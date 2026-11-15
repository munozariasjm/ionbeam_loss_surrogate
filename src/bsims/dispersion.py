from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .hazard import Linear

LOGPHI_CLAMP = (-4.0, 12.0)
PHI_BIAS_INIT = 6.0


def betabinom_nll(logit_p, logphi, m, s, reduction="mean"):
    p = jax.nn.sigmoid(logit_p)
    phi = jnp.exp(jnp.clip(logphi, *LOGPHI_CLAMP))
    a = jnp.clip(p * phi, 1e-4, None)
    b = jnp.clip((1.0 - p) * phi, 1e-4, None)
    lg = jax.lax.lgamma
    ll = lg(s + a) + lg(m - s + b) - lg(m + a + b) + lg(a + b) - lg(a) - lg(b)
    alive = (m > 0).astype(ll.dtype)
    loss = -jnp.sum(ll * alive)
    return loss if reduction == "sum" else loss / jnp.clip(jnp.sum(alive), 1.0, None)


class DispHead(eqx.Module):
    l1: Linear
    l2: Linear

    def __init__(self, n_features, n_dets, hidden=128, *, key):
        k1, k2 = jax.random.split(key)
        self.l1 = Linear(n_features, hidden, key=k1)
        l2 = Linear(hidden, n_dets, key=k2)
        self.l2 = eqx.tree_at(
            lambda m: (m.weight, m.bias), l2,
            (l2.weight * 1e-2, jnp.full((n_dets,), PHI_BIAS_INIT, jnp.float32)))

    def __call__(self, z):
        return self.l2(jax.nn.silu(self.l1(z)))


@eqx.filter_jit
def _step(head, opt, ostate, z, lp, m, s, lr):
    loss, g = eqx.filter_value_and_grad(
        lambda h: betabinom_nll(lp, h(z), m, s))(head)
    upd, ostate = opt.update(g, ostate, eqx.filter(head, eqx.is_inexact_array))
    return eqx.apply_updates(head, jax.tree.map(lambda u: -lr * u, upd)), ostate, loss


def fit(Ztr, lptr, mtr, str_, Zva, lpva, mva, sva, *, seed=0, max_steps=4000,
        lr=1e-2, batch=1024, patience=30, hidden=128):
    import optax

    rng = np.random.RandomState(seed)
    Ztr, lptr, mtr, str_ = map(jnp.asarray, (Ztr, lptr, mtr, str_))
    Zva, lpva, mva, sva = map(jnp.asarray, (Zva, lpva, mva, sva))
    head = DispHead(Ztr.shape[1], mtr.shape[1], hidden=hidden,
                    key=jax.random.PRNGKey(seed))
    opt = optax.chain(optax.add_decayed_weights(1e-5), optax.scale_by_adam())
    ostate = opt.init(eqx.filter(head, eqx.is_inexact_array))

    @eqx.filter_jit
    def vloss(h):
        return betabinom_nll(lpva, h(Zva), mva, sva)

    n = len(Ztr)
    n_epochs = max(1, math.ceil(max_steps / max(1, math.ceil(n / batch))))
    best, best_head, bad, steps = float("inf"), None, 0, 0
    for ep in range(n_epochs):
        lr_ep = jnp.float32(lr * (1.0 + math.cos(math.pi * ep / n_epochs)) / 2.0)
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            head, ostate, _ = _step(head, opt, ostate, Ztr[j], lptr[j],
                                    mtr[j], str_[j], lr_ep)
            steps += 1
            if steps >= max_steps:
                break
        vl = float(vloss(head))
        if vl < best - 1e-5:
            best, bad, best_head = vl, 0, head
        else:
            bad += 1
            if bad >= patience:
                break
        if steps >= max_steps:
            break
    return (best_head if best_head is not None else head,
            dict(steps=steps, best_val_nll=best, n_epochs_planned=n_epochs))


def predict_phi(head, Z):
    return np.asarray(jnp.exp(jnp.clip(head(jnp.asarray(Z)), *LOGPHI_CLAMP)))


def var_inflation(m, phi):
    return (m + phi) / (1.0 + phi)


def conditional_survival(log_T, n_pinned=2):
    p = np.exp(np.diff(np.asarray(log_T, np.float64), axis=1)[:, n_pinned - 1:])
    return np.clip(p, 1e-9, 1 - 1e-9)


def survival_chain(p, phi=None, n_draws=2000, n_launched=10000, seed=0):
    rng = np.random.default_rng(seed)
    n, K = p.shape
    cur = np.full((n_draws, n), float(n_launched))
    out = np.empty((n_draws, n, K))
    for j in range(K):
        pj = np.repeat(np.clip(p[:, j], 1e-9, 1 - 1e-9)[None], n_draws, 0)
        q = pj if phi is None else rng.beta(
            np.clip(pj * phi[:, j][None], 1e-4, None),
            np.clip((1 - pj) * phi[:, j][None], 1e-4, None))
        cur = rng.binomial(cur.astype(np.int64), q).astype(float)
        out[:, :, j] = cur / n_launched
    return out


def predict_interval(draws, level=0.68):
    q = (1.0 - level) / 2.0
    return (np.quantile(draws, q, axis=0), np.quantile(draws, 1 - q, axis=0))


def coverage(draws, T_meas, level=0.68, deliverers_only=True):
    lo, hi = predict_interval(draws, level)
    T = np.asarray(T_meas, np.float64)
    ok = (T >= lo) & (T <= hi)
    if not deliverers_only:
        return ok.mean(0)
    live = T > 0
    return np.array([ok[live[:, j], j].mean() if live[:, j].any() else np.nan
                     for j in range(T.shape[1])])
