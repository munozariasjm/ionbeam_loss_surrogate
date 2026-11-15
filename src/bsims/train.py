from __future__ import annotations

import math
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from sklearn.preprocessing import StandardScaler

from .hazard import GRUHazard

_LN10 = math.log(10.0)
DET10 = -1


def binomial_nll(logits, m, s):
    alive = (m > 0).astype(logits.dtype)
    target = jnp.where(m > 0, s / jnp.clip(m, 1e-12, None), 0.0)
    bce = (jnp.maximum(logits, 0) - logits * target
           + jnp.logaddexp(0.0, -jnp.abs(logits)))
    return jnp.sum(bce * m * alive) / jnp.clip(jnp.sum(m * alive), 1.0, None)


def log10_T_mse(logits, m, s):
    alive = m > 0
    deliv = jnp.all(alive & (s > 0), axis=1)
    r = jnp.log(jnp.clip(s, 1e-12, None)) - jnp.log(jnp.clip(m, 1e-12, None))
    t_true = jnp.sum(jnp.where(alive, r, 0.0), axis=1) / _LN10
    t_pred = jnp.sum(jax.nn.log_sigmoid(logits), axis=1) / _LN10
    e = jnp.where(deliv, t_pred - t_true, 0.0)
    return jnp.sum(e ** 2) / jnp.clip(jnp.sum(deliv), 1.0, None)


def objective(logits, m, s, obj="logT", lambda_log=0.1):
    if obj == "binom":
        return binomial_nll(logits, m, s)
    if obj == "logT":
        return binomial_nll(logits, m, s) + lambda_log * log10_T_mse(logits, m, s)
    raise ValueError(f"obj={obj!r} must be 'binom' or 'logT'")


def _make_opt(wd=1e-5):
    return optax.chain(optax.add_decayed_weights(wd), optax.scale_by_adam())


@eqx.filter_jit
def _step(model, opt, opt_state, x, m, s, lr, clip, obj, lambda_log):
    loss, grads = eqx.filter_value_and_grad(
        lambda net: objective(net(x), m, s, obj, lambda_log))(model)
    gnorm = optax.global_norm(grads)
    coef = jnp.minimum(clip / (gnorm + 1e-6), 1.0)
    grads = jax.tree.map(lambda g: g * coef, grads)
    upd, opt_state = opt.update(grads, opt_state,
                                eqx.filter(model, eqx.is_inexact_array))
    return eqx.apply_updates(model, jax.tree.map(lambda u: -lr * u, upd)), \
        opt_state, gnorm


class Survival:
    def __init__(self, n_features, n_dets, *, seed=0, d=48, tied=False,
                 obj="logT", lambda_log=0.1, lr=2e-3, wd=1e-5, batch=1024,
                 patience=30, max_steps=20000, clip=1.0, deliverers_only=False,
                 standardize=True, n_pinned=2, gate_prior=None):
        self.n_features, self.n_dets = n_features, n_dets
        self.n_pinned, self.gate_prior = n_pinned, gate_prior
        self.seed, self.d, self.tied = seed, d, tied
        self.obj, self.lambda_log = obj, lambda_log
        self.lr, self.wd, self.batch = lr, wd, batch
        self.patience, self.max_steps, self.clip = patience, max_steps, clip
        self.deliverers_only, self.standardize = deliverers_only, standardize
        self.steps_completed, self.best_epoch = 0, -1
        self.converged = False
        self.clip_bind_frac = float("nan")

    def _prep(self, X):
        return jnp.asarray(self.sx.transform(X) if self.sx is not None else X,
                           jnp.float32)

    def fit(self, Xtr, mtr, str_, Xva, mva, sva, patience_window=None):
        if self.deliverers_only:
            keep = str_[:, DET10] > 0
            if keep.sum() >= 256:
                Xtr, mtr, str_ = Xtr[keep], mtr[keep], str_[keep]

        rng = np.random.RandomState(self.seed)
        self.sx = StandardScaler().fit(Xtr) if self.standardize else None
        Xt, Xv = self._prep(Xtr), self._prep(Xva)
        mt, st = jnp.asarray(mtr, jnp.float32), jnp.asarray(str_, jnp.float32)
        mv, sv = jnp.asarray(mva, jnp.float32), jnp.asarray(sva, jnp.float32)

        model = GRUHazard(self.n_features, self.n_dets, d=self.d, tied=self.tied,
                          n_pinned=self.n_pinned, gate_prior=self.gate_prior,
                          key=jax.random.PRNGKey(self.seed))
        opt = _make_opt(self.wd)
        opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
        clip = jnp.float32(self.clip)

        @eqx.filter_jit
        def vloss(net):
            return objective(net(Xv), mv, sv, self.obj, self.lambda_log)

        n = len(Xt)
        steps_per_epoch = max(1, math.ceil(n / self.batch))
        n_epochs = max(1, math.ceil(self.max_steps / steps_per_epoch))
        patience = self.patience
        if patience_window:
            patience = max(1, round(patience_window / steps_per_epoch))
        self.n_epochs_planned, self.patience_used = n_epochs, patience

        best, best_model, bad, steps, binds = float("inf"), None, 0, 0, 0
        t0 = time.time()
        for ep in range(n_epochs):
            lr = jnp.float32(self.lr * (1.0 + math.cos(math.pi * ep / n_epochs)) / 2.0)
            perm = rng.permutation(n)
            for i in range(0, n, self.batch):
                j = perm[i:i + self.batch]
                model, opt_state, gnorm = _step(
                    model, opt, opt_state, Xt[j], mt[j], st[j], lr, clip,
                    self.obj, self.lambda_log)
                binds += int(float(gnorm) > self.clip)
                steps += 1
                if steps >= self.max_steps:
                    break

            vl = float(vloss(model))
            if vl < best - 1e-5:
                best, bad, best_model, self.best_epoch = vl, 0, model, ep
            else:
                bad += 1
                if bad >= patience:
                    break
            if steps >= self.max_steps:
                break

        self.net = best_model if best_model is not None else model
        self.best_val = best
        self.steps_completed = steps
        self.converged = True
        self.clip_bind_frac = binds / max(steps, 1)
        self.train_s = time.time() - t0
        return self

    def predict_p(self, X):
        return np.asarray(jax.nn.sigmoid(self.net(self._prep(X))))

    def n_params(self):
        return int(sum(x.size for x in
                       jax.tree.leaves(eqx.filter(self.net, eqx.is_inexact_array))))
