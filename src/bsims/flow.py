from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

NDET, NFEAT = 10, 6
K_BINS = 8
B_TAIL = 5.0
MIN_BIN = 1e-3
MIN_DERIV = 1e-3
N_FF = 16
N_FF_S = 8
EMB_V, EMB_C = 64, 128

SIG_MIN, SIG_MAX = 0.01, 0.5

LR = 3e-3

MASKS = {6: ((0, 1, 2), (3, 4, 5), (0, 2, 4), (1, 3, 5), (0, 3, 4), (1, 2, 5)),
         5: ((0, 1, 2), (3, 4), (0, 3), (1, 2, 4), (0, 2, 4), (1, 3))}
N_COUPLE = 6

_SOFTPLUS_INV_1 = float(np.log(np.expm1(1.0)))


@dataclasses.dataclass
class Cfg:
    steps: int = 20000
    batch: int = 64
    lr: float = LR
    width: int = 256
    ions: int = 64
    seed: int = 0
    sig_min: float = SIG_MIN
    sig_max: float = SIG_MAX
    sig_eval: float = SIG_MIN
    nfeat: int = NFEAT
    train_planes: tuple = None


def _rqs(x, uw, uh, ud, inverse):
    K = uw.shape[-1]
    w = MIN_BIN + (1.0 - MIN_BIN * K) * jax.nn.softmax(uw, -1)
    h = MIN_BIN + (1.0 - MIN_BIN * K) * jax.nn.softmax(uh, -1)
    xk = -B_TAIL + 2.0 * B_TAIL * jnp.cumsum(w, -1)
    yk = -B_TAIL + 2.0 * B_TAIL * jnp.cumsum(h, -1)
    pad = jnp.full(xk.shape[:-1] + (1,), -B_TAIL)
    xk = jnp.concatenate([pad, xk], -1)
    yk = jnp.concatenate([pad, yk], -1)
    one = jnp.ones(ud.shape[:-1] + (1,))
    dk = jnp.concatenate([one, MIN_DERIV + jax.nn.softplus(ud), one], -1)

    inside = (x > -B_TAIL) & (x < B_TAIL)
    knots = yk if inverse else xk
    xc = jnp.clip(x, -B_TAIL + 1e-6, B_TAIL - 1e-6)
    i = jnp.clip(jnp.sum(xc[..., None] >= knots[..., 1:-1], -1), 0, K - 1)

    take = lambda a: jnp.take_along_axis(a, i[..., None], -1)[..., 0]
    xlo, xhi = take(xk[..., :-1]), take(xk[..., 1:])
    ylo, yhi = take(yk[..., :-1]), take(yk[..., 1:])
    dlo, dhi = take(dk[..., :-1]), take(dk[..., 1:])
    wb, hb = xhi - xlo, yhi - ylo
    sl = hb / wb

    if inverse:
        r = xc - ylo
        c2 = hb * (sl - dlo) + r * (dhi + dlo - 2.0 * sl)
        c1 = hb * dlo - r * (dhi + dlo - 2.0 * sl)
        c0 = -sl * r
        disc = jnp.maximum(c1 * c1 - 4.0 * c2 * c0, 0.0)
        xi = jnp.clip(2.0 * c0 / (-c1 - jnp.sqrt(disc)), 0.0, 1.0)
        out = xi * wb + xlo
    else:
        xi = jnp.clip((xc - xlo) / wb, 0.0, 1.0)
        num = hb * (sl * xi * xi + dlo * xi * (1.0 - xi))
        den = sl + (dhi + dlo - 2.0 * sl) * xi * (1.0 - xi)
        out = ylo + num / den

    q = xi * (1.0 - xi)
    den = sl + (dhi + dlo - 2.0 * sl) * q
    ld = (2.0 * jnp.log(sl)
          + jnp.log(dhi * xi * xi + 2.0 * sl * q + dlo * (1.0 - xi) ** 2)
          - 2.0 * jnp.log(den))
    ld = -ld if inverse else ld
    return jnp.where(inside, out, x), jnp.where(inside, ld, 0.0)


class Coupling(eqx.Module):
    net: eqx.nn.MLP
    act: tuple = eqx.field(static=True)
    psv: tuple = eqx.field(static=True)

    def __init__(self, act, width, depth, ctx, key, nfeat=NFEAT):
        self.act = tuple(act)
        self.psv = tuple(j for j in range(nfeat) if j not in act)
        n_out = len(self.act) * (3 * K_BINS - 1)
        net = eqx.nn.MLP(len(self.psv) + ctx, n_out, width, depth,
                         activation=jax.nn.gelu, key=key)
        b = np.zeros(n_out, np.float32)
        b.reshape(len(self.act), 3 * K_BINS - 1)[:, 2 * K_BINS:] = _SOFTPLUS_INV_1
        self.net = eqx.tree_at(
            lambda m: (m.layers[-1].weight, m.layers[-1].bias), net,
            (jnp.zeros_like(net.layers[-1].weight), jnp.asarray(b)))

    def __call__(self, x, ctx, inverse=False):
        xp = x[:, jnp.array(self.psv)]
        p = jax.vmap(self.net)(jnp.concatenate([xp, ctx], -1))
        p = p.reshape(x.shape[0], len(self.act), 3 * K_BINS - 1)
        y, ld = _rqs(x[:, jnp.array(self.act)], p[..., :K_BINS],
                     p[..., K_BINS:2 * K_BINS], p[..., 2 * K_BINS:], inverse)
        return x.at[:, jnp.array(self.act)].set(y), ld.sum(-1)


class Flow(eqx.Module):
    couplings: list
    volt_enc: eqx.nn.MLP
    cond: eqx.nn.MLP
    gate_logits: jax.Array
    v_mu: jax.Array
    v_sd: jax.Array
    ff: jax.Array
    ffs: jax.Array
    s_all: jax.Array
    s_max: float = eqx.field(static=True)
    sig_eval: float = eqx.field(static=True, default=SIG_MIN)
    sig_min: float = eqx.field(static=True, default=SIG_MIN)
    sig_max: float = eqx.field(static=True, default=SIG_MAX)
    nfeat: int = eqx.field(static=True, default=NFEAT)

    def _gate(self, s):
        sc = jnp.clip(s, self.s_all[0], self.s_all[-1])
        i = jnp.clip(jnp.searchsorted(self.s_all, sc, side="right") - 1,
                     0, self.s_all.shape[0] - 2)
        lo, hi = self.s_all[i], self.s_all[i + 1]
        w = ((sc - lo) / (hi - lo))[:, None]
        return (1.0 - w) * self.gate_logits[i] + w * self.gate_logits[i + 1]

    def ctx(self, V, s, log_sig):
        g = jax.nn.sigmoid(self._gate(s))
        e = jax.vmap(self.volt_enc)(((V - self.v_mu) / self.v_sd) * g)
        sn = (s / self.s_max)[:, None]
        ph = 2.0 * jnp.pi * sn * self.ff
        t = ((log_sig - np.log(self.sig_min))
             / (np.log(self.sig_max) - np.log(self.sig_min)))[:, None]
        phs = 2.0 * jnp.pi * t * self.ffs
        return jax.vmap(self.cond)(jnp.concatenate(
            [e, jnp.sin(ph), jnp.cos(ph), sn, jnp.sin(phs), jnp.cos(phs), t], -1))

    def logp(self, x, c):
        ld = jnp.zeros(x.shape[0])
        for cp in self.couplings:
            x, l = cp(x, c)
            ld = ld + l
        return -0.5 * jnp.sum(x * x, -1) - 0.5 * x.shape[-1] * jnp.log(2 * jnp.pi) + ld

    def inv(self, z, c):
        for cp in reversed(self.couplings):
            z, _ = cp(z, c, inverse=True)
        return z


def build(V, s, cfg: Cfg, key) -> Flow:
    nv = V.shape[1]
    keys = jax.random.split(key, N_COUPLE + 2)
    s = np.asarray(s, np.float64)
    s_max = float(max(s.max(), 1.0))

    fk = (s / s_max)[:, None]
    fj = (np.arange(nv) / max(nv - 1, 1))[None, :]
    g0 = 1.5 - 4.0 * np.maximum(fj - fk, 0.0)

    sd = V.std(0)
    return Flow(
        couplings=[Coupling(m, cfg.width, 2, EMB_C, keys[i], nfeat=cfg.nfeat)
                   for i, m in enumerate(MASKS[cfg.nfeat])],
        volt_enc=eqx.nn.MLP(nv, EMB_V, 128, 1, activation=jax.nn.gelu, key=keys[-2]),
        cond=eqx.nn.MLP(EMB_V + 2 * N_FF + 1 + 2 * N_FF_S + 1, EMB_C, EMB_C, 1,
                        activation=jax.nn.gelu, key=keys[-1]),
        gate_logits=jnp.asarray(g0, jnp.float32),
        v_mu=jnp.asarray(V.mean(0)),
        v_sd=jnp.asarray(np.where(sd < 1e-8, 1.0, sd)),
        ff=jnp.asarray(np.geomspace(0.5, 32.0, N_FF), jnp.float32),
        ffs=jnp.asarray(np.geomspace(0.5, 8.0, N_FF_S), jnp.float32),
        s_all=jnp.asarray(s, jnp.float32), s_max=s_max,
        sig_eval=float(cfg.sig_eval), sig_min=float(cfg.sig_min),
        sig_max=float(cfg.sig_max), nfeat=int(cfg.nfeat))


def _spec(model):
    fs = jax.tree.map(eqx.is_inexact_array, model)
    return eqx.tree_at(lambda m: (m.v_mu, m.v_sd, m.ff, m.ffs, m.s_all), fs,
                       replace=(False,) * 5)


def n_params(model) -> int:
    return int(sum(x.size for x in jax.tree.leaves(eqx.filter(model, _spec(model)))))


def with_sig_eval(model, sig) -> Flow:
    return dataclasses.replace(model, sig_eval=float(sig))


def train(model, d, train_idx, cfg: Cfg, key, verbose=0) -> Flow:
    lo_, hi_ = np.log(cfg.sig_min), np.log(cfg.sig_max)
    mu, sd = d.scaler()
    mu, sd = np.asarray(mu, np.float32), np.asarray(sd, np.float32)
    s_all = np.asarray(d.s, np.float32)
    tr = np.asarray(train_idx)
    kept = np.asarray(d.M[tr]).sum(2)
    ndet, nfeat = d.X.shape[1], d.X.shape[3]
    if nfeat != model.nfeat:
        raise ValueError(f"data is {nfeat}-D, the flow is {model.nfeat}-D")
    pools = [tr[kept[:, k] >= 8] for k in range(ndet)]
    keptd = {k: dict(zip(pools[k], kept[kept[:, k] >= 8, k])) for k in range(ndet)}
    allowed = (set(range(ndet)) if cfg.train_planes is None
               else {int(k) - 1 for k in cfg.train_planes})
    planes = np.array([k for k in range(ndet) if len(pools[k]) and k in allowed])
    if len(planes) == 0:
        return model

    mk = np.array([max(kept[:, k].mean(), 1.0) for k in range(ndet)])
    wk = np.sqrt(mk.max() / mk).astype(np.float32)
    wk /= wk.mean()

    P, M_ION = int(cfg.batch), int(cfg.ions)
    spec = _spec(model)
    sch = optax.warmup_cosine_decay_schedule(
        cfg.lr * 0.05, cfg.lr, max(1, cfg.steps // 20), cfg.steps, cfg.lr * 0.02)
    opt = optax.chain(optax.clip_by_global_norm(5.0),
                      optax.adamw(sch, weight_decay=1e-5))
    ostate = opt.init(eqx.filter(model, spec))

    @eqx.filter_jit
    def step(model, ostate, x, V, sb, w, sig, nkey):
        x = x + sig[:, None, None] * jax.random.normal(nkey, x.shape)
        ls = jnp.log(sig)

        def loss_fn(m):
            c = m.ctx(V, sb, ls)
            n = x.shape[1]
            lp = m.logp(x.reshape(-1, x.shape[-1]), jnp.repeat(c, n, axis=0))
            return -jnp.sum(lp.reshape(x.shape[0], n).mean(1) * w) / jnp.sum(w)

        diff, static = eqx.partition(model, spec)
        lo, gr = eqx.filter_value_and_grad(
            lambda dm: loss_fn(eqx.combine(dm, static)))(diff)
        upd, ostate = opt.update(gr, ostate, diff)
        return eqx.combine(eqx.apply_updates(diff, upd), static), ostate, lo

    rng = np.random.default_rng(cfg.seed + 12345)
    nkeys = jax.random.split(key, 2)[0]
    for t in range(int(cfg.steps)):
        ks = rng.choice(planes, size=P)
        idxs = np.empty(P, np.int64)
        for k in np.unique(ks):
            m = ks == k
            idxs[m] = rng.choice(pools[k], size=int(m.sum()))
        nk = np.array([keptd[int(k)][int(i)] for k, i in zip(ks, idxs)])
        raw = np.asarray(d.X[idxs, ks], np.float32)
        sel = (rng.random((P, M_ION)) * nk[:, None]).astype(np.int64)
        xb = np.take_along_axis(raw, sel[:, :, None], 1)
        xb = (xb - mu[ks][:, None]) / sd[ks][:, None]
        sig = np.exp(rng.uniform(lo_, hi_, size=P)).astype(np.float32)
        model, ostate, lo = step(model, ostate, jnp.asarray(xb),
                                 jnp.asarray(d.V[idxs]), jnp.asarray(s_all[ks]),
                                 jnp.asarray(wk[ks]), jnp.asarray(sig),
                                 jax.random.fold_in(nkeys, t))
        if verbose and t % max(1, cfg.steps // 20) == 0:
            print(f"  [flow] step {t:6d}  nll {float(lo):8.4f}", flush=True)
    return model


@eqx.filter_jit
def _draw(model, u, s, z, ls):
    c = model.ctx(u[None], s[None], ls[None])
    return model.inv(z, jnp.repeat(c, z.shape[0], 0))


@eqx.filter_jit
def _logp(model, u, s, x, ls):
    c = model.ctx(u[None], s[None], ls[None])
    return model.logp(x, jnp.repeat(c, x.shape[0], 0))


def _ls(model, sig):
    return jnp.asarray(np.log(float(model.sig_eval if sig is None else sig)),
                       jnp.float32)


def sample(model, u, s, n, key, sig=None):
    npad = int(2 ** int(np.ceil(np.log2(max(n, 8)))))
    z = jax.random.normal(key, (npad, model.nfeat))
    return _draw(model, jnp.asarray(u, jnp.float32), jnp.asarray(s, jnp.float32),
                 z, _ls(model, sig))[:n]


def log_prob(model, u, s, x, sig=None):
    return _logp(model, jnp.asarray(u, jnp.float32), jnp.asarray(s, jnp.float32),
                 jnp.asarray(x, jnp.float32), _ls(model, sig))
