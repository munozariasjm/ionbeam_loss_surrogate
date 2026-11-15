from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

SEGMENTS = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11),
            (12, 13), (14, 15), (16, 17, 18), (19, 20, 21))

NDET = 10
DET_START = 2


def _uniform(key, shape, lim):
    return jax.random.uniform(key, shape, jnp.float32, -lim, lim)


class Linear(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, fin, fout, *, key):
        k1, k2 = jax.random.split(key)
        lim = 1.0 / math.sqrt(fin)
        self.weight = _uniform(k1, (fout, fin), lim)
        self.bias = _uniform(k2, (fout,), lim)

    def __call__(self, x):
        return x @ self.weight.T + self.bias


class GRUCell(eqx.Module):
    weight_ih: jax.Array
    weight_hh: jax.Array
    bias_ih: jax.Array
    bias_hh: jax.Array

    def __init__(self, input_size, hidden_size, *, key):
        k = jax.random.split(key, 4)
        lim = 1.0 / math.sqrt(hidden_size)
        self.weight_ih = _uniform(k[0], (3 * hidden_size, input_size), lim)
        self.weight_hh = _uniform(k[1], (3 * hidden_size, hidden_size), lim)
        self.bias_ih = _uniform(k[2], (3 * hidden_size,), lim)
        self.bias_hh = _uniform(k[3], (3 * hidden_size,), lim)

    def __call__(self, x, h):
        gi = x @ self.weight_ih.T + self.bias_ih
        gh = h @ self.weight_hh.T + self.bias_hh
        ir, iz, inn = jnp.split(gi, 3, axis=-1)
        hr, hz, hn = jnp.split(gh, 3, axis=-1)
        r = jax.nn.sigmoid(ir + hr)
        z = jax.nn.sigmoid(iz + hz)
        n = jnp.tanh(inn + r * hn)
        return (1.0 - z) * n + z * h


class SegGate(eqx.Module):
    g: jax.Array

    def __init__(self, n_features, n_dets, prior=None):
        if prior is None:
            prior = np.full((n_dets, n_features), -1.0, np.float32)
            for k, cols in enumerate(SEGMENTS[:n_dets]):
                prior[k, [c for c in cols if c < n_features]] = 1.0
        self.g = jnp.asarray(np.asarray(prior, np.float32))

    def __call__(self, u):
        return u[:, None, :] * jax.nn.sigmoid(self.g)[None]


class GRUHazard(eqx.Module):
    gate: SegGate
    enc: tuple
    inp: list
    cell: list
    out: list
    emb: jax.Array
    bias: jax.Array
    n_dets: int = eqx.field(static=True)
    tied: bool = eqx.field(static=True)
    n_pinned: int = eqx.field(static=True, default=DET_START)

    def __init__(self, n_features, n_dets, d=48, tied=False, n_pinned=DET_START,
                 gate_prior=None, *, key):
        self.n_dets, self.tied, self.n_pinned = n_dets, tied, int(n_pinned)
        k = jax.random.split(key, 4)
        self.gate = SegGate(n_features, n_dets, gate_prior)
        ke = jax.random.split(k[0], 2)
        self.enc = (Linear(n_features, d, key=ke[0]), Linear(d, d, key=ke[1]))
        n = 1 if tied else n_dets
        self.inp = [Linear(n_features, d, key=kk) for kk in jax.random.split(k[1], n)]
        self.cell = [GRUCell(d, d, key=kk) for kk in jax.random.split(k[2], n)]
        self.out = [Linear(d, 1, key=kk) for kk in jax.random.split(k[3], n)]
        self.emb = jnp.zeros((n_dets, d), jnp.float32)
        self.bias = jnp.zeros((n_dets,), jnp.float32)

    def __call__(self, u):
        xs = self.gate(u)
        h = self.enc[1](jax.nn.silu(self.enc[0](u)))
        logits = []
        for k in range(self.n_dets):
            i = 0 if self.tied else k
            h = self.cell[i](self.inp[i](xs[:, k]) + self.emb[k], h)
            logits.append(self.out[i](h)[..., 0] + self.bias[k])
        return jnp.stack(logits, axis=1)


def log_T(net, u_std):
    lp = jax.nn.log_sigmoid(net(u_std))
    pinned = jnp.zeros((u_std.shape[0], net.n_pinned), lp.dtype)
    return jnp.concatenate([pinned, jnp.cumsum(lp, axis=1)], axis=1)


def n_params(net) -> int:
    return int(sum(x.size for x in
                   jax.tree.leaves(eqx.filter(net, eqx.is_inexact_array))))


class Head(eqx.Module):
    net: GRUHazard
    mu: jax.Array
    sd: jax.Array

    def __init__(self, net, mu, sd):
        self.net = net
        self.mu = jnp.asarray(mu, jnp.float32)
        self.sd = jnp.asarray(sd, jnp.float32)

    @property
    def n_pinned(self):
        return self.net.n_pinned

    def log_T(self, u):
        u = jnp.atleast_2d(jnp.asarray(u, jnp.float32))
        return log_T(self.net, (u - self.mu) / self.sd)

    def T(self, u):
        return jnp.exp(self.log_T(u))
