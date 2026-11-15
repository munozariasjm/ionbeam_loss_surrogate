from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from . import flow as F
from .data import N_LAUNCHED, S_REF


class Beamline(eqx.Module):
    transmission: eqx.Module
    flow: F.Flow
    mu: jax.Array
    sd: jax.Array
    s: jax.Array
    sig_eval: float = eqx.field(static=True, default=F.SIG_MIN)

    def __init__(self, transmission, flow, mu, sd, s=S_REF, sig_eval=None):
        self.transmission = transmission
        self.flow = flow
        self.mu = jnp.asarray(mu, jnp.float32)
        self.sd = jnp.asarray(sd, jnp.float32)
        self.s = jnp.asarray(s, jnp.float32)
        self.sig_eval = float(flow.sig_eval if sig_eval is None else sig_eval)

    def log_T(self, u):
        return self.transmission.log_T(u)

    def T(self, u):
        return jnp.exp(self.log_T(u))

    def expected_counts(self, u, n_launched=N_LAUNCHED):
        return n_launched * self.T(u)

    def _one(self, u):
        u = jnp.asarray(u, jnp.float32)
        if u.ndim != 1:
            raise ValueError(f"expected one voltage vector, got shape "
                             f"{tuple(u.shape)}; only log_T/T/expected_counts "
                             "are batched.")
        return u

    def sample(self, u, k, n, key, sig=None):
        u = self._one(u)
        x = F.sample(self.flow, u, self.s[k - 1], n, key,
                     self.sig_eval if sig is None else sig)
        return x * self.sd[k - 1] + self.mu[k - 1]

    def log_1m_T(self, u, k):
        if k <= self.transmission.n_pinned:
            return jnp.zeros(())
        lt = jnp.minimum(self.log_T(u)[0, k - 1], -1e-7)
        return jnp.where(lt > -0.6931472,
                         jnp.log(-jnp.expm1(lt)), jnp.log1p(-jnp.exp(lt)))

    def log_prob_cloud(self, u, k, X, sig=None, n_launched=N_LAUNCHED):
        u = self._one(u)
        X = jnp.asarray(X, jnp.float32)
        m = X.shape[0]
        if m > n_launched:
            raise ValueError(f"cloud has {m} ions but n_launched={n_launched}.")
        mass = (m * self.log_T(u)[0, k - 1]
                + (n_launched - m) * self.log_1m_T(u, k))
        if m == 0:
            return mass
        xs = (X - self.mu[k - 1]) / self.sd[k - 1]
        lp = F.log_prob(self.flow, u, self.s[k - 1], xs,
                        self.sig_eval if sig is None else sig)
        return mass + jnp.sum(lp) - m * jnp.sum(jnp.log(self.sd[k - 1]))
