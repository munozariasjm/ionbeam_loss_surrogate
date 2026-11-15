from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .hazard import Head


class Ensemble(eqx.Module):
    members: list

    def __init__(self, members):
        if not members:
            raise ValueError("an ensemble needs at least one member")
        self.members = list(members)

    def __len__(self):
        return len(self.members)

    @property
    def n_pinned(self):
        return self.members[0].n_pinned

    def member_log_T(self, u):
        return jnp.stack([m.log_T(u) for m in self.members])

    def log_T(self, u):
        lt = self.member_log_T(u)
        return jax.scipy.special.logsumexp(lt, axis=0) - jnp.log(lt.shape[0])

    def T(self, u):
        return jnp.exp(self.log_T(u))

    def spread(self, u):
        t = jnp.exp(self.member_log_T(u))
        return t.mean(0), t.std(0, ddof=1)

    @staticmethod
    def load(paths):
        from .checkpoint import load_hazard

        return Ensemble([Head(*load_hazard(p)[:3]) for p in paths])


def ensemble_from_dir(directory, pattern="hazard_ens_m*.eqx"):
    import pathlib
    import re

    paths = sorted(pathlib.Path(directory).glob(pattern),
                   key=lambda p: int(re.findall(r"(\d+)", p.stem)[-1]))
    if not paths:
        raise FileNotFoundError(f"no members matching {pattern} in {directory}")
    return Ensemble.load(paths), paths
