from __future__ import annotations

import pathlib
import pickle

import equinox as eqx
import jax
import numpy as np

from . import flow as F
from .data import S_REF
from .dispersion import DispHead
from .hazard import GRUHazard


def save(path, hparams, tree):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dict(hparams), f)
        eqx.tree_serialise_leaves(f, tree)
    return path


def load(path, skeleton):
    with open(pathlib.Path(path), "rb") as f:
        h = pickle.load(f)
        return h, eqx.tree_deserialise_leaves(f, skeleton(h))


def load_hazard(path):
    h, net = load(path, lambda h: GRUHazard(
        h["n_features"], h["n_dets"], d=h["d"], tied=h["tied"],
        n_pinned=h.get("n_pinned", 2), key=jax.random.PRNGKey(0)))
    return net, np.asarray(h["mean_"]), np.asarray(h["scale_"]), h


def load_flow(path, s=S_REF, n_voltages=22):
    def skeleton(h):
        nv = int(h.get("n_voltages", n_voltages))
        cfg = F.Cfg(width=h["width"], sig_eval=h["sig_eval"],
                    sig_min=h.get("sig_min", F.SIG_MIN),
                    sig_max=h.get("sig_max", F.SIG_MAX),
                    nfeat=int(h.get("nfeat", F.NFEAT)))
        return F.build(np.zeros((2, nv), np.float32), np.asarray(h.get("s", s)),
                       cfg, jax.random.PRNGKey(0))

    h, model = load(path, skeleton)
    return model, h


def load_dispersion(path):
    h, head = load(path, lambda h: DispHead(
        h["n_features"], h["n_dets"], hidden=h.get("hidden", 128),
        key=jax.random.PRNGKey(0)))
    return head, np.asarray(h["mean_"]), np.asarray(h["scale_"]), h
