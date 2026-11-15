from __future__ import annotations

import os
import pathlib

import numpy as np
import pandas as pd

NDET, NFEAT = 10, 6
FEATS = ("x1", "x2", "a1", "a2", "ke", "dz")
DET_COLS = [f"det{k}_n" for k in range(1, NDET + 1)]
DET_START = 2
DET10 = -1
N_LAUNCHED = 10000

PINNED_ELECTRODE = "V36"

S_REF = np.array([0.0, 98.1244888305664, 188.1256866455078, 347.36248779296875,
                  462.7955017089844, 608.9094848632812, 691.5440673828125,
                  886.9576416015625, 1014.1102294921875, 1120.850830078125])


def root(path=None) -> pathlib.Path:
    return pathlib.Path(path or os.environ.get("BSIMS_DATA", "data"))


class Data:
    def __init__(self, path=None, n=0, seed=0):
        self.path = p = root(path)
        idx = pd.read_parquet(p / "index.parquet")
        cfg = pd.read_parquet(p / "configs.parquet").set_index("config_id")
        cfg = cfg.reindex(idx.config_id.to_numpy())
        if cfg.isna().any().any():
            raise RuntimeError("configs.parquet is missing config_ids present "
                               "in index.parquet.")

        if PINNED_ELECTRODE in cfg.columns:
            pin = cfg[PINNED_ELECTRODE].to_numpy()
            if pin.min() != pin.max():
                raise RuntimeError(
                    f"{PINNED_ELECTRODE} varies ({pin.min()}..{pin.max()}); it "
                    "carries signal here and must not be dropped as a constant.")
        self.volt_names = [c for c in cfg.columns if c != PINNED_ELECTRODE]
        self.V = cfg[self.volt_names].to_numpy(np.float32)

        fr = np.load(p / "frames.npz")
        self.legacy = legacy = "names" not in fr.files
        self.names = ([f"det{k}" for k in range(1, NDET + 1)] if legacy
                      else [str(x) for x in fr["names"]])
        self.feats = tuple(FEATS if legacy else [str(x) for x in fr["feats"]])
        self.n_pinned = DET_START if legacy else int(fr["n_pinned"])
        self.ndet = len(self.names)
        self.det_cols = [f"{d}_n" for d in self.names]

        self.config_id = idx.config_id.to_numpy()
        self.layer = idx.layer.to_numpy().astype(str)
        self.n_launched = idx.n_launched.to_numpy(np.float64)
        self.N = idx[self.det_cols].to_numpy(np.int64)
        if not (np.diff(self.N, axis=1) <= 0).all():
            raise RuntimeError("survivor counts must be non-increasing along "
                               "the beam.")
        if not (self.N[:, 0] <= self.n_launched).all():
            raise RuntimeError("det1 count exceeds n_launched.")

        self.X = np.load(p / "X.npy", mmap_mode="r")
        self.M = np.load(p / "M.npy", mmap_mode="r")
        if len(self.X) != len(idx) or len(self.M) != len(idx):
            raise RuntimeError(f"X/M have {len(self.X)}/{len(self.M)} rows, "
                               f"index.parquet has {len(idx)}.")

        if n and n < len(idx):
            sel = np.sort(np.random.default_rng(seed).choice(len(idx), n,
                                                             replace=False))
            self.X, self.M = self.X[sel], self.M[sel]
            for at in ("V", "config_id", "layer", "n_launched", "N"):
                setattr(self, at, getattr(self, at)[sel])

        self.n = len(self.config_id)
        self.T = self.N / self.n_launched[:, None]

        o = fr["origin"][1:NDET + 1] if legacy else fr["origin"]
        self.s = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(o, axis=0), axis=1))]
        ).astype(np.float32)
        if legacy and not np.allclose(self.s, S_REF, atol=0.06):
            raise RuntimeError(f"arc length {np.round(self.s, 1).tolist()} does "
                               f"not match the documented geometry {S_REF.tolist()}.")
        if self.X.shape[1:] != (self.ndet, self.X.shape[2], len(self.feats)):
            raise RuntimeError(f"X is {self.X.shape}, expected (n, {self.ndet}, C, "
                               f"{len(self.feats)}) for {self.names}/{self.feats}.")

    def __repr__(self):
        return f"Data(n={self.n}, path={str(self.path)!r})"

    def scaler(self):
        f = self.path / "scaler.npz"
        if not f.exists():
            raise FileNotFoundError(
                f"{f} is missing. Refitting it makes every previously measured "
                "number incomparable; restore it, or regenerate deliberately "
                "with bsims.data.fit_scaler.")
        z = np.load(f)
        return z["mu"], z["sd"]

    def name(self, k):
        return self.names[k - 1]

    def cloud(self, i, k, standardise=None):
        x = np.asarray(self.X[i, k - 1], np.float32)[np.asarray(self.M[i, k - 1])]
        if standardise is not None:
            mu, sd = standardise
            x = (x - mu[k - 1]) / sd[k - 1]
        return x

    @property
    def trials(self):
        return np.concatenate([self.n_launched[:, None],
                               self.N[:, :-1].astype(np.float64)], axis=1)

    def survival_targets(self):
        k = self.n_pinned
        return self.trials[:, k:], self.N[:, k:].astype(np.float64)


def load(path=None, n=0, seed=0) -> Data:
    return Data(path, n=n, seed=seed)


def load_splits(path=None):
    f = root(path) / "splits.npz"
    if not f.exists():
        raise FileNotFoundError(f"{f} is missing; every recorded number is "
                                "conditioned on this exact draw.")
    return np.load(f)


def layer_masks(z, split):
    idx = z[split]
    layers = (set(np.asarray(z["layer"]).astype(str).tolist())
              if "layer" in z.files else None)
    out = {"all": np.ones(len(idx), bool)}
    for k in z.files:
        if not k.startswith(f"{split}_"):
            continue
        name = k[len(split) + 1:]
        if layers is not None and name not in layers:
            continue
        m = np.asarray(z[k])
        if m.dtype != bool or m.shape != (len(idx),):
            continue
        out[name] = m
    return idx, out


def fit_scaler(path=None, chunk=2500):
    p = root(path)
    X = np.load(p / "X.npy", mmap_mode="r")
    M = np.load(p / "M.npy", mmap_mode="r")
    ndet, nfeat = X.shape[1], X.shape[3]
    c = np.zeros(ndet)
    s1 = np.zeros((ndet, nfeat))
    s2 = np.zeros((ndet, nfeat))
    for a in range(0, len(X), chunk):
        x, m = np.asarray(X[a:a + chunk], np.float64), np.asarray(M[a:a + chunk])
        for k in range(ndet):
            v = x[:, k][m[:, k]]
            c[k] += len(v)
            s1[k] += v.sum(0)
            s2[k] += (v ** 2).sum(0)
    if (c < 100).any():
        raise RuntimeError(f"too few ions on some plane: {c.tolist()}")
    mu = s1 / c[:, None]
    sd = np.sqrt(np.maximum(s2 / c[:, None] - mu ** 2, 0.0))
    sd = np.where(sd < 1e-8, 1.0, sd)
    fr = np.load(p / "frames.npz")
    feats = fr["feats"] if "feats" in fr.files else np.array(FEATS)
    np.savez(p / "scaler.npz", mu=mu, sd=sd, n_ions=c.astype(np.int64), feats=feats)
    return mu, sd
