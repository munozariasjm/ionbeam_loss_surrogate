#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time

import jax
import numpy as np

from bsims import checkpoint as ckpt
from bsims import data as D
from bsims import flow as F


def parse():
    p = argparse.ArgumentParser(description='Train the phase-space head.')
    p.add_argument("--data", default=None, help="dataset dir [$BSIMS_DATA]")
    p.add_argument("--out", default="models/flow.eqx")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--lr", type=float, default=F.LR)
    p.add_argument("--batch", type=int, default=64, help="configurations per step")
    p.add_argument("--ions", type=int, default=64, help="ions per configuration")
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--sig-min", type=float, default=F.SIG_MIN)
    p.add_argument("--sig-max", type=float, default=F.SIG_MAX)
    p.add_argument("--sig-eval", type=float, default=F.SIG_MIN)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    a = parse()
    if a.lr >= 1e-2:
        raise SystemExit(f"--lr {a.lr}: 1e-2 is the tuned value WITHOUT sigma "
                         "conditioning and is measurably worse with it. Every "
                         "reported number was measured at 3e-3.")
    d = D.load(a.data)
    tr = np.asarray(D.load_splits(a.data)["train"])
    cfg = F.Cfg(steps=a.steps, batch=a.batch, lr=a.lr, width=a.width,
                ions=a.ions, seed=a.seed, sig_min=a.sig_min, sig_max=a.sig_max,
                sig_eval=a.sig_eval, nfeat=int(d.X.shape[3]))

    key = jax.random.PRNGKey(a.seed)
    t0 = time.time()
    model = F.train(F.build(d.V[tr], d.s, cfg, key), d, tr, cfg, key,
                    verbose=a.verbose)
    dt = time.time() - t0

    hp = dict(width=a.width, sig_eval=a.sig_eval, sig_min=a.sig_min,
              sig_max=a.sig_max, steps=a.steps, lr=a.lr, batch=a.batch,
              ions=a.ions, seed=a.seed, n_voltages=int(d.V.shape[1]),
              nfeat=int(d.X.shape[3]), s=np.asarray(d.s, np.float64))
    p = ckpt.save(a.out, hp, model)
    print(json.dumps(dict(weights=str(p), params=F.n_params(model),
                          steps=a.steps, seconds=round(dt, 1))))


if __name__ == "__main__":
    main()
