#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import numpy as np

from bsims import checkpoint as ckpt
from bsims import data as D
from bsims.train import Survival

PATIENCE_WINDOW = 1140


def parse():
    p = argparse.ArgumentParser(description='Train one transmission head.')
    p.add_argument("--data", default=None, help="dataset dir [$BSIMS_DATA]")
    p.add_argument("--out", default="models/hazard.eqx")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=0,
                   help="resample the training rows with this rng seed; 0 = off")
    p.add_argument("--subsample", type=float, default=1.0,
                   help="fraction of distinct training rows, for a learning curve")
    p.add_argument("--obj", default="logT", choices=("logT", "binom"))
    p.add_argument("--lambda-log", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--deliverers-only", action="store_true",
                   help="ablation: drop configurations that deliver no ions")
    p.add_argument("--gate-prior", default="auto", choices=("auto", "segments", "flat"),
                   help="SegGate init: DerickV2 electrode groups, or every gate "
                        "half open. auto = segments for the DerickV2 store, else flat")
    return p.parse_args()


def main():
    a = parse()
    d = D.load(a.data)
    z = D.load_splits(a.data)
    tr, va = np.asarray(z["train"]), np.asarray(z["val"])
    m, s = d.survival_targets()

    rng = np.random.default_rng(a.bootstrap or a.seed)
    if a.subsample < 1.0:
        tr = rng.choice(tr, round(a.subsample * len(tr)), replace=False)
    if a.bootstrap:
        tr = rng.choice(tr, len(tr), replace=True)

    flat = a.gate_prior == "flat" or (a.gate_prior == "auto" and not d.legacy)
    gate_prior = np.zeros((s.shape[1], d.V.shape[1]), np.float32) if flat else None
    sv = Survival(d.V.shape[1], s.shape[1], seed=a.seed, d=a.dim, obj=a.obj,
                  lambda_log=a.lambda_log, lr=a.lr, batch=a.batch,
                  wd=a.wd, clip=a.clip,
                  patience=a.patience, max_steps=a.max_steps,
                  deliverers_only=a.deliverers_only, n_pinned=d.n_pinned,
                  gate_prior=gate_prior)
    sv.fit(d.V[tr], m[tr], s[tr], d.V[va], m[va], s[va],
           patience_window=PATIENCE_WINDOW)

    hp = dict(n_features=d.V.shape[1], n_dets=int(s.shape[1]), d=a.dim,
              tied=False, n_pinned=int(d.n_pinned), gate_prior="flat" if flat else "segments",
              mean_=sv.sx.mean_, scale_=sv.sx.scale_, obj=a.obj,
              lambda_log=a.lambda_log, lr=a.lr, batch=a.batch, wd=a.wd,
              clip=a.clip, seed=a.seed,
              bootstrap=a.bootstrap, subsample=a.subsample,
              n_train=len(tr), n_train_unique=int(len(np.unique(tr))),
              max_steps=a.max_steps, steps=sv.steps_completed)
    p = ckpt.save(a.out, hp, sv.net)
    print(json.dumps(dict(weights=str(p), params=sv.n_params(),
                          steps=sv.steps_completed, best_epoch=sv.best_epoch,
                          val_nll=round(sv.best_val, 6),
                          n_train=len(tr), seconds=round(sv.train_s, 1))))


if __name__ == "__main__":
    main()
