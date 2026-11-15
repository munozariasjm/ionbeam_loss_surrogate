#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.preprocessing import StandardScaler

from bsims import checkpoint as ckpt
from bsims import data as D
from bsims import dispersion as DISP
from bsims.ensemble import ensemble_from_dir
from bsims.hazard import Head


def parse():
    p = argparse.ArgumentParser(description='Fit the beta-binomial dispersion head against a transmission model.')
    p.add_argument("--data", default=None, help="dataset dir [$BSIMS_DATA]")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--hazard", help="a single transmission checkpoint")
    g.add_argument("--ensemble", help="a directory of ensemble members")
    p.add_argument("--out", default="models/dispersion.eqx")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--max-steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-2)
    return p.parse_args()


def main():
    a = parse()
    d = D.load(a.data)
    z = D.load_splits(a.data)
    tr, va = np.asarray(z["train"]), np.asarray(z["val"])
    head = (ensemble_from_dir(a.ensemble)[0] if a.ensemble
            else Head(*ckpt.load_hazard(a.hazard)[:3]))
    m, s = d.survival_targets()
    p = DISP.conditional_survival(head.log_T(d.V), d.n_pinned)
    lp = np.log(p) - np.log1p(-p)
    sx = StandardScaler().fit(d.V[tr])
    Z = sx.transform(d.V).astype(np.float32)
    dh, info = DISP.fit(Z[tr], lp[tr], m[tr], s[tr], Z[va], lp[va], m[va], s[va],
                        seed=a.seed, max_steps=a.max_steps, lr=a.lr, hidden=a.hidden)
    hp = dict(n_features=int(d.V.shape[1]), n_dets=int(m.shape[1]), hidden=a.hidden,
              mean_=sx.mean_, scale_=sx.scale_, seed=a.seed, **info)
    out = ckpt.save(a.out, hp, dh)
    phi = DISP.predict_phi(dh, Z[va])
    print(json.dumps(dict(weights=str(out), steps=info["steps"],
                          val_nll=round(float(info["best_val_nll"]), 5),
                          median_phi_last_plane=round(float(np.median(phi[:, -1])), 2))))


if __name__ == "__main__":
    main()
