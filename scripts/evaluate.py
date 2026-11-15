#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import pathlib

import jax
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, roc_curve

from bsims import checkpoint as ckpt
from bsims import data as D
from bsims import dispersion as DISP
from bsims import flow as F
from bsims.beamline import Beamline
from bsims.ensemble import ensemble_from_dir
from bsims.hazard import Head

SPLITS = ("test", "test_natural", "test_top", "to_optimum")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=None, help="dataset dir [$BSIMS_DATA]")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--hazard", help="a single transmission checkpoint")
    g.add_argument("--ensemble", help="a directory of ensemble members")
    p.add_argument("--flow", default=None, help="phase-space checkpoint")
    p.add_argument("--dispersion", default=None,
                   help="dispersion head, for count coverage")
    p.add_argument("--plane", type=int, default=10, help="plane for shape metrics")
    p.add_argument("--n-cfg", type=int, default=64, help="configurations per row")
    p.add_argument("--ions", type=int, default=64, help="ions per cloud")
    p.add_argument("--splits", nargs="+", default=list(SPLITS))
    p.add_argument("--chain-draws", type=int, default=2000,
                   help="Monte Carlo draws behind the count intervals")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="metrics.csv")
    return p.parse_args()


def transmission_metrics(log_t10, delivered, t_true_pp, thr):
    t_pred = 100.0 * np.exp(log_t10)
    yhat = log_t10 >= thr
    tpr = float(yhat[delivered].mean()) if delivered.any() else np.nan
    tnr = float((~yhat[~delivered]).mean()) if (~delivered).any() else np.nan
    m = dict(n=len(log_t10), n_deliv=int(delivered.sum()),
             deliv_frac=round(float(delivered.mean()), 6),
             auc=(round(float(roc_auc_score(delivered, log_t10)), 6)
                  if 0 < delivered.sum() < len(log_t10) else None),
             balacc=round((tpr + tnr) / 2, 6))
    k = max(1, round(0.01 * len(log_t10)))
    top = set(np.argsort(-t_true_pp)[:k].tolist())
    m.update(spearman=round(float(spearmanr(t_true_pp, t_pred).statistic), 6),
             top1_recall=round(len(top & set(np.argsort(-t_pred)[:k].tolist())) / k, 6))

    a, b = t_true_pp[delivered], t_pred[delivered]
    if len(a) < 2:
        return m
    ss = float(((a - a.mean()) ** 2).sum())
    la, lb = np.log10(np.clip(a, 1e-12, None)), np.log10(np.clip(b, 1e-12, None))
    lss = float(((la - la.mean()) ** 2).sum())
    m.update(
        r2=round(1 - float(((a - b) ** 2).sum()) / ss, 6) if ss > 0 else None,
        log_r2=round(1 - float(((la - lb) ** 2).sum()) / lss, 6) if lss > 0 else None,
        rmse_pp=round(float(np.sqrt(((a - b) ** 2).mean())), 6),
        bias_pp=round(float((b - a).mean()), 6))
    return m


def shape_metrics(bl, d, idx, rng, k, n_cfg, ions):
    kept = np.asarray(d.M[idx, k - 1]).sum(1)
    e = np.asarray(idx)[kept >= ions]
    if len(e) == 0:
        return dict(flow_n_cfg=0)
    sel = rng.choice(e, min(n_cfg, len(e)), replace=False)
    mu, sd = d.scaler()
    per_ion, cloud = [], []
    for j in sel:
        X = d.cloud(int(j), k)[:ions]
        xs = (X - mu[k - 1]) / sd[k - 1]
        per_ion.append(float(np.mean(np.asarray(
            F.log_prob(bl.flow, d.V[j], d.s[k - 1], xs, bl.sig_eval)))))
        cloud.append(float(bl.log_prob_cloud(d.V[j], k, X, n_launched=len(X))))
    return dict(flow_n_cfg=len(sel),
                flow_logp_ion=round(float(np.mean(per_ion)), 4),
                composed_logp_cloud=round(float(np.mean(cloud)), 4))


def main():
    a = parse()
    d = D.load(a.data)
    z = D.load_splits(a.data)
    rng = np.random.default_rng(a.seed)

    if a.ensemble:
        head, paths = ensemble_from_dir(a.ensemble)
        model_id = f"ensemble[{len(head)}]:{pathlib.Path(a.ensemble).name}"
    else:
        head = Head(*ckpt.load_hazard(a.hazard)[:3])
        model_id = pathlib.Path(a.hazard).name

    mu, sd = d.scaler()
    flw = ckpt.load_flow(a.flow, s=d.s)[0] if a.flow else None
    bl = Beamline(head, flw, mu, sd, s=d.s) if flw else None

    log_t = np.asarray(head.log_T(d.V), np.float64)[:, D.DET10]
    delivered = d.N[:, D.DET10] > 0
    t_true = 100.0 * d.N[:, D.DET10] / d.n_launched

    va = np.asarray(z["val"])
    fpr, tpr, thr = roc_curve(delivered[va], log_t[va])
    cut = float(thr[np.argmax((tpr + (1 - fpr)) / 2)])

    phi = None
    if a.dispersion:
        dh, dmu, dsd, _ = ckpt.load_dispersion(a.dispersion)
        phi = DISP.predict_phi(dh, (d.V - dmu) / dsd)
    T_meas = d.N[:, d.n_pinned:] / d.n_launched[:, None]

    rows = []
    for split in a.splits:
        if split not in z.files:
            continue
        idx, masks = D.layer_masks(z, split)
        for layer, mask in masks.items():
            i = idx[mask]
            if len(i) < 2:
                continue
            r = dict(model=model_id, split=split, layer=layer)
            r.update(transmission_metrics(log_t[i], delivered[i], t_true[i], cut))
            if phi is not None:
                p = DISP.conditional_survival(head.log_T(d.V[i]), d.n_pinned)
                draws = DISP.survival_chain(p, phi[i], n_draws=a.chain_draws,
                                            n_launched=int(d.n_launched[i[0]]),
                                            seed=a.seed)
                for lvl in (0.68, 0.95):
                    cov = DISP.coverage(draws, T_meas[i], level=lvl)
                    r[f"cov{int(lvl * 100)}_det10"] = round(float(cov[-1]), 6)
            if bl is not None:
                r.update(shape_metrics(bl, d, i, rng, a.plane, a.n_cfg, a.ions))
            rows.append(r)
            print(f"{split:>13s} {layer:>8s}  n={r['n']:>6d}  auc={r.get('auc')}  "
                  f"R2={r.get('r2')}  logR2={r.get('log_r2')}  "
                  f"RMSE={r.get('rmse_pp')} pp")

    cols = sorted({k for r in rows for k in r},
                  key=lambda k: (k not in ("model", "split", "layer"), k))
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
