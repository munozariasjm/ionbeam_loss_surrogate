#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax

from bsims import checkpoint as ckpt
from bsims.ensemble import ensemble_from_dir


def parse():
    p = argparse.ArgumentParser(description='Gradient ascent on transmission through the surrogate.')
    p.add_argument("--models", default="models", help="directory of members")
    p.add_argument("--flow", default="models/flow_s0.eqx")
    p.add_argument("--meta", default="models/meta.npz")
    p.add_argument("--plane", type=int, default=10, help="plane to maximise")
    p.add_argument("--restarts", type=int, default=256)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top", type=int, default=3, help="candidates to report")
    p.add_argument("--separation", type=float, default=1.0,
                   help="minimum mutual distance between reported candidates, "
                        "in standardised voltage units")
    p.add_argument("--width-target", type=float, default=None,
                   help="RMS transverse width to hold, mm")
    p.add_argument("--width-plane", type=int, default=5)
    p.add_argument("--width-draws", type=int, default=96,
                   help="common random numbers inside the constraint")
    p.add_argument("--width-tol", type=float, default=0.05,
                   help="relative tolerance for calling a candidate feasible")
    p.add_argument("--dual-lr", type=float, default=0.1)
    p.add_argument("--rho", type=float, default=5.0, help="quadratic penalty")
    p.add_argument("--out", default="optimize.csv")
    return p.parse_args()


def main():
    a = parse()
    meta = np.load(a.meta, allow_pickle=True)
    ens, paths = ensemble_from_dir(a.models)
    lo = jnp.asarray(meta["box_lo"], jnp.float32)
    hi = jnp.asarray(meta["box_hi"], jnp.float32)
    names = [str(v) for v in meta["volt_names"]]
    ruler = np.asarray(meta["hz_sd"])[0]

    def to_volts(w):
        return lo + (hi - lo) * 0.5 * (jnp.tanh(w) + 1.0)

    def log_T(u):
        return ens.log_T(u)[:, a.plane - 1]

    constrained = a.width_target is not None
    if constrained:
        flw = ckpt.load_flow(a.flow, s=meta["s"])[0]
        s_w = jnp.float32(meta["s"][a.width_plane - 1])
        mu_w = jnp.asarray(meta["ion_mu"][a.width_plane - 1][:2], jnp.float32)
        sd_w = jnp.asarray(meta["ion_sd"][a.width_plane - 1][:2], jnp.float32)
        ls = jnp.float32(np.log(flw.sig_eval))
        Zfix = jax.random.normal(jax.random.PRNGKey(a.seed + 7), (a.width_draws, flw.nfeat))
        Zind = jax.random.normal(jax.random.PRNGKey(a.seed + 77), (2048, flw.nfeat))

        def width(u, Z):
            n, nz = u.shape[0], Z.shape[0]
            c = flw.ctx(u, jnp.full((n,), s_w), jnp.full((n,), ls))
            x = flw.inv(jnp.tile(Z, (n, 1)), jnp.repeat(c, nz, axis=0))
            xy = x.reshape(n, nz, flw.nfeat)[:, :, :2] * sd_w + mu_w
            return jnp.sqrt(xy[:, :, 0].var(1) + xy[:, :, 1].var(1))

        eval_width = jax.jit(lambda u: width(u, Zind))

    opt = optax.adam(a.lr)

    @jax.jit
    def step(w, state, lam):
        def loss(w):
            u = to_volts(w)
            lt = log_T(u)
            if not constrained:
                return -jnp.sum(lt), (lt, jnp.zeros_like(lt))
            c = width(u, Zfix) - a.width_target
            return -(jnp.sum(lt) - jnp.sum(lam * c + 0.5 * a.rho * c * c)), (lt, c)

        (_, (lt, c)), g = jax.value_and_grad(loss, has_aux=True)(w)
        upd, state = opt.update(g, state)
        return optax.apply_updates(w, upd), state, \
            jnp.clip(lam + a.dual_lr * c, -50.0, 50.0), lt, c

    rng = np.random.default_rng(a.seed)
    f0 = np.clip(rng.uniform(0.0, 1.0, (a.restarts, len(names))), 0.02, 0.98)
    w = jnp.asarray(np.arctanh(2.0 * f0 - 1.0), jnp.float32)
    state = opt.init(w)
    lam = jnp.zeros(a.restarts)

    t10_start = 100.0 * np.exp(np.asarray(log_T(to_volts(w))))
    for _ in range(a.steps):
        w, state, lam, lt, c = step(w, state, lam)

    U = np.asarray(to_volts(w))
    mean, sd = (np.asarray(x) for x in ens.spread(U))
    t_mean = 100.0 * mean[:, a.plane - 1]
    t_sd = 100.0 * sd[:, a.plane - 1]
    widths = (np.asarray(eval_width(jnp.asarray(U))) if constrained
              else np.full(len(U), np.nan))
    feasible = (np.abs(widths - a.width_target) < a.width_tol * a.width_target
                if constrained else np.ones(len(U), bool))

    Uz = U / ruler
    order = np.argsort(-np.where(feasible, t_mean, -np.inf))
    keep = []
    for j in order:
        if not feasible[j]:
            break
        if all(np.linalg.norm(Uz[j] - Uz[q]) >= a.separation for q in keep):
            keep.append(int(j))
        if len(keep) == a.top:
            break

    best_in_data = float(np.max(meta["opt_meas_T10"]))
    print(f"{len(ens)} members, {a.restarts} restarts x {a.steps} steps, "
          f"maximising T_{a.plane}")
    print(f"start T{a.plane}: median {np.median(t10_start):.3f}%  "
          f"max {t10_start.max():.3f}%")
    for r, j in enumerate(keep, 1):
        line = (f"  #{r} predicted {t_mean[j]:6.3f} +- {t_sd[j]:.3f} %  "
                f"(member spread, not a calibrated interval)")
        if constrained:
            line += f"  width {widths[j]:.3f} mm"
        print(line)
    print(f"best SIMION-verified setting shipped with these weights: "
          f"{best_in_data:.2f}%")

    with open(a.out, "w", newline="") as f:
        cols = (["rank", "predicted_T_mean_pct", "predicted_T_sd_pct",
                 "width_mm", "feasible"] + names)
        wcsv = csv.DictWriter(f, fieldnames=cols)
        wcsv.writeheader()
        for r, j in enumerate(keep, 1):
            wcsv.writerow(dict(rank=r,
                               predicted_T_mean_pct=round(float(t_mean[j]), 4),
                               predicted_T_sd_pct=round(float(t_sd[j]), 4),
                               width_mm=round(float(widths[j]), 4),
                               feasible=int(feasible[j]),
                               **{n: round(float(U[j, i]), 3)
                                  for i, n in enumerate(names)}))
    np.savez(a.out.replace(".csv", ".npz"), u=U, t_mean=t_mean, t_sd=t_sd,
             width_mm=widths, feasible=feasible, keep=np.array(keep),
             volt_names=np.array(names))
    print(json.dumps(dict(csv=a.out, npz=a.out.replace(".csv", ".npz"),
                          n_reported=len(keep))))


if __name__ == "__main__":
    main()
