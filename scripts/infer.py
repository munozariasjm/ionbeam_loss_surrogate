#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv

import jax
import jax.numpy as jnp
import numpy as np
import optax

from bsims import checkpoint as ckpt
from bsims.ensemble import ensemble_from_dir

CHI2_68 = 2.2788746

def parse():
    p = argparse.ArgumentParser(description="Simulation Based Inference of the beamline's detector planes.")
    p.add_argument("--models", default="models")
    p.add_argument("--flow", default="models/flow_s0.eqx")
    p.add_argument("--meta", default="models/meta.npz")
    p.add_argument("--data", default=None,
                   help="dataset dir; adds the exit image and the true ions")
    p.add_argument("--optimum", type=int, default=1,
                   help="without --data, which released setting is observed")
    p.add_argument("--observed", type=int, nargs="+", default=[4, 7, 10])
    p.add_argument("--dark", type=int, nargs="+", default=[3, 6, 9])
    p.add_argument("--image-ions", type=int, default=256,
                   help="observed positions at the exit plane")
    p.add_argument("--image-draws", type=int, default=128,
                   help="flow draws behind the image likelihood")
    p.add_argument("--restarts", type=int, default=32)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--draws", type=int, default=192, help="voltage draws")
    p.add_argument("--ions", type=int, default=256, help="ions per draw")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="inference.csv")
    p.add_argument("--npz", default=None,
                   help="also save the predictive clouds, for plotting")
    return p.parse_args()


def log1mexp(a):
    a = jnp.minimum(a, -1e-7)
    return jnp.where(a > -0.6931472, jnp.log(-jnp.expm1(a)), jnp.log1p(-jnp.exp(a)))


def ellipse_area(xy, w):
    m = np.average(xy, 0, weights=w)
    c = np.cov(xy.T, aweights=w)
    return float(np.pi * CHI2_68 * np.sqrt(max(np.linalg.det(c), 0.0))), m


def observation(a, meta):
    if a.data is None:
        c = np.asarray(meta["opt_meas_counts"][a.optimum], np.float64)
        print(f"observing released setting {a.optimum}, measured T10 "
              f"{meta['opt_meas_T10'][a.optimum]:.2f}%")
        return c, int(meta["n_launched"]), None, None

    from bsims import data as D

    d = D.load(a.data)
    z = D.load_splits(a.data)
    idx = np.asarray(z["test_top"])
    i = int(idx[np.argmax(d.N[idx, D.DET10])])
    exit_plane = max(a.observed)
    img = d.cloud(i, exit_plane)[:a.image_ions, :2]
    print(f"observing held-out configuration {int(d.config_id[i])}, measured "
          f"T({d.name(exit_plane)}) "
          f"{100 * d.N[i, D.DET10] / d.n_launched[i]:.2f}%, "
          f"{len(img)} imaged ions at {d.name(exit_plane)}")
    truth = {k: d.cloud(i, k)[:, :2] for k in a.dark}
    return (d.N[i].astype(np.float64), int(d.n_launched[i]), img, truth)


def main():
    a = parse()
    meta = np.load(a.meta, allow_pickle=True)
    ens, _ = ensemble_from_dir(a.models)
    flw = ckpt.load_flow(a.flow, s=meta["s"])[0]
    s_all = jnp.asarray(meta["s"], jnp.float32)
    ion_mu = jnp.asarray(meta["ion_mu"], jnp.float32)
    ion_sd = jnp.asarray(meta["ion_sd"], jnp.float32)
    mu_v = jnp.asarray(meta["volt_mu"], jnp.float32)
    sd_v = jnp.asarray(meta["volt_sd"], jnp.float32)
    ls = jnp.float32(np.log(flw.sig_eval))
    nv = len(mu_v)

    names = ([str(x) for x in meta["names"]] if "names" in meta.files
             else [f"det{k}" for k in range(1, len(meta["s"]) + 1)])
    nm = lambda k: names[k - 1]
    counts, n_launched, image, truth = observation(a, meta)
    obs = sorted(a.observed)
    exit_plane = obs[-1]
    n_obs = jnp.asarray([counts[k - 1] for k in obs], jnp.float32)
    print("counts: " + ", ".join(f"{nm(k)}={int(counts[k-1])}" for k in obs)
          + f" of {n_launched} launched")

    volts = lambda z: z * sd_v + mu_v

    def count_ll(z):
        lt = ens.log_T(volts(z)[None])[0]
        prev_n, prev_lt, ll = jnp.float32(n_launched), jnp.float32(0.0), 0.0
        for i, k in enumerate(obs):
            lp = lt[k - 1] - prev_lt
            ll = ll + n_obs[i] * lp + (prev_n - n_obs[i]) * log1mexp(lp)
            prev_n, prev_lt = n_obs[i], lt[k - 1]
        return ll

    if image is not None:
        Y = jnp.asarray(image, jnp.float32)
        Zimg = jax.random.normal(jax.random.PRNGKey(a.seed + 3),
                                 (a.image_draws, flw.nfeat))

        def image_ll(z):
            u = volts(z)
            c = flw.ctx(u[None], s_all[exit_plane - 1][None], ls[None])
            x = flw.inv(Zimg, jnp.repeat(c, a.image_draws, 0))
            xy = x[:, :2] * ion_sd[exit_plane - 1][:2] + ion_mu[exit_plane - 1][:2]
            m = xy.mean(0)
            c2 = jnp.cov(xy.T) + 1e-6 * jnp.eye(2)
            r = Y - m
            sol = jnp.linalg.solve(c2, r.T).T
            return -0.5 * (jnp.sum(r * sol)
                           + len(Y) * jnp.linalg.slogdet(c2)[1])
    else:
        image_ll = lambda z: 0.0

    def neg_logpost(z):
        return -(count_ll(z) + image_ll(z) - 0.5 * jnp.sum(z ** 2))

    rng = np.random.default_rng(a.seed)
    Z = jnp.asarray(rng.normal(size=(a.restarts, nv)), jnp.float32)
    opt = optax.adam(a.lr)
    state = opt.init(Z)

    @jax.jit
    def step(Z, state):
        v, g = jax.value_and_grad(
            lambda Zb: jnp.sum(jax.vmap(neg_logpost)(Zb)))(Z)
        upd, state = opt.update(g, state)
        return optax.apply_updates(Z, upd), state, v

    for _ in range(a.steps):
        Z, state, _ = step(Z, state)
    z_map = jnp.asarray(Z[int(np.argmin(np.asarray(jax.vmap(neg_logpost)(Z))))])

    H = np.asarray(jax.hessian(neg_logpost)(z_map), np.float64)
    H = 0.5 * (H + H.T)
    ev, V = np.linalg.eigh(H)
    cov = V @ np.diag(1.0 / np.maximum(ev, 1.0)) @ V.T
    sd_post = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
    print(f"voltage posterior: median contraction {np.median(1.0/sd_post):.2f}x, "
          f"{int((ev > 10).sum())} of {nv} directions constrained by the data")

    L = np.linalg.cholesky(cov + 1e-9 * np.eye(nv))
    z_prior = rng.normal(size=(a.draws, nv))
    z_post = np.asarray(z_map) + rng.normal(size=(a.draws, nv)) @ L.T

    @jax.jit
    def cloud(zc, k, key):
        u = volts(zc)
        z = jax.random.normal(key, (a.ions, flw.nfeat))
        c = flw.ctx(u[None], s_all[k - 1][None], ls[None])
        x = flw.inv(z, jnp.repeat(c, a.ions, 0)) * ion_sd[k - 1] + ion_mu[k - 1]
        return x[:, :2], ens.log_T(u[None])[0, k - 1]

    def predictive(zs, k):
        xy, w = [], []
        for i, zc in enumerate(zs):
            x, lt = cloud(jnp.asarray(zc, jnp.float32), k,
                          jax.random.fold_in(jax.random.PRNGKey(a.seed), i))
            xy.append(np.asarray(x))
            w.append(np.full(a.ions, float(np.exp(lt))))
        return np.concatenate(xy), np.concatenate(w)

    store = dict(dark=np.asarray(a.dark), observed=np.asarray(obs),
                 names=np.asarray(names, dtype=str),
                 n_launched=np.int64(n_launched), counts=np.asarray(counts),
                 volt_sd_post=sd_post, n_constrained=np.int64((ev > 10).sum()),
                 n_volt=np.int64(nv))
    if image is not None:
        store["image"] = np.asarray(image, np.float32)
    rows = []
    for k in a.dark:
        Xp, wp = predictive(z_prior, k)
        Xq, wq = predictive(z_post, k)
        store[f"prior_{k}"], store[f"prior_w_{k}"] = Xp.astype(np.float32), wp
        store[f"post_{k}"], store[f"post_w_{k}"] = Xq.astype(np.float32), wq
        if truth is not None:
            store[f"true_{k}"] = np.asarray(truth[k], np.float32)
        (ap, _), (aq, mq) = ellipse_area(Xp, wp), ellipse_area(Xq, wq)
        r = dict(plane=k, detector=nm(k), prior_area_mm2=round(ap, 4),
                 posterior_area_mm2=round(aq, 4),
                 contraction=round(ap / max(aq, 1e-12), 2),
                 posterior_x1_mm=round(float(mq[0]), 4),
                 posterior_x2_mm=round(float(mq[1]), 4))
        line = (f"  {nm(k):<6s} 68% area  prior {ap:9.2f}  posterior {aq:8.2f} mm^2"
                f"  contraction {ap / max(aq, 1e-12):7.1f}x")
        if truth is not None:
            t = truth[k]
            r["true_x1_mm"] = round(float(t[:, 0].mean()), 4)
            r["true_x2_mm"] = round(float(t[:, 1].mean()), 4)
            r["centroid_err_mm"] = round(
                float(np.hypot(*(np.asarray(mq) - t.mean(0)))), 4)
            line += f"  centroid err {r['centroid_err_mm']:.3f} mm"
        rows.append(r)
        print(line)

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", a.out)
    if a.npz:
        np.savez_compressed(a.npz, **store)
        print("wrote", a.npz)


if __name__ == "__main__":
    main()
