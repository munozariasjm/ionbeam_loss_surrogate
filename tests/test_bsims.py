import os
import pathlib

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bsims import checkpoint as ckpt
from bsims import data as D
from bsims import dispersion as DISP
from bsims import flow as F
from bsims import hazard as H
from bsims import train as T
from bsims.beamline import Beamline
from bsims.ensemble import ensemble_from_dir

MODELS = pathlib.Path(__file__).resolve().parents[1] / "models"
KEY = jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def released():
    ens, _ = ensemble_from_dir(MODELS)
    flw = ckpt.load_flow(MODELS / "flow_s0.eqx")[0]
    meta = np.load(MODELS / "meta.npz", allow_pickle=True)
    bl = Beamline(ens, flw, meta["ion_mu"], meta["ion_sd"], s=meta["s"])
    return ens, flw, bl, meta


def test_released_parameter_count():
    net = H.GRUHazard(22, 8, d=48, tied=False, key=KEY)
    assert H.n_params(net) == 126144


def test_transmission_is_a_survival_function(released):
    ens, _, _, meta = released
    lt = np.asarray(ens.log_T(meta["opt_U"]))
    assert lt.shape == (7, 10)
    assert np.allclose(lt[:, :H.DET_START], 0.0)
    assert (np.diff(lt, axis=1) <= 1e-6).all()
    assert (lt <= 1e-6).all()


def test_log_T_survives_where_T_underflows(released):
    ens, _, _, meta = released
    u = jnp.asarray(meta["opt_U"][1] * 100.0, jnp.float32)
    lt = ens.log_T(u[None])[0, -1]
    assert float(jnp.exp(lt)) == 0.0
    assert np.isfinite(float(lt)) and float(lt) < -87
    g = jax.grad(lambda v: ens.log_T(v[None])[0, -1])(u)
    assert np.isfinite(np.asarray(g)).all()


def test_ensemble_reproduces_the_released_predictions(released):
    ens, _, _, meta = released
    mean, sd = (np.asarray(x)[:, -1] * 100.0 for x in ens.spread(meta["opt_U"]))
    assert np.abs(mean - meta["opt_pred_mean"]).max() < 1e-3
    assert np.abs(sd - meta["opt_pred_sd"]).max() < 1e-3


def test_ensemble_members_are_distinct(released):
    ens, _, _, meta = released
    per_member = np.asarray(ens.member_log_T(meta["opt_U"]))[:, :, -1]
    assert len(ens) == 12
    assert per_member.std(0).min() > 1e-3


def test_binomial_nll_is_minimised_by_the_empirical_rate():
    m = jnp.asarray(np.full((4, 8), 1000.0), jnp.float32)
    s = jnp.asarray(np.full((4, 8), 400.0), jnp.float32)
    exact = jax.scipy.special.logit(jnp.clip(s / m, 1e-6, 1 - 1e-6))
    best = float(T.binomial_nll(exact, m, s))
    for delta in (-0.5, -0.1, 0.1, 0.5):
        assert float(T.binomial_nll(exact + delta, m, s)) > best


def test_log10_T_term_vanishes_on_an_exact_fit():
    m = jnp.asarray(np.full((4, 8), 1000.0), jnp.float32)
    s = jnp.asarray(np.full((4, 8), 400.0), jnp.float32)
    exact = jax.scipy.special.logit(jnp.clip(s / m, 1e-6, 1 - 1e-6))
    assert float(T.log10_T_mse(exact, m, s)) < 1e-8
    assert float(T.log10_T_mse(exact - 1.0, m, s)) > 0.1
    assert abs(float(T.objective(exact - 1.0, m, s, "logT"))
               - float(T.objective(exact - 1.0, m, s, "binom"))
               - 0.1 * float(T.log10_T_mse(exact - 1.0, m, s))) < 1e-5


def test_dead_cells_do_not_contribute():
    m = jnp.asarray(np.full((2, 8), 100.0), jnp.float32).at[1].set(0.0)
    s = jnp.asarray(np.full((2, 8), 40.0), jnp.float32).at[1].set(0.0)
    lg = jnp.zeros((2, 8))
    one = T.binomial_nll(lg[:1], m[:1], s[:1])
    assert abs(float(T.binomial_nll(lg, m, s)) - float(one)) < 1e-6


def test_flow_coupling_layers_are_invertible(released):
    _, flw, _, meta = released
    u = jnp.asarray(meta["opt_U"][1], jnp.float32)
    s = jnp.asarray(meta["s"][9], jnp.float32)
    c = flw.ctx(u[None], s[None], jnp.asarray([np.log(flw.sig_eval)], jnp.float32))
    z = jax.random.normal(KEY, (64, 6))
    c64 = jnp.repeat(c, 64, 0)
    x = flw.inv(z, c64)
    back = x
    for cp in flw.couplings:
        back, _ = cp(back, c64)
    assert float(jnp.abs(back - z).max()) < 1e-3


def test_flow_density_is_normalised_enough_to_be_a_density(released):
    _, flw, _, meta = released
    u, s = meta["opt_U"][1], meta["s"][9]
    x = F.sample(flw, u, s, 4096, KEY)
    lp = np.asarray(F.log_prob(flw, u, s, x))
    assert np.isfinite(lp).all()
    assert lp.std() > 0


def test_sampling_is_deterministic_in_the_key(released):
    _, flw, _, meta = released
    u, s = meta["opt_U"][1], meta["s"][9]
    a = np.asarray(F.sample(flw, u, s, 32, jax.random.PRNGKey(3)))
    b = np.asarray(F.sample(flw, u, s, 32, jax.random.PRNGKey(3)))
    c = np.asarray(F.sample(flw, u, s, 32, jax.random.PRNGKey(4)))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_empty_cloud_is_the_most_informative_observation(released):
    _, _, bl, meta = released
    u = jnp.asarray(meta["opt_U"][1], jnp.float32)
    empty = jnp.zeros((0, 6))
    v = float(bl.log_prob_cloud(u, 10, empty))
    assert np.isfinite(v) and v < 0
    expected = int(meta["n_launched"]) * float(bl.log_1m_T(u, 10))
    assert abs(v - expected) < 1e-3
    g = np.asarray(jax.grad(lambda w: bl.log_prob_cloud(w, 10, empty))(u))
    assert np.isfinite(g).all() and np.abs(g).max() > 0


def test_likelihood_carries_the_binomial_failure_term(released):
    _, _, bl, meta = released
    u = jnp.asarray(meta["opt_U"][1], jnp.float32)
    X = bl.sample(u, 10, 64, KEY)
    full = float(bl.log_prob_cloud(u, 10, X, n_launched=10000))
    sub = float(bl.log_prob_cloud(u, 10, X, n_launched=64))
    assert full < sub


def test_composed_likelihood_is_differentiable(released):
    _, _, bl, meta = released
    u = jnp.asarray(meta["opt_U"][1], jnp.float32)
    X = bl.sample(u, 10, 128, KEY)
    g = np.asarray(jax.grad(
        lambda w: bl.log_prob_cloud(w, 10, X, n_launched=len(X)))(u))
    assert np.isfinite(g).all() and np.abs(g).max() > 0


def test_pinned_planes_have_no_failure_term(released):
    _, _, bl, meta = released
    u = jnp.asarray(meta["opt_U"][1], jnp.float32)
    for k in (1, 2):
        assert float(bl.log_1m_T(u, k)) == 0.0


def test_binomial_chain_is_calibrated_when_the_mean_is_exact():
    rng = np.random.default_rng(0)
    p = np.clip(rng.uniform(0.6, 0.99, (400, 8)), 1e-9, 1 - 1e-9)
    truth = DISP.survival_chain(p, None, n_draws=1, n_launched=10000, seed=1)[0]
    draws = DISP.survival_chain(p, None, n_draws=2000, n_launched=10000, seed=2)
    cov = DISP.coverage(draws, truth, level=0.68)
    assert np.all(np.abs(cov - 0.68) < 0.06)


def test_dispersion_widens_the_interval():
    rng = np.random.default_rng(0)
    p = np.full((200, 8), 0.9)
    phi = np.full((200, 8), 5.0)
    wide = DISP.survival_chain(p, phi, n_draws=1000, n_launched=10000, seed=0)
    tight = DISP.survival_chain(p, None, n_draws=1000, n_launched=10000, seed=0)
    lo_w, hi_w = DISP.predict_interval(wide)
    lo_t, hi_t = DISP.predict_interval(tight)
    assert ((hi_w - lo_w) > (hi_t - lo_t)).all()
    assert DISP.var_inflation(10000.0, 5.0) > 1.0


def test_released_dispersion_head_loads():
    head, mu, sd, h = ckpt.load_dispersion(MODELS / "dispersion.eqx")
    meta = np.load(MODELS / "meta.npz", allow_pickle=True)
    phi = DISP.predict_phi(head, (meta["opt_U"] - mu) / sd)
    assert phi.shape == (7, 8) and (phi > 0).all()


@pytest.mark.skipif(not os.environ.get("BSIMS_DATA"),
                    reason="set BSIMS_DATA to run the dataset checks")
def test_dataset_contract():
    d = D.load()
    z = D.load_splits()
    assert d.V.shape[1] == 22 and d.N.shape[1] == 10
    assert (np.diff(d.N, axis=1) <= 0).all()
    assert np.allclose(d.s, D.S_REF, atol=0.06)
    m, s = d.survival_targets()
    assert m.shape == s.shape == (d.n, 8)
    assert (s <= m).all()
    assert np.array_equal(m[:, 0], d.N[:, 1])
    blocks = [np.asarray(z[k]) for k in ("train", "val", "test")]
    assert len(set(np.concatenate(blocks).tolist())) == sum(map(len, blocks))


def test_flow_sigma_range_is_carried_by_the_model():
    import dataclasses
    V, s = np.zeros((2, 22), np.float32), D.S_REF
    a = F.build(V, s, F.Cfg(width=8, sig_min=0.02, sig_max=0.2), KEY)
    assert (a.sig_min, a.sig_max) == (0.02, 0.2)
    b = dataclasses.replace(a, sig_min=F.SIG_MIN, sig_max=F.SIG_MAX)
    ctx = lambda m, sig: np.asarray(m.ctx(jnp.zeros((1, 22)), jnp.zeros(1),
                                          jnp.log(jnp.full(1, sig))))
    assert np.abs(ctx(a, 0.02) - ctx(b, F.SIG_MIN)).max() < 1e-5
    assert np.abs(ctx(a, 0.02) - ctx(b, 0.02)).max() > 1e-3
    assert F.with_sig_eval(a, 0.05).sig_min == 0.02


def test_flow_sigma_range_survives_a_round_trip(tmp_path):
    V, s = np.zeros((2, 22), np.float32), D.S_REF
    cfg = F.Cfg(width=8, sig_min=0.02, sig_max=0.2, sig_eval=0.02)
    m = F.build(V, s, cfg, KEY)
    p = ckpt.save(tmp_path / "f.eqx",
                  dict(width=8, sig_eval=0.02, sig_min=0.02, sig_max=0.2,
                       n_voltages=22, s=np.asarray(s, np.float64)), m)
    back = ckpt.load_flow(p)[0]
    assert (back.sig_min, back.sig_max) == (0.02, 0.2)


def test_released_flow_keeps_the_default_sigma_range(released):
    _, flw, _, meta = released
    assert (flw.sig_min, flw.sig_max) == (F.SIG_MIN, F.SIG_MAX)
    lp = np.asarray(F.log_prob(flw, meta["opt_U"][1], meta["s"][9],
                               F.sample(flw, meta["opt_U"][1], meta["s"][9],
                                        16, KEY)))
    assert np.isfinite(lp).all()
