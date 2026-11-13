from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo/data"


def _feat(V, rng=None, u=None, v=None, tsa=290.0, tw=340.0, tout=270.0, occn=0.0, hour=6.0):
    from aether.zone_model import make_features
    rng = rng or np.random.default_rng(0)
    u = rng.uniform(0, 1, V) if u is None else np.broadcast_to(u, (V,)).astype(float)
    v = rng.uniform(0, 1, V) if v is None else np.broadcast_to(v, (V,)).astype(float)
    return make_features(u=u, v=v, tsa=np.full(V, tsa), tw=tw, tout=tout, occn=occn, hour=hour)


def test_rollout_matches_recursion():
    from aether.zone_model import TAU, regressors, rollout
    rng = np.random.default_rng(1)
    theta = rng.uniform(0, 0.3, (2, 10))
    T0, Tm0 = np.array([293.0, 291.0]), np.array([292.0, 292.5])
    steps = [_feat(2, rng, hour=6 + i / 12) for i in range(3)]
    prev = _feat(2, rng)
    traj = rollout(theta, T0, Tm0, steps, prev)
    T, Tm, fp = T0.copy(), Tm0.copy(), prev
    for i, f in enumerate(steps):
        x = np.stack([(f["tout"] - T) / 10, f["u"] * (f["tsa"] - T) / 10, f["v"] * (f["tw"] - T) / 50,
                      fp["u"] * (fp["tsa"] - T) / 10, fp["v"] * (fp["tw"] - T) / 50, (Tm - T) / 10,
                      np.full(2, f["occn"]), np.full(2, f["hs"]), np.full(2, f["hc"]), np.ones(2)], -1)
        np.testing.assert_allclose(regressors(T, Tm, f, fp), x)
        Tm = Tm + (T - Tm) / TAU
        T = T + (x * theta).sum(1)
        fp = f
        np.testing.assert_allclose(traj[i + 1], T)


def test_kf_recovers_synthetic():
    from aether.identifier import Prior, ZoneIdentifier
    from aether.zone_model import TAU, regressors
    rng = np.random.default_rng(2)
    V = 3
    theta = np.array([[0.15, 2.0, 4.0, 0.5, 0.5, 3.0, 0.1, 0.02, 0.01, 0.0]] * V) * rng.uniform(0.7, 1.3, (V, 1))
    # vague prior (P0=4) => slow forgetting (q_rw at the lower tuning bound); stationary truth
    ident = ZoneIdentifier(V, Prior(m0=np.zeros(10), P0=np.full(10, 4.0), R=0.02 ** 2), q_rw=1e-6)
    T, Tm, fp = np.full(V, 293.0), np.full(V, 293.0), _feat(V, rng)
    for t in range(2016):
        f = _feat(V, rng, tout=265 + 10 * np.sin(t / 50), occn=float(rng.random() > 0.6), hour=(t / 12) % 24)
        x = regressors(T, Tm, f, fp)
        Tn = T + (x * theta).sum(1) + rng.normal(0, 0.02, V)
        ident.update(x, Tn - T)
        Tm, T, fp = Tm + (T - Tm) / TAU, Tn, f
    for j in (0, 1, 2, 5):  # a, b, r, k
        err = np.abs(ident.theta[:, j] - theta[:, j])
        assert np.all(err < 0.1 * np.abs(theta[:, j]) + 0.02), (j, ident.theta[:, j], theta[:, j])


def _replay(npz):
    """Snapshots from a demo log (already physical units) via an identity-normalised parser."""
    from aether.obs import ObsParser
    d = np.load(npz)
    meta = eval(str(d["meta"]), {"np": np})
    fake = {"observation_field_names": list(d["fields"]), "observation_normalization_metadata": {},
            "ahu_zone_assignments": meta["ahu_zone_assignments"], "zone_action_indices": meta["zone_action_indices"]}
    parser = ObsParser(fake)
    return parser, [parser.parse(o) for o in d["obs"].astype(np.float64)]


def test_simulator_accuracy():
    from aether.identifier import Prior, ZoneIdentifier, fit_batch
    from aether.zone_model import features, regressors

    def design(parser, snaps):
        F = [features(s, parser) for s in snaps]
        Tm = [snaps[0].T]
        for s in snaps[:-1]:
            Tm.append(Tm[-1] + (s.T - Tm[-1]) / 24)
        X = np.stack([regressors(snaps[t].T, Tm[t], F[t], F[t - 1]) for t in range(1, len(snaps) - 1)])
        y = np.stack([snaps[t + 1].T - snaps[t].T for t in range(1, len(snaps) - 1)])
        return X, y

    fits = [fit_batch(*design(*_replay(DEMO / f"building_0502__{w}__pid_water.npz"))) for w in ("fairbanks_1", "phoenix_5")]
    ths = np.concatenate([f[0] for f in fits])
    prior = Prior(m0=np.median(ths, 0), P0=np.var(ths, 0) + 1e-8, R=float(np.mean([f[1] for f in fits])))

    parser, snaps = _replay(DEMO / "building_train_1__phoenix_5__pid_water.npz")
    ident = ZoneIdentifier(len(parser.vavs), prior)
    for s in snaps:
        ident.observe(s, features(s, parser))
    assert ident.n_scored.min() > 1000
    med = np.median(ident.rmse_since(288))  # 1-h RMSE over days 2-7, per zone
    assert med <= 0.5, med


def test_nonfinite_falls_back():
    from aether.identifier import Prior, ZoneIdentifier
    from aether.zone_model import regressors
    V = 3
    prior = Prior(m0=np.full(10, 0.1), P0=np.full(10, 1.0), R=1e-3)
    ident = ZoneIdentifier(V, prior)
    rng = np.random.default_rng(3)
    T, Tm = np.full(V, 293.0), np.full(V, 293.0)
    for _ in range(50):
        f = _feat(V, rng)
        ident.update(regressors(T, Tm, f, f), rng.normal(0, 0.05, V))
    y = rng.normal(0, 0.05, V)
    y[1] = np.nan
    ident.update(regressors(T, Tm, f, f), y)
    assert np.all(np.isfinite(ident.theta)) and np.all(np.isfinite(ident.sd))
    np.testing.assert_allclose(ident.theta[1], prior.m0)
    assert not ident.trusted[1]
    assert not np.allclose(ident.theta[0], prior.m0)


def test_airflow_feature_scales_with_sqrt_pressure():
    """Zone airflow in the simulator is damper x max_flow x sqrt(P / 2000); the model's u must follow it."""
    from aether.zone_model import features
    from tests.test_policy import _snap

    class P1:
        zone_ahu = np.zeros(3, int)
        n_occ_ref = 6.0

    s = _snap(V=3)
    s.ahu_on = np.ones(1, bool)
    s.damper = np.array([0.2, 0.5, 1.0])
    s.pressure = np.array([500.0])
    np.testing.assert_allclose(features(s, P1())["u"], s.damper * 0.5)
    s.pressure = np.array([2000.0])
    np.testing.assert_allclose(features(s, P1())["u"], s.damper)
    s.ahu_on = np.zeros(1, bool)
    np.testing.assert_allclose(features(s, P1())["u"], 0.0)


def test_offset_estimator_recovers_constant_disturbance():
    """Offset-free prediction: a frozen prior plus an EMA of one-step residuals absorbs a constant
    unmodelled heat input (sun, people the model under-counts) into the bias column."""
    from aether.identifier import Prior, ZoneIdentifier
    from aether.zone_model import TAU, regressors
    rng = np.random.default_rng(4)
    V = 3
    m0 = np.array([0.15, 2.0, 4.0, 0.5, 0.5, 3.0, 0.1, 0.02, 0.01, 0.0])
    base = ZoneIdentifier(V, Prior(m0=m0, P0=np.full(10, 1e-6), R=1e-4), frozen=True)
    ident = ZoneIdentifier(V, Prior(m0=m0, P0=np.full(10, 1e-6), R=1e-4), frozen=True, offset_alpha=1 / 12)
    T, Tm, fp = np.full(V, 293.0), np.full(V, 293.0), _feat(V, rng)
    snap = type("S", (), {})()
    for t in range(600):
        f = _feat(V, rng, tout=270 + 5 * np.sin(t / 40), hour=(t / 12) % 24)
        snap.T = T
        for i in (base, ident):
            i.observe(snap, f)
        Tn = T + (regressors(T, Tm, f, fp) * m0).sum(1) + 0.05 + rng.normal(0, 0.01, V)
        Tm, T, fp = Tm + (T - Tm) / TAU, Tn, f
    np.testing.assert_allclose(ident.bias, 0.05, atol=0.01)
    np.testing.assert_allclose(ident.theta[:, 9], m0[9] + ident.bias)
    np.testing.assert_allclose(base.theta, np.repeat(m0[None], V, 0))  # default: no offset


def test_actuation_scale_converges_to_truth():
    """A zone whose real airflow/reheat response is half the prior's: the per-zone scale g converges near 0.5;
    with adaptation off g stays exactly 1."""
    import numpy as np
    from aether.identifier import Prior, ZoneIdentifier
    from aether.zone_model import make_features, regressors
    prior = Prior.load("aether/prior.json")
    true = prior.m0.copy()
    true[1:5] *= 0.5
    rng = np.random.default_rng(0)
    V = 3
    on = ZoneIdentifier(V, prior, frozen=True, offset_alpha=1 / 12, act_rho=1 - 1 / 864)
    off = ZoneIdentifier(V, prior, frozen=True, offset_alpha=1 / 12)
    T = np.full(V, 293.0)
    Tm, fp = T.copy(), None
    for t in range(288 * 5):
        heat = (t // 6) % 2 == 0  # alternate heating and cooling every 30 min: excitation
        f = make_features(np.full(V, 0.3 if heat else 0.6), np.full(V, 1.0 if heat else 0.0),
                          np.full(V, 286.0), 340.0, 273.0, 0.0, (t / 12) % 24)
        fp = fp or f
        for ident in (on, off):
            ident.observe(type("S", (), {"T": T})(), f)
        x = regressors(T, Tm, f, fp)
        Tm = Tm + (T - Tm) / 24
        T = T + x @ true + rng.normal(0, 0.02, V)
        fp = f
    assert np.all(np.abs(on.g - 0.5) < 0.15), on.g
    assert np.array_equal(off.g, np.ones(V))
