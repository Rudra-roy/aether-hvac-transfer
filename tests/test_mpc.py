import numpy as np

from tests.test_policy import META_SCHED, _snap

C = 273.15


class _Parser:
    def __init__(self, V=4, K=2):
        self.vavs = [f"VAV_{i + 1}" for i in range(V)]
        self.ahus = [f"ahu_{k + 1}" for k in range(K)]
        self.zone_ahu = np.repeat(np.arange(K), V // K)
        self.groups = [np.flatnonzero(self.zone_ahu == k) for k in range(K)]
        self.n_occ_ref = 2.0 * V


def _ident(V, a=0.1, b=2.0, r=4.0, k=2.0, T=None):
    from aether.identifier import Prior, ZoneIdentifier
    from aether.zone_model import make_features
    m0 = np.array([a, b, r, 0.3, 0.3, k, 0.0, 0.0, 0.0, 0.0])
    ident = ZoneIdentifier(V, Prior(m0=m0, P0=np.full(10, 1e-6), R=1e-4), frozen=True)
    T = np.full(V, 22.0 + C) if T is None else np.asarray(T, float)
    f = make_features(np.zeros(V), np.zeros(V), np.full(V, 291.0), 330.0, 270.0, 0.0, 8.0)
    ident._hist.append((T.copy(), T.copy(), f))
    return ident


def test_cold_zone_gets_heat_warm_zone_idles():
    """Occupied, winter. One zone below band, one in band, one above: heat / idle / cool."""
    from aether.mpc import Planner
    from aether.params import Params
    P = _Parser(V=4, K=2)
    T = np.array([19.0, 22.5, 25.5, 22.5]) + C
    ident = _ident(4, a=0.02, T=T)  # a well-insulated building: an in-band zone stays in band for the horizon
    snap = _snap(V=4, hour=9.0, now=True, occ=8.0, T=22.0)
    snap.T = T.copy()
    snap.hsp[:], snap.csp[:] = 294.0, 297.0
    snap.tout = 265.0
    plan = Planner(P, Params(), META_SCHED)
    cmd, info = plan.step(snap, ident)
    assert cmd.reheat[0] >= 0.5 and cmd.ahu_run[0] and cmd.pump_run
    # the in-band zone gets at most the cold zone's heat (ventilation air at the damper floor is
    # 12 C here, so holding the band legitimately needs some reheat even when in band)
    assert cmd.reheat[1] <= cmd.reheat[0] and cmd.damper[1] <= Params().heat_damper
    assert cmd.damper[2] >= 0.5 and cmd.reheat[2] == 0.0 and cmd.ahu_run[1]  # cooling at half or full airflow
    assert info["n_plans"] >= 20


def test_unoccupied_night_mild_does_nothing():
    """Setback, nobody in, zones inside the eco band, next comfort start far away: plant off."""
    from aether.mpc import Planner
    from aether.params import Params
    P = _Parser(V=4, K=2)
    ident = _ident(4)
    snap = _snap(V=4, hour=1.0, dow=1, occ=0.0, now=False, soon=False, T=22.0)
    snap.hsp[:], snap.csp[:] = 289.0, 298.0
    snap.tout = 285.0
    cmd, _ = Planner(P, Params(), META_SCHED).step(snap, ident)
    assert not cmd.ahu_run.any() and not cmd.pump_run and cmd.tw_sp == 310.0


def test_prestart_before_occupancy_in_cold():
    """05:15 on a weekday at -20 C, zones at 17 C: the planner must already be heating."""
    from aether.mpc import Planner
    from aether.params import Params
    P = _Parser(V=4, K=2)
    T = np.full(4, 17.0 + C)
    ident = _ident(4, T=T)
    snap = _snap(V=4, hour=5.25, dow=1, occ=0.0, now=False, soon=True, T=17.0)
    snap.T = T.copy()
    snap.hsp[:], snap.csp[:] = 289.0, 298.0  # eco band still in force at 05:15
    snap.tout = 253.0
    cmd, _ = Planner(P, Params(), META_SCHED).step(snap, ident)
    assert cmd.pump_run and (cmd.reheat >= 0.5).all() and cmd.ahu_run.all()


def test_objective_matches_reward_shape():
    """Comfort cost is zero inside the band, linear to 5 K, quadratic beyond; only when occupied."""
    from aether.mpc import comfort_regret
    dev = np.array([0.0, 2.5, 5.0, 10.0])
    r = comfort_regret(dev)
    assert r[0] == 0 and abs(r[1] - 0.5) < 1e-9 and abs(r[2] - 1.0) < 1e-9 and abs(r[3] - 2.0) < 1e-9


def test_night_floor_and_optimal_start():
    """Unoccupied: a zone below Layer A's recoverable floor gets heat, one above it coasts; with the
    optimal-start lead covering the horizon the comfort band applies before opening."""
    from aether.mpc import Planner
    from aether.params import Params
    P = _Parser(V=4, K=2)
    T = np.array([12.0, 18.0, 18.0, 18.0]) + C
    ident = _ident(4, a=0.02, T=T)  # insulated: the 18 C zone stays above 15 C over the horizon
    snap = _snap(V=4, hour=1.0, dow=1, occ=0.0, now=False, soon=False, T=18.0)
    snap.T = T.copy()
    snap.hsp[:], snap.csp[:] = 285.0, 300.0
    snap.tout = 250.0
    floor = np.full(4, 15.0 + C)
    cmd, _ = Planner(P, Params(), META_SCHED).step(snap, ident, lead=np.zeros(4, int), floor=floor)
    assert cmd.reheat[0] > 0 and cmd.pump_run and cmd.ahu_run[0]
    assert cmd.reheat[1] == 0.0
    # same night, no floor, but every zone's lead reaches past the horizon: preheat toward the band
    cmd, _ = Planner(P, Params(), META_SCHED).step(snap, ident, lead=np.full(4, 10 ** 4), floor=None)
    assert (cmd.reheat > 0).all() and cmd.pump_run


def test_batched_rollout_equals_one_rollout_per_candidate():
    """Stacking candidates along the zone axis must give each candidate exactly its own rollout."""
    from aether.mpc import Planner
    from aether.params import Params
    from aether.zone_model import make_features, rollout
    P = _Parser(V=4, K=2)
    T = np.array([19.0, 22.5, 25.5, 21.0]) + C
    ident = _ident(4, a=0.05, T=T)
    ident.theta[:] = ident.theta * np.linspace(0.8, 1.2, 4)[:, None]  # zone-specific parameters
    snap = _snap(V=4, hour=7.5, now=False, occ=3.0, T=22.0)
    snap.T, snap.tout = T.copy(), 268.0
    plan = Planner(P, Params(), META_SCHED)
    rng = np.random.default_rng(0)
    items = [(g, rng.random(len(g)), rng.random(len(g)), 285.0 + 5 * rng.random(), tw)
             for g, tw in ((P.groups[0], 330.0), (P.groups[1], 340.0), (P.groups[0], 310.0), (np.arange(4), 335.0))]
    got = plan._rollout(ident, items, snap)
    T0, Tm0, fp = ident.state
    for (g, u, v, tsa, tw), traj in zip(items, got):
        steps = [make_features(u, v, np.full(len(g), tsa), tw, snap.tout, snap.occ / P.n_occ_ref, (snap.hour + j / 12) % 24) for j in range(plan.H)]
        prev = {k: (x[g] if isinstance(x, np.ndarray) and x.shape == (4,) else x) for k, x in fp.items()}
        np.testing.assert_array_equal(traj, rollout(ident.theta[g], T0[g], Tm0[g], steps, prev)[1:])


def test_module_cost_functions_match_planner():
    """The planner's zone inputs and costs are module functions (the value solver reuses them); the methods delegate."""
    from aether.mpc import Planner, mass_after, zone_inputs, zone_step_costs
    from aether.params import Params
    P = _Parser(V=4, K=2)
    T = np.array([19.0, 22.5, 25.5, 22.5]) + C
    ident = _ident(4, a=0.02, T=T)
    snap = _snap(V=4, hour=9.0, now=True, occ=8.0, T=22.0)
    snap.T, snap.tout = T.copy(), 265.0
    plan = Planner(P, Params(), META_SCHED)
    lo, hi, occ = plan._bands(snap)
    g = P.groups[0]
    for za in ("idle", "heat_half", "heat", "cool_half", "cool"):
        args = (snap, g, za, True, 290.0, True, 330.0, 925.0)
        d, u, v, tsa, mixed = plan._zone_inputs(*args)
        d2, u2, v2, tsa2, mixed2 = zone_inputs(plan.params, len(g), snap.T[g].mean(), snap.tout, za, True, 290.0, True, 330.0, 925.0)
        for a, b in ((d, d2), (u, u2), (v, v2), (tsa, tsa2), (mixed, mixed2)):
            np.testing.assert_array_equal(a, b)
        # array return-air temperature: elementwise equal to the scalar call
        _, _, _, tsa3, mixed3 = zone_inputs(plan.params, len(g), np.full(len(g), snap.T[g].mean()), snap.tout, za, True, 290.0, True, 330.0, 925.0)
        np.testing.assert_array_equal(tsa3, np.full(len(g), tsa2))
        np.testing.assert_array_equal(mixed3, np.full(len(g), mixed2))
        traj = plan._rollout(ident, [(g, u, v, tsa, 330.0)], snap)[0]
        np.testing.assert_allclose(plan._zone_costs(traj, lo[:, g], hi[:, g], occ[:, g], d, v, 925.0, tsa, 330.0, 2),
                                   zone_step_costs(traj, lo[:, g], hi[:, g], occ[:, g], d, v, 925.0, tsa, 330.0, 2,
                                                   plan.params.band_margin, P.n_occ_ref) / plan.H, rtol=1e-12, atol=0)
    T0, Tm0, _ = ident.state
    traj = np.array([[1.0, 2.0], [3.0, 4.0]]) + 290.0
    Tm = Tm0[:2].copy()
    for row in (T0[:2], traj[0]):
        Tm = Tm + (row - Tm) / 24
    np.testing.assert_array_equal(mass_after(Tm0[:2], T0[:2], traj), Tm)


def _prior_ident(V, T):
    """Frozen population prior, zones resting at T (the state the planner and the value both read)."""
    from aether.identifier import Prior, ZoneIdentifier
    from aether.zone_model import make_features
    ident = ZoneIdentifier(V, Prior.load("aether/prior.json"), frozen=True)
    f = make_features(np.zeros(V), np.zeros(V), np.full(V, 285.0), 310.0, 243.15, 0.0, 4.0)
    ident._hist.append((np.asarray(T, float).copy(), np.asarray(T, float).copy(), f))
    return ident


def _cold_dawn(V=4, T=289.0):
    """Monday 04:00, opening at 06:00 (2 h: beyond the 90-min horizon), -30 C outside, empty building."""
    snap = _snap(V=V, hour=4.0, now=False, occ=0.0, T=T)
    snap.tout = C - 30.0
    return snap


def test_value_makes_planner_preheat():
    """Without a terminal term the horizon cannot see an opening 2 h away and the planner idles; the learned value
    makes it heat every zone."""
    from aether.identifier import Prior
    from aether.mpc import Planner
    from aether.params import Params
    from aether.value import ZoneValue
    P = _Parser(V=4, K=2)
    # 10 C: at 16 C the prior expects a fast enough recovery after opening that the value too waits
    snap, ident = _cold_dawn(T=283.0), _prior_ident(4, np.full(4, 283.0))
    cmd0, _ = Planner(P, Params(), META_SCHED).step(snap, ident, lead=None, floor=None)
    assert (cmd0.reheat == 0).all()
    value = ZoneValue.solve(Prior.load("aether/prior.json").m0, Params(), META_SCHED, K=2, n_occ_ref=P.n_occ_ref)
    cmd1, _ = Planner(P, Params(), META_SCHED, value=value).step(snap, ident, lead=None, floor=None)
    assert (cmd1.reheat > 0).all() and cmd1.pump_run


def test_terminal_nonfinite_is_zero():
    """A zone whose temperature is not finite gets no terminal term, so it cannot hijack the argmin; the command stays
    finite and zones in other AHU groups act as without the fault."""
    from aether.identifier import Prior
    from aether.mpc import Planner
    from aether.params import Params
    from aether.value import ZoneValue
    P = _Parser(V=4, K=2)
    value = ZoneValue.solve(Prior.load("aether/prior.json").m0, Params(), META_SCHED, K=2, n_occ_ref=P.n_occ_ref)
    snap = _cold_dawn()
    T = np.full(4, 289.0)
    cmd_ok, _ = Planner(P, Params(), META_SCHED, value=value).step(snap, _prior_ident(4, T), lead=None, floor=None)
    T[0] = np.nan
    plan = Planner(P, Params(), META_SCHED, value=value)
    term = plan._terminal(snap, _prior_ident(4, T), np.arange(4), np.tile(T, (plan.H, 1)))
    assert term[0] == 0.0 and np.isfinite(term).all() and (term[1:] > 0).all()
    cmd, _ = plan.step(snap, _prior_ident(4, T), lead=None, floor=None)
    for x in (cmd.damper, cmd.reheat, cmd.sat, cmd.pressure):
        assert np.isfinite(x).all()
    g = P.groups[1]
    np.testing.assert_array_equal(cmd.damper[g], cmd_ok.damper[g])
    np.testing.assert_array_equal(cmd.reheat[g], cmd_ok.reheat[g])


def test_econ_recirc_same_supply_cheaper_plant():
    from aether.mpc import COP, RHO_CP, econ_recirc
    from aether.planner import RECIRC_MAX
    tr, sat, flow, p = 297.0, 285.15, 10.0, 200.0
    coil = lambda tout: flow * RHO_CP * max(RECIRC_MAX * tr + (1 - RECIRC_MAX) * tout - sat, 0) / COP
    for tout in (270.0, 280.0, 285.0, 290.0, 295.0):
        r, w = econ_recirc(tr, tout, sat, flow, p)
        mixed = r * tr + (1 - r) * tout
        mixed_max = RECIRC_MAX * tr + (1 - RECIRC_MAX) * tout
        assert np.isclose(min(mixed, sat), min(mixed_max, sat))  # zones see the same supply air
        assert w <= coil(tout) + 1e-9 and 0.0 <= r <= RECIRC_MAX
    assert econ_recirc(tr, 305.0, sat, flow, p) == (RECIRC_MAX, coil(305.0))  # hot outside: no economising
    assert econ_recirc(tr, 280.0, sat, flow, p)[1] < 0.2 * coil(280.0)  # cool outside: most of the coil work is free


def test_holiday_rule():
    from types import SimpleNamespace as S
    from aether.supervisor import holiday
    day = S(comfort_now=True, hour=9.5)
    assert holiday(day, seen_today=False, after_hour=9.0)          # comfort period, nobody by 9:30
    assert not holiday(day, seen_today=True, after_hour=9.0)       # someone came in today
    assert not holiday(S(comfort_now=True, hour=8.0), False, 9.0)  # too early to tell
    assert not holiday(S(comfort_now=False, hour=10.0), False, 9.0)  # night or weekend: nothing to rest
    assert not holiday(day, False, 0.0)                            # off by default


def test_discharge_matches_coil_physics():
    """The VAV coil (vav.py): tsa + e (tw - tsa), e = min(1, water/air capacity), capped at 308.15 K (verified
    against the live simulator to 1e-13 K on 928 box/valve/supply cases, scratchpad discheck)."""
    from aether.zone_model import KAPPA, T_DIS_MAX, discharge
    f = lambda u, v, tsa, tw=334.0, dp=10.0: float(discharge({"u": u, "v": v, "tsa": tsa, "tw": tw, "dp": dp}))
    assert f(0.5, 0.0, 290.0) == 290.0                                   # valve shut: supply air passes through
    assert f(0.05, 1.0, 290.0) == T_DIS_MAX                              # low airflow, full valve: saturates at 35 C
    e = KAPPA * np.sqrt(10.0) * 0.1 / 0.8
    assert abs(f(0.8, 0.1, 290.0) - (290.0 + e * 44.0)) < 1e-9 and e < 1  # water-limited: heat independent of airflow
    assert f(0.5, 1.0, 309.0) == 309.0                                   # already above the cap: the coil adds nothing
