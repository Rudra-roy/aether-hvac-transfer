import numpy as np
import pytest

from tests.conftest import SHIPPED, make_env

META_SCHED = {"morning_start_hour": 6, "evening_start_hour": 19, "comfort_temp_window": (294.0, 297.0),
              "eco_temp_window": (289.0, 298.0)}


def _snap(V=3, hour=4.0, dow=1, occ=0.0, now=False, soon=False, T=292.0):
    from aether.obs import Snapshot
    return Snapshot(T=np.full(V, T), hsp=np.full(V, 289.0), csp=np.full(V, 298.0), damper=np.full(V, 0.1),
                    reheat=np.zeros(V), tsa_sp=np.full(1, 291.0), recirc=np.full(1, 0.7), flow=np.zeros(1), pressure=np.full(1, 2000.0),
                    ahu_on=np.zeros(1, bool), tsa_eff=np.full(V, 291.0), tout=270.0, tw=330.0, occ=occ,
                    comfort_now=now, comfort_soon=soon, hour=hour, dow=dow)


def _run(scenario, arm, steps, params=None):
    from aether.policy import AetherPolicy
    from smart_buildings.evaluate import validate_action
    env, tracker, meta = make_env(scenario)
    pol = AetherPolicy(meta, env.action_spec(), params=params, arm=arm)
    tracker.reset()
    obs = tracker.last_obs
    for _ in range(steps):
        a = validate_action(pol.get_action(obs), env.action_spec())
        assert np.all(np.isfinite(a["continuous_action"])) and np.all(np.abs(a["continuous_action"]) <= 1)
        obs, *_ = tracker.step(a)
    return pol, tracker


def _layout_scenarios():
    from harness.gen_pool import load_split
    return [sorted((SHIPPED / b).iterdir())[0] for b in sorted(p.name for p in SHIPPED.iterdir())] + load_split("pool_train")


@pytest.mark.parametrize("arm", ["rules"])
def test_layout_every_building(arm):
    for s in _layout_scenarios():
        _run(s, arm, 5)


def test_deterministic():
    s = sorted((SHIPPED / "building_train_8").glob("phoenix_*"))[0]
    r1 = list(_run(s, "rules", 288)[1].rewards)
    r2 = list(_run(s, "rules", 288)[1].rewards)
    assert r1 == r2


def test_weekend_no_precondition():
    from aether import supervisor
    from aether.params import Params
    p = Params()
    lead = np.full(3, 12)  # 1 h
    sat = supervisor.modes(_snap(hour=5.5, dow=5), lead, p, META_SCHED)
    assert not np.any(sat == supervisor.PRECONDITION)
    wk = supervisor.modes(_snap(hour=5.5, dow=1), lead, p, META_SCHED)  # Tuesday 05:30, comfort at 06:00
    assert np.all(wk == supervisor.PRECONDITION)
    assert supervisor.steps_to_comfort(_snap(hour=18.0, dow=4), META_SCHED) == 12 * (24 + 24 + 6 + 6)  # Fri 18:00 -> Mon 06:00
    assert np.all(supervisor.modes(_snap(hour=10, dow=1, now=True), lead, p, META_SCHED) == supervisor.OCCUPIED)


@pytest.mark.slow
def test_fairbanks_floor():
    """Design-spec criterion: no catastrophic scenario (score <= 2x the best baseline on that scenario)."""
    import json
    from tests.conftest import ROOT
    pol, tr = _run(SHIPPED / "building_train_1/fairbanks_1", "rules", 8640)
    base = min(abs(json.loads((ROOT / f"results/baselines/building_train_1/fairbanks_1/{p}/summary.json").read_text())["total_reward"])
               for p in ("bang_bang", "fixed_setpoint", "pid", "pid_water"))
    assert abs(sum(tr.rewards)) <= 2 * base, (abs(sum(tr.rewards)) / 8640, base / 8640)


def _ident_with(a_phys_per_h, V=2, b=2.0, r=4.0):
    """Identifier whose zones differ only in heat-loss rate a (1/h), past the trust gate."""
    from aether.identifier import Prior, ZoneIdentifier
    m0 = np.array([0.1, b, r, 0.5, 0.5, 2.0, 0.0, 0.0, 0.0, 0.0])
    ident = ZoneIdentifier(V, Prior(m0=m0, P0=np.full(10, 1e-6), R=1e-4), frozen=True)
    ident.prior = Prior(m0=m0, P0=np.full(10, 1e-6), R=1e-4)
    th = np.repeat(m0[None], V, 0)
    th[:, 0] = np.asarray(a_phys_per_h) / 12 * 10
    ident.frozen, ident.m = False, th
    return ident


def test_lead_monotone():
    from aether import supervisor
    from aether.obs import ObsParser
    from aether.params import Params
    from aether.zone_model import make_features
    ident = _ident_with([0.1, 0.3])
    snap = _snap(V=2, hour=3.0, dow=1, T=290.0)
    snap.tout = 255.0
    f = make_features(np.zeros(2), np.zeros(2), np.full(2, 280.0), 340.0, 255.0, 0.0, 3.0)
    ident._hist.append((snap.T.copy(), snap.T.copy(), f))
    parser = type("P", (), {"zone_ahu": np.zeros(2, int), "n_occ_ref": 4.0})()
    lead = supervisor.lead_steps(ident, snap, Params(), parser, META_SCHED, trusted=np.ones(2, bool))
    assert lead[1] >= lead[0] > 0


def test_prior_arm_always_trusted():
    pol, _ = _run(SHIPPED / "building_train_8" / sorted(p.name for p in (SHIPPED / "building_train_8").iterdir())[0], "prior", 60)
    assert pol.ident.trusted.all()
    np.testing.assert_allclose(pol.ident.theta, np.repeat(pol.ident.prior.m0[None], len(pol.parser.vavs), 0))


def test_passive_layout():
    for s in _layout_scenarios()[:8]:
        _run(s, "passive", 5)


@pytest.mark.parametrize("arm", ["prior", "passive", "active", "passive_guard", "full"])
def test_all_arms_layout(arm):
    from tests.test_actions import _largest_scenario
    for s in _layout_scenarios()[:3] + [_largest_scenario()[0]]:
        pol, _ = _run(s, arm, 30)
        st = pol.stats()
        assert {"guard_interventions", "probe_steps", "trusted_fraction_end"} <= set(st)


class _P2:
    vavs = ["VAV_1", "VAV_2", "VAV_3"]
    ahus = ["ahu_1"]
    zone_ahu = np.zeros(3, int)
    groups = [np.arange(3)]


def test_economiser_never_undercools():
    """Winter: one zone over band, two under. Mixed air must not fall below the cooling supply target."""
    from aether import planner
    from aether.params import Params
    from aether.supervisor import OCCUPIED
    p = Params()
    snap = _snap(V=3, hour=10, now=True)
    snap.hsp[:], snap.csp[:] = 294.0, 297.0
    snap.T[:] = [298.5, 292.0, 292.0]
    snap.tout = 251.0  # -22 C
    cmd = planner.plan(snap, np.full(3, OCCUPIED, np.int8), np.ones((3, 2)), p, _P2(), META_SCHED)
    mixed = cmd.recirc[0] * snap.T.mean() + (1 - cmd.recirc[0]) * snap.tout
    assert mixed >= min(p.sat_cool, 0.7 * snap.T.mean() + 0.3 * snap.tout) - 0.5, (cmd.recirc, mixed - 273.15)


def test_heating_damper_tracks_reheat():
    from aether import planner
    from aether.params import Params
    from aether.supervisor import OCCUPIED
    p = Params()
    snap = _snap(V=3, hour=10, now=True, T=290.0)
    snap.hsp[:], snap.csp[:] = 294.0, 297.0
    snap.tout = 275.0  # mild: mixed air only ~4.5 K below the zone -> full airflow through the coil
    cmd = planner.plan(snap, np.full(3, OCCUPIED, np.int8), np.ones((3, 2)), p, _P2(), META_SCHED)
    assert np.all(cmd.reheat == 1.0) and np.allclose(cmd.damper, 1.0)
    snap.tout = 235.0  # deep cold: mixed air ~16.5 K below the zone -> airflow cut to the cold-end gain
    cmd = planner.plan(snap, np.full(3, OCCUPIED, np.int8), np.ones((3, 2)), p, _P2(), META_SCHED)
    assert np.allclose(cmd.damper, 0.1 + p.heat_damper)


def test_hot_water_idles_without_reheat_demand():
    """No zone below its target => pump off and the loop at its lowest setpoint (tank sits outdoors)."""
    from aether import planner
    from aether.params import Params
    from aether.supervisor import OCCUPIED
    snap = _snap(V=3, hour=10, now=True, T=295.5)
    snap.hsp[:], snap.csp[:] = 294.0, 297.0
    snap.tout = 250.0
    cmd = planner.plan(snap, np.full(3, OCCUPIED, np.int8), np.ones((3, 2)), Params(), _P2(), META_SCHED)
    assert not cmd.pump_run and cmd.tw_sp == 310.0
    snap.T[1] = 292.0  # one zone needs heat: loop comes back at the outdoor-reset temperature
    cmd = planner.plan(snap, np.full(3, OCCUPIED, np.int8), np.ones((3, 2)), Params(), _P2(), META_SCHED)
    assert cmd.pump_run and cmd.tw_sp == Params().tw_max


def test_mpc_arm_layout_and_determinism():
    """Arm 'mpc' (planner over the frozen prior, rule fallback) runs on three buildings + the largest, deterministically."""
    from tests.test_actions import _largest_scenario
    for s in _layout_scenarios()[:2] + [_largest_scenario()[0]]:
        pol, _ = _run(s, "mpc", 24)
        assert pol.stats()["planner_calls"] >= 8
    s = _layout_scenarios()[0]
    r1 = list(_run(s, "mpc", 60)[1].rewards)
    r2 = list(_run(s, "mpc", 60)[1].rewards)
    assert r1 == r2
