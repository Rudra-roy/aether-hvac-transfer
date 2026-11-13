import copy

import numpy as np
import pytest

from tests.conftest import SHIPPED, make_env


def _scenarios():
    from harness.gen_pool import load_split
    return [SHIPPED / "building_train_1/fairbanks_1", load_split("pool_test")[0]]


def _rollout(scenario, steps):
    """Yield (parser, snapshot, tracker) for the pre-step obs of every step under pid_water."""
    from aether.obs import ObsParser
    from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy

    env, tracker, meta = make_env(scenario)
    parser = ObsParser(meta)
    pol = PIDWaterResetPolicy(meta, env.action_spec())
    tracker.reset()
    obs = tracker.last_obs
    for _ in range(steps):
        yield parser, parser.parse(obs), tracker
        obs, *_ = tracker.step(pol.get_action(obs))


@pytest.mark.parametrize("scenario", _scenarios(), ids=lambda p: f"{p.parent.name}/{p.name}")
def test_matches_simulator(scenario):
    prev = None
    n = 0
    for parser, snap, tracker in _rollout(scenario, 300):
        if prev is not None:
            truth = dict(zip(tracker.vavs.keys(), tracker.zone_temps[-1]))
            np.testing.assert_allclose(snap.T, [truth[v] for v in parser.vavs], atol=1e-4)
            assert abs(snap.tout - tracker.ahu_log[-1]["outside_air"]) < 1e-3
            n += 1
        prev = snap
        assert np.all(np.isfinite(snap.tsa_eff)) and np.all(snap.tsa_eff <= snap.tsa_sp[parser.zone_ahu] + 1e-9)
    assert n == 299


def test_time_decoding():
    hours, dows, stamps = [], [], []
    for _, snap, tracker in _rollout(SHIPPED / "building_train_1/fairbanks_1", 600):
        if tracker.timestamps:
            stamps.append(tracker.timestamps[-1])
            hours.append(snap.hour)
            dows.append(snap.dow)
    for h, d, ts in zip(hours, dows, stamps):
        assert abs(h - (ts.hour + ts.minute / 60)) < 1e-6
        assert d == ts.weekday()


def test_missing_zone_sensor():
    from aether.obs import ObsParser

    env, tracker, meta = make_env(SHIPPED / "building_train_1/fairbanks_1")
    m = copy.copy(meta)
    m["observation_field_names"] = [f.replace("VAV_3_", "XXX_3_") if f.startswith("VAV_3_") else f
                                    for f in meta["observation_field_names"]]
    parser = ObsParser(m)
    assert "VAV_3" not in parser.vavs and len(parser.vavs) == len(ObsParser(meta).vavs) - 1
    tracker.reset()
    snap = parser.parse(tracker.last_obs)
    assert snap.T.shape == (len(parser.vavs),) and np.all(np.isfinite(snap.tsa_eff))
