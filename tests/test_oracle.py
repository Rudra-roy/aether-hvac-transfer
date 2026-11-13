import numpy as np

from tests.conftest import SHIPPED, make_env


def test_weather_oracle_matches_simulator():
    """Oracle outside temperature at step t equals what the simulator applied at step t (weather ablation only)."""
    from aether.oracle import WeatherOracle
    from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy
    scen = SHIPPED / "building_train_1/fairbanks_1"
    env, tracker, meta = make_env(scen)
    oracle = WeatherOracle.from_scenario(scen)
    pol = PIDWaterResetPolicy(meta, env.action_spec())
    tracker.reset()
    obs = tracker.last_obs
    for t in range(60):
        obs, *_ = tracker.step(pol.get_action(obs))
        assert abs(oracle.tout(t + 1) - tracker.ahu_log[-1]["outside_air"]) < 0.05, t
    # a 1-h-ahead lookup returns a different value when the weather is moving
    assert any(abs(oracle.tout(t + 12) - oracle.tout(t)) > 0.1 for t in range(0, 60, 6))


def test_mpc_arm_with_oracle_weather_runs_and_differs():
    """aether:mpc+oracle plans on the true future temperature; it runs on a real scenario and does not
    produce the identical action stream as persistence weather (so the ablation measures something)."""
    from harness.run_parallel import resolve_policy
    from smart_buildings.evaluate import validate_action
    scen = SHIPPED / "building_train_1/fairbanks_1"
    streams = []
    for key in ("aether:mpc", "aether:mpc+oracle"):
        env, tracker, meta = make_env(scen)
        pol = resolve_policy(key, scenario_dir=scen)(meta, env.action_spec())
        tracker.reset()
        obs, out = tracker.last_obs, []
        for _ in range(2 * 288):
            a = validate_action(pol.get_action(obs), env.action_spec())
            out.append(a["continuous_action"].copy())
            obs, *_ = tracker.step(a)
        streams.append(np.array(out))
        assert pol.stats()["planner_calls"] > 0
    assert not np.array_equal(streams[0], streams[1])


def test_weather_oracle_on_generated_pool_scenario():
    """Generated pool scenarios live outside the kit; the oracle must still find and align their weather."""
    from pathlib import Path
    from aether.oracle import WeatherOracle
    from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy
    scen = Path(__file__).resolve().parents[1] / "pool/scenarios/building_1501/fairbanks_6"
    env, tracker, meta = make_env(scen)
    oracle = WeatherOracle.from_scenario(scen)
    pol = PIDWaterResetPolicy(meta, env.action_spec())
    tracker.reset()
    obs = tracker.last_obs
    for t in range(24):
        obs, *_ = tracker.step(pol.get_action(obs))
        assert abs(oracle.tout(t + 1) - tracker.ahu_log[-1]["outside_air"]) < 0.05, t
