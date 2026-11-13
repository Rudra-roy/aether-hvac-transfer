import functools

import numpy as np
import pytest

from tests.conftest import SHIPPED, make_env


def _metas():
    from harness.gen_pool import load_split
    scen = [sorted((SHIPPED / b).iterdir())[0] for b in ("building_train_1", "building_train_2", "building_train_4", "building_train_5", "building_train_8")]
    scen += load_split("pool_train")
    return [make_env(s)[2] for s in scen]


@functools.lru_cache(maxsize=1)
def _largest_scenario():
    """Pool scenario with the most AHUs, from the generated configs (no env construction)."""
    import re
    from harness.gen_pool import load_split
    pools = load_split("pool_train") + load_split("pool_test") + load_split("large") + load_split("minneapolis")
    n_ahu = lambda s: len(set(re.findall(r"ahu_(\d+)", (s / "config.gin").read_text())))
    best = max(pools, key=n_ahu)
    return best, n_ahu(best)


def _largest():
    from aether.obs import ObsParser
    s, _ = _largest_scenario()
    env, tr, meta = make_env(s)
    return len(ObsParser(meta).ahus), s, env, tr, meta


def _random_command(writer, rng):
    from aether.actions import Command
    V, K = len(writer.parser.vavs), len(writer.parser.ahus)
    return Command(damper=rng.uniform(0, 1, V), reheat=rng.uniform(0, 1, V), ahu_run=rng.random(K) > 0.5,
                   sat=rng.uniform(285, 305, K), pressure=rng.uniform(0, 2000, K), recirc=rng.uniform(0, 1, K),
                   pump_run=bool(rng.random() > 0.5), tw_sp=float(rng.uniform(310, 350)), dp_sp=float(rng.uniform(0, 20)))


def _check_roundtrip(meta, rng):
    from aether.actions import ActionWriter
    w = ActionWriter(meta)
    cmd = _random_command(w, rng)
    act = w.write(cmd)
    K, V = len(w.parser.ahus), len(meta["zone_action_indices"])
    assert act["continuous_action"].dtype == np.float32 and act["discrete_action"].dtype == np.int32
    assert act["continuous_action"].shape == (3 * K + 2 + 2 * V,) and act["discrete_action"].shape == (K + 1,)
    assert w.last_written == set(range(meta["n_continuous_action"]))
    back = w.decode(act)
    for name, span in (("damper", 1), ("reheat", 1), ("sat", 20), ("pressure", 2000), ("recirc", 1)):
        np.testing.assert_allclose(getattr(back, name), getattr(cmd, name), atol=1e-4 * span)
    np.testing.assert_array_equal(back.ahu_run, cmd.ahu_run)
    assert back.pump_run == cmd.pump_run and abs(back.tw_sp - cmd.tw_sp) < 4e-3 and abs(back.dp_sp - cmd.dp_sp) < 2e-3


def test_roundtrip_all_slots():
    rng = np.random.default_rng(0)
    for meta in _metas():
        _check_roundtrip(meta, rng)


def test_bounds_and_dtypes():
    from aether.actions import ActionWriter, Command
    meta = make_env(SHIPPED / "building_train_1/fairbanks_1")[2]
    w = ActionWriter(meta)
    V, K = len(w.parser.vavs), len(w.parser.ahus)
    cmd = Command.default(V, K)
    cmd.damper[:] = 5.0
    cmd.sat[:] = 100.0
    cmd.tw_sp = 1e6
    act = w.write(cmd)
    c = act["continuous_action"]
    assert np.all(c >= -1) and np.all(c <= 1) and np.all(np.isfinite(c))
    assert set(np.unique(act["discrete_action"])) <= {0, 1}


def test_largest_building():
    k, s, env, tracker, meta = _largest()
    assert k >= 4
    _check_roundtrip(meta, np.random.default_rng(1))


def test_simulator_honours_mapping():
    """Zone slots map 0-1 linearly; AHU k's run command drives AHU k (checked on the building with most AHUs)."""
    from aether.actions import ActionWriter, Command
    k, s, env, tracker, meta = _largest()
    w = ActionWriter(meta)
    V, K = len(w.parser.vavs), len(w.parser.ahus)
    tracker.reset()
    for target in (0, K - 1):
        cmd = Command.default(V, K)
        cmd.ahu_run[target] = True
        cmd.pressure[:] = 1000.0
        cmd.damper[:] = 0.55
        cmd.reheat[:] = 0.3
        for _ in range(3):
            obs, *_ = tracker.step(w.write(cmd))
        snap = w.parser.parse(obs)
        assert snap.flow[target] > 0 and np.all(np.delete(snap.flow, target) == 0)
        np.testing.assert_allclose(snap.damper, 0.55, atol=1e-3)
        np.testing.assert_allclose(snap.reheat, 0.3, atol=1e-3)
