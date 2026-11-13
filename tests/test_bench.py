def test_runtime_budget():
    """Full-arm rollout time per step <= 3x bang-bang on the largest pool building (rule allows 10x)."""
    from harness.bench import bench
    from tests.test_actions import _largest_scenario
    s, _ = _largest_scenario()
    base = bench(s, "bang_bang", steps=288)
    ours = bench(s, "aether:full", steps=288)
    ratio = ours["total_seconds"] / base["total_seconds"]
    assert ratio <= 3.0, (ratio, base, ours)
    assert ours["get_action_ms_p99"] < 1000
