import functools
import time

import numpy as np

from tests.test_policy import META_SCHED as META, _snap

C = 273.15


@functools.lru_cache(maxsize=None)
def _solved(morning=6):
    from aether.identifier import Prior
    from aether.params import Params
    from aether.value import ZoneValue
    meta = {**META, "morning_start_hour": morning}
    return ZoneValue.solve(Prior.load("aether/prior.json").m0, Params(), meta, K=2, n_occ_ref=8.0)


def test_bellman_holds_on_grid():
    """On grid nodes the stored value is the minimum over actions of cost + value of the successor (both phases)."""
    from aether.value import BIAS_GRID, D_GRID, K_MAX, T_GRID, TOUT_GRID
    v = _solved()
    rng = np.random.default_rng(0)
    for occupied, cmax in ((False, K_MAX), (True, v.c_day)):
        for c in rng.choice(np.arange(0, cmax + 1, 3), 6):
            n = 50
            tout, T = rng.choice(TOUT_GRID, n), rng.choice(T_GRID, n)
            d, b = rng.choice(D_GRID, n), rng.choice(BIAS_GRID, n)
            np.testing.assert_allclose(v.lookup(occupied, int(c), tout, T, d, b), v.q(occupied, int(c), tout, T, d, b).min(0), rtol=0, atol=1e-9)


def test_cold_before_opening_costs_more():
    """One hour before opening in deep cold, a colder zone has a strictly larger cost-to-go."""
    v = _solved()
    T = np.array([283.0, 287.0, 291.0])
    val = v.lookup(False, 12, np.full(3, C - 20), T, np.zeros(3), np.zeros(3))
    assert np.isfinite(val).all() and val[0] > val[1] > val[2]


def test_solve_fast_and_finite():
    from aether.identifier import Prior
    from aether.params import Params
    from aether.value import ZoneValue
    t = time.perf_counter()
    v = ZoneValue.solve(Prior.load("aether/prior.json").m0, Params(), META, K=2, n_occ_ref=8.0)
    dt = time.perf_counter() - t
    print(f"solve {dt:.1f} s")
    assert dt < 30.0
    assert v.unocc.shape == (33, 9, 71, 5, 5) and v.occ.shape == (156 // 3 + 1, 9, 71, 5, 5)
    assert np.isfinite(v.unocc).all() and np.isfinite(v.occ).all()


def test_clock_ahead_cases():
    from aether.value import K_MAX, clock_ahead
    c_day = 12 * (19 - 6)
    assert clock_ahead(_snap(hour=10.0, now=True), META, 18) == (True, 12 * 9 - 18)       # comfort on
    assert clock_ahead(_snap(hour=4.0), META, 18) == (False, 24 - 18)                    # 2 h before opening
    assert clock_ahead(_snap(hour=5.5), META, 18) == (True, c_day - 12)                  # opens inside the horizon
    assert clock_ahead(_snap(hour=18.5, now=True), META, 18) == (False, K_MAX)           # closes inside the horizon
    assert clock_ahead(_snap(hour=12.0, dow=5), META, 18) == (False, K_MAX)              # Saturday noon


def test_lookup_clamps_offgrid():
    from aether.value import BIAS_MAX
    v = _solved()
    off = v.lookup(False, 30, np.array([C - 50]), np.array([260.0]), np.zeros(1), np.array([0.2]))
    edge = v.lookup(False, 30, np.array([C - 35]), np.array([275.0]), np.zeros(1), np.array([BIAS_MAX]))
    assert np.isfinite(off).all() and np.array_equal(off, edge)


def test_solve_depends_on_schedule():
    a, b = _solved(6), _solved(8)
    assert a.occ.shape != b.occ.shape or not np.array_equal(a.occ, b.occ)
