import numpy as np

from tests.conftest import SHIPPED

C = 273.15


def test_collect_logs_consistent_zone_transitions(tmp_path):
    """Two days of behaviour data: one row per 15-min macro step per zone, costs finite and non-negative, comfort
    charged whenever an occupied zone leaves the band by more than the margin, clock nodes as the value uses them."""
    from aether.value import K_MAX
    from harness.collect import collect
    d = collect(SHIPPED / "building_train_1/fairbanks_1", "mpc_off", steps=576, out=tmp_path / "x.npz")
    n, V = d["T"].shape
    assert n == 576 // 3 and V > 0 and (tmp_path / "x.npz").exists()
    for k in ("Tm", "bias", "cost", "comfort"):
        assert d[k].shape == (n, V) and np.isfinite(d[k]).all()
    assert (d["cost"] >= 0).all() and (d["comfort"] >= 0).all() and (d["comfort"] <= d["cost"] + 1e-15).all()
    assert d["occ"].shape == d["c"].shape == d["tout"].shape == (n,)
    assert ((d["c"] % 3) == 0).all() and (d["c"] >= 3).all() and (d["c"][~d["occ"]] <= K_MAX).all()
    # comfort is charged exactly when an occupied zone ended a step out of the inset band
    occ_rows = d["occ"] & (d["out_of_band"].sum(1) > 0)
    assert occ_rows.any() and (d["comfort"][occ_rows].sum(1) > 0).all()
    assert (d["comfort"][~d["occ"]] == 0).all()
    assert d["T"].min() > C - 30 and d["T"].max() < C + 45
