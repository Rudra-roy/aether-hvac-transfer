"""Behaviour data for per-zone transitions on the value's 15-min clock grid, with the observed zone cost.

    CUDA_VISIBLE_DEVICES="" nice -n 19 uv run python -m harness.collect <arm> <scenario dir> [<scenario dir> ...]
        -> results/e2_data/<building>/<weather>/<arm>.npz

Row k is the state at the start of macro step k (steps 3k..3k+2) and the cost summed over those 3 steps. The cost is
the planner's own zone_step_costs on the OBSERVED temperature after each step and the applied actuation, plus the
zone's cooling-coil share: the same units as aether.value.ZoneValue, measured instead of predicted.
"""
import sys
from pathlib import Path

import numpy as np

from aether.mpc import RECIRC_MAX, zone_step_costs
from aether.params import Params
from aether.planner import DAMPER_MIN
from aether.policy import AetherPolicy
from aether.value import MACRO, c_day, clock_ahead, clock_node, zone_cool_cost
from harness.report import params_for
from harness.run_parallel import ROOT
from tests.conftest import make_env

OUT = ROOT / "results/e2_data"


def _step_cost(pol, snap, cmd, T_next, K):
    """(comfort, total) per zone for one 5-min step, from the applied command and the temperature it produced."""
    p, ahu = pol.parser, pol.parser.zone_ahu
    on = cmd.ahu_run[ahu]
    d = np.where(on, cmd.damper, DAMPER_MIN)
    pressure = np.where(on, cmd.pressure[ahu], 200.0)
    v = cmd.reheat * float(cmd.pump_run) * on
    lo, hi = pol.meta["comfort_temp_window"] if snap.comfort_now else (-np.inf, np.inf)
    occ = np.full((1, len(T_next)), snap.comfort_now)
    args = (lo, hi, occ, d, v, pressure, snap.tsa_eff, snap.tw, K, pol.params.band_margin, p.n_occ_ref)
    total = zone_step_costs(T_next[None], *args)
    comfort = zone_step_costs(T_next[None], lo, hi, occ, np.zeros_like(d), np.zeros_like(v), 0.0, snap.tsa_eff, snap.tw, K,
                              pol.params.band_margin, p.n_occ_ref)  # comfort part only (zero flow -> zero energy)
    ret = np.array([snap.T[ahu == k].mean() if (ahu == k).any() else snap.tout for k in range(len(p.ahus))])
    mixed = (RECIRC_MAX * ret + (1 - RECIRC_MAX) * snap.tout)[ahu]
    total = total + np.where(on, zone_cool_cost(d, pressure, mixed, cmd.sat[ahu], K), 0.0)
    return comfort, total


def collect(scenario: Path, arm: str, steps: int = 8640, out: Path | None = None) -> dict:
    env, tr, meta = make_env(scenario)
    pol = AetherPolicy(meta, env.action_spec(), params=Params.from_json(params_for(arm)), arm=arm)
    K, cd = len(pol.parser.ahus), c_day(meta)
    lo_band, hi_band = meta["comfort_temp_window"]
    tr.reset()
    obs = tr.last_obs
    rows = {k: [] for k in ("occ", "c", "tout", "T", "Tm", "bias", "cost", "comfort", "out_of_band")}
    for t in range(steps - steps % MACRO):
        snap = pol.parser.parse(obs)
        a = pol.get_action(obs)
        if t % MACRO == 0:
            T, Tm, _ = pol.ident.state
            occ, c = clock_node(*clock_ahead(snap, meta, 0), cd)
            for k, x in (("occ", occ), ("c", c), ("tout", snap.tout), ("T", T.copy()), ("Tm", Tm.copy()), ("bias", pol.ident.bias.copy())):
                rows[k].append(x)
            acc_c, acc, oob = 0.0, 0.0, np.zeros(len(T), bool)
        obs, *_ = tr.step(a)
        T_next = pol.parser.parse(obs).T
        comfort, total = _step_cost(pol, snap, pol.writer.decode(a), T_next, K)
        acc_c, acc = acc_c + comfort, acc + total
        m = pol.params.band_margin
        oob |= snap.comfort_now & ((T_next < lo_band + m) | (T_next > hi_band - m))
        if t % MACRO == MACRO - 1:
            rows["cost"].append(acc)
            rows["comfort"].append(acc_c)
            rows["out_of_band"].append(oob)
    d = {k: np.array(v) for k, v in rows.items()}
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **d)
    return d


if __name__ == "__main__":
    arm = sys.argv[1]
    for s in map(Path, sys.argv[2:]):
        d = collect(s, arm, out=OUT / s.parent.name / s.name / f"{arm}.npz")
        print(s.parent.name, s.name, arm, "rows", d["T"].shape, "mean zone cost/step", round(float(d["cost"].mean() / MACRO), 6), flush=True)
