"""Signed 1-h open-loop prediction error of the frozen prior, by room size range and zone mode (review R4).

Closed loop under the planner (econ configuration, 7 days), so the actions are the ones the model is used for.
From day 2 on, every 3rd step: predict 12 steps ahead with the logged actions, error = predicted - simulated (K).
Mode per zone at the start: heat (reheat valve > 0.05), cool (airflow > 0.05 and supply air 0.5 K below the zone),
else idle. Gains: the prior as fitted (scale 1) and with the size correction (airflow and reheat gains x 0.5).

    uv run python -m harness.signed_err <scenario_dir> [more ...]  -> results/signed_err/<building>_<station>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

from aether.params import Params
from aether.policy import AetherPolicy, load_prior
from aether.zone_model import features, rollout
from harness.fit_prior import _env, design
from smart_buildings import evaluate as ev

ROOT = Path(__file__).resolve().parents[1]
PARAMS = ROOT / "results/sweep/2016_stress/econ/params.json"
STEPS, H, START, STRIDE = 2016, 12, 288, 3


def run(scen: Path) -> dict:
    env, tracker, meta = _env(scen)
    p = json.loads(PARAMS.read_text())
    pol = AetherPolicy(meta, env.action_spec(), params=Params(**{k: v for k, v in p.items() if k in Params.__dataclass_fields__}), arm="mpc_off")
    tracker.reset()
    obs, log = tracker.last_obs, []
    for _ in range(STEPS):
        s = pol.parser.parse(obs)
        log.append((s.T, features(s, pol.parser), s))
        obs, *_ = tracker.step(ev.validate_action(pol.get_action(obs), env.action_spec()))
    _, _, Tm = design(log)
    m0 = load_prior().m0
    out = {}
    for scale in (1.0, 0.5):
        th = m0.copy()
        th[1:5] *= scale  # as AetherPolicy applies Params.act_scale
        th = np.repeat(th[None], len(log[0][0]), 0)
        err = {"heat": [], "cool": [], "idle": []}
        for t in range(START, len(log) - H, STRIDE):
            T, f, _ = log[t]
            e = rollout(th, T, Tm[t], [log[t + k][1] for k in range(H)], log[t - 1][1])[-1] - log[t + H][0]
            heat = f["v"] > 0.05
            cool = ~heat & (f["u"] > 0.05) & (f["tsa"] < T - 0.5)
            for m, mask in (("heat", heat), ("cool", cool), ("idle", ~heat & ~cool)):
                err[m].extend(e[mask].tolist())
        out[str(scale)] = {m: {"n": len(v), "mean": float(np.mean(v)) if v else None} for m, v in err.items()}
    return out


if __name__ == "__main__":
    for a in sys.argv[1:]:
        scen = Path(a).resolve()
        res = {"scenario": f"{scen.parent.name}/{scen.name}", "by_scale": run(scen)}
        dst = ROOT / "results/signed_err" / f"{scen.parent.name}_{scen.name}.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(res, indent=1))
        print(res)
