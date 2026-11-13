"""Lowest zone temperature while the building is empty (snap.occ == 0), any policy; same statistic as
AetherPolicy.stats()['min_T_unocc']. Appends one JSON line to results/min_unocc/<policy>.jsonl.

    uv run python -m harness.min_unocc <scenario_dir> <steps> <policy key> [params.json] [label]
"""
import json
import sys
from pathlib import Path

import numpy as np

from aether.policy import AetherPolicy
from harness.fit_prior import _env
from harness.run_parallel import resolve_policy
from smart_buildings import evaluate as ev

ROOT = Path(__file__).resolve().parents[1]


def main(argv):
    scen, steps, key = Path(argv[0]).resolve(), int(argv[1]), argv[2]
    params = Path(argv[3]).resolve() if len(argv) > 3 else None
    label = argv[4] if len(argv) > 4 else key
    env, tracker, meta = _env(scen)
    pol = resolve_policy(key, params)(meta, env.action_spec())
    parser = AetherPolicy(meta, env.action_spec(), arm="mpc_off").parser
    tracker.reset()
    obs, lo = tracker.last_obs, np.inf
    for _ in range(steps):
        s = parser.parse(obs)
        if s.occ == 0:
            lo = min(lo, float(np.nanmin(s.T)))
        obs, *_ = tracker.step(ev.validate_action(pol.get_action(obs), env.action_spec()))
    rec = {"scenario": f"{scen.parent.name}/{scen.name}", "policy": label, "steps": steps, "min_T_unocc_C": round(lo - 273.15, 2)}
    out = ROOT / "results/min_unocc" / f"{label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec)


if __name__ == "__main__":
    main(sys.argv[1:])
