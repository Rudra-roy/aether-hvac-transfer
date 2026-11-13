"""Per-step zone-temperature trajectories for the paper's behaviour figure (CPU only).

    uv run python -m harness.paper_traj <policy_key> [steps] [scenario]
writes results/paper_traj/<scenario>/<policy>.npz (hour, dow, tout, occupied, T percentiles, band medians).
"""
import sys

import numpy as np
from smart_buildings import evaluate as ev

from aether.obs import ObsParser
from harness.report import params_for
from harness.run_parallel import ROOT, run_one

key = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 1152
scen = ROOT / "pool/scenarios" / (sys.argv[3] if len(sys.argv) > 3 else "building_1401/fairbanks_3")
out = ROOT / "results/paper_traj" / scen.parent.name / scen.name
log = []
_get = ev.get_policy_action
parser = None


def get_policy_action(policy, obs):
    global parser
    if parser is None:
        parser = ObsParser(policy.meta if hasattr(policy, "meta") else ev._last_meta)
    s = parser.parse(obs)
    log.append((s.hour, s.dow, s.tout, s.comfort_now, *np.percentile(s.T, [10, 50, 90]), np.median(s.hsp), np.median(s.csp)))
    return _get(policy, obs)


_meta = ev.create_building_meta_data


def create_building_meta_data(*a, **k):
    ev._last_meta = _meta(*a, **k)
    return ev._last_meta


ev.get_policy_action = get_policy_action
ev.create_building_meta_data = create_building_meta_data
arm = key.split(":")[1] if key.startswith("aether:") else key
params = params_for(arm) if key.startswith("aether:") else None
r = run_one(scen, key, steps, ROOT / "results/paper_traj/runs", params)
a = np.array(log, float)
out.mkdir(parents=True, exist_ok=True)
np.savez(out / f"{key.replace(':', '-')}.npz", hour=a[:, 0], dow=a[:, 1], tout=a[:, 2], occ=a[:, 3],
         t10=a[:, 4], t50=a[:, 5], t90=a[:, 6], hsp=a[:, 7], csp=a[:, 8], score=abs(r["total_reward"]) / len(a))
print(key, f"{abs(r['total_reward']) / len(a):.4f}", len(a))
