"""Equal-budget CMA-ES tuning per arm (A fair comparison needs identical budgets for `prior` and `full`).

    uv run python -m harness.tune --arm full --split pool_train --evals 64 --out results/tune/full
    uv run python -m harness.tune --arm full --smoke --out results/tune/smoke   # plumbing check

Never sees pool_test, shipped LOSO test folds or the leaderboard.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cma
import numpy as np

from aether.params import BOUNDS, Params
from harness.gen_pool import load_split
from harness.run_parallel import run_many
from harness.score_table import score

# what each arm can actually use (others would be dead dimensions)
NAMES = {
    "rules": ["band_margin", "kp_heat", "kp_cool", "heat_damper", "sat_cool", "sat_idle", "p_max", "tw_max", "tw_min",
              "default_lead_min", "setback_margin", "econ_margin"],
}
NAMES["prior"] = NAMES["rules"] + ["max_lead_min"]
NAMES["passive"] = NAMES["prior"] + ["q_rw"]
NAMES["full"] = NAMES["passive"] + ["guard_h", "guard_margin", "probe_budget_ahu_h"]
# the planner: its own comfort margin and weight, plus the Layer A scalars it consumes (lead, floor, plant settings)
NAMES["mpc_off"] = ["band_margin", "plan_margin", "comfort_w", "heat_damper", "sat_cool", "sat_idle", "p_max", "tw_max", "tw_min",
                    "default_lead_min", "max_lead_min", "setback_margin", "econ_margin", "floor_shift", "lead_scale"]
PID_BOUNDS = {"kp": (0.05, 3.0), "ki": (0.0, 0.2), "kd": (0.0, 1.0), "zone_damper_full_k": (0.5, 4.0)}
PID_X0 = {"kp": 0.5, "ki": 0.02, "kd": 0.1, "zone_damper_full_k": 2.0}


def candidate_file(arm: str, x: np.ndarray, path: Path, base: Params | None = None) -> dict:
    if arm == "pid_tuned":
        d = {n: lo + float(xi) * (hi - lo) for xi, (n, (lo, hi)) in zip(np.clip(x, 0, 1), PID_BOUNDS.items())}
        path.write_text(json.dumps(d))
        return d
    p = Params.from_unit(x, NAMES[arm], base)
    p.to_json(path)
    return json.loads(path.read_text())


def x0_for(arm: str, base: Params | None = None) -> np.ndarray:
    if arm == "pid_tuned":
        return np.array([(PID_X0[n] - lo) / (hi - lo) for n, (lo, hi) in PID_BOUNDS.items()])
    p = base or Params()
    if p.plan_margin < 0:  # start where the shipped planner is: its margin is Layer A's
        p = Params(**{**p.__dict__, "plan_margin": p.band_margin})
    return p.to_unit(NAMES[arm])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(NAMES) + ["pid_tuned"])
    ap.add_argument("--split", default="pool_train")
    ap.add_argument("--evals", type=int, default=64)
    ap.add_argument("--popsize", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--steps", type=int, default=8640)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--x0", type=Path, default=None, help="warm start (and values of untuned fields) from a params json")
    ap.add_argument("--sigma", type=float, default=0.25)
    ap.add_argument("--smoke", action="store_true", help="plumbing check: 2 scenarios x 288 steps, popsize 2, 2 evals")
    a = ap.parse_args(argv)
    if a.smoke:
        a.split, a.steps, a.evals, a.popsize = "pool_train10", 288, 2, 2
    assert a.split not in ("pool_test",) and not a.split.endswith(":test"), "tuning must not see test splits"
    scen = load_split(a.split)[:2] if a.smoke else load_split(a.split)
    policy = "pid_tuned" if a.arm == "pid_tuned" else f"aether:{a.arm}"
    out = a.out
    (out / "cands").mkdir(parents=True, exist_ok=True)
    base = Params.from_json(a.x0) if a.x0 else None
    es = cma.CMAEvolutionStrategy(x0_for(a.arm, base), a.sigma, {"bounds": [0, 1], "popsize": a.popsize, "seed": a.seed + 1, "verbose": -9})
    hist, best = [], (np.inf, None)
    n_eval, gen = 0, 0
    t0 = time.time()
    while n_eval < a.evals:
        xs = es.ask()[: a.evals - n_eval]
        files = []
        for i, x in enumerate(xs):
            f = out / "cands" / f"e{n_eval + i:04d}.json"
            candidate_file(a.arm, x, f, base)
            files.append(f)
        jobs = [(s, policy, f, out / "runs" / f.stem) for f in files for s in scen]
        res = list(run_many(jobs, a.steps, out / "runs", a.workers))
        fits = []
        for f in files:
            tot = {r["scenario"]: r["total_reward"] for r in res if r["params"] and Path(r["params"]).name == f.name}
            fits.append(score(tot, a.steps))
        for i, (x, f, s) in enumerate(zip(xs, files, fits)):
            hist.append({"eval": n_eval + i, "gen": gen, "score": s, "params": f.read_text()})
            if s < best[0]:
                best = (s, json.loads(f.read_text()))
        if len(xs) == a.popsize:
            es.tell(xs, fits)
        n_eval += len(xs)
        gen += 1
        print(f"gen {gen}: evals {n_eval}/{a.evals} best {best[0]:.5f} gen-min {min(fits):.5f} ({time.time() - t0:.0f}s)", flush=True)
    with open(out / "history.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["eval", "gen", "score", "params"])
        w.writeheader()
        w.writerows(hist)
    (out / "best.json").write_text(json.dumps({**best[1], "_score": best[0], "_split": a.split, "_steps": a.steps,
                                               "_evals": a.evals, "_arm": a.arm}, indent=1))
    print(f"best {best[0]:.5f} -> {out / 'best.json'}")


if __name__ == "__main__":
    main()
