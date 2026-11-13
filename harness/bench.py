"""Runtime benchmark: our policy vs bang-bang (competition limit: <= 10x bang-bang per rollout).

    uv run python -m harness.bench [--steps 2016]   -> results/bench/BENCH.md
"""

import argparse
import os
import resource
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from harness.run_parallel import KIT, ROOT, resolve_policy

POLICIES = ["bang_bang", "pid_water", "aether:rules", "aether:prior", "aether:passive", "aether:active", "aether:passive_guard", "aether:full"]


def bench(scenario_dir, policy_key: str, steps: int = 2016) -> dict:
    from smart_buildings import evaluate as ev
    from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker
    scenario_dir = Path(scenario_dir).resolve()
    cwd = os.getcwd()
    os.chdir(KIT)
    try:
        env = ev.load_hybrid_environment(scenario_dir / "config.gin")
    finally:
        os.chdir(cwd)
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    tracker = SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False)
    meta = ev.create_building_meta_data(env, scenario_dir)
    pol = resolve_policy(policy_key)(meta, env.action_spec())
    tracker.reset()
    obs = tracker.last_obs
    ga, st = [], []
    for _ in range(steps):
        t0 = time.perf_counter()
        a = pol.get_action(obs)
        t1 = time.perf_counter()
        obs, *_ = tracker.step(a)
        ga.append(t1 - t0)
        st.append(time.perf_counter() - t1)
    ga, st = np.array(ga) * 1e3, np.array(st)
    return {"policy": policy_key, "zones": len(meta["zone_action_indices"]), "ahus": len(meta["temp_indices"]),
            "seconds_per_step": float(st.mean()), "get_action_ms_p50": float(np.median(ga)),
            "get_action_ms_p99": float(np.percentile(ga, 99)), "total_seconds": float(st.sum() + ga.sum() / 1e3),
            "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def _job(args):
    return bench(*args)


def main(argv=None):
    from harness.gen_pool import load_split
    from tests.test_actions import _largest_scenario
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2016)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    scen = [sorted((KIT / "smart_buildings/training_scenarios/building_train_2").iterdir())[0], _largest_scenario()[0]]
    jobs = [(s, p, a.steps) for s in scen for p in POLICIES]
    with Pool(a.workers, maxtasksperchild=1) as pool:
        rows = pool.map(_job, jobs)
    lines = [f"# Runtime benchmark ({a.steps} steps, {a.workers} concurrent jobs)", "",
             "| scenario | zones | AHUs | policy | sim s/step | get_action ms p50 | p99 | total s | x bang-bang | peak RSS MB |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for s in scen:
        sub = [r for (sc, _, _), r in zip(jobs, rows) if sc == s]
        bb = next(r for r in sub if r["policy"] == "bang_bang")["total_seconds"]
        for r in sub:
            lines.append(f"| {s.parent.name}/{s.name} | {r['zones']} | {r['ahus']} | {r['policy']} | {r['seconds_per_step']:.4f} | "
                         f"{r['get_action_ms_p50']:.2f} | {r['get_action_ms_p99']:.2f} | {r['total_seconds']:.0f} | "
                         f"{r['total_seconds'] / bb:.2f} | {r['peak_rss_mb']:.0f} |")
    out = ROOT / "results/bench"
    out.mkdir(parents=True, exist_ok=True)
    (out / "BENCH.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
