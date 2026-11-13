"""Run (policy, scenario) jobs in parallel on the unmodified kit simulator.

    uv run python -m harness.run_parallel --policies pid_water aether:full \
        --scenarios 'external/starter-kit/smart_buildings/training_scenarios/*/*' --out results/x --workers 28

--scenarios takes globs of scenario dirs (each holding config.gin) or a split name from
pool/pool.yaml / harness/folds.yaml (see harness.gen_pool.load_split).
"""

import argparse
import glob
import json
import sys
import os
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "external/starter-kit"  # config.gin paths are relative to the kit root


def resolve_policy(key: str, params_path: Path | None = None, scenario_dir: Path | None = None):
    """Factory (meta, action_spec) -> policy for a policy key. 'aether:<arm>+oracle' adds true weather (ablation only)."""
    if key.startswith("aether:"):
        from aether.params import Params
        from aether.policy import AetherPolicy
        arm = key.split(":", 1)[1]
        oracle = None
        if arm.endswith("+oracle"):
            from aether.oracle import WeatherOracle
            arm = arm[: -len("+oracle")]
            oracle = WeatherOracle.from_scenario(scenario_dir)
        params = Params.from_json(params_path) if params_path else None
        residual = None
        if arm == "mpc_r":  # learned residual weights live next to the params file
            import numpy as np
            residual = dict(np.load(Path(params_path).with_name("residual.npz")))
        return lambda meta, spec: AetherPolicy(meta, spec, params=params, arm=arm, oracle=oracle, residual=residual)
    if key.startswith("ext:"):  # third-party controller for analysis: ext:<module>:<Class>, module on PYTHONPATH
        import importlib
        mod, cls_name = key[4:].rsplit(":", 1)
        cls = getattr(importlib.import_module(mod), cls_name)
        return lambda meta, spec: cls(meta, spec)
    if key == "pid_tuned":
        from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy
        kw = {k: v for k, v in json.loads(Path(params_path).read_text()).items() if not k.startswith("_")} if params_path else {}
        return lambda meta, spec: PIDWaterResetPolicy(meta, spec, **kw)
    from smart_buildings.evaluate import POLICY_MAP, initialize_policy
    cls = POLICY_MAP[key]
    return lambda meta, spec: initialize_policy(cls, meta, spec)


def policy_dirname(key: str) -> str:
    return key.replace(":", "-")


def run_one(scenario_dir: Path, policy_key: str, steps: int, out_dir: Path, params_path: Path | None = None) -> dict:
    from smart_buildings import evaluate as ev
    from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker
    from harness.billing_fix import apply as billing_fix

    billing_fix()  # analysis only: AETHER_BILLING_FIX=1 bills gas at the actual tank temperature
    scenario_dir, out_dir = Path(scenario_dir).resolve(), Path(out_dir).resolve()
    params_path = Path(params_path).resolve() if params_path else None
    t0 = time.time()
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
    policy = resolve_policy(policy_key, params_path, scenario_dir)(meta, env.action_spec())
    compact = ev.run_policy(env, tracker, policy, steps)
    rc = tracker.reward_components
    summary = {
        "policy": policy_key,
        "total_reward": float(np.sum(tracker.rewards)),
        "total_steps": len(tracker.rewards),
        "avg_reward": float(np.mean(tracker.rewards)),
        "total_normalized_productivity_regret": float(sum(r["normalized_productivity_regret"] for r in rc)),
        "total_normalized_energy_cost": float(sum(r["normalized_energy_cost"] for r in rc)),
        "total_normalized_carbon_emission": float(sum(r["normalized_carbon_emission"] for r in rc)),
        # fuel split for the score anatomy: dollars and kg over the episode
        "total_electricity_cost": float(sum(r["electricity_energy_cost"] for r in rc)),
        "total_natural_gas_cost": float(sum(r["natural_gas_energy_cost"] for r in rc)),
        "total_carbon_kg": float(sum(r["carbon_emitted"] for r in rc)),
        "seconds_per_step": compact["seconds_per_step"],
        "wall_seconds": time.time() - t0,
        "params": str(params_path) if params_path else None,
    }
    if hasattr(policy, "stats"):
        summary["policy_stats"] = policy.stats()
    d = Path(out_dir) / scenario_dir.parent.name / scenario_dir.name / policy_dirname(policy_key)
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    return {"scenario": f"{scenario_dir.parent.name}/{scenario_dir.name}", **summary}


def _job(args):
    try:
        return run_one(*args)
    except Exception:  # re-raise as a plain, picklable error that carries the traceback
        raise RuntimeError(f"job {args[:2]} failed:\n{traceback.format_exc()}") from None


def _job_soft(args):
    try:
        return _job(args)
    except RuntimeError as e:  # report the failure, keep the other rollouts going
        print(e, file=sys.stderr, flush=True)
        return None


def run_many(jobs: list[tuple], steps: int, out_dir: Path, workers: int, params_path: Path | None = None,
             strict: bool = True):
    """jobs: (scenario, policy) or (scenario, policy, params_path, out_dir) tuples. Yields results as they finish.
    strict=False logs a failed rollout to stderr and skips it (evaluation); strict=True raises (tuning)."""
    args = [(Path(j[0]), j[1], steps, Path(j[3]) if len(j) > 3 else Path(out_dir), j[2] if len(j) > 2 else params_path) for j in jobs]
    fn = _job if strict else _job_soft
    if workers <= 1:
        yield from (r for r in map(fn, args) if r is not None)
        return
    with Pool(min(workers, len(args)), maxtasksperchild=1) as pool:
        yield from (r for r in pool.imap_unordered(fn, args) if r is not None)


def expand_scenarios(specs: list[str]) -> list[Path]:
    out: list[Path] = []
    for s in specs:
        hits = sorted(Path(p) for p in glob.glob(s) if (Path(p) / "config.gin").exists())
        if not hits:
            from harness.gen_pool import load_split
            hits = load_split(s)
        out += hits
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=8640)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--params", type=Path, default=None)
    a = ap.parse_args(argv)
    scen = expand_scenarios(a.scenarios)
    jobs = [(s, p) for p in a.policies for s in scen]
    t0 = time.time()
    for r in run_many(jobs, a.steps, a.out, a.workers, a.params):
        print(f"{r['scenario']:<32} {r['policy']:<16} total={r['total_reward']:12.3f} wall={r['wall_seconds']:7.1f}s", flush=True)
    print(f"done: {len(jobs)} jobs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
