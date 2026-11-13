"""Screen planner-cost variants of mpc_off on development scenarios (energy is 79% of our score).

    CUDA_VISIBLE_DEVICES="" nice -n 19 uv run python -m harness.sweep run  <steps> <variant=key:val,key:val> ...
    uv run python -m harness.sweep score <steps>

Each variant = the tuned parameters (results/tune/prior/best.json) + overrides, run as aether:mpc_off on
subset10 ∪ pool_train10 into results/sweep/<steps>/<variant>/. Never the held-out sets.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from harness.gen_pool import load_split
from harness.report import boot_ci
from harness.run_parallel import ROOT, run_many
from harness.score_table import load_summaries

BASE = Path(__import__("os").environ.get("SWEEP_BASE", ROOT / "results/tune/prior/best.json"))  # SWEEP_BASE: another params file
DEV = ("subset10", "pool_train10")
# SWEEP_SPLITS=pool_test,shipped reads held-out sets (once, for a chosen candidate) into results/sweep/<steps>_<splits>/
SPLITS = tuple(__import__("os").environ.get("SWEEP_SPLITS", ",".join(DEV)).split(","))
TAG = "" if SPLITS == DEV else "_" + "+".join(SPLITS)
ARM = __import__("os").environ.get("SWEEP_ARM", "mpc_off")  # mpc_r: put residual.npz in the variant dir first


def run(steps: int, variants: dict[str, dict], workers: int = 16):
    jobs = []
    for name, over in variants.items():
        root = ROOT / f"results/sweep/{steps}{TAG}/{name}"
        root.mkdir(parents=True, exist_ok=True)
        pj = root / "params.json"
        pj.write_text(json.dumps(json.loads(BASE.read_text()) | over, indent=1))
        for s in [s for split in SPLITS for s in load_split(split)]:
            if not (root / s.parent.name / s.name / f"aether-{ARM}/summary.json").exists():
                jobs.append((s, f"aether:{ARM}", pj, root))
    if __import__("os").environ.get("SWEEP_REVERSE"):  # a second machine works the same list from the other end
        jobs = jobs[::-1]
    print(len(jobs), "rollouts", flush=True)
    for r in run_many(jobs, steps, ROOT / "results/sweep", workers, strict=False):
        print(r["scenario"], r["params"].split("/")[-2], round(abs(r["total_reward"]) / r["total_steps"], 5), flush=True)


def score(steps: int, base: str = "base"):
    root = ROOT / f"results/sweep/{steps}{TAG}"
    sc, parts = {}, {}
    for v in sorted(p.name for p in root.iterdir() if p.is_dir()):
        summ = load_summaries(root / v)
        pick = lambda d: d.get("aether-mpc_off") or next((x for k, x in d.items() if k.startswith("aether-")), None)
        summ = {k: pick(d) for k, d in summ.items() if pick(d)}
        sc[v] = {k: abs(d["total_reward"]) / d["total_steps"] for k, d in summ.items()}
        parts[v] = {k: [0.6 * abs(d["total_normalized_productivity_regret"]), 0.2 * d["total_normalized_energy_cost"],
                        0.2 * d["total_normalized_carbon_emission"]] for k, d in summ.items()}
    full = len(sc[base])
    sc = {v: x for v, x in sc.items() if len(x) == full}  # complete variants only (a running sweep has partial ones)
    keys = sorted(set.intersection(*map(set, sc.values())))
    b = np.mean([sc[base][k] for k in keys])
    print(f"n={len(keys)} scenarios, {steps} steps; '{base}' = {b:.5f}")
    for v in sorted(sc, key=lambda v: np.mean([sc[v][k] for k in keys])):
        m, lo, hi = boot_ci({k: sc[v][k] for k in keys}, {k: sc[base][k] for k in keys})
        st = defaultdict(list)
        for k in keys:
            st[k.split("/")[1].rsplit("_", 1)[0]].append(sc[v][k] / sc[base][k] - 1)
        c, e, ca = (np.mean([parts[v][k][i] for k in keys]) / steps for i in range(3))
        print(f"{v:14s} {np.mean([sc[v][k] for k in keys]):.5f} ({100 * m / b:+5.1f}% [{100 * lo / b:+5.1f}, {100 * hi / b:+5.1f}])"
              f" | comfort {c:.5f} energy {e:.5f} carbon {ca:.5f} | " + " ".join(f"{s[:4]} {100 * np.mean(x):+4.0f}%" for s, x in sorted(st.items())))


if __name__ == "__main__":
    cmd, steps = sys.argv[1], int(sys.argv[2])
    if cmd == "run":
        variants = {}
        for spec in sys.argv[3:]:
            name, _, kv = spec.partition("=")
            variants[name] = {k: float(v) for k, v in (x.split(":") for x in kv.split(",") if x)}
        run(steps, variants, workers=int(__import__("os").environ.get("SWEEP_WORKERS", 24)))
    else:
        score(steps)
