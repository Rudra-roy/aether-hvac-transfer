"""Score anatomy. Where does a policy's score go, by reward component and station, plus the lever table.

    uv run python -m harness.anatomy --runs results/round1/runs/shipped --policies pid_water aether-prior aether-mpc
        -> results/round1/ANATOMY.md and fig_anatomy.pdf
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from harness.gen_pool import STATIONS
from harness.score_table import load_summaries

ROOT = Path(__file__).resolve().parents[1]
W = {"comfort": 0.6, "energy": 0.2, "carbon": 0.2}


def _components(s: dict) -> dict:
    n = s["total_steps"]
    c = W["comfort"] * abs(s["total_normalized_productivity_regret"]) / n
    e = W["energy"] * s["total_normalized_energy_cost"] / n
    k = W["carbon"] * s["total_normalized_carbon_emission"] / n
    return {"score": abs(s["total_reward"]) / n, "comfort": c, "energy": e, "carbon": k}


def decompose(runs_dir: Path, policy: str) -> dict:
    """{'mean': {...}, 'by_station': {station: {...}}, 'by_scenario': {key: {...}}} for one policy."""
    summ = load_summaries(runs_dir)
    rows = {k: _components(d[policy]) for k, d in summ.items() if policy in d}
    def avg(keys):
        return {f: float(np.mean([rows[k][f] for k in keys])) for f in ("score", "comfort", "energy", "carbon")}
    by_station = defaultdict(list)
    for k in rows:
        by_station[k.split("/")[1].split("_")[0]].append(k)
    return {"mean": avg(list(rows)), "by_station": {s: avg(v) for s, v in by_station.items()}, "by_scenario": rows}


def lever_table(x0_dir: Path) -> list[tuple[str, float, float]]:
    """(lever, mean score change %, comfort change) from the lever jsonl files against pid_water."""
    out = []
    reset = [json.loads(l) for l in (x0_dir / "reset_idle.jsonl").read_text().splitlines() if l.strip()]
    base = {}
    for f in (ROOT / "results/baselines").glob("*/*/pid_water/summary.json"):
        base[f"{f.parts[-4]}/{f.parts[-3]}"] = _components(json.loads(f.read_text()))
    keys = [r["scenario"] for r in reset if r["scenario"] in base]
    if keys:
        b = np.mean([base[k]["score"] for k in keys])
        r = np.mean([x["score"] for x in reset if x["scenario"] in base])
        dc = np.mean([x["comfort"] - base[x["scenario"]]["comfort"] for x in reset if x["scenario"] in base])
        out.append(("hot-water outdoor reset + loop idle (30 shipped)", 100 * (r / b - 1), float(dc)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "results/round1/runs/shipped")
    ap.add_argument("--policies", nargs="+", default=["pid_water", "aether-prior", "aether-mpc"])
    ap.add_argument("--out", type=Path, default=ROOT / "results/round1")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    L = ["# Score anatomy", "", f"Runs: `{a.runs}`. Score = comfort + energy + carbon shares (weights 0.6/0.2/0.2 applied).", ""]
    dec = {p: decompose(a.runs, p) for p in a.policies}
    dec = {p: d for p, d in dec.items() if d["by_scenario"]}
    L += ["| policy | score | comfort | energy | carbon | " + " | ".join(STATIONS) + " |", "|---|---|---|---|---|" + "---|" * len(STATIONS)]
    for p, d in dec.items():
        m = d["mean"]
        L.append(f"| {p} | **{m['score']:.4f}** | {m['comfort']:.4f} | {m['energy']:.4f} | {m['carbon']:.4f} | "
                 + " | ".join(f"{d['by_station'][s]['score']:.4f}" if s in d["by_station"] else "—" for s in STATIONS) + " |")
    levers = lever_table(ROOT / "results/x0") if (ROOT / "results/x0/reset_idle.jsonl").exists() else []
    if levers:
        L += ["", "## Single levers on pid_water", "", "| lever | mean score change | mean comfort change |", "|---|---|---|"]
        L += [f"| {n} | {d:+.1f}% | {c:+.4f} |" for n, d, c in levers]
    (a.out / "ANATOMY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    if dec:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, len(STATIONS), figsize=(12, 3.2), sharey=True)
        cols = {"comfort": "#2a78d6", "energy": "#eb6834", "carbon": "#1baf7a"}
        for ax, st in zip(axes, STATIONS):
            x = np.arange(len(dec))
            bottom = np.zeros(len(dec))
            for f in ("comfort", "energy", "carbon"):
                v = np.array([d["by_station"].get(st, {}).get(f, 0.0) for d in dec.values()])
                ax.bar(x, v, bottom=bottom, color=cols[f], label=f if st == STATIONS[0] else None, width=0.7)
                bottom += v
            ax.set_title(st)
            ax.set_xticks(x, [p.replace("aether-", "") for p in dec], rotation=45, ha="right", fontsize=8)
        axes[0].set_ylabel("score share (lower is better)")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(a.out / "fig_anatomy.pdf")


if __name__ == "__main__":
    main()
