"""Score a results tree exactly like the kit's score_leaderboard.py, plus diagnostics.

    uv run python -m harness.score_table results/baselines [--compare external/starter-kit/baseline_returns.json]
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

STEPS = 8640


def load_results(results_dir: Path) -> dict[str, dict[str, float]]:
    """{"building/weather": {policy: total_reward}} (same glob as the kit's load_tree)."""
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for f in Path(results_dir).glob("building_*/*/*/summary.json"):
        out[f"{f.parts[-4]}/{f.parts[-3]}"][f.parts[-2]] = float(json.loads(f.read_text())["total_reward"])
    return dict(out)


def load_summaries(results_dir: Path) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    for f in Path(results_dir).glob("building_*/*/*/summary.json"):
        out[f"{f.parts[-4]}/{f.parts[-3]}"][f.parts[-2]] = json.loads(f.read_text())
    return dict(out)


def score(totals: dict[str, float], steps: int = STEPS) -> float:
    """Leaderboard metric: mean over scenarios of |total_reward / steps|. Lower is better."""
    return sum(abs(t) / steps for t in totals.values()) / len(totals)


def by_policy(tree: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    pol: dict[str, dict[str, float]] = defaultdict(dict)
    for key, d in tree.items():
        for p, v in d.items():
            pol[p][key] = v
    return dict(pol)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--compare", type=Path, help="json {scenario: {policy: total}} to check reproduction against")
    a = ap.parse_args(argv)
    summ = load_summaries(a.results_dir)
    steps = {s["total_steps"] for d in summ.values() for s in d.values()}
    if steps != {STEPS}:
        print(f"WARNING: episode lengths {sorted(steps)}; only {STEPS}-step results are comparable")
    pol = by_policy(load_results(a.results_dir))
    names = sorted(pol, key=lambda p: score(pol[p]))
    keys = sorted({k for d in pol.values() for k in d})
    print(f"{'scenario':<34}" + "".join(f"{p[:14]:>15}" for p in names))
    for k in keys:
        print(f"{k:<34}" + "".join(f"{abs(pol[p][k]) / STEPS:15.5f}" if k in pol[p] else f"{'—':>15}" for p in names))
    print(f"{'SCORE (mean, lower better)':<34}" + "".join(f"{score(pol[p]):15.5f}" for p in names))
    print(f"{'n scenarios':<34}" + "".join(f"{len(pol[p]):>15}" for p in names))
    for p in names:
        worst = max(pol[p], key=lambda k: abs(pol[p][k]))
        print(f"worst scenario {p}: {worst} {abs(pol[p][worst]) / STEPS:.5f}")
    ok = True
    if a.compare:
        ref = json.loads(a.compare.read_text())
        for p in names:
            diffs = [abs(pol[p][k] - ref[k][p]) / abs(ref[k][p]) for k in pol[p] if k in ref and p in ref[k]]
            if diffs:
                mx = max(diffs)
                ok &= mx <= 1e-6
                print(f"reproduction {p}: {len(diffs)} scenarios, max relative diff {mx:.2e} {'OK' if mx <= 1e-6 else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
