"""Round One evaluation: rollouts and tables.

    uv run python -m harness.report run [--splits pool_test shipped minneapolis] [--workers 28]
    uv run python -m harness.report tables      -> results/round1/REPORT.md (+ figures)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from harness.gen_pool import STATIONS, load_split
from harness.run_parallel import ROOT, policy_dirname, run_many
from harness.score_table import STEPS, load_summaries

OUT = ROOT / "results/round1"
ARMS = ["rules", "prior", "mpc", "mpc_off", "mpc_v", "mpc_vp", "mpc_guard", "passive", "active", "passive_guard", "full"]
BASELINES = ["bang_bang", "fixed_setpoint", "pid", "pid_water"]
TUNE = ROOT / "results/tune"


def params_for(arm: str) -> Path | None:
    """Tuned parameters: the arm's own if tuned, else the shared Layer A parameters (tuned on the prior arm).
    `rules` and `pid_tuned` only use their own. '+oracle' arms share their base arm's parameters."""
    base = arm.split("+")[0]
    for cand in ((base,) if base in ("rules", "pid_tuned") else (base, "prior")):
        f = TUNE / cand / "best.json"
        if f.exists():
            return f
    return None


def run(splits, workers, arms=None, baselines=None, out_root: Path = OUT):
    """out_root: results tree (final-round runs go to their own, never mixed with Round One)."""
    arms = ARMS if arms is None else arms
    baselines = BASELINES if baselines is None else baselines
    jobs = []
    for split in splits:
        out = out_root / "runs" / split
        for s in load_split(split):
            for arm in arms:
                d = out / s.parent.name / s.name / policy_dirname(f"aether:{arm}") / "summary.json"
                if not d.exists():
                    jobs.append((s, f"aether:{arm}", params_for(arm), out))
            if "pid_tuned" in baselines and (TUNE / "pid_tuned/best.json").exists() \
                    and not (out / s.parent.name / s.name / "pid_tuned" / "summary.json").exists():
                jobs.append((s, "pid_tuned", TUNE / "pid_tuned/best.json", out))
            for b in baselines:
                if b != "pid_tuned" and not (out / s.parent.name / s.name / b / "summary.json").exists():
                    jobs.append((s, b, None, out))
    if __import__("os").environ.get("SWEEP_REVERSE"):  # a second machine works the list from the other end
        jobs = jobs[::-1]
    print(f"{len(jobs)} rollouts")
    for r in run_many(jobs, STEPS, out_root / "runs", workers, strict=False):
        print(f"{r['scenario']:<30} {r['policy']:<20} {abs(r['total_reward']) / STEPS:.5f}", flush=True)


def _scores(split, out_root: Path = OUT):
    """{policy: {scenario: |mean reward|}}, {policy: {scenario: summary}}"""
    summ = load_summaries(out_root / "runs" / split)
    sc, full = defaultdict(dict), defaultdict(dict)
    for k, d in summ.items():
        for p, s in d.items():
            sc[p][k] = abs(s["total_reward"]) / s["total_steps"]
            full[p][k] = s
    return sc, full


def boot_ci(a: dict, b: dict, n=10000, seed=0):
    """Paired bootstrap over scenarios of mean(a - b): (mean, lo95, hi95)."""
    keys = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in keys])
    rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def tables():
    L = ["# Round One results", "", "Score = mean over scenarios of |mean per-step reward| (lower is better), 8640-step episodes.", ""]
    names = [policy_dirname(f"aether:{a}") for a in ARMS + ["mpc_off+oracle", "mpc+oracle"]] + ["pid_tuned"] + BASELINES
    for split in ("pool_test", "shipped", "minneapolis"):
        if not (OUT / "runs" / split).exists():
            continue
        sc, full = _scores(split)
        pols = [p for p in names if p in sc]
        L += [f"## {split} ({len(next(iter(sc.values())))} scenarios)", "", "| policy | score | worst scenario | comfort | energy | carbon | guard / zone-step | probe steps |", "|---|---|---|---|---|---|---|---|"]
        for p in pols:
            v = sc[p]
            w = max(v, key=v.get)
            comp = {c: np.mean([abs(full[p][k][f"total_normalized_{c}"]) / STEPS for k in v]) for c in ("productivity_regret", "energy_cost", "carbon_emission")}
            gi = [full[p][k].get("policy_stats", {}).get("guard_interventions", 0) for k in v]
            ps = [full[p][k].get("policy_stats", {}).get("probe_steps", 0) for k in v]
            L.append(f"| {p} | **{np.mean(list(v.values())):.5f}** | {w} {v[w]:.4f} | {comp['productivity_regret']:.4f} | "
                     f"{comp['energy_cost']:.4f} | {comp['carbon_emission']:.4f} | {np.mean(gi) / STEPS:.2f} | {np.mean(ps):.0f} |")
        L.append("")
        pairs = [("aether-mpc", "aether-prior", "planner (mpc) vs population-tuned rule (prior)"),
                 ("aether-mpc_off", "aether-mpc", "offset-free bias on vs off"),
                 ("aether-mpc_off+oracle", "aether-mpc_off", "oracle weather vs persistence (ship candidate)"),
                 ("aether-mpc_off", "pid_water", "ship candidate (mpc_off) vs pid_water"),
                 ("aether-mpc_guard", "aether-mpc", "tail shield on vs off"),
                 ("aether-full", "aether-prior", "online adaptation (full) vs frozen prior"),
                 ("aether-passive", "aether-prior", "no guard or probe: passive vs prior"),
                 ("aether-passive_guard", "aether-passive", "guard on vs off"),
                 ("aether-active", "aether-passive", "probing on vs off"),
                 ("aether-full", "pid_water", "full vs pid_water")]
        L += ["| comparison | mean diff (neg = first better) | 95% CI (paired bootstrap) |", "|---|---|---|"]
        for a, b, lab in pairs:
            if a in sc and b in sc:
                m, lo, hi = boot_ci(sc[a], sc[b])
                L.append(f"| {lab} | {m:+.5f} | [{lo:+.5f}, {hi:+.5f}]{' *' if hi < 0 or lo > 0 else ''} |")
        L.append("")
        if split == "shipped":
            L += ["### Per held-out station (descriptive; worst fold not used for testing)", "",
                  "| policy | " + " | ".join(STATIONS) + " | worst |", "|---|" + "---|" * (len(STATIONS) + 1)]
            for p in pols:
                fold = {st: np.mean([v for k, v in sc[p].items() if k.split("/")[1].startswith(st)]) for st in STATIONS}
                L.append(f"| {p} | " + " | ".join(f"{fold[st]:.4f}" for st in STATIONS) + f" | {max(fold.values()):.4f} |")
            L.append("")
        if split == "pool_test":
            L += ["### median-zone rolling 1-h identification error by day (K)", "", "| arm | " + " | ".join(f"d{d}" for d in range(1, 30, 4)) + " |", "|---|" + "---|" * 8]
            for a in ("aether-passive", "aether-active"):
                if a in full:
                    days = np.array([[x if x is not None else np.nan for x in s.get("policy_stats", {}).get("id_rmse_by_day", [])][:29] for s in full[a].values()], float)
                    med = np.nanmedian(days, 0)
                    L.append(f"| {a} | " + " | ".join(f"{med[d - 1]:.3f}" for d in range(1, 30, 4)) + " |")
            L.append("")
            if "bang_bang" in full:
                L += ["### Runtime vs bang-bang (wall seconds per rollout, same batch)", "", "| policy | median ratio | max ratio |", "|---|---|---|"]
                for p in pols:
                    r = [full[p][k]["wall_seconds"] / full["bang_bang"][k]["wall_seconds"] for k in full[p] if k in full["bang_bang"]]
                    L.append(f"| {p} | {np.median(r):.2f} | {np.max(r):.2f} |")
                L.append("")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n")
    _figures()
    print("\n".join(L))


def _figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not (OUT / "runs/pool_test").exists():
        return
    sc, _ = _scores("pool_test")
    pols = [p for p in [policy_dirname(f"aether:{a}") for a in ARMS] + ["pid_tuned"] + BASELINES if p in sc]
    keys = sorted(next(iter(sc.values())))
    M = np.array([[sc[p].get(k, np.nan) for k in keys] for p in pols])
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(np.log10(M), aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(pols)), pols)
    ax.set_xticks(range(len(keys)), [k.split("/")[1] for k in keys], rotation=90, fontsize=7)
    fig.colorbar(im, label="log10 |mean reward|")
    fig.tight_layout()
    fig.savefig(OUT / "fig_heatmap_pool_test.pdf")
    fig, ax = plt.subplots(figsize=(7, 3.5))
    means = [np.nanmean(M[i]) for i in range(len(pols))]
    ax.barh(pols, means, color="#2a78d6")
    ax.invert_yaxis()
    ax.set_xlabel("score on pool_test (lower is better)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ablation_pool_test.pdf")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "tables"])
    ap.add_argument("--splits", nargs="+", default=["pool_test", "shipped", "minneapolis"])
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--baselines", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    run(a.splits, a.workers, a.arms, a.baselines, a.out) if a.cmd == "run" else tables()


if __name__ == "__main__":
    main()
