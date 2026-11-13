"""Collect every number the Round One paper prints into paper/numbers.json (missing experiments -> null).

    uv run python -m harness.paper_numbers
"""

import csv
import json
import re
from pathlib import Path

import numpy as np

from harness.anatomy import _components, decompose, lever_table
from harness.score_table import load_summaries

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/numbers.json"


def _arm_table(runs: Path, policies: list[str]) -> dict:
    if not runs.exists():
        return {}
    out = {}
    for p in policies:
        d = decompose(runs, p)
        if d["by_scenario"]:
            out[p] = {"n": len(d["by_scenario"]), **{k: round(v, 5) for k, v in d["mean"].items()},
                      "by_station": {s: round(v["score"], 5) for s, v in d["by_station"].items()},
                      "worst": max(d["by_scenario"].items(), key=lambda kv: kv[1]["score"])[0]}
    return out


def _paired(runs: Path, a: str, b: str, n=10000, seed=0):
    """Paired bootstrap of mean(score_a - score_b) over shared scenarios."""
    if not runs.exists():
        return None
    s = load_summaries(runs)
    keys = [k for k, d in s.items() if a in d and b in d]
    if len(keys) < 3:
        return None
    diff = np.array([abs(s[k][a]["total_reward"]) / s[k][a]["total_steps"] - abs(s[k][b]["total_reward"]) / s[k][b]["total_steps"] for k in keys])
    bs = diff[np.random.default_rng(seed).integers(0, len(diff), (n, len(diff)))].mean(1)
    m = float(np.mean([abs(s[k][b]["total_reward"]) / s[k][b]["total_steps"] for k in keys]))
    return {"n": len(keys), "mean": round(float(diff.mean()), 5), "lo": round(float(np.percentile(bs, 2.5)), 5),
            "hi": round(float(np.percentile(bs, 97.5)), 5), "pct": round(100 * diff.mean() / m, 1),
            "pct_lo": round(100 * np.percentile(bs, 2.5) / m, 1), "pct_hi": round(100 * np.percentile(bs, 97.5) / m, 1)}


def _idcheck():
    f = ROOT / "results/idcheck/REPORT.md"
    if not f.exists():
        return None
    m = re.search(r"\*\*median\*\* \| \| \*\*([\d.]+)\*\* \| \*\*([\d.]+)\*\* \| \*\*([\d.]+)\*\* \| \*\*([\d.]+)\*\* \| \*\*([\d.]+)\*\*", f.read_text())
    if not m:
        return None
    prior, h24, h48, h120, trusted = map(float, m.groups())
    return {"prior": prior, "online_24h": h24, "online_48h": h48, "online_120h": h120, "trusted_end": trusted,
            "improvement_48h_pct": round(100 * (1 - h48 / prior), 1), "error_change_48h_pct": round(100 * (h48 / prior - 1), 1)}


def _tuning(arm="prior"):
    d = ROOT / "results/tune" / arm
    if not (d / "history.csv").exists():
        return None
    rows = list(csv.DictReader(open(d / "history.csv")))
    best = json.loads((d / "best.json").read_text()) if (d / "best.json").exists() else {}
    return {"evals": len(rows), "first": round(float(rows[0]["score"]), 5), "best": round(min(float(r["score"]) for r in rows), 5),
            "params": {k: v for k, v in best.items() if not k.startswith("_")}}


def _rows(runs: Path, policy: str) -> dict:
    """{scenario: components} for one policy in one results tree (empty if absent)."""
    if not runs.exists():
        return {}
    return {k: _components(d[policy]) for k, d in load_summaries(runs).items() if policy in d}


def _is_large(key: str) -> bool:
    """Stress buildings alternate shipped / generator-default size ranges by slot (harness/gen_pool.add_stress)."""
    return ((int(key.split("/")[0].split("_")[1]) - 11000) // 100) % 2 == 1


def _summ(rows: dict) -> dict | None:
    if not rows:
        return None
    avg = lambda ks: {f: round(float(np.mean([rows[k][f] for k in ks])), 5) for f in ("score", "comfort", "energy", "carbon")}
    small, large = [k for k in rows if not _is_large(k)], [k for k in rows if _is_large(k)]
    return {"n": len(rows), **avg(list(rows)), "small": avg(small)["score"] if small else None, "large": avg(large)["score"] if large else None}


def _paired_rows(a: dict, b: dict, n=10000, seed=0):
    """Paired bootstrap of mean(a - b) over shared scenarios; also as % of b's mean."""
    keys = sorted(set(a) & set(b))
    if len(keys) < 3:
        return None
    diff = np.array([a[k]["score"] - b[k]["score"] for k in keys])
    bs = diff[np.random.default_rng(seed).integers(0, len(diff), (n, len(diff)))].mean(1)
    m = float(np.mean([b[k]["score"] for k in keys]))
    st = {}
    for k in keys:
        st.setdefault(k.split("/")[1].rsplit("_", 1)[0], []).append(100 * (a[k]["score"] / b[k]["score"] - 1))
    return {"n": len(keys), "mean": round(float(diff.mean()), 5), "lo": round(float(np.percentile(bs, 2.5)), 5),
            "hi": round(float(np.percentile(bs, 97.5)), 5), "pct": round(100 * diff.mean() / m, 1),
            "pct_lo": round(100 * np.percentile(bs, 2.5) / m, 1), "pct_hi": round(100 * np.percentile(bs, 97.5) / m, 1),
            "by_station_pct": {s: round(float(np.mean(v)), 1) for s, v in st.items()}, "wins": int((diff < 0).sum())}


def _diff_of_diffs(p: dict, b: dict, i: dict, n=10000, seed=0):
    """Paired bootstrap of (i - b) - (b - p) over shared scenarios, as % of p's mean: is the effect beyond the
    model (inversion minus perfect-model planner) larger than the model effect (perfect model minus prior)?"""
    keys = sorted(set(p) & set(b) & set(i))
    if len(keys) < 3:
        return None
    P, B, I = (np.array([d[k]["score"] for k in keys]) for d in (p, b, i))
    dd, m = (I - B) - (B - P), P.mean()
    bs = dd[np.random.default_rng(seed).integers(0, len(dd), (n, len(dd)))].mean(1)
    return {"n": len(keys), "pct": round(100 * dd.mean() / m, 1), "pct_lo": round(100 * np.percentile(bs, 2.5) / m, 1),
            "pct_hi": round(100 * np.percentile(bs, 97.5) / m, 1), "wins": int((dd < 0).sum())}


def _holiday_scenarios(tree: Path) -> int | None:
    """Scenarios in which the holiday rest fired at least once (policy_stats.holiday_steps > 0)."""
    s = load_summaries(tree) if tree.exists() else {}
    hits = [d["aether-mpc_off"].get("policy_stats", {}).get("holiday_steps", 0) for d in s.values() if "aether-mpc_off" in d]
    return sum(h > 0 for h in hits) if hits else None


STRESS30 = {  # arm -> (results tree, policy dir); 48 stress scenarios, 30 days
    "pid_water": ("results/round1/runs/stress", "pid_water"),
    "rules": ("results/round1/runs/stress", "aether-rules"),
    "r1": ("results/round1/runs/stress", "aether-mpc_off"),  # the Round One policy (tuned mpc_off)
    "c5": ("results/sweep/8640_stress/c5", "aether-mpc_off"),
    "c6": ("results/sweep/8640_stress/c6", "aether-mpc_off"),
    "econ": ("results/sweep/8640_stress/econ", "aether-mpc_off"),
    "zc": ("results/sweep/8640_stress/zc", "aether-mpc_off"),
    "hol9zc": ("results/sweep/8640_stress/hol9zc", "aether-mpc_off"),
    "inv": ("results/sweep/8640_stress/inv", "aether-inv"),  # one-step inversion arm (aether/inv.py), zc params
}
STRESS7 = ("c5", "a05", "econ", "zc", "hol9", "hol9zc", "ls1", "tank", "zcfix", "tankfix", "area1", "zcfloor", "ra1", "ra1s1", "zcf278")  # ra1*: reheat through airflow (prior_reheat_air.json), halving kept / removed; 7-day stress ablations (results/sweep/2016_stress);
STRESS7X = {  # 7-day stress, other policies: arm -> (results tree, policy dir)
    "inv": ("results/sweep/2016_stress/inv", "aether-inv"),
    "ctrlw5": ("results/sweep/2016_stress/entrants", "ext-ctrl_w5-CtrlW5"),
    "sb278": ("results/sweep/2016_stress/sb278", "aether-inv"),  # inversion held above 278 K while empty  # public controller CtrlW5 (MorS1337/sb26-stop-heating, Apache-2.0)
    "invfix": ("results/sweep/2016_stress/invfix", "aether-inv"),  # inversion with gas billed at the actual tank temperature (AETHER_BILLING_FIX=1)
}
# *fix = same configuration scored with corrected gas billing (AETHER_BILLING_FIX=1, harness/billing_fix.py)


def _min_unocc(tree: Path, policy="aether-mpc_off") -> dict | None:
    """Lowest zone temperature while unoccupied (policy_stats.min_T_unocc, K) over scenarios, in C."""
    s = load_summaries(tree) if tree.exists() else {}
    v = [d[policy]["policy_stats"]["min_T_unocc"] - 273.15 for d in s.values()
         if policy in d and "min_T_unocc" in d[policy].get("policy_stats", {})]
    return {"n": len(v), "min": round(min(v), 1), "p10": round(float(np.percentile(v, 10)), 1),
            "median": round(float(np.median(v)), 1)} if v else None


def _cold(tree: Path) -> dict | None:
    """Coldest fresh grid scenarios: lowest empty-building zone temperature (C) of the final configuration against
    the official baseline and the initial planner with its setback floor (harness.min_unocc, results/min_unocc)."""
    zc = {k: d["aether-mpc_off"]["policy_stats"]["min_T_unocc"] - 273.15 for k, d in load_summaries(tree).items()
          if "min_T_unocc" in d.get("aether-mpc_off", {}).get("policy_stats", {})} if tree.exists() else {}
    o = {p: {r["scenario"]: r["min_T_unocc_C"] for r in map(json.loads, (ROOT / f"results/min_unocc/{p}.jsonl").read_text().splitlines())}
         for p in ("pid_water", "r1") if (ROOT / f"results/min_unocc/{p}.jsonl").exists()}
    keys = sorted(set(zc) & set(o.get("pid_water", {})) & set(o.get("r1", {})))
    if not keys:
        return None
    col = lambda v: {"min": round(min(v), 1), "max": round(max(v), 1)}
    return {"n": len(keys), "zc": col([zc[k] for k in keys]), "pid_water": col([o["pid_water"][k] for k in keys]),
            "r1": col([o["r1"][k] for k in keys]), "gap_vs_pid": col([o["pid_water"][k] - zc[k] for k in keys])}


def _signed_err() -> dict | None:
    """Mean signed 1-h prediction error (K, predicted - simulated) by size range, zone mode and gain scale
    (harness.signed_err on the eight bound scenarios); None until all eight are in."""
    fs = sorted((ROOT / "results/signed_err").glob("*.json"))
    if len(fs) < 8:
        return None
    rs = [json.loads(f.read_text()) for f in fs]
    out = {}
    for size, big in (("small", False), ("large", True)):
        for scale, tag in (("1.0", "fitted"), ("0.5", "halved")):
            for mode in ("heat", "cool", "idle"):
                cells = [r["by_scale"][scale][mode] for r in rs if _is_large(r["scenario"]) == big and r["by_scale"][scale][mode]["n"]]
                n = sum(c["n"] for c in cells)
                out.setdefault(size, {}).setdefault(tag, {})[mode] = round(sum(c["mean"] * c["n"] for c in cells) / n, 2) if n else None
    return out


def _ahu_off(tree: Path) -> dict | None:
    """Share of occupied zone-steps whose AHU is off (policy_stats occ_zone_steps*), per policy over the stress grid."""
    out = {}
    for pol, tag in (("aether-mpc_off", "planner"), ("aether-inv", "inv")):
        st = [d[pol]["policy_stats"] for d in (load_summaries(tree) if tree.exists() else {}).values()
              if d.get(pol, {}).get("policy_stats", {}).get("occ_zone_steps")]
        if len(st) == 48:
            fr = [x["occ_zone_steps_ahu_off"] / x["occ_zone_steps"] for x in st]
            out[tag] = {"pooled": round(100 * sum(x["occ_zone_steps_ahu_off"] for x in st) / sum(x["occ_zone_steps"] for x in st), 1),
                        "median": round(100 * float(np.median(fr)), 1), "max": round(100 * max(fr), 1)}
    return out or None


def _rl_big() -> dict:
    """Fresh grid replication: rl_big split (other seeds, strips 2/4/6/8), 7 days."""
    r = {"pid_water": _rows(ROOT / "results/round1/runs/rl_big7", "pid_water"),
         **{a: _rows(ROOT / "results/sweep/2016_rl_big" / a, "aether-mpc_off") for a in ("c5", "zc")},
         "inv": _rows(ROOT / "results/sweep/2016_rl_big/inv", "aether-inv"),
         "rules": _rows(ROOT / "results/sweep/2016_rl_big/rules", "aether-rules")}
    r = {a: v if len(v) == 46 else {} for a, v in r.items()}  # rl_big has 46 scenarios; an arm enters only when complete
    return {"summ": {a: _summ(v) for a, v in r.items()},
            "ci": {f"{a}_vs_{b}": _paired_rows(r[a], r[b]) for a, b in (("zc", "c5"), ("zc", "pid_water"), ("c5", "pid_water"), ("inv", "zc"), ("zc", "rules"))},
            "min_unocc": {a: _min_unocc(ROOT / "results/sweep/2016_rl_big" / a) for a in ("c5", "zc")},
            "cold": _cold(ROOT / "results/sweep/2016_rl_big/zc")}


def _stress() -> dict:
    full = lambda r: r if len(r) == 48 else {}  # an arm enters the paper only with the whole grid
    s30 = {a: full(_rows(ROOT / d, p)) for a, (d, p) in STRESS30.items()}
    s7 = {a: full(_rows(ROOT / "results/sweep/2016_stress" / a, "aether-mpc_off")) for a in STRESS7}
    s7.update({a: full(_rows(ROOT / d, pol)) for a, (d, pol) in STRESS7X.items()})
    bound = _rows(ROOT / "results/sweep/2016_stress/simbound", "aether-mpc_off")
    bound = bound if len(bound) == 8 else {}  # all eight bound scenarios or none
    econ8 = {k: v for k, v in s7["econ"].items() if k in bound}
    inv8 = {k: v for k, v in s7["inv"].items() if k in bound}  # the inversion on the same eight scenarios
    return {
        "s30": {a: _summ(r) for a, r in s30.items()},
        "s30_ci": {f"{a}_vs_{b}": _paired_rows(s30[a], s30[b]) for a, b in (
            ("zc", "pid_water"), ("zc", "rules"), ("zc", "r1"), ("c6", "c5"), ("econ", "c6"), ("zc", "econ"), ("hol9zc", "zc"), ("c5", "r1"), ("inv", "zc"))},
        "s7": {a: _summ(r) for a, r in s7.items()},
        "s7_ci": {f"{a}_vs_{b}": _paired_rows(s7[a], s7[b]) for a, b in (
            ("a05", "c5"), ("econ", "a05"), ("zc", "econ"), ("hol9", "econ"), ("hol9zc", "econ"), ("ls1", "a05"), ("tank", "a05"), ("tank", "econ"), ("zcfix", "zc"), ("tankfix", "zcfix"), ("zc", "c5"), ("area1", "zc"), ("inv", "zc"), ("inv", "ctrlw5"), ("ctrlw5", "zc"), ("zcfloor", "zc"), ("sb278", "inv"), ("ra1", "zc"), ("ra1s1", "zc"), ("zcf278", "zc"), ("invfix", "inv"), ("invfix", "zcfix"))},
        "bound": {"model": _summ(econ8), "sim": _summ(bound), "ci": _paired_rows(bound, econ8),
                  "inv": _summ(inv8), "ci_inv": _paired_rows(inv8, econ8), "ci_inv_sim": _paired_rows(inv8, bound),
                  "ci_rest_minus_model": _diff_of_diffs(econ8, bound, inv8)},
        "holiday_scenarios": _holiday_scenarios(ROOT / "results/sweep/8640_stress/hol9zc"),
        "rl_big": _rl_big(),
        "min_unocc_f278": _min_unocc(ROOT / "results/sweep/2016_stress/zcf278"),  # planner with a 5 C floor (floor_abs 278.15)
        "ahu_off": _ahu_off(ROOT / "results/sweep/2016_stress/ahucount"),
        "signed_err": _signed_err(),
    }


KAGGLE = {"first": 0.01823, "c5": 0.01623, "zc": 0.01602, "inv": 0.01032, "ctrlw5": 0.01024, "leader": 0.0043}  # public leaderboard, read off Kaggle (not selected on); ctrlw5 as reported by its authors


def main():
    base = ["pid_water", "pid", "fixed_setpoint", "bang_bang"]
    arms = ["aether-rules", "aether-prior", "aether-passive", "aether-mpc", "aether-mpc_off", "aether-mpc_guard", "aether-full",
            "aether-mpc_off+oracle", "pid_tuned"]
    n = {
        "baselines_shipped": _arm_table(ROOT / "results/baselines", base),
        "subset10": _arm_table(ROOT / "results/layerA_untuned", ["pid_water"] + arms),
        "subset10_value_proxy": _arm_table(ROOT / "results/x2_floor", ["aether-mpc"]),
        "offset3": _arm_table(ROOT / "results/x2_offset", ["aether-mpc_off"]),  # planner + Layer A lead/floor
        "levers": [{"lever": a, "delta_pct": round(b, 1), "dcomfort": round(c, 5)} for a, b, c in lever_table(ROOT / "results/x0")]
        if (ROOT / "results/x0/reset_idle.jsonl").exists() else [],
        "idcheck": _idcheck(),
        "stress": _stress(),
        "kaggle": KAGGLE,
        "tuning_prior": _tuning("prior"),
        "tuning_rules": _tuning("rules"),
        "tuning_pid": _tuning("pid_tuned"),
        "x5_guard": _arm_table(ROOT / "results/x5_guard", ["aether-prior", "aether-passive", "aether-active", "aether-passive_guard", "aether-full"]),
        "pool_test": _arm_table(ROOT / "results/round1/runs/pool_test", ["pid_water"] + arms),
        "shipped": _arm_table(ROOT / "results/round1/runs/shipped", ["pid_water"] + arms),
        "ci": {  # primary split pool_test; shipped set as a robustness check
            "ship_vs_pid_pool_test": _paired(ROOT / "results/round1/runs/pool_test", "aether-mpc_off", "pid_water"),
            "ship_vs_rules_pool_test": _paired(ROOT / "results/round1/runs/pool_test", "aether-mpc_off", "aether-rules"),
            "rq1_mpc_vs_prior_pool_test": _paired(ROOT / "results/round1/runs/pool_test", "aether-mpc", "aether-prior"),
            "offset_vs_none_pool_test": _paired(ROOT / "results/round1/runs/pool_test", "aether-mpc_off", "aether-mpc"),
            "prior_vs_pid_pool_test": _paired(ROOT / "results/round1/runs/pool_test", "aether-prior", "pid_water"),
            "rules_vs_pid_pool_test": _paired(ROOT / "results/round1/runs/pool_test", "aether-rules", "pid_water"),
            "pidtuned_vs_pid_pool_test": _paired(ROOT / "results/round1/runs/pool_test", "pid_tuned", "pid_water"),
            "x3a_oracle_vs_persistence_pool_test": _paired(ROOT / "results/round1/runs/pool_test", "aether-mpc_off+oracle", "aether-mpc_off"),
            "x3c_online_vs_frozen_pool_test": _paired(ROOT / "results/round1/runs/pool_test", "aether-passive", "aether-prior"),
            "x4_guard_pool_test": _paired(ROOT / "results/round1/runs/pool_test", "aether-mpc_guard", "aether-mpc"),
            "full_vs_prior_pool_test": _paired(ROOT / "results/round1/runs/pool_test", "aether-full", "aether-prior"),
            "ship_vs_pid_shipped": _paired(ROOT / "results/round1/runs/shipped", "aether-mpc_off", "pid_water"),
            "ship_vs_rules_shipped": _paired(ROOT / "results/round1/runs/shipped", "aether-mpc_off", "aether-rules"),
            "pidtuned_vs_pid_shipped": _paired(ROOT / "results/round1/runs/shipped", "pid_tuned", "pid_water"),
            "x3c_online_vs_frozen_shipped": _paired(ROOT / "results/round1/runs/shipped", "aether-passive", "aether-prior"),
            "x3a_oracle_vs_persistence_shipped": _paired(ROOT / "results/round1/runs/shipped", "aether-mpc_off+oracle", "aether-mpc_off"),
        },
    }
    # the subset10 comparison is read from the rules/prior/mpc runs and pid_water from the baseline tree
    if n["subset10"]:
        keys = None
        s = load_summaries(ROOT / "results/layerA_untuned")
        keys = sorted(s)
        b = load_summaries(ROOT / "results/baselines")
        pid = [abs(b[k]["pid_water"]["total_reward"]) / 8640 for k in keys if k in b]
        n["subset10"]["pid_water"] = {"n": len(pid), "score": round(float(np.mean(pid)), 5)} if pid else None
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(n, indent=1))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
