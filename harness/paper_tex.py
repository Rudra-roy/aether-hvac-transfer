"""paper/numbers.json -> paper/latex/numbers.tex (macros + table bodies for the LaTeX paper).

    uv run python -m harness.paper_numbers && uv run python -m harness.paper_tex

Every leaf of numbers.json becomes \\N{a/b/c} (as stored) and \\F{a/b/c} (4 decimals); \\CI{key} prints a paired
bootstrap interval. A missing value renders as red "pending", never as a silent blank.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, OUT = ROOT / "paper/numbers.json", ROOT / "paper/latex/numbers.tex"


def tt(s: str) -> str:
    return r"\texttt{" + s.replace("_", r"\_") + "}"


def fx(x: float, d: int) -> str:
    """Fixed decimals, trailing zeros dropped, never scientific notation or "-0"."""
    t = f"{x:.{d}f}".rstrip("0").rstrip(".")
    return "0" if t in ("-0", "") else t


def sg(x: float) -> str:
    """Signed, one decimal, never "-0.0"."""
    t = f"{x:+.1f}"
    return "0.0" if t in ("-0.0", "+0.0") else t


def f4(x) -> str:
    return r"\pending" if x is None else f"{x:.4f}"


def t4(x) -> str:  # tables: fixed four decimals so columns align
    return r"\pending" if x is None else f"{x:.4f}"


def get(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def leaves(d, prefix=""):
    if isinstance(d, dict):
        for k, v in d.items():
            yield from leaves(v, f"{prefix}{k}/")
    elif not isinstance(d, list):
        yield prefix[:-1], d


def rows(cells: list[list[str]]) -> str:
    return "\n".join(" & ".join(r) + r" \\" for r in cells)


BASE_LABEL = {"pid_water": "PID with water reset", "pid": "PID", "fixed_setpoint": "Fixed setpoint", "bang_bang": "Bang--bang"}
LABEL = {"pid_water": "Official baseline", "pid_tuned": "Baseline, retuned", "aether-rules": "Tuned rules",
         "aether-prior": "Rules, model start", "aether-mpc": "Planner", "aether-mpc_off": "Planner, offset"}
STRESS_LABEL = {"pid_water": "Official baseline", "rules": "Tuned rules", "r1": "Planner, initial",
                "c5": "With deeper setback", "c6": "With size correction", "econ": "With economizer",
                "zc": "With zone cooling cost", "inv": "One step inversion~\\cite{mors2026stopheating}"}


WORDS = "zero one two three four five six seven eight nine ten eleven twelve".split()


def build(n: dict) -> str:
    out = [r"\providecommand{\pending}{\textcolor{red}{\textbf{[TBD]}}}",
           r"\newcommand{\N}[1]{\ifcsname N:#1\endcsname\csname N:#1\endcsname\else\pending\fi}",
           r"\newcommand{\F}[1]{\ifcsname F:#1\endcsname\csname F:#1\endcsname\else\pending\fi}",
           r"\newcommand{\CI}[1]{\ifcsname CI:#1\endcsname\csname CI:#1\endcsname\else\pending\fi}",
           r"\newcommand{\FF}[1]{\ifcsname FF:#1\endcsname\csname FF:#1\endcsname\else\pending\fi}",
           r"\newcommand{\CIpct}[1]{\ifcsname CIpct:#1\endcsname\csname CIpct:#1\endcsname\else\pending\fi}",
           r"\newcommand{\Nw}[1]{\ifcsname W:#1\endcsname\csname W:#1\endcsname\else\pending\fi}"]  # small counts in words
    for k, v in leaves(n):
        if v is None:
            continue
        shown = f"${sg(v)}$" if isinstance(v, float) and "pct" in k else f"${v}$" if isinstance(v, float) and v < 0 else v  # true minus signs
        out.append(rf"\expandafter\def\csname N:{k}\endcsname{{{shown}}}")
        if isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 12:
            out.append(rf"\expandafter\def\csname W:{k}\endcsname{{{WORDS[v]}}}")
        if isinstance(v, float):
            out.append(rf"\expandafter\def\csname F:{k}\endcsname{{{f4(v)}}}")
            out.append(rf"\expandafter\def\csname FF:{k}\endcsname{{{v:.5f}}}")  # five decimals for small differences
    for k, c in (n.get("ci") or {}).items():
        if c:
            out.append(rf"\expandafter\def\csname CI:{k}\endcsname{{${sg(c['pct'])}$\%, 95\% CI [${sg(c['pct_lo'])}$, ${sg(c['pct_hi'])}$]}}")

    st0 = n.get("stress") or {}
    pcts = {**(st0.get("s7_ci") or {}), **(st0.get("s30_ci") or {}), "bound": get(st0, "bound", "ci"),
            "bound_inv": get(st0, "bound", "ci_inv"), "bound_inv_sim": get(st0, "bound", "ci_inv_sim"), "bound_diff": get(st0, "bound", "ci_rest_minus_model")}
    pcts.update({f"{d}/{k}": c for d in ("s7", "s30") for k, c in (st0.get(f"{d}_ci") or {}).items()})
    pcts.update({f"rlbig/{k}": c for k, c in (get(st0, "rl_big", "ci") or {}).items()})  # unambiguous when a key exists at both lengths
    for k, c in pcts.items():
        if c:
            out.append(rf"\expandafter\def\csname CIpct:{k}\endcsname{{${sg(c['pct'])}$\%, 95\% CI [${sg(c['pct_lo'])}$, ${sg(c['pct_hi'])}$]}}")

    def arm_row(name, r, cols):
        return [name] + [str(get(r, "n")) if c == "n" else t4(get(r, *c.split("/"))) for c in cols]

    sc = ["score", "comfort", "energy", "carbon"]
    tabs = {
        "TabBase": [arm_row(BASE_LABEL[p], get(n, "baselines_shipped", p), sc + [f"by_station/{s}" for s in ("fairbanks", "phoenix", "singapore", "atlanta")])
                    for p in ("pid_water", "pid", "fixed_setpoint", "bang_bang")],
        "TabLever": [[l["lever"], f"{l['delta_pct']}\\%", f"{l['dcomfort']}"] for l in n.get("levers", [])],
        "TabDev": [arm_row(name, r, ["n"] + sc + ["by_station/fairbanks", "by_station/phoenix"]) for name, r in (
            ("aether-rules", get(n, "subset10", "aether-rules")), ("aether-prior", get(n, "subset10", "aether-prior")),
            ("aether-mpc, no terminal value", get(n, "subset10", "aether-mpc")), ("aether-mpc", get(n, "subset10_value_proxy", "aether-mpc")))],
        "TabMain": [arm_row(p, get(n, "pool_test", p), ["n"] + sc)
                    for p in ("pid_water", "pid_tuned", "aether-rules", "aether-prior", "aether-mpc", "aether-mpc_off")],
        "TabShipped": [arm_row(p, get(n, "shipped", p), ["n"] + sc + [f"by_station/{s}" for s in ("fairbanks", "phoenix")])
                       for p in ("pid_water", "pid_tuned", "aether-rules", "aether-prior", "aether-mpc", "aether-mpc_off")],
    }
    arms = ("pid_water", "pid_tuned", "aether-rules", "aether-prior", "aether-mpc", "aether-mpc_off")
    best = {s: min((get(n, s, p, "score") for p in arms if get(n, s, p, "score") is not None), default=None) for s in ("pool_test", "shipped")}

    def both(p):  # pool_test | shipped side by side, best score in bold
        cells = [LABEL[p]]
        for s in ("pool_test", "shipped"):
            for c in sc:
                v = t4(get(n, s, p, c))
                cells.append(rf"\textbf{{{v}}}" if c == "score" and get(n, s, p, c) == best[s] else v)
        return cells
    tabs["TabBoth"] = [both(p) for p in arms]

    st = get(n, "stress") or {}

    def pct(c):
        return r"\pending" if not c else rf"${sg(c['pct'])}$ [${sg(c['pct_lo'])}$, ${sg(c['pct_hi'])}$]"

    s30 = [("pid_water", None), ("rules", None), ("r1", None), ("c5", "c5_vs_r1"), ("c6", "c6_vs_c5"), ("econ", "econ_vs_c6"), ("zc", "zc_vs_econ"), ("inv", "inv_vs_zc")]
    tabs["TabStress"] = [[STRESS_LABEL[a]] + [t4(get(st, "s30", a, c)) for c in ("score", "comfort", "energy", "carbon", "small", "large")]
                         + [pct(get(st, "s30_ci", k)) if k else "--"] for a, k in s30]
    abl = [("a05_vs_c5", "Size correction"), ("econ_vs_a05", "Economizer in the planner"),
           ("zc_vs_econ", "Zone cooling cost"),
           ("area1_vs_zc", "Floor plan gain instead of halving"), ("zcfloor_vs_zc", "Setback floor restored"), ("zcf278_vs_zc", "Planner with a soft 5\\degC{} floor when empty"),
           ("sb278_vs_inv", "Inversion with a 5\\degC{} floor when empty"),
           ("ra1_vs_zc", "Reheat through airflow"), ("ra1s1_vs_zc", "Reheat through airflow, no halving"), ("tank_vs_econ", "Billing inconsistency, not used")]
    tabs["TabAbl"] = [[lab, pct(get(st, "s7_ci", k))] for k, lab in abl]
    for name, cells in tabs.items():
        out.append(rf"\newcommand{{\{name}}}{{{rows(cells)}}}")
    return "\n".join(out) + "\n"


def main():
    OUT.write_text(build(json.loads(SRC.read_text())))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    t = build({"a": {"b": 0.123456, "c": None}, "ci": {"k": {"n": 3, "mean": -0.1, "lo": -0.2, "hi": -0.0, "pct": -10.0, "pct_lo": -20.0, "pct_hi": 0.04}}})
    assert fx(1e-05, 5) == "0.00001" and fx(-0.000001, 5) == "0"
    assert r"\csname N:a/b\endcsname{0.123456}" in t and r"\csname F:a/b\endcsname{0.1235}" in t and "N:a/c" not in t
    assert r"CI:k\endcsname{$-10.0$\%, 95\% CI [$-20.0$, $0.0$]}" in t
    main()
