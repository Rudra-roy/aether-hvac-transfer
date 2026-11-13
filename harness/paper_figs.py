"""Figures for the Round One paper, from paper/numbers.json and results/paper_traj (CPU only).

    uv run python -m harness.paper_figs      -> paper/latex/figs/*.pdf
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/latex/figs"
N = json.loads((ROOT / "paper/numbers.json").read_text())
TRAJ = ROOT / "results/paper_traj/building_1401/fairbanks_3"

# reference palette (dataviz skill), first three slots: validated all-pairs in light mode
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
TEXTWIDTH = 5.5  # inches, NeurIPS body

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],  # matches Figure 1
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

ARMS = [("pid_water", "Official baseline"), ("pid_tuned", "Baseline, retuned"), ("aether-rules", "Tuned rules"),
        ("aether-prior", "Rules, model start"), ("aether-mpc", "Planner"), ("aether-mpc_off", "Planner, offset")]


CI_ROWS = [  # plain label, CI key stem, reference arm for the relative scale
    ("Our planner vs official baseline", "ship_vs_pid", "pid_water"),
    ("Our planner vs tuned rules", "ship_vs_rules", "aether-rules"),
    ("Tuned rules vs official baseline", "rules_vs_pid", "pid_water"),
    ("Retuning the official baseline", "pidtuned_vs_pid", "pid_water"),
    ("Planning vs rules, same model", "rq1_mpc_vs_prior", "aether-prior"),
    ("Adding the per zone offset", "offset_vs_none", "aether-mpc"),
    ("Perfect weather forecast", "x3a_oracle_vs_persistence", "aether-mpc_off"),
    ("Online model identification", "x3c_online_vs_frozen", "aether-prior"),
    ("Predictive comfort filter", "x4_guard", "aether-mpc"),
]
GROUPS = [(0, "Controllers compared"), (4, "Single components")]  # first row index, header


def _pct(stem, split, ref):
    c = N["ci"].get(f"{stem}_{split}")
    if not c:
        return None
    base = N[split][ref]["score"]
    return tuple(100 * c[k] / base for k in ("mean", "lo", "hi"))


def _fmt(v):
    return f"{v:+.0f}%".replace("-", "−")


def forest():
    """Diverging bars: paired change in score (% of the reference mean) on the held out set, its 95% bootstrap interval as
    a thin line, and the shipped scenarios as an open circle where run. Left of zero is better."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    lim = (-106, 72)
    edge = lim[1] - 8  # bars beyond this are cut and marked with an arrow
    gaps = [sum(1 for g, _ in GROUPS if g <= i) for i in range(len(CI_ROWS))]
    y = [i + 1.15 * g for i, g in enumerate(gaps)]  # a header slot before each group
    fig, ax = plt.subplots(figsize=(0.8 * TEXTWIDTH, 2.25))  # printed at 0.8 linewidth, so fonts print at size
    for (lab, stem, ref), yi in zip(CI_ROWS, y):
        h = _pct(stem, "pool_test", ref)
        if h:
            m, lo, hi = h
            col = BLUE if m < 0 else ORANGE
            ax.barh(yi, max(min(m, edge), lim[0]), height=0.62, color=col, linewidth=0, zorder=2)
            ax.plot([max(lo, lim[0]), min(hi, edge)], [yi, yi], color=INK2, linewidth=0.9, zorder=3, solid_capstyle="butt")
            if m > edge:  # off scale: arrow at the edge, the value above it
                ax.annotate("", xy=(lim[1] - 1, yi), xytext=(edge, yi), arrowprops=dict(arrowstyle="-|>", color=col, lw=1.4, shrinkA=0, shrinkB=0))
                ax.text(edge - 1, yi, _fmt(m), ha="right", va="center", fontsize=7, color="white", weight="bold", zorder=5)
            elif m < -5:
                ax.text(max(lo, lim[0]) - 2, yi, _fmt(m), ha="right", va="center", fontsize=7, color=INK)
            else:
                ax.text(min(hi, edge) + 2, yi, _fmt(m), ha="left", va="center", fontsize=7, color=INK)
        s = _pct(stem, "shipped", ref)
        if s:
            ax.plot(s[0], yi, "o", ms=4.2, mfc="white", mec=INK, mew=0.9, zorder=4)
    for g, txt in GROUPS:
        ax.text(-0.01, y[g] - 1.1, txt, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7, color=INK, weight="bold")
    ax.axvline(0, color=INK2, linewidth=0.8, zorder=1)
    ax.set_yticks(y, [r[0] for r in CI_ROWS])
    ax.set_ylim(y[-1] + 0.6, -1.7)
    ax.set_xlim(*lim)
    ax.set_xticks([-100, -75, -50, -25, 0, 25, 50])
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=7, labelcolor=INK)
    ax.set_xlabel("Change in score (%)    ← better    worse →")
    ax.legend(handles=[Patch(color=BLUE, label="Held out, better"), Patch(color=ORANGE, label="Held out, worse"),
                       Line2D([], [], color=INK2, linewidth=0.9, label="95% interval"),
                       Line2D([], [], marker="o", ms=4.2, mfc="white", mec=INK, linestyle="", label="Shipped scenarios")],
              ncol=2, loc="lower right", bbox_to_anchor=(1.0, 1.0), handlelength=1.2, columnspacing=1.2, fontsize=6.5)
    fig.savefig(OUT / "forest.pdf")
    fig.savefig(OUT / "preview/forest.png", dpi=220)
    plt.close(fig)


TRAJ_ARMS = [("pid_water", "Official baseline"), ("aether-rules", "Tuned rules"), ("aether-mpc_off", "Planner, offset")]


def trajectories():
    """Outside temperature and zone-temperature spread over four days, one panel per controller."""
    runs = {k: np.load(TRAJ / f"{k}.npz") for k, _ in TRAJ_ARMS if (TRAJ / f"{k}.npz").exists()}
    if len(runs) < len(TRAJ_ARMS):
        print("trajectories missing:", [k for k, _ in TRAJ_ARMS if k not in runs])
        return
    ref = runs["pid_water"]
    n = min(len(r["hour"]) for r in runs.values())
    t = ref["hour"][0] / 24 + np.arange(n) / 288  # days since midnight before the episode start
    fig, axes = plt.subplots(4, 1, figsize=(TEXTWIDTH, 3.2), sharex=True, gridspec_kw={"height_ratios": [0.7, 1, 1, 1], "hspace": 0.55})
    axes[0].plot(t, ref["tout"][:n] - 273.15, color=INK2, linewidth=1)
    axes[0].set_ylabel("Outside\n(°C)")
    occ = ref["occ"][:n] > 0.5
    occ[1:] &= occ[:-1]  # the occupancy flag leads the setpoint sensors by one step
    for ax, (k, lab) in zip(axes[1:], TRAJ_ARMS):
        r = runs[k]
        lo, hi = r["hsp"][:n] - 273.15, r["csp"][:n] - 273.15
        ax.fill_between(t, r["t10"][:n] - 273.15, r["t90"][:n] - 273.15, color=BLUE, alpha=0.22, linewidth=0, label="Zones, 10th to 90th percentile")
        ax.plot(t, r["t50"][:n] - 273.15, color=BLUE, linewidth=0.8, label="Median zone")
        for edge, name in ((lo, "Comfort band, occupied hours"), (hi, None)):  # band edges drawn on top
            ax.plot(t, np.where(occ, edge, np.nan), color=INK, linewidth=0.9, label=name)
        ax.set_ylabel("Zone (°C)")
        ax.set_title(f"{lab}, score {float(r['score']):.4f} over these four days", color=INK, loc="left", pad=2)
        ax.set_ylim(10, 28)
    h, l = axes[1].get_legend_handles_labels()
    axes[0].legend(h, l, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02), handlelength=1.4)
    days = "Mon Tue Wed Thu Fri Sat Sun".split()
    d0 = int(ref["dow"][0])
    axes[-1].set_xticks(range(6), [days[(d0 + i) % 7] for i in range(6)])
    axes[-1].set_xlim(t[0], t[-1])
    axes[-1].set_xlabel("Local time, ticks at midnight")
    for ax in axes:
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    fig.align_ylabels(axes)
    fig.savefig(OUT / "trajectories.pdf")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    forest()
    trajectories()
    print("figures ->", OUT)
