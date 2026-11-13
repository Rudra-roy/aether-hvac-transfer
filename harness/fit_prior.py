"""Population prior from generated buildings, and the identifier go/no-go check.

    uv run python -m harness.fit_prior fit    # pool_train: pid_water + probing logs -> aether/prior.json
    uv run python -m harness.fit_prior check  # pool_test: online identifier vs prior -> results/idcheck/REPORT.md
"""

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from aether.identifier import Prior, ZoneIdentifier, fit_batch
from aether.obs import ObsParser
from aether.zone_model import TAU, features, regressors, rollout
from harness.gen_pool import load_split

ROOT = Path(__file__).resolve().parents[1]
KIT = ROOT / "external/starter-kit"
DAYS = int(os.environ.get("AETHER_DAYS", 7))
CHECKPOINTS_H = (24, 48, 120)
TEST_FROM = 5 * 288  # evaluation window: days 6-7
AIR = bool(int(os.environ.get("AETHER_REHEAT_AIR", 0)))  # discharge-temperature airflow terms (Params.reheat_air)
PRIOR_PATH = ROOT / ("aether/prior_reheat_air.json" if AIR else "aether/prior.json")


def _env(scenario):
    from smart_buildings import evaluate as ev
    from smart_buildings.smart_control.utils.scenario_generator import SimulationTracker
    cwd = os.getcwd()
    os.chdir(KIT)
    try:
        env = ev.load_hybrid_environment(Path(scenario).resolve() / "config.gin")
    finally:
        os.chdir(cwd)
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    return env, SimulationTracker(env, vmin=280, vmax=310, capture_snapshots=False), ev.create_building_meta_data(env, Path(scenario))


def _probe_policy_cls():
    from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy

    class ProbeData(PIDWaterResetPolicy):
        """pid_water + 02:00-03:00 pseudo-random excitation of one AHU group per night (data collection only)."""

        def __init__(self, meta, spec):
            super().__init__(meta, spec)
            from aether.actions import ActionWriter
            self.parser, self.writer = ObsParser(meta), None
            self.writer = ActionWriter(meta, self.parser)
            self.rng, self.n, self.level = np.random.default_rng(0), 0, None

        def get_action(self, obs):
            a = super().get_action(obs)
            s = self.parser.parse(obs)
            self.n += 1
            if not (2 <= s.hour < 3):
                return a
            cmd = self.writer.decode(a)
            k = (self.n // 288) % len(self.parser.ahus)
            g = self.parser.groups[k]
            if self.level is None or self.n % 6 == 0 or len(self.level[0]) != len(g):
                self.level = (self.rng.choice([0.2, 1.0], len(g)), self.rng.choice([0.0, 1.0], len(g)), float(self.rng.choice([286.0, 297.0])))
            cmd.damper[g], cmd.reheat[g] = self.level[0], self.level[1]
            cmd.ahu_run[k], cmd.sat[k], cmd.pressure[k], cmd.pump_run = True, self.level[2], 600.0, True
            return self.writer.write(cmd)

    return ProbeData


def _log(scenario, probe: bool):
    """(parser, list of (T, features)) for a DAYS-long pid_water(+probe) rollout."""
    from smart_buildings.baselines.pid_plus import PIDWaterResetPolicy
    env, tracker, meta = _env(scenario)
    parser = ObsParser(meta)
    pol = (_probe_policy_cls() if probe else PIDWaterResetPolicy)(meta, env.action_spec())
    tracker.reset()
    obs, out = tracker.last_obs, []
    for _ in range(DAYS * 288):
        s = parser.parse(obs)
        out.append((s.T, features(s, parser, AIR), s))
        obs, *_ = tracker.step(pol.get_action(obs))
    return parser, out


def design(log):
    Tm = [log[0][0]]
    for T, _, _ in log[:-1]:
        Tm.append(Tm[-1] + (T - Tm[-1]) / TAU)
    X = np.stack([regressors(log[t][0], Tm[t], log[t][1], log[t - 1][1]) for t in range(1, len(log) - 1)])
    y = np.stack([log[t + 1][0] - log[t][0] for t in range(1, len(log) - 1)])
    return X, y, Tm


def eval_theta(theta, log, Tm, t0, step=3):
    """Per-zone RMSE of 1-h predictions with fixed theta from every `step`-th start in [t0, end-12)."""
    errs = []
    for s in range(t0, len(log) - 12, step):
        pred = rollout(theta, log[s][0], Tm[s], [log[s + k][1] for k in range(12)], log[s - 1][1])[-1]
        errs.append(pred - log[s + 12][0])
    return np.sqrt(np.mean(np.square(errs), 0))


def _fit_job(args):
    scenario, probe = args
    _, log = _log(scenario, probe)
    X, y, _ = design(log)
    th, rv = fit_batch(X, y)
    return th, rv


def _check_job(args):
    scenario, prior_d, q_rw = args
    prior = Prior(np.array(prior_d["m0"]), np.array(prior_d["P0"]), prior_d["R"])
    parser, log = _log(scenario, False)
    ident = ZoneIdentifier(len(parser.vavs), prior, q_rw=q_rw)
    snaps = {}
    for t, (_, f, s) in enumerate(log):
        ident.observe(s, f)
        if t in [h * 12 for h in CHECKPOINTS_H]:
            snaps[t // 12] = ident.theta.copy()
    _, _, Tm = design(log)
    V = len(parser.vavs)
    res = {"scenario": f"{Path(scenario).parent.name}/{Path(scenario).name}", "zones": V,
           "prior": float(np.median(eval_theta(np.repeat(prior.m0[None], V, 0), log, Tm, TEST_FROM))),
           "trusted_end": float(ident.trusted.mean())}
    for h, th in snaps.items():
        res[f"online_{h}h"] = float(np.median(eval_theta(th, log, Tm, TEST_FROM)))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "check"])
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--split", default="pool_test")
    ap.add_argument("--q-rw", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        scen = load_split("pool_train")
        with Pool(a.workers, maxtasksperchild=1) as pool:
            fits = pool.map(_fit_job, [(s, p) for s in scen for p in (False, True)])
        ths = np.concatenate([f[0] for f in fits])
        prior = Prior(np.median(ths, 0), np.var(ths, 0) + 1e-8, float(np.mean([f[1] for f in fits])))
        prior.save(PRIOR_PATH, fitted_on=[f"{s.parent.name}/{s.name}" for s in scen], n_zones=int(len(ths)))
        print("prior m0", np.round(prior.m0, 4), "R", prior.R, "zones", len(ths))
        return
    prior_d = json.loads(PRIOR_PATH.read_text())
    scen = load_split(a.split)[: a.limit]
    with Pool(a.workers, maxtasksperchild=1) as pool:
        rows = pool.map(_check_job, [(s, prior_d, a.q_rw) for s in scen])
    out = ROOT / "results/idcheck"
    out.mkdir(parents=True, exist_ok=True)
    keys = ["prior"] + [f"online_{h}h" for h in CHECKPOINTS_H] + ["trusted_end"]
    lines = [f"# Identifier check ({a.split}, q_rw={a.q_rw:g}, pid_water, 7 days)", "",
             "Median-zone 1-h prediction RMSE (K) on days 6-7, with theta frozen at each checkpoint.", "",
             "| scenario | zones | " + " | ".join(keys) + " |", "|---|---|" + "---|" * len(keys)]
    for r in sorted(rows, key=lambda r: r["scenario"]):
        lines.append(f"| {r['scenario']} | {r['zones']} | " + " | ".join(f"{r[k]:.3f}" for k in keys) + " |")
    med = {k: float(np.median([r[k] for r in rows])) for k in keys}
    lines.append("| **median** | | " + " | ".join(f"**{med[k]:.3f}**" for k in keys) + " |")
    go = med["online_48h"] <= 0.75 and med["online_48h"] <= 0.7 * med["prior"]
    lines += ["", f"Criterion: median online_48h <= 0.75 K and >= 30% below prior "
              f"({med['online_48h']:.3f} vs prior {med['prior']:.3f}). **Decision: {'GO' if go else 'NO-GO'}**"]
    name = "REPORT.md" if (a.split, a.q_rw, a.limit, AIR) == ("pool_test", 1e-4, None, False) else f"REPORT_{a.split}_q{a.q_rw:g}{'_air' if AIR else ''}.md"
    (out / name).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-3:]))


if __name__ == "__main__":
    main()
