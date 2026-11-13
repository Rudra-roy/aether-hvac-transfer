"""Perfect-model bound: the planner with the true simulator as its model (experiment only, never in a submission).

    OMP_NUM_THREADS=1 uv run python -m harness.sim_bound <scenario_dir> <params.json> <steps> <out_root> [children]

Same policy (aether:mpc_off) and same candidate search; only the planner's rollouts change: each candidate
(plant setting x one zone action per group) is played for the horizon on a forked copy of the live simulator.
Forking copies memory copy-on-write, so a fresh child per candidate starts exactly at the current state.
Writes <out_root>/<building>/<scenario>/aether-mpc_off/summary.json in run_parallel's format.
"""
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

from aether.actions import Command
from aether.mpc import RECIRC_MAX, TW_IDLE

_JOB = None  # (env, parser, actions, H), inherited by forked children


def _play(i):
    env, parser, actions, H = _JOB
    out = []
    for _ in range(H):
        obs, *_ = env.step(actions[i])
        out.append(parser.parse(obs).T)
    return np.stack(out)


class SimRollout:
    def __init__(self, env, policy, children: int):
        from smart_buildings import evaluate as ev
        self.env, self.parser, self.writer, self.children = env, policy.parser, policy.writer, children
        self.validate = lambda a: ev.validate_action(a, env.action_spec())
        self.ctx = mp.get_context("fork")
        self.calls, self.t0, self.name = 0, time.time(), ""

    def __call__(self, cand: dict, H: int) -> dict:
        global _JOB
        groups, V, K = self.parser.groups, len(self.parser.vavs), len(self.parser.ahus)
        variants = sorted({(key[0], key[1], key[3], key[4], key[5], key[6]) for key in cand})
        actions = []
        for pump_on, tw, ahu_on, sat, pressure, za in variants:
            cmd = Command.default(V, K)
            for k, g in enumerate(groups):
                key = (pump_on, tw, k, ahu_on, sat, pressure, za)
                if len(g) == 0 or key not in cand:
                    continue
                d, _, v, _, _ = cand[key]
                cmd.ahu_run[k], cmd.sat[k], cmd.pressure[k] = ahu_on, sat, pressure if ahu_on else 200.0
                cmd.damper[g], cmd.reheat[g], cmd.recirc[k] = d, v, RECIRC_MAX
            cmd.pump_run = bool(pump_on and (cmd.reheat > 0).any())
            cmd.tw_sp = tw if cmd.pump_run else TW_IDLE
            actions.append(self.validate(self.writer.write(cmd)))
        _JOB = (self.env, self.parser, actions, H)
        with self.ctx.Pool(self.children, maxtasksperchild=1) as pool:
            res = pool.map(_play, range(len(actions)), chunksize=1)
        _JOB = None
        self.calls += 1
        if self.calls % 96 == 0:  # every simulated day at every=3
            print(self.name, "day", self.calls // 96, "min", round((time.time() - self.t0) / 60, 1), flush=True)
        by_variant = dict(zip(variants, res))
        return {key: by_variant[(key[0], key[1], key[3], key[4], key[5], key[6])][:, groups[key[2]]] for key in cand}


def main(scenario, params_path, steps, out_root, children=10):
    from smart_buildings import evaluate as ev
    from harness.fit_prior import _env
    from harness.run_parallel import resolve_policy
    scenario = Path(scenario).resolve()
    t0 = time.time()
    env, tracker, meta = _env(scenario)
    policy = resolve_policy("aether:mpc_off", Path(params_path).resolve())(meta, env.action_spec())
    policy.planner.sim = SimRollout(env, policy, children)
    policy.planner.sim.name = f"{scenario.parent.name}/{scenario.name}"
    compact = ev.run_policy(env, tracker, policy, steps)
    rc = tracker.reward_components
    summary = {
        "policy": "aether:mpc_off+simbound", "total_reward": float(np.sum(tracker.rewards)), "total_steps": len(tracker.rewards),
        "total_normalized_productivity_regret": float(sum(r["normalized_productivity_regret"] for r in rc)),
        "total_normalized_energy_cost": float(sum(r["normalized_energy_cost"] for r in rc)),
        "total_normalized_carbon_emission": float(sum(r["normalized_carbon_emission"] for r in rc)),
        "seconds_per_step": compact["seconds_per_step"], "wall_seconds": time.time() - t0, "params": str(params_path),
        "policy_stats": policy.stats(),
    }
    d = Path(out_root) / scenario.parent.name / scenario.name / "aether-mpc_off"
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    print(scenario.name, round(abs(summary["total_reward"]) / summary["total_steps"], 5), round(summary["wall_seconds"]), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], *(int(x) for x in sys.argv[5:6]))
