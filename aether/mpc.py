"""Layer B: enumerative short-horizon planner over the population prior (MPC-lite).

Every planning call, for the whole building:
  global settings  G  in {loop off, loop on at tw_min, loop on at tw_max}
  per AHU group    S  in {AHU off} x {SAT cool/idle} x {pressure floor/max}
  per zone         Z  in {idle, heat half, heat full, cool half, cool full}
Zones are conditionally independent given (G, S) under the per-zone model, so the best Z is chosen
per zone for every (G, S), then the best S per group given G, then the best G. Each candidate is
held constant over the horizon and scored with the competition reward reconstructed from the
simulator's own formulas; the first step of the best plan is applied and the search repeats.
Enumeration evaluates the bilinear airflow term exactly; weather is held at its current value.
"""

import numpy as np

from aether.actions import Command
from aether.planner import DAMPER_MIN, RECIRC_MAX, TW_IDLE
from aether.supervisor import STEPS_PER_H, steps_to_comfort
from aether.zone_model import TAU, ahu_share, discharge, flow_fraction, make_features, rollout

RHO_CP = 1.2 * 1006.0          # J/(m3 K), air
VAV_MAX_FLOW = 2.0             # m3/s per box (kit configs)
AHU_MAX_FLOW = 20.0            # m3/s per handler (kit configs)
FAN_EFF, COP = 0.9, 3.5
ELEC_PER_AHU, GAS_PER_AHU = 80000.0, 205000.0   # reward normalisation denominators, W per AHU
PRICE_E, PRICE_G = 0.18 / 3.6e6, 0.03 / 3.6e6   # $/J (16-20 c/kWh electricity; gas ~5-8x cheaper)
CARBON_E, CARBON_G = 0.10 / 3.6e6, 0.181 / 3.6e6  # kg/J
TANK_UA = 60.0                 # W/K tank loss to outdoor air (approximation)
REHEAT_EFF = 0.5               # share of (tw - tsa) the coil adds to the air (approximation)
ZONE_ACTIONS = ("idle", "heat_half", "heat", "cool_half", "cool")


def econ_recirc(t_return: float, tout: float, sat: float, flow: float, pressure: float) -> tuple[float, float]:
    """(recirculation, coil + extra exhaust-fan W) for an AHU that runs. Below the return air, outside air can replace
    coil work: mix so the mixed air lands on the supply setpoint (never below it), which leaves the supply air
    min(mixed, sat) unchanged, so zone dynamics are the same and only the plant bill changes. Kept only when cheaper
    than the coil at maximum recirculation, since the exhaust fan moves the extra outside air."""
    def watts(r):
        mixed = r * t_return + (1 - r) * tout
        return flow * RHO_CP * max(mixed - sat, 0.0) / COP + flow * (RECIRC_MAX - r) * pressure / FAN_EFF
    best = (RECIRC_MAX, watts(RECIRC_MAX))
    if tout < t_return:
        r = float(np.clip((sat - tout) / (t_return - tout), 0.0, RECIRC_MAX))
        if watts(r) < best[1]:
            best = (r, watts(r))
    return best


def comfort_regret(dev: np.ndarray) -> np.ndarray:
    """Productivity shortfall: linear to 5 K, quadratic beyond (the simulator's curve)."""
    return np.where(dev <= 5.0, dev / 5.0, 1.0 + ((dev - 5.0) / 5.0) ** 2)


def zone_inputs(params, n: int, t_return, tout: float, z_action: str, ahu_on: bool, sat: float, pump_on: bool, tw: float,
                pressure: float = 2000.0):
    """(damper, airflow fraction, reheat, supply air, mixed air) for n zones under one zone action and plant setting.
    t_return: return-air temperature, scalar or (n,); tsa and mixed broadcast with it."""
    d = np.full(n, DAMPER_MIN)
    v = np.zeros(n)
    if z_action == "heat_half":
        d[:], v[:] = params.heat_damper, 0.5
    elif z_action == "heat":
        d[:], v[:] = params.heat_damper, 1.0
    elif z_action == "heat_full":  # damper fully open on a saturated reheat coil: a large room needs the airflow, not the valve
        d[:], v[:] = 1.0, 1.0
    elif z_action == "cool_half":
        d[:] = 0.5
    elif z_action == "cool":
        d[:] = 1.0
    if not pump_on or not ahu_on:  # a reheat coil only heats air that flows through it
        v[:] = 0.0
    if not ahu_on:
        d[:] = DAMPER_MIN
    u = flow_fraction(d, pressure) * float(ahu_on)  # true airflow: the pressure floor gives 32% of design flow
    mixed = RECIRC_MAX * t_return + (1 - RECIRC_MAX) * tout
    tsa = np.minimum(mixed, sat) if ahu_on else mixed
    return d, u, v * float(pump_on), tsa, mixed


def _cost_parts(traj, lo, hi, occ, d, v, pressure, tsa, tw, K, band_margin, air: bool = False):
    """Summed occupied comfort regret over traj's steps, and the per-step normalised energy and carbon of the zone's own use."""
    dev = np.maximum(0.0, np.maximum(lo + band_margin - traj, traj - (hi - band_margin)))
    regret = (comfort_regret(dev) * occ).sum(0)
    frac = flow_fraction(d, pressure)
    if air:  # d is one AHU group under one action: the AHU's flow cap scales every box (ahu_share)
        frac = frac * ahu_share(frac, np.zeros(np.shape(frac)[-1], int))
    flow = frac * VAV_MAX_FLOW
    fan_w = flow * pressure / FAN_EFF
    if air:  # the coil heat the simulator computes: flow x rho cp x (discharge - supply air)
        reheat_w = flow * RHO_CP * np.maximum(discharge({"u": flow / VAV_MAX_FLOW, "v": v, "tsa": tsa, "tw": tw}) - tsa, 0.0)
    else:
        reheat_w = v * flow * RHO_CP * REHEAT_EFF * np.maximum(tw - tsa, 0.0)
    e_norm = (PRICE_E * fan_w + PRICE_G * reheat_w) / (PRICE_E * ELEC_PER_AHU + PRICE_G * GAS_PER_AHU) / K
    c_norm = (CARBON_E * fan_w + CARBON_G * reheat_w) / (CARBON_E * ELEC_PER_AHU + CARBON_G * GAS_PER_AHU) / K
    return regret, e_norm, c_norm


def zone_step_costs(traj, lo, hi, occ, d, v, pressure, tsa, tw, K: int, band_margin: float, n_occ_ref: float) -> np.ndarray:
    """(n,) normalised cost per zone SUMMED over traj's steps: 0.6 comfort share + 0.2 energy + 0.2 carbon of its own use."""
    regret, e_norm, c_norm = _cost_parts(traj, lo, hi, occ, d, v, pressure, tsa, tw, K, band_margin)
    return 0.6 * regret / max(n_occ_ref / 2.0, 1.0) + traj.shape[0] * (0.2 * e_norm + 0.2 * c_norm)


def mass_after(Tm0, T0, traj) -> np.ndarray:
    """Mass node after the trajectory: Tm <- Tm + (T - Tm)/TAU applied from T0 through traj[:-1] (as zone_model.rollout)."""
    Tm = np.array(Tm0, float)
    for T in (T0, *traj[:-1]):
        Tm = Tm + (T - Tm) / TAU
    return Tm


class Planner:
    def __init__(self, parser, params, meta, horizon: int = 18, every: int = 3, oracle=None, value=None):
        """value: a ZoneValue (aether/value.py); when set, each zone candidate also pays the learned cost-to-go at the
        horizon end (in per-step units, V / H)."""
        self.parser, self.params, self.meta, self.oracle, self.value = parser, params, meta, oracle, value
        self._t0 = 0  # absolute step of the current planning call (for the oracle)
        self.H, self.every = horizon, every
        self.t = 0
        self._last = None
        self.n_plans = 0
        self.sim = None  # bound experiment only (harness/sim_bound.py): callable(cand, H) -> trajectories from the true simulator

    # ------------------------------------------------------------------ schedule over the horizon
    def _bands(self, snap, lead=None, floor=None):
        """(lo, hi, scored) per future step (H, V). The comfort band applies while occupied and, before
        opening, from Layer A's optimal-start lead on (the 1.5-h horizon alone cannot see a long
        recovery). At other unoccupied steps only falling below Layer A's recoverable floor is scored."""
        V = len(snap.T)
        clo, chi = self.meta["comfort_temp_window"]
        occ_now = snap.comfort_now or snap.occ > 0
        k0 = np.zeros(V) if occ_now else steps_to_comfort(snap, self.meta) - (0 if lead is None else np.asarray(lead))
        j = np.arange(1, self.H + 1)[:, None]
        band = j >= np.broadcast_to(k0, (V,))[None]
        fl = np.full(V, -np.inf) if floor is None else np.asarray(floor, float)
        lo = np.where(band, clo, fl[None])
        hi = np.where(band, chi, np.inf)
        return lo, hi, band | np.isfinite(fl)[None]

    # ------------------------------------------------------------------ one candidate, all zones
    def _zone_inputs(self, snap, g, z_action, ahu_on, sat, pump_on, tw, pressure=2000.0):
        return zone_inputs(self.params, len(g), snap.T[g].mean(), snap.tout, z_action, ahu_on, sat, pump_on, tw, pressure)

    def _rollout(self, ident, items, snap):
        """Roll out many candidates in one batch. items: [(g, u, v, tsa, tw)]; returns [(H, len(g))] in order.
        The model is per zone, so stacking candidates along the zone axis gives each one exactly its own rollout."""
        T0, Tm0, fp = ident.state
        idx = np.concatenate([it[0] for it in items])
        u, v = np.concatenate([it[1] for it in items]), np.concatenate([it[2] for it in items])
        if self.params.reheat_air > 0:  # one candidate = one AHU group under one action: apply that AHU's flow cap
            u = u * np.concatenate([ahu_share(it[1], np.zeros(len(it[0]), int)) for it in items])
        tsa = np.concatenate([np.full(len(it[0]), it[3]) for it in items])
        tw = np.concatenate([np.full(len(it[0]), float(it[4])) for it in items])
        tout = [self.oracle.tout(self._t0 + 1 + j) if self.oracle is not None else snap.tout for j in range(self.H)]
        steps = []
        for j in range(self.H):
            f = make_features(u, v, tsa, 0.0, tout[j], snap.occ / self.parser.n_occ_ref, (snap.hour + j / STEPS_PER_H) % 24,
                              self.params.reheat_air > 0)
            f["tw"] = tw  # per candidate
            steps.append(f)
        prev = {k: (val[idx] if isinstance(val, np.ndarray) and val.shape == (len(ident.theta),) else val) for k, val in fp.items()}
        traj = rollout(ident.theta[idx], T0[idx], Tm0[idx], steps, prev)[1:]  # (H, sum n)
        return np.split(traj, np.cumsum([len(it[0]) for it in items])[:-1], axis=1)

    def _terminal(self, snap, ident, idx, traj):
        """(n,) learned cost-to-go at the horizon end for zones idx with trajectories traj (H, n), per-step units;
        0 where it is not finite, so a faulty sensor cannot hijack the argmin."""
        from aether.value import clock_ahead  # aether.value imports this module
        T0, Tm0, _ = ident.state
        occ_h, c_h = clock_ahead(snap, self.meta, self.H)
        tout_h = self.oracle.tout(self._t0 + self.H) if self.oracle is not None else snap.tout
        bias = ident.bias[idx] if ident.offset_alpha > 0 else np.zeros(len(idx))
        T_h = traj[-1]
        with np.errstate(invalid="ignore"):
            term = self.value.lookup(occ_h, c_h, tout_h, T_h, mass_after(Tm0[idx], T0[idx], traj) - T_h, bias) / self.H
        return np.where(np.isfinite(term), term, 0.0)

    def _zone_costs(self, traj, lo, hi, occ, d, v, pressure, tsa, tw, K, cool_w=None):
        """(n,) normalised cost per zone, per-step average over the horizon (zone_step_costs / H, same arithmetic order as before).
        cool_w: (n,) the zone's own share of the cooling-coil power (zone_cool), else the coil is charged only at group level."""
        p = self.params
        margin = p.band_margin if p.plan_margin < 0 else p.plan_margin
        regret, e_norm, c_norm = _cost_parts(traj, lo, hi, occ, d, v, pressure, tsa, tw, K, margin, p.reheat_air > 0)
        comfort = regret / self.H / max(self.parser.n_occ_ref / 2.0, 1.0) * p.comfort_w  # x * 1.0 is exact
        if cool_w is not None:
            return 0.6 * comfort + 0.2 * e_norm + 0.2 * c_norm + self._cool_cost(cool_w, K)
        return 0.6 * comfort + 0.2 * e_norm + 0.2 * c_norm

    @staticmethod
    def _cool_cost(cool_w, K):
        """Normalised energy + carbon share of cooling-coil electricity (W), as _plant_cost charges it."""
        e = PRICE_E * cool_w / (PRICE_E * ELEC_PER_AHU + PRICE_G * GAS_PER_AHU) / K
        c = CARBON_E * cool_w / (CARBON_E * ELEC_PER_AHU + CARBON_G * GAS_PER_AHU) / K
        return 0.2 * e + 0.2 * c

    def _plant_cost(self, flow_total, mixed, sat, ahu_on, pump_on, tw, snap, K, recirc=None):
        """recirc: (fraction, coil + extra exhaust W) from econ_recirc replaces the coil at maximum recirculation."""
        cool_w = (recirc[1] if recirc else flow_total * RHO_CP * max(mixed - sat, 0.0) / COP) if ahu_on else 0.0
        tank_w = TANK_UA * max(tw - snap.tout, 0.0) if pump_on else 0.0
        e = (PRICE_E * cool_w + PRICE_G * tank_w) / (PRICE_E * ELEC_PER_AHU + PRICE_G * GAS_PER_AHU) / K
        c = (CARBON_E * cool_w + CARBON_G * tank_w) / (CARBON_E * ELEC_PER_AHU + CARBON_G * GAS_PER_AHU) / K
        return 0.2 * e + 0.2 * c

    # ------------------------------------------------------------------ search
    def plan(self, snap, ident, lead=None, floor=None):
        p = self.params
        V, K = len(snap.T), len(self.parser.ahus)
        lo, hi, occ = self._bands(snap, lead, floor)
        globals_ = [(False, TW_IDLE), (True, p.tw_min), (True, p.tw_max)]
        # one AHU supply temperature serves all its zones: intermediate levels let cooling and reheat on one AHU balance
        sats = (p.sat_cool, p.sat_idle) if p.sat_mid < 1 else tuple(np.linspace(p.sat_cool, p.sat_idle, int(p.sat_mid) + 2))
        pressures = (200.0, p.p_max) if p.p_mid <= 0 else (200.0, p.p_mid, p.p_max)
        group_settings = [(False, p.sat_idle, 200.0)] + [(True, s, pr) for s in sats for pr in pressures]
        actions = ZONE_ACTIONS + (("heat_full",) if p.heat_full > 0 else ())

        def skip(za, pump_on, ahu_on):
            return (za.startswith("heat") and not pump_on) or (za.startswith("cool") and not ahu_on)

        # every candidate's inputs first, then one batched rollout (the per-candidate Python loop dominated runtime)
        cand = {}
        for pump_on, tw in globals_:
            for k, g in enumerate(self.parser.groups):
                for ahu_on, sat, pressure in group_settings if len(g) else ():
                    for za in actions:
                        if not skip(za, pump_on, ahu_on):
                            cand[pump_on, tw, k, ahu_on, sat, pressure, za] = self._zone_inputs(snap, g, za, ahu_on, sat, pump_on, tw, pressure)
        if self.sim is not None:
            trajs = self.sim(cand, self.H)
        else:
            trajs = dict(zip(cand, self._rollout(ident, [(self.parser.groups[key[2]], c[1], c[2], c[3], key[1]) for key, c in cand.items()], snap)))
        term = {}
        if self.value is not None:  # one batched lookup for every candidate
            sizes = [len(self.parser.groups[key[2]]) for key in trajs]
            idx = np.concatenate([self.parser.groups[key[2]] for key in trajs])
            t = self._terminal(snap, ident, idx, np.concatenate(list(trajs.values()), axis=1))
            term = dict(zip(trajs, np.split(t, np.cumsum(sizes)[:-1])))
        best = None
        n_plans = 0
        for pump_on, tw in globals_:
            total = 0.0
            choice = []
            for k, g in enumerate(self.parser.groups):
                if len(g) == 0:
                    choice.append(None)
                    continue
                gbest = None
                for ahu_on, sat, pressure in group_settings:
                    zc = np.full((len(actions), len(g)), np.inf)
                    inputs = []
                    for ai, za in enumerate(actions):
                        if skip(za, pump_on, ahu_on):
                            inputs.append(None)
                            continue
                        key = (pump_on, tw, k, ahu_on, sat, pressure, za)
                        d, u, v, tsa, mixed = cand[key]
                        traj = trajs[key]
                        zcw = u * VAV_MAX_FLOW * RHO_CP * np.maximum(mixed - tsa, 0.0) / COP if p.zone_cool > 0 else None
                        zc[ai] = self._zone_costs(traj, lo[:, g], hi[:, g], occ[:, g], d, v, pressure, tsa, tw, K, zcw)
                        if term:
                            zc[ai] = zc[ai] + term[key]
                        inputs.append((d, v, tsa, mixed, zcw))
                        n_plans += 1
                    ai_best = zc.argmin(0)
                    d = np.array([inputs[a][0][i] for i, a in enumerate(ai_best)])
                    v = np.array([inputs[a][1][i] for i, a in enumerate(ai_best)])
                    tsa_mixed = next(x for x in inputs if x is not None)
                    flow_total = min((flow_fraction(d, pressure) * VAV_MAX_FLOW).sum(), AHU_MAX_FLOW) if ahu_on else 0.0
                    rc = econ_recirc(snap.T[g].mean(), snap.tout, sat, flow_total, pressure) if p.plan_econ > 0 and ahu_on else None
                    cost = zc[ai_best, np.arange(len(g))].sum() + self._plant_cost(flow_total, tsa_mixed[3], sat, ahu_on, pump_on, tw, snap, K, rc)
                    if p.zone_cool > 0:  # the zones already paid their coil shares: keep the group total exact
                        cost -= self._cool_cost(sum(inputs[a][4][i] for i, a in enumerate(ai_best)), K)
                    if gbest is None or cost < gbest[0]:
                        gbest = (cost, ahu_on, sat, pressure, d, v, rc[0] if rc else RECIRC_MAX)
                total += gbest[0]
                choice.append(gbest)
            if best is None or total < best[0]:
                best = (total, pump_on, tw, choice)
        self.n_plans = n_plans
        _, pump_on, tw, choice = best
        cmd = Command.default(V, K)
        for k, g in enumerate(self.parser.groups):
            if choice[k] is None:
                continue
            _, ahu_on, sat, pressure, d, v, recirc = choice[k]
            cmd.ahu_run[k], cmd.sat[k], cmd.pressure[k] = ahu_on, sat, pressure if ahu_on else 200.0
            cmd.damper[g], cmd.reheat[g] = d, v
            cmd.recirc[k] = recirc
        cmd.pump_run = bool(pump_on and (cmd.reheat > 0).any())
        cmd.tw_sp = tw if cmd.pump_run else TW_IDLE
        return cmd, {"n_plans": n_plans, "cost": best[0]}

    def step(self, snap, ident, t: int | None = None, lead=None, floor=None):
        """Replan every `every` steps; otherwise repeat the last plan's first step.
        lead, floor: Layer A's optimal-start lead (steps) and recoverable setback floor (K) per zone."""
        if t is not None:
            self._t0 = t
        if self._last is None or self.t % self.every == 0:
            self._last = self.plan(snap, ident, lead, floor)
        self.t += 1
        cmd, info = self._last
        return cmd.copy(), info
