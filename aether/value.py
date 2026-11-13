"""Learned terminal value: per-zone backward induction on the frozen population prior.

A zone under the prior is a small MDP. State: zone temperature T, mass-node offset d = Tm - T, outside temperature
(held, persistence), the zone's offset bias, and a schedule clock (unoccupied: steps to the next comfort start;
occupied: steps to its end). Actions: the planner's zone actions at a nominal plant (table below). Cost: the planner's
own per-zone reward reconstruction (zone_step_costs) plus the zone's cooling-coil share. Value iteration on a grid with
multilinear interpolation (an averager, hence stable: Gordon 1995) gives V(s), the sum of future per-step costs.
This is model-based RL in the approximate-DP sense: value iteration on a learned model, not model-free RL.

Clock nodes step by MACRO steps. The phase boundaries are aliases, so the chain is finite:
unoccupied c=0 ("opening now") is occupied c=C_day, occupied c=0 ("closing now") is unoccupied K_MAX, and
unoccupied K_MAX (8 h or more before opening: nights, weekends) is a self-loop discounted by GAMMA_TAIL.
"""

import itertools

import numpy as np

from aether.identifier import BIAS_MAX
from aether.mpc import (CARBON_E, CARBON_G, COP, ELEC_PER_AHU, GAS_PER_AHU, PRICE_E, PRICE_G, RHO_CP, VAV_MAX_FLOW,
                        mass_after, zone_inputs, zone_step_costs)
from aether.planner import TW_IDLE
from aether.supervisor import STEPS_PER_H, steps_to_comfort
from aether.zone_model import flow_fraction, make_features, rollout

T_GRID = np.arange(275.0, 310.01, 0.5)
D_GRID = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
TOUT_GRID = 273.15 + np.arange(-35.0, 45.1, 10.0)
BIAS_GRID = np.linspace(-BIAS_MAX, BIAS_MAX, 5)
GRIDS = (TOUT_GRID, T_GRID, D_GRID, BIAS_GRID)
SHAPE = tuple(len(g) for g in GRIDS)
MACRO = 3          # steps per DP transition (15 min)
K_MAX = 96         # steps (8 h); further from opening than this is one discounted self-loop node
GAMMA_TAIL = 0.95  # per macro step at K_MAX
OCCN_OCCUPIED = 0.5  # nominal occupancy share while occupied
TOL, MAX_ITER = 1e-10, 3000  # K_MAX self-loop: max-abs change (tight enough that the Bellman residual is < 1e-9)


def c_day(meta) -> int:
    """Steps of comfort per day, floored to a multiple of MACRO."""
    return int(STEPS_PER_H * (float(meta["evening_start_hour"]) - float(meta["morning_start_hour"]))) // MACRO * MACRO


def clock_ahead(snap, meta, ahead: int) -> tuple[bool, int]:
    """(occupied, steps remaining in that phase) `ahead` steps from now. Unoccupied clamps to K_MAX; a comfort end
    inside `ahead` rolls into unoccupied K_MAX."""
    if snap.comfort_now:
        rem = max(int(round(STEPS_PER_H * (float(meta["evening_start_hour"]) - snap.hour))), 0)
        return (True, rem - ahead) if ahead < rem else (False, K_MAX)
    s = steps_to_comfort(snap, meta)
    if s > ahead:
        return False, min(s - ahead, K_MAX)
    return True, max(c_day(meta) - (ahead - s), 0)


def clock_node(occupied: bool, c: int, c_day_: int) -> tuple[bool, int]:
    """Canonical (occupied, clock) node: clock floored to MACRO, unoccupied clamped to K_MAX, and the phase boundaries
    aliased (unoccupied c=0 is occupied C_day; occupied c=0 is unoccupied K_MAX)."""
    k = c // MACRO
    if occupied:
        k = min(k, c_day_ // MACRO)
        return (False, K_MAX) if k == 0 else (True, k * MACRO)
    k = min(k, K_MAX // MACRO)
    return (True, c_day_) if k == 0 else (False, k * MACRO)


def zone_cool_cost(damper, pressure, mixed, sat, K: int):
    """Per-step normalised cost (0.2 energy + 0.2 carbon) of a zone's own share of its AHU's cooling coil."""
    cool_w = flow_fraction(damper, pressure) * VAV_MAX_FLOW * RHO_CP * np.maximum(mixed - sat, 0.0) / COP
    e = PRICE_E * cool_w / (PRICE_E * ELEC_PER_AHU + PRICE_G * GAS_PER_AHU) / K
    cc = CARBON_E * cool_w / (CARBON_E * ELEC_PER_AHU + CARBON_G * GAS_PER_AHU) / K
    return 0.2 * e + 0.2 * cc


def _corners(tout, T, d, bias):
    """Flat table indices (16, n) and multilinear weights (16, n); every axis clamps to its grid."""
    idx, frac = [], []
    for g, x in zip(GRIDS, (tout, T, d, bias)):
        x = np.clip(np.asarray(x, float), g[0], g[-1])
        i = np.clip(np.searchsorted(g, x, "right") - 1, 0, len(g) - 2)
        idx.append(i)
        frac.append((x - g[i]) / (g[i + 1] - g[i]))
    I, W = [], []
    for corner in itertools.product((0, 1), repeat=4):
        I.append(np.ravel_multi_index([i + c for i, c in zip(idx, corner)], SHAPE))
        W.append(np.prod([f if c else 1.0 - f for f, c in zip(frac, corner)], axis=0))
    return np.array(I), np.array(W)


def _interp(table, corners):
    I, W = corners
    return (table.ravel()[I] * W).sum(0)


class ZoneValue:
    def _actions(self):
        p = self.params
        return [("idle", False, p.sat_idle, 200.0, False, TW_IDLE),           # off
                ("idle", True, p.sat_idle, 200.0, False, TW_IDLE),            # vent
                ("heat_half", True, p.sat_idle, 200.0, True, p.tw_max),
                ("heat", True, p.sat_idle, 200.0, True, p.tw_max),
                ("cool_half", True, p.sat_cool, p.p_max, False, TW_IDLE),
                ("cool", True, p.sat_cool, p.p_max, False, TW_IDLE)]

    def _node(self, occupied: bool, c: int):
        return clock_node(occupied, c, self.c_day)

    def _table(self, occupied, c):
        return self.occ[c // MACRO] if occupied else self.unocc[c // MACRO]

    def _step(self, action, occupied, c, tout, T, d, bias):
        """One macro step from node (occupied, c) under one action: (cost (n,), T_next, d_next)."""
        za, ahu_on, sat, pressure, pump_on, tw = action
        n = len(T)
        hour0 = (float(self.meta["evening_start_hour" if occupied else "morning_start_hour"]) - c / STEPS_PER_H) % 24
        dmp, u, v, tsa, mixed = zone_inputs(self.params, n, T, tout, za, ahu_on, sat, pump_on, tw, pressure)
        steps = []
        for j in range(MACRO):
            f = make_features(u, v, tsa, tw, 0.0, OCCN_OCCUPIED if occupied else 0.0, (hour0 + j / STEPS_PER_H) % 24)
            f["tout"] = tout  # per state
            steps.append(f)
        th = np.tile(self.theta, (n, 1))
        th[:, -1] += bias
        traj = rollout(th, T, T + d, steps, steps[0])[1:]  # previous-step inputs = the held action
        Tm = mass_after(T + d, T, traj)
        lo, hi = self.meta["comfort_temp_window"] if occupied else (-np.inf, np.inf)
        cost = zone_step_costs(traj, lo, hi, np.full(traj.shape, occupied), dmp, v, pressure, tsa, tw, self.K,
                               self.params.band_margin, self.n_occ_ref)
        if ahu_on:  # the zone's own cooling-coil share; tank loss is building-level and omitted
            cost = cost + MACRO * zone_cool_cost(dmp, pressure, mixed, sat, self.K)
        return cost, traj[-1], Tm - traj[-1]

    def q(self, occupied, c, tout, T, d, bias) -> np.ndarray:
        """(6, n): per-action macro-step cost + value of the successor (discounted only on the K_MAX self-loop)."""
        occupied, c = self._node(occupied, c)
        tout, T, d, bias = (np.asarray(x, float) for x in (tout, T, d, bias))
        nxt, disc = ((False, K_MAX), GAMMA_TAIL) if (not occupied and c == K_MAX) else ((occupied, c - MACRO), 1.0)
        nxt = self._node(*nxt)
        out = []
        for a in self._actions():
            cost, Tn, dn = self._step(a, occupied, c, tout, T, d, bias)
            out.append(cost + disc * _interp(self._table(*nxt), _corners(tout, Tn, dn, bias)))
        return np.array(out)

    def lookup(self, occupied, c, tout, T, d, bias) -> np.ndarray:
        """(n,) value at the nearest lower clock node, multilinear in (tout, T, d, bias), every axis clamped."""
        T = np.asarray(T, float)
        b = np.broadcast_to
        return _interp(self._table(*self._node(occupied, c)), _corners(b(tout, T.shape), T, b(d, T.shape), b(bias, T.shape)))

    @classmethod
    def solve(cls, theta, params, meta, K: int, n_occ_ref: float) -> "ZoneValue":
        self = cls()
        self.theta, self.params, self.meta, self.K, self.n_occ_ref = np.asarray(theta, float), params, meta, K, float(n_occ_ref)
        self.c_day = c_day(meta)
        self.unocc = np.zeros((K_MAX // MACRO + 1, *SHAPE))
        self.occ = np.zeros((self.c_day // MACRO + 1, *SHAPE))
        tout, T, d, b = (x.ravel() for x in np.meshgrid(*GRIDS, indexing="ij"))
        # 1. far from opening: discounted self-loop; transitions and costs do not depend on V, so precompute them
        trans = [self._step(a, False, K_MAX, tout, T, d, b) for a in self._actions()]
        trans = [(cost, _corners(tout, Tn, dn, b)) for cost, Tn, dn in trans]
        V = np.zeros(len(T))
        for _ in range(MAX_ITER):
            Vn = np.min([cost + GAMMA_TAIL * _interp(V, cr) for cost, cr in trans], axis=0)
            done = np.abs(Vn - V).max() < TOL
            V = Vn
            if done:
                break
        self.unocc[-1] = V.reshape(SHAPE)
        # 2. the comfort day, from its end backwards; 3. the hours before opening
        self.occ[0] = self.unocc[-1]
        for k in range(1, len(self.occ)):
            self.occ[k] = self.q(True, k * MACRO, tout, T, d, b).min(0).reshape(SHAPE)
        self.unocc[0] = self.occ[-1]
        for k in range(1, len(self.unocc) - 1):
            self.unocc[k] = self.q(False, k * MACRO, tout, T, d, b).min(0).reshape(SHAPE)
        return self
