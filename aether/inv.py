"""Arm inv: a one-step inversion controller (numpy only).

Every five minutes each zone's airflow is solved from an implicit one-step model so that the zone lands on the
nearer edge of its band, the AHU static pressure is set so the neediest damper is fully open, and the AHU is left
off whenever the building's inertia keeps its zones inside the band for one step. A per-zone bias is learned
online from the one-step residual. The mechanisms follow the public Apache-2.0 controller of Daniel Wu as
extended by MorS1337 (sb26-stop-heating, CtrlW5): floor-plan response gain, implicit model, flow inversion,
matched pressure, AHU off by inertia, standby band when empty, hot water 310 K unless heating recently.
Reimplemented here on our observation parser, command writer and metadata geometry; no learned residual ensemble.
"""

import numpy as np
from collections import deque

from aether.actions import Command
from aether.obs import zone_geometry

RHO_CP = 1.2 * 1006.0                       # J/(m3 K)
VAV_MAX_FLOW = 2.0                          # m3/s per box (every kit config)
VALVE = 4180.0 * 1000.0 * 0.108 / 3600.0    # W/K per sqrt(bar), the kit's reheat valve
P_DESIGN, P_MIN, D_MIN, RECIRC_MAX, T_DIS_MAX = 2000.0, 200.0, 0.1, 0.7, 308.15
FAN_EFF, COP = 0.9, 3.5
SAT_TARGETS = np.array([285.0, 287.0, 289.0, 291.0, 293.0, 295.0, 297.0, 299.0, 302.0, 305.0])
RESIDUAL_RATE = 0.35    # EMA rate of the per-zone one-step residual
SLOW_TAU = 48.0         # steps, slow (mass) temperature estimate


class InvController:
    def __init__(self, meta, parser, params):
        self.parser, self.p = parser, params
        V = len(parser.vavs)
        geo = zone_geometry(meta, parser.vavs)
        self.area, ua = (np.full(V, 55.0), np.full(V, 18.0)) if geo is None else (np.maximum(geo[0], 1.0), np.maximum(geo[1] * 0.2 * 3.0 * 1.3, 0.5))
        g0 = np.clip(300000.0 / (3800.0 * self.area + 21000.0 * np.sqrt(self.area)), 0.08, 2.5)
        rs = params.inv_response
        # drift per step = theta[:, :5] . [1, (tout - T)/20, (slow - T)/5, (group - T)/5, occupied] + residual; theta[:, 5] = K per kW
        self.theta = np.column_stack((g0 * self.area * 2e-3 * rs, g0 * ua * 20e-3 * rs, np.full(V, 0.18), np.full(V, 0.08),
                                      g0 * self.area * 8e-3 * rs, g0 * rs))
        self.comfort, self.eco = tuple(map(float, meta["comfort_temp_window"])), tuple(map(float, meta["eco_temp_window"]))
        self.start = float(meta["morning_start_hour"])
        self.residual, self.slow, self.prev = np.zeros(V), None, None
        self.hist = deque(maxlen=3)
        self.t, self.last_heat = 0, -1000
        self.empty_steps, self.released = 0, False
        self.tw_state = params.inv_tw
        # occupant tracker: per-zone head count estimated from the building count and each zone's one-step innovation
        self.count, self.seen, self.cap, self.max_occ = np.zeros(V, int), np.zeros(V, bool), 2, 0.0
        self.inn, self.prev_inn, self.prev_occ, self.pending = None, None, None, 0

    # ------------------------------------------------------------------ model
    def _group_mean(self, T):
        out = np.zeros(len(T))
        for g in self.parser.groups:
            if len(g):
                out[g] = np.average(T[g], weights=self.area[g])
        return out

    def _features(self, T, tout, mode, slow):
        return np.column_stack((np.ones(len(T)), np.clip((tout - T) / 20.0, -5, 5), np.clip((slow - T) / 5.0, -5, 5),
                                np.clip((self._group_mean(T) - T) / 5.0, -5, 5), np.full(len(T), float(mode))))

    def _drift(self, T, tout, mode):
        return np.clip(np.einsum("ij,ij->i", self.theta[:, :5], self._features(T, tout, mode, self.slow)) + self.residual, -5.0, 5.0)

    def _learn(self, observed):
        if self.prev is None or len(self.hist) < 2:
            return
        a = self.hist[-2]  # the action that produced this observation (sensors are one step old)
        power = (a["cond"] * (a["supply"] - observed) + a["reheat"]) / 1000.0
        x = np.column_stack((self._features(self.prev, a["tout"], a["mode"], a["slow"]), power))
        remaining = np.clip(observed - self.prev, -15, 15) - np.einsum("ij,ij->i", self.theta, x)
        self.residual = np.clip((1 - RESIDUAL_RATE) * self.residual + RESIDUAL_RATE * remaining, -5.0, 5.0)
        self.inn = remaining

    # ------------------------------------------------------------------ occupants
    def _track(self, snap):
        """A change of the observed head count shows one step later as a step of about 70 W in the one-step innovation of
        the zones concerned. Departures go to the zones with the largest drop among those holding someone, arrivals to
        the largest rise among those with room. The count resets each night."""
        occ = float(snap.occ)
        if snap.hour < 4.0:
            if self.max_occ > 0:
                self.cap = max(1, int(round(self.max_occ / len(self.count))))
            self.count[:], self.seen[:], self.pending = 0, False, 0
        elif self.inn is not None and self.prev_inn is not None and self.pending:
            sig = (self.inn - self.prev_inn) / (0.07 * np.maximum(self.theta[:, 5], 1e-6))
            sig = np.nan_to_num(sig)
            if self.pending < 0:
                cand = np.flatnonzero(self.count > 0)
                order = cand[np.argsort(sig[cand])]
                k = -self.pending
                pick = order[:k]
                if self.p.inv_occ_gap > 0 and len(order) > k:  # assign only a clear drop, else leave every count (safe side)
                    pick = pick[(sig[pick] < -0.5) & (sig[order[k]] - sig[pick] > self.p.inv_occ_gap)]
                self.count[pick] -= 1
            else:
                cand = np.flatnonzero(self.count < self.cap)
                pick = cand[np.argsort(-sig[cand])[:self.pending]]
                self.count[pick] += 1
                self.seen[pick] = True
        self.pending = int(round(occ - self.prev_occ)) if self.prev_occ is not None and snap.hour >= 4.0 else 0
        self.prev_occ, self.prev_inn = occ, (None if self.inn is None else self.inn.copy())
        self.max_occ = max(self.max_occ, occ)

    # ------------------------------------------------------------------ band
    def _band(self, snap, T):
        p, now, soon = self.p, snap.comfort_now, snap.comfort_soon
        V = len(T)
        comfort_day = snap.dow < 5
        remaining = (self.start - snap.hour) % 24.0
        if now:
            low, high = np.maximum(snap.hsp, self.comfort[0]), np.minimum(snap.csp, self.comfort[1])
            if p.inv_release > 0:  # the building emptied after noon: nobody left to be uncomfortable
                self.empty_steps = 0 if snap.occ > 0 else self.empty_steps + 1
                self.released = (self.released or (self.empty_steps >= 3 and snap.hour >= 12.0)) and snap.occ == 0
            margin = p.inv_margin
        else:
            self.empty_steps, self.released = 0, False
            margin = 0.0
            if comfort_day and (soon or remaining <= p.inv_preheat_h):
                prog = float(np.clip(1.0 - remaining / max(p.inv_preheat_h, 0.1), 0.0, 1.0))
                low = np.full(V, self.eco[0] + prog * (self.comfort[0] - self.eco[0]))
                high = np.full(V, self.eco[1] + prog * (self.comfort[1] - self.eco[1]))
                margin = p.inv_margin
            else:
                low, high = np.full(V, p.inv_standby), np.full(V, 307.0)
        if self.released:
            low, high, margin = np.full(V, p.inv_standby), np.full(V, 307.0), 0.0
        low, high = low + margin, high - margin
        low = np.minimum(low, T + np.maximum((low - T) / p.inv_recovery, 0.0))
        high = np.maximum(high, T - np.maximum((T - high) / p.inv_recovery, 0.0))
        return low, high

    # ------------------------------------------------------------------ one AHU
    def _choose(self, g, T, free, low, high, tout, water):
        t, d, lo, hi = T[g], free[g], low[g], high[g]
        resp = self.theta[g, 5]
        inside = float(np.average(t, weights=self.area[g]))
        targets = np.concatenate((SAT_TARGETS, [float(np.clip(np.min(hi) - 1.5, 285.0, 305.0))]))
        sp = np.repeat(targets, 2)
        recirc = np.full_like(sp, RECIRC_MAX)
        if tout < inside - 0.5:  # economizer candidates: just enough outside air to reach the target
            recirc[1::2] = np.clip((targets - tout) / max(inside - tout, 1e-6), 0.0, RECIRC_MAX)
        mixed = recirc * inside + (1.0 - recirc) * tout
        supply = np.minimum(mixed, sp)
        sa = supply[:, None]
        cool_need = np.maximum(t + d - hi, 0.0) * 1000.0 / resp          # kW-steps to remove
        heat_need = np.maximum(lo - t - d, 0.0) * 1000.0 / resp
        cooling_flow = cool_need[None] / (RHO_CP * np.maximum(hi[None] - sa, 0.05))
        heating_flow = heat_need[None] / (RHO_CP * np.maximum(T_DIS_MAX - lo[None], 0.5))
        max_heat_flow = VALVE * np.sqrt(20.0) * np.maximum(water - sa, 0.0) / (RHO_CP * np.maximum(T_DIS_MAX - sa, 0.1))
        needed = np.maximum(cooling_flow, np.minimum(heating_flow, max_heat_flow))
        frac = np.max(needed / VAV_MAX_FLOW, axis=1)
        pressure = np.clip(P_DESIGN * frac ** 2, P_MIN, P_DESIGN)
        pr = np.sqrt(pressure / P_DESIGN)
        dampers = np.clip(needed / (VAV_MAX_FLOW * pr[:, None]), D_MIN, 1.0)
        flow = dampers * VAV_MAX_FLOW * pr[:, None]
        cond = flow * RHO_CP
        wanted = np.maximum((lo[None] - t[None] - d[None]) * 1000.0 / resp[None] + cond * (lo[None] - sa), 0.0)
        limit = np.minimum(cond * np.maximum(T_DIS_MAX - sa, 0.0), VALVE * np.sqrt(20.0) * np.maximum(water - sa, 0.0))
        reheat = np.minimum(wanted, limit)
        predicted = (t[None] + d[None] + resp[None] * (cond * sa + reheat) / 1000.0) / (1.0 + resp[None] * cond / 1000.0)
        viol = np.maximum(lo[None] - predicted, 0.0) + np.maximum(predicted - hi[None], 0.0)
        total_flow = flow.sum(1)
        fan = total_flow * pressure * (1.0 + self.p.inv_exhaust * (1.0 - recirc)) / FAN_EFF  # intake fan (+ exhaust fan when inv_exhaust)
        chiller = total_flow * RHO_CP * np.maximum(mixed - supply, 0.0) / COP
        obj = fan + chiller + self.p.inv_gas_ratio * reheat.sum(1) + self.p.inv_comfort_cost * np.sum(viol + 2.0 * viol ** 2, axis=1)
        off_viol = np.maximum(lo - t - d, 0.0) + np.maximum(t + d - hi, 0.0)
        off_obj = self.p.inv_comfort_cost * np.sum(off_viol + 2.0 * off_viol ** 2)
        b = int(np.argmin(obj))
        if off_obj <= obj[b]:
            return dict(run=False, pressure=0.0, sat=305.0, recirc=RECIRC_MAX, damper=np.full(len(g), D_MIN),
                        cond=np.zeros(len(g)), supply=np.full(len(g), mixed[b]), reheat=np.zeros(len(g)))
        return dict(run=True, pressure=float(pressure[b]), sat=float(sp[b]), recirc=float(recirc[b]), damper=dampers[b],
                    cond=cond[b], supply=np.full(len(g), supply[b]), reheat=reheat[b])

    # ------------------------------------------------------------------ step
    def step(self, snap) -> Command:
        observed = snap.T
        V, K = len(observed), len(self.parser.ahus)
        if self.slow is None:
            self.slow = observed.copy()
        self._learn(observed)
        T = observed.copy()
        if self.hist:  # the sensor lags one step: predict where the zone is now under the last command
            a = self.hist[-1]
            gain = self.theta[:, 5] / 1000.0
            T = (observed + self._drift(observed, a["tout"], a["mode"]) + gain * (a["cond"] * a["supply"] + a["reheat"])) / (1.0 + gain * a["cond"])
            T = np.clip(T, observed - 8.0, observed + 8.0)
        if self.p.inv_occ_track > 0:
            self._track(snap)
        low, high = self._band(snap, T)
        if self.p.inv_occ_track > 0 and snap.comfort_now and snap.hour >= self.p.inv_occ_hour:
            gone = self.seen & (self.count == 0)  # every occupant of the zone has left: nobody to be uncomfortable
            if self.p.inv_occ_relax > 0:  # widen the band of an empty zone by a bounded amount (caps the cost of a wrong call)
                low, high = np.where(gone, low - self.p.inv_occ_relax, low), np.where(gone, high + self.p.inv_occ_relax, high)
            else:
                low, high = np.where(gone, self.p.inv_standby, low), np.where(gone, 307.0, high)
        free = self._drift(T, snap.tout, snap.comfort_now)
        if np.any(T + free < low):
            self.last_heat = self.t
        cmd = Command.default(V, K)
        cond, supply, reheat = np.zeros(V), np.full(V, snap.tout), np.zeros(V)
        for k, g in enumerate(self.parser.groups):
            if not len(g):
                continue
            c = self._choose(g, T, free, low, high, snap.tout, snap.tw)
            cmd.ahu_run[k], cmd.sat[k], cmd.pressure[k], cmd.recirc[k] = c["run"], c["sat"], c["pressure"], c["recirc"]
            cmd.damper[g], cond[g], supply[g], reheat[g] = c["damper"], c["cond"], c["supply"], c["reheat"]
        if np.any(reheat > 10.0):
            self.last_heat = self.t
        need = reheat / (VALVE * np.maximum(snap.tw - supply, 1.0))       # sqrt(bar) each valve needs fully open
        if self.p.inv_tw_reset > 0:  # demand-based hot-water reset: raise while the loop is at capacity, trim otherwise
            at_cap = float(np.max(need)) >= np.sqrt(20.0) * 0.98
            self.tw_state = float(np.clip(self.tw_state + (self.p.inv_tw_reset if at_cap else -0.5 * self.p.inv_tw_reset), self.p.inv_tw_min, 350.0))
        cmd.tw_sp = (self.tw_state if self.p.inv_tw_reset > 0 else self.p.inv_tw) if self.t - self.last_heat < 24 else 310.0
        dp = float(np.clip(np.max(need) ** 2 * 1.05, 0.0, 20.0))
        valves = np.clip(need / np.sqrt(dp), 0.0, 1.0) if dp > 1e-8 else np.zeros(V)
        cmd.reheat, cmd.pump_run, cmd.dp_sp = valves, dp > 1e-8, dp
        actual = np.minimum(VALVE * np.sqrt(dp) * valves * np.maximum(snap.tw - supply, 0.0), cond * np.maximum(T_DIS_MAX - supply, 0.0))
        self.hist.append(dict(cond=cond, supply=supply, reheat=actual, tout=snap.tout, mode=snap.comfort_now, slow=self.slow.copy()))
        self.slow += (T - self.slow) / SLOW_TAU
        self.prev = observed.copy()
        self.t += 1
        return cmd
