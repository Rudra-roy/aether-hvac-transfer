"""AetherPolicy: TransferPolicy-compatible composition of the controller units.

Arms (ablations):
  rules          fixed default lead, no model
  prior          model uses (optimal start, setback floor, gains) with theta frozen at the prior mean (control)
  passive        + online identification
  active         + comfort-free probing
  passive_guard  passive + comfort guard
  full           active + comfort guard
"""

import dataclasses
from pathlib import Path

import numpy as np

from aether.guard import apply
from aether.planner import TW_IDLE, plan
from aether.supervisor import OCCUPIED, holiday, lead_steps, modes, setback_floor, steps_to_comfort
from aether.actions import ActionWriter
from aether.identifier import Prior, ZoneIdentifier
from aether.inv import InvController
from aether.obs import AREA_REF, ObsParser, zone_areas
from aether.mpc import Planner
from aether.params import Params
from aether.probe import Prober
from aether.residual import residual_apply, residual_features, residual_forward
from aether.value import ZoneValue
from aether.zone_model import features

ARMS = {  # use_model, identify, probe, guard, plan
    "rules": (False, False, False, False, False),
    "prior": (True, False, False, False, False),
    "passive": (True, True, False, False, False),
    "active": (True, True, True, False, False),
    "passive_guard": (True, True, False, True, False),
    "full": (True, True, True, True, False),
    "mpc": (True, False, False, False, True),         # Layer B over the frozen prior, Layer A fallback
    "mpc_guard": (True, False, False, True, True),
    "mpc_off": (True, False, False, False, True),     # mpc + offset-free bias estimator
    "mpc_v": (True, False, False, False, True),       # mpc_off with the learned terminal value instead of Layer A's proxies
    "mpc_vp": (True, False, False, False, True),      # mpc_off with the learned terminal value and the proxies
    "mpc_r": (True, False, False, False, True),       # mpc_off + a learned per-zone residual on damper and reheat
    "inv": (False, False, False, False, False),       # one-step inversion controller (aether/inv.py), no planner, no prior
}
OFFSET_ARMS = {"mpc_off", "mpc_v", "mpc_vp", "mpc_r"}
VALUE_ARMS = {"mpc_v", "mpc_vp"}
OFFSET_ALPHA = 1 / 12  # 1-h EMA of one-step residuals
LEAD_EVERY, FLOOR_EVERY = 3, 12  # steps between optimal-start / setback-floor recomputations
GAIN_CLIP = (0.25, 4.0)


def load_prior(air: bool = False) -> Prior:
    """The fitted population prior shipped next to this file (the submission bundle embeds it instead).
    air: the prior fitted with the discharge-temperature airflow terms (Params.reheat_air)."""
    return Prior.load(Path(__file__).with_name("prior_reheat_air.json" if air else "prior.json"))


class AetherPolicy:
    name = "aether"

    def __init__(self, building_meta_data, action_spec, params: Params | None = None, arm: str = "full", prior: Prior | None = None,
                 oracle=None, residual: dict | None = None):
        """oracle: a WeatherOracle (weather ablation only, never in a submission) giving the true future outside temperature."""
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}; options {sorted(ARMS)}")
        self.meta, self.action_spec, self.arm = building_meta_data, action_spec, arm
        self.params = params or Params()
        self.use_model, self.identify, self.probe, self.guard, self.plan = ARMS[arm]
        self.parser = ObsParser(building_meta_data)
        self.writer = ActionWriter(building_meta_data, self.parser)
        V = len(self.parser.vavs)
        self.gains = np.ones((V, 2))
        self.lead = np.full(V, int(round(self.params.default_lead_min / 5)))
        self.floor = None
        self.ident = None
        if self.use_model:
            self.prior = prior or load_prior(self.params.reheat_air > 0)
            if self.params.act_scale != 1.0:  # scale the actuation gains (zone_model NAMES: a, b, r, b1, r1, ...)
                m0 = self.prior.m0.copy()
                m0[1:5] *= self.params.act_scale
                self.prior = Prior(m0, self.prior.P0, self.prior.R)
            if self.params.phys_scale != 1.0:  # every flux per unit capacity (a..s2), not the constant bias d
                m0 = self.prior.m0.copy()
                m0[:9] *= self.params.phys_scale
                self.prior = Prior(m0, self.prior.P0, self.prior.R)
            rho = 1 - 1 / (288 * self.params.act_adapt_days) if self.params.act_adapt_days > 0 else 0.0
            base = None
            if self.params.act_area > 0:  # per-zone actuation gains from the floor plan: a bigger room responds more slowly
                A = zone_areas(building_meta_data, self.parser.vavs)
                if A is not None:
                    cap = lambda a: 3800.0 * a + 21000.0 * np.sqrt(a)  # effective capacity vs area, the form of Wu's public controller (Apache-2.0)
                    s = np.clip((cap(AREA_REF) / cap(A)) ** self.params.act_area, 0.1, 3.0)
                    base = np.repeat(self.prior.m0[None], V, 0)
                    base[:, 1:5] *= s[:, None]
            self.ident = ZoneIdentifier(V, self.prior, q_rw=self.params.q_rw, frozen=not self.identify,
                                       offset_alpha=OFFSET_ALPHA if arm in OFFSET_ARMS else 0.0, act_rho=rho, base=base)
        self.prober = Prober(self.parser, self.params) if self.probe else None
        self.oracle = oracle
        value = ZoneValue.solve(self.prior.m0, self.params, building_meta_data, K=len(self.parser.ahus),
                                n_occ_ref=self.parser.n_occ_ref) if arm in VALUE_ARMS else None
        self.planner = Planner(self.parser, self.params, building_meta_data, oracle=oracle, value=value) if self.plan else None
        self.inv = InvController(building_meta_data, self.parser, self.params) if arm == "inv" else None
        self.residual = residual if arm == "mpc_r" else None  # weights (aether/residual.py)
        self.residual_features = None
        self.residual_noise = None  # training only: np.random.Generator for exploration; log of (features, pre-tanh action)
        self.residual_log = []
        self.t = 0
        self._hol_day, self._hol_seen = -1, False
        self._stats = {"guard_interventions": 0, "probe_steps": 0, "trusted_fraction_end": 0.0, "planner_calls": 0}

    def _model_uses(self, snap) -> None:
        trusted = self.ident.trusted
        if self.t % LEAD_EVERY == 0:
            self.lead = lead_steps(self.ident, snap, self.params, self.parser, self.meta, trusted)
            if self.params.lead_scale != 1.0:
                self.lead = np.round(self.lead * self.params.lead_scale).astype(int)
        if self.t % FLOOR_EVERY == 0:
            self.floor = setback_floor(self.ident, snap, self.params, self.parser, self.meta, trusted)
            if self.params.floor_shift:
                self.floor = self.floor - self.params.floor_shift
            if self.params.floor_abs:  # analysis: hold empty zones above an absolute floor
                self.floor = np.maximum(self.floor, self.params.floor_abs)
            m0 = self.prior.m0
            g = np.stack([self.ident.theta[:, 2] / m0[2], self.ident.theta[:, 1] / m0[1]], 1)
            self.gains = np.where(trusted[:, None], np.clip(np.nan_to_num(g, nan=1.0), *GAIN_CLIP), 1.0)
        self._stats["trusted_fraction_end"] = float(trusted.mean())
        if self.t % 288 == 0 and self.t > 0:  # daily: median-zone rolling 1-h hindcast error (learning curve)
            r = self.ident.rmse_1h
            self._stats.setdefault("id_rmse_by_day", []).append(float(np.median(r[np.isfinite(r)])) if np.isfinite(r).any() else None)

    def get_action(self, obs):
        snap = self.parser.parse(obs)
        if self.inv is not None:
            self.t += 1
            cmd = self.inv.step(snap)
            self._ahu_off_stat(snap, cmd)
            return self.writer.write(cmd)
        if self.params.holiday_hour > 0:
            if snap.dow != self._hol_day:
                self._hol_day, self._hol_seen = snap.dow, False
            self._hol_seen = self._hol_seen or snap.occ > 0
            if holiday(snap, self._hol_seen, self.params.holiday_hour):  # rest the building; anyone arriving ends it
                lo, hi = self.meta["eco_temp_window"]
                snap = dataclasses.replace(snap, comfort_now=False, comfort_soon=False, hsp=np.full_like(snap.hsp, lo), csp=np.full_like(snap.csp, hi))
                self._stats["holiday_steps"] = self._stats.get("holiday_steps", 0) + 1
        if self.use_model:
            self.ident.observe(snap, features(snap, self.parser, self.params.reheat_air > 0))
            self._model_uses(snap)
        zone_modes = modes(snap, self.lead, self.params, self.meta)
        cmd = plan(snap, zone_modes, self.gains, self.params, self.parser, self.meta, self.floor)
        if self.planner is not None:
            # Layer B: planner output for groups whose zones all pass the trust gate; Layer A elsewhere
            proxies = self.arm != "mpc_v"  # mpc_v: the value replaces the lead and floor inside the planner
            pcmd, _ = self.planner.step(snap, self.ident, t=self.t, lead=self.lead if proxies else None, floor=self.floor if proxies else None)
            if self.planner.t % self.planner.every == 1:
                self._stats["planner_calls"] += 1
            trusted = self.ident.trusted
            for k, g in enumerate(self.parser.groups):
                if len(g) and trusted[g].all():
                    cmd.damper[g], cmd.reheat[g] = pcmd.damper[g], pcmd.reheat[g]
                    cmd.ahu_run[k], cmd.sat[k], cmd.pressure[k], cmd.recirc[k] = pcmd.ahu_run[k], pcmd.sat[k], pcmd.pressure[k], pcmd.recirc[k]
            cmd.pump_run = bool((cmd.reheat > 0).any())
            cmd.tw_sp = pcmd.tw_sp if cmd.pump_run else TW_IDLE
        if self.residual is not None:
            x = residual_features(snap, self.ident, cmd, self.parser, self.meta)
            pre = residual_forward(self.residual, x)
            if self.residual_noise is not None:
                pre = pre + np.exp(self.residual["log_std"]) * self.residual_noise.standard_normal(pre.shape)
                self.residual_log.append((x.astype(np.float32), pre.astype(np.float32)))
            self.residual_features = x
            residual_apply(cmd, pre, self.params)
        if self.prober is not None:
            cmd, probed = self.prober.apply(cmd, snap, zone_modes, self.ident.trusted)
            self._stats["probe_steps"] += int(probed.any())
        if self.guard:
            occ = zone_modes == OCCUPIED
            soon = steps_to_comfort(snap, self.meta) <= self.params.guard_h
            clo, chi = self.meta["comfort_temp_window"]
            watch = occ | soon
            lo, hi = np.where(occ, snap.hsp, clo), np.where(occ, snap.csp, chi)
            cmd, hit = apply(cmd, snap, self.ident, self.params, self.parser, lo, hi, watch)
            self._stats["guard_interventions"] += int(hit.sum())
        if self.params.tank_cycle > 0 and cmd.pump_run and snap.tw >= cmd.tw_sp - self.params.tank_cycle:
            # BILLING-BUG EXPLOIT (starter-kit PR #2), experiment only, off by default: coils draw on the tank's actual
            # temperature but gas is billed from the setpoint, and a tank cooling toward a lower setpoint is credited.
            # Coast at the lowest setpoint while the tank is within tank_cycle K of the target; reheat for a step below.
            cmd.tw_sp = TW_IDLE
        if snap.occ == 0:  # statistic only: lowest zone temperature while the building is empty
            self._stats["min_T_unocc"] = min(self._stats.get("min_T_unocc", np.inf), float(np.nanmin(snap.T)))
        self._ahu_off_stat(snap, cmd)
        self.t += 1
        return self.writer.write(cmd)

    def _ahu_off_stat(self, snap, cmd):
        """Statistic only: zone steps with people in the building, and how many of them had the zone's AHU off."""
        if snap.occ > 0:
            off = ~np.asarray(cmd.ahu_run, dtype=bool)[self.parser.zone_ahu]
            self._stats["occ_zone_steps"] = self._stats.get("occ_zone_steps", 0) + int(off.size)
            self._stats["occ_zone_steps_ahu_off"] = self._stats.get("occ_zone_steps_ahu_off", 0) + int(off.sum())

    def stats(self) -> dict:
        return dict(self._stats)
