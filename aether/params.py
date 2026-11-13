"""Tunable scalars of the controller, with the bounds CMA-ES searches over."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np


@dataclass
class Params:
    band_margin: float = 0.5        # K inside the comfort band edge (occupied / pre-conditioning)
    kp_heat: float = 0.8            # reheat fraction per K of heating error
    kp_cool: float = 0.8            # damper fraction per K of cooling error
    heat_damper: float = 0.4        # heating airflow gain when mixed air is >= 16 K below the zone (0.9 when mild)
    sat_cool: float = 286.0         # K, AHU supply setpoint when a served zone needs cooling
    sat_idle: float = 291.0         # K, otherwise
    p_max: float = 1200.0           # Pa, static pressure at full damper demand
    tw_max: float = 350.0           # K, hot-water supply at T_out <= -20 C
    tw_min: float = 320.0           # K, hot-water supply at T_out >= 15 C
    default_lead_min: float = 60.0  # pre-conditioning lead without a trusted model
    max_lead_min: float = 180.0
    setback_margin: float = 0.5     # K inside the eco band
    econ_margin: float = 1.0        # K outside-air advantage before economising
    guard_h: float = 12.0           # steps
    guard_margin: float = 0.2       # K
    probe_budget_ahu_h: float = 1.0
    q_rw: float = 1e-6              # chosen on pool_train (results/idcheck/REPORT_pool_train_q*.md)
    plan_margin: float = -1.0       # K inset of the band in the PLANNER's comfort cost; < 0: use band_margin (Layer A's)
    comfort_w: float = 1.0          # weight on the planner's comfort cost relative to its energy and carbon costs
    floor_shift: float = 0.0        # K subtracted from the recoverable setback floor (deeper night setback)
    floor_abs: float = 0.0          # K, absolute minimum of the setback floor (0 = off; 278.15 = a 5 C freeze limit)
    lead_scale: float = 1.0         # multiplies the optimal-start lead (earlier recovery)
    sat_mid: float = 0.0            # intermediate AHU supply temperatures the planner may pick between sat_cool and sat_idle
    act_scale: float = 1.0          # multiplies the prior's actuation gains (airflow b, b1; reheat r, r1): slower-responding rooms
    act_adapt_days: float = 0.0     # > 0: estimate a per-zone actuation scale online (forgetting time in days), starting at act_scale
    # arm inv (aether/inv.py): one-step inversion controller
    inv_margin: float = 0.12        # K inside the comfort band the inversion aims for while occupied or preconditioning
    inv_preheat_h: float = 2.0      # h before the comfort window over which the band ramps from standby to comfort
    inv_standby: float = 262.0      # K, lowest allowed zone temperature while the building is empty (262 = never binds)
    inv_tw: float = 336.32          # K, hot-water target while any zone heated in the last 2 h (310 K otherwise)
    inv_comfort_cost: float = 65000.0  # W-equivalent per K of predicted one-step band violation in the AHU objective
    inv_gas_ratio: float = 0.5      # score weight of one W of gas relative to one W of electricity (0.2 energy + 0.2 carbon)
    inv_response: float = 0.75      # scale on the floor-plan response gain 300000 / (3800 A + 21000 sqrt A)
    inv_release: float = 1.0        # > 0: drop to standby when the building is empty after 12:00 during comfort mode
    inv_tw_reset: float = 0.0       # K per step, > 0: hot-water target rises while the water loop is at capacity (dp cap, a valve fully open) and trims at half that rate otherwise, within [inv_tw_min, 350]
    inv_tw_min: float = 326.0       # K, floor of the hot-water reset
    inv_exhaust: float = 0.0        # 1: the AHU objective also bills the exhaust fan, flow x (1 - recirc) x pressure, as the simulator does
    inv_recovery: float = 1.5       # steps over which a band move is requested (never the whole recovery in one step)
    inv_occ_track: float = 0.0     # > 0: track per-zone occupants from the head count and one-step innovations; zones whose occupants have all left get the standby band
    inv_occ_hour: float = 13.0     # h, the tracker's empty zones are released only from this hour on (departures)
    inv_occ_gap: float = 0.0      # > 0: a departure is assigned only if its zone's drop beats the next zone's by this many occupant units
    inv_occ_relax: float = 0.0    # K, > 0: an empty zone's band is widened by this much instead of dropping to standby
    heat_full: float = 0.0          # > 0: adds the zone action heat_full (damper 1, reheat 1) to the planner's five levels
    p_mid: float = 0.0              # Pa, > 0: a third static-pressure candidate between the 200 Pa floor and p_max
    act_area: float = 0.0           # > 0: per-zone actuation gains scaled by (C(109 m2) / C(zone area))**act_area from the metadata zone_map
    phys_scale: float = 1.0         # multiplies every physical coefficient of the prior (a..s2; not the bias d): bigger thermal capacity
    holiday_hour: float = 0.0       # > 0: a weekday comfort period with nobody in by this hour is a holiday; rest the building
    zone_cool: float = 0.0          # > 0: each zone's planner cost includes its own share of the cooling-coil energy
    tank_cycle: float = 0.0         # > 0: EXPLOIT of the tank billing bug (PR #2), hysteresis K; experiment only, never the honest arm
    reheat_air: float = 0.0         # > 0: reheat reaches the zone with the air (discharge temperature); uses aether/prior_reheat_air.json
    plan_econ: float = 0.0        # > 0: the planner economises (outside air instead of the coil when it is cheaper; same supply air)

    @staticmethod
    def from_json(path) -> "Params":
        d = json.loads(Path(path).read_text())
        return Params(**{k: v for k, v in d.items() if k in {f.name for f in fields(Params)}})

    def to_json(self, path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @staticmethod
    def from_unit(x: np.ndarray, names: list[str], base: "Params | None" = None) -> "Params":
        """CMA-ES coordinates in [0,1] -> Params (log scale for LOG_SCALE names)."""
        d = asdict(base or Params())
        for xi, n in zip(np.clip(x, 0, 1), names):
            lo, hi = BOUNDS[n]
            d[n] = math.exp(math.log(lo) + xi * (math.log(hi) - math.log(lo))) if n in LOG_SCALE else lo + xi * (hi - lo)
        return Params(**d)

    def to_unit(self, names: list[str]) -> np.ndarray:
        out = []
        for n in names:
            lo, hi, v = *BOUNDS[n], getattr(self, n)
            out.append((math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) if n in LOG_SCALE else (v - lo) / (hi - lo))
        return np.clip(np.array(out), 0, 1)


BOUNDS: dict[str, tuple[float, float]] = {
    "band_margin": (0.1, 1.5), "kp_heat": (0.2, 3.0), "kp_cool": (0.2, 3.0), "heat_damper": (0.0, 0.9), "sat_cool": (285.0, 290.0),
    "sat_idle": (287.0, 295.0), "p_max": (400.0, 2000.0), "tw_max": (330.0, 350.0), "tw_min": (310.0, 335.0),
    "default_lead_min": (0.0, 180.0), "max_lead_min": (60.0, 240.0), "setback_margin": (0.0, 2.0),
    "econ_margin": (0.0, 3.0), "guard_h": (6.0, 24.0), "guard_margin": (0.0, 1.0),
    "probe_budget_ahu_h": (0.0, 3.0), "q_rw": (1e-6, 1e-3), "plan_margin": (0.0, 1.0), "comfort_w": (0.2, 1.5),
    "floor_shift": (0.0, 10.0), "lead_scale": (0.5, 3.0), "sat_mid": (0.0, 4.0),
}
LOG_SCALE = {"q_rw"}
