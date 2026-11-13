"""Physical command from zone modes: band-edge targets, zone loops, AHU and hot-water resets."""

import numpy as np

from aether.actions import Command
from aether.supervisor import OCCUPIED, PRECONDITION

T_OUT_COLD, T_OUT_WARM = 253.15, 288.15  # hot-water reset end points (-20 C, 15 C)
TW_IDLE = 310.0  # lowest commandable supply-water temperature
DAMPER_MIN = 0.1


def targets(snap, modes, params, meta, setback_floor=None):
    """(lo, hi) per zone: the band edges moved inward by the mode's margin."""
    clo, chi = meta["comfort_temp_window"]
    lo = np.where(modes == PRECONDITION, clo, snap.hsp).astype(float)
    hi = np.where(modes == PRECONDITION, chi, snap.csp).astype(float)
    m = np.where(modes == OCCUPIED, params.band_margin, np.where(modes == PRECONDITION, params.band_margin, params.setback_margin))
    lo, hi = lo + m, hi - m
    if setback_floor is not None:
        lo = np.where((modes != OCCUPIED) & (modes != PRECONDITION), np.maximum(lo, setback_floor), lo)
    mid = (lo + hi) / 2
    bad = lo > hi
    return np.where(bad, mid, lo), np.where(bad, mid, hi)


RECIRC_MAX = 0.7  # simulator cap
DEFICIT_RAMP_K = (8.0, 16.0)  # zone minus mixed-air temperature over which the heating airflow gain ramps down
HEAT_DAMPER_MILD = 0.9


def economiser(t_return: float, tout: float, sat: float, cooling: bool, margin: float) -> float:
    """Recirculation fraction. When cooling and outside air is colder than return air, mix so the
    mixed air lands on the supply target, never colder (a winter blast would freeze the neighbours)."""
    if not cooling or tout >= t_return - margin:
        return RECIRC_MAX
    r = (sat - tout) / max(t_return - tout, 1e-6)
    return float(np.clip(r, 0.0, RECIRC_MAX))


def plan(snap, modes, gains, params, parser, meta, setback_floor=None) -> Command:
    V, K = len(parser.vavs), len(parser.ahus)
    lo, hi = targets(snap, modes, params, meta, setback_floor)
    e_h, e_c = np.maximum(lo - snap.T, 0), np.maximum(snap.T - hi, 0)
    reheat = np.clip(params.kp_heat * e_h / gains[:, 0], 0, 1)
    cool = np.clip(params.kp_cool * e_c / gains[:, 1], 0, 1)
    # Heating damper tracks the valve. The coil needs airflow, so in mild weather the gain is high; when the
    # mixed air is far below the zone, extra air overwhelms the coil and the gain drops to heat_damper
    # (measured: fairbanks_1 at -33 C vs fairbanks_5 / singapore / denver).
    deficit = snap.T - (RECIRC_MAX * snap.T + (1 - RECIRC_MAX) * snap.tout)
    gain = np.interp(deficit, DEFICIT_RAMP_K, [HEAT_DAMPER_MILD, params.heat_damper])
    damper = np.where(e_c > 0, DAMPER_MIN + (1 - DAMPER_MIN) * cool, np.minimum(1.0, DAMPER_MIN + gain * reheat))
    cmd = Command.default(V, K)
    cmd.damper, cmd.reheat = damper, reheat
    occupied = modes == OCCUPIED
    for k, g in enumerate(parser.groups):
        if len(g) == 0:
            continue
        heat, coolg = (e_h[g] > 0).any(), (e_c[g] > 0).any()
        cmd.ahu_run[k] = heat or coolg or occupied[g].any()
        cmd.sat[k] = params.sat_cool if coolg else params.sat_idle
        cmd.recirc[k] = economiser(snap.T[g].mean(), snap.tout, cmd.sat[k], coolg, params.econ_margin)
        demand = (damper[g].max() - DAMPER_MIN) / (1 - DAMPER_MIN)
        cmd.pressure[k] = 200.0 + (params.p_max - 200.0) * demand
    cmd.pump_run = bool((reheat > 0).any())
    f = np.clip((snap.tout - T_OUT_COLD) / (T_OUT_WARM - T_OUT_COLD), 0, 1)
    # Outdoor reset of the hot-water supply. When no zone needs reheat the loop idles at the lowest
    # setpoint: the tank sits outdoors and keeping it hot is pure dissipation (measured: idling is
    # comfort-free and cuts Phoenix 7%; a fixed cap instead costs +135% on Fairbanks).
    cmd.tw_sp = float(params.tw_max + f * (params.tw_min - params.tw_max)) if cmd.pump_run else TW_IDLE
    cmd.dp_sp = 10.0
    return cmd
