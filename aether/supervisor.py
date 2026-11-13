"""Zone operating modes from the comfort schedule (metadata) and the observed clock.

Comfort mode runs on weekdays from morning_start_hour to evening_start_hour (holidays are not
observable). A zone is OCCUPIED when comfort mode is on or people are present, PRECONDITION when
comfort starts within its lead time, SETBACK otherwise.
"""

import numpy as np

OCCUPIED, PRECONDITION, SETBACK = 0, 1, 2
STEPS_PER_H = 12


def steps_to_comfort(snap, meta) -> int:
    """Steps until the next weekday comfort start (0 while comfort mode is on)."""
    if snap.comfort_now:
        return 0
    start = float(meta["morning_start_hour"])
    h, d = snap.hour, snap.dow
    ahead = 0.0
    for _ in range(8):
        if d < 5 and h < start:
            return int(round((ahead + start - h) * STEPS_PER_H))
        ahead += 24 - h
        h, d = 0.0, (d + 1) % 7
    return 10 ** 6


def holiday(snap, seen_today: bool, after_hour: float) -> bool:
    """A weekday comfort period with nobody in by after_hour is a holiday. The simulator's occupancy model keeps people
    home on US federal holidays but its setpoint schedule does not know them, so the comfort period runs for no one."""
    return after_hour > 0 and snap.comfort_now and not seen_today and snap.hour >= after_hour


def modes(snap, lead_steps: np.ndarray, params, meta) -> np.ndarray:
    V = len(snap.T)
    if snap.comfort_now or snap.occ > 0:
        return np.full(V, OCCUPIED, np.int8)
    k = steps_to_comfort(snap, meta)
    return np.where((k > 0) & (k <= lead_steps), PRECONDITION, SETBACK).astype(np.int8)


def _plan_steps(snap, params, n: int, mode: str) -> list[dict]:
    """Constant full-heat or full-cool inputs for n steps from now (AHU on)."""
    from aether.zone_model import flow_fraction, make_features
    V = len(snap.T)
    mixed = 0.7 * snap.T + 0.3 * snap.tout
    if mode == "heat":
        u, v, tsa = np.full(V, flow_fraction(params.heat_damper, params.p_max)), np.ones(V), np.minimum(mixed, params.sat_idle)
    else:
        u, v, tsa = np.full(V, flow_fraction(1.0, params.p_max)), np.zeros(V), np.minimum(mixed, params.sat_cool)
    return [make_features(u, v, tsa, params.tw_max, snap.tout, 0.0, (snap.hour + j / STEPS_PER_H) % 24, params.reheat_air > 0)
            for j in range(n)]


def _first_reach(traj: np.ndarray, target: np.ndarray, above: bool, kmax: int) -> np.ndarray:
    hit = traj[1:] >= target if above else traj[1:] <= target
    return np.where(hit.any(0), hit.argmax(0) + 1, kmax)


def lead_steps(ident, snap, params, parser, meta, trusted=None) -> np.ndarray:
    """Optimal start: steps of full heating/cooling the model needs to bring each zone into the
    comfort band shrunk by band_margin. Untrusted zones get the default lead."""
    from aether.zone_model import rollout
    V = len(snap.T)
    kmax = int(round(params.max_lead_min / 5))
    default = int(round(params.default_lead_min / 5))
    trusted = ident.trusted if trusted is None else trusted
    T, Tm, fp = ident.state
    clo, chi = meta["comfort_temp_window"]
    tlo, thi = clo + params.band_margin, chi - params.band_margin
    lead = np.zeros(V, int)
    need_h, need_c = T < tlo, T > thi
    if need_h.any():
        traj = rollout(ident.theta, T, Tm, _plan_steps(snap, params, kmax, "heat"), fp)
        lead = np.where(need_h, _first_reach(traj, np.full(V, tlo), True, kmax), lead)
    if need_c.any():
        traj = rollout(ident.theta, T, Tm, _plan_steps(snap, params, kmax, "cool"), fp)
        lead = np.where(need_c, _first_reach(traj, np.full(V, thi), False, kmax), lead)
    return np.where(trusted, lead, default)


def setback_floor(ident, snap, params, parser, meta, trusted=None) -> np.ndarray:
    """Lowest temperature from which max_lead of full heating still reaches the comfort band
    (bisection, mass node assumed settled at the start temperature). -inf for untrusted zones."""
    from aether.zone_model import rollout
    trusted = ident.trusted if trusted is None else trusted
    kmax = int(round(params.max_lead_min / 5))
    tlo = meta["comfort_temp_window"][0] + params.band_margin
    _, _, fp = ident.state
    steps = _plan_steps(snap, params, kmax, "heat")
    lo, hi = np.full(len(snap.T), 270.0), np.full(len(snap.T), tlo)
    for _ in range(8):
        mid = (lo + hi) / 2
        ok = rollout(ident.theta, mid, mid, steps, fp)[-1] >= tlo
        hi, lo = np.where(ok, mid, hi), np.where(ok, lo, mid)
    return np.where(trusted, hi, -np.inf)
