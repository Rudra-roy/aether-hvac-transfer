"""Predictive comfort guard (heuristic predictive safety filter).

For watched zones (occupied now or within guard_h steps), roll the zone model forward holding the
planned command. If the forecast leaves the band (shrunk by guard_margin), override that zone with
full corrective action. Untrusted zones use their last-12-step linear trend instead of the model.
"""

import numpy as np

from aether.zone_model import flow_fraction, make_features, rollout


def _held_steps(cmd, snap, parser, n):
    on = cmd.ahu_run[parser.zone_ahu].astype(float)
    ret = np.array([snap.T[g].mean() if len(g) else snap.T.mean() for g in parser.groups])
    mixed = cmd.recirc * ret + (1 - cmd.recirc) * snap.tout
    tsa = np.minimum(mixed, cmd.sat)[parser.zone_ahu]
    v = on * cmd.reheat * float(cmd.pump_run)
    tw = cmd.tw_sp if cmd.pump_run else snap.tw
    u = on * flow_fraction(cmd.damper, cmd.pressure[parser.zone_ahu])
    return [make_features(u, v, tsa, tw, snap.tout, snap.occ / parser.n_occ_ref, (snap.hour + j / 12) % 24)
            for j in range(n)]


def apply(cmd, snap, ident, params, parser, band_lo, band_hi, watch, trusted=None):
    """Returns (command with overrides, overridden mask (V,))."""
    trusted = ident.trusted if trusted is None else trusted
    n = int(round(params.guard_h))
    T, Tm, fp = ident.state
    traj = rollout(ident.theta, T, Tm, _held_steps(cmd, snap, parser, n), fp)[1:]
    T_old = ident._hist[0][0]
    slope = (T - T_old) / max(len(ident._hist) - 1, 1)
    trend = T[None] + slope[None] * np.arange(1, n + 1)[:, None]
    pred = np.where(trusted[None], traj, trend)
    low = watch & (pred.min(0) < band_lo + params.guard_margin)
    high = watch & ~low & (pred.max(0) > band_hi - params.guard_margin)
    out = cmd.copy()
    if low.any():
        out.reheat = np.where(low, 1.0, out.reheat)
        out.damper = np.where(low, 0.3, out.damper)
        out.pump_run = True
        out.tw_sp = max(out.tw_sp, params.tw_max - 5.0)
    if high.any():
        out.damper = np.where(high, 1.0, out.damper)
        out.reheat = np.where(high, 0.0, out.reheat)
    for k, g in enumerate(parser.groups):
        if len(g) and (low[g].any() or high[g].any()):
            out.ahu_run[k] = True
            out.pressure[k] = max(out.pressure[k], 600.0)
            if high[g].any():
                out.sat[k] = params.sat_cool
                out.pressure[k] = max(out.pressure[k], params.p_max)
    mask = low | high
    out.overridden = mask
    return out, mask
