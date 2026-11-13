"""Per-zone grey-box thermal model, linear in its 10 parameters (validated in demo/pipeline.py).

dT = a (Tout-T) + b u (Tsa-T) + r v (Tw-T)                 envelope, airflow, reheat
   + b1 u' (Tsa'-T) + r1 v' (Tw'-T)                         same, one step earlier (delivery lag)
   + k (Tm-T) + c occ + s1 sin(tod) + s2 cos(tod) + d       slow mass node, people, sun/schedule, bias
u = AHU-on x damper x sqrt(P/2000) (the simulator's zone airflow), v = AHU-on x reheat valve, Tsa = air actually supplied (Snapshot.tsa_eff),
Tm = 2-h exponential average of T. Columns are scaled (/10, /50) so all parameters are O(1).
"""

import numpy as np

TAU = 24  # mass-node time constant, steps (2 h)
NAMES = ("a", "b", "r", "b1", "r1", "k", "c", "s1", "s2", "d")
P = len(NAMES)


T_DIS_MAX = 308.15  # K, the VAV box's maximum discharge temperature (vav.py)
# water / air heat-capacity rate of one box per unit valve, per unit airflow fraction, at 1 bar pump head:
# water flow = valve x Kv 0.108 x sqrt(dp) m3/h, air flow = u x 2.0 m3/s (identical in all 206 generated and shipped configs)
KAPPA = 4180.0 * 1000.0 * 0.108 / 3600 / (1006.0 * 1.2 * 2.0)


def make_features(u, v, tsa, tw, tout, occn, hour, air: bool = False) -> dict:
    """air: the airflow terms carry the box's discharge temperature (reheat arrives with the air), see discharge()."""
    h = 2 * np.pi * hour / 24
    f = {"u": np.asarray(u, float), "v": np.asarray(v, float), "tsa": np.asarray(tsa, float), "tw": float(tw),
         "tout": float(tout), "occn": float(occn), "hs": float(np.sin(h)), "hc": float(np.cos(h))}
    if air:
        f["air"] = True
    return f


def discharge(f: dict) -> np.ndarray:
    """Air leaving the VAV box, as the simulator's reheat coil computes it: tsa + e (tw - tsa), effectiveness
    e = min(1, water capacity / air capacity), and never hotter than T_DIS_MAX. The zone then receives
    flow x rho cp x (discharge - T): reheat scales with airflow and saturates, unlike a radiator term r v (tw - T)."""
    u, tsa = np.asarray(f["u"], float), np.asarray(f["tsa"], float)
    # dp: observed pump head when identifying (0 with the pump off, so a stale open valve adds nothing); planned
    # steps omit it and get the 10 bar aether always commands (the pre-action observation reads 0 at a pump start)
    e = np.minimum(1.0, KAPPA * np.sqrt(f.get("dp", 10.0)) * np.asarray(f["v"], float) / np.maximum(u, 1e-9))
    gap = f["tw"] - tsa
    dis = tsa + e * gap
    return np.where(gap > 0, np.minimum(dis, np.maximum(T_DIS_MAX, tsa)), dis)


AIR_SHARE_CAP = 20.0 / 2.0  # AHU max flow / VAV max flow (20 and 2.0 m3/s in all 208 generated and shipped configs)


def ahu_share(u, zone_ahu) -> np.ndarray:
    """(n,) share of its demanded air each box receives. An AHU moves at most AIR_SHARE_CAP boxes' worth of design
    flow; above that the simulator scales every box on it down equally (simulator_flexible_floor_plan supply_air_flow_ratio).
    The water side is not scaled (header_resistance 0), so a capped box heats less air to a hotter discharge."""
    u, zone_ahu = np.asarray(u, float), np.asarray(zone_ahu)
    tot = np.bincount(zone_ahu, weights=u)
    return np.minimum(1.0, AIR_SHARE_CAP / np.maximum(tot, 1e-9))[zone_ahu]


P_DESIGN = 2000.0  # Pa; simulator zone flow = damper x max_flow x sqrt(P / P_DESIGN)


def flow_fraction(damper, pressure):
    """Zone airflow as a fraction of design flow, as the simulator computes it."""
    return np.asarray(damper, float) * np.sqrt(np.clip(np.asarray(pressure, float), 0.0, None) / P_DESIGN)


def features(snap, parser, air: bool = False) -> dict:
    """Model inputs for the interval starting at this snapshot (actuation reported now acts over the next step).
    u is the zone's true airflow fraction (pressure matters: 200 Pa gives 32% of design flow); v is the
    reheat valve (scaling it by airflow was measured to fit much worse)."""
    on = snap.ahu_on[parser.zone_ahu].astype(float)
    u = on * flow_fraction(snap.damper, snap.pressure[parser.zone_ahu])
    if air:
        u = u * ahu_share(u, parser.zone_ahu)
    f = make_features(u, on * snap.reheat, snap.tsa_eff, snap.tw, snap.tout, snap.occ / parser.n_occ_ref, snap.hour, air)
    if air:
        f["dp"] = snap.dp
    return f


def regressors(T, Tm, f: dict, fp: dict) -> np.ndarray:
    """(V, 10) regressor matrix for state (T, Tm), current inputs f and previous-step inputs fp."""
    one = np.ones_like(T)
    ts, tsp = (discharge(f), discharge(fp)) if f.get("air") else (f["tsa"], fp["tsa"])
    return np.stack([
        (f["tout"] - T) / 10,
        f["u"] * (ts - T) / 10, f["v"] * (f["tw"] - T) / 50,
        fp["u"] * (tsp - T) / 10, fp["v"] * (fp["tw"] - T) / 50,
        (Tm - T) / 10, f["occn"] * one, f["hs"] * one, f["hc"] * one, one], -1)


def rollout(theta: np.ndarray, T0: np.ndarray, Tm0: np.ndarray, steps: list[dict], prev: dict) -> np.ndarray:
    """Simulate len(steps) steps; returns (len(steps)+1, V) temperatures including T0."""
    T, Tm, fp = np.array(T0, float), np.array(Tm0, float), prev
    out = [T.copy()]
    for f in steps:
        dT = np.einsum("vp,vp->v", regressors(T, Tm, f, fp), theta)
        Tm = Tm + (T - Tm) / TAU
        T = T + dT
        fp = f
        out.append(T.copy())
    return np.stack(out)
