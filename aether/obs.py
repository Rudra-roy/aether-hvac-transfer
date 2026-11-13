"""Observation vector -> named physical arrays (Kelvin, fractions), built from metadata only.

Simulator facts this relies on (verified against the simulator, see tests/test_obs.py):
- only VAV_* fields and two hws_* fields arrive normalised (native = obs*std + mean); the rest is native;
- the AHU supply-air sensor echoes its setpoint, so the air actually supplied is
  min(mixed, setpoint), mixed = recirc*return + (1-recirc)*outside, return ~ mean T of the AHU's zones.
"""

from dataclasses import dataclass

import numpy as np

_NORMALISED_HWS = ("hws_differential_pressure_setpoint", "hws_run_status")


@dataclass
class Snapshot:
    T: np.ndarray        # (V,) zone air temperature, K
    hsp: np.ndarray      # (V,) heating setpoint sensor (lower band edge), K
    csp: np.ndarray      # (V,) cooling setpoint sensor (upper band edge), K
    damper: np.ndarray   # (V,) damper position taken, 0-1
    reheat: np.ndarray   # (V,) reheat valve position taken, 0-1
    tsa_sp: np.ndarray   # (K,) AHU supply-air setpoint (what the sensor reports), K
    recirc: np.ndarray   # (K,) recirculation fraction
    flow: np.ndarray     # (K,) AHU supply airflow
    pressure: np.ndarray # (K,) AHU supply static pressure sensor, Pa (zone flow ~ damper x sqrt(P/2000))
    ahu_on: np.ndarray   # (K,) bool, flow > 0
    tsa_eff: np.ndarray  # (V,) air actually supplied to each zone (before reheat), K
    tout: float
    tw: float
    occ: float
    comfort_now: bool
    comfort_soon: bool
    hour: float          # [0, 24)
    dow: int             # 0 = Monday ... 6 = Sunday
    dp: float = 10.0     # hot-water pump differential pressure setpoint, bar


AREA_REF = 109.0  # m2, median zone area of the 20 pool_train buildings the prior was fitted on (11 Nov)


def zone_geometry(meta, vavs) -> tuple[np.ndarray, np.ndarray] | None:
    """(area m2, exterior edge count) per VAV zone from the metadata zone_map (0 interior, 1 wall, 2 exterior; 0.2 m cells).
    The generator numbers rooms by 4-connected interior components in raster order, so room_N is component N.
    An edge is an interior cell side facing the exterior directly or across one wall cell."""
    zm = meta.get("zone_map") if hasattr(meta, "get") else None
    if zm is None:
        return None
    grid = np.asarray(zm)
    if grid.ndim != 2:
        return None
    H, W = grid.shape
    gf = grid.ravel()
    g = gf == 0
    seen = np.zeros(g.size, bool)
    counts, edges = [], []
    for start in np.flatnonzero(g):  # flood fill, a few seconds per 10^6 cells, once at start-up
        if seen[start]:
            continue
        stack, n, e = [int(start)], 0, 0
        seen[start] = True
        while stack:
            p = stack.pop()
            n += 1
            r, c = divmod(p, W)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < H and 0 <= cc < W):
                    continue
                q = rr * W + cc
                if g[q]:
                    if not seen[q]:
                        seen[q] = True
                        stack.append(q)
                elif gf[q] == 2:
                    e += 1
                elif 0 <= rr + dr < H and 0 <= cc + dc < W and gf[(rr + dr) * W + cc + dc] == 2:
                    e += 1
        counts.append(n)
        edges.append(e)
    idx = [int(v.rsplit("_", 1)[1]) - 1 for v in vavs]
    if min(idx) < 0 or max(idx) >= len(counts):
        return None
    return 0.04 * np.array([counts[i] for i in idx], float), np.array([edges[i] for i in idx], float)


def zone_areas(meta, vavs) -> np.ndarray | None:
    """Floor area in m2 per VAV zone (zone_geometry), None without a usable zone_map."""
    geo = zone_geometry(meta, vavs)
    return None if geo is None else geo[0]


class ObsParser:
    def __init__(self, meta):
        f = list(meta["observation_field_names"])
        idx = {n: i for i, n in enumerate(f)}
        norm = meta["observation_normalization_metadata"]
        self.mu, self.sd = np.zeros(len(f)), np.ones(len(f))
        for i, n in enumerate(f):
            k = next((k for k in norm if n.endswith(k)), None)
            if k is not None and (n.startswith("VAV_") or n in _NORMALISED_HWS):
                self.mu[i], self.sd[i] = norm[k]["sample_mean"], norm[k]["sample_std"]

        num = lambda s: int(s.rsplit("_", 1)[1])
        self.ahus = sorted(meta["ahu_zone_assignments"], key=num)
        ahu_of = {r.replace("room_", "VAV_"): a for a, rooms in meta["ahu_zone_assignments"].items() for r in rooms}
        self.vavs = sorted((v for v in meta["zone_action_indices"]
                            if f"{v}_zone_air_temperature_sensor" in idx and v in ahu_of), key=num)
        self.zone_ahu = np.array([self.ahus.index(ahu_of[v]) for v in self.vavs], dtype=int)
        self.groups = [np.flatnonzero(self.zone_ahu == k) for k in range(len(self.ahus))]
        self.n_occ_ref = 2.0 * max(len(self.vavs), 1)

        z = lambda s: np.array([idx[f"{v}_{s}"] for v in self.vavs], dtype=int)
        a = lambda s: np.array([idx[f"ahs_{x}_{s}"] for x in self.ahus], dtype=int)
        self._T, self._hsp, self._csp = z("zone_air_temperature_sensor"), z("zone_air_heating_temperature_setpoint_sensor"), z("zone_air_cooling_temperature_setpoint_sensor")
        self._damper, self._reheat = z("supply_air_damper_percentage_command"), z("reheat_valve_percentage_command")
        self._tsa, self._recirc, self._flow = a("supply_air_temperature_sensor"), a("recirculation_percentage_command"), a("supply_air_flowrate_sensor")
        self._pres = a("supply_air_static_pressure_sensor")
        self._tout = idx[f"ahs_{self.ahus[0]}_outside_air_temperature_sensor"]
        self._tw, self._occ = idx["hws_supply_water_temperature_sensor"], idx["num_occupants"]
        self._now, self._soon = idx["comfort_mode_now"], idx["comfort_mode_soon"]
        self._dp = idx.get("hws_differential_pressure_setpoint")
        self._hs, self._hc = idx["hod_sin_000"], idx["hod_cos_000"]
        self._ds, self._dc = idx["dow_sin_000"], idx["dow_cos_000"]

    def parse(self, obs: np.ndarray) -> Snapshot:
        x = np.asarray(obs, dtype=np.float64) * self.sd + self.mu
        T = x[self._T]
        tout = float(x[self._tout])
        tsa_sp, recirc, flow = x[self._tsa], x[self._recirc], x[self._flow]
        ret = np.array([T[g].mean() if len(g) else T.mean() for g in self.groups])
        mixed = recirc * ret + (1 - recirc) * tout
        tsa_eff = np.minimum(mixed, tsa_sp)[self.zone_ahu]
        hour = (np.arctan2(x[self._hs], x[self._hc]) / (2 * np.pi) * 24) % 24
        dow = int(np.floor((np.arctan2(x[self._ds], x[self._dc]) / (2 * np.pi) * 7) % 7 + 1e-6)) % 7
        return Snapshot(T=T, hsp=x[self._hsp], csp=x[self._csp], damper=x[self._damper], reheat=x[self._reheat],
                        tsa_sp=tsa_sp, recirc=recirc, flow=flow, pressure=x[self._pres], ahu_on=flow > 0, tsa_eff=tsa_eff, tout=tout,
                        tw=float(x[self._tw]), occ=float(x[self._occ]), comfort_now=bool(x[self._now] > 0.5),
                        comfort_soon=bool(x[self._soon] > 0.5), hour=float(hour), dow=dow,
                        dp=float(x[self._dp]) if self._dp is not None else 10.0)
