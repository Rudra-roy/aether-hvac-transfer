"""Physical setpoints -> the competition's normalised hybrid action. Every continuous slot is written.

Layout (from metadata only): per AHU k: temp_indices[k], pressure_indices[k], recirculation_indices[k];
hot-water supply temperature: water_indices[0]; pump differential pressure: pressure_indices[K]
(the kit files it under "pressure"); per VAV: zone_action_indices[vav][damper|reheat].
Discrete: AHU 1..K run commands, then the hot-water pump.
Zone slots have no entry in action_normalization_metadata; they map 0-1 linearly onto [-1, 1]
(verified against the simulator in tests/test_actions.py).
"""

from dataclasses import dataclass, field

import numpy as np

from aether.obs import ObsParser

DAMPER, REHEAT = "supply_air_damper_percentage_command", "reheat_valve_percentage_command"
_DEFAULT_RANGES = {"sat": (285.0, 305.0), "pressure": (0.0, 2000.0), "recirc": (0.0, 1.0), "tw": (310.0, 350.0), "dp": (0.0, 20.0)}


@dataclass
class Command:
    damper: np.ndarray          # (V,) 0-1
    reheat: np.ndarray          # (V,) 0-1
    ahu_run: np.ndarray         # (K,) bool
    sat: np.ndarray             # (K,) K
    pressure: np.ndarray        # (K,) Pa
    recirc: np.ndarray          # (K,) 0-1
    pump_run: bool = False
    tw_sp: float = 330.0        # K
    dp_sp: float = 10.0
    overridden: np.ndarray = field(default=None, repr=False)  # (V,) bool, set by guard/prober

    @staticmethod
    def default(V: int, K: int) -> "Command":
        return Command(damper=np.full(V, 0.1), reheat=np.zeros(V), ahu_run=np.zeros(K, bool), sat=np.full(K, 291.0),
                       pressure=np.full(K, 200.0), recirc=np.full(K, 0.7), pump_run=False, tw_sp=330.0, dp_sp=10.0)

    def copy(self) -> "Command":
        return Command(self.damper.copy(), self.reheat.copy(), self.ahu_run.copy(), self.sat.copy(), self.pressure.copy(),
                       self.recirc.copy(), self.pump_run, self.tw_sp, self.dp_sp,
                       None if self.overridden is None else self.overridden.copy())


class ActionWriter:
    def __init__(self, meta, parser: ObsParser | None = None):
        self.parser = parser or ObsParser(meta)
        self.n = int(meta["n_continuous_action"])
        self.nd = int(meta["n_discrete_action"])
        an = meta["action_normalization_metadata"]
        rng = lambda name, key: (an[name]["min_native_value"], an[name]["max_native_value"]) if name in an else _DEFAULT_RANGES[key]
        K = len(self.parser.ahus)
        self.K = K
        self.temp = list(meta["temp_indices"])
        self.press = list(meta["pressure_indices"])
        self.recirc = list(meta["recirculation_indices"])
        self.water = list(meta["water_indices"])
        self.r_sat = [rng(f"{a}_supply_air_temperature_setpoint", "sat") for a in self.parser.ahus]
        self.r_p = [rng(f"{a}_supply_air_static_pressure_setpoint", "pressure") for a in self.parser.ahus]
        self.r_rc = [rng(f"{a}_recirculation_percentage_command", "recirc") for a in self.parser.ahus]
        self.r_tw = rng("supply_water_temperature_setpoint", "tw")
        self.r_dp = rng("differential_pressure_setpoint", "dp")
        pos = {v: i for i, v in enumerate(self.parser.vavs)}
        self.zones = [(meta["zone_action_indices"][v][DAMPER], meta["zone_action_indices"][v][REHEAT], pos.get(v))
                      for v in meta["zone_action_indices"]]
        self.last_written: set[int] = set()

    @staticmethod
    def _n(x, lo_hi):
        lo, hi = lo_hi
        return np.clip(2.0 * (np.asarray(x, float) - lo) / (hi - lo) - 1.0, -1.0, 1.0)

    @staticmethod
    def _d(a, lo_hi):
        lo, hi = lo_hi
        return lo + (np.asarray(a, float) + 1.0) / 2.0 * (hi - lo)

    def write(self, cmd: Command) -> dict:
        c = np.zeros(self.n, np.float64)
        w = set()

        def put(i, v):
            c[i] = np.nan_to_num(v, nan=0.0)
            w.add(i)

        for k in range(self.K):
            put(self.temp[k], self._n(cmd.sat[k], self.r_sat[k]))
            put(self.press[k], self._n(cmd.pressure[k], self.r_p[k]))
            put(self.recirc[k], self._n(cmd.recirc[k], self.r_rc[k]))
        for i in self.water:
            put(i, self._n(cmd.tw_sp, self.r_tw))
        for i in self.press[self.K:]:
            put(i, self._n(cmd.dp_sp, self.r_dp))
        for di, ri, z in self.zones:
            d, r = (0.1, 0.0) if z is None else (cmd.damper[z], cmd.reheat[z])
            put(di, self._n(d, (0.0, 1.0)))
            put(ri, self._n(r, (0.0, 1.0)))
        for i in range(self.n):  # any slot the metadata does not describe: explicit mid-range
            if i not in w:
                put(i, 0.0)
        self.last_written = w
        disc = np.zeros(self.nd, np.int32)
        disc[: self.K] = np.asarray(cmd.ahu_run, bool)
        disc[-1] = int(bool(cmd.pump_run))
        return {"continuous_action": np.clip(c, -1, 1).astype(np.float32), "discrete_action": disc}

    def decode(self, action: dict) -> Command:
        c, d = np.asarray(action["continuous_action"], float), np.asarray(action["discrete_action"])
        V = len(self.parser.vavs)
        damper, reheat = np.zeros(V), np.zeros(V)
        for di, ri, z in self.zones:
            if z is not None:
                damper[z], reheat[z] = self._d(c[di], (0, 1)), self._d(c[ri], (0, 1))
        K = self.K
        return Command(damper=damper, reheat=reheat, ahu_run=d[:K].astype(bool),
                       sat=np.array([self._d(c[self.temp[k]], self.r_sat[k]) for k in range(K)]),
                       pressure=np.array([self._d(c[self.press[k]], self.r_p[k]) for k in range(K)]),
                       recirc=np.array([self._d(c[self.recirc[k]], self.r_rc[k]) for k in range(K)]),
                       pump_run=bool(d[-1]), tw_sp=float(self._d(c[self.water[0]], self.r_tw)),
                       dp_sp=float(self._d(c[self.press[K]], self.r_dp)) if len(self.press) > K else 10.0)
