"""Oracle weather for the weather ablation only. Never part of a submission: it reads the scenario's weather file.

The simulator linearly interpolates hourly station temperatures (TempC) from the strip's
start_timestamp; step t is start + 5 t minutes.
"""

import csv
from pathlib import Path

import numpy as np
import yaml

STEP_MIN = 5


class WeatherOracle:
    def __init__(self, hours: np.ndarray, temps_k: np.ndarray, start_hour_index: float):
        self.hours, self.temps, self.h0 = hours, temps_k, start_hour_index

    @staticmethod
    def from_scenario(scenario_dir) -> "WeatherOracle":
        scenario_dir = Path(scenario_dir)
        cfg = yaml.safe_load((scenario_dir / "config.yaml").read_text())
        w = cfg["weather"]
        kit = Path(__file__).resolve().parents[1] / "external/starter-kit"  # generated pool scenarios live outside the kit
        kit_weather = next(p for p in [*scenario_dir.resolve().parents, kit] if (p / "weather").is_dir() or (p / "smart_buildings/weather").is_dir())
        base = kit_weather / "weather" if (kit_weather / "weather").is_dir() else kit_weather / "smart_buildings/weather"
        rows = list(csv.DictReader(open(base / w["weather_csv"])))
        temps = np.array([float(r["TempC"]) for r in rows]) + 273.15
        stamps = [r["Time"] for r in rows]  # YYYYMMDD-HHMM, hourly
        start = str(w["start_timestamp"])  # "2023-01-01 06:00"
        key = start[:4] + start[5:7] + start[8:10] + "-" + start[11:13] + start[14:16]
        h0 = stamps.index(key[:11] + "00")
        frac = int(key[11:13]) / 60.0
        return WeatherOracle(np.arange(len(temps)), temps, h0 + frac)

    def tout(self, step: int) -> float:
        h = self.h0 + step * STEP_MIN / 60.0
        return float(np.interp(h, self.hours, self.temps))
