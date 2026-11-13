"""Corrected gas billing for analysis runs only (AETHER_BILLING_FIX=1). Never part of a policy.

The kit bills the boiler's flow heating and tank loss at the hot-water SETPOINT, while the coils receive the
tank's actual temperature (boiler.py compute_thermal_energy_rate). This patch bills both at the actual tank
temperature. It changes accounting only: the simulated temperatures and every action are unchanged.
"""
import os


def apply() -> bool:
    if os.environ.get("AETHER_BILLING_FIX") != "1":
        return False
    from smart_buildings.smart_control.simulator import boiler

    orig = boiler.Boiler.compute_thermal_energy_rate

    def fixed(self, return_water_temp, outside_temp, total_flow_rate):
        sp = self._supply_water_temperature_setpoint
        self._supply_water_temperature_setpoint = self._current_temperature  # bill what the coils actually get
        try:
            return orig(self, return_water_temp, outside_temp, total_flow_rate)
        finally:
            self._supply_water_temperature_setpoint = sp

    boiler.Boiler.compute_thermal_energy_rate = fixed
    return True
