"""Per-zone residual on the planner's zone commands: a small shared MLP, numpy inference (trained by harness/rl.py).

delta = BOUND * tanh(mlp(features)), added to each zone's damper and reheat. The last layer starts at zero, so an
untrained residual is exactly the base controller. The residual has no tank-setpoint lever (no tank cycling).
"""

import numpy as np

from aether.planner import DAMPER_MIN, TW_IDLE
from aether.supervisor import steps_to_comfort

N_FEAT, HIDDEN, BOUND = 16, 64, 0.3


def residual_init(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {"W1": rng.normal(0, 1 / np.sqrt(N_FEAT), (N_FEAT, HIDDEN)), "b1": np.zeros(HIDDEN),
            "W2": rng.normal(0, 1 / np.sqrt(HIDDEN), (HIDDEN, HIDDEN)), "b2": np.zeros(HIDDEN),
            "W3": np.zeros((HIDDEN, 2)), "b3": np.zeros(2), "log_std": np.full(2, -1.0)}


def residual_features(snap, ident, cmd, parser, meta) -> np.ndarray:
    """(V, N_FEAT) per-zone inputs: comfort margins, mass node, offset, weather, clock, the planner's own command."""
    T, Tm, _ = ident.state
    ahu = parser.zone_ahu
    h = 2 * np.pi * snap.hour / 24
    one = np.ones_like(T)
    stc = min(steps_to_comfort(snap, meta), 144) / 144
    x = np.stack([(T - snap.hsp) / 2, (snap.csp - T) / 2, (Tm - T) / 2, ident.bias * 10, (snap.tout - T) / 20,
                  np.sin(h) * one, np.cos(h) * one, float(snap.comfort_now) * one, stc * one, float(snap.dow >= 5) * one,
                  cmd.damper, cmd.reheat, cmd.ahu_run[ahu].astype(float), (cmd.sat[ahu] - T) / 10,
                  float(cmd.pump_run) * one, snap.occ / parser.n_occ_ref * one], 1)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def residual_forward(w: dict, x: np.ndarray) -> np.ndarray:
    """(V, 2) pre-tanh mean of the correction."""
    h = np.tanh(x @ w["W1"] + w["b1"])
    h = np.tanh(h @ w["W2"] + w["b2"])
    return h @ w["W3"] + w["b3"]


def residual_apply(cmd, pre: np.ndarray, params) -> None:
    """Add BOUND*tanh(pre) to damper and reheat in place; untouched where the correction is exactly zero."""
    delta = BOUND * np.tanh(pre)
    if not delta.any():
        return
    cmd.damper = np.where(delta[:, 0] != 0, np.clip(cmd.damper + delta[:, 0], DAMPER_MIN, 1.0), cmd.damper)
    cmd.reheat = np.where(delta[:, 1] != 0, np.clip(cmd.reheat + delta[:, 1], 0.0, 1.0), cmd.reheat)
    pump = bool((cmd.reheat > 0).any())
    if pump and not cmd.pump_run:
        cmd.tw_sp = max(cmd.tw_sp, params.tw_min)  # the residual may start the pump, never lower the tank setpoint
    elif not pump:
        cmd.tw_sp = TW_IDLE
    cmd.pump_run = pump
