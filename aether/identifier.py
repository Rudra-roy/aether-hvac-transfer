"""Online per-zone identification: batched Kalman filter with random-walk parameters.

- population prior (m0, P0, R) fitted on generated buildings (harness/fit_prior.py);
- excitation gate: a step updates a zone only if it carries information ((s - R) / R > gate);
- trust gate: a > -sd_a, b > 0, sd_b < 0.5 b, rolling 24-h RMSE of the filter's own 1-h hindcasts
  < trust_rmse with >= 36 scored hindcasts;
- non-finite input or a diverged estimate resets that zone to the prior (sensor fault / blow-up).
"""

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aether.zone_model import P, TAU, regressors, rollout

H = 12          # hindcast horizon for the trust check (1 h)
WINDOW = 288    # rolling window for the trust RMSE (24 h)
MIN_SCORED = 36
ACT = slice(1, 5)  # zone_model columns b, r, b1, r1: airflow and reheat, current and one step back
ACT_REF2 = 0.01    # K^2: typical squared one-step actuation effect, scales the prior's pseudo-steps
G_MIN, G_MAX = 0.2, 1.5
BIAS_MAX = 0.1  # K/step; an offset beyond this is a fault, not a disturbance


@dataclass
class Prior:
    m0: np.ndarray
    P0: np.ndarray
    R: float

    @staticmethod
    def load(path) -> "Prior":
        d = json.loads(Path(path).read_text())
        return Prior(np.array(d["m0"], float), np.array(d["P0"], float), float(d["R"]))

    def save(self, path, **extra) -> None:
        Path(path).write_text(json.dumps({"m0": self.m0.tolist(), "P0": self.P0.tolist(), "R": self.R, **extra}, indent=1))


def fit_batch(X: np.ndarray, y: np.ndarray, lam: float = 1e-4) -> tuple[np.ndarray, float]:
    """Ridge least squares per zone. X (N,V,P), y (N,V) -> theta (V,P), mean residual variance."""
    th = np.stack([np.linalg.solve(X[:, z].T @ X[:, z] + lam * np.eye(P), X[:, z].T @ y[:, z]) for z in range(X.shape[1])])
    return th, float(np.var(y - np.einsum("tzp,zp->tz", X, th)))


class ZoneIdentifier:
    def __init__(self, n_zones: int, prior: Prior, q_rw: float = 1e-6, gate: float = 0.01,
                 trust_rmse: float = 0.75, frozen: bool = False, offset_alpha: float = 0.0, act_rho: float = 0.0,
                 act_n0: float = 288.0, base: np.ndarray | None = None):
        """offset_alpha > 0: offset-free prediction. A per-zone additive bias (K/step), the EMA of one-step
        residuals with this weight, is added to the constant column of theta (one parameter per zone,
        unlike the full filter; cf. disturbance estimation in offset-free MPC)."""
        self.V, self.prior, self.gate, self.trust_rmse, self.frozen = n_zones, prior, gate, trust_rmse, frozen
        self.base = base.astype(float) if base is not None else np.repeat(prior.m0[None], n_zones, 0).astype(float)  # (V, P) per-zone prior
        self.offset_alpha = float(offset_alpha)
        self.bias = np.zeros(n_zones)
        # act_rho > 0: a per-zone scale g on the actuation gains (b, r, b1, r1), forgetting least squares of the
        # residual on the actuation term, shrunk to the prior's scale with act_n0 pseudo-steps; 0 = fixed (bitwise off)
        self.act_rho, self.act_n0 = float(act_rho), float(act_n0)
        self.g = np.ones(n_zones)
        self._g0 = np.ones(n_zones)
        self._saa, self._say = np.zeros(n_zones), np.zeros(n_zones)
        self.R = float(prior.R)
        self.Q = q_rw * np.diag(prior.P0)
        self.m = self.base.copy()
        self.Pm = np.repeat(np.diag(prior.P0)[None], n_zones, 0).astype(float)
        self.Tm = None
        self._hist = deque(maxlen=H + 2)        # (T, Tm, features) per step
        self._theta_hist = deque(maxlen=H + 1)  # theta known at each step
        self._err2 = np.full((WINDOW, n_zones), np.nan)
        self._k = 0
        self.n_scored = np.zeros(n_zones, int)
        self.err_log: list[np.ndarray] = []     # per-step 1-h hindcast errors (NaN before the first)
        self.n_updates = np.zeros(n_zones, int)

    # ---------------------------------------------------------------- estimates
    @property
    def theta(self) -> np.ndarray:
        th = self.base if self.frozen else self.m
        if self.act_rho > 0:
            th = th.copy()
            th[:, ACT] *= self.g[:, None]
        if self.offset_alpha > 0:
            th = th.copy()
            th[:, -1] += self.bias
        return th

    @property
    def sd(self) -> np.ndarray:
        return np.sqrt(np.maximum(np.einsum("vii->vi", self.Pm), 0))

    @property
    def rmse_1h(self) -> np.ndarray:
        n = np.isfinite(self._err2).sum(0)
        r = np.sqrt(np.nansum(self._err2, 0) / np.maximum(n, 1))
        return np.where(n >= MIN_SCORED, r, np.inf)

    @property
    def trusted(self) -> np.ndarray:
        if self.frozen:
            return np.ones(self.V, bool)
        th, sd = self.m, self.sd
        return (th[:, 0] > -sd[:, 0]) & (th[:, 1] > 0) & (sd[:, 1] < 0.5 * np.abs(th[:, 1])) & (self.rmse_1h < self.trust_rmse)

    def rmse_since(self, step: int) -> np.ndarray:
        e = np.array(self.err_log[step:])
        n = np.isfinite(e).sum(0)
        return np.where(n > 0, np.sqrt(np.nansum(e ** 2, 0) / np.maximum(n, 1)), np.inf)

    # ---------------------------------------------------------------- updates
    def _reset(self, zones: np.ndarray) -> None:
        self.m[zones] = self.prior.m0
        self.Pm[zones] = np.diag(self.prior.P0)
        self._err2[:, zones] = np.nan

    def update(self, x: np.ndarray, y: np.ndarray) -> None:
        """One Kalman step for all zones: x (V,P) regressors, y (V,) observed dT."""
        bad = ~np.isfinite(y) | ~np.isfinite(x).all(1)
        if bad.any():
            self._reset(bad)
        if self.frozen:
            return
        x, y = np.where(bad[:, None], 0.0, x), np.where(bad, 0.0, y)
        self.Pm = self.Pm + self.Q
        Px = np.einsum("vij,vj->vi", self.Pm, x)
        s = np.einsum("vi,vi->v", x, Px) + self.R
        on = ((s - self.R) / self.R > self.gate) & ~bad
        k = Px / s[:, None] * on[:, None]
        self.m = self.m + k * (y - np.einsum("vi,vi->v", x, self.m))[:, None]
        self.Pm = self.Pm - np.einsum("vi,vj->vij", k, Px)
        self.n_updates += on
        diverged = ~np.isfinite(self.m).all(1) | (np.abs(self.m) > 1e3).any(1) | ~np.isfinite(self.Pm).all((1, 2))
        if diverged.any():
            self._reset(diverged)

    def observe(self, snap, feats: dict) -> None:
        """Call once per step with the current snapshot and its model inputs."""
        T = np.asarray(snap.T, float)
        if self.Tm is None:
            self.Tm = T.copy()
        if self._hist:
            Tp, Tmp, fp = self._hist[-1]
            fpp = self._hist[-2][2] if len(self._hist) > 1 else fp
            x = regressors(Tp, Tmp, fp, fpp)
            base = self.base if self.frozen else self.m
            if self.act_rho > 0:  # fit g: (dT - non-actuation part - bias) ~ g * actuation part
                act = np.nan_to_num(np.einsum("vp,vp->v", x[:, ACT], base[:, ACT]))
                rest = np.nan_to_num(np.einsum("vp,vp->v", x, base) - act)
                y = np.nan_to_num(T - Tp) - rest - self.bias
                self._saa = self.act_rho * self._saa + act * act
                self._say = self.act_rho * self._say + act * y
                w0 = self.act_n0 * ACT_REF2
                self.g = np.clip((self._say + w0 * self._g0) / (self._saa + w0), G_MIN, G_MAX)
                base = base.copy()
                base[:, ACT] *= self.g[:, None]
            if self.offset_alpha > 0:  # residual of the model without the bias (no feedback through it)
                e1 = np.nan_to_num((T - Tp) - np.einsum("vp,vp->v", x, base))
                self.bias = np.clip((1 - self.offset_alpha) * self.bias + self.offset_alpha * e1, -BIAS_MAX, BIAS_MAX)
            self.update(x, T - Tp)
            self.Tm = Tmp + (Tp - Tmp) / TAU
        self._hist.append((T.copy(), self.Tm.copy(), feats))
        self._theta_hist.append(self.theta.copy())
        e = np.full(self.V, np.nan)
        if len(self._hist) == H + 2:  # hindcast: state 12 steps ago, theta known then, inputs that actually followed
            T0, Tm0, _ = self._hist[1]
            steps = [h[2] for h in list(self._hist)[1:H + 1]]
            pred = rollout(self._theta_hist[0], T0, Tm0, steps, self._hist[0][2])[-1]
            e = pred - T
            self._err2[self._k % WINDOW] = e ** 2
            self._k += 1
            self.n_scored += np.isfinite(e)
        self.err_log.append(e)

    @property
    def state(self):
        """(T, Tm, features) of the latest step."""
        return self._hist[-1]

    def predict(self, steps: list[dict]) -> np.ndarray:
        """Roll the current model forward from the latest state; (len(steps)+1, V)."""
        T, Tm, fp = self._hist[-1]
        return rollout(self.theta, T, Tm, steps, fp)
