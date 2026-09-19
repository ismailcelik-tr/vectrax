"""Constant-velocity Kalman filter over (cx, cy, w, h). Time step from timestamps."""

import numpy as np

from vectrax.tracking.config import NS_PER_S, TrackingConfig
from vectrax.tracking.geometry import Box

__all__ = ["COAST_MARGIN", "MIN_SIZE", "CvKalman"]

MIN_SIZE = 0.005  # frame fraction
COAST_MARGIN = 0.5  # how far beyond the frame a predicted center may go

_DIM = 4
_H = np.hstack([np.eye(_DIM), np.zeros((_DIM, _DIM))])


class CvKalman:
    def __init__(self, box: Box, t_ns: int, cfg: TrackingConfig):
        self._cfg = cfg
        self._t = t_ns
        self._last_dt = 0.0
        self._x = np.array([box.cx, box.cy, box.w, box.h, 0, 0, 0, 0], dtype=np.float64)
        self._P = np.diag([cfg.meas_std**2] * _DIM + [cfg.init_vel_std**2] * _DIM)
        self._R = np.eye(_DIM) * cfg.meas_std**2
        self._acc = np.array([cfg.acc_std, cfg.acc_std, cfg.size_acc_std, cfg.size_acc_std])

    @property
    def box(self) -> Box:
        return Box(*(float(v) for v in self._x[:_DIM]))

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self._x[4]), float(self._x[5])

    @property
    def position_variance(self) -> float:
        return float(self._P[0, 0] + self._P[1, 1])

    @property
    def t_ns(self) -> int:
        return self._t

    def predict(self, t_ns: int) -> None:
        if t_ns < self._t:
            raise ValueError(f"predict into the past: {t_ns} < {self._t}")

        dt = (t_ns - self._t) / NS_PER_S
        self._t = t_ns
        self._last_dt = dt
        if dt == 0:
            return

        F = np.eye(2 * _DIM)
        F[:_DIM, _DIM:] = np.eye(_DIM) * dt
        # Piecewise white-acceleration noise per axis.
        q = self._acc**2
        Q = np.zeros((2 * _DIM, 2 * _DIM))
        Q[:_DIM, :_DIM] = np.diag(q * dt**4 / 4)
        Q[:_DIM, _DIM:] = np.diag(q * dt**3 / 2)
        Q[_DIM:, :_DIM] = np.diag(q * dt**3 / 2)
        Q[_DIM:, _DIM:] = np.diag(q * dt**2)

        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q
        self._clamp()

    def coast(self) -> None:
        """Call after predict when no measurement arrived: decay velocity so an
        unseen target's estimate settles instead of running away."""
        self._x[_DIM:] *= np.exp(-self._last_dt * NS_PER_S / self._cfg.coast_tau_ns)

    def residual(self, box: Box) -> float:
        """Squared Mahalanobis distance of a measurement from the prediction."""
        y, S = self._innovation(box)
        return float(y @ np.linalg.solve(S, y))

    def update(self, box: Box) -> float:
        y, S = self._innovation(box)
        K = self._P @ _H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (np.eye(2 * _DIM) - K @ _H) @ self._P
        self._clamp()
        return float(y @ np.linalg.solve(S, y))

    def _clamp(self):
        for i in (0, 1):
            lo, hi = -COAST_MARGIN, 1 + COAST_MARGIN
            if not lo <= self._x[i] <= hi:
                self._x[i] = min(max(self._x[i], lo), hi)
                self._x[_DIM + i] = 0.0

        for i in (2, 3):
            if self._x[i] < MIN_SIZE:
                self._x[i] = MIN_SIZE
                self._x[_DIM + i] = max(self._x[_DIM + i], 0.0)

    def _innovation(self, box):
        z = np.array([box.cx, box.cy, box.w, box.h])
        return z - _H @ self._x, _H @ self._P @ _H.T + self._R
