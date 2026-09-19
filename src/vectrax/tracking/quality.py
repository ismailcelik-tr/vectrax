"""Track quality: raw signals kept for diagnostics, one combined score for decisions."""

from dataclasses import dataclass

from vectrax.tracking.config import TrackingConfig

__all__ = ["TrackQuality"]


@dataclass(frozen=True, slots=True)
class TrackQuality:
    propagator_score: float | None  # None = no observation this frame
    motion_residual: float | None  # squared Mahalanobis vs Kalman prediction

    def combined(self, cfg: TrackingConfig) -> float | None:
        if self.propagator_score is None:
            return None

        if self.motion_residual is None or self.motion_residual <= cfg.motion_gate:
            return self.propagator_score

        # PROVISIONAL: outside the gate, scale down in proportion.
        return self.propagator_score * cfg.motion_gate / self.motion_residual
