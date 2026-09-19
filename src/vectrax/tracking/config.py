"""Tracking thresholds and noise. All defaults PROVISIONAL — tune on fixtures."""

from dataclasses import dataclass

__all__ = ["TrackingConfig"]

NS_PER_S = 1_000_000_000
CHI2_4DOF_99 = 13.28


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    # Quality thresholds on TrackQuality.combined (0..1).
    min_quality: float = 0.3
    good_quality: float = 0.6
    confirm_frames: int = 3
    init_timeout_ns: int = 1 * NS_PER_S
    occlusion_timeout_ns: int = 3 * NS_PER_S

    # Kalman, normalized units (frame fraction) per second.
    meas_std: float = 0.005
    acc_std: float = 3.0
    size_acc_std: float = 0.5
    init_vel_std: float = 1.0
    motion_gate: float = CHI2_4DOF_99

    def __post_init__(self):
        if not 0 <= self.min_quality < self.good_quality <= 1:
            raise ValueError("need 0 <= min_quality < good_quality <= 1")

        if self.confirm_frames < 1:
            raise ValueError("confirm_frames must be >= 1")

        if self.init_timeout_ns <= 0 or self.occlusion_timeout_ns <= 0:
            raise ValueError("timeouts must be > 0")

        for name in ("meas_std", "acc_std", "size_acc_std", "init_vel_std", "motion_gate"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")
