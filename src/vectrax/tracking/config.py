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
    # Band around each threshold; stops TRACKING/DEGRADED/OCCLUDED flapping.
    quality_hysteresis: float = 0.1
    # Leaving INITIALIZING needs confirm_frames good frames out of the last
    # confirm_window. NanoTrack's box jitters on a 3-frame rhythm, so good
    # frames arrive one in three; consecutive ones never confirm (ROADMAP).
    # The window is well inside init_timeout_ns, so a wrong box still times out.
    confirm_frames: int = 3
    confirm_window: int = 9
    init_timeout_ns: int = 1 * NS_PER_S
    occlusion_timeout_ns: int = 3 * NS_PER_S

    # Kalman, normalized units (frame fraction) per second.
    meas_std: float = 0.005
    acc_std: float = 3.0
    size_acc_std: float = 0.5
    init_vel_std: float = 1.0
    motion_gate: float = CHI2_4DOF_99
    # While coasting (no visible observation), velocity decays with this time constant.
    coast_tau_ns: int = 3 * NS_PER_S // 10

    # Detection ↔ track association. The label weight only ranks candidates
    # that already pass the IoU gate (R1: the detector's class is a hint).
    assoc_min_iou: float = 0.3
    assoc_label_weight: float = 1.2
    # A matched detection counts as "visible" for this long after the frame it
    # saw; it bridges the frames between detections (stride 2 plus latency).
    detection_hold_ns: int = 2 * NS_PER_S // 10

    def __post_init__(self):
        if not 0 <= self.min_quality < self.good_quality <= 1:
            raise ValueError("need 0 <= min_quality < good_quality <= 1")

        h = self.quality_hysteresis
        if h < 0 or self.min_quality + h > self.good_quality - h:
            raise ValueError("hysteresis bands around min_quality and good_quality overlap")

        if self.confirm_frames < 1:
            raise ValueError("confirm_frames must be >= 1")

        if not 0 < self.confirm_frames <= self.confirm_window:
            raise ValueError("need 0 < confirm_frames <= confirm_window")

        if min(self.init_timeout_ns, self.occlusion_timeout_ns, self.coast_tau_ns, self.detection_hold_ns) <= 0:
            raise ValueError("timeouts must be > 0")

        for name in ("meas_std", "acc_std", "size_acc_std", "init_vel_std", "motion_gate"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0")

        if not 0 < self.assoc_min_iou <= 1:
            raise ValueError("need 0 < assoc_min_iou <= 1")

        if self.assoc_label_weight < 1:
            raise ValueError("assoc_label_weight must be >= 1")
