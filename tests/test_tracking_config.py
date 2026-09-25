import pytest

from vectrax.tracking.config import TrackingConfig
from vectrax.tracking.geometry import Box
from vectrax.tracking.quality import TrackQuality


def test_defaults_are_valid():
    TrackingConfig()


@pytest.mark.parametrize("kwargs", [
    {"min_quality": 0.7, "good_quality": 0.6},
    {"good_quality": 1.5},
    {"min_quality": -0.1},
    {"confirm_frames": 0},
    {"occlusion_timeout_ns": 0},
    {"init_timeout_ns": -1},
    {"meas_std": 0},
    {"assoc_min_iou": 0},
    {"assoc_min_iou": 1.1},
    {"assoc_label_weight": 0.9},
])
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        TrackingConfig(**kwargs)


def test_box_pixel_round_trip():
    b = Box.from_xywh_px(320, 180, 128, 72, img_w=1280, img_h=720)

    assert (b.cx, b.cy, b.w, b.h) == pytest.approx((0.3, 0.3, 0.1, 0.1))
    assert b.to_xywh_px(1280, 720) == (320, 180, 128, 72)


def test_quality_without_observation_is_none():
    assert TrackQuality(propagator_score=None, motion_residual=None).combined(TrackingConfig()) is None


def test_quality_penalizes_motion_outside_gate():
    cfg = TrackingConfig()
    inside = TrackQuality(propagator_score=0.8, motion_residual=cfg.motion_gate / 2)
    outside = TrackQuality(propagator_score=0.8, motion_residual=cfg.motion_gate * 4)

    assert inside.combined(cfg) == pytest.approx(0.8)
    assert outside.combined(cfg) < inside.combined(cfg)


def test_hysteresis_bands_must_not_overlap():
    with pytest.raises(ValueError):
        TrackingConfig(min_quality=0.4, good_quality=0.6, quality_hysteresis=0.15)
