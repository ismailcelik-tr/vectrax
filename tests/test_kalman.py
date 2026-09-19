import pytest

from vectrax.tracking.config import TrackingConfig
from vectrax.tracking.geometry import Box
from vectrax.tracking.kalman import COAST_MARGIN, MIN_SIZE, CvKalman

FRAME_NS = 33_333_333
SPEED = 0.3  # frame widths per second


def _moving(n, speed=SPEED):
    """Boxes moving right at `speed`/s, one per frame."""
    return [(i * FRAME_NS, Box(0.2 + speed * i * FRAME_NS / 1e9, 0.5, 0.1, 0.1)) for i in range(n)]


def _track(samples):
    t0, b0 = samples[0]
    kf = CvKalman(b0, t0, TrackingConfig())
    for t, b in samples[1:]:
        kf.predict(t)
        kf.update(b)

    return kf


def test_learns_constant_velocity():
    kf = _track(_moving(30))

    vx, vy = kf.velocity
    assert vx == pytest.approx(SPEED, abs=0.02)
    assert vy == pytest.approx(0, abs=0.01)


def test_prediction_extrapolates_by_elapsed_time():
    samples = _moving(30)
    kf = _track(samples)
    t_last, b_last = samples[-1]

    kf.predict(t_last + 1_000_000_000)

    assert kf.box.cx == pytest.approx(b_last.cx + SPEED, abs=0.03)


def test_predict_grows_uncertainty_update_shrinks_it():
    kf = _track(_moving(10))
    settled = kf.position_variance

    kf.predict(20 * FRAME_NS)
    grown = kf.position_variance
    kf.update(Box(0.2 + SPEED * 20 * FRAME_NS / 1e9, 0.5, 0.1, 0.1))

    assert grown > settled
    assert kf.position_variance < grown


def test_outlier_has_larger_residual():
    samples = _moving(15)
    kf = _track(samples)
    t_next = 15 * FRAME_NS
    expected = Box(0.2 + SPEED * t_next / 1e9, 0.5, 0.1, 0.1)

    kf.predict(t_next)
    near = kf.residual(expected)
    far = kf.residual(Box(0.9, 0.1, 0.1, 0.1))

    assert far > 10 * near


def test_time_never_goes_back():
    kf = CvKalman(Box(0.5, 0.5, 0.1, 0.1), 1_000, TrackingConfig())

    with pytest.raises(ValueError):
        kf.predict(999)


def _fast_shrinking():
    """Target moving right fast while its box shrinks, then lost."""
    samples = [(i * FRAME_NS, Box(0.5 + 0.03 * i, 0.5, 0.2 - 0.004 * i, 0.2 - 0.004 * i)) for i in range(10)]
    return _track(samples), samples[-1][0]


def test_coasting_damps_velocity():
    kf, t = _fast_shrinking()
    speed = abs(kf.velocity[0])

    for i in range(1, 31):
        kf.predict(t + i * FRAME_NS)
        kf.coast()

    assert abs(kf.velocity[0]) < speed / 10


def test_coasting_stays_in_bounds():
    kf, t = _fast_shrinking()

    for i in range(1, 91):
        kf.predict(t + i * FRAME_NS)
        kf.coast()

    b = kf.box
    assert -COAST_MARGIN <= b.cx <= 1 + COAST_MARGIN
    assert b.w >= MIN_SIZE and b.h >= MIN_SIZE


def test_update_never_yields_negative_size():
    kf = CvKalman(Box(0.5, 0.5, 0.01, 0.01), 0, TrackingConfig())

    kf.predict(FRAME_NS)
    kf.update(Box(0.5, 0.5, -0.05, -0.05))

    assert kf.box.w >= MIN_SIZE and kf.box.h >= MIN_SIZE
