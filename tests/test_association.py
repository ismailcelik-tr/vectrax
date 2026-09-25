from vectrax.evaluation.metrics import iou
from vectrax.tracking.association import associate
from vectrax.tracking.config import TrackingConfig
from vectrax.tracking.geometry import Box
from vectrax.tracking.observation import Observation, Origin

CFG = TrackingConfig()
SIZE = 0.2


def _box(cx, size=SIZE):
    return Box(cx, 0.5, size, size)


def _det(box, label=None):
    return Observation(0, 0, box, 0.9, Origin.DETECTOR, label)


def _iou(a, b):
    return iou((a.cx - a.w / 2, a.cy - a.h / 2, a.w, a.h), (b.cx - b.w / 2, b.cy - b.h / 2, b.w, b.h))


def test_nothing_to_match():
    assert associate({}, [], CFG) == {}
    assert associate({1: _box(0.5)}, [], CFG) == {}
    assert associate({}, [_det(_box(0.5))], CFG) == {}


def test_a_detection_on_a_track_is_matched():
    det = _det(_box(0.51))

    assert associate({1: _box(0.5)}, [det], CFG) == {1: det}


def test_a_detection_below_min_iou_is_not_matched():
    track, det = _box(0.5), _det(_box(0.65))
    assert _iou(track, det.box) < CFG.assoc_min_iou

    assert associate({1: track}, [det], CFG) == {}


def test_side_by_side_targets_keep_their_own_detections():
    near_a, near_b = _det(_box(0.41)), _det(_box(0.51))

    assert associate({1: _box(0.40), 2: _box(0.52)}, [near_b, near_a], CFG) == {1: near_a, 2: near_b}


def test_an_undetected_neighbour_does_not_take_the_other_targets_detection():
    # The 2b hijack: B is an object the detector cannot see; A's detection
    # overlaps B enough to pass the gate, so a per-track best would pull B onto A.
    a, b = _box(0.40), _box(0.48)
    det_a = _det(a)
    assert _iou(b, det_a.box) >= CFG.assoc_min_iou

    assert associate({1: a, 2: b}, [det_a], CFG) == {1: det_a}


def test_a_detection_tied_between_two_tracks_goes_to_neither():
    # Binary fractions, so both overlaps are exactly 1/3.
    size = 0.25
    between = _det(_box(0.5, size))

    assert associate({1: _box(0.375, size), 2: _box(0.625, size)}, [between], CFG) == {}


def test_matching_label_breaks_a_near_tie():
    track = _box(0.5)
    cup, person = _det(_box(0.54), "cup"), _det(_box(0.53), "person")
    assert _iou(track, person.box) > _iou(track, cup.box)

    assert associate({1: track}, [cup, person], CFG) == {1: person}
    assert associate({1: track}, [cup, person], CFG, labels={1: "cup"}) == {1: cup}


def test_a_different_label_still_matches():
    # R1: the target may be something the detector names wrongly.
    bottle = _det(_box(0.51), "bottle")

    assert associate({1: _box(0.5)}, [bottle], CFG, labels={1: "cup"}) == {1: bottle}


def test_a_matching_label_never_lifts_an_overlap_past_the_gate():
    track, cup = _box(0.5), _det(_box(0.6125), "cup")
    overlap = _iou(track, cup.box)
    assert overlap < CFG.assoc_min_iou < overlap * CFG.assoc_label_weight

    assert associate({1: track}, [cup], CFG, labels={1: "cup"}) == {}
