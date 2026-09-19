import json

import cv2
import numpy as np
import pytest

from vectrax.frames import PixelFormat
from vectrax.sources.file import FileSource

W, H, FPS = 64, 48, 30


def _video(path, n):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(n):
        writer.write(np.full((H, W, 3), i * 10 % 256, np.uint8))

    writer.release()


def _sidecar(path, stamps):
    path.write_text(json.dumps({"fps": FPS, "stamps": stamps}))


def _drain(src):
    out = []
    while (p := src.read(timeout_s=0)) is not None:
        out.append(p)

    return out


def test_uses_sidecar_timestamps(tmp_path):
    video = tmp_path / "clip.mp4"
    _video(video, 3)
    stamps = [{"capture_ns": 1_000 + i, "arrival_ns": 2_000 + i} for i in range(3)]
    _sidecar(tmp_path / "clip.json", stamps)

    with FileSource(video) as src:
        packets = _drain(src)

    assert [p.frame_id for p in packets] == [0, 1, 2]
    assert [p.capture_ns for p in packets] == [1_000, 1_001, 1_002]
    assert [p.arrival_ns for p in packets] == [2_000, 2_001, 2_002]
    assert packets[0].pixel_format is PixelFormat.BGR
    assert (packets[0].width, packets[0].height) == (W, H)
    assert packets[0].source_id == "file:clip"


def test_without_sidecar_synthesizes_from_fps(tmp_path):
    video = tmp_path / "clip.mp4"
    _video(video, 3)

    with FileSource(video) as src:
        packets = _drain(src)

    assert [p.capture_ns for p in packets] == [round(i * 1_000_000_000 / FPS) for i in range(3)]
    assert all(p.age_ns == 0 for p in packets)


def test_more_frames_than_stamps_rejected(tmp_path):
    video = tmp_path / "clip.mp4"
    _video(video, 3)
    _sidecar(tmp_path / "clip.json", [{"capture_ns": 0, "arrival_ns": 0}])

    with FileSource(video) as src, pytest.raises(ValueError):
        _drain(src)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        FileSource(tmp_path / "nope.mp4")
