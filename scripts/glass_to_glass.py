"""Glass-to-glass latency via screen flash.

The screen toggles black/white; the camera sees the room brightness change.
Latency = first frame past the brightness midpoint − toggle time. Includes
display latency, identical for every camera, so cameras stay comparable.

Setup: dim room, face the camera, sit close to the screen.

Usage:
  uv run scripts/glass_to_glass.py --device MacBook
  uv run scripts/glass_to_glass.py --device Fox --label wired
"""

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from avf_capture import AvfCapture, ensure_permission, find_device

TOGGLES = 60
HALF_PERIOD_S = 0.5
SETTLE_S = 2.0
SAMPLE_STRIDE = 16
MIN_CONTRAST = 8.0
WHITE = 255
NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000
WINDOW = "vectrax-flash"
OUT_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results" / "glass"


def _stats_ms(values_ns):
    a = np.asarray(values_ns, dtype=np.float64) / NS_PER_MS
    return {
        "n": int(a.size),
        "mean": round(float(a.mean()), 1),
        "p50": round(float(np.percentile(a, 50)), 1),
        "p95": round(float(np.percentile(a, 95)), 1),
        "min": round(float(a.min()), 1),
        "max": round(float(a.max()), 1),
    }


def _analyze(frames, toggles):
    cap = np.array([f[0] for f in frames], dtype=np.int64)
    arr = np.array([f[1] for f in frames], dtype=np.int64)
    lum = np.array([f[2] for f in frames])

    half_ns = int(HALF_PERIOD_S * NS_PER_S)

    # Settled level = frames in the second half of each state.
    def level(white):
        sel = [(arr >= t + half_ns // 2) & (arr < t + half_ns) for t, w in toggles if w == white]
        return float(np.median(lum[np.any(sel, axis=0)]))

    low, high = level(False), level(True)
    if high - low < MIN_CONTRAST:
        raise SystemExit(f"Contrast too low ({high - low:.1f}). Dim the room, sit closer to the screen.")

    mid = (low + high) / 2
    arrival_lat, pts_lat = [], []
    for t, white in toggles:
        window = np.nonzero((arr > t) & (arr < t + half_ns))[0]
        crossed = [i for i in window if (lum[i] > mid) == white]
        if not crossed:
            continue

        i = crossed[0]
        arrival_lat.append(arr[i] - t)
        pts_lat.append(cap[i] - t)

    if not arrival_lat:
        raise SystemExit("No brightness transition detected.")

    return {
        "levels": {"black": round(low, 1), "white": round(high, 1)},
        "toggle_to_arrival_ms": _stats_ms(arrival_lat),
        "toggle_to_pts_ms": _stats_ms(pts_lat),
        "missed_toggles": len(toggles) - len(arrival_lat),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    ensure_permission()
    device = find_device(args.device)
    frames = []
    lock = threading.Lock()

    def on_frame(view, capture_ns, arrival_ns):
        if view.shape[:2] != (args.height, args.width):
            return

        lum = float(view[::SAMPLE_STRIDE, ::SAMPLE_STRIDE, :3].mean())
        with lock:
            frames.append((capture_ns, arrival_ns, lum))

    capture = AvfCapture(device, args.width, args.height, args.fps, on_frame)
    cv2.namedWindow(WINDOW, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    black = np.zeros((64, 64), np.uint8)
    white = np.full((64, 64), WHITE, np.uint8)

    capture.start()
    cv2.imshow(WINDOW, black)
    settle_end = time.monotonic() + SETTLE_S
    while time.monotonic() < settle_end:
        cv2.waitKey(10)

    toggles = []
    for i in range(TOGGLES):
        is_white = i % 2 == 0
        cv2.imshow(WINDOW, white if is_white else black)
        cv2.waitKey(1)
        toggles.append((time.monotonic_ns(), is_white))
        next_at = time.monotonic() + HALF_PERIOD_S
        while time.monotonic() < next_at:
            cv2.waitKey(5)

    capture.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    with lock:
        result = _analyze(list(frames), toggles)

    report = {
        "device": {"name": device.localizedName(), "id": device.uniqueID()},
        "label": args.label,
        "requested": {"width": args.width, "height": args.height, "fps": args.fps},
        "toggles": TOGGLES,
        "half_period_s": HALF_PERIOD_S,
        "avf_dropped": capture.dropped,
        **result,
    }
    print(json.dumps(report, indent=2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = device.localizedName().split()[0].lower()
    suffix = f"_{args.label}" if args.label else ""
    out = OUT_DIR / f"{slug}{suffix}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
