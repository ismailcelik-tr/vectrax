"""Record a fixture clip: H.264 video + per-frame timestamp sidecar.

Usage:
  uv run scripts/record_fixture.py --name single_target --seconds 20

Preview window shows a countdown, then REC. Press q to stop early.
Output: data/fixtures/<name>.mp4 and <name>.json.
"""

import argparse
import json
import platform
import queue
import subprocess
import threading
import time
from pathlib import Path

import cv2

from vectrax.sources.avf import AvfCapture, ensure_permission, find_device

DEFAULT_COUNTDOWN_S = 3
WRITE_QUEUE_FRAMES = 60
PREVIEW_SCALE = 0.5
CRF = 12
KEY_QUIT = ord("q")
WINDOW = "vectrax-record"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "fixtures"


def _ffmpeg(path, width, height, fps):
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", str(CRF), "-pix_fmt", "yuv420p", str(path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--device", default="MacBook")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--note", default="", help="what happens in the clip")
    p.add_argument("--countdown", type=float, default=DEFAULT_COUNTDOWN_S)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    video = OUT_DIR / f"{args.name}.mp4"
    sidecar = OUT_DIR / f"{args.name}.json"
    if video.exists():
        raise SystemExit(f"{video} exists; pick another --name or delete it")

    ensure_permission()
    device = find_device(args.device)
    size = (args.height, args.width)

    latest = {"frame": None}
    recording = threading.Event()
    pending = queue.Queue(maxsize=WRITE_QUEUE_FRAMES)
    stamps = []
    skipped = {"queue_full": 0}

    def on_frame(view, capture_ns, arrival_ns):
        if view.shape[:2] != size:
            return

        frame = view.copy()
        latest["frame"] = frame
        if not recording.is_set():
            return

        try:
            pending.put_nowait(frame)
            stamps.append({"capture_ns": capture_ns, "arrival_ns": arrival_ns})
        except queue.Full:
            skipped["queue_full"] += 1

    encoder = _ffmpeg(video, args.width, args.height, args.fps)

    def writer():
        while True:
            frame = pending.get()
            if frame is None:
                return

            encoder.stdin.write(frame.tobytes())

    write_thread = threading.Thread(target=writer, daemon=True)
    write_thread.start()

    capture = AvfCapture(device, args.width, args.height, args.fps, on_frame)
    capture.start()
    cv2.namedWindow(WINDOW)
    start = time.monotonic()
    rec_start = None
    while True:
        now = time.monotonic()
        if rec_start is None and now - start >= args.countdown:
            rec_start = now
            recording.set()

        if rec_start is not None and now - rec_start >= args.seconds:
            break

        frame = latest["frame"]
        if frame is not None:
            shown = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR), None, fx=PREVIEW_SCALE, fy=PREVIEW_SCALE)
            if rec_start is None:
                label = f"{args.name}: starts in {args.countdown - (now - start):.0f}"
                color = (0, 200, 255)
            else:
                label = f"REC {args.name} {now - rec_start:.1f}/{args.seconds:.0f}s"
                color = (0, 0, 255)

            cv2.putText(shown, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow(WINDOW, shown)

        if cv2.waitKey(10) & 0xFF == KEY_QUIT:
            break

    recording.clear()
    capture.stop()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    pending.put(None)
    write_thread.join()
    encoder.stdin.close()
    encoder.wait()

    meta = {
        "name": args.name,
        "note": args.note,
        "device": {"name": device.localizedName(), "id": device.uniqueID()},
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "frames": len(stamps),
        "avf_dropped": capture.dropped,
        "queue_full_skipped": skipped["queue_full"],
        "macos": platform.mac_ver()[0],
        "clock": "time.monotonic_ns; capture_ns = sensor PTS (ADR-003)",
        "stamps": stamps,
    }
    sidecar.write_text(json.dumps(meta, indent=1))
    print(f"Saved {video} ({len(stamps)} frames, dropped={capture.dropped}, "
          f"queue_full={skipped['queue_full']})")


if __name__ == "__main__":
    main()
