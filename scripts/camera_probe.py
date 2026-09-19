"""Phase 0 camera probe: capture timing per device and capture method.

Usage:
  uv run scripts/camera_probe.py --list
  uv run scripts/camera_probe.py --device MacBook
  uv run scripts/camera_probe.py --device Fox --label wired
"""

import argparse
import json
import platform
import resource
import subprocess
import threading
import time
from pathlib import Path

import AVFoundation as AVF
import CoreMedia as CM
import cv2
import numpy as np
import objc
import Quartz
from Foundation import NSObject
from libdispatch import DISPATCH_QUEUE_SERIAL, dispatch_queue_create

WARMUP_FRAMES = 30
GAP_FACTOR = 1.5
BGRA_CHANNELS = 4
NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000
PERMISSION_TIMEOUT_S = 60
OUT_DIR = Path(__file__).resolve().parent.parent / "benchmarks" / "results" / "probe"

METHOD_AVF = "avf"
METHOD_OPENCV = "opencv"
METHOD_BOTH = "both"


def _devices():
    return list(AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo))


def _find_device(query):
    q = query.lower()
    matches = [d for d in _devices() if q in d.localizedName().lower() or query == d.uniqueID()]
    if len(matches) != 1:
        raise SystemExit(f"'{query}' matched {len(matches)} devices; use --list")

    return matches[0]


def _formats(device):
    out = set()
    for f in device.formats():
        dims = CM.CMVideoFormatDescriptionGetDimensions(f.formatDescription())
        for r in f.videoSupportedFrameRateRanges():
            out.add((dims.width, dims.height, r.maxFrameRate()))

    return sorted(out)


def _list():
    for d in _devices():
        print(f"{d.localizedName()}  id={d.uniqueID()}  type={d.deviceType()}")
        for w, h, fps in _formats(d):
            print(f"    {w}x{h} @ {fps:g}")

    print(f"Center Stage enabled: {AVF.AVCaptureDevice.isCenterStageEnabled()}")


def _ensure_permission():
    status = AVF.AVCaptureDevice.authorizationStatusForMediaType_(AVF.AVMediaTypeVideo)
    if status == AVF.AVAuthorizationStatusAuthorized:
        return

    if status != AVF.AVAuthorizationStatusNotDetermined:
        raise SystemExit("Camera denied: System Settings → Privacy & Security → Camera → enable your terminal")

    done = threading.Event()
    granted = []

    def handler(ok):
        granted.append(ok)
        done.set()

    AVF.AVCaptureDevice.requestAccessForMediaType_completionHandler_(AVF.AVMediaTypeVideo, handler)
    done.wait(PERMISSION_TIMEOUT_S)
    if not granted or not granted[0]:
        raise SystemExit("Camera permission not granted")


def _stats_ms(values_ns):
    a = np.asarray(values_ns, dtype=np.float64) / NS_PER_MS
    if a.size == 0:
        return None

    return {
        "mean": round(float(a.mean()), 3),
        "p50": round(float(np.percentile(a, 50)), 3),
        "p95": round(float(np.percentile(a, 95)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "max": round(float(a.max()), 3),
    }


def _timing(arrival_ns, fps):
    intervals = np.diff(np.asarray(arrival_ns, dtype=np.int64))
    span_s = (arrival_ns[-1] - arrival_ns[0]) / NS_PER_S
    nominal_ns = NS_PER_S / fps
    return {
        "frames": len(arrival_ns),
        "effective_fps": round((len(arrival_ns) - 1) / span_s, 2),
        "interval_ms": _stats_ms(intervals),
        "gaps": int((intervals > GAP_FACTOR * nominal_ns).sum()),
    }


class _CpuMeter:
    def __enter__(self):
        self._wall = time.monotonic_ns()
        self._cpu = self._cpu_ns()
        return self

    def __exit__(self, *exc):
        wall = time.monotonic_ns() - self._wall
        self.percent = round(100 * (self._cpu_ns() - self._cpu) / wall, 1)

    @staticmethod
    def _cpu_ns():
        r = resource.getrusage(resource.RUSAGE_SELF)
        return int((r.ru_utime + r.ru_stime) * NS_PER_S)


class _Delegate(NSObject, protocols=[objc.protocolNamed("AVCaptureVideoDataOutputSampleBufferDelegate")]):
    def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sample, connection):
        arrival = time.monotonic_ns()
        host_now = CM.CMTimeGetSeconds(CM.CMClockGetTime(CM.CMClockGetHostTimeClock()))
        pts = CM.CMTimeGetSeconds(CM.CMSampleBufferGetPresentationTimeStamp(sample))

        # Copy cost = what building a FramePacket from a CVPixelBuffer costs.
        t0 = time.monotonic_ns()
        pix = CM.CMSampleBufferGetImageBuffer(sample)
        Quartz.CVPixelBufferLockBaseAddress(pix, Quartz.kCVPixelBufferLock_ReadOnly)
        h = Quartz.CVPixelBufferGetHeight(pix)
        w = Quartz.CVPixelBufferGetWidth(pix)
        bpr = Quartz.CVPixelBufferGetBytesPerRow(pix)
        raw = Quartz.CVPixelBufferGetBaseAddress(pix).as_buffer(bpr * h)
        img = np.frombuffer(raw, np.uint8).reshape(h, bpr)[:, : w * BGRA_CHANNELS].reshape(h, w, BGRA_CHANNELS).copy()
        Quartz.CVPixelBufferUnlockBaseAddress(pix, Quartz.kCVPixelBufferLock_ReadOnly)
        copy_ns = time.monotonic_ns() - t0

        r = self.rec
        r["arrival"].append(arrival)
        r["pts_s"].append(pts)
        r["delivery_s"].append(host_now - pts)
        r["copy"].append(copy_ns)
        r["shapes"].append(img.shape)
        if len(r["arrival"]) >= r["target"]:
            r["done"].set()

    def captureOutput_didDropSampleBuffer_fromConnection_(self, output, sample, connection):
        self.rec["dropped"] += 1


def _set_format(device, width, height, fps):
    for f in device.formats():
        dims = CM.CMVideoFormatDescriptionGetDimensions(f.formatDescription())
        if (dims.width, dims.height) != (width, height):
            continue

        if not any(r.minFrameRate() <= fps <= r.maxFrameRate() for r in f.videoSupportedFrameRateRanges()):
            continue

        ok, err = device.lockForConfiguration_(None)
        if not ok:
            raise SystemExit(f"lockForConfiguration failed: {err}")

        device.setActiveFormat_(f)
        device.setActiveVideoMinFrameDuration_(CM.CMTimeMake(1, fps))
        device.setActiveVideoMaxFrameDuration_(CM.CMTimeMake(1, fps))
        device.unlockForConfiguration()
        return

    raise SystemExit(f"{device.localizedName()} has no {width}x{height}@{fps} format")


def _probe_avf(device, width, height, fps, frames):
    total = frames + WARMUP_FRAMES
    rec = {"arrival": [], "pts_s": [], "delivery_s": [], "copy": [], "dropped": 0,
           "target": total, "done": threading.Event(), "shapes": []}

    session = AVF.AVCaptureSession.alloc().init()
    dev_input, err = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
    if dev_input is None:
        raise SystemExit(f"Cannot open device: {err}")

    output = AVF.AVCaptureVideoDataOutput.alloc().init()
    output.setAlwaysDiscardsLateVideoFrames_(True)
    output.setVideoSettings_({Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA})
    delegate = _Delegate.alloc().init()
    delegate.rec = rec
    queue = dispatch_queue_create(b"vectrax.probe", DISPATCH_QUEUE_SERIAL)
    output.setSampleBufferDelegate_queue_(delegate, queue)

    session.beginConfiguration()
    session.addInput_(dev_input)
    session.addOutput_(output)
    session.commitConfiguration()

    t_open = time.monotonic_ns()
    with _CpuMeter() as cpu:
        session.startRunning()
        # macOS: startRunning re-applies the session preset and overrides
        # activeFormat; InputPriority preset is unsupported. Set it after.
        _set_format(device, width, height, fps)
        finished = rec["done"].wait(total / fps * 3 + 10)
        session.stopRunning()

    if not finished:
        raise SystemExit(f"Timed out after {len(rec['arrival'])}/{total} frames")

    arrival = rec["arrival"][WARMUP_FRAMES:]
    pts_ns = (np.asarray(rec["pts_s"][WARMUP_FRAMES:]) * NS_PER_S).astype(np.int64)
    delivery_ns = np.asarray(rec["delivery_s"][WARMUP_FRAMES:]) * NS_PER_S
    shapes = rec["shapes"][WARMUP_FRAMES:]
    return {
        "method": METHOD_AVF,
        "actual_shape": list(shapes[-1]),
        "shape_mismatch_frames": sum(s[:2] != (height, width) for s in shapes),
        "first_frame_ms": round((rec["arrival"][0] - t_open) / NS_PER_MS, 1),
        "arrival": _timing(arrival, fps),
        "sensor_pts_interval_ms": _stats_ms(np.diff(pts_ns)),
        "pts_to_delivery_ms": _stats_ms(delivery_ns),
        "copy_to_numpy_ms": _stats_ms(rec["copy"][WARMUP_FRAMES:]),
        "avf_dropped": rec["dropped"],
        "cpu_percent": cpu.percent,
    }


def _opencv_index(device):
    # ASSUMPTION: OpenCV's AVFoundation index follows AVF device order.
    # _verify_opencv_index checks it with a resolution only this device has.
    uids = [d.uniqueID() for d in _devices()]
    return uids.index(device.uniqueID())


def _verify_opencv_index(device, index):
    others = {(w, h) for d in _devices() if d.uniqueID() != device.uniqueID() for w, h, _ in _formats(d)}
    unique = [(w, h) for w, h, _ in _formats(device) if (w, h) not in others]
    if not unique:
        return "unverified (no unique resolution)"

    w, h = unique[0]
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ok, img = cap.read()
    cap.release()
    if ok and img.shape[1] == w and img.shape[0] == h:
        return f"verified via {w}x{h}"

    return f"MISMATCH: requested {w}x{h}, got {None if not ok else img.shape[1::-1]}"


def _probe_opencv(device, width, height, fps, frames):
    index = _opencv_index(device)
    check = _verify_opencv_index(device, index)

    t_open = time.monotonic_ns()
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise SystemExit(f"OpenCV cannot open index {index}")

    arrival, block = [], []
    first_ns = None
    shape = None
    with _CpuMeter() as cpu:
        for _ in range(frames + WARMUP_FRAMES):
            t0 = time.monotonic_ns()
            ok, img = cap.read()
            t1 = time.monotonic_ns()
            if not ok:
                break

            first_ns = first_ns or t1
            arrival.append(t1)
            block.append(t1 - t0)
            shape = img.shape

    cap.release()
    if len(arrival) <= WARMUP_FRAMES + 1:
        raise SystemExit("OpenCV delivered too few frames")

    return {
        "method": METHOD_OPENCV,
        "index": index,
        "index_check": check,
        "actual_shape": list(shape),
        "reported_fps": cap.get(cv2.CAP_PROP_FPS),
        "first_frame_ms": round((first_ns - t_open) / NS_PER_MS, 1),
        "arrival": _timing(arrival[WARMUP_FRAMES:], fps),
        "read_block_ms": _stats_ms(block[WARMUP_FRAMES:]),
        "cpu_percent": cpu.percent,
    }


def _env():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, check=False).stdout.splitlines()[0]
    return {
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "opencv": cv2.__version__,
        "git_sha": sha,
        "power": power,
        "center_stage": bool(AVF.AVCaptureDevice.isCenterStageEnabled()),
    }


def _print(result):
    a = result["arrival"]
    print(f"\n[{result['method']}] shape={result['actual_shape']} fps={a['effective_fps']} "
          f"gaps={a['gaps']} cpu={result['cpu_percent']}% first_frame={result['first_frame_ms']}ms")
    print(f"  arrival interval ms: {a['interval_ms']}")
    for key in ("sensor_pts_interval_ms", "pts_to_delivery_ms", "copy_to_numpy_ms", "read_block_ms"):
        if key in result:
            print(f"  {key}: {result[key]}")

    for key in ("avf_dropped", "shape_mismatch_frames", "index_check"):
        if key in result:
            print(f"  {key}: {result[key]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true")
    p.add_argument("--device", help="name substring or uniqueID")
    p.add_argument("--method", choices=[METHOD_AVF, METHOD_OPENCV, METHOD_BOTH], default=METHOD_BOTH)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--label", default="", help="free text, e.g. wired / wireless")
    args = p.parse_args()

    if args.list:
        _list()
        return

    if not args.device:
        p.error("--device or --list required")

    _ensure_permission()
    device = _find_device(args.device)
    report = {
        "device": {"name": device.localizedName(), "id": device.uniqueID(), "type": str(device.deviceType())},
        "label": args.label,
        "requested": {"width": args.width, "height": args.height, "fps": args.fps, "frames": args.frames},
        "env": _env(),
        "results": [],
    }
    print(f"Probing {device.localizedName()} {args.width}x{args.height}@{args.fps} ({args.label or 'no label'})")

    methods = [METHOD_AVF, METHOD_OPENCV] if args.method == METHOD_BOTH else [args.method]
    for m in methods:
        probe = _probe_avf if m == METHOD_AVF else _probe_opencv
        result = probe(device, args.width, args.height, args.fps, args.frames)
        report["results"].append(result)
        _print(result)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = device.localizedName().split()[0].lower()
    suffix = f"_{args.label}" if args.label else ""
    out = OUT_DIR / f"{slug}{suffix}_{args.width}x{args.height}@{args.fps}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
