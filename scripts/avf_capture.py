"""AVFoundation capture shared by Phase 0 scripts. Prototype of MacCamera."""

import threading
import time

import AVFoundation as AVF
import CoreMedia as CM
import numpy as np
import objc
import Quartz
from Foundation import NSObject
from libdispatch import DISPATCH_QUEUE_SERIAL, dispatch_queue_create

__all__ = ["AvfCapture", "ensure_permission", "find_device"]

BGRA_CHANNELS = 4
NS_PER_S = 1_000_000_000
PERMISSION_TIMEOUT_S = 60


def _devices():
    return list(AVF.AVCaptureDevice.devicesWithMediaType_(AVF.AVMediaTypeVideo))


def find_device(query):
    q = query.lower()
    matches = [d for d in _devices() if q in d.localizedName().lower() or query == d.uniqueID()]
    if len(matches) != 1:
        raise SystemExit(f"'{query}' matched {len(matches)} devices")

    return matches[0]


def ensure_permission():
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


class _Delegate(NSObject, protocols=[objc.protocolNamed("AVCaptureVideoDataOutputSampleBufferDelegate")]):
    def captureOutput_didOutputSampleBuffer_fromConnection_(self, output, sample, connection):
        arrival_ns = time.monotonic_ns()
        host_now_s = CM.CMTimeGetSeconds(CM.CMClockGetTime(CM.CMClockGetHostTimeClock()))
        pts_s = CM.CMTimeGetSeconds(CM.CMSampleBufferGetPresentationTimeStamp(sample))
        # PTS is on the CoreMedia host clock; express it on time.monotonic_ns.
        capture_ns = arrival_ns - int((host_now_s - pts_s) * NS_PER_S)

        pix = CM.CMSampleBufferGetImageBuffer(sample)
        Quartz.CVPixelBufferLockBaseAddress(pix, Quartz.kCVPixelBufferLock_ReadOnly)
        try:
            h = Quartz.CVPixelBufferGetHeight(pix)
            w = Quartz.CVPixelBufferGetWidth(pix)
            bpr = Quartz.CVPixelBufferGetBytesPerRow(pix)
            raw = Quartz.CVPixelBufferGetBaseAddress(pix).as_buffer(bpr * h)
            view = np.frombuffer(raw, np.uint8).reshape(h, bpr)[:, : w * BGRA_CHANNELS].reshape(h, w, BGRA_CHANNELS)
            self.on_frame(view, capture_ns, arrival_ns)
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pix, Quartz.kCVPixelBufferLock_ReadOnly)

    def captureOutput_didDropSampleBuffer_fromConnection_(self, output, sample, connection):
        self.dropped += 1


class AvfCapture:
    """on_frame(bgra_view, capture_ns, arrival_ns) runs on the capture queue.
    The view is valid only during the call; copy to keep it."""

    def __init__(self, device, width, height, fps, on_frame):
        self._device = device
        self._size = (width, height)
        self._fps = fps
        self._session = AVF.AVCaptureSession.alloc().init()
        dev_input, err = AVF.AVCaptureDeviceInput.deviceInputWithDevice_error_(device, None)
        if dev_input is None:
            raise SystemExit(f"Cannot open device: {err}")

        output = AVF.AVCaptureVideoDataOutput.alloc().init()
        output.setAlwaysDiscardsLateVideoFrames_(True)
        output.setVideoSettings_({Quartz.kCVPixelBufferPixelFormatTypeKey: Quartz.kCVPixelFormatType_32BGRA})
        self._delegate = _Delegate.alloc().init()
        self._delegate.on_frame = on_frame
        self._delegate.dropped = 0
        self._queue = dispatch_queue_create(b"vectrax.capture", DISPATCH_QUEUE_SERIAL)
        output.setSampleBufferDelegate_queue_(self._delegate, self._queue)

        self._session.beginConfiguration()
        self._session.addInput_(dev_input)
        self._session.addOutput_(output)
        self._session.commitConfiguration()

    @property
    def dropped(self):
        return self._delegate.dropped

    def start(self):
        self._session.startRunning()
        # startRunning re-applies the session preset (ADR-001); set format after.
        self._set_format()

    def stop(self):
        self._session.stopRunning()

    def _set_format(self):
        width, height = self._size
        for f in self._device.formats():
            dims = CM.CMVideoFormatDescriptionGetDimensions(f.formatDescription())
            if (dims.width, dims.height) != (width, height):
                continue

            if not any(r.minFrameRate() <= self._fps <= r.maxFrameRate() for r in f.videoSupportedFrameRateRanges()):
                continue

            ok, err = self._device.lockForConfiguration_(None)
            if not ok:
                raise SystemExit(f"lockForConfiguration failed: {err}")

            self._device.setActiveFormat_(f)
            self._device.setActiveVideoMinFrameDuration_(CM.CMTimeMake(1, self._fps))
            self._device.setActiveVideoMaxFrameDuration_(CM.CMTimeMake(1, self._fps))
            self._device.unlockForConfiguration()
            return

        raise SystemExit(f"{self._device.localizedName()} has no {width}x{height}@{self._fps} format")
