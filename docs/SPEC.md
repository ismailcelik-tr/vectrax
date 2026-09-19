# VectraX — Specification

## Purpose
Real-time, operator-driven visual tracking and camera guidance for
non-weaponized use (monitoring, robotics, inspection, research). The operator
chooses targets. v1 never actuates hardware; guidance is displayed only.

## Open requirements — owner answers before Phase 2
- R1 Target domain: people / vehicles / arbitrary operator-boxed objects.
  Decides detector class set, open-vocabulary need, SOT vs MOT emphasis.
- R2 Scene: indoor/outdoor, target size in pixels, motion speed.
- R3 Latency goal: p95 capture→guidance ≤ ? ms at ? fps, ? resolution.
- R4 Max simultaneous targets.
- R5 License: research-only or commercial? (Ultralytics = AGPL-3.0;
  MOT17/MOT20 = non-commercial.)
- R6 Recording privacy: retention, storage location, face blurring.
Decisions depending on an unanswered R are PROVISIONAL.

## Non-goals (v1)
Physical camera control. Weapon/engagement logic. Microservices, brokers,
databases in the real-time path. Browser-side perception. Anomaly detection
and dedicated classifiers until a use case with acceptance criteria exists.

## Pipeline
CameraSource → FramePacket → LatestFrameBuffer (bounded, drop-oldest)
→ pipeline tick:
  1. Propagate each track (cheap, every frame)
  2. Merge finished async inference results (late-result fusion)
  3. TrackManager: associate, update quality, run state machine
  4. Predict, FOV analysis, guidance
  5. Publish TrackSnapshot → UI, recorder, metrics
InferenceScheduler runs models on a worker; the tick never waits on it.

## Fixed architectural decisions
- Two tracking roles:
  - Propagator: per-target, per-frame, class-agnostic. Candidates: OpenCV
    CSRT, NanoTrack, ViTTrack, optical flow, Kalman-only.
  - Associator: runs when detections arrive. Candidates: ByteTrack,
    OC-SORT, BoT-SORT association. These assume per-frame detections;
    running them with a sparse detector is an experiment, not a given.
- One state estimator (Kalman) per track, owned by TrackManager. Third-party
  trackers are used for association only, not as a second state owner.
- Late results: every result carries its input frame_id and capture_ns. It
  is associated against the track state at that time (short per-track state
  history) and rolled forward. Results older than max_result_age are
  dropped and counted.
- Clock is injected. Live = monotonic system clock; replay/tests = virtual.
- Run modes: realtime (drops frames, async inference) and deterministic
  (every frame, synchronous inference, reproducible). Evaluation uses
  deterministic.
- Tracks live in normalized image coordinates. CameraModel converts to
  angles. Guidance reports degrees only when the camera is calibrated;
  otherwise direction + urgency.
- Concurrency: capture thread, pipeline thread, one inference worker.
  Move to processes only if GIL contention is measured.
- UI v1: local OpenCV window with click and box selection (macOS: GUI on main
  thread). A web UI later consumes the same TrackSnapshot stream.
- Sources: CameraSource interface; implement MacCamera and FileSource now.
  USB (UVC) and RTSP are later implementations of the same interface.
- Future CUDA/TensorRT/edge: behind the model Backend interface only.

## Data structures (starting point, refine in code)
FramePacket     {frame_id, source_id, capture_ns, source_pts?, image, w, h, pixel_format}
Observation     {frame_id, capture_ns, bbox, score, class_id?, embedding?, origin}
TrackState      {track_id, state, class_id?, bbox, kf_mean, kf_cov, quality,
                 history, last_obs_ns}
TrackQuality    {det_score?, propagator_score, motion_residual,
                 appearance_sim?, frames_since_obs, cov_trace}
Guidance        {track_id, direction, magnitude_deg?, urgency, time_to_exit_s?, confidence}
Event           {type, track_id?, frame_id, ns, payload}
InferenceRecord {model, backend, trigger, input_frame_id, roi?, start_ns, end_ns}

## Track state machine
INITIALIZING, TRACKING, DEGRADED, OCCLUDED, REACQUIRING, LOST, PAUSED, STOPPED.
Guards are named, config-driven thresholds. Operator commands (pause, resume,
stop, reselect, request reacquire) are transitions. Every transition has a
unit test. Reacquisition never takes an ID owned by an active track;
ambiguous candidates keep the track in REACQUIRING.

## Timing
Stamps: capture, preprocess, infer_start/end, track, predict, guide, render.
Report p50/p95/p99/max, frame age at render, queue depth, drops.
capture_ns is arrival time, not exposure time. Measure glass-to-glass once
externally (film a ms clock) and record it.

## Evaluation data
data/fixtures/: short clips recorded by the owner on this camera: single
target, two crossing targets, 1–3 s full occlusion, exit and re-entry, fast
motion, low light. Annotated in CVAT (MOT format). Ground truth for every
experiment. Public datasets only if R5 permits.

## Model evaluation
Detectors (small matrix): one Ultralytics nano (YOLO11n or YOLO26n) as
reference; one Apache-2.0 alternative (RF-DETR nano, D-FINE-N or YOLOX-nano);
open-vocabulary (YOLOE/YOLO-World) only if R1 = arbitrary objects.
Backends: PyTorch CPU, PyTorch MPS, Core ML (CPU/GPU/ANE compute units),
ONNX Runtime CPU + CoreML EP. Export/compile time reported separately.
Quality: mAP on a fixed COCO val subset (sanity) + precision/recall on
fixtures. Tracking: HOTA, IDF1, ID switches (TrackEval) on fixtures.
Re-ID ladder: none → color histogram → generic embedding (e.g. DINOv2-S).
OSNet only if R1 = people.
Protocol: on AC power, Low Power Mode off, warm-up excluded, sustained runs
long enough to expose throttling; record macOS, Python, package versions,
git SHA. Accelerator usage needs `powermetrics` (sudo): ask the owner.
Vendor numbers are reference only.

## Mac camera
Disable Center Stage. Log actual FPS (auto exposure drops it in low light).
Device indices are unstable (Continuity Camera). Terminal needs camera access.

## Phases (each ends: tests green, numbers measured, ADR entry, commit)
0 Environment + camera probe + fixture recording and annotation.
1 Walking skeleton: FileSource, MacCamera, FramePacket, Clock, bounded
  buffer, deterministic mode, metrics, OpenCV UI, box selection, one
  Propagator, TrackManager + state machine. No detector.
2 Detector/backend benchmark harness → ADR choosing detector.
3 Scheduler, async inference, late-result fusion, Associator, multi-target.
4 Occlusion, reacquisition scoring, Re-ID ladder experiments.
5 Prediction, FOV exit, guidance, checkerboard calibration.
6 Session recording, evaluation runner, reports.
7 Only with a use case: classification, anomaly detection, web UI, PTZ.

## Docs
ARCHITECTURE.md, DECISIONS.md (ADR log), PERFORMANCE.md (measured only),
ROADMAP.md. Subsystem docs are written when the subsystem is built.
