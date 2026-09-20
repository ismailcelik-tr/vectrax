# VectraX — Specification

## Purpose
Real-time, operator-driven visual tracking and camera guidance for
non-weaponized use (monitoring, robotics, inspection, research). The operator
chooses targets. v1 never actuates hardware; guidance is displayed only.

## Requirements (answered 2026-09-19)
- R1 Target domain: arbitrary operator-boxed objects. Class-agnostic
  Propagator (SOT) is the primary path; detector classes are hints only.
- R2 Scene: indoor desk/room, near range, moderate motion.
- R3 Latency goal at 30 fps, 720p (ADR-003, ADR-007):
  - R3a p95 sensor PTS → guidance ≤ 100 ms (system; also future PTZ).
  - R3b p95 sensor PTS → render ≤ 120 ms (operator display; renderer
    will change).
- R4 Targets: v1 meets R3 with 1–3 targets. 10+ is the next milestone,
  scoped by measured per-target cost.
- R5 License: research now, possibly commercial later. Runtime
  dependencies must be commercial-safe (Apache/MIT/BSD). AGPL models and
  non-commercial datasets are allowed only as benchmark references.
- R6 Recording privacy: local only (data/, git-ignored), no blurring.
  Record only the owner and consenting people.

## Non-goals (v1)
Physical camera control. Weapon/engagement logic: out of scope for v1 and
deliberately left open beyond it — the owner may widen it, and that takes its
own ADR covering authorization, accountability and a safety review, not a
line edit here. Perception (detect, track, warn) carries no such condition.
Microservices, brokers, databases in the real-time path. Browser-side
perception. Anomaly detection and dedicated classifiers until a use case with
acceptance criteria exists.

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
FramePacket     {frame_id, source_id, capture_ns (sensor PTS), arrival_ns, image, w, h, pixel_format}
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
capture_ns is the sensor PTS on the host monotonic clock (ADR-003); it may
still lag exposure. Measure glass-to-glass once externally (film a ms
clock) and record it.

## Evaluation data
data/fixtures/: short clips recorded by the owner on this camera: single
target, two crossing targets, 1–3 s full occlusion, exit and re-entry, fast
motion, low light. Annotated in CVAT (MOT format). Ground truth for every
experiment. Public datasets only if R5 permits.

## Model evaluation
Detectors (small matrix): one Ultralytics nano (YOLO11n or YOLO26n) as
reference only (R5); Apache-2.0 candidates (RF-DETR nano, D-FINE-N or
YOLOX-nano) for runtime; a class-agnostic or open-vocabulary option for
reacquisition of arbitrary objects (R1), license checked before use.
Backends: PyTorch CPU, PyTorch MPS, Core ML (CPU/GPU/ANE compute units),
ONNX Runtime CPU + CoreML EP. Export/compile time reported separately.
Quality: mAP on a fixed COCO val subset (sanity) + precision/recall on
fixtures. Tracking: HOTA, IDF1, ID switches (TrackEval) on fixtures.
Re-ID ladder: none → color histogram → generic embedding (e.g. DINOv2-S).
OSNet is out (person-only, R1).
Protocol: on AC power, Low Power Mode off, warm-up excluded, sustained runs
long enough to expose throttling; record macOS, Python, package versions,
git SHA. Accelerator usage via `macmon` (no sudo).
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
