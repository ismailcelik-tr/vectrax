# Fixtures

Ground-truth clips for every experiment (SPEC: Evaluation data). Recorded
on the MacBook camera, 1280x720 @ 30. Stored in `data/fixtures/`
(git-ignored, R6). Annotated in CVAT, exported as MOT 1.1.

Record: `uv run scripts/record_fixture.py --name <name> --seconds <s> --note "<text>"`

| Name | s | Content | Tests |
|---|---|---|---|
| `single_target` | 20 | Cup moved slowly across the desk | baseline propagation |
| `crossing_targets` | 20 | Two similar cups swap sides twice | ID switch |
| `near_targets` | 20 | Two similar cups approach and overlap, no swap | association under proximity |
| `occlusion` | 25 | Cup hidden behind a book 1–3 s, three times | OCCLUDED → TRACKING |
| `exit_reentry` | 25 | Owner walks out of frame, returns after ~3 s | LOST → reacquisition |
| `fast_motion` | 15 | Cup moved fast, with direction changes | propagator limits, blur |
| `non_coco` | 20 | Transparent food container moved slowly | R1: object outside detector classes |
| `low_light` | 20 | Dim light, cup moved slowly | auto exposure, fps drop |

Rules: camera fixed, nobody but the owner or consenting people in frame,
one clip per scenario; re-record rather than edit.

## Annotation (CVAT)
- One track per physical object, same ID for the whole clip.
- Keyframe every 10–15 frames and at every direction change; CVAT interpolates.
- Occluded or out of frame: mark the track `outside` for those frames.
- Partially visible: keep the box on the visible part, set `occluded`.
- Cut by the frame edge is truncation, not occlusion: no `occluded` flag.
- A normal grip (fingers on the object) is not occlusion.

## Export format (`scripts/cvat_tasks.py export <name>` → `<name>.gt.zip`)
MOT 1.1 `gt/gt.txt`: `frame,id,x,y,w,h,not_ignored,class,visibility`.
- Frames are 1-based: VectraX `frame_id = frame - 1`.
- Outside frames have no row. `visibility` 0.0 = occluded, 1.0 = visible.
- `class` indexes `gt/labels.txt` (1-based): cup, person, object.
