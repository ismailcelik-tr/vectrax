# VectraX

Spec: docs/SPEC.md. Decisions: docs/DECISIONS.md. Progress: docs/ROADMAP.md.

## Rules
- Never write a performance or quality number you did not measure in this
  repo. Every number in docs/PERFORMANCE.md cites the command and raw output file.
- Decisions without local evidence are marked PROVISIONAL.
- Real-time path: no database, no network, never blocks on inference.
- Models and trackers return Observations. Only TrackManager creates,
  transitions or deletes tracks and assigns IDs.
- All time comes from the injected Clock (monotonic ns). No time.time().
- You cannot see the camera or click the UI. Verify behavior with
  data/fixtures clips in deterministic mode. Ask the owner to run camera scripts.
- Algorithmic code: failing test first. Perf measurements live in
  benchmarks/, never in tests/.
- Do not create modules, docs or abstractions before the phase that needs them.

## Commands
- uv run pytest
- uv run ruff check .
