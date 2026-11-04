# Forearm Tactile Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans; execute directly in this session as requested by the user.

**Goal:** Make the forearm tactile prefix configurable without vendor-specific code.
**Architecture:** Extend PressureConfig and PressureEncoder, preserve pi05 prefix/Flow Matching, resolve optional JSON at config construction.
**Tech Stack:** Python, NumPy, PyTorch, pytest.
**Spec:** docs/superpowers/specs/2026-11-03-forearm-tactile-design.md

## Global Constraints
- Use local Conda Python and CPU-only sbatch validation on h200-ali-01.
- Optional features disabled preserve old encoder state shapes and outputs.
- Preserve current-value information and invalid/saturated metadata.
- No vendor interfaces, high-rate controller, mixed-grid or vector-force implementation.

## Review Focus
- Malformed/nonfinite coordinates must fail before model construction.
- Missing/stale/reversed-time/saturated history cannot fabricate deltas.
- Invalid values cannot poison tokens or gradients.
- Checkpoint JSON nested tuples/lists must compare equivalently; semantic changes must fail.
- Resolved config must survive deployment without source JSON.

### Task 1: Config and checkpoint contracts
Files: pressure_config.py, pressure_training.py, tests/vtla/test_forearm_config.py, tests/vtla/test_pressure_training.py.
- [x] Add failing JSON roundtrip/validation and legacy-metadata tests; run and inspect failures.
- [x] Implement optional geometry, signal metadata, timing, differences fields; strict JSON loader and canonical serializer.
- [x] Normalize saved legacy metadata through default config fields; enforce mismatches.
- [x] Run component suite.

### Task 2: Geometry and temporal features
Files: pressure_pytorch.py, tests/vtla/test_forearm_encoder.py.
- [x] Add failing tests for geometry effects, actual-time differences, masks, saturation, and gradients.
- [x] Add optional patch coordinate MLP and difference/value-availability input channels.
- [x] Preserve current inputs, default parameters, and existing mask logic.
- [x] Run full component suite.

### Task 3: Training configuration and usage
Files: training/config.py, scripts/compute_norm_stats.py, examples/pressure/forearm.json, README_VTLA.zh-CN.md, tests/pressure_integration/test_forearm_config.py.
- [x] Test config resolution and rejection through real config objects where dependencies exist.
- [x] Add consumed pressure_config_path to TrainConfig, get_config, and normalization CLI; add forearm preset.
- [x] Document single-forearm JSON, common grid constraint, physical coordinates, delta semantics, and CLI usage.
- [x] Run lint, component suite, integration availability check, and diff review. Record exact evidence and limitations.

## Validation and review evidence

- Baseline: Slurm 52364, 111 component tests passed.
- Config RED: 52369, expected missing fields/loader failures; GREEN 52370, 129 passed.
- Encoder RED: 52373, expected missing delta channels and coordinate effect; backward mask fix validated in 52379 (141 passed, two pending resolver tests failed as expected).
- Resolver RED: 52379 missing resolver; GREEN 52386, 143 component tests passed.
- Full pipeline: 52396, four integration tests passed, including forearm XYZ+deltas training/save/resume/infer and actual CLI JSON resolution.
- Review: fresh read-only reviewer requested by requesting-code-review. Found tiny coordinate scale float32 overflow; regression RED 52412, fixed by bounded scale. None scale regression RED 52421, fixed by explicit rejection.
- Review decisions: common-grid/vector-channel constraints and controller/vendor/full-weight work remain outside scope as stated in the spec. These do not affect the provided scalar forearm model contract.
- Bare upstream pytest: 52398 collection error `src/openpi/conftest.py: ModuleNotFoundError: pynvml`; component/integration results above remain independently verified.
- Final verification recorded in docs/VALIDATION.md after last run. No model training beyond random tiny CPU integration fixtures; no GPU allocation needed.
- Final: 52434 components 148 passed; 52452 integration 4 passed; 52453 Ruff passed; 52454 nine files formatted; git diff --check passed.
- Final review: string metadata cannot be interpreted character-by-character as sensor slots; RED 52431, fixed and included in final component run. No unresolved review findings.
- Implementation stayed in the requested checkout on feature branch feat/forearm-tactile-config; no worktree, merge, push, or notification was needed.
