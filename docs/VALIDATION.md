# Validation record

Date: 2026-10-29. Upstream baseline: `215abfb217dbac7d5f1273282331b9b1866c0479`.

## Local evidence

- Windows, Python 3.11.1, PyTorch 2.7.1+cpu, NumPy 1.26.4.
- `torch.cuda.is_available()` returned `False`.
- `python -m pytest tests/vtla -q`: **93 passed in 9.52s**.
- Ruff check and format check: passed for the 21 modified/new Python files.
- Python compilation and `git diff --check`: passed.
- AST comparison against the upstream commit: `embed_suffix`, `denoise_step`, noise/time sampling, and the flow interpolation/target expressions are unchanged.

Subsequent CPU component validation after the checkpoint and image-resize fixes: `109 passed in 15.61s` locally on Windows. A GitHub-hosted Linux CPU integration run passed the full synthetic-data → real small openpi model → optimizer → final checkpoint → policy inference → resume path. With fixed action noise, changing valid pressure changed predicted actions; after resuming, pressure-encoder weights changed. The smoke uses a randomly initialized model and a locally generated test tokenizer; see the CPU integration workflow for the latest result.

## What the tests establish

NPZ reading and causal alignment, fixed signal normalization, missing/bad/saturated taxel handling, pressure tokens and validity, prefix masks and positions, finite gradients to the pressure branch, controlled base initialization, strict VTLA restoration, and checkpoint asset integrity.

The integration harness calls real `PI0Pytorch` methods with a small two-stream attention test double. It checks the integration's joint/cached dataflow without loading the official backbone or downloading weights.

## Not established

No full official π0.5 checkpoint was loaded or trained. No real sensor recording, device calibration, robot task result, full-backbone numerical parity, or closed-loop latency was measured. The example sensor geometry is not a hardware specification.

The component workflow does not install the complete upstream stack. A separate CPU integration workflow installs the dependencies needed by the small real-model/data pipeline. Neither workflow validates official π0.5 weights or real hardware. Original upstream workflows are retained in `docs/upstream-workflows/` for reference.

## 2026-11-03: configurable forearm tactile prefix

- Node: h200-ali-01, CPU-only Slurm batch jobs (no GPU requested). Conda Python 3.11.16, torch 2.7.1+cu128; CPU tensors used throughout.
- Final components: job 52434, `python -m pytest tests/vtla -q`: **148 passed in 2.35s**.
- Final real tiny-model pipeline and config/CLI tests: job 52452, `python -m pytest tests/pressure_integration -q`: **4 passed in 14.38s**. This smoke enables forearm XYZ geometry and temporal differences, trains a random tiny model, saves final checkpoint, deploys without original training manifest, samples actions, resumes optimization, and verifies changed pressure affects output.
- Ruff: job 52453, all nine changed/new Python files passed. Format: job 52454, nine files already formatted. `git diff --check` passed.
- RED-to-GREEN regressions cover JSON fields and strictness, physical-coordinate effects and patch ordering, causal differences and masks, finite gradients, metadata/legacy defaults, consumed external config paths, tiny scale overflow, and accidental string-to-slot conversion.
- A fresh read-only code review identified coordinate-scale overflow and malformed sensor-name strings; both were fixed with regression tests.
- Bare upstream `pytest -q --maxfail=5`: job 52398 could not collect `src/openpi` because `pynvml` is absent. This is separate from the passing VTLA/component and real tiny-model integration suites.
- Dependencies were installed only in an isolated validation environment over the existing local Conda runtime. No official pi05 weights, real sensor measurements, real robot performance or closed-loop latency were validated.

Example: `examples/pressure/forearm.json`. The 4x4 grid, one forearm slot, and four-frame history are configurable placeholders, not confirmed hardware parameters. Common scalar grid input is supported; mixed sizes require explicit common-grid padding/masks or separate configurations.
