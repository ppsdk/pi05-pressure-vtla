# Validation record

Initial validation date: 2026-10-29. Upstream baseline: `215abfb217dbac7d5f1273282331b9b1866c0479`.

## Local evidence

- Windows, Python 3.11.1, PyTorch 2.7.1+cpu, NumPy 1.26.4.
- `torch.cuda.is_available()` returned `False`.
- `python -m pytest tests/vtla -q`: 93 passed in 9.52s.
- Ruff check and format check: passed for the 21 modified/new Python files.
- Python compilation and `git diff --check`: passed.
- At that revision, AST comparison against the upstream commit: `embed_suffix`, `denoise_step`, noise/time sampling, and the flow interpolation/target expressions are unchanged.

Subsequent CPU component validation after the checkpoint and image-resize fixes: `109 passed in 15.61s` locally on Windows. A GitHub-hosted Linux CPU integration run passed the full synthetic-data → real small openpi model → optimizer → final checkpoint → policy inference → resume path. With fixed action noise, changing valid pressure changed predicted actions; after resuming, pressure-encoder weights changed. The smoke uses a randomly initialized model and a locally generated test tokenizer; see the CPU integration workflow for the latest result.

## What the tests establish

The tests cover NPZ reading, causal alignment, fixed signal normalization, missing/bad/saturated taxels, pressure tokens, prefix masks and positions, gradients, base initialization, strict VTLA restoration, and checkpoint asset integrity.

The integration harness calls real `PI0Pytorch` methods with a small two-stream attention test double. It checks the integration's joint/cached dataflow without loading the official backbone or downloading weights.

## Not established

No full official π0.5 checkpoint was loaded or trained. No real sensor recording, device calibration, robot task result, full-backbone numerical parity, or closed-loop latency was measured. The example sensor geometry is not a hardware specification.

The component workflow does not install the complete upstream stack. A separate CPU integration workflow installs the dependencies needed by the small real-model/data pipeline. Neither workflow validates official π0.5 weights or real hardware. Original upstream workflows are retained in `docs/upstream-workflows/` for reference.

## 2026-11-03: configurable forearm tactile prefix

- Node: h200-ali-01, CPU-only Slurm batch jobs (no GPU requested). Conda Python 3.11.16, torch 2.7.1+cu128; CPU tensors used throughout.
- Final components: job 52434, `python -m pytest tests/vtla -q`: 148 passed in 2.35s.
- Final real tiny-model pipeline and config/CLI tests: job 52452, `python -m pytest tests/pressure_integration -q`: 4 passed in 14.38s. This smoke enables forearm XYZ geometry and temporal differences, trains a random tiny model, saves final checkpoint, deploys without original training manifest, samples actions, resumes optimization, and verifies changed pressure affects output.
- Ruff: job 52453, all nine changed/new Python files passed. Format: job 52454, nine files already formatted. `git diff --check` passed.
- RED-to-GREEN regressions cover JSON fields and strictness, physical-coordinate effects and patch ordering, causal differences and masks, finite gradients, metadata/legacy defaults, consumed external config paths, tiny scale overflow, and accidental string-to-slot conversion.
- A fresh read-only code review identified coordinate-scale overflow and malformed sensor-name strings; both were fixed with regression tests.
- Bare upstream `pytest -q --maxfail=5`: job 52398 could not collect `src/openpi` because `pynvml` is absent. This is separate from the passing VTLA/component and real tiny-model integration suites.
- Dependencies were installed only in an isolated validation environment over the existing local Conda runtime. No official pi05 weights, real sensor measurements, real robot performance or closed-loop latency were validated.

Example: `examples/pressure/forearm.json`. The 4x4 grid, one forearm slot, and four-frame history are configurable placeholders, not confirmed hardware parameters. Common scalar grid input is supported; mixed sizes require explicit common-grid padding/masks or separate configurations.

## 2026-11-14: LingBot-Touch forearm adaptation

- Windows CPU, Python 3.11, PyTorch 2.7.1+cpu, NumPy 1.26.4.
- `python -m pytest tests/vtla -q`: 166 passed. The suite includes dynamic missingness, spatial-swap sensitivity, signed resistance calibration, frozen-backbone gradients, full checkpoints, and real PI0 residual methods with a test-double VLA.
- The official Tiny bundle from ModelScope contains 5,455,488 encoder parameters. Its `encoder.pt` SHA256 is `ddb8b435963348fff5b8a83bb48f995978dc8841d82f87be007c775bf5a88055`.
- `scripts/check_lingbot_forearm.py` ran with that bundle and synthetic signals/geometry. With the added skin embedding disabled and a matching single-finger reference, the maximum absolute forward difference was `1.8477439880371094e-06`. Output shape was `[1,6,2048]`.
- Frozen-backbone adapter gradients, unfrozen pretrained-block gradients, and self-contained safetensors restoration passed. Ruff, format, compilation and whitespace checks passed for the changed project Python files.
- Forearm XYZ, skin identity and real-age encoding replace the hand-specific input convention. These changes require task adaptation; the reference comparison establishes weight/operation reuse, not zero-shot forearm accuracy.
- The full training CLI and real tiny-PI0 pipeline were not run for this change. The local component environment lacks JAX, Flax, Transformers and Tyro. No official full π0.5 weights, real robot training, GPU training or latency measurement were used.

The current residual path adds a tactile correction in `denoise_step`; the original 2026-10-29 AST statement above applies to the earlier prefix-only revision. Noise/time sampling and the Flow Matching target remain unchanged.

## Training guidance

[The two-stage guide](lingbot-forearm.zh-CN.md#两阶段微调) describes frozen-Tiny adapter training, optional pretrained-encoder fine-tuning, initialization, and same-stage resume. Suggested learning rates and step counts are starting points without real-data validation. They are not measured task results.

## 2026-11-15: training contracts and Windows integration

- Windows, isolated `.venv-integration`, Python 3.11.1, torch 2.7.1+cpu, NumPy 1.26.4, JAX 0.5.3, Flax 0.10.2 and Transformers 4.53.2 with the repository replacement files. CUDA was unavailable.
- `python -m pytest tests/vtla -q`: 185 passed in 11.24s.
- `python -m pytest tests/pressure_integration -q --tb=short`: 9 passed in 57.53s. Run these two suites separately; their existing module names overlap under pytest's default import mode.
- The real small-model pipeline now covers both the MLP prefix and the LingBot residual. Synthetic NPZ, a local test SentencePiece vocabulary and random small model weights exercise statistics generation, actual model training, final checkpoint, deployment without training assets, and same-stage resume.
- The LingBot case checks that stage one leaves pretrained weights unchanged. Stage two loads the selected first-stage checkpoint, updates the unfrozen encoder, preserves base-policy weights, and resumes with the pretrained/adaptor LR ratio still 0.1.
- New component checks reject same-shaped hardware/calibration/statistics drift, missing full-VTLA metadata, bundle reinitialization, changed optimizer groups/names/scales/settings, and accidental experiment-directory reuse. Plain pi0/pi05 initialization and legacy optimizer loading remain supported; legacy optimizers emit a warning because parameter names were not recorded.
- Initial integration runs exposed Windows drive paths being mistaken for URL schemes, and NumPy LR scalars preventing restricted optimizer-file loading. Both were fixed, then the affected suites passed. New optimizer LR values are Python floats; legacy optimizer/metadata loading behavior is retained.
- Ruff lint and format checks passed for the 12 changed/new Python files; compilation and `git diff --check` passed. The CPU integration workflow now includes Linux and Windows runners; local results do not establish hosted workflow status.

This run does not use official full pi05 weights, real sensor measurements or robot demonstrations. It does not establish task quality, GPU/DDP behavior or latency. Resume restores model, optimizer and step; data-iterator position and all RNG states are not restored, so whole-training bitwise continuation is not claimed.
