# Upstream provenance

- Source: [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi).
- Baseline commit: [`215abfb217dbac7d5f1273282331b9b1866c0479`](https://github.com/Physical-Intelligence/openpi/commit/215abfb217dbac7d5f1273282331b9b1866c0479).
- This repository is a derivative engineering implementation, not an official Physical Intelligence release.
- Changes add numeric tactile prefix and action-flow residual paths to PyTorch π0.5, with data/training/checkpoint adapters, tests, and documentation.
- LingBot-Touch inference code is vendored from `Robbyant/lingbot-touch@dec8025b212e7099b235cc2ef13a4921a7e59545`; forearm adaptation is maintained separately. Its Apache-2.0 license and third-party notices are retained under `src/openpi/models_pytorch/lingbot_touch/`.
- Original `LICENSE`, `LICENSE_GEMMA.txt`, source notices, and third-party references are retained.
- The original project README is retained in [`docs/UPSTREAM_README.md`](docs/UPSTREAM_README.md). Relative links inside that historical copy refer to the upstream project layout.
- The upstream workflows are retained as documentation; this repository runs a separate CPU component workflow.
- No model weights, real recordings, hardware calibration assets, or credentials are distributed here.
