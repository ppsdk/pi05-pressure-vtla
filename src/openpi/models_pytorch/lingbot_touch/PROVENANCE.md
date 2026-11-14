# Vendored LingBot-Touch inference core

Source: https://github.com/Robbyant/lingbot-touch

Revision: `dec8025b212e7099b235cc2ef13a4921a7e59545` (retrieved 2026-11-14).
The seven Python files are copied without modification from
`tactile_jepa_inference/`. Apache-2.0 LICENSE and upstream third-party notices
are retained here. No training code, sample data or pretrained weights are vendored.

Forearm adaptations live separately in `../lingbot_forearm.py`: numeric sensor
slots replace five fingers; real age and forearm XYZ replace the finger geometry
contract; attention receives per-batch dynamic validity; robot-state-conditioned
queries replace mean readout. The original parameter names/shapes are retained
inside `LingBotForearmEncoder.pretrained` for strict bundle initialization.
