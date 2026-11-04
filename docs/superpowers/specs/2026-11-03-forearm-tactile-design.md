# Configurable forearm tactile prefix

User scope: install tactile sensing on the forearm; implement now with configurable model parameters. Vendor interfaces and high-rate controllers are outside scope.

Keep the existing numeric patch MLP and pi05 prefix/action architecture. Configure sensor count, common grid shape, patch size, history, hidden width, age limit, names/locations, signal type/unit, and optional nominal sample period. Nominal period documents acquisition; actual age continues to determine temporal features.

Add optional per-taxel forearm-local XYZ coordinates in millimetres, shaped [S,R*C,3], row-major. A coordinate MLP encodes the coordinates of every taxel in each patch, scaled by a configurable positive length. Coordinates are optional; existing learned patch IDs remain the default. Coordinates from different sensors must use the documented common forearm frame.

Add optional adjacent-history differences, preserving absolute normalized values. A difference is available only when both endpoints are valid, nonsaturated, finite, and acquisition times strictly increase. Optional maximum gap rejects disconnected history. Missing pairs contribute zero and a false difference-valid flag. No prediction, future samples, or fabricated repeated observations.

Use a strict JSON PressureConfig loader and JSON-compatible serialization. TrainConfig consumes an optional pressure_config_path and resolves it into model.pressure, then clears the source path so deployment does not depend on that file. Normalization CLI accepts the same path. Add a separate pi05_pressure_forearm example: one forearm slot, placeholder 4x4 geometry, per-taxel tokens, four-frame history, optional differences enabled. This example is not a hardware specification.

All resolved config fields are checkpoint contract fields. Old configs without new fields are interpreted with defaults. Existing parameter names/shapes remain unchanged when geometry and differences are disabled. Mixed grid sizes and vector force channels are outside this implementation; pad/mask into a common grid or train separate configs.

Validation: CPU component regression, JSON/config rejection, physical-coordinate effects, temporal causality/masks/gradients, save/load metadata compatibility. Run compute through sbatch on the current H200 node, using its local Conda environment. No full official weights or real hardware claimed.

Approach: extend the established encoder rather than replace it with an image encoder or add a controller. Independent numeric tokens retain measurement magnitude and spatial structure with a small, configurable parameter footprint. The user requested direct implementation; optional research variants remain deferred.
