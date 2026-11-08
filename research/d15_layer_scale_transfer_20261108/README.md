# D15-related layer-scale intervention

This saved experiment studies a function-preserving parameterization change: multiply the first layer by c and divide the final layer by c, then optionally insert two trainable identity-linear factors. This is related to D15 layer-scale initialization questions. It tests one compensating scale intervention and does not cover the full ?/? initialization family.

It does not test the original D1 optimizer-specific effective-time coordinate U. The new proxy V is a separate initialization-only function-displacement measure.

## Results

- Preregistered source commit: `87a6839053384b781833bdfb349370617d329f4d` (historically created under the D1 exploration path; this migration corrects current classification without rewriting history).
- 5,184 candidate paths, 64 SGD references; raw trajectories remain in the original external archive `AIQ_KB_DATA/runs/d1_scaling_exploration_20261107/r002_parameterization/` for provenance.
- V had lower shared-point prediction error in 507/591 eligible scale-transfer pairs (85.79%).
- Four-weight-layer holdout: V covered 267/288 paths; median complete-path MAE 12.88%, failing the registered 1% threshold.
- Full formulation, settings, figures and limitations: `docs/????/D15_layer_scale_transfer.html`.

## Reproduction from saved arrays

Run from the repository root: `python -X utf8 research/d15_layer_scale_transfer_20261108/analyze_r002.py`, then `verify_r002.py`, `plot_r002.py`, and `render_r002_report.py`. These consume saved external arrays and do not train. The training entry point and frozen source hashes are preserved from the preregistered run.
