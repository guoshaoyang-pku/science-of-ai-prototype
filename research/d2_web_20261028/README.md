# D2 readable web report — 2026-10-28

This is a presentation update, not a new experiment. It condenses the original
D2 page and adds the saved diagonal-ablation and five-percent selection results.
The default page text is reduced by about 30%; task-level results, formulas and
historical failures remain available in disclosures and linked evidence.

## Rebuild

From the repository root, supply the external data root containing the completed
runs (do not start the research runners):

```sh
.venv/bin/python research/d2_web_20261028/build_page.py --data-root /path/to/AIQ_KB_DATA
.venv/bin/python research/d2_web_20261028/render_charts.py --plot-data research/d2_web_20261028/plot_data.json
```

The page builder verifies eight exported files against the committed result
manifests before copying them byte-for-byte. `export_manifest.json` records each
source-relative path and hash. It never edits old runs or archived evidence.

The portable plot data was extracted from saved summary, row and focus artifacts.
The first render independently recounted the 45,360 saved condition-method rows.
Its input hashes and display-only curve sampling are recorded in `plot_data.json`.
No minima are recomputed by the page renderer.

## Interpretation

- Pareto axes: retained cross-term budget ceiling versus stopping-quality rates.
  Only 0%, 5%, and 100% are measured; connecting lines are reading guides.
- Strict flooring means actual retention may be below 5%, including zero.
- Full reconstruction defines the reference, rather than being an independent
  successful predictor. Retained size is not runtime or training cost.
- Old scaling-law failures and new sparse-model successes have different tests
  and denominators. The page explicitly distinguishes them.
- All results remain development evidence. The methods use clean targets and
  validation geometry; no train-only, learned-feature, or speedup claim is made.

## Validation

Desktop and 390-pixel mobile previews: MathJax rendered without errors, both
images loaded, disclosure table contains all twelve recipes, no document overflow.
Local href/src targets checked. Independent content audit corrected the Pareto
dominance definition and verified formulas, counts, task shapes and source pins.
Repository suite: 113 tests and 19 subtests passed. Handoff: 2590 files verified.
Only the D2 page, its new figures, supporting exports, and this build directory
are changed by the presentation update; homepage and shared stylesheet are intact.
