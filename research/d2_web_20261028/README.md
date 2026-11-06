# D2 readable web report — updated 2026-10-29

This is a presentation update, not a new experiment. The report now starts with
why generalization can worsen, then separates refuted original predictions,
conditional mathematical facts, development ablations, failures and open questions.
The homepage marks the original D2 claims as refuted without denying the observed
phenomenon or the existing five-percent development result. Historical report and
literature entrances point back to the current status; archived measurements stay intact.

## Rebuild

From the repository root, supply the external data root containing the completed
runs (do not start the research runners):

```sh
.venv/bin/python research/d2_web_20261028/build_page.py --data-root /path/to/AIQ_KB_DATA
.venv/bin/python research/d2_web_20261028/render_charts.py --plot-data research/d2_web_20261028/plot_data.json
.venv/bin/python research/d2_web_20261028/render_refutation.py
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

## Previous validation (2026-10-28)

Desktop and 390-pixel mobile previews: MathJax rendered without errors, both
images loaded, disclosure table contains all twelve recipes, no document overflow.
Local href/src targets checked. Independent content audit corrected the Pareto
dominance definition and verified formulas, counts, task shapes and source pins.
Repository suite: 113 tests and 19 subtests passed. Handoff: 2590 files verified.
Only the D2 page, its new figures, supporting exports, and this build directory
are changed by the presentation update; homepage and shared stylesheet are intact.

## 2026-10-29 evidence presentation

`render_refutation.py` reads the saved twelve per-recipe expanded scaling counts
without fitting or recomputing trajectories. `refutation_plot_data.json` pins the
source hash and plotted counts. It distinguishes the 1232/2700 expanded test from
the original 57/60 leave-sample-size-out test. The existing Pareto and focus plots
remain unchanged, with their measured scope and limitations explicitly retained.

Current content distinguishes refuted predictions, counterexamples to stronger
interpretations, and claims not yet established. The next causal-mechanism study
is only proposed; no new experiment has been preregistered or started. The homepage
change is limited to the existing D2 entry and its update date. Upstream D14 and
other directions are preserved.

Current validation is recorded in `validation_20261029.json`: 113 tests and
19 subtests passed; 92 local links/anchors passed; desktop and 390px layouts
have no document overflow, MathJax errors, or broken figures. The worktree's
handoff command stops at its uninitialized submodule; the original checkout
passed all 2590 manifest-file checks and the pinned upstream check. No submodule
or archived evidence was changed.
