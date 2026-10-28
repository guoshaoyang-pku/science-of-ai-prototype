# Changelog

## 2026-10-28 — `sai-taxonomy` branch

Adds the English-language taxonomy and literature survey layer:

- `taxonomy/` — new. Nine pages defining the SAI framework: dual definition (object/system), three-level epistemic ladder with machine-checkable gates, five-operator research loop and discipline invariants, metrics (equivalent compute, compression ratio, SAI Index), milestones M0–M6 with heroes table, ten research families, anonymized prototype results, and a reading guide.
- `survey/` — new. English classification tables for 932 papers / 872 concepts across RL, pre-training, and interpretability, generated from the private annotated corpus: level tags (strong vs conditional/candidate/fragment), scope boundaries for highlighted law/science entries, and per-entry review-depth records. No PDFs, no local paths, no working annotations.
- `figures/` — new. Four standalone-compiled TikZ figures (SVG + PNG + sources in `figures/src/`): epistemic hierarchy + minimal loop, full state space, positioning map, prototype measurements. Model identities anonymized (Frontier-1…10).
- `README.md` — added an English entry section pointing at `taxonomy/` and `survey/`.

No changes to code, evidence, or existing documentation. No internal codenames, provider names, cost figures, or local paths in the new material.
