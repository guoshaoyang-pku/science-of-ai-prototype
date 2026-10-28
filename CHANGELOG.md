# Changelog

## 2026-10-28 — `sai-taxonomy` branch

Adds the English-language taxonomy and literature survey layer:

- `taxonomy/` — new. Nine pages defining the SAI framework: dual definition (object/system), three-level epistemic ladder with machine-checkable gates, five-operator research loop and discipline invariants, metrics (equivalent compute, compression ratio, SAI Index), milestones M0–M6 with heroes table, ten research families, anonymized prototype results, and a reading guide.
- `survey/` — new. English classification tables for 932 papers / 872 concepts across RL, pre-training, and interpretability, generated from the private annotated corpus: level tags (strong vs conditional/candidate/fragment), scope boundaries for highlighted law/science entries, and per-entry review-depth records. No PDFs, no local paths, no working annotations.
- `figures/` — new. Four standalone-compiled TikZ figures (SVG + PNG + sources in `figures/src/`): epistemic hierarchy + minimal loop, full state space, positioning map, prototype measurements. Model identities anonymized (Frontier-1…10).
- `README.md` — added an English entry section pointing at `taxonomy/` and `survey/`.

No changes to code, evidence, or existing documentation. No internal codenames, provider names, cost figures, or local paths in the new material.

## 2026-10-28 (later) — merged the parallel research line

Per user instruction, merged the parallel session's prepared candidates into this branch under `sai-paper/`:

- `sai-paper/handoffs/` — merge handoff (sanitized: internal Feishu doc links redacted, local absolute paths normalized), state snapshot, merge inventory, and the user's original hand-drawn three-axis sketch.
- `sai-paper/literature/` — 38 corpus files: per-direction JSON records (interp 317 / RL 271 / pretrain 344 papers), generated catalogs and concept maps, coverage protocols, and 14 boundary-case documents.
- `sai-paper/tools/` — `build_catalog.py`, `build_concept_map.py`, the four exact counterexample programs (all verified exit 0 by the parallel session), and `export_survey_public.py`. `build_panel_source_map.py` omitted: it depends on the frozen panel and `references.bib`, which are not migrated.
- `sai-paper/README.md` / `.gitignore` — mirrored with a public-mirror note prepended.

Not merged (per the handoff's own dispositions): the frozen manuscript and survey drafts, the Chinese panel (known P1 content issues, user has not approved the fixes), download/fetch scripts (network tools; one contains a contact email), `notes/` (internal meeting snapshots), `papers/` and `.cache/` (local-only source archives), and the superseded handoff snapshot.
