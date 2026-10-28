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

## 2026-10-28 (night) — SAI definition and panel P1 fixes

- `sai-paper/SAI_definition.md` — new. Full SAI definition: the three levels operationalized by how a claim's boundary is known (observed at tested points / measured by systematic extension / derived from independently checkable premises); the three axes under an explicit contract (Ω, μ, ℓ, ε) with error e, cost c (direct training = 1) and non-applicability u; level, scope, evidence protocol and review depth recorded separately; the automated SAI system as a derived notion; open problems. Sources marked as user intent, existing protocol, or proposal.
- `sai-paper/report/` — new in the mirror. The Chinese panel with the four P1 issues fixed: PUCT top-40 = +3.3 and top-20 = −3.3 points; level criteria separated from machine-checkability, pre-registration and blind tests; a new three-axis figure redrawn from the user's sketch; Science allows a single conditional derivation and separates in-model correctness from real-system mapping. Overbroad wording narrowed; the 51-paper RL tally labeled as the old draft's sample. Built with `SAI_PUBLIC=1`, which anonymizes model identities as Frontier-1…10; local builds keep the real names. `gen_charts.py` reads `PANEL_STYLE_DIR` instead of a hard-coded local path.
- `sai-paper/literature/panel_source_map.*` — regenerated for the new panel level labels; counts unchanged (53 rows, 59 references, 23 fully matched, 30 unmatched). Its `panel_sha256` refers to the local, non-anonymized build.
- `sai-paper/README.md`, handoff §4 — entry for the definition; status of the fixes and what remains open (whole-paper labels such as AlphaFold, M0–M6 progression, the "Hu 2025" author).
