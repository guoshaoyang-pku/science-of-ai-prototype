# Milestones and Heroes

## The M0–M6 ladder for systems

Operational tests, not capability claims. A system passes a milestone only with **re-checkable evidence** of the test.

| Milestone | Test |
|---|---|
| **M0 — KB lift** | Injected accumulated knowledge measurably improves solving on held-out problems (paired arms, same solver) |
| **M1 — Regularity discovery** | The system proposes claims that pass machine verification on fresh cells, beyond its seed knowledge |
| **M2 — Surprise** | At least one verified claim contradicts the priors of expert humans and frontier models (checked against closed-book judgments) |
| **M3 — Frontier lift** | The accumulated knowledge raises the score of the *strongest available* model, not just weaker ones |
| **M4 — Explanatory coverage** | The KB explains essentially every problem in a domain: frontier accuracy pushed from ~75% toward 90% on a sealed set, with per-problem attribution |
| **M5 — Rigorous science** | The system synthesizes L3 claims directly: mechanisms with pre-registered novel predictions that pass blind tests, without human authorship of the prediction |
| **M6 — Paper-grade science** | Multiple results pass expert human review as publishable science, with the machine-verified evidence chain attached |

M0–M2 test the knowledge loop; M3–M4 test whether knowledge is *better than the model's weights*; M5–M6 test the law→science ascent.

## Heroes table

Existing systems against the ladder. Scoring is conservative: ✓ requires published, re-checkable evidence of the operational test; ~ = partial; ✗ = not passed; n/a = not applicable to the system's domain.

| System | M0 | M1 | M2 | M3 | M4 | M5 | M6 |
|---|---|---|---|---|---|---|---|
| Agent memory (ExpeL, Voyager, AWM, SkillRL, Cheatsheet) | ✓ | ~ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Program search (FunSearch, AlphaEvolve, OpenEvolve) | n/a | ~ | ✓† | ✗ | ✗ | ✗ | ✗ |
| AI Scientist v1/v2; Agentic Scientist | ✗ | ~ | ~ | ✗ | ✗ | ✗ | ~* |
| Zochi / Locus | ✗ | ~ | ~ | ✗ | ✗ | ✗ | ~* |
| AI co-scientist | ✗ | ~ | ✓† | ✗ | ✗ | ✗ | ✗ |
| Robin (biomedicine) | ✗ | ~ | ✓† | ✗ | ✗ | ✗ | ✗ |
| Law-finders elsewhere (Eureqa, AI Feynman) | n/a | ✓† | ✓† | n/a | n/a | ~† | ✗ |
| **SAI-min (this work, in ML)** | ✓ | ✓ | ✓ | ✗ | ✗ | ~ | ✗ |

† In the system's native (non-ML) domain. \* Peer-review-conferred (workshop/main-conference acceptance) without a machine-verified evidence chain — M6 by sociology, not by pre-registered prediction.

**SAI-min scoring detail.** M0: +14.2 / +15.8 pt paired lifts; M1: 34 hypotheses verified on fresh cells, 9 falsified; M2: verified claims contradicting frontier-model priors; M5 partial: pre-registered mechanism tests with recorded refutations, human-audited, not yet a confirmed novel theory. Evidence in [results.md](results.md).

## The pattern

Memory systems pass M0 and sometimes M1. Discovery machines pass M1–M2 **in their own (non-ML) domains**. AI-scientist systems reach M2 and touch M6 via peer review but skip the verification ladder entirely. **Nobody passes M3 or M4; M5 is open in ML.** The gap is not capability — it is that no existing system keeps a persistent, machine-verified claim store across campaigns.

## Human audit note

An experienced researcher reviewed the prototype's direction corpus: D1, D2, D4, D7 (phenomenological discoveries with good abstraction/attribution) judged genuinely valuable; D3, D5, D6 likely non-viable as modeled; D9 under-powered; D12 complex and partially pre-existing. Verdict: *"the KB's distillation of the science of AI far exceeded expectations, but current-stage findings still need an experienced researcher to gatekeep."* That is exactly why M5 scores partial: the loop demonstrably produces and kills mechanistic hypotheses under discipline, but the confirmed-novel-theory bar is not yet cleared without human audit.
