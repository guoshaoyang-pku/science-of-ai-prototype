# Prototype Results

What the loop produced when instantiated training-free on published infrastructure (the ArchitectureIQ benchmark, the KB-loop prototype, the fuzzy-direction research program in this repository). Model identities are anonymized in public figures; the underlying numbers are from the public leaderboard and this repo's `evidence/`.

![prototype measurements](../figures/results.svg)

## 1. Pricing the knowledge models already carry

Before adding knowledge, we price what is already there. Knowledge-free baseline: **truncated probing** — train every candidate to a fraction ρ of its budget and pick the current leader.

- On the main split, probing at ρ = 10% selects the eventual winner on **79.4%** of problems — above *every* closed-book frontier model; the strongest (77.6%) carries knowledge worth ρ* = 8.5%.
- Ten frontier models span ρ* = 1.0%–8.5% (accuracies 55.8%–77.6%).
- Reaching 90% selection costs ρ ≈ 30% on main but ρ ≈ 75% on the flip-problem split, where 10%-probing drops to 43.6–53.2%: flip problems are decided late, and that is where knowledge is most valuable. On flip problems all ten models score 41–53% against a 33% random floor.

**Reading:** the field's frontier knowledge is still cheap in compute terms — most of what the best models know is recoverable from a small, well-organized KB.

## 2. The KB loop (Solve–Discover–Summarize)

Three phases per epoch. **Solve:** the model sees only the problem and the KB (read-only, no execution tools). **Discover:** after answers are revealed, the model gains tools and full epoch history; on mistakes it may propose up to five quantitative falsifiable hypotheses (≤2000 tokens) with backtests. **Summarize:** a high-effort summarizer adds/revises/merges/deletes KB entries with epoch snapshots and a no-regression guard. Claims carry usage-credit counters; credibility is the Laplace-smoothed rate (s+1)/(s+f+2); claims below 0.45 with ≥3 evaluations are eliminated. Seed KB: 294 items.

**Paired-arm lifts** (identical problems and seeds, with vs without KB):

| Line | Without | With | Lift |
|---|---|---|---|
| Mixed (150 instances, 5 epochs) | 0.660 | 0.720 | **+6.0 pt** |
| Architecture (240 instances, 16 epochs) | 0.454 | 0.596 | **+14.2 pt** |
| Credibility-ranked top-40 + 10 trial slots (120 matched) | 0.458 | 0.617 | **+15.8 pt** |
| PUCT-style top-40 | 0.492 | 0.525 | +3.3 pt |
| PUCT top-20 | 0.450 | 0.417 | **−3.3 pt** |

The lift is largest where the baseline is weakest — knowledge is worth most exactly on problem types the model cannot already answer. Of **76 newly proposed hypotheses, 34 were cited and verified** in subsequent solving, **9 were falsified and eliminated**; the strongest discovery climbed to credibility 0.82 (s=8, f=1); the KB grew 294 → 316 (mixed) and 364 (architecture). Findings include a U-shaped performance curve family and a claim retracted after contradicting established results — retraction machinery working as designed.

**Retrieval policy is a promotion gate.** Ungated exploration (PUCT top-20) rotates unverified claims into context, dilutes credit among near-duplicates, and lands *below* the no-knowledge baseline. Operational lessons: rank by verified accuracy with a bounded trial period for new claims; ~40 hints (≈10k tokens) fit modern contexts; deduplicate aggressively.

## 3. The fuzzy-direction campaign (L2 → L3 gate)

A supervised automated research program over **fuzzy directions**: wish-list conjectures that a research round must turn into precise formulations and controlled experiments. One sandboxed session per round, CPU-scale tensor experiments ≤20 min; every round follows the invariant chain: read state → **pre-register** hypothesis and decision criteria in a commit *before* running → execute → record findings (failed predictions included) → update the central KB and living report.

Scale at snapshot: 12–13 directions, **113 rounds launched, 93 valid**, **128 central claims including registered refutations**, 480 commits as audit trail; development and sealed OOD sets strictly separated.

**Produced (verified law family):**
- **Effective time** U = Σ_t η_t: training curves under different learning-rate schedules collapse when reparameterized by U (D1), with per-optimizer displacement formulas carrying measured pass rates.
- **Overfitting-onset identity** with exact σ² dependence and n-dependence emerging under spectral concentration (D2).
- **Hessian-spectrum control** of training-curve shape, including delay of the descent transition (D4).
- **Sigmoidal universal fits** of curve families, with cascade-segment counts matching resolvable scales.

**Refuted — and why that matters more:** the D2 mechanism hypothesis (population normalization halves the aggregate onset-coefficient spread) was killed by its own blind test: predicted halving vs observed range growth 0.0158 → 0.0741 (ratio 4.68). The surviving statement is narrower and truer. A D13 conjecture ("deeper networks transition later") died at R² = 0.044. These are **pre-registered falsifications of the system's own hypotheses** — the behavior that distinguishes L3-producing loops from manuscript-producing ones.

**Human audit:** D1, D2, D4, D7 judged genuinely valuable; D3, D5, D6 likely non-viable; D9 under-powered; D12 partially pre-existing. Verdict: the KB's distillation far exceeded expectations, but current-stage findings still need an experienced researcher to gatekeep — hence M5 = partial.

## 4. Infrastructure results (loop is not benchmark-specific)

**DSL search:** within a fixed finite-dimensional architecture language, random search with performance-feedback mutation over 100 forecasting tasks on 25 datasets reached or tied **58 leaderboard bests across 17 datasets** (52 strictly ahead at three decimals); largest gain: NASDAQ 48-step record 0.989 → 0.677 test MSE (−31.56%), all four NASDAQ horizons improved 18.5–31.6%. Winners are dense in the DSL — evidence this region of Ω is compressible and law-minable.

**Task infrastructure:** a JSON↔Python bijection covers most open-source architectures, making designs enumerable and comparable — edit distances between real architectures become computable (e.g., 39 edits between two frontier architectures) — enabling the next measurement: search efficiency of LLM-guided proposals vs random walk vs MCMC over the same space.

## 5. The honest negative: solver-RL fails at small scale

We trained a 0.8B model with RLVR on flip-style 3-way judgment problems (best frontier score on the set: 52/450-scale). The model crossed the 0.33 random floor only slowly; an early truncation penalty collapsed response length toward 16 tokens — the policy learned to emit an empty-thought template and guess — and the final reward of 0.49 reflected overfitting (500 steps × 32 problems ≈ 30 epochs). Late chain-of-thought samples are answer-guessing, not reasoning. A mixed easier corpus fit within ≈10 epochs with the same non-thinking collapse.

Readings: (i) length-penalty shaping on judgment tasks destroys the reasoning substrate before it exists — cold-start distillation should precede RL; (ii) test-time scaling shows no clear benefit on the main leaderboard — CoT may be the wrong instrument for this judgment class; (iii) small models are doubly disadvantaged. **The training-free KB route dominates at current scale**: knowledge accumulated *outside* the weights is verifiable, retractable, and does not collapse.

## 6. The survey audit

51 classic RL papers classified under the ladder criteria: **43 knowledge, 3 law, 5 science** — the quantitative backbone of the "the field is almost all L1" claim. Full per-paper classification with notes in the companion [survey](../survey/).
