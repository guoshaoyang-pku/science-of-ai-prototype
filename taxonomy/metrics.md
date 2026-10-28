# Metrics: how to measure knowledge

The taxonomy is only useful if it changes what you can measure. Three instruments, all computable from the loop's state.

## Equivalent compute (ρ*)

**Question:** how much training compute would you need to *relearn from scratch* what a knowledge base already encodes?

**Definition:** ρ* is the smallest probing budget ρ such that Solve-with-K matches the target accuracy. If a model needs ρ* = 8.5% of a 10%-budget probing sweep to reach its closed-book accuracy, then the knowledge it "carries" is worth about that much compute.

**Measured values** (from the prototype, see [results.md](results.md)): ten frontier models span ρ* = 1.0%–8.5%; the 10%-budget probe with KB beats every closed-book model (79.4%). Reading: the field's frontier knowledge is still cheap in compute terms — most of what the best models know is recoverable from a small, well-organized KB.

## Compression ratio

**Question:** how much does the KB shrink the state space you must search?

**Definition:** compression = |Ω_raw| / |Ω_guided|, where Ω_guided is the effective search space after K prunes the unpromising regions. A KB that merely stores facts compresses little; a KB that encodes *applicability boundaries* compresses a lot, because most of Ω is excluded by the regions C of its claims.

**Measured values:** the fuzzy-direction campaign's KB-guided proposal step concentrates trials into a region roughly 15–40× smaller than uniform search on the same budget; the compression grows with KB maturity, which is the empirical signature of ascent.

## SAI Index

**Question:** how far up the ladder has a system climbed, in one number?

**Definition:** a weighted count of verified claims by level — SAI Index = w1·N(L1) + w2·N(L2) + w3·N(L3), with w1 ≪ w2 ≪ w3 and the constraint that a higher-level claim only counts if its lower-level support is intact. The index is **monotone under promotion, non-monotone under refutation** — losing a law costs more than gaining ten observations.

**Measured values:** the prototype's KB grew 294 → 364 verified items over one campaign with 76 hypotheses (34 verified, 9 refuted, rest narrowed) — index movement dominated by two L2 candidates (effective-time, PUCT scaling). The RL-survey audit scored 51 works: 43 L1 / 3 L2 / 5 L3 — the field's index is L1-heavy, so the marginal value of a verified law is high.

## Why these three

Equivalent compute prices knowledge in the field's own currency (compute). Compression measures the loop's *search* advantage. The SAI Index measures *ascent*. All three are computable from the state S — no external judge, no charitable reading.
