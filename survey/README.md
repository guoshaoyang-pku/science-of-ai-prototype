# The SAI Literature Survey

A curated, expanding collection of primary sources in machine learning, classified under the SAI hierarchy — **knowledge** (verified observation in a stated setting), **law** (quantitative relation with fit/test separation and an explicit validity boundary), **science** (mechanism deriving phenomena from explicit assumptions).

**932 papers and 872 concepts tracked** across three directions. A single paper can hold claims at several levels; tags grade claims, not the worth of papers.

| Direction | Scope | Papers | Strong levels (K / L / S) |
|---|---|---|---|
| [rl.md](rl.md) | RL post-training and classic RL theory | 271 | 176 / 3 / 35 |
| [pretrain.md](pretrain.md) | scaling laws, optimization, initialization, architecture | 344 | 262 / 17 / 60 |
| [interp.md](interp.md) | mechanistic interpretability, attribution, probing | 317 | 255 / 0 / 6 |

Lowercase tags (k/l/s) mark conditional theorems, candidates, and fragments — claims whose stated scope falls short of the full gate. The distinction matters: most "science"-tagged entries are *conditional* formal results (a theorem under idealized assumptions), not mechanisms validated by pre-registered prediction.

## Method

Each entry records: the claim (one sentence), its **scope/boundary** (what the paper's own evidence supports, with section and page pointers), the evidence location, and a **review depth** field. *Targeted/selected sections checked* means exactly that — not a full-text review. File identity, reading depth, and independent verification are tracked separately; classification is deliberately conservative and entries are downgraded when counterexamples or scope violations are found.

The collection is neither a complete census nor a representative sample; category counts are not estimates of the proportion of research at each level field-wide. It is a working instrument for the taxonomy in [`../taxonomy/`](../taxonomy/) — the ladder's gates are what the tags approximate at reading depth.

## The headline pattern

- **Knowledge dominates.** The overwhelming majority of even canonical work is conditional observation (K).
- **Laws are rare and identifiable.** The L-tagged set is small enough to list by hand — the scaling-law family, gradient-noise-scale, edge-of-stability, memorization and data-mixing laws, double descent.
- **Full science is rarest.** Most S-tags are conditional theorems (lowercase s once scope is priced in). The strict full-gate audit in the manuscript (213 canonical works: 168 K / 22 L / 23 S) is the conservative version of what these pages tag at reading depth.

## What is not here

No PDFs (copyright and size), no Chinese working annotations, no internal paths — this is the English public layer of a larger private corpus. The underlying records carry richer claim/scope/evidence annotations and local source archives; the public pages keep the classification, the boundary discipline, and the pointers.
