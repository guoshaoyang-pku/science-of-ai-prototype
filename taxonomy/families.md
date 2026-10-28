# The Ten Families

The classification question for any body of work is triple: **what level of claim does it produce**, **how strong is its verification**, and **does its knowledge accumulate and ascend**.

![positioning map](../figures/positioning.svg)

| # | Family | Producer | Level | Verification | Accumulation / ascent |
|---|---|---|---|---|---|
| 1 | Empirical ML & systems papers | Human | L1 (unstructured) | Peer review; not machine-checkable | None; knowledge decays with model generations |
| 2 | Empirical laws | Human | L2 | Fit/holdout on real runs | Islands; boundaries stripped in citation chains |
| 3 | Interpretability | Human | L1–L3 fragments | Intervention experiments; model-specific | Mechanism-rich, boundary-poor |
| 4 | Learning theory | Human | L3 (enclaves) | Proof + idealized experiments | Rigorous but disconnected from frontier practice |
| 5 | AutoML / NAS / program search | Machine | L1 artifacts | **Strong** (executable evaluators) | Implicit in archives; no claims, no ascent |
| 6 | Agent memory / experience | Machine | L1 | Task-success correlation | Accumulates; ungated; can hurt (dead-end accumulation) |
| 7 | AI-scientist systems | Machine | L1 manuscripts, some L2 hypotheses | LLM/human review (verification gap) | Per-campaign; no persistent K |
| 8 | Discovery machines, other domains | Machine | L2–L3 (their domains) | **Strong** (cheap oracles) | Existence proof: ascent works where oracles are cheap |
| 9 | Benchmarks | — | measure L1 ability only | Ground-truth tasks | No instrument for retention/promotion/prediction |
| 10 | Curation & meta-science | Human+LLM | organizes L1 | Extraction quality; audits | Catalogs findings; creates no verified claims |

**No existing family couples strong verification with explicit ascent.** The natural-science machines come closest, in domains with cheap oracles.

## Family notes

**1. Empirical ML & systems** — the field's volume. GRPO stability, GSPO clipping, PPO vs TRPO, bag of tricks, Megatron/ZeRO/FlashAttention regimes, frontier-report bundles. Under the gates these are L1 *candidates*: conditions live in prose, adjudication needs charitable reading, no machine can re-check on a fresh cell. Modelpedia's catalog of 1000+ extracted findings from two years of ICLR — effectively unretrievable without automation — quantifies the loss.

**2. Empirical laws** — the clearest L2 family: Kaplan scaling laws, Chinchilla compute-optimal reallocation, data-constrained variants, knowledge-capacity laws (bits/parameter), reward-overoptimization laws, sigmoidal RL compute–performance curves, test-time compute laws, double descent, grokking phase behavior, μP width-transfer. Their failure mode is sociological: validity boundaries are stripped as citations propagate.

**3. Interpretability** — mechanism-rich, boundary-poor: induction heads, monosemantic features at scale, grokking circuits. Genuine mechanistic insight, but claims are model-specific fragments without machine-checkable regions — L1–L3 *fragments*, not verified clusters.

**4. Learning theory** — ML's most rigorous L3, in enclaves: NTK convergence/generalization, lazy-vs-rich dichotomy, infinite-width feature learning, statistical-mechanical derivations of scaling exponents; classical RL theory (Q-learning convergence, policy gradient theorem). Limitation is domain, not rigor: assumptions carve out enclaves frontier practice has left. The Math-of-AI program — training-as-pure-mathematics — is the L3-by-derivation route, complementary to L3-by-pre-registered-prediction.

**5. AutoML / NAS / program search** — the strongest *verifiers* in the map: NAS, FunSearch, AlphaEvolve evolve against executable evaluators and have produced genuinely new algorithms. In the operator view they implement Propose and Verify at near-oracle strength — and almost no K: nothing states *why* winners win or where patterns transfer. Our own DSL search shares the diagnosis: 58 SOTA results across 17 time-series datasets are valuable as *material* for knowledge extraction, not as knowledge.

**6. Agent memory** — ExpeL, Voyager, AWM, SkillRL, Cheatsheet: accumulates across tasks but ungated — the corpus can hurt (dead-end accumulation) precisely because nothing verifies before storing.

**7. AI-scientist systems** — AI Scientist v1/v2, Agentic Scientist, Zochi/Locus, co-scientist, Agent Laboratory, Robin, MLE-STAR. Nominally all five operators, but Verify is peer review (human or LLM), not programmatic adjudication on fresh cells, and K does not persist across campaigns. The recurring diagnosis in the surveys is the *verification gap*.

**8. Discovery machines in other domains** — Eureqa, GNoME, AlphaFold: L2–L3 output with strong gates where the oracle is cheap (symbolic physics, materials, structure prediction). The existence proof that machine ascent works — where experiments are cheap.

**9. Benchmarks** — MLE-bench, RE-Bench, PaperBench, DSBench, MLGym, ScienceAgentBench, SWE-bench, METR horizons, ArchitectureIQ: every current instrument reads L1 — task performance *now*. None measures retention, promotion, held-out prediction of discovered laws, or blind-falsification survival. The missing instrument family (an "SAI-Bench" for M0–M6) is the most valuable near-term community artifact this framework points to.

**10. Curation & meta-science** — Modelpedia, verification-gap analyses, AI-scientist surveys: organizes L1, creates no verified claims. Useful substrate; not ascent.
