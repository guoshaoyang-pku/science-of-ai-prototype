#!/usr/bin/env python3
"""Export the public (English) survey pages for the science-of-ai-prototype repo.

Reads the private literature JSONs (Chinese claim/scope annotations, local PDF
paths) and emits English-only markdown: per-direction stats, a full paper table
(level + linked title + year + review depth), and hand-translated highlights
for Law/Science and iconic entries. Raw JSONs and local PDF paths are never
emitted. Run from the sai-paper root:

    python3 tools/export_survey_public.py /path/to/science-of-ai-prototype/survey
"""
import json
import sys
from collections import Counter
from pathlib import Path

DIRECTIONS = {
    "rl": ("Reinforcement learning", "RL post-training and classic RL theory"),
    "pretrain": ("Pre-training", "scaling laws, optimization, initialization, architecture"),
    "interp": ("Interpretability", "mechanistic interpretability, attribution, probing"),
}

# Hand-translated claims (English) for Law/Science and iconic Knowledge entries.
# Keyed by paper title.
HIGHLIGHTS = {
    # ---- RL: Science ----
    "Average Cost Temporal-Difference Learning": (
        "Science",
        "Thm. 1: almost-sure convergence of average-cost TD(lambda) — the average-cost estimate tends to the true mean, linear parameters to a unique projected fixed point.",
        "Fixed finite irreducible aperiodic Markov chain; full-column-rank linear features whose span excludes the all-ones vector; lambda in [0,1), Robbins–Monro stepsizes. Policy evaluation only — not a guarantee for nonlinear approximation or control."),
    "Learning Algorithms for Markov Decision Processes with Average Cost": (
        "Science",
        "Convergence proofs for average-cost learning: relative-value-iteration Q-learning and a stochastic-shortest-path reduction with two-timescale updates, under stated visitation and normalization conditions.",
        "Finite state/action; every stationary policy has a single recurrent class with a common recurrent reference state; per-variant simulation and boundedness conditions."),
    "Simulation-Based Optimization of Markov Reward Processes": (
        "Science",
        "Likelihood-ratio average-reward updates over regenerative cycles of a single trajectory; Prop. 3: performance iterates converge and the performance gradient tends to zero.",
        "Finite parameterized Markov reward processes; Assumptions 1–4 (aperiodicity, common regeneration state, twice-differentiable bounded rewards/transitions, square-summable stepsizes)."),
    "Infinite-Horizon Policy-Gradient Estimation": (
        "Science",
        "GPOMDP estimates the average-reward gradient from a single partially observed trajectory via decaying eligibility traces; for fixed beta<1 the estimate converges to a biased gradient object that tends to the true gradient as beta→1.",
        "Finite hidden states/actions/observations; fixed differentiable stochastic policy; unique stationary distribution per parameter; bounded rewards and score ratios."),
    # ---- Pretrain: Law ----
    "An Empirical Model of Large-Batch Training": (
        "Law",
        "The gradient noise scale predicts the steps/compute trade-off across batch sizes; reported across multiple tasks.",
        "Assumes per-sample gradient variance scaling as 1/B and a local quadratic approximation with tuned learning rates; not a global-optimality or generalization theorem."),
    "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability": (
        "Law",
        "In full-batch gradient descent, the top Hessian eigenvalue rises to and hovers near the local-quadratic stability threshold 2/eta.",
        "Full-batch GD on specified networks/data/loss/stepsize; the 2/eta threshold is a quadratic-model notion — does not transfer to Adam, arbitrary minibatches, or global convergence claims."),
    "Deep Learning Scaling is Predictable, Empirically": (
        "Law",
        "Empirical power-law fits of generalization error vs training-data size, across six model families and four application domains.",
        "Exponents are empirical estimates over the fitted regime; not guaranteed stable across domains or at very small/large data."),
    "Scaling Laws for Neural Language Models": (
        "Law",
        "Empirical power laws for LM test loss vs non-embedding parameters, data, and training compute.",
        "Each fit has bottleneck conditions (parameter-limited needs convergence; data-limited uses early stopping; compute-limited needs specified batch/allocation); coefficients are recipe-dependent, not universal."),
    "Scaling Laws for Autoregressive Generative Modeling": (
        "Law",
        "Cross-entropy power-law-plus-constant fits for autoregressive Transformers across image, video, multimodal, and math tasks.",
        "Decoder-only autoregressive objective; modalities/encodings/fit ranges differ; the authors themselves discuss scaling-exponent inconsistencies (Section 6)."),
    "Training Compute-Optimal Large Language Models": (
        "Law",
        "Under fixed training FLOPs, the loss-optimal allocation between parameters and tokens (Chinchilla).",
        "The authors' Transformer, data, loss, budget grid, and per-budget cosine schedules; the equal-FLOPs exponents (~0.49/0.51) are not claims about total deployment or inference cost."),
    "Scaling Data-Constrained Language Models": (
        "Law",
        "Fitted relations for the decaying value of repeated data and compute allocation under data constraints.",
        "GPT-2 architecture, C4 subsets, specified budgets; '4 epochs is nearly free' is an observation within the measured range, not a universal threshold."),
    "Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance": (
        "Law",
        "Fitted relations between domain mixture ratios and domain losses, used to select data mixtures.",
        "Fit and validated within the authors' domains, model sizes, and budgets; extrapolation to arbitrary domains not guaranteed."),
    "Quantifying Memorization Across Neural Language Models": (
        "Law",
        "Relations among text extractability, training-data duplication, and model scale in the tested LMs.",
        "Specific prefix lengths and extraction protocols; measured within the authors' model family."),
    "Deep Double Descent: Where Bigger Models and More Data Hurt": (
        "Law",
        "Double descent over model capacity and training time in the tested CNN/ResNet/Transformer setups; in some IWSLT settings more samples temporarily worsen test perplexity.",
        "An empirical law candidate bounded by data/noise/regularization/optimizer/stopping; 'generalized double descent' is explicitly informal."),
    "Resolving Discrepancies in Compute-Optimal Scaling of Language Models": (
        "Law",
        "Decomposes how compute-optimal scaling estimates depend on output-layer compute, warmup, and optimizer tuning across two web corpora.",
        "Within the paper's reproduction, compute accounting, data, and search ranges; factors need not transfer across all architectures."),
    "A Constructive Prediction of the Generalization Error Across Scales": (
        "Law",
        "Empirical fit of test error as a surface over parameter count and data size for a fixed architecture-scaling policy, with cross-scale extrapolation tests.",
        "PTB/WikiText-2/103 and image classification; mostly width scaling; not a theorem for arbitrary architectures or training policies."),
    # ---- Pretrain: Science (iconic) ----
    "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer": (
        "Science",
        "Maximal-update parametrization (muP) controls feature-update magnitude under width scaling; muTransfer's near-optimal HP transfer at finite width is a separately tested empirical claim.",
        "Theory rests on model/optimizer/width-scaling rules; tables of parametrization are prescriptions, not proofs that optimal HPs must transfer; depth and regularization scale differently."),
    "Neural Tangent Kernel: Convergence and Generalization in Neural Networks": (
        "Science",
        "In the infinite-width limit of the paper's fully connected nets, the NTK at initialization converges, and under added conditions stays constant during finite-time training.",
        "Thm. 1 requires Lipschitz activations; Thm. 2 adds twice-differentiable bounded-second-derivative activations and sequential infinite-width limits; numerical checks use ReLU."),
    "On Lazy Training in Differentiable Programming": (
        "Science",
        "For scaled objectives R(alpha h(w))/alpha^2, Thm. 2.2 bounds the gap between the nonlinear gradient flow and its linearization at initialization as alpha→∞.",
        "Requires differentiability with local Lipschitz structure, h(w0)=0, fixed finite horizon; the O(1/alpha) bound's constants grow with T; not a statement about arbitrary finite networks."),
    "The Implicit Bias of Gradient Descent on Separable Data": (
        "Science",
        "Under the paper's assumptions, gradient descent's weight direction converges to the hard-margin L2 max-margin classifier while the norm diverges.",
        "Linearly separable data, homogeneous linear predictor, smooth exponentially tailed losses, small fixed stepsize; not a theorem about deep nets, inseparable data, or arbitrary optimizers."),
    "On the Convergence of Adam and Beyond": (
        "Science",
        "Constructs convex problems where Adam-style exponential moving-average updates fail to converge; proposes AMSGrad with a convergence analysis.",
        "Counterexample and analysis live in the paper's specified functions/settings; not a claim that Adam fails on all training tasks."),
    "Deep Information Propagation": (
        "Science",
        "Mean-field analysis of signal-propagation depth scales and criticality conditions in random networks, including dropout's effect on the critical point.",
        "Independent Gaussian random weights and mean-field assumptions; not a limit on all finite residual network depths."),
    "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks": (
        "Science",
        "Exact learning dynamics for deep linear networks: learning stages and their depth dependence under specified conditions.",
        "Deep linear, continuous-time small-stepsize, specific initialization/data conditions; not a training law for nonlinear LLMs."),
    "Explaining Neural Scaling Laws": (
        "Science",
        "Conditional bounds on loss vs data/model scale in variance-limited and resolution-limited models.",
        "Thm. 1 needs output-concentration and loss regularity; Thm. 2 needs a compact data manifold, Lipschitz conditions, i.i.d. sampling, and interpolation; bounds are not exact LLM exponents."),
    "A Solvable Model of Neural Scaling Laws": (
        "Science",
        "In a solvable latent-generative + random-feature + linear-regression model, analysis of how test loss scales with data and features and when scaling breaks.",
        "Idealized spectra, random features, closed-form optima; does not model finite-compute optimization or prove Transformer spectra conform."),
    "Feature Learning in Infinite-Width Neural Networks": (
        "Science",
        "In the abc-parametrization infinite-width limit, stable nontrivial training dynamics fall into two classes: feature learning or kernel (lazy) gradient descent.",
        "Fixed input dimension, MLPs, Gaussian init, specified SGD/loss; training time independent of width; not a claim about arbitrary finite-width networks or Transformers."),
    "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality": (
        "Knowledge+Science",
        "For given time-varying SSM coefficients and zero initial state, an algebraic correspondence between state recurrences and semiseparable matrix forms; under scalar-identity state structure, a connection to masked kernel attention — motivating Mamba-2.",
        "The equivalence is operator-level under specified coefficients/state dimension/zero init; input-dependent coefficients make blocks generally non-fixed-linear; no softmax."),
    "On Layer Normalization in the Transformer Architecture": (
        "Science",
        "Gradient-scale analysis of Pre-LN vs Post-LN at a simplified Transformer initialization; warmup studied.",
        "Single-head, square weights, zero-init Q/K (uniform attention) assumptions; initialization-time gradient scales do not imply warmup is never needed."),
    "Decoupled Weight Decay Regularization": (
        "Science",
        "For standard momentum-free SGD, the update equivalence lambda' = lambda/alpha reproduces decoupled decay; for non-scalar preconditioned updates no single scalar L2 coefficient reproduces the rule.",
        "Prop. 1 is the specified SGD update only; AdamW's practical performance is a separate empirical layer."),
    "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification": (
        "Science",
        "Under independence, zero-bias, and symmetric-init assumptions, forward variance preservation for ReLU gives Var(W)=2/fan_in; the backward condition gives 2/fan_out — different directional init scales.",
        "Analysis is at initialization for ReLU convnets; E[x^2] is not the variance of nonzero-mean ReLU outputs; the two scales need not both hold layer-wise for unequal widths."),
    # ---- Interp: Science (iconic) ----
    "A Unified Approach to Interpreting Model Predictions": (
        "Science",
        "Within additive feature-attribution models plus three stated properties, Thm. 1: the unique attribution satisfying them is the Shapley value; SHAP is the practical approximation.",
        "Limited to the specified explanation model class and the local-accuracy/missingness/consistency axioms; the theorem does not fix the feature-missing distribution nor confer causal meaning."),
    "Understanding Black-box Predictions via Influence Functions": (
        "Knowledge+Science",
        "Influence of a training point on parameters and test loss via local derivatives of upweighted perturbations and an inverse-Hessian approximation.",
        "Derivation assumes twice-differentiable empirical risk, strict convexity, positive-definite Hessian at the optimum; in non-convex deep nets it is an experimental approximation."),
    "A Value for n-Person Games": (
        "Science",
        "For cooperative games with a finite carrier, the value is unique under symmetry, efficiency, and additivity axioms, given by the weighted marginal-contribution formula.",
        "Transferable-utility cooperative games; uniqueness is for the stated set function — it does not decide feature-missing conventions, causality, or fairness choices in ML attribution."),
    "Toy Models of Superposition": (
        "Knowledge+Science",
        "In sparse-feature toy models: superposition, phase transitions, and polytope geometry of feature representations.",
        "Small ReLU networks with sparse synthetic features; the authors explicitly state generalization to real networks is unknown; exact phase boundaries not verified."),
    "Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching": (
        "Knowledge+Science",
        "A constructive counterexample: directions causally disconnected from output can be superposed with dormant output-altering directions so subspace patching yields misleading causal effects.",
        "The counterexample negates unconditional faithfulness; it does not show all subspace patching is untrustworthy."),
    "Causal Abstractions of Neural Networks": (
        "Knowledge+Science",
        "A linear-network construction showing representations can be perfectly probeable yet causally inert; interchange interventions compare high-level variables with neural representations.",
        "Counterexample limited to the explicit linear construction; MQNLI alignment experiments limited to the specified models/interventions."),
    "Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations": (
        "Knowledge+Science",
        "For latent-variable models with factorized priors, an entangling bijection preserving the prior shows the observation distribution alone cannot identify a unique disentanglement; plus a multi-dataset empirical comparison.",
        "Thm. 1 assumes d>1 and factorized-density priors; does not rule out identifiability with supervision, data structure, or inductive biases."),
    "LEACE: Perfect linear concept erasure in closed form": (
        "Knowledge+Science",
        "Closed-form minimal-perturbation linear concept erasure; studies concept removal vs model behavior.",
        "Theory assumes categorical labels with nonzero class probabilities, linear predictors, convex losses; does not rule out nonlinear-probe recovery of the information."),
    # ---- RL: classic Knowledge icons ----
    "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning": (
        "Knowledge",
        "RL-first post-training (GRPO over verified answers) elicits long chain-of-thought reasoning; R1-Zero shows emergent reasoning without SFT, with documented readability failures.",
        "Specific base models, reward rules, and training recipes; the emergence observations are tied to the reported setting."),
}

REVIEW_MAP = [
    ("selected_sections_checked", "selected sections checked"),
    ("targeted_primary_sections", "targeted primary sections checked"),
    ("selected_method_sections", "method sections checked"),
    ("theorem_assumptions", "theorem assumptions checked"),
    ("proofs_and_experiments", "method/proof/empirical sections checked"),
    ("proof", "proof statements checked"),
]


def review_en(raw: str) -> str:
    first = (raw or "").split(";")[0].strip()
    for pat, label in REVIEW_MAP:
        if pat in first:
            return label
    if any(ord(ch) > 0x2E7F for ch in first):  # CJK
        return "targeted sections checked"
    return (first[:60] or "targeted sections checked").replace("_", " ")


# Qualifiers that downgrade a level tag to its weak (lowercase) form:
# conditional/candidate/fragment/partial-scope claims are not full-level claims.
WEAK_MARKERS = ("条件化", "候选", "局部", "反例", "限定", "待审", "未审", "碎片", "框架", "指南")


def levels_of(classification: str):
    """K/L/S = full claim at that level; k/l/s = conditional, candidate, or fragment."""
    cls = classification or ""
    weak = any(m in cls for m in WEAK_MARKERS)
    out = []
    for word, letter in (("Knowledge", "K"), ("Law", "L"), ("Science", "S")):
        if word in cls:
            out.append(letter.lower() if weak else letter)
    return "+".join(out) if out else "k"


def strong_levels(classification: str):
    return [c for c in levels_of(classification).split("+") if c.isupper()]


def first_author(authors: str) -> str:
    a = (authors or "").strip()
    if a.startswith("["):
        a = a.strip("[]").split(",")[0].strip().strip("'\"")
    else:
        a = a.split(",")[0]
    return a + " et al."


def main(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    totals = Counter()
    for key, (title, subtitle) in DIRECTIONS.items():
        d = json.loads(Path(f"literature/{key}.json").read_text())
        papers = d["papers"]
        concepts = d.get("concepts", [])
        tier = Counter()   # uppercase letter = strong/full claim at that level
        tier_weak = Counter()
        for p in papers:
            for letter in levels_of(p.get("classification", "")).split("+"):
                if letter.isupper():
                    tier[letter] += 1
                    totals[letter] += 1
                else:
                    tier_weak[letter.upper()] += 1
                    totals["weak"] += 1
        totals["papers"] += len(papers)
        totals["concepts"] += len(concepts)

        lines = []
        lines.append(f"# Survey — {title}")
        lines.append("")
        snapshot = d.get("as_of") or "rolling"
        lines.append(f"*{subtitle}. Snapshot: {snapshot}. "
                     f"{len(papers)} papers, {len(concepts)} concepts tracked.*")
        lines.append("")
        lines.append(f"Level counts (a paper can hold several; **bold** = full claim at that level): "
                     f"**{tier['K']} knowledge · {tier['L']} law · {tier['S']} science**, "
                     f"plus {tier_weak['K']}+{tier_weak['L']}+{tier_weak['S']} conditional / candidate / fragment tags.")
        lines.append("")
        lines.append("Levels: **K** = verified observation in a stated setting; **L** = quantitative "
                     "relation with fit/test separation and an explicit validity boundary; "
                     "**S** = mechanism deriving phenomena from explicit assumptions. "
                     "**Lowercase** (k/l/s) marks conditional theorems, candidates, and fragments — "
                     "claims whose stated scope falls short of the full gate. "
                     "The hierarchy grades the claim, not the worth of the paper. "
                     "Review depth is recorded per entry; *targeted/selected sections checked* means "
                     "exactly that — not a full-text review.")
        lines.append("")
        hi = [(p, HIGHLIGHTS[p["title"]]) for p in papers if p.get("title") in HIGHLIGHTS]
        if hi:
            lines.append("## Highlighted law/science entries")
            lines.append("")
            for p, (lvl, claim, scope) in hi:
                url = p.get("url") or ""
                t = p.get("title", "?")
                head = f"**[{t}]({url})**" if url else f"**{t}**"
                yr = p.get("year") or ""
                tail = f" ({yr})" if yr else ""
                lines.append(f"- {head}{tail} — *{lvl}*. {claim} *Boundary:* {scope}")
            lines.append("")
        lines.append("## Full classification table")
        lines.append("")
        lines.append("| Level | Paper | Year | Review depth |")
        lines.append("|---|---|---|---|")
        for p in sorted(papers, key=lambda p: (str(p.get("year") or "9999"), p.get("title", ""))):
            t = p.get("title", "?")
            url = p.get("url") or ""
            cell = f"[{t}]({url})" if url else t
            lines.append(f"| {levels_of(p.get('classification',''))} | {cell} "
                         f"| {p.get('year','?')} | {review_en(p.get('review_level',''))} |")
        lines.append("")
        (outdir / f"{key}.md").write_text("\n".join(lines))
        print(f"{key}: {len(papers)} papers ({tier['K']}K/{tier['L']}L/{tier['S']}S), "
              f"{len(concepts)} concepts, {len(hi)} highlights")

    print(f"TOTAL: {totals['papers']} papers ({totals['K']}K/{totals['L']}L/{totals['S']}S), "
          f"{totals['concepts']} concepts")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
