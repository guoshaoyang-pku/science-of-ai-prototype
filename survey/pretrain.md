# Survey — Pre-training

*scaling laws, optimization, initialization, architecture. Snapshot: rolling. 344 papers, 305 concepts tracked.*

Level counts (a paper can hold several; **bold** = full claim at that level): **262 knowledge · 17 law · 60 science**, plus 14+0+15 conditional / candidate / fragment tags.

Levels: **K** = verified observation in a stated setting; **L** = quantitative relation with fit/test separation and an explicit validity boundary; **S** = mechanism deriving phenomena from explicit assumptions. **Lowercase** (k/l/s) marks conditional theorems, candidates, and fragments — claims whose stated scope falls short of the full gate. The hierarchy grades the claim, not the worth of the paper. Review depth is recorded per entry; *targeted/selected sections checked* means exactly that — not a full-text review.

## Highlighted law/science entries

- **[The Implicit Bias of Gradient Descent on Separable Data](https://arxiv.org/abs/1710.10345)** — *Science*. Under the paper's assumptions, gradient descent's weight direction converges to the hard-margin L2 max-margin classifier while the norm diverges. *Boundary:* Linearly separable data, homogeneous linear predictor, smooth exponentially tailed losses, small fixed stepsize; not a theorem about deep nets, inseparable data, or arbitrary optimizers.
- **[Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060)** — *Knowledge+Science*. For given time-varying SSM coefficients and zero initial state, an algebraic correspondence between state recurrences and semiseparable matrix forms; under scalar-identity state structure, a connection to masked kernel attention — motivating Mamba-2. *Boundary:* The equivalence is operator-level under specified coefficients/state dimension/zero init; input-dependent coefficients make blocks generally non-fixed-linear; no softmax.
- **[On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)** — *Science*. Gradient-scale analysis of Pre-LN vs Post-LN at a simplified Transformer initialization; warmup studied. *Boundary:* Single-head, square weights, zero-init Q/K (uniform attention) assumptions; initialization-time gradient scales do not imply warmup is never needed.
- **[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)** — *Science*. For standard momentum-free SGD, the update equivalence lambda' = lambda/alpha reproduces decoupled decay; for non-scalar preconditioned updates no single scalar L2 coefficient reproduces the rule. *Boundary:* Prop. 1 is the specified SGD update only; AdamW's practical performance is a separate empirical layer.
- **[An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)** — *Law*. The gradient noise scale predicts the steps/compute trade-off across batch sizes; reported across multiple tasks. *Boundary:* Assumes per-sample gradient variance scaling as 1/B and a local quadratic approximation with tuned learning rates; not a global-optimality or generalization theorem.
- **[Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/abs/2103.00065)** — *Law*. In full-batch gradient descent, the top Hessian eigenvalue rises to and hovers near the local-quadratic stability threshold 2/eta. *Boundary:* Full-batch GD on specified networks/data/loss/stepsize; the 2/eta threshold is a quadratic-model notion — does not transfer to Adam, arbitrary minibatches, or global convergence claims.
- **[Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)** — *Science*. Maximal-update parametrization (muP) controls feature-update magnitude under width scaling; muTransfer's near-optimal HP transfer at finite width is a separately tested empirical claim. *Boundary:* Theory rests on model/optimizer/width-scaling rules; tables of parametrization are prescriptions, not proofs that optimal HPs must transfer; depth and regularization scale differently.
- **[Feature Learning in Infinite-Width Neural Networks](https://arxiv.org/abs/2011.14522)** — *Science*. In the abc-parametrization infinite-width limit, stable nontrivial training dynamics fall into two classes: feature learning or kernel (lazy) gradient descent. *Boundary:* Fixed input dimension, MLPs, Gaussian init, specified SGD/loss; training time independent of width; not a claim about arbitrary finite-width networks or Transformers.
- **[Deep Learning Scaling is Predictable, Empirically](https://arxiv.org/abs/1712.00409)** — *Law*. Empirical power-law fits of generalization error vs training-data size, across six model families and four application domains. *Boundary:* Exponents are empirical estimates over the fitted regime; not guaranteed stable across domains or at very small/large data.
- **[A Constructive Prediction of the Generalization Error Across Scales](https://arxiv.org/abs/1909.12673)** — *Law*. Empirical fit of test error as a surface over parameter count and data size for a fixed architecture-scaling policy, with cross-scale extrapolation tests. *Boundary:* PTB/WikiText-2/103 and image classification; mostly width scaling; not a theorem for arbitrary architectures or training policies.
- **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)** — *Law*. Empirical power laws for LM test loss vs non-embedding parameters, data, and training compute. *Boundary:* Each fit has bottleneck conditions (parameter-limited needs convergence; data-limited uses early stopping; compute-limited needs specified batch/allocation); coefficients are recipe-dependent, not universal.
- **[Scaling Laws for Autoregressive Generative Modeling](https://arxiv.org/abs/2010.14701)** — *Law*. Cross-entropy power-law-plus-constant fits for autoregressive Transformers across image, video, multimodal, and math tasks. *Boundary:* Decoder-only autoregressive objective; modalities/encodings/fit ranges differ; the authors themselves discuss scaling-exponent inconsistencies (Section 6).
- **[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)** — *Law*. Under fixed training FLOPs, the loss-optimal allocation between parameters and tokens (Chinchilla). *Boundary:* The authors' Transformer, data, loss, budget grid, and per-budget cosine schedules; the equal-FLOPs exponents (~0.49/0.51) are not claims about total deployment or inference cost.
- **[Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)** — *Law*. Fitted relations for the decaying value of repeated data and compute allocation under data constraints. *Boundary:* GPT-2 architecture, C4 subsets, specified budgets; '4 epochs is nearly free' is an observation within the measured range, not a universal threshold.
- **[Resolving Discrepancies in Compute-Optimal Scaling of Language Models](https://arxiv.org/abs/2406.19146)** — *Law*. Decomposes how compute-optimal scaling estimates depend on output-layer compute, warmup, and optimizer tuning across two web corpora. *Boundary:* Within the paper's reproduction, compute accounting, data, and search ranges; factors need not transfer across all architectures.
- **[Explaining Neural Scaling Laws](https://arxiv.org/abs/2102.06701)** — *Science*. Conditional bounds on loss vs data/model scale in variance-limited and resolution-limited models. *Boundary:* Thm. 1 needs output-concentration and loss regularity; Thm. 2 needs a compact data manifold, Lipschitz conditions, i.i.d. sampling, and interpolation; bounds are not exact LLM exponents.
- **[A Solvable Model of Neural Scaling Laws](https://arxiv.org/abs/2210.16859)** — *Science*. In a solvable latent-generative + random-feature + linear-regression model, analysis of how test loss scales with data and features and when scaling breaks. *Boundary:* Idealized spectra, random features, closed-form optima; does not model finite-compute optimization or prove Transformer spectra conform.
- **[Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572)** — *Science*. In the infinite-width limit of the paper's fully connected nets, the NTK at initialization converges, and under added conditions stays constant during finite-time training. *Boundary:* Thm. 1 requires Lipschitz activations; Thm. 2 adds twice-differentiable bounded-second-derivative activations and sequential infinite-width limits; numerical checks use ReLU.
- **[On Lazy Training in Differentiable Programming](https://arxiv.org/abs/1812.07956)** — *Science*. For scaled objectives R(alpha h(w))/alpha^2, Thm. 2.2 bounds the gap between the nonlinear gradient flow and its linearization at initialization as alpha→∞. *Boundary:* Requires differentiability with local Lipschitz structure, h(w0)=0, fixed finite horizon; the O(1/alpha) bound's constants grow with T; not a statement about arbitrary finite networks.
- **[Deep Double Descent: Where Bigger Models and More Data Hurt](https://arxiv.org/abs/1912.02292)** — *Law*. Double descent over model capacity and training time in the tested CNN/ResNet/Transformer setups; in some IWSLT settings more samples temporarily worsen test perplexity. *Boundary:* An empirical law candidate bounded by data/noise/regularization/optimizer/stopping; 'generalized double descent' is explicitly informal.
- **[Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)** — *Science*. Under independence, zero-bias, and symmetric-init assumptions, forward variance preservation for ReLU gives Var(W)=2/fan_in; the backward condition gives 2/fan_out — different directional init scales. *Boundary:* Analysis is at initialization for ReLU convnets; E[x^2] is not the variance of nonzero-mean ReLU outputs; the two scales need not both hold layer-wise for unequal widths.
- **[Deep Information Propagation](https://arxiv.org/abs/1611.01232)** — *Science*. Mean-field analysis of signal-propagation depth scales and criticality conditions in random networks, including dropout's effect on the critical point. *Boundary:* Independent Gaussian random weights and mean-field assumptions; not a limit on all finite residual network depths.
- **[Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/abs/1312.6120)** — *Science*. Exact learning dynamics for deep linear networks: learning stages and their depth dependence under specified conditions. *Boundary:* Deep linear, continuous-time small-stepsize, specific initialization/data conditions; not a training law for nonlinear LLMs.
- **[On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237)** — *Science*. Constructs convex problems where Adam-style exponential moving-average updates fail to converge; proposes AMSGrad with a convergence analysis. *Boundary:* Counterexample and analysis live in the paper's specified functions/settings; not a claim that Adam fails on all training tasks.
- **[Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance](https://arxiv.org/abs/2403.16952)** — *Law*. Fitted relations between domain mixture ratios and domain losses, used to select data mixtures. *Boundary:* Fit and validated within the authors' domains, model sizes, and budgets; extrapolation to arbitrary domains not guaranteed.
- **[Quantifying Memorization Across Neural Language Models](https://arxiv.org/abs/2202.07646)** — *Law*. Relations among text extractability, training-data duplication, and model scale in the tested LMs. *Boundary:* Specific prefix lengths and extraction protocols; measured within the authors' model family.

## Full classification table

| Level | Paper | Year | Review depth |
|---|---|---|---|
| K | [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861) | ? | selected sections checked |
| L | [A Bayesian Perspective on Generalization and Stochastic Gradient Descent](https://arxiv.org/abs/1710.06451) | ? | selected sections checked |
| K | [A Closer Look at Memorization in Deep Networks](https://arxiv.org/abs/1706.05394) | ? | selected method or introduction sections checked |
| k+s | [A Connection Between Score Matching and Denoising Autoencoders](https://www.iro.umontreal.ca/~vincentp/Publications/DenoisingScoreMatching_NeuralComp2011.pdf) | ? | selected sections checked |
| L | [A Constructive Prediction of the Generalization Error Across Scales](https://arxiv.org/abs/1909.12673) | ? | selected sections checked |
| K+S | [A Fast Learning Algorithm for Deep Belief Nets](https://www.cs.toronto.edu/~hinton/absps/ncfast.pdf) | ? | selected primary sections checked |
| S | [A General Lower Bound on the Number of Examples Needed for Learning](https://www.cis.upenn.edu/~mkearns/papers/lower.pdf) | ? | selected primary lower bound statement and construction chec |
| K+S | [A Learning Algorithm for Boltzmann Machines](https://www.cs.toronto.edu/~hinton/absps/cogscibm.pdf) | ? | selected primary sections and derivation structure checked |
| S | [A Logical Calculus of the Ideas Immanent in Nervous Activity](https://home.csulb.edu/~cwallis/382/readings/482/mccolloch.logical.calculus.ideas.1943.pdf) | ? | selected primary formal assumptions and theorem statements c |
| K | [A Neural Probabilistic Language Model](https://jmlr.csail.mit.edu/papers/volume3/bengio03a/bengio03a.pdf) | ? | selected sections checked |
| K | [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) | ? | selected sections checked |
| S | [A Simple Neural Network Generating an Interactive Memory](https://papers.baulab.info/papers/also/Anderson-1972.pdf) | ? | selected primary linear storage construction checked |
| K+S | [A Simplified Neuron Model as a Principal Component Analyzer](https://neurophysics.ucsd.edu/courses/physics_171/Oja_1982.pdf) | ? | selected primary update and mean ode theorem checked |
| S | [A Solvable Model of Neural Scaling Laws](https://arxiv.org/abs/2210.16859) | ? | selected sections checked |
| S | [A Stochastic Approximation Method](https://www.columbia.edu/~ww2040/8100F16/RM51.pdf) | ? | selected sections checked |
| K | [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948) | ? | selected sections checked |
| k+s | [A Theoretical Analysis of Contrastive Unsupervised Representation Learning](https://arxiv.org/abs/1902.09229) | ? | selected theorem statement checked |
| S | [A Theory of the Learnable](https://courses.grainger.illinois.edu/ece544na/fa2014/valiant84.pdf) | ? | selected primary protocol and theorem A argument checked |
| K | [ADADELTA: An Adaptive Learning Rate Method](https://arxiv.org/abs/1212.5701) | ? | selected sections checked |
| K | [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) | ? | selected sections checked |
| K | [Accurate prediction of protein structures and interactions using a 3-track neural network](https://doi.org/10.1126/science.abj8754) | ? | selected sections checked |
| K | [Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://doi.org/10.1038/s41586-024-07487-w) | ? | selected sections checked |
| K | [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) | ? | selected sections checked |
| K | [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235) | ? | selected sections checked |
| K | [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) | ? | selected sections checked |
| S | [Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](https://jmlr.csail.mit.edu/papers/volume12/duchi11a/duchi11a.pdf) | ? | theorem assumptions checked |
| L | [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) | ? | selected sections checked |
| K | [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) | ? | selected sections checked |
| K | [Analyzing & Reducing the Need for Learning Rate Warmup in GPT Training](https://arxiv.org/abs/2410.23922) | ? | selected sections checked |
| K | [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) | ? | selected sections checked |
| K | [Atlas: Few-shot Learning with Retrieval Augmented Language Models](https://arxiv.org/abs/2208.03299) | ? | selected sections checked |
| K | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | ? | selected sections checked |
| K | [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) | ? | selected sections checked |
| K | [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) | ? | method sections checked |
| K | [BASE Layers: Simplifying Training of Large, Sparse Models](https://arxiv.org/abs/2103.16716) | ? | selected sections checked |
| K | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) | ? | selected sections checked |
| K | [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254) | ? | selected sections checked |
| K | [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) | ? | selected sections checked |
| K | [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100) | ? | selected sections checked |
| K | [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230) | ? | selected method or introduction sections checked |
| K | [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) | ? | method sections checked |
| S | [Benign Overfitting in Linear Regression](https://arxiv.org/abs/1906.11300) | ? | theorem assumptions checked |
| k+s | [Beyond neural scaling laws: beating power law scaling via data pruning](https://arxiv.org/abs/2206.14486) | ? | selected sections checked |
| K | [Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models](https://arxiv.org/abs/2206.04615) | ? | selected sections checked |
| k+s | [Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) | ? | selected theorem statement checked |
| K | [Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences](https://doi.org/10.1073/pnas.2016239118) | ? | selected sections checked |
| K | [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733) | ? | selected sections checked |
| K | [Born-Again Neural Networks](https://arxiv.org/abs/1805.04770) | ? | selected sections checked |
| K | [ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models](https://arxiv.org/abs/2105.13626) | ? | selected sections checked |
| S | [Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data](https://arxiv.org/abs/1703.11008) | ? | theorem assumptions checked |
| S | [Correlation Matrix Memories](https://lucidar.me/en/neural-networks/files/1972-correlation-matrix-memories.pdf) | ? | selected primary linear recall identity and scope checked |
| S | [Cryptographic Limitations on Learning Boolean Formulae and Finite Automata](https://www.cis.upenn.edu/~mkearns/papers/cryptojacm.pdf) | ? | selected primary learning model RSA reduction and theorem st |
| K | [Curriculum Learning](https://icml.cc/2009/papers/119.pdf) | ? | selected sections checked |
| K | [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/abs/1905.04899) | ? | selected sections checked |
| K | [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) | ? | selected sections checked |
| K | [DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genomes](https://arxiv.org/abs/2306.15006) | ? | selected sections checked |
| K | [DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome](https://doi.org/10.1093/bioinformatics/btab083) | ? | selected sections checked |
| L | [Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance](https://arxiv.org/abs/2403.16952) | ? | selected sections checked |
| K | [Data Selection for Language Models via Importance Resampling](https://arxiv.org/abs/2302.03169) | ? | selected sections checked |
| K | [DataComp-LM: In search of the next generation of training sets for language models](https://arxiv.org/abs/2406.11794) | ? | selected sections checked |
| K | [De novo design of protein structure and function with RFdiffusion](https://doi.org/10.1038/s41586-023-06415-8) | ? | selected sections checked |
| K | [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) | ? | selected sections checked |
| S | [Decision Theoretic Generalizations of the PAC Model for Neural Net and Other Learning Applications](https://bpb-us-e1.wpmucdn.com/sites.ucsc.edu/dist/4/821/files/2017/07/Decision-theoretic-generalizations-PAC-model-neural-net-learning-applications.pdf) | ? | selected primary decision model and estimation optimization  |
| S | [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | ? | theorem assumptions checked |
| K | [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) | ? | selected sections checked |
| L | [Deep Double Descent: Where Bigger Models and More Data Hurt](https://arxiv.org/abs/1912.02292) | ? | selected sections checked |
| S | [Deep Information Propagation](https://arxiv.org/abs/1611.01232) | ? | selected sections checked |
| L | [Deep Learning Scaling is Predictable, Empirically](https://arxiv.org/abs/1712.00409) | ? | selected sections checked |
| K | [Deep Mutual Learning](https://arxiv.org/abs/1706.00384) | ? | selected sections checked |
| K | [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382) | ? | selected sections checked |
| K | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | ? | selected sections checked |
| K | [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) | ? | selected sections checked |
| K | [Deep contextualized word representations](https://arxiv.org/abs/1802.05365) | ? | selected sections checked |
| K | [DeepNet: Scaling Transformers to 1,000 Layers](https://arxiv.org/abs/2203.00555) | ? | selected sections checked |
| K | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | ? | method sections checked |
| S | [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852) | ? | theorem assumptions checked |
| K | [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) | ? | selected sections checked |
| K | [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | ? | selected sections checked |
| K | [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) | ? | selected sections checked |
| K | [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) | ? | selected sections checked |
| K | [Density Estimation using Real NVP](https://arxiv.org/abs/1605.08803) | ? | selected sections checked |
| K | [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) | ? | selected sections checked |
| K | [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) | ? | selected sections checked |
| K | [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) | ? | selected sections checked |
| K | [DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining](https://arxiv.org/abs/2305.10429) | ? | selected sections checked |
| K | [Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus](https://arxiv.org/abs/2104.08758) | ? | method sections checked |
| K | [Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research](https://arxiv.org/abs/2402.00159) | ? | selected sections checked |
| K | [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964) | ? | selected sections checked |
| K | [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.csail.mit.edu/papers/volume15/srivastava14a/srivastava14a.pdf) | ? | selected sections checked |
| K | [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) | ? | method sections checked |
| K | [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) | ? | selected sections checked |
| K | [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) | ? | selected sections checked |
| K | [Efficient Learning of Sparse Representations with an Energy-Based Model](https://proceedings.neurips.cc/paper/2006/file/87f4d79e36d68c3031ccf6c55e9bbd39-Paper.pdf) | ? | selected primary sections checked |
| K | [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) | ? | method sections checked |
| K | [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946) | ? | selected sections checked |
| K | [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396) | ? | selected sections checked |
| K | [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682) | ? | method sections checked |
| K | [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294) | ? | selected sections checked |
| S | [Equivalence of Models for Polynomial Learnability](https://www.cis.upenn.edu/~mkearns/papers/equivmodels.pdf) | ? | selected primary Lemma3 3 conversion checked with OCR and vi |
| k+s | [Estimation of Non-Normalized Statistical Models by Score Matching](https://www.jmlr.org/papers/v6/hyvarinen05a.html) | ? | selected theorem statement checked |
| K | [Evolutionary-scale prediction of atomic-level protein structure with a language model](https://doi.org/10.1126/science.ade2574) | ? | selected sections checked |
| S | [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/abs/1312.6120) | ? | selected sections checked |
| S | [Explaining Neural Scaling Laws](https://arxiv.org/abs/2102.06701) | ? | selected sections checked |
| S | [Exploring Generalization in Deep Learning](https://arxiv.org/abs/1706.08947) | ? | theorem assumptions checked |
| K | [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566) | ? | selected sections checked |
| K | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) | ? | selected sections checked |
| K | [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) | ? | method sections checked |
| K | [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805) | ? | method sections checked |
| K | [Extracting and Composing Robust Features with Denoising Autoencoders](https://icml.cc/2008/papers/592.pdf) | ? | selected primary sections checked |
| K | [Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/abs/2509.02046) | ? | selected sections checked |
| S | [Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization](https://arxiv.org/abs/2606.16899) | ? | theorem assumptions checked |
| K | [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) | ? | method sections checked |
| K | [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](https://arxiv.org/abs/1511.07289) | ? | selected sections checked |
| S | [Feature Learning in Infinite-Width Neural Networks](https://arxiv.org/abs/2011.14522) | ? | selected sections checked |
| K | [Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652) | ? | method sections checked |
| K | [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) | ? | selected sections checked |
| K | [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) | ? | method sections checked |
| S | [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) | ? | theorem assumptions checked |
| K | [Flat Minima](https://people.idsia.ch/~juergen/fm/) | ? | selected sections checked |
| k+s | [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | ? | selected theorem statement checked |
| k+s | [Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) | ? | selected theorem statement checked |
| K | [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | ? | selected sections checked |
| K | [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905) | ? | method sections checked |
| K | [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) | ? | selected sections checked |
| K | [GPT-NeoX-20B: An Open-Source Autoregressive Language Model](https://arxiv.org/abs/2204.06745) | ? | method sections checked |
| K | [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) | ? | method sections checked |
| K | [GPipe: Easy Scaling with Micro-Batch Pipeline Parallelism](https://arxiv.org/abs/1811.06965) | ? | selected sections checked |
| K | [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) | ? | method sections checked |
| K | [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) | ? | selected sections checked |
| K | [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415) | ? | selected sections checked |
| K | [Gemini: A Family of Highly Capable Multimodal Models](https://arxiv.org/abs/2312.11805) | ? | selected primary architecture training evaluation protocol a |
| K | [Generalization through Memorization: Nearest Neighbor Language Models](https://arxiv.org/abs/1911.00172) | ? | selected sections checked |
| K | [Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) | ? | selected sections checked |
| k+s | [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) | ? | selected theorem statement checked |
| K | [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) | ? | selected sections checked |
| K | [Generative Pretraining From Pixels](https://proceedings.mlr.press/v119/chen20s.html) | ? | method sections checked |
| K | [GloVe: Global Vectors for Word Representation](https://aclanthology.org/D14-1162.pdf) | ? | method sections checked |
| K | [Glow: Generative Flow with Invertible 1×1 Convolutions](https://arxiv.org/abs/1807.03039) | ? | selected sections checked |
| K | [Going deeper with convolutions](https://arxiv.org/abs/1409.4842) | ? | selected sections checked |
| K | [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation](https://arxiv.org/abs/1609.08144) | ? | selected sections checked |
| L | [Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability](https://arxiv.org/abs/2103.00065) | ? | selected sections checked |
| K | [Gradient-Based Learning Applied to Document Recognition](https://leon.bottou.org/publications/pdf/ieee-1998.pdf) | ? | selected sections checked |
| K | [Greedy Layer-Wise Training of Deep Networks](https://proceedings.neurips.cc/paper/2006/file/5da713a690c067105aeb2fae32403405-Paper.pdf) | ? | selected primary sections checked |
| K | [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) | ? | selected sections checked |
| K | [Group Normalization](https://arxiv.org/abs/1803.08494) | ? | selected sections checked |
| K | [Hierarchical Probabilistic Neural Network Language Model](https://proceedings.mlr.press/r5/morin05a.html) | ? | selected sections checked |
| K | [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) | ? | selected sections checked |
| S | [High-dimensional dynamics of generalization error in neural networks](https://arxiv.org/abs/1710.03667) | ? | theorem assumptions checked |
| K | [Highly accurate protein structure prediction with AlphaFold](https://doi.org/10.1038/s41586-021-03819-2) | ? | selected sections checked |
| K | [Holistic Evaluation of Language Models](https://arxiv.org/abs/2211.09110) | ? | selected sections checked |
| k+s | [How (not) to Train your Generative Model: Scheduled Sampling, Likelihood, Adversary?](https://arxiv.org/abs/1511.05101) | ? | selected theorem statement checked |
| K | [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) | ? | selected sections checked |
| K | [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units](https://arxiv.org/abs/2106.07447) | ? | method sections checked |
| S | [Hyperball May Not Be a Free Lunch](https://arxiv.org/abs/2607.22444) | ? | method sections checked |
| S | [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) | ? | selected sections checked |
| K | [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html) | ? | selected sections checked |
| K | [Improved protein structure prediction using potentials from deep learning](https://doi.org/10.1038/s41586-019-1923-7) | ? | selected sections checked |
| K | [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | ? | selected sections checked |
| K | [Improving language models by retrieving from trillions of tokens](https://arxiv.org/abs/2112.04426) | ? | selected sections checked |
| K | [Investigating Data Contamination for Pre-training Language Models](https://arxiv.org/abs/2401.06059) | ? | selected sections checked |
| K | [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) | ? | method sections checked |
| K | [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | ? | selected sections checked |
| K | [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language-models.pdf) | ? | selected sections checked |
| K | [Language models enable zero-shot prediction of the effects of mutations on protein function](https://papers.neurips.cc/paper_files/paper/2021/hash/f51338d736f95dd42427296047067694-Abstract.html) | ? | selected sections checked |
| K | [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962) | ? | selected sections checked |
| K | [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096) | ? | selected sections checked |
| K | [Latent Retrieval for Weakly Supervised Open Domain Question Answering](https://arxiv.org/abs/1906.00300) | ? | selected sections checked |
| K | [Layer Normalization](https://arxiv.org/abs/1607.06450) | ? | method sections checked |
| S | [Learnability and the Vapnik-Chervonenkis Dimension](https://trhvidsten.com/docs/classics/Blumer-1989.pdf) | ? | selected primary statistical vs computational conditions che |
| K | [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) | ? | method sections checked |
| S | [Learning Quickly When Irrelevant Attributes Abound: A New Linear-Threshold Algorithm](https://www.cs.yale.edu/homes/aspnes/pinewiki/attachments/WinnowAlgorithm/winnow.pdf) | ? | selected primary WINNOW1 update and mistake bound argument c |
| K | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | ? | selected sections checked |
| S | [Learning curves for Gaussian process regression: Approximations and bounds](https://arxiv.org/abs/cond-mat/0105015) | ? | selected derivation and approximation assumptions checked |
| K | [Learning representations by back-propagating errors](https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf) | ? | selected sections checked |
| K | [Leveraging Passage Retrieval with Generative Models for Open Domain Question Answering](https://arxiv.org/abs/2007.01282) | ? | selected sections checked |
| K | [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/abs/2307.09288) | ? | method sections checked |
| K | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) | ? | method sections checked |
| K | [Long Short-Term Memory](https://people.idsia.ch/~juergen/lstm1997-2024head.pdf) | ? | selected sections checked |
| K | [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) | ? | selected sections checked |
| K | [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) | ? | method sections checked |
| K | [MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://arxiv.org/abs/1905.02450) | ? | selected sections checked |
| K | [MSA Transformer](https://proceedings.mlr.press/v139/rao21a.html) | ? | selected sections checked |
| K | [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) | ? | method sections checked |
| K | [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) | ? | selected sections checked |
| K | [Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results](https://arxiv.org/abs/1703.01780) | ? | selected sections checked |
| K | [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) | ? | selected sections checked |
| K | [Mistral 7B](https://arxiv.org/abs/2310.06825) | ? | selected sections checked |
| K | [Mixed Precision Training](https://arxiv.org/abs/1710.03740) | ? | selected sections checked |
| K | [Mixtral of Experts](https://arxiv.org/abs/2401.04088) | ? | method sections checked |
| K | [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) | ? | selected sections checked |
| K | [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207) | ? | selected method or introduction sections checked |
| K | [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) | ? | selected sections checked |
| K | [NICE: Non-linear Independent Components Estimation](https://arxiv.org/abs/1410.8516) | ? | selected sections checked |
| K | [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) | ? | selected sections checked |
| K | [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) | ? | selected sections checked |
| K | [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) | ? | selected sections checked |
| S | [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572) | ? | selected sections checked |
| K+S | [Neural networks and physical systems with emergent collective computational abilities](https://authors.library.caltech.edu/records/w41x7-8bn13) | ? | primary body ocr read |
| S | [Neurons with Graded Response Have Collective Computational Properties like Those of Two-State Neurons](https://redwood.berkeley.edu/wp-content/uploads/2018/08/hopfield84.pdf) | ? | selected primary continuous model and energy identity checke |
| S | [Noise-contrastive estimation: A new estimation principle for unnormalized statistical models](https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf) | ? | theorem assumptions checked |
| K | [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838) | ? | method sections checked |
| K | [OPT: Open Pre-trained Transformer Language Models](https://arxiv.org/abs/2205.01068) | ? | selected sections checked |
| K | [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) | ? | selected sections checked |
| K+S | [On Contrastive Divergence Learning](https://proceedings.mlr.press/r5/carreira-perpinan05a.html) | ? | selected primary sections checked |
| S | [On Convergence Proofs for Perceptrons](https://cs.uwaterloo.ca/~y328yu/classics/novikoff.pdf) | ? | proof statements checked |
| K | [On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima](https://arxiv.org/abs/1609.04836) | ? | selected sections checked |
| S | [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) | ? | selected sections checked |
| S | [On Lazy Training in Differentiable Programming](https://arxiv.org/abs/1812.07956) | ? | theorem assumptions checked |
| K+S | [On the Convergence Properties of Contrastive Divergence](https://proceedings.mlr.press/v9/sutskever10a.html) | ? | selected primary sections checked |
| S | [On the Convergence of Adam and Beyond](https://arxiv.org/abs/1904.09237) | ? | selected sections checked |
| S | [On the Uniform Convergence of Relative Frequencies of Events to Their Probabilities](https://courses.grainger.illinois.edu/ece544na/fa2014/vapnik71.pdf) | ? | selected primary growth bound and sufficient condition check |
| K | [On the Variance of the Adaptive Learning Rate and Beyond](https://arxiv.org/abs/1908.03265) | ? | selected sections checked |
| K | [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063) | ? | selected sections checked |
| K | [On the importance of initialization and momentum in deep learning](https://proceedings.mlr.press/v28/sutskever13.pdf) | ? | method sections checked |
| K | [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) | ? | method sections checked |
| S | [PAC-Bayesian Generalisation Error Bounds for Gaussian Process Classification](https://www.jmlr.org/papers/v3/seeger02a.html) | ? | theorem assumptions checked |
| K | [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) | ? | method sections checked |
| K | [Paloma: A Benchmark for Evaluating Language Model Fit](https://arxiv.org/abs/2312.10523) | ? | method sections checked |
| K | [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759) | ? | selected sections checked |
| K | [Pre-training Molecular Graph Representation with 3D Geometry](https://arxiv.org/abs/2110.07728) | ? | selected sections checked |
| K | [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) | ? | method sections checked |
| K | [Prevalence of Neural Collapse during the terminal phase of deep learning training](https://arxiv.org/abs/2008.08186) | ? | method sections checked |
| K | [ProGen: Language Modeling for Protein Generation](https://arxiv.org/abs/2004.03497) | ? | selected sections checked |
| K | [Professor Forcing: A New Algorithm for Training Recurrent Networks](https://arxiv.org/abs/1610.09038) | ? | selected sections checked |
| K | [ProtTrans: Towards Cracking the Language of Life’s Code Through Self-Supervised Learning](https://arxiv.org/abs/2007.06225) | ? | selected sections checked |
| K | [Protein complex prediction with AlphaFold-Multimer](https://doi.org/10.1101/2021.10.04.463034) | ? | selected sections checked |
| K | [Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling](https://arxiv.org/abs/2304.01373) | ? | method sections checked |
| K | [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | ? | method sections checked |
| L | [Quantifying Memorization Across Neural Language Models](https://arxiv.org/abs/2202.07646) | ? | selected sections checked |
| K | [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) | ? | method sections checked |
| K | [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909) | ? | selected sections checked |
| K | [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/abs/1909.13719) | ? | selected sections checked |
| K+S | [Reconciling Kaplan and Chinchilla Scaling Laws](https://arxiv.org/abs/2406.12907) | ? | selected sections checked |
| L | [Reconciling modern machine learning practice and the bias-variance trade-off](https://arxiv.org/abs/1812.11118) | ? | selected sections checked |
| K | [Recurrent neural network based language model](https://www.isca-archive.org/interspeech_2010/mikolov10_interspeech.pdf) | ? | selected sections checked |
| K | [RedPajama: an Open Dataset for Training Large Language Models](https://arxiv.org/abs/2411.12372) | ? | method sections checked |
| K | [Reducing Activation Recomputation in Large Transformer Models](https://arxiv.org/abs/2205.05198) | ? | selected sections checked |
| K | [Reducing the Dimensionality of Data with Neural Networks](https://www.cs.toronto.edu/~rsalakhu/papers/science.pdf) | ? | selected primary sections checked |
| K | [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451) | ? | selected method or introduction sections checked |
| K | [Regularization of Neural Networks using DropConnect](https://proceedings.mlr.press/v28/wan13.pdf) | ? | method sections checked |
| K | [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182) | ? | selected sections checked |
| K | [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) | ? | selected sections checked |
| L | [Resolving Discrepancies in Compute-Optimal Scaling of Language Models](https://arxiv.org/abs/2406.19146) | ? | selected sections checked |
| K | [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) | ? | selected sections checked |
| K | [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) | ? | selected sections checked |
| K | [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | ? | selected sections checked |
| s | [Reverse-time diffusion equation models](https://www.sciencedirect.com/science/article/pii/0304414982900515) | ? | selected theorem statement checked |
| K | [Ring Attention with Blockwise Transformers for Near-Infinite Context](https://arxiv.org/abs/2310.01889) | ? | selected sections checked |
| K | [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) | ? | selected sections checked |
| K | [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | ? | method sections checked |
| K | [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) | ? | method sections checked |
| K | [Robust deep learning–based protein sequence design using ProteinMPNN](https://doi.org/10.1126/science.add2187) | ? | selected sections checked |
| K | [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | ? | method sections checked |
| K | [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983) | ? | selected sections checked |
| K+S | [SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321) | ? | selected sections and conditional derivation checked |
| L | [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264) | ? | selected sections checked |
| K | [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446) | ? | method sections checked |
| L | [Scaling Laws and Interpretability of Learning from Repeated Data](https://arxiv.org/abs/2205.10487) | ? | selected sections checked |
| L | [Scaling Laws for Autoregressive Generative Modeling](https://arxiv.org/abs/2010.14701) | ? | selected sections checked |
| L | [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | ? | selected sections checked |
| L | [Scaling Laws for Neural Machine Translation](https://arxiv.org/abs/2109.07740) | ? | method sections checked |
| K | [Scaling Laws from the Data Manifold Dimension](https://www.jmlr.org/papers/v23/20-1111.html) | ? | selected sections checked |
| K | [Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918) | ? | selected method or introduction sections checked |
| K | [Scaling to Very Very Large Corpora for Natural Language Disambiguation](https://aclanthology.org/P01-1005/) | ? | selected sections checked |
| K | [Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks](https://arxiv.org/abs/1506.03099) | ? | selected sections checked |
| k+s | [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) | ? | selected sections checked |
| k+s | [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515) | ? | selected theorem statement checked |
| K | [Self-Organized Formation of Topologically Correct Feature Maps](https://tcosmo.github.io/assets/soms/doc/kohonen1982.pdf) | ? | selected primary update simulations and failure sections che |
| K | [Self-Supervised Graph Transformer on Large-Scale Molecular Data](https://arxiv.org/abs/2007.02835) | ? | selected sections checked |
| K | [SemDeDup: Data-efficient learning at web-scale through semantic deduplication](https://arxiv.org/abs/2303.09540) | ? | selected sections checked |
| K | [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226) | ? | selected sections checked |
| K | [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) | ? | selected sections checked |
| K | [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947) | ? | selected sections checked |
| K | [Shampoo: Preconditioned Stochastic Tensor Optimization](https://arxiv.org/abs/1802.09568) | ? | selected sections checked |
| k+s | [Sharp Minima Can Generalize For Deep Nets](https://arxiv.org/abs/1703.04933) | ? | method sections checked |
| K | [Sharpness-Aware Minimization for Efficiently Improving Generalization](https://arxiv.org/abs/2010.01412) | ? | method sections checked |
| K | [Sigmoid Loss for Language Image Pre-Training](https://arxiv.org/abs/2303.15343) | ? | selected method or introduction sections checked |
| K | [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821) | ? | selected sections checked |
| K | [Simulating 500 million years of evolution with a language model](https://doi.org/10.1126/science.ads0018) | ? | selected sections checked |
| S | [Some PAC-Bayesian Theorems](https://doi.org/10.1023/A:1007618624809) | ? | theorem assumptions checked |
| K | [SpanBERT: Improving Pre-training by Representing and Predicting Spans](https://arxiv.org/abs/1907.10529) | ? | selected sections checked |
| K | [Sparse deep belief net model for visual area V2](https://proceedings.neurips.cc/paper/2007/file/4daa3db355ef2b0e64b472968cb70f0d-Paper.pdf) | ? | selected primary sections checked |
| S | [Spectral Bias and Task-Model Alignment Explain Generalization in Kernel Regression and Infinitely Wide Neural Networks](https://arxiv.org/abs/2006.13198) | ? | selected derivation and approximation assumptions checked |
| S | [Spectrum Dependent Learning Curves in Kernel Regression and Wide Neural Networks](https://arxiv.org/abs/2002.02561) | ? | selected derivation and approximation assumptions checked |
| S | [Stability and Generalization](https://www.jmlr.org/papers/v2/bousquet02a.html) | ? | theorem assumptions checked |
| K | [Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion](https://jmlr.org/papers/volume11/vincent10a/vincent10a.pdf) | ? | selected primary sections checked |
| S | [Stochastic Gradient Descent as Approximate Bayesian Inference](https://arxiv.org/abs/1704.04289) | ? | selected derivation and approximation assumptions checked |
| K | [Strategies for Pre-training Graph Neural Networks](https://arxiv.org/abs/1905.12265) | ? | selected sections checked |
| K | [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://arxiv.org/abs/1804.10959) | ? | selected sections checked |
| S | [Surprises in High-Dimensional Ridgeless Least Squares Interpolation](https://arxiv.org/abs/1903.08560) | ? | theorem assumptions checked |
| K | [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) | ? | selected sections checked |
| K | [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) | ? | method sections checked |
| K | [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675) | ? | selected sections checked |
| S | [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466) | ? | selected sections checked |
| K | [Textbooks Are All You Need](https://arxiv.org/abs/2306.11644) | ? | selected method or introduction sections checked |
| S | [The Anisotropic Noise in Stochastic Gradient Descent: Its Behavior of Escaping from Sharp Minima and Regularization Effects](https://arxiv.org/abs/1803.00195) | ? | selected derivation and approximation assumptions checked |
| K | [The Curse of Recursion: Training on Generated Data Makes Models Forget](https://arxiv.org/abs/2305.17493) | ? | selected method or introduction sections checked |
| K | [The Difficulty of Training Deep Architectures and the Effect of Unsupervised Pre-Training](https://proceedings.mlr.press/v5/erhan09a.html) | ? | selected sections checked |
| K | [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557) | ? | selected sections checked |
| K+S | [The Helmholtz Machine](https://www.cs.toronto.edu/~hinton/absps/helmholtz.pdf) | ? | selected primary sections checked |
| S | [The Implicit Bias of Gradient Descent on Separable Data](https://arxiv.org/abs/1710.10345) | ? | selected theorem statement checked |
| K | [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783) | ? | selected sections checked |
| K | [The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://arxiv.org/abs/1803.03635) | ? | method sections checked |
| K | [The Marginal Value of Adaptive Gradient Methods in Machine Learning](https://arxiv.org/abs/1705.08292) | ? | selected method or introduction sections checked |
| K | [The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain](https://web.engr.oregonstate.edu/~huanlian/teaching/ML/2020spring/extra/rosenblatt-1958.pdf) | ? | selected primary model and probability claims checked |
| K | [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027) | ? | selected sections checked |
| K | [The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only](https://arxiv.org/abs/2306.01116) | ? | selected sections checked |
| K | [The Wake-Sleep Algorithm for Unsupervised Neural Networks](https://www.cs.toronto.edu/~hinton/absps/ws.pdf) | ? | selected primary sections checked |
| S | [The generalization error of random features regression: Precise asymptotics and double descent curve](https://arxiv.org/abs/1908.05355) | ? | theorem assumptions checked |
| L | [Three Factors Influencing Minima in SGD](https://arxiv.org/abs/1711.04623) | ? | selected sections checked |
| K | [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351) | ? | selected sections checked |
| K | [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) | ? | method sections checked |
| S | [Train faster, generalize better: Stability of stochastic gradient descent](https://arxiv.org/abs/1509.01240) | ? | theorem assumptions checked |
| L | [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | ? | selected sections checked |
| K | [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) | ? | method sections checked |
| K | [Training Products of Experts by Minimizing Contrastive Divergence](https://www.cs.toronto.edu/~hinton/absps/nccd.pdf) | ? | selected primary sections checked |
| K | [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) | ? | method sections checked |
| K | [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236) | ? | selected sections checked |
| K+S | [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060) | ? | method sections checked |
| S | [Two models of double descent for weak features](https://arxiv.org/abs/1903.07571) | ? | theorem assumptions checked |
| K | [UL2: Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131) | ? | selected sections checked |
| k+s | [Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere](https://arxiv.org/abs/2005.10242) | ? | selected theorem statement checked |
| K | [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530) | ? | selected sections checked |
| S | [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) | ? | theorem assumptions checked |
| K | [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) | ? | selected sections checked |
| K | [Universal Transformers](https://arxiv.org/abs/1807.03819) | ? | selected method or introduction sections checked |
| K | [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) | ? | selected sections checked |
| K | [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) | ? | selected sections checked |
| K | [Using the Output Embedding to Improve Language Models](https://arxiv.org/abs/1608.05859) | ? | selected sections checked |
| K | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) | ? | selected sections checked |
| K | [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602) | ? | selected sections checked |
| K | [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) | ? | selected sections checked |
| k+s | [Wasserstein GAN](https://arxiv.org/abs/1701.07875) | ? | selected theorem statement checked |
| K | [WaveNet: A Generative Model for Raw Audio](https://arxiv.org/abs/1609.03499) | ? | selected sections checked |
| K | [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868) | ? | selected sections checked |
| K | [Why Does Unsupervised Pre-training Help Deep Learning?](https://jmlr.org/papers/volume11/erhan10a/erhan10a.pdf) | ? | selected primary sections checked |
| K | [Why Warmup the Learning Rate? Underlying Mechanisms and Improvements](https://arxiv.org/abs/2406.09405) | ? | selected sections checked |
| K | [Wide Residual Networks](https://arxiv.org/abs/1605.07146) | ? | selected sections checked |
| K | [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) | ? | selected sections checked |
| K | [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | ? | selected sections checked |
| K | [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092) | ? | method sections checked |
| K | [mT5: A massively multilingual pre-trained text-to-text transformer](https://arxiv.org/abs/2010.11934) | ? | selected sections checked |
| K | [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412) | ? | selected sections checked |
| K | [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) | ? | method sections checked |
