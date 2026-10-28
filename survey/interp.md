# Survey — Interpretability

*mechanistic interpretability, attribution, probing. Snapshot: rolling. 317 papers, 302 concepts tracked.*

Level counts (a paper can hold several; **bold** = full claim at that level): **255 knowledge · 0 law · 6 science**, plus 58+1+25 conditional / candidate / fragment tags.

Levels: **K** = verified observation in a stated setting; **L** = quantitative relation with fit/test separation and an explicit validity boundary; **S** = mechanism deriving phenomena from explicit assumptions. **Lowercase** (k/l/s) marks conditional theorems, candidates, and fragments — claims whose stated scope falls short of the full gate. The hierarchy grades the claim, not the worth of the paper. Review depth is recorded per entry; *targeted/selected sections checked* means exactly that — not a full-text review.

## Highlighted law/science entries

- **[Toy Models of Superposition](https://arxiv.org/abs/2209.10652)** — *Knowledge+Science*. In sparse-feature toy models: superposition, phase transitions, and polytope geometry of feature representations. *Boundary:* Small ReLU networks with sparse synthetic features; the authors explicitly state generalization to real networks is unknown; exact phase boundaries not verified.
- **[Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching](https://arxiv.org/abs/2311.17030)** — *Knowledge+Science*. A constructive counterexample: directions causally disconnected from output can be superposed with dormant output-altering directions so subspace patching yields misleading causal effects. *Boundary:* The counterexample negates unconditional faithfulness; it does not show all subspace patching is untrustworthy.
- **[Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730)** — *Knowledge+Science*. Influence of a training point on parameters and test loss via local derivatives of upweighted perturbations and an inverse-Hessian approximation. *Boundary:* Derivation assumes twice-differentiable empirical risk, strict convexity, positive-definite Hessian at the optimum; in non-convex deep nets it is an experimental approximation.
- **[A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)** — *Science*. Within additive feature-attribution models plus three stated properties, Thm. 1: the unique attribution satisfying them is the Shapley value; SHAP is the practical approximation. *Boundary:* Limited to the specified explanation model class and the local-accuracy/missingness/consistency axioms; the theorem does not fix the feature-missing distribution nor confer causal meaning.
- **[Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997)** — *Knowledge+Science*. A linear-network construction showing representations can be perfectly probeable yet causally inert; interchange interventions compare high-level variables with neural representations. *Boundary:* Counterexample limited to the explicit linear construction; MQNLI alignment experiments limited to the specified models/interventions.
- **[LEACE: Perfect linear concept erasure in closed form](https://arxiv.org/abs/2306.03819)** — *Knowledge+Science*. Closed-form minimal-perturbation linear concept erasure; studies concept removal vs model behavior. *Boundary:* Theory assumes categorical labels with nonzero class probabilities, linear predictors, convex losses; does not rule out nonlinear-probe recovery of the information.
- **[Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://arxiv.org/abs/1811.12359)** — *Knowledge+Science*. For latent-variable models with factorized priors, an entangling bijection preserving the prior shows the observation distribution alone cannot identify a unique disentanglement; plus a multi-dataset empirical comparison. *Boundary:* Thm. 1 assumes d>1 and factorized-density priors; does not rule out identifiability with supervision, data structure, or inductive biases.
- **[A Value for n-Person Games](https://doi.org/10.1515/9781400881970-018)** — *Science*. For cooperative games with a finite carrier, the value is unique under symmetry, efficiency, and additivity axioms, given by the weighted marginal-contribution formula. *Boundary:* Transferable-utility cooperative games; uniqueness is for the stated set function — it does not decide feature-missing conventions, causality, or fairness choices in ML attribution.

## Full classification table

| Level | Paper | Year | Review depth |
|---|---|---|---|
| K | [A Benchmark for Interpretability Methods in Deep Neural Networks](https://arxiv.org/abs/1806.10758) | ? | targeted sections checked |
| K | [A Category-free Neural Population Supports Evolving Demands during Decision-Making](https://doi.org/10.1038/nn.3865) | ? | targeted sections checked |
| k | [A Geometric Notion of Causal Probing](https://arxiv.org/abs/2307.15054) | ? | targeted sections checked |
| K | [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) | ? | targeted sections checked |
| K | [A Multimodal Automated Interpretability Agent](https://arxiv.org/abs/2404.14394) | ? | targeted sections checked |
| K | [A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models](https://arxiv.org/abs/2407.02646) | ? | targeted sections checked |
| K | [A Pragmatic Way to Measure Chain-of-Thought Monitorability](https://arxiv.org/abs/2510.23966) | ? | targeted sections checked |
| K | [A Primer in BERTology: What we know about how BERT works](https://arxiv.org/abs/2002.12327) | ? | targeted sections checked |
| K | [A Structural Probe for Finding Syntax in Word Representations](https://aclanthology.org/N19-1419/) | ? | targeted sections checked |
| k+s | [A Toy Model of Universality: Reverse Engineering How Networks Learn Group Operations](https://arxiv.org/abs/2302.03025) | ? | targeted sections checked |
| k+s | [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) | ? | targeted sections checked |
| S | [A Value for n-Person Games](https://doi.org/10.1515/9781400881970-018) | ? | targeted sections checked |
| K | [A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](https://arxiv.org/abs/2409.14507) | ? | targeted sections checked |
| K | [A survey of algorithmic recourse: contrastive explanations and consequential recommendations](https://arxiv.org/abs/2010.04050) | ? | targeted sections checked |
| S | [A technical note on bilinear layers for interpretability](https://arxiv.org/abs/2305.03452) | ? | targeted sections checked |
| K | [Acquisition of Chess Knowledge in AlphaZero](https://arxiv.org/abs/2111.09259) | ? | targeted sections checked |
| K | [Actionable Recourse in Linear Classification](https://arxiv.org/abs/1809.06514) | ? | targeted sections checked |
| K | [Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/abs/1905.02175) | ? | targeted sections checked |
| K | [Adversarial Examples Are Not Real Features](https://arxiv.org/abs/2310.18936) | ? | targeted sections checked |
| K | [Algorithmic Recourse: from Counterfactual Explanations to Interventions](https://arxiv.org/abs/2002.06278) | ? | targeted sections checked |
| k+s | [Algorithmic recourse under imperfect causal knowledge: a probabilistic approach](https://arxiv.org/abs/2006.06831) | ? | targeted sections checked |
| K | [All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously](https://arxiv.org/abs/1801.01489) | ? | targeted sections checked |
| K | [Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals](https://arxiv.org/abs/2006.00995) | ? | targeted sections checked |
| K | [An Interpretability Illusion for BERT](https://arxiv.org/abs/2104.07143) | ? | targeted sections checked |
| K | [An Overview of Early Vision in InceptionV1](https://distill.pub/2020/circuits/early-vision/) | ? | targeted sections checked |
| K | [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://arxiv.org/abs/1905.09418) | ? | targeted sections checked |
| K | [Analyzing Transformers in Embedding Space](https://arxiv.org/abs/2209.02535) | ? | targeted sections checked |
| K | [Anchors: High-Precision Model-Agnostic Explanations](https://ojs.aaai.org/index.php/AAAI/article/view/11491) | ? | targeted sections checked |
| K | [Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet](https://arxiv.org/abs/1904.00760) | ? | targeted sections checked |
| K | [Are Neural Nets Modular? Inspecting Functional Modularity Through Differentiable Weight Masks](https://arxiv.org/abs/2010.02066) | ? | targeted sections checked |
| K | [Are Sixteen Heads Really Better than One?](https://arxiv.org/abs/1905.10650) | ? | targeted sections checked |
| K | [Are Sparse Autoencoder Benchmarks Reliable?](https://arxiv.org/abs/2605.18229) | ? | targeted sections checked |
| K | [AtP*: An efficient and scalable method for localizing LLM behaviour to components](https://arxiv.org/abs/2403.00745) | ? | targeted sections checked |
| K | [Attention is not Explanation](https://arxiv.org/abs/1902.10186) | ? | targeted sections checked |
| K | [Attention is not not Explanation](https://arxiv.org/abs/1908.04626) | ? | targeted sections checked |
| K | [Attribution Patching Outperforms Automated Circuit Discovery](https://arxiv.org/abs/2310.10348) | ? | targeted sections checked |
| K | [Auditing Visualizations: Transparency Methods Struggle to Detect Anomalous Behavior](https://arxiv.org/abs/2206.13498) | ? | targeted sections checked |
| k+s | [Axiomatic Attribution for Deep Networks](https://arxiv.org/abs/1703.01365) | ? | targeted sections checked |
| k | [BERT Rediscovers the Classical NLP Pipeline](https://arxiv.org/abs/1905.05950) | ? | targeted sections checked |
| K | [Backpack Language Models](https://aclanthology.org/2023.acl-long.506/) | ? | targeted sections checked |
| k+s | [Backward Lens: Projecting Language Model Gradients into the Vocabulary Space](https://aclanthology.org/2024.emnlp-main.142/) | ? | targeted sections checked |
| K | [BatchTopK Sparse Autoencoders](https://arxiv.org/abs/2412.06410) | ? | targeted sections checked |
| K | [Beyond Model Interpretability: Socio-Structural Explanations in Machine Learning](https://arxiv.org/abs/2409.03632) | ? | targeted sections checked |
| k | [Beyond Word Importance: Contextual Decomposition to Extract Interactions from LSTMs](https://arxiv.org/abs/1801.05453) | ? | targeted sections checked |
| K | [Black-Box Access is Insufficient for Rigorous AI Audits](https://arxiv.org/abs/2401.14446) | ? | targeted sections checked |
| K | [Bridging the Human-AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero](https://arxiv.org/abs/2310.16410) | ? | targeted sections checked |
| K | [Building High-level Features Using Large Scale Unsupervised Learning](https://arxiv.org/abs/1112.6209) | ? | targeted sections checked |
| K | [CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks](https://arxiv.org/abs/2102.03322) | ? | targeted sections checked |
| k+s | [Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability](https://arxiv.org/abs/2301.04709) | ? | targeted sections checked |
| k+s | [Causal Abstractions of Neural Networks](https://arxiv.org/abs/2106.02997) | ? | targeted sections checked |
| K | [Causal Analysis of Syntactic Agreement Mechanisms in Neural Language Models](https://arxiv.org/abs/2106.06087) | ? | targeted sections checked |
| K | [Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias](https://arxiv.org/abs/2004.12265) | ? | targeted sections checked |
| K | [Causal Scrubbing: a method for rigorously testing interpretability hypotheses](https://www.alignmentforum.org/s/h95ayYYwMebGEYN5y/p/JvZhhzycHu2Yd57RN) | ? | targeted sections checked |
| K | [Causal Shapley Values: Exploiting Causal Knowledge to Explain Individual Predictions of Complex Models](https://arxiv.org/abs/2011.01625) | ? | targeted sections checked |
| k+s | [Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations](https://arxiv.org/abs/1811.12359) | ? | targeted sections checked |
| K | [Circuit Component Reuse Across Tasks in Transformer Language Models](https://arxiv.org/abs/2310.08744) | ? | targeted sections checked |
| k | [Circuit Tracing: Revealing Computational Graphs in Language Models](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) | ? | targeted sections checked |
| K | [Codebook Features: Sparse and Discrete Interpretability for Neural Networks](https://arxiv.org/abs/2310.17230) | ? | targeted sections checked |
| K+S | [Compact Proofs of Model Performance via Mechanistic Interpretability](https://arxiv.org/abs/2406.11779) | ? | targeted sections checked |
| K | [Compositional Explanations of Neurons](https://arxiv.org/abs/2006.14032) | ? | targeted sections checked |
| k+s | [Concept Activation Regions: A Generalized Framework For Concept-Based Explanations](https://arxiv.org/abs/2209.11222) | ? | targeted sections checked |
| K | [Concept Bottleneck Models](https://arxiv.org/abs/2007.04612) | ? | targeted sections checked |
| K | [Concept Whitening for Interpretable Image Recognition](https://arxiv.org/abs/2002.01650) | ? | targeted sections checked |
| K | [Context-dependent computation by recurrent dynamics in prefrontal cortex](https://doi.org/10.1038/nature12742) | ? | targeted sections checked |
| k | [Contrastive Explanation: A Structural-Model Approach](https://arxiv.org/abs/1811.03163) | ? | targeted sections checked |
| K | [Convergent Learning: Do Different Neural Networks Learn the Same Representations?](https://arxiv.org/abs/1511.07543) | ? | targeted sections checked |
| K | [Copy Suppression: Comprehensively Understanding an Attention Head](https://arxiv.org/abs/2310.04625) | ? | targeted sections checked |
| K | [Counterfactual Explanations Can Be Manipulated](https://arxiv.org/abs/2106.02666) | ? | targeted sections checked |
| K | [Counterfactual Explanations without Opening the Black Box: Automated Decisions and the GDPR](https://arxiv.org/abs/1711.00399) | ? | targeted sections checked |
| K | [Coup probes: Catching catastrophes with probes trained off-policy](https://www.alignmentforum.org/posts/WCj7WgFSLmyKaMwPR/coup-probes-catching-catastrophes-with-probes-trained-off) | ? | targeted sections checked |
| k | [Current themes in mechanistic interpretability research](https://www.lesswrong.com/posts/Jgs7LQwmvErxR9BCC/current-themes-in-mechanistic-interpretability-research) | ? | targeted sections checked |
| K | [Data Shapley: Equitable Valuation of Data for Machine Learning](https://arxiv.org/abs/1904.02868) | ? | targeted sections checked |
| K | [Datamodels: Predicting Predictions from Training Data](https://arxiv.org/abs/2202.00622) | ? | targeted sections checked |
| K | [Debugging Tests for Model Explanations](https://arxiv.org/abs/2011.05429) | ? | targeted sections checked |
| K | [Decomposing The Dark Matter of Sparse Autoencoders](https://arxiv.org/abs/2410.14670) | ? | targeted sections checked |
| K | [Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps](https://arxiv.org/abs/1312.6034) | ? | targeted sections checked |
| K | [Designing and Interpreting Probes with Control Tasks](https://arxiv.org/abs/1909.03368) | ? | targeted sections checked |
| K | [Diagnostics for Deep Neural Networks with Automated Copy/Paste Attacks](https://arxiv.org/abs/2211.10024) | ? | targeted sections checked |
| K | [Discovering Latent Knowledge in Language Models Without Supervision](https://arxiv.org/abs/2212.03827) | ? | targeted sections checked |
| K | [Dissecting Contextual Word Embeddings: Architecture and Representation](https://aclanthology.org/D18-1179/) | ? | targeted sections checked |
| K | [Dissecting Language Models: Machine Unlearning via Selective Pruning](https://arxiv.org/abs/2403.01267) | ? | targeted sections checked |
| K | [Dissecting Recall of Factual Associations in Auto-Regressive Language Models](https://arxiv.org/abs/2304.14767) | ? | targeted sections checked |
| K | [Do Concept Bottleneck Models Learn as Intended?](https://arxiv.org/abs/2105.04289) | ? | targeted sections checked |
| K | [Do Vision Transformers See Like Convolutional Neural Networks?](https://arxiv.org/abs/2108.08810) | ? | targeted sections checked |
| K | [Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla](https://arxiv.org/abs/2307.09458) | ? | targeted sections checked |
| k | [Does Localization Inform Editing? Surprising Differences in Causality-Based Localization vs. Knowledge Editing in Language Models](https://arxiv.org/abs/2301.04213) | ? | targeted sections checked |
| K | [Does Transformer Interpretability Transfer to RNNs?](https://arxiv.org/abs/2404.05971) | ? | targeted sections checked |
| K | [ERASER: A Benchmark to Evaluate Rationalized NLP Models](https://arxiv.org/abs/1911.03429) | ? | targeted sections checked |
| K | [Eliciting Latent Knowledge from Quirky Language Models](https://arxiv.org/abs/2312.01037) | ? | targeted sections checked |
| K | [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112) | ? | targeted sections checked |
| K | [Emergent Linear Representations in World Models of Self-Supervised Sequence Models](https://arxiv.org/abs/2309.00941) | ? | targeted sections checked |
| K | [Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task](https://arxiv.org/abs/2210.13382) | ? | targeted sections checked |
| K | [Estimating Training Data Influence by Tracing Gradient Descent](https://arxiv.org/abs/2002.08484) | ? | targeted sections checked |
| k | [Evaluating Feature Attribution Methods in the Image Domain](https://arxiv.org/abs/2202.12270) | ? | targeted sections checked |
| K+S | [Evaluating the Robustness of Interpretability Methods through Explanation Invariance and Equivariance](https://arxiv.org/abs/2304.06715) | ? | targeted sections checked |
| K | [Exemplary Natural Images Explain CNN Activations Better than State-of-the-Art Feature Visualization](https://arxiv.org/abs/2010.12606) | ? | targeted sections checked |
| k | [Explainable Graph Neural Networks Under Fire](https://arxiv.org/abs/2406.06417) | ? | targeted sections checked |
| k | [Explaining Explanations: Axiomatic Feature Interactions for Deep Networks](https://arxiv.org/abs/2002.04138) | ? | targeted sections checked |
| K | [Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations](https://arxiv.org/abs/1905.07697) | ? | targeted sections checked |
| K | [Explaining NonLinear Classification Decisions with Deep Taylor Decomposition](https://arxiv.org/abs/1512.02479) | ? | targeted sections checked |
| k | [Explaining by Removing: A Unified Framework for Model Explanation](https://arxiv.org/abs/2011.14878) | ? | targeted sections checked |
| k | [Explaining individual predictions when features are dependent: More accurate approximations to Shapley values](https://arxiv.org/abs/1903.10464) | ? | targeted sections checked |
| K | [Explanation in Artificial Intelligence: Insights from the Social Sciences](https://arxiv.org/abs/1706.07269) | ? | targeted sections checked |
| K | [Explanations can be manipulated and geometry is to blame](https://arxiv.org/abs/1906.07983) | ? | targeted sections checked |
| K | [FACE: Feasible and Actionable Counterfactual Explanations](https://arxiv.org/abs/1909.09369) | ? | targeted sections checked |
| K | [FIND: A Function Description Benchmark for Evaluating Interpretability Methods](https://arxiv.org/abs/2309.03886) | ? | targeted sections checked |
| k+s | [Faith-Shap: The Faithful Shapley Interaction Index](https://arxiv.org/abs/2203.00870) | ? | targeted sections checked |
| K | [Faithful Chain-of-Thought Reasoning](https://arxiv.org/abs/2301.13379) | ? | targeted sections checked |
| K | [Faithfulness Tests for Natural Language Explanations](https://arxiv.org/abs/2305.18029) | ? | targeted sections checked |
| K | [Faithfulness vs. Plausibility: On the (Un)Reliability of Explanations from Large Language Models](https://arxiv.org/abs/2402.04614) | ? | targeted sections checked |
| K | [FastSHAP: Real-Time Shapley Value Estimation](https://arxiv.org/abs/2107.07436) | ? | targeted sections checked |
| K | [Feature Visualization](https://distill.pub/2017/feature-visualization/) | ? | targeted sections checked |
| k | [Feature relevance quantification in explainable AI: A causal problem](https://arxiv.org/abs/1910.13413) | ? | targeted sections checked |
| K | [Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations](https://arxiv.org/abs/2303.02536) | ? | targeted sections checked |
| K | [Finding Neurons in a Haystack: Case Studies with Sparse Probing](https://arxiv.org/abs/2305.01610) | ? | targeted sections checked |
| K | [Finding Sparse Linear Connections between Features in LLMs](https://www.lesswrong.com/posts/7fxusXdkMNmAhkAfc/finding-sparse-linear-connections-between-features-in-llms) | ? | targeted sections checked |
| K | [Finding Transformer Circuits with Edge Pruning](https://arxiv.org/abs/2406.16778) | ? | targeted sections checked |
| K | [Fine-tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking](https://arxiv.org/abs/2402.14811) | ? | targeted sections checked |
| K | [Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods](https://arxiv.org/abs/1911.02508) | ? | targeted sections checked |
| K | [Fooling Neural Network Interpretations via Adversarial Model Manipulation](https://arxiv.org/abs/1902.02041) | ? | targeted sections checked |
| k | [Function Vectors in Large Language Models](https://arxiv.org/abs/2310.15213) | ? | targeted sections checked |
| K | [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894) | ? | targeted sections checked |
| K | [General Pitfalls of Model-Agnostic Interpretation Methods for Machine Learning Models](https://arxiv.org/abs/2007.04131) | ? | targeted sections checked |
| K | [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391) | ? | targeted sections checked |
| K | [Gradient Routing: Masking Gradients to Localize Computation in Neural Networks](https://arxiv.org/abs/2410.04332) | ? | targeted sections checked |
| k | [GraphFramEx: Towards Systematic Evaluation of Explainability Methods for Graph Neural Networks](https://arxiv.org/abs/2206.09677) | ? | targeted sections checked |
| K | [GraphLIME: Local Interpretable Model Explanations for Graph Neural Networks](https://arxiv.org/abs/2001.06216) | ? | targeted sections checked |
| K | [Grokking Group Multiplication with Cosets](https://arxiv.org/abs/2312.06581) | ? | targeted sections checked |
| K | [How Can We Know What Language Models Know?](https://arxiv.org/abs/1911.12543) | ? | targeted sections checked |
| K | [How do Decisions Emerge across Layers in Neural Models? Interpretation with Differentiable Masking](https://arxiv.org/abs/2004.14992) | ? | targeted sections checked |
| K | [How do Humans Understand Explanations from Machine Learning Systems? An Evaluation of the Human-Interpretability of Explanation](https://arxiv.org/abs/1802.00682) | ? | targeted sections checked |
| K | [How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model](https://arxiv.org/abs/2305.00586) | ? | targeted sections checked |
| k | [How to use and interpret activation patching](https://arxiv.org/abs/2404.15255) | ? | targeted sections checked |
| K | [Hypothesis Testing the Circuit Hypothesis in LLMs](https://arxiv.org/abs/2410.13032) | ? | targeted sections checked |
| K | [Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning](https://arxiv.org/abs/2405.12241) | ? | targeted sections checked |
| K | [ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness](https://arxiv.org/abs/1811.12231) | ? | targeted sections checked |
| K | [ImageNet-trained CNNs are not biased towards texture: Revisiting feature reliance through controlled suppression](https://arxiv.org/abs/2509.20234) | ? | targeted sections checked |
| K | [Improving Dictionary Learning with Gated Sparse Autoencoders](https://arxiv.org/abs/2404.16014) | ? | targeted sections checked |
| k | [In-context Learning and Induction Heads](https://arxiv.org/abs/2209.11895) | ? | targeted sections checked |
| K | [Inducing Causal Structure for Interpretable Neural Networks](https://arxiv.org/abs/2112.00826) | ? | targeted sections checked |
| K | [Inference-Time Intervention: Eliciting Truthful Answers from a Language Model](https://arxiv.org/abs/2306.03341) | ? | targeted sections checked |
| K | [Influence Functions in Deep Learning Are Fragile](https://arxiv.org/abs/2006.14651) | ? | targeted sections checked |
| K | [Information-Theoretic Probing for Linguistic Structure](https://aclanthology.org/2020.acl-main.420/) | ? | targeted sections checked |
| K | [Information-Theoretic Probing with Minimum Description Length](https://arxiv.org/abs/2003.12298) | ? | targeted sections checked |
| K+S | [Input Switched Affine Networks: An RNN Architecture Designed for Interpretability](https://arxiv.org/abs/1611.09434) | ? | targeted sections checked |
| K | [Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)](https://arxiv.org/abs/1711.11279) | ? | targeted sections checked |
| K | [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593) | ? | targeted sections checked |
| K | [Interpretable Explanations of Black Boxes by Meaningful Perturbation](https://arxiv.org/abs/1704.03296) | ? | targeted sections checked |
| K | [Interpretation of Neural Networks is Fragile](https://arxiv.org/abs/1710.10547) | ? | targeted sections checked |
| K | [Intriguing properties of neural networks](https://arxiv.org/abs/1312.6199) | ? | targeted sections checked |
| k+s | [Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching](https://arxiv.org/abs/2311.17030) | ? | targeted sections checked |
| k+s | [Isolating Sources of Disentanglement in VAEs](https://arxiv.org/abs/1802.04942) | ? | targeted sections checked |
| K | [Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders](https://arxiv.org/abs/2407.14435) | ? | targeted sections checked |
| K | [Knowledge Neurons in Pretrained Transformers](https://arxiv.org/abs/2104.08696) | ? | targeted sections checked |
| k+s | [LEACE: Perfect linear concept erasure in closed form](https://arxiv.org/abs/2306.03819) | ? | targeted sections checked |
| K | [Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting](https://arxiv.org/abs/2305.04388) | ? | targeted sections checked |
| K | [Language Models Implement Simple Word2Vec-style Vector Arithmetic](https://arxiv.org/abs/2305.16130) | ? | targeted sections checked |
| K | [Language Models Represent Space and Time](https://arxiv.org/abs/2310.02207) | ? | targeted sections checked |
| K | [Language Models as Knowledge Bases?](https://arxiv.org/abs/1909.01066) | ? | targeted sections checked |
| K | [Language models can explain neurons in language models](https://openaipublic.blob.core.windows.net/neuron-explainer/paper/index.html) | ? | targeted sections checked |
| K | [Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Large_Multi-modal_Models_Can_Interpret_Features_in_Large_Multi-modal_Models_ICCV_2025_paper.html) | ? | targeted sections checked |
| k+s | [Learning Important Features Through Propagating Activation Differences](https://arxiv.org/abs/1704.02685) | ? | targeted sections checked |
| K | [Learning Multi-Level Features with Matryoshka Sparse Autoencoders](https://arxiv.org/abs/2503.17547) | ? | targeted sections checked |
| k+s | [Learning Sparse Neural Networks through L0 Regularization](https://arxiv.org/abs/1712.01312) | ? | targeted sections checked |
| K | [Learning to Deceive with Attention-Based Explanations](https://aclanthology.org/2020.acl-main.432/) | ? | targeted sections checked |
| K | [Learning to Explain: An Information-Theoretic Perspective on Model Interpretation](https://arxiv.org/abs/1802.07814) | ? | targeted sections checked |
| k | [Learning to Faithfully Rationalize by Construction](https://arxiv.org/abs/2005.00115) | ? | targeted sections checked |
| K | [Linear Adversarial Concept Erasure](https://arxiv.org/abs/2201.12091) | ? | targeted sections checked |
| k | [Linguistic Knowledge and Transferability of Contextual Representations](https://arxiv.org/abs/1903.08855) | ? | targeted sections checked |
| K | [Linguistic Regularities in Continuous Space Word Representations](https://aclanthology.org/N13-1090/) | ? | targeted sections checked |
| K | [Localizing Model Behavior with Path Patching](https://arxiv.org/abs/2304.05969) | ? | targeted sections checked |
| K | [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) | ? | targeted sections checked |
| K | [Locating and Editing Factual Associations in Mamba](https://arxiv.org/abs/2404.03646) | ? | targeted sections checked |
| K | [MIB: A Mechanistic Interpretability Benchmark](https://arxiv.org/abs/2504.13151) | ? | targeted sections checked |
| K | [Manipulating and Measuring Model Interpretability](https://arxiv.org/abs/1802.07810) | ? | targeted sections checked |
| K | [Mass-Editing Memory in a Transformer](https://arxiv.org/abs/2210.07229) | ? | targeted sections checked |
| K | [Meaningfully Debugging Model Mistakes using Conceptual Counterfactual Explanations](https://arxiv.org/abs/2106.12723) | ? | targeted sections checked |
| K | [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702) | ? | targeted sections checked |
| K | [Mechanistic Interpretability for AI Safety -- A Review](https://arxiv.org/abs/2404.14082) | ? | targeted sections checked |
| K | [Mechanistic?](https://arxiv.org/abs/2410.09087) | ? | targeted sections checked |
| K | [Missed Causes and Ambiguous Effects: Counterfactuals Pose Challenges for Interpreting Neural Networks](https://arxiv.org/abs/2407.04690) | ? | targeted sections checked |
| K | [Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation](https://arxiv.org/abs/2503.11926) | ? | targeted sections checked |
| K | [Multifaceted Feature Visualization: Uncovering the Different Types of Features Learned By Each Neuron in Deep Neural Networks](https://arxiv.org/abs/1602.03616) | ? | targeted sections checked |
| K | [Multimodal Neurons in Artificial Neural Networks](https://distill.pub/2021/multimodal-neurons/) | ? | targeted sections checked |
| K | [Natural Language Descriptions of Deep Visual Features](https://arxiv.org/abs/2201.11114) | ? | targeted sections checked |
| K | [Network Dissection: Quantifying Interpretability of Deep Visual Representations](https://arxiv.org/abs/1704.05796) | ? | targeted sections checked |
| K | [Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://arxiv.org/abs/2004.13912) | ? | targeted sections checked |
| k | [Neural and Conceptual Interpretation of PDP Models](https://web.stanford.edu/~jlmcc/papers/PDP/Volume%202/Chap22_PDP86.pdf) | ? | targeted sections checked |
| k | [Neuron Shapley: Discovering the Responsible Neurons](https://arxiv.org/abs/2002.09815) | ? | targeted sections checked |
| K | [Noise or Signal: The Role of Image Backgrounds in Object Recognition](https://arxiv.org/abs/2006.09994) | ? | targeted sections checked |
| K | [Not All Language Model Features Are One-Dimensionally Linear](https://arxiv.org/abs/2405.14860) | ? | targeted sections checked |
| K | [Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection](https://arxiv.org/abs/2004.07667) | ? | targeted sections checked |
| K | [Object Detectors Emerge in Deep Scene CNNs](https://arxiv.org/abs/1412.6856) | ? | targeted sections checked |
| K | [On Completeness-aware Concept-Based Explanations in Deep Neural Networks](https://arxiv.org/abs/1910.07969) | ? | targeted sections checked |
| k+s | [On Counterfactual Explanations under Predictive Multiplicity](https://arxiv.org/abs/2006.13132) | ? | targeted sections checked |
| K | [On Explainability of Graph Neural Networks via Subgraph Explorations](https://arxiv.org/abs/2102.05152) | ? | targeted sections checked |
| K | [On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140) | ? | targeted sections checked |
| K | [On the (In)fidelity and Sensitivity of Explanations](https://arxiv.org/abs/1901.09392) | ? | targeted sections checked |
| K | [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html) | ? | targeted sections checked |
| s | [On the Complexity of Neural Computation in Superposition](https://arxiv.org/abs/2409.15318) | ? | targeted sections checked |
| K | [On the Robustness of Interpretability Methods](https://arxiv.org/abs/1806.08049) | ? | targeted sections checked |
| K | [Open Problems in Mechanistic Interpretability](https://arxiv.org/abs/2501.16496) | ? | targeted sections checked |
| K | [Opening the AI Black Box: Program Synthesis via Mechanistic Interpretability](https://arxiv.org/abs/2402.05110) | ? | targeted sections checked |
| k+s | [Optimal Sparse Decision Trees](https://arxiv.org/abs/1904.12847) | ? | targeted sections checked |
| k+s | [PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks](https://arxiv.org/abs/2010.05788) | ? | targeted sections checked |
| K | [Parameterized Explainer for Graph Neural Network](https://arxiv.org/abs/2011.04573) | ? | targeted sections checked |
| K | [Pathologies of Neural Models Make Interpretations Difficult](https://aclanthology.org/D18-1407/) | ? | targeted sections checked |
| K | [Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation](https://arxiv.org/abs/1309.6392) | ? | targeted sections checked |
| K | [Preserving Causal Constraints in Counterfactual Explanations for Machine Learning Classifiers](https://arxiv.org/abs/1912.03277) | ? | targeted sections checked |
| K | [Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video](https://arxiv.org/abs/2504.19475) | ? | targeted sections checked |
| K | [Probabilistically Robust Recourse: Navigating the Trade-offs between Costs and Robustness in Algorithmic Recourse](https://arxiv.org/abs/2203.06768) | ? | targeted sections checked |
| K | [Probing Classifiers: Promises, Shortcomings, and Advances](https://arxiv.org/abs/2102.12452) | ? | targeted sections checked |
| K | [Probing for semantic evidence of composition by means of simple classification tasks](https://aclanthology.org/W16-2524/) | ? | targeted sections checked |
| K | [Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance?](https://aclanthology.org/2021.eacl-main.295/) | ? | targeted sections checked |
| k | [Problems with Shapley-value-based explanations as feature importance measures](https://arxiv.org/abs/2002.11097) | ? | targeted sections checked |
| k+s | [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217) | ? | targeted sections checked |
| K | [Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928) | ? | targeted sections checked |
| K | [Quantifying Feature Space Universality Across Large Language Models via Sparse Autoencoders](https://arxiv.org/abs/2410.06981) | ? | targeted sections checked |
| K | [RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations](https://arxiv.org/abs/2402.17700) | ? | targeted sections checked |
| K | [RISE: Randomized Input Sampling for Explanation of Black-box Models](https://arxiv.org/abs/1806.07421) | ? | targeted sections checked |
| K | [Rationalizing Neural Predictions](https://arxiv.org/abs/1606.04155) | ? | targeted sections checked |
| K | [Reasoning Models Don't Always Say What They Think](https://arxiv.org/abs/2505.05410) | ? | targeted sections checked |
| K | [Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks](https://arxiv.org/abs/2307.02477) | ? | targeted sections checked |
| K | [Red Teaming Deep Neural Networks with Feature Synthesis Tools](https://arxiv.org/abs/2302.10894) | ? | targeted sections checked |
| K | [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717) | ? | targeted sections checked |
| k | [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405) | ? | targeted sections checked |
| K | [Representer Point Selection for Explaining Deep Neural Networks](https://arxiv.org/abs/1811.09720) | ? | targeted sections checked |
| K | [Revisiting Sanity Checks for Saliency Maps](https://arxiv.org/abs/2110.14297) | ? | targeted sections checked |
| K | [Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations](https://arxiv.org/abs/1703.03717) | ? | targeted sections checked |
| K | [Rigorously Assessing Natural Language Explanations of Neurons](https://aclanthology.org/2023.blackboxnlp-1.24/) | ? | targeted sections checked |
| K | [SAE-V: Interpreting Multimodal Models for Enhanced Alignment](https://arxiv.org/abs/2502.17514) | ? | targeted sections checked |
| K | [SAEBench: A Comprehensive Benchmark for Sparse Autoencoders in Language Model Interpretability](https://arxiv.org/abs/2503.09532) | ? | targeted sections checked |
| k | [SAGE: Scalable Ground Truth Evaluations for Large Sparse Autoencoders](https://arxiv.org/abs/2410.07456) | ? | targeted sections checked |
| K | [SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability](https://arxiv.org/abs/1706.05806) | ? | targeted sections checked |
| K | [Sanity Checks for Saliency Maps](https://arxiv.org/abs/1810.03292) | ? | targeted sections checked |
| K | [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://arxiv.org/abs/2605.29358) | ? | targeted sections checked |
| k+l | [Scaling and evaluating sparse autoencoders](https://arxiv.org/abs/2406.04093) | ? | targeted sections checked |
| K | [Seeing Is Believing: Brain-Inspired Modular Training for Mechanistic Interpretability](https://arxiv.org/abs/2305.08746) | ? | targeted sections checked |
| K | [Shape Representation in Parallel Systems](https://ijcai.org/proceedings/1981/217) | ? | targeted sections checked |
| K | [Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780) | ? | targeted sections checked |
| k+s | [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414) | ? | targeted sections checked |
| K | [Skeletonization: A Technique for Trimming the Fat from a Network via Relevance Assessment](https://proceedings.neurips.cc/paper_files/paper/1988/file/07e1cd7dca89a1678042477183b7ac3f-Paper.pdf) | ? | targeted sections checked |
| K | [SmoothGrad: removing noise by adding noise](https://arxiv.org/abs/1706.03825) | ? | targeted sections checked |
| k | [Sparse Autoencoders Do Not Find Canonical Units of Analysis](https://arxiv.org/abs/2502.04878) | ? | targeted sections checked |
| K | [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600) | ? | targeted sections checked |
| K | [Sparse Crosscoders for Cross-Layer Features and Model Diffing](https://transformer-circuits.pub/2024/crosscoders/index.html) | ? | targeted sections checked |
| K | [Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](https://arxiv.org/abs/2403.19647) | ? | targeted sections checked |
| K | [Steering Language Models With Activation Engineering](https://arxiv.org/abs/2308.10248) | ? | targeted sections checked |
| K | [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) | ? | targeted sections checked |
| K | [Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead](https://arxiv.org/abs/1811.10154) | ? | targeted sections checked |
| K | [Studying Large Language Model Generalization with Influence Functions](https://arxiv.org/abs/2308.03296) | ? | targeted sections checked |
| K | [Synthesizing the preferred inputs for neurons in neural networks via deep generator networks](https://arxiv.org/abs/1605.09304) | ? | targeted sections checked |
| K | [TRAK: Attributing Model Behavior at Scale](https://arxiv.org/abs/2303.14186) | ? | targeted sections checked |
| K | [The 'Strong' Feature Hypothesis could be wrong.](https://www.alignmentforum.org/posts/tojtPCCRpKLSHBdpn/the-strong-feature-hypothesis-could-be-wrong) | ? | targeted sections checked |
| K | [The (Un)reliability of saliency methods](https://arxiv.org/abs/1711.00867) | ? | targeted sections checked |
| K | [The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/) | ? | targeted sections checked |
| k | [The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective](https://arxiv.org/abs/2202.01602) | ? | targeted sections checked |
| K | [The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets](https://arxiv.org/abs/2310.06824) | ? | targeted sections checked |
| K | [The Hydra Effect: Emergent Self-repair in Language Model Computations](https://arxiv.org/abs/2307.15771) | ? | targeted sections checked |
| s | [The Impossibility of Eliciting Latent Knowledge](https://arxiv.org/abs/2606.12268) | ? | targeted sections checked |
| k+s | [The Linear Representation Hypothesis and the Geometry of Large Language Models](https://arxiv.org/abs/2311.03658) | ? | targeted sections checked |
| K | [The Local Interaction Basis: Identifying Computationally-Relevant and Sparsely Interacting Features in Neural Networks](https://arxiv.org/abs/2405.10928) | ? | targeted sections checked |
| k | [The Many Shapley Values for Model Explanation](https://arxiv.org/abs/1908.08474) | ? | targeted sections checked |
| K | [The Mythos of Model Interpretability](https://arxiv.org/abs/1606.03490) | ? | targeted sections checked |
| K | [The Quest for the Right Mediator: A History, Survey, and Theoretical Grounding of Causal Interpretability](https://arxiv.org/abs/2408.01416) | ? | targeted sections checked |
| k | [The elephant in the interpretability room: Why use attention as explanation when we have saliency methods?](https://arxiv.org/abs/2010.05607) | ? | targeted sections checked |
| K | [The importance of mixed selectivity in complex cognitive tasks](https://doi.org/10.1038/nature12160) | ? | targeted sections checked |
| K | [The mechanistic basis of data dependence and abrupt learning in an in-context classification task](https://arxiv.org/abs/2312.03002) | ? | targeted sections checked |
| K | [Toward Transparent AI: A Survey on Interpreting the Inner Structures of Deep Neural Networks](https://arxiv.org/abs/2207.13243) | ? | targeted sections checked |
| k | [Towards A Rigorous Science of Interpretable Machine Learning](https://arxiv.org/abs/1702.08608) | ? | targeted sections checked |
| K | [Towards Automated Circuit Discovery for Mechanistic Interpretability](https://arxiv.org/abs/2304.14997) | ? | targeted sections checked |
| K | [Towards Automatic Concept-based Explanations](https://arxiv.org/abs/1902.03129) | ? | targeted sections checked |
| K | [Towards Best Practices of Activation Patching in Language Models: Metrics and Methods](https://arxiv.org/abs/2309.16042) | ? | targeted sections checked |
| K | [Towards Faithful Model Explanation in NLP: A Survey](https://arxiv.org/abs/2209.11326) | ? | targeted sections checked |
| K | [Towards Faithfully Interpretable NLP Systems: How should we define and evaluate faithfulness?](https://arxiv.org/abs/2004.03685) | ? | targeted sections checked |
| K | [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html) | ? | targeted sections checked |
| K | [Towards Principled Evaluations of Sparse Autoencoders for Interpretability and Control](https://arxiv.org/abs/2405.08366) | ? | targeted sections checked |
| K | [Towards Robust Interpretability with Self-Explaining Neural Networks](https://arxiv.org/abs/1806.07538) | ? | targeted sections checked |
| K | [Towards Robust and Reliable Algorithmic Recourse](https://arxiv.org/abs/2102.13620) | ? | targeted sections checked |
| K | [Towards Vision-Language Mechanistic Interpretability: A Causal Tracing Tool for BLIP](https://openaccess.thecvf.com/content/ICCV2023W/CLVL/html/Palit_Towards_Vision-Language_Mechanistic_Interpretability_A_Causal_Tracing_Tool_for_BLIP_ICCVW_2023_paper.html) | ? | targeted sections checked |
| k | [Towards better understanding of gradient-based attribution methods for Deep Neural Networks](https://arxiv.org/abs/1711.06104) | ? | targeted sections checked |
| k | [Towards falsifiable interpretability research](https://arxiv.org/abs/2010.12016) | ? | targeted sections checked |
| k+s | [Toy Models of Superposition](https://arxiv.org/abs/2209.10652) | ? | targeted sections checked |
| K | [Tracr: Compiled Transformers as a Laboratory for Interpretability](https://arxiv.org/abs/2301.05062) | ? | targeted sections checked |
| K | [Transcoders Find Interpretable LLM Feature Circuits](https://arxiv.org/abs/2406.11944) | ? | targeted sections checked |
| K | [Transformer Circuit Faithfulness Metrics Are Not Robust](https://arxiv.org/abs/2407.08734) | ? | targeted sections checked |
| K | [Transformer Feed-Forward Layers Are Key-Value Memories](https://arxiv.org/abs/2012.14913) | ? | targeted sections checked |
| K | [Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space](https://arxiv.org/abs/2203.14680) | ? | targeted sections checked |
| K | [Transformer Visualization via Dictionary Learning: Contextualized Embedding as a Linear Superposition of Transformer Factors](https://aclanthology.org/2021.deelio-1.1/) | ? | targeted sections checked |
| K | [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958) | ? | targeted sections checked |
| k+s | [Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730) | ? | targeted sections checked |
| K | [Understanding Deep Image Representations by Inverting Them](https://arxiv.org/abs/1412.0035) | ? | targeted sections checked |
| K | [Understanding Deep Networks via Extremal Perturbations and Smooth Masks](https://arxiv.org/abs/1910.08485) | ? | targeted sections checked |
| k | [Understanding Global Feature Contributions With Additive Importance Measures](https://arxiv.org/abs/2004.00668) | ? | targeted sections checked |
| K | [Understanding Neural Networks Through Deep Visualization](https://arxiv.org/abs/1506.06579) | ? | targeted sections checked |
| K | [Understanding Neural Networks through Representation Erasure](https://arxiv.org/abs/1612.08220) | ? | targeted sections checked |
| K | [Understanding intermediate layers using linear classifier probes](https://arxiv.org/abs/1610.01644) | ? | targeted sections checked |
| K | [Understanding the Role of Individual Units in a Deep Neural Network](https://arxiv.org/abs/2009.05041) | ? | targeted sections checked |
| K | [Unmasking Clever Hans Predictors and Assessing What Machines Really Learn](https://arxiv.org/abs/1902.10178) | ? | targeted sections checked |
| K | [Unrestricted Permutation forces Extrapolation: Variable Importance Requires at least One More Model, or There Is No Free Variable Importance](https://arxiv.org/abs/1905.03151) | ? | targeted sections checked |
| K | [Unstable Features, Reproducible Subspaces: Understanding Seed Dependence in Sparse Autoencoders](https://arxiv.org/abs/2606.12138) | ? | targeted sections checked |
| K | [Visualisation and ‘Diagnostic Classifiers’ Reveal how Recurrent and Recursive Neural Networks Process Hierarchical Structure](https://arxiv.org/abs/1711.10203) | ? | targeted sections checked |
| K | [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) | ? | targeted sections checked |
| K | [Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models](https://arxiv.org/abs/1612.08468) | ? | targeted sections checked |
| K+S | [Weight-Based Decomposition: A Case for Bilinear MLPs](https://arxiv.org/abs/2406.03947) | ? | targeted sections checked |
| K | [What Does BERT Look At? An Analysis of BERT's Attention](https://arxiv.org/abs/1906.04341) | ? | targeted sections checked |
| K | [What Is One Grain of Sand in the Desert? Analyzing Individual Neurons in Deep NLP Models](https://arxiv.org/abs/1812.09355) | ? | targeted sections checked |
| k | [What made you do this? Understanding black-box decisions with sufficient input subsets](https://arxiv.org/abs/1810.03805) | ? | targeted sections checked |
| k | [What you can cram into a single $&!#* vector: Probing sentence embeddings for linguistic properties](https://arxiv.org/abs/1805.01070) | ? | targeted sections checked |
| K | [What’s in an Embedding? Analyzing Word Embeddings through Multilingual Evaluation](https://aclanthology.org/D15-1246/) | ? | targeted sections checked |
| K | [When Chain of Thought is Necessary, Language Models Struggle to Evade Monitors](https://arxiv.org/abs/2507.05246) | ? | targeted sections checked |
| K | [Why Should I Trust You?: Explaining the Predictions of Any Classifier](https://arxiv.org/abs/1602.04938) | ? | targeted sections checked |
| K | [Zoom In: An Introduction to Circuits](https://distill.pub/2020/circuits/zoom-in/) | ? | targeted sections checked |
| K | [Zorro: Valid, Sparse, and Stable Explanations in Graph Neural Networks](https://arxiv.org/abs/2105.08621) | ? | targeted sections checked |
| K | [[Interim research report] Taking features out of superposition with sparse autoencoders](https://www.conjecture.dev/research/interim-research-report-taking-features-out-of-superposition-with-sparse-autoencoders) | ? | targeted sections checked |
| K | [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) | ? | targeted sections checked |
| K | [k-Sparse Autoencoders](https://arxiv.org/abs/1312.5663) | ? | targeted sections checked |
