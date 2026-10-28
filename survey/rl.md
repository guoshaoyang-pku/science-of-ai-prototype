# Survey — Reinforcement learning

*RL post-training and classic RL theory. Snapshot: 2026-10-28. 271 papers, 265 concepts tracked.*

Level counts (a paper can hold several; **bold** = full claim at that level): **176 knowledge · 3 law · 35 science**, plus 43+1+65 conditional / candidate / fragment tags.

Levels: **K** = verified observation in a stated setting; **L** = quantitative relation with fit/test separation and an explicit validity boundary; **S** = mechanism deriving phenomena from explicit assumptions. **Lowercase** (k/l/s) marks conditional theorems, candidates, and fragments — claims whose stated scope falls short of the full gate. The hierarchy grades the claim, not the worth of the paper. Review depth is recorded per entry; *targeted/selected sections checked* means exactly that — not a full-text review.

## Highlighted law/science entries

- **[Average Cost Temporal-Difference Learning](https://web.stanford.edu/~bvr/pubs/average-td.pdf)** (1999) — *Science*. Thm. 1: almost-sure convergence of average-cost TD(lambda) — the average-cost estimate tends to the true mean, linear parameters to a unique projected fixed point. *Boundary:* Fixed finite irreducible aperiodic Markov chain; full-column-rank linear features whose span excludes the all-ones vector; lambda in [0,1), Robbins–Monro stepsizes. Policy evaluation only — not a guarantee for nonlinear approximation or control.
- **[Learning Algorithms for Markov Decision Processes with Average Cost](https://www.mit.edu/~dimitrib/q_learn.pdf)** (2001) — *Science*. Convergence proofs for average-cost learning: relative-value-iteration Q-learning and a stochastic-shortest-path reduction with two-timescale updates, under stated visitation and normalization conditions. *Boundary:* Finite state/action; every stationary policy has a single recurrent class with a common recurrent reference state; per-variant simulation and boundedness conditions.
- **[Simulation-Based Optimization of Markov Reward Processes](https://www.mit.edu/~jnt/Papers/J083-01-mar-MDP.pdf)** (2001) — *Science*. Likelihood-ratio average-reward updates over regenerative cycles of a single trajectory; Prop. 3: performance iterates converge and the performance gradient tends to zero. *Boundary:* Finite parameterized Markov reward processes; Assumptions 1–4 (aperiodicity, common regeneration state, twice-differentiable bounded rewards/transitions, square-summable stepsizes).
- **[Infinite-Horizon Policy-Gradient Estimation](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume15/baxter01a.pdf)** (2001) — *Science*. GPOMDP estimates the average-reward gradient from a single partially observed trajectory via decaying eligibility traces; for fixed beta<1 the estimate converges to a biased gradient object that tends to the true gradient as beta→1. *Boundary:* Finite hidden states/actions/observations; fixed differentiable stochastic policy; unique stationary distribution per parameter; bounded rewards and score ratios.
- **[DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)** — *Knowledge*. RL-first post-training (GRPO over verified answers) elicits long chain-of-thought reasoning; R1-Zero shows emergent reasoning without SFT, with documented readability failures. *Boundary:* Specific base models, reward rules, and training recipes; the emergence observations are tied to the reported setting.

## Full classification table

| Level | Paper | Year | Review depth |
|---|---|---|---|
| K | [Some Studies in Machine Learning Using the Game of Checkers](https://people.eecs.berkeley.edu/~russell/classes/cs294/s11/readings/Samuel:1959.pdf) | 1959 | targeted primary method and scope checked |
| K | [Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problems](https://www.cs.colostate.edu/~anderson/newsite/publications/barto-sutton-anderson-83.pdf) | 1983 | targeted primary method and scope checked |
| K | [A Reinforcement Learning Method for Maximizing Undiscounted Rewards](https://doi.org/10.1016/B978-1-55860-307-3.50045-9) | 1993 | targeted author uploaded remote pdf text checked |
| K | [Temporal Difference Learning and TD-Gammon](https://doi.org/10.1145/203330.203343) | 1995 | targeted licensed original article transcription checked |
| k | [Average Reward Reinforcement Learning: Foundations, Algorithms, and Empirical Results](https://doi.org/10.1007/BF00114727) | 1996 | targeted primary method and scope checked |
| S | [Average Cost Temporal-Difference Learning](https://web.stanford.edu/~bvr/pubs/average-td.pdf) | 1999 | targeted theorem statements and assumptions checked |
| S | [Infinite-Horizon Policy-Gradient Estimation](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume15/baxter01a.pdf) | 2001 | targeted theorem statements and assumptions checked |
| S | [Learning Algorithms for Markov Decision Processes with Average Cost](https://www.mit.edu/~dimitrib/q_learn.pdf) | 2001 | targeted theorem statements and assumptions checked |
| S | [Simulation-Based Optimization of Markov Reward Processes](https://www.mit.edu/~jnt/Papers/J083-01-mar-MDP.pdf) | 2001 | targeted theorem statements and assumptions checked |
| k+s | [(More) Efficient Reinforcement Learning via Posterior Sampling](https://arxiv.org/abs/1306.0940) | ? | targeted primary sections checked |
| K | [A Closer Look at Deep Policy Gradients](https://arxiv.org/abs/1811.02553) | ? | method/proof/empirical sections checked |
| K | [A Comprehensive Survey on Safe Reinforcement Learning](https://www.jmlr.org/papers/v16/garcia15a.html) | ? | targeted survey taxonomy and reference chain checked |
| k+s | [A Convergent O(n) Algorithm for Off-policy Temporal-difference Learning with Linear Function Approximation](https://proceedings.neurips.cc/paper_files/paper/2008/hash/e0c641195b27425bb056ac56f8953d24-Abstract.html) | ? | targeted primary sections checked |
| s | [A Counterexample in Stochastic Optimum Control](https://doi.org/10.1137/0306011) | ? | official abstract plus non author mirror claim location |
| K | [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887) | ? | method/proof/empirical sections checked |
| s | [A Finite Time Analysis of Temporal Difference Learning With Linear Function Approximation](https://arxiv.org/abs/1806.02450) | ? | method/proof/empirical sections checked |
| k+s | [A General Theoretical Paradigm to Understand Learning from Human Preferences](https://arxiv.org/abs/2310.12036) | ? | targeted primary sections checked |
| K | [A Laplacian Framework for Option Discovery in Reinforcement Learning](https://arxiv.org/abs/1703.00956) | ? | method/proof/empirical sections checked |
| K | [A Lyapunov-based Approach to Safe Reinforcement Learning](https://arxiv.org/abs/1805.07708) | ? | targeted primary sections checked |
| K | [A Minimalist Approach to Offline Reinforcement Learning](https://arxiv.org/abs/2106.06860) | ? | method/proof/empirical sections checked |
| K | [A Natural Policy Gradient](https://papers.nips.cc/paper_files/paper/2001/hash/4b86abe48d358ecf194c56c69108433e-Abstract.html) | ? | method/proof/empirical sections checked |
| S | [A New Approach to Linear Filtering and Prediction Problems](https://doi.org/10.1115/1.3662552) | ? | targeted primary reprint model and derivation checked |
| K | [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://arxiv.org/abs/1011.0686) | ? | method/proof/empirical sections checked |
| k+s | [A Sparse Sampling Algorithm for Near-Optimal Planning in Large Markov Decision Processes](https://ijcai.org/proceedings/1999/188) | ? | targeted primary sections checked |
| K+S | [A Survey of Partially Observable Markov Decision Processes: Theory, Models, and Algorithms](https://pubsonline.informs.org/doi/10.1287/mnsc.28.1.1) | ? | selected primary survey model and statement sections checked |
| k+s | [A Theoretical and Empirical Analysis of Expected Sarsa](https://www.ai.rug.nl/~mwiering/GROUP/ARTICLES/expected_sarsa.pdf) | ? | proof statements checked |
| k+s | [A Theory of Regularized Markov Decision Processes](https://arxiv.org/abs/1901.11275) | ? | targeted primary sections checked |
| K | [A Tour of Reinforcement Learning: The View from Continuous Control](https://arxiv.org/abs/1806.09460) | ? | targeted survey sections and reference chain checked |
| K | [A Tutorial on Thompson Sampling](https://arxiv.org/abs/1707.02038) | ? | proof statements checked |
| K | [AWAC: Accelerating Online Reinforcement Learning with Offline Datasets](https://arxiv.org/abs/2006.09359) | ? | method/proof/empirical sections checked |
| S | [Actor-Critic Algorithms](https://proceedings.neurips.cc/paper/1999/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) | ? | method/proof/empirical sections checked |
| K | [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) | ? | method/proof/empirical sections checked |
| K | [Agent57: Outperforming the Atari Human Benchmark](https://arxiv.org/abs/2003.13350) | ? | method/proof/empirical sections checked |
| K+S | [Algorithms for Inverse Reinforcement Learning](https://www.cl.cam.ac.uk/~ey204/teaching/ACS/R244_2022_2023/papers/NG_ICML_2000.pdf) | ? | proof statements checked |
| k+s | [An Analysis of Categorical Distributional Reinforcement Learning](https://arxiv.org/abs/1802.08163) | ? | targeted primary sections checked |
| s | [An Analysis of Temporal-Difference Learning with Function Approximation](https://web.mit.edu/~jnt/www/Papers/J063-97-bvr-td.pdf) | ? | targeted primary sections checked |
| k+s | [An Emphatic Approach to the Problem of Off-policy Temporal-Difference Learning](https://arxiv.org/abs/1503.04269) | ? | proof statements checked |
| k+s | [Apprenticeship Learning via Inverse Reinforcement Learning](https://www.cs.cmu.edu/~jeanoh/16-785/papers/abbeel-icml2004-apprentice.pdf) | ? | method/proof/empirical sections checked |
| k+s | [Approximately Optimal Approximate Reinforcement Learning](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/KakadeLangford-icml2002.pdf) | ? | proof statements checked |
| K | [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) | ? | method/proof/empirical sections checked |
| s | [Asynchronous Stochastic Approximation and Q-Learning](https://www.mit.edu/~jnt/Papers/J052-94-jnt-q.pdf) | ? | proof statements checked |
| K | [Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs](https://arxiv.org/abs/2402.14740) | ? | targeted method sections |
| k+s | [Bandit based Monte-Carlo Planning](https://doi.org/10.1007/11871842_29) | ? | proof statements checked |
| K | [Batch Policy Learning under Constraints](https://arxiv.org/abs/1903.08738) | ? | targeted primary sections checked |
| K | [Benchmarking Batch Deep Reinforcement Learning Algorithms](https://arxiv.org/abs/1910.01708) | ? | targeted primary sections checked |
| K | [Benchmarks for Deep Off-Policy Evaluation](https://arxiv.org/abs/2103.16596) | ? | targeted primary sections checked |
| K | [Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning](https://www.sciencedirect.com/science/article/pii/S0004370299000521) | ? | proof statements checked |
| k+s | [Bridging the Gap Between Value and Policy Based Reinforcement Learning](https://arxiv.org/abs/1702.08892) | ? | method/proof/empirical sections checked |
| K | [Buy 4 REINFORCE Samples, Get a Baseline for Free!](https://wouterkool.github.io/publication/buy-4-samples-free-baseline/) | ? | source abstract only |
| s | [Certainty Equivalence is Efficient for Linear Quadratic Control](https://papers.neurips.cc/paper_files/paper/2019/hash/5dbc8390f17e019d300d5a162c3ce3bc-Abstract.html) | ? | targeted primary definitions assumptions and theorem stateme |
| S | [Coherent Measures of Risk](https://doi.org/10.1111/1467-9965.00068) | ? | selected primary axioms representation and counterexample re |
| k+s | [Combining policy gradient and Q-learning](https://arxiv.org/abs/1611.01626) | ? | proof statements checked |
| s | [Computationally Feasible Bounds for Partially Observed Markov Decision Processes](https://pubsonline.informs.org/doi/10.1287/opre.39.1.162) | ? | source abstract only |
| S | [Conditional Value-at-Risk for General Loss Distributions](https://doi.org/10.1016/S0378-4266(02)00271-6) | ? | selected primary atom boundary definition and variational th |
| K | [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779) | ? | targeted theorem and algorithm sections |
| K | [Constrained Policy Optimization](https://arxiv.org/abs/1705.10528) | ? | targeted method sections |
| S | [Contextual Decision Processes with Low Bellman Rank are PAC-Learnable](https://proceedings.mlr.press/v70/jiang17c.html) | ? | targeted primary definitions assumptions and theorem stateme |
| K | [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) | ? | method/proof/empirical sections checked |
| S | [Contributions to the Theory of Optimal Control](https://www.ee.iitb.ac.in/~belur/ee640/optimal-classic-paper.pdf) | ? | targeted primary reprint model and theorem conditions checke |
| s | [Convergence Results for Single-Step On-Policy Reinforcement-Learning Algorithms](https://web.eecs.umich.edu/~baveja/Papers/MLjournal5.pdf) | ? | proof statements checked |
| K | [Counterfactual Multi-Agent Policy Gradients](https://arxiv.org/abs/1705.08926) | ? | method/proof/empirical sections checked |
| K | [Curiosity-driven Exploration by Self-supervised Prediction](https://arxiv.org/abs/1705.05363) | ? | method/proof/empirical sections checked |
| K | [D4RL: Datasets for Deep Data-Driven Reinforcement Learning](https://arxiv.org/abs/2004.07219) | ? | method/proof/empirical sections checked |
| K | [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) | ? | method/proof/empirical sections checked |
| K | [Data-Efficient Hierarchical Reinforcement Learning](https://arxiv.org/abs/1805.08296) | ? | method/proof/empirical sections checked |
| k+s | [Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning](https://arxiv.org/abs/1604.00923) | ? | targeted primary sections checked |
| K | [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345) | ? | targeted method sections |
| K | [Deep Exploration via Bootstrapped DQN](https://arxiv.org/abs/1602.04621) | ? | method/proof/empirical sections checked |
| K | [Deep Q-learning from Demonstrations](https://arxiv.org/abs/1704.03732) | ? | targeted primary sections checked |
| K | [Deep Recurrent Q-Learning for Partially Observable MDPs](https://arxiv.org/abs/1507.06527) | ? | method/proof/empirical sections checked |
| K | [Deep Reinforcement Learning and the Deadly Triad](https://arxiv.org/abs/1812.02648) | ? | method/proof/empirical sections checked |
| K | [Deep Reinforcement Learning at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264) | ? | targeted method and empirical sections |
| K | [Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models](https://arxiv.org/abs/1805.12114) | ? | method/proof/empirical sections checked |
| K | [Deep Reinforcement Learning that Matters](https://arxiv.org/abs/1709.06560) | ? | method/proof/empirical sections checked |
| K | [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) | ? | targeted method and empirical sections |
| K | [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741) | ? | method/proof/empirical sections checked |
| K | [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) | ? | method/proof/empirical sections checked |
| K | [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) | ? | targeted method sections |
| s | [Defining and Characterizing Reward Hacking](https://arxiv.org/abs/2209.13085) | ? | proof statements checked |
| K | [Deterministic Policy Gradient Algorithms](https://proceedings.mlr.press/v32/silver14.html) | ? | targeted theorem and algorithm sections |
| K | [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) | ? | method/proof/empirical sections checked |
| s | [Discrete Dynamic Programming](https://doi.org/10.1214/aoms/1177704593) | ? | primary bibliographic collection original abstract only |
| K | [Distributed Distributional Deterministic Policy Gradients](https://arxiv.org/abs/1804.08617) | ? | method/proof/empirical sections checked |
| K | [Distributed Prioritized Experience Replay](https://arxiv.org/abs/1803.00933) | ? | method/proof/empirical sections checked |
| K | [Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044) | ? | method/proof/empirical sections checked |
| s | [Distributionally Robust Markov Decision Processes](https://proceedings.neurips.cc/paper/2010/hash/19f3cd308f1455b3fa09a282e0d496f4-Abstract.html) | ? | targeted primary theorem statement and assumptions checked |
| K | [Diversity is All You Need: Learning Skills without a Reward Function](https://arxiv.org/abs/1802.06070) | ? | targeted primary sections checked |
| K | [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?](https://arxiv.org/abs/2504.13837) | ? | method/proof/empirical sections checked |
| k+s | [Double Q-learning](https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) | ? | proof statements checked |
| k+s | [Doubly Robust Off-policy Value Evaluation for Reinforcement Learning](https://proceedings.mlr.press/v48/jiang16.html) | ? | targeted primary sections checked |
| K | [Dream to Control: Learning Behaviors by Latent Imagination](https://arxiv.org/abs/1912.01603) | ? | method/proof/empirical sections checked |
| K | [Dual Control Theory. I](https://www.mathnet.ru/eng/at12624) | ? | targeted primary russian problem formulation checked |
| S | [Dual Control Theory. II](https://www.mathnet.ru/eng/at12665) | ? | targeted primary russian model and backward derivation check |
| K | [DualDICE: Behavior-Agnostic Estimation of Discounted Stationary Distribution Corrections](https://arxiv.org/abs/1906.04733) | ? | targeted primary sections checked |
| K | [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581) | ? | method/proof/empirical sections checked |
| K | [Dynamics-Aware Unsupervised Discovery of Skills](https://arxiv.org/abs/1907.01657) | ? | method/proof/empirical sections checked |
| K | [Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables](https://arxiv.org/abs/1903.08254) | ? | targeted primary sections checked |
| k+s | [Eligibility Traces for Off-Policy Policy Evaluation](https://web.eecs.umich.edu/~baveja/Papers/OffPolicy.pdf) | ? | targeted primary sections checked |
| K | [Empirical Study of Off-Policy Policy Evaluation for Reinforcement Learning](https://arxiv.org/abs/1911.06854) | ? | targeted primary sections checked |
| K | [Episodic Curiosity through Reachability](https://arxiv.org/abs/1810.02274) | ? | targeted primary sections checked |
| K | [Evolution Strategies as a Scalable Alternative to Reinforcement Learning](https://arxiv.org/abs/1703.03864) | ? | targeted primary sections checked |
| K | [Expected Policy Gradients for Reinforcement Learning](https://arxiv.org/abs/1801.03326) | ? | targeted primary sections checked |
| K | [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894) | ? | method/proof/empirical sections checked |
| k+s | [Fast Gradient-Descent Methods for Temporal-Difference Learning with Linear Function Approximation](https://icml.cc/Conferences/2009/papers/546.pdf) | ? | targeted primary sections checked |
| K | [FeUdal Networks for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1703.01161) | ? | method/proof/empirical sections checked |
| K | [Feudal Reinforcement Learning](https://papers.nips.cc/paper/1992/hash/d14220ee66aeec73c49038385428ec4c-Abstract.html) | ? | method/proof/empirical sections checked |
| K | [Finding Approximate POMDP Solutions Through Belief Compression](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume23/roy05a-html/) | ? | selected primary representation and planning sections checke |
| s | [Finite-Sample Analysis of Least-Squares Policy Iteration](https://jmlr.org/papers/v13/lazaric12a.html) | ? | proof statements checked |
| K+S | [Finite-time Analysis of the Multiarmed Bandit Problem](https://doi.org/10.1023/A:1013689704352) | ? | proof statements checked |
| k+s | [GQ(λ): A general gradient algorithm for temporal-difference prediction learning with eligibility traces](https://www.atlantis-press.com/proceedings/agi10/1920) | ? | targeted primary sections checked |
| K | [GenDICE: Generalized Offline Estimation of Stationary Values](https://arxiv.org/abs/2002.09072) | ? | targeted primary sections checked |
| k+s | [Generalization and Exploration via Randomized Value Functions](https://proceedings.mlr.press/v48/osband16.html) | ? | method/proof/empirical sections checked |
| K | [Generalization in Reinforcement Learning: Successful Examples Using Sparse Coarse Coding](https://proceedings.neurips.cc/paper/1995/hash/8f1d43620bc6bb580df6e80b0dc05c48-Abstract.html) | ? | method/proof/empirical sections checked |
| K | [Generative Adversarial Imitation Learning](https://arxiv.org/abs/1606.03476) | ? | method/proof/empirical sections checked |
| s | [Global Convergence of Policy Gradient Methods for the Linear Quadratic Regulator](https://proceedings.mlr.press/v80/fazel18a.html) | ? | targeted primary definitions assumptions and theorem stateme |
| K | [Go-Explore: a New Approach for Hard-Exploration Problems](https://arxiv.org/abs/1901.10995) | ? | method/proof/empirical sections checked |
| K | [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071) | ? | targeted method and empirical sections |
| K+S | [Heuristic Search Value Iteration for POMDPs](https://longhorizon.org/trey/pubs/b2hd-smith04hsvi.html) | ? | selected model algorithm and guarantee statements checked |
| K | [Hierarchical Deep Reinforcement Learning: Integrating Temporal Abstraction and Intrinsic Motivation](https://arxiv.org/abs/1604.06057) | ? | method/proof/empirical sections checked |
| K | [Hierarchical Reinforcement Learning with the MAXQ Value Function Decomposition](https://arxiv.org/abs/cs/9905014) | ? | method/proof/empirical sections checked |
| k+s | [High Confidence Off-Policy Evaluation](https://ojs.aaai.org/index.php/AAAI/article/view/9541) | ? | targeted primary sections checked |
| K | [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) | ? | targeted theorem and algorithm sections |
| K | [Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) | ? | method/proof/empirical sections checked |
| K | [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) | ? | targeted empirical sections |
| K | [IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures](https://arxiv.org/abs/1802.01561) | ? | method/proof/empirical sections checked |
| k+s | [Imitating Past Successes can be Very Suboptimal](https://arxiv.org/abs/2206.03378) | ? | method/proof/empirical sections checked |
| K | [Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO](https://arxiv.org/abs/2005.12729) | ? | method/proof/empirical sections checked |
| K | [Implicit Quantile Networks for Distributional Reinforcement Learning](https://arxiv.org/abs/1806.06923) | ? | method/proof/empirical sections checked |
| K | [Improving Generalisation for Temporal Difference Learning: The Successor Representation](https://www.gatsby.ucl.ac.uk/~dayan/papers/d93b.pdf) | ? | targeted primary sections checked |
| K+S | [Incremental Pruning: A Simple, Fast, Exact Method for Partially Observable Markov Decision Processes](https://arxiv.org/abs/1302.1525) | ? | selected model algorithm and identity checked |
| K | [Integrated Architectures for Learning, Planning, and Reacting Based on Approximating Dynamic Programming](https://mlanthology.org/icml/1990/sutton1990icml-integrated/) | ? | method/proof/empirical sections checked |
| k+s | [Is Pessimism Provably Efficient for Offline RL?](https://arxiv.org/abs/2012.15085) | ? | targeted primary sections checked |
| S | [Is a Good Representation Sufficient for Sample Efficient Reinforcement Learning?](https://arxiv.org/abs/1910.03016) | ? | targeted primary definitions assumptions and theorem stateme |
| K | [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) | ? | targeted primary sections checked |
| K | [Learning Latent Dynamics for Planning from Pixels](https://arxiv.org/abs/1811.04551) | ? | method/proof/empirical sections checked |
| K | [Learning Multi-Level Hierarchies with Hindsight](https://arxiv.org/abs/1712.00948) | ? | method/proof/empirical sections checked |
| K | [Learning Multiagent Communication with Backpropagation](https://arxiv.org/abs/1605.07736) | ? | targeted primary sections checked |
| s | [Learning Rates for Q-learning](https://www.jmlr.org/papers/v5/evendar03a.html) | ? | targeted primary sections checked |
| s | [Learning Without Mixing: Towards A Sharp Analysis of Linear System Identification](https://arxiv.org/abs/1802.08334) | ? | targeted primary definitions assumptions and theorem stateme |
| s | [Learning Without State-Estimation in Partially Observable Markovian Decision Processes](https://people.csail.mit.edu/tommi/rl.html) | ? | targeted primary counterexample and assumptions checked |
| K | [Learning to Communicate with Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1605.06676) | ? | targeted primary sections checked |
| K | [Learning to Predict by the Methods of Temporal Differences](https://link.springer.com/article/10.1007/BF00115009) | ? | method/proof/empirical sections checked |
| k+s | [Learning to Reach Goals via Iterated Supervised Learning](https://arxiv.org/abs/1912.06088) | ? | method/proof/empirical sections checked |
| K | [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) | ? | method/proof/empirical sections checked |
| K | [Least-Squares Policy Iteration](https://jmlr.org/papers/v4/lagoudakis03a.html) | ? | proof statements checked |
| K | [Let’s Verify Step by Step](https://arxiv.org/abs/2305.20050) | ? | method/proof/empirical sections checked |
| K | [MOPO: Model-based Offline Policy Optimization](https://arxiv.org/abs/2005.13239) | ? | method/proof/empirical sections checked |
| K | [MOReL: Model-Based Offline Reinforcement Learning](https://arxiv.org/abs/2005.05951) | ? | targeted primary sections checked |
| K | [Markov games as a framework for multi-agent reinforcement learning](https://mlanthology.org/icml/1994/littman1994icml-markov/) | ? | targeted primary sections checked |
| K | [Mastering Atari with Discrete World Models](https://arxiv.org/abs/2010.02193) | ? | method/proof/empirical sections checked |
| K | [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265) | ? | method/proof/empirical sections checked |
| K | [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/abs/1712.01815) | ? | method/proof/empirical sections checked |
| K | [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104) | ? | method/proof/empirical sections checked |
| K | [Mastering the game of Go with deep neural networks and tree search](https://research.google/pubs/mastering-the-game-of-go-with-deep-neural-networks-and-tree-search/) | ? | method/proof/empirical sections checked |
| K | [Mastering the game of Go without human knowledge](https://discovery.ucl.ac.uk/id/eprint/10045895/) | ? | method/proof/empirical sections checked |
| K | [Maximum Entropy Inverse Reinforcement Learning](https://www.cs.cmu.edu/~bziebart/publications/maxentirl-bziebart.pdf) | ? | method/proof/empirical sections checked |
| K | [Maximum a Posteriori Policy Optimisation](https://arxiv.org/abs/1806.06920) | ? | method/proof/empirical sections checked |
| k+s | [Minimax Regret Bounds for Reinforcement Learning](https://proceedings.mlr.press/v70/azar17a.html) | ? | proof statements checked |
| s | [Model-based RL in Contextual Decision Processes: PAC bounds and Exponential Improvements over Model-free Approaches](https://proceedings.mlr.press/v99/sun19a.html) | ? | targeted primary definitions assumptions and theorem stateme |
| K | [Monte-Carlo Planning in Large POMDPs](https://papers.neurips.cc/paper_files/paper/2010/hash/edfbe1afcf9246bb0d40eb4d8027d90f-Abstract.html) | ? | method/proof/empirical sections checked |
| K | [Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) | ? | method/proof/empirical sections checked |
| K | [Multi-Agent Reinforcement Learning: Independent vs. Cooperative Agents](https://mlanthology.org/icml/1993/tan1993icml-multi/) | ? | targeted primary sections checked |
| k+s | [Near-Optimal Reinforcement Learning in Polynomial Time](https://www.cis.upenn.edu/~mkearns/papers/barbados/ks-e3.pdf) | ? | targeted primary sections checked |
| k+s | [Near-optimal Regret Bounds for Reinforcement Learning](https://www.jmlr.org/papers/v11/jaksch10a.html) | ? | proof statements checked |
| K | [Neural Episodic Control](https://arxiv.org/abs/1703.01988) | ? | method/proof/empirical sections checked |
| K | [Never Give Up: Learning Directed Exploration Strategies](https://arxiv.org/abs/2002.06038) | ? | targeted primary sections checked |
| S | [New Results in Linear Filtering and Prediction Theory](https://doi.org/10.1115/1.3658902) | ? | targeted primary model and theorem statements checked |
| K | [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295) | ? | targeted primary sections checked |
| K | [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) | ? | targeted primary sections checked |
| k+s | [Off-Policy Actor-Critic](https://arxiv.org/abs/1205.4839) | ? | proof statements checked |
| K | [Off-Policy Deep Reinforcement Learning without Exploration](https://arxiv.org/abs/1812.02900) | ? | method/proof/empirical sections checked |
| K | [Off-Policy Evaluation via the Regularized Lagrangian](https://arxiv.org/abs/2007.03438) | ? | targeted primary sections checked |
| K | [Offline RL Without Off-Policy Evaluation](https://arxiv.org/abs/2106.08909) | ? | targeted primary sections checked |
| K | [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169) | ? | method/proof/empirical sections checked |
| s | [On the Sample Complexity of the Linear Quadratic Regulator](https://arxiv.org/abs/1710.01688) | ? | targeted primary definitions assumptions and theorem stateme |
| S | [On the Theory of Policy Gradient Methods: Optimality, Approximation, and Distribution Shift](https://arxiv.org/abs/1908.00261) | ? | proof statements checked |
| K | [On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) | ? | targeted primary sections checked |
| K | [Online and Offline Reinforcement Learning by Planning with a Learned Model](https://arxiv.org/abs/2104.06294) | ? | method/proof/empirical sections checked |
| S | [Optimal Control of Markov Processes with Incomplete State Information I](https://doi.org/10.1016/0022-247X(65)90154-X) | ? | proof statements checked |
| S | [Optimization of Conditional Value-at-Risk](https://doi.org/10.21314/JOR.2000.038) | ? | selected primary assumptions and theorem statements read |
| K | [PILCO: A Model-Based and Data-Efficient Approach to Policy Search](https://icml.cc/2011/papers/323_icmlpaper.pdf) | ? | method/proof/empirical sections checked |
| K | [Phasic Policy Gradient](https://arxiv.org/abs/2009.04416) | ? | targeted primary sections checked |
| K | [Planning and Acting in Partially Observable Stochastic Domains](https://doi.org/10.1016/S0004-3702(98)00023-X) | ? | targeted primary model and derivation checked |
| K | [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) | ? | method/proof/empirical sections checked |
| K+S | [Point-Based POMDP Algorithms: Improved Analysis and Implementation](https://longhorizon.org/trey/pubs/b2hd-smith05hsvi.html) | ? | selected corrected primary theorems and author erratum check |
| K+S | [Point-based Value Iteration: An Anytime Algorithm for POMDPs](https://robots.stanford.edu/papers/Pineau03a.html) | ? | selected primary method and error bound checked |
| S | [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) | ? | proof statements checked |
| s | [Policy Gradient for Coherent Risk Measures](https://arxiv.org/abs/1502.03919) | ? | targeted primary theorem statement and assumptions checked |
| S | [Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) | ? | proof statements checked |
| K | [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) | ? | method/proof/empirical sections checked |
| K | [Prioritized Sweeping: Reinforcement Learning with Less Data and Less Real Time](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa15/optreadings/MooreAtkeson-1993-prioritized-sweeping.pdf) | ? | method/proof/empirical sections checked |
| s | [Provably Efficient Reinforcement Learning with Linear Function Approximation](https://proceedings.mlr.press/v125/jin20a.html) | ? | targeted primary definitions assumptions and theorem stateme |
| K | [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | ? | targeted algorithm sections |
| k+s | [Q(λ) with Off-Policy Corrections](https://arxiv.org/abs/1602.04951) | ? | targeted primary sections checked |
| K | [Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic](https://arxiv.org/abs/1611.02247) | ? | targeted primary sections checked |
| S | [Q-learning](https://link.springer.com/article/10.1007/BF00992698) | ? | proof statements checked |
| K | [QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) | ? | targeted method sections |
| k+s | [QPLEX: Duplex Dueling Multi-Agent Q-Learning](https://arxiv.org/abs/2008.01062) | ? | targeted primary sections checked |
| k+s | [QTRAN: Learning to Factorize with Transformation for Cooperative Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1905.05408) | ? | targeted primary sections checked |
| k+s | [R-max – A General Polynomial Time Algorithm for Near-Optimal Reinforcement Learning](https://jmlr.org/papers/v3/brafman02a.html) | ? | proof statements checked |
| K | [REINFORCE++: Stabilizing Critic-Free Policy Optimization with Global Advantage Normalization](https://arxiv.org/abs/2501.03262) | ? | method/proof/empirical sections checked |
| K | [RLAIF vs. RLHF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267) | ? | method/proof/empirical sections checked |
| K | [RL²: Fast Reinforcement Learning via Slow Reinforcement Learning](https://arxiv.org/abs/1611.02779) | ? | targeted primary sections checked |
| k+s | [RUDDER: Return Decomposition for Delayed Rewards](https://arxiv.org/abs/1806.07857) | ? | proof statements checked |
| K | [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298) | ? | method/proof/empirical sections checked |
| K | [Randomized Prior Functions for Deep Reinforcement Learning](https://arxiv.org/abs/1806.03335) | ? | method/proof/empirical sections checked |
| K | [Recurrent Experience Replay in Distributed Reinforcement Learning](https://willdabney.com/publication/r2d2/) | ? | targeted primary review draft sections checked |
| S | [Recursive Multiple-Priors](https://doi.org/10.1016/S0022-0531(03)00097-8) | ? | selected primary domain axioms and representation theorem re |
| s | [Regret Bounds for the Adaptive Control of Linear Quadratic Systems](https://proceedings.mlr.press/v19/abbasi-yadkori11a.html) | ? | targeted primary definitions assumptions and theorem stateme |
| s | [Reinforcement Learning Algorithm for Partially Observable Markov Decision Problems](https://proceedings.neurips.cc/paper/1994/hash/1c1d4df596d01da60385f0bb17a4a9e0-Abstract.html) | ? | targeted primary theorem statement and assumptions checked |
| k+s | [Reinforcement Learning in Finite MDPs: PAC Analysis](https://jmlr.org/papers/v10/strehl09a.html) | ? | proof statements checked |
| K | [Reinforcement Learning with Deep Energy-Based Policies](https://arxiv.org/abs/1702.08165) | ? | method/proof/empirical sections checked |
| k+s | [Reinforcement Learning with Replacing Eligibility Traces](https://web.eecs.umich.edu/~baveja/Papers/Replace.pdf) | ? | targeted primary sections checked |
| K | [Reinforcement Learning with Unsupervised Auxiliary Tasks](https://arxiv.org/abs/1611.05397) | ? | method/proof/empirical sections checked |
| K | [Reinforcement Learning with Verifiable Rewards Implicitly Incentivizes Correct Reasoning in Base LLMs](https://arxiv.org/abs/2506.14245) | ? | method/proof/empirical sections checked |
| k+s | [Reinforcement Learning with a Corrupted Reward Channel](https://arxiv.org/abs/1705.08417) | ? | targeted primary sections checked |
| K | [Reinforcement Learning: A Survey](https://arxiv.org/abs/cs/9605103) | ? | proof statements checked |
| K | [Relative Entropy Policy Search](https://ojs.aaai.org/index.php/AAAI/article/view/7727) | ? | method/proof/empirical sections checked |
| S | [Residual Algorithms: Reinforcement Learning with Function Approximation](https://doi.org/10.1016/B978-1-55860-377-6.50013-X) | ? | targeted counterexample sections |
| k+l | [Revisiting Fundamentals of Experience Replay](https://arxiv.org/abs/2007.06700) | ? | targeted primary sections checked |
| K | [Reward Constrained Policy Optimization](https://arxiv.org/abs/1805.11074) | ? | targeted primary sections checked |
| S | [Risk-Averse Dynamic Programming for Markov Decision Processes](https://doi.org/10.1007/s10107-010-0393-3) | ? | selected primary risk axioms and finite discounted DP condit |
| K+S | [Risk-Sensitive Markov Decision Processes](https://doi.org/10.1287/mnsc.18.7.356) | ? | primary reprint selected equations assumptions and appendix  |
| K | [Risk-Sensitive Reinforcement Learning Applied to Control under Constraints](https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume24/geibel05a.pdf) | ? | targeted primary sections checked |
| K | [Risk-Sensitive and Robust Decision-Making: a CVaR Optimization Approach](https://arxiv.org/abs/1506.02188) | ? | targeted primary sections checked |
| s | [Robust Control of Markov Decision Processes with Uncertain Transition Matrices](https://doi.org/10.1287/opre.1050.0216) | ? | proof statements checked |
| S | [Robust Dynamic Programming](https://doi.org/10.1287/moor.1040.0129) | ? | remote author manuscript selected original text conditions c |
| K+S | [SARSOP: Efficient Point-Based POMDP Planning by Approximating Optimally Reachable Belief Spaces](https://www.roboticsproceedings.org/rss04/p9.html) | ? | selected primary method and cited theorem statements checked |
| k+s | [Safe Policy Improvement with Baseline Bootstrapping](https://arxiv.org/abs/1712.06924) | ? | targeted primary sections checked |
| K | [Safe and Efficient Off-Policy Reinforcement Learning](https://arxiv.org/abs/1606.02647) | ? | method/proof/empirical sections checked |
| K | [Sample Efficient Actor-Critic with Experience Replay](https://arxiv.org/abs/1611.01224) | ? | method/proof/empirical sections checked |
| S | [Sample-Optimal Parametric Q-Learning Using Linearly Additive Features](https://proceedings.mlr.press/v97/yang19b.html) | ? | targeted primary definitions assumptions and theorem stateme |
| L | [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) | ? | targeted method and empirical sections |
| L | [Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms](https://arxiv.org/abs/2406.02900) | ? | method/proof/empirical sections checked |
| K | [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) | ? | method/proof/empirical sections checked |
| K | [SimPO: Simple Preference Optimization with a Reference-Free Reward](https://arxiv.org/abs/2405.14734) | ? | targeted primary sections checked |
| K | [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://doi.org/10.1007/BF00992696) | ? | method/proof/empirical sections checked |
| K | [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/abs/1812.05905) | ? | method/proof/empirical sections checked |
| K | [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) | ? | method/proof/empirical sections checked |
| K | [Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction](https://arxiv.org/abs/1906.00949) | ? | method/proof/empirical sections checked |
| k+s | [Successor Features for Transfer in Reinforcement Learning](https://papers.nips.cc/paper_files/paper/2017/hash/350db081a661525235354dd3e19b8c05-Abstract.html) | ? | targeted primary sections checked |
| K | [TD-MPC2: Scalable, Robust World Models for Continuous Control](https://arxiv.org/abs/2310.16828) | ? | method/proof/empirical sections checked |
| S | [Team Decision Theory and Information Structures in Optimal Control Problems—Part I](https://doi.org/10.1109/TAC.1972.1099850) | ? | proof statements checked |
| K | [Temporal Difference Learning for Model Predictive Control](https://arxiv.org/abs/2203.04955) | ? | method/proof/empirical sections checked |
| K | [The Arcade Learning Environment: An Evaluation Platform for General Agents](https://arxiv.org/abs/1207.4708) | ? | method/proof/empirical sections checked |
| L | [The Art of Scaling Reinforcement Learning Compute for LLMs](https://arxiv.org/abs/2510.13786) | ? | method/proof/empirical sections checked |
| k+s | [The Cramer Distance as a Solution to Biased Wasserstein Gradients](https://arxiv.org/abs/1705.10743) | ? | proof statements checked |
| K | [The Mirage of Action-Dependent Baselines in Reinforcement Learning](https://arxiv.org/abs/1802.10031) | ? | targeted primary sections checked |
| K | [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://arxiv.org/abs/2403.17031) | ? | method/proof/empirical sections checked |
| k+s | [The Nonstochastic Multiarmed Bandit Problem](https://doi.org/10.1137/S0097539701398375) | ? | proof statements checked |
| s | [The Optimal Control of Partially Observable Markov Processes over a Finite Horizon](https://pubsonline.informs.org/doi/10.1287/opre.21.5.1071) | ? | source abstract only |
| s | [The Optimal Control of Partially Observable Markov Processes over the Infinite Horizon: Discounted Costs](https://pubsonline.informs.org/doi/10.1287/opre.26.2.282) | ? | source abstract only |
| K | [The Option-Critic Architecture](https://arxiv.org/abs/1609.05140) | ? | method/proof/empirical sections checked |
| K | [The Reactor: A fast and sample-efficient Actor-Critic agent for Reinforcement Learning](https://arxiv.org/abs/1704.04651) | ? | method/proof/empirical sections checked |
| K | [The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) | ? | targeted primary sections checked |
| K | [Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback](https://arxiv.org/abs/2204.05862) | ? | method/proof/empirical sections checked |
| K | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | ? | method/proof/empirical sections checked |
| K | [Tree-Based Batch Mode Reinforcement Learning](https://www.jmlr.org/papers/v6/ernst05a.html) | ? | targeted primary sections checked |
| K | [True Online Temporal-Difference Learning](https://arxiv.org/abs/1512.04087) | ? | method/proof/empirical sections checked |
| K | [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) | ? | targeted theorem and algorithm sections |
| K | [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783) | ? | targeted method and empirical sections |
| K | [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/abs/1606.01868) | ? | method/proof/empirical sections checked |
| K | [Universal Value Function Approximators](https://proceedings.mlr.press/v37/schaul15.html) | ? | targeted primary sections checked |
| K | [V-MPO: On-Policy Maximum a Posteriori Policy Optimization for Discrete and Continuous Control](https://arxiv.org/abs/1909.12238) | ? | targeted primary sections checked |
| K | [VIME: Variational Information Maximizing Exploration](https://arxiv.org/abs/1605.09674) | ? | targeted primary sections checked |
| K | [Value-Decomposition Networks For Cooperative Multi-Agent Learning](https://arxiv.org/abs/1706.05296) | ? | method/proof/empirical sections checked |
| K+S | [Value-Directed Compression of POMDPs](https://papers.nips.cc/paper_files/paper/2002/hash/14ea0d5b0cf49525d1866cb1e95ada5d-Abstract.html) | ? | selected linear sufficient conditions and algebra checked |
| K | [VariBAD: A Very Good Method for Bayes-Adaptive Deep RL via Meta-Learning](https://arxiv.org/abs/1910.08348) | ? | targeted primary sections checked |
| k+s | [Variance Reduction for Policy Gradient with Action-Dependent Factorized Baselines](https://arxiv.org/abs/1803.07246) | ? | targeted primary sections checked |
| K | [Variational Intrinsic Control](https://arxiv.org/abs/1611.07507) | ? | method/proof/empirical sections checked |
| K | [When to Trust Your Model: Model-Based Policy Optimization](https://arxiv.org/abs/1906.08253) | ? | targeted method sections |
