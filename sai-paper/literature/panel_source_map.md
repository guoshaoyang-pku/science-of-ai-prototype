# 核心 panel 的代表工作与文献证据对照

逐项覆盖 panel 支撑清单的 53 行；合并行保留多个来源。这里只关联已有书目与三领域记录，不修改 panel、冻结论文或参考文献。书目定位取自 references.bib，不表示本次审查了所有来源正文。

匹配依据为标识或标准化题名；来源命中不等于分类一致、主张成立或全文精读。未匹配项只说明尚未在三份目录定位，不表示工作不存在或不重要。

当前 53 行含 59 个来源引用：23 行全部匹配、0 行部分匹配、30 行未匹配。来源引用不是去重论文数，不能换算为领域覆盖率。

| Panel 工作及原标签 | 来源及当前文献主标签 | 应区分的主张／待核事项 |
|---|---|---|
| Kaplan 2020（Scaling Laws for Neural Language Models） [L2] | [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)<br>pretrain：Law | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Hoffmann 2022（Chinchilla） [L2] | [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)<br>pretrain：Law | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Muennighoff 2023（数据受限缩放） [L2] | [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264)<br>pretrain：Law | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Yang 2022（μP / Tensor Programs 系列） [L2] | [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](https://arxiv.org/abs/2203.03466)<br>pretrain：Science<br><br>[Feature Learning in Infinite-Width Neural Networks](https://arxiv.org/abs/2011.14522)<br>pretrain：Science | 系列中的理论、参数化方法与迁移实验应分开，不能只比较整篇主标签。 |
| Zhang 2017 / Belkin 2019（双下降） [L2] | [Understanding deep learning requires rethinking generalization](https://arxiv.org/abs/1611.03530)<br>pretrain：Knowledge<br><br>[Reconciling modern machine learning practice and the bias-variance trade-off](https://arxiv.org/abs/1812.11118)<br>pretrain：Law | 随机标签/像素记忆和泛化限制不等于双下降证据；两篇来源应分开。 |
| Power 2022（grokking） [L2] | [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)<br>pretrain：Knowledge | 延迟泛化是条件观察；跨条件经验关系可另列 Law 候选。 |
| Gao 2022（奖励过优化） [L2] | [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760)<br>rl：Law | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Hu 2025（RL 计算缩放的艺术） [L2] | [The Art of Scaling Reinforcement Learning Compute for LLMs](https://arxiv.org/abs/2510.13786)<br>rl：Law | 原文首页作者为 Devvrit Khatri 等；panel 的 Hu 作者标记待修正。摘要报告超过 40 万 GPU 小时；本次核对该表述，未审计算力日志。 |
| 本工作（有效时间 U 与过拟合起点） [L2] | [Science of AI Prototype: Fuzzy-Direction Automated Research Reports](https://guoshaoyang-pku.github.io/science-of-ai-prototype/)<br>三线目录未匹配；本表未新增原文审阅 | 项目内研究报告；需要独立核对假设、测量与反例。 |
| Jacot 2018（NTK） [L3] | [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572)<br>pretrain：Science | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Chizat 2019（lazy / rich） [L3] | [On Lazy Training in Differentiable Programming](https://arxiv.org/abs/1812.07956)<br>pretrain：Science | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Bahri 2023（缩放律的统计力学） [L3] | [Explaining Neural Scaling Laws](https://arxiv.org/abs/2102.06701)<br>pretrain：Science | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Agarwal 2020（策略梯度理论） [L3] | [On the Theory of Policy Gradient Methods: Optimality, Approximation, and Distribution Shift](https://arxiv.org/abs/1908.00261)<br>rl：Science | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Watkins 1992（Q-learning） [L3] | [Q-learning](http://dx.doi.org/10.1007/BF00992698)<br>rl：Science | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Hyperball 2026 及其批评 [L2→L3] | [Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization](https://arxiv.org/abs/2606.16899)<br>pretrain：Science<br><br>[Hyperball May Not Be a Free Lunch](https://arxiv.org/abs/2607.22444)<br>pretrain：Science | 按用户口径，指定理论分析可列 Science；方法、实测及批评各有独立范围。 |
| Olsson 2022（induction heads） [L2 碎片] | [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)<br>interp：Knowledge（限定实证与机制假说） | 小模型干预证据与大模型普遍机制假说分开。 |
| Elhage 2022（叠加） [L1–L2] | [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_models/index.html)<br>interp：Science（玩具理论候选）＋Knowledge | 玩具模型数学性质与实际网络的经验主张分开。 |
| Bricken 2023 / Templeton 2024（单义性） [L1–L2] | [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html)<br>interp：Knowledge<br><br>[Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)<br>interp：Knowledge | 特征抽取方法/观测为 Knowledge；若有经验缩放关系，须单列关系及跨条件证据。 |
| Nanda 2023（grokking 机制） [L3（域内）] | [Progress measures for grokking via mechanistic interpretability](https://arxiv.org/abs/2301.05217)<br>interp：Knowledge＋Science（局部机制候选） | 指定模块加法电路的证据，不等于已知所有 grokking 的适用边界。 |
| Shao 2024（GRPO / DeepSeekMath） [L1] | [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)<br>rl：Knowledge | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Zheng 2025（GSPO） [L1] | [Group Sequence Policy Optimization](https://arxiv.org/abs/2507.18071)<br>rl：Knowledge | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Dao 2022（FlashAttention） [L1] | [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)<br>pretrain：Science | 方法与性能实测可为 Knowledge；指定内存模型下的 IO 定理是不同的 Science 主张。 |
| Hu 2021（LoRA） [L1] | [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)<br>pretrain：Knowledge | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Shinn 2023（Reflexion） [L1] | [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Wang 2023（Voyager） [L1] | [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Zhao 2023（ExpeL） [L1] | [ExpeL: LLM Agents Are Experiential Learners](https://arxiv.org/abs/2308.10144)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Wang 2024（Agent Workflow Memory） [L1] | [Agent Workflow Memory](https://arxiv.org/abs/2409.07429)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Jiang 2025（G-Memory） [L1] | [G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems](https://arxiv.org/abs/2506.07398)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Suzgun 2025（Dynamic Cheatsheet） [L1] | [Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory](https://arxiv.org/abs/2504.07952)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Lu 2024（AI Scientist） [L1 稿] | [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Yamada 2025（AI Scientist v2） [L1 稿] | [The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search](https://arxiv.org/abs/2504.08066)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Intology 2025（Zochi / Locus） [L1 稿] | [Zochi: Technical Report](https://intology.ai/blog/zochi-tech-report)<br>三线目录未匹配；本表未新增原文审阅<br><br>[Locus: A Long-Horizon Artificial Scientist](https://intology.ai/about)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Gottweis 2025（co-scientist） [L2 假说] | [Accelerating scientific discovery with Co-Scientist](https://arxiv.org/abs/2502.18864)<br>三线目录未匹配；本表未新增原文审阅 | 排序假说不自动是 Law，需核对具体关系与跨条件证据。 |
| Schmidgall 2025（Agent Laboratory） [L1] | [Agent Laboratory: Using LLM Agents as Research Assistants](https://arxiv.org/abs/2501.04227)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Ghareeb 2025（Robin） [L2 假说] | [Robin: A multi-agent system for automating scientific discovery](https://arxiv.org/abs/2505.13400)<br>三线目录未匹配；本表未新增原文审阅 | 系统产生的假说与实验支持的发现应逐条区分。 |
| Nam 2025: MLE-STAR（带搜索与定向精炼的 ML 工程智能体） [L1] | [MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement](https://arxiv.org/abs/2506.15692)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Romera-Paredes 2024（FunSearch） [L1 工件] | [Mathematical discoveries from program search with large language models](http://dx.doi.org/10.1038/s41586-023-06924-6)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| DeepMind 2025（AlphaEvolve） [L1 工件] | [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Fawzi 2022（AlphaTensor） [L1 工件] | [Discovering faster matrix multiplication algorithms with reinforcement learning](http://dx.doi.org/10.1038/s41586-022-05172-4)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Langley 1987（BACON） [L2] | Scientific Discovery: Computational Explorations of the Creative Processes（书目未提供直接链接）<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| King 2004 / 2009（Adam 机器人科学家） [L2–L3] | [Functional genomic hypothesis generation and experimentation by a robot scientist](http://dx.doi.org/10.1038/nature02236)<br>三线目录未匹配；本表未新增原文审阅<br><br>[The Automation of Science](http://dx.doi.org/10.1126/science.1165620)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Schmidt 2009（Eureqa） [L2] | [Distilling Free-Form Natural Laws from Experimental Data](http://dx.doi.org/10.1126/science.1165893)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Udrescu 2020（AI Feynman） [L2] | [AI Feynman: a Physics-Inspired Method for Symbolic Regression](https://arxiv.org/abs/1905.11481)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Merchant 2023（GNoME） [L2] | [Scaling deep learning for materials discovery](http://dx.doi.org/10.1038/s41586-023-06735-9)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Jumper 2021（AlphaFold） [L3（域内）] | [Highly accurate protein structure prediction with AlphaFold](http://dx.doi.org/10.1038/s41586-021-03819-2)<br>pretrain：Knowledge | 结构预测方法及评测属于 Knowledge 主张；预测精度本身不证明第一性原理边界。 |
| Bernat 2026（Modelpedia） [L1 组织] | [Modelpedia: A Catalog of Model Findings for the Meta-Science of AI](https://arxiv.org/abs/2609.01090)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| 验证缺口调查 2026 [审计] | [Autonomous Research Agents: A Survey of AI Scientists and the Verification Gap](https://arxiv.org/abs/2608.05179)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Chan 2024（MLE-bench） [L1 仪器] | [MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering](https://arxiv.org/abs/2410.07095)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Aronovich 2024（RE-Bench） [L1 仪器] | [RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts](https://arxiv.org/abs/2411.15114)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| Fajman 2025（PaperBench） [L1 仪器] | [PaperBench: Evaluating AI's Ability to Replicate AI Research](https://arxiv.org/abs/2504.01848)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| METR 2025（长任务视野） [L1 仪器] | [Measuring AI Ability to Complete Long Software Tasks](https://arxiv.org/abs/2503.14499)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| ArchitectureIQ 2026 [L1 仪器（最接近知识）] | [ArchitectureIQ: A Benchmark for Judging Neural Network Training Outcomes](https://architecture-iq.com/)<br>三线目录未匹配；本表未新增原文审阅 | 需结合原记录的 scope 和阅读深度核对本行主张。 |
| 本工作（SAI-min） [L1→L2] | [Science of AI: Knowledge Base Loop Prototype](https://guoshaoyang-pku.github.io/blogs/kb_site/)<br>三线目录未匹配；本表未新增原文审阅 | 配对性能、引用归因、独立真值核验是不同证据，不由来源匹配判定晋级。 |

完整匹配记录及证据字段见 [panel_source_map.json](panel_source_map.json)；文件核查见 [audit.json](audit.json)。本表只是 panel 的证据入口，不替代逐条原文审阅。

Panel SHA-256：875cc6a1f8db9a266a6c1d1105dec1849faf2732da7e28d24a6945d8c9e12cb4。生成时间：2026-10-28T12:33:53.697690+00:00。
