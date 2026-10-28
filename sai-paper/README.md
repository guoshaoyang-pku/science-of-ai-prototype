> **Public mirror note (2026-10-28, branch `sai-taxonomy`).** This directory is the public layer of a larger local research workspace. Present here: `handoffs/` (merge handoff, state snapshot, inventory, original three-axis sketch), `literature/` (932 paper records, 872 concepts, coverage/case documents), `tools/` (catalog builders, exact counterexample scripts, public survey exporter). Intentionally local-only: `papers/` (full-text PDFs), `.cache/`, `notes/` (internal meeting snapshots), `report/` (Chinese panel — known content issues, pending revision), and the frozen manuscript (`main.tex`/`main.pdf`/`sections/`/`figures/`/`references.bib`, not to be published or extended). Links to those materials below refer to the local workspace and are dead ends in this mirror. `tools/build_panel_source_map.py` is not mirrored because it depends on the frozen panel and `references.bib`.

# Science of AI 文献与概念图谱

本项目围绕可解释性（interp）、强化学习（RL）和预训练（pretrain），收集原始文献、下载可公开获取的论文 PDF，并以 Knowledge、Law、Science 标记论文的主要认识主张及其范围。

## 核心入口

- **[中文汇报 panel](report/sai-core-figures.html)**：对人汇报、续作讨论、向外介绍 SAI 的第一阅读面。其证据与分类仍需逐条核查，不能把交接中的“已核对”视为全部结论已获独立验证。
- **[当前状态与合并交接](handoffs/SAI当前状态与合并交接_2026-10-28.md)**：保存用户定义、932 条文献／872 个概念的快照、panel 未修问题、11 个主 PDF 缺口及恢复步骤。目标为 `guoshaoyang-pku/science-of-ai-prototype`；按用户最新指令，等待另一条线后再 merge。
- [文件合并清单](handoffs/merge_inventory_2026-10-28.json)与[状态快照](handoffs/state_2026-10-28.json)：仅为本地准备，没有暂存、commit 或 push。改动 panel 后仍须重跑生成、自检并目检图形。
- [Panel 来源对照](literature/panel_source_map.md)：将支撑清单的 53 行、59 个来源引用关联到现有三领域目录，保留分类差异、未匹配来源和审阅范围。来源匹配不代表主张验证。
- `main.pdf` 是冻结的次级学术产物：不投、不扩、不发。以下 `survey/` 草稿同样不是本轮续作入口；当前目录持续更新，不意味着这些 PDF 已同步重生成。

## 既有论文草稿与证据卡

- `survey/science_of_ai_concept_catalogue.pdf`：按方向与层次列出概念、定义或主张、scope、证据、审阅程度和来源；复合分类会重复列出。对应 `survey/concept_paper.tex`、`survey/concepts/` 与 `tools/render_concept_tex.py`。
- `survey/science_of_ai_literature_catalogue.pdf`：三方向论文分类附录，逐条列出主张、scope、证据位置、审阅程度和原文链接；复合分类会重复列出。对应源文件为 `survey/catalogue_paper.tex`、`survey/catalogue/` 和 `tools/render_catalogue_tex.py`。
- `survey/science_of_ai_survey_draft.pdf` 与 `survey_paper.tex`：按当前 Knowledge–Law–Science 口径整理的独立论文草稿，包含三轴图和三个方向的代表性比较；作者信息尚未确定。引用由 DOI 一手元数据核验，清单见 `survey/reference_manifest.json`。
- `survey/interp_evidence.md`、`survey/rl_evidence.md`、`survey/pretrain_evidence.md`：供正文扩写使用的 47 篇定向原文证据卡，含具体主张、范围和未完成核查；不是系统性全领域样本。
- `survey/interp_adjudication.md`、`survey/rl_adjudication.md`、`survey/pretrain_adjudication.md`：对原标签已知情况下的定向复核，保留修改理由和未决事项；不构成盲法标注或标注一致率。
- 原有 `main.tex` 是包含自动化原型和另一套分类门槛的冻结草稿，尚未按当前口径统一审阅。`references.bib` 中 `watkins1992qlearning` 已修正为 Watkins–Dayan 的 Q-learning；旧版错误记录不再代表当前文件。本轮未修改冻结稿及其参考文献。

## 使用

- `literature/interp.json`、`literature/rl.json`、`literature/pretrain.json`：分方向的论文与概念条目，包含来源、主张、scope、审阅程度和本地 PDF 状态。
- `literature/by_level.md`：按 Knowledge、Law、Science 分组的论文与概念入口；多标签条目可出现在多个组中。
- `literature/by_claim.md`：已拆分的具体主张及其范围、原文位置和核查程度；尚未拆分的论文不会出现在该页；没有单独审阅等级的次要主张会明确标示沿用所属论文的等级。
- `literature/concept_sources.md`、`concept_source_map.json`：按来源标识关联概念与已收录论文；主文、匿名审稿稿和海报分别列出，来源匹配不等于语义覆盖或概念已获验证。
- `literature/concept_coverage_gaps.md`：反向列出未被概念显式引用的已收论文。先核查同义概念，再补来源或定义；不要求为每篇论文机械新建概念。
- `literature/catalog.md`：由以上文件生成的按方向可阅读总表。`literature/audit.json`：同一次生成的 PDF 检查和结构化统计；原论文、原生网页、教材资源和海报等补充材料分别审计并记录；标题、arXiv ID 和文件哈希的潜在重复分组保留供人工判断。
- `literature/science_boundary_cases.md`：二次系统的完整稳定边界、同简化噪声尺度下不同单步效果的精确例子，以及 Q-learning、积分梯度和 Hyperball 的条件化理论边界。
- `literature/law_scope_matrix.md`：十条经验 Law／候选主张已研究的条件轴、仍需保留的范围与未知。
- `literature/probe_intervention_scope.md`：用可枚举的二元反例区分 probe 可读出、模型使用、观察不可辨识与分布外干预。
- `literature/adversarial_features_scope_case.md`：比较对抗特征的单范式与跨范式定义，并记录原文表格和公式中需要进一步核查的不一致。
- `literature/superposition_encoding_scope_case.md`：针对一篇 superposition 理论论文的精确编码下界给出可枚举反例，并说明哪些较弱计数界仍成立。
- `literature/interp_reference_audit.md`：基于一篇可解释性综述参考文献的非随机覆盖抽查；记录已收录与遗漏，不作为总体覆盖率估计。
- `literature/emergence_metric_scope.md`：对照涌现研究与测量解释，区分固定长度下的指标变换、相关误差和有限样本边界。
- `literature/rl_policy_scope_case.md`：以单状态双动作系统区分同策略与离策略价值目标，并明确收敛定理和样本训练的条件。
- `literature/average_reward_scope_case.md`：用精确两状态系统区分折扣回报与长期平均回报，推导策略排序在 γ=5/8 的反转边界，并给出指定测度下第三轴为 3/8 的实例。
- `literature/rectangular_uncertainty_scope_case.md`：用两步系统区分联合不确定集与逐行独立的矩形包，精确推导误差 (1−δ)/2 及给定容差下的不适用测度；说明局部最坏情形的拼接为何需要条件。
- `literature/contrastive_divergence_scope_case.md`：对照 CD 的固定点偏差、非梯度更新与正则化固定点存在性，给出两个精确有限模型核算，并记录原文符号修正及其不影响的结论。
- `literature/goal_relabeling_scope_case.md`：比较 GCSL 与结果条件模仿理论，并用一步随机系统展示重标记可能优于混合基线、却劣于原目标条件策略。
- `literature/nonapplicability_measure_protocol.md`：第三轴的条件空间、测度、失败判据、零失败置信上限与跨分布边界。
- `literature/finite_scope_identifiability.md`：构造两个匹配全部有限观测但未测范围几乎相反的多项式系统，并区分自适应构造与固定系统的随机测试。
- `literature/counterfactual_scope_case.md`：用二元结构因果模型对照总体干预与个体反事实，给出可逐项核算的不可辨识反例。
- `literature/scaling_compute_scope.md`：比较 Kaplan、Chinchilla 和后续受控研究，推导经验损失模型中的计算最优配置，并标明模型外推边界。
- `literature/grokking_scope_case.md`：用三个 grokking 研究拆分观察、经验关系候选与有限群表示算法的数学边界，区分模型内定理与训练中的现实映射。
- `literature/cross_domain_cases.md`：Q-learning、规模规律和显著图的跨层次对照，说明方法、经验关系与理论证明的边界如何区别。
- `literature/anchor_cases.md`：Hyperball、scaling law、GRPO、GSPO 的分类依据与原文范围。各方向的 `*_coverage.md` 记录缺口，`pretrain_reading_cards.md` 记录已定向检查的原文。
- `literature/coverage_protocol.md`：收录、证据和后续扩展协议，区分原文、网页与补充材料。
- `papers/`：按方向保存的本地论文 PDF；它由 `.gitignore` 排除，避免大量原始文献进入 Git。
- `notes/` 和原有 `references*.bib`：此前整理的材料。它们与新图谱并存，不应视为已经全部核验。

运行 `python3 tools/build_catalog.py` 可重新核对清单和已声明的 PDF、计算并比对已声明的 SHA-256、文件大小与页数，并更新总表。该工具不会下载、覆盖或删除 PDF，也不会将缺失文件伪装成下载成功。清单字段缺失、声明的 PDF 不可用或文件身份可疑时，它会列出问题并以非零状态退出；原生网页另行记录。检查使用本机 `pdfinfo`；缺少该工具时会如实报告受限。运行 `python3 tools/build_concept_map.py` 可根据最新总审计重建概念来源 Markdown、结构化 JSON 和反向核查队列；若论文清单与审计不一致，会要求先重建总审计。即使总审计因原文缺失返回非零，仍会生成包含缺口的审计文件，概念关联可以继续使用它。

运行 `python3 tools/build_panel_source_map.py` 可重建 panel 的来源对照；只写 `literature/panel_source_map.json` 与 `.md`。它只读 panel、书目和三份目录，不修改 panel 或冻结稿；若 panel 的行数、次序或工作名变化，会停止并要求人工更新映射。

## 分类口径

- **Knowledge**：论文在具体条件下提出方法、设计系统或报告现象；要写清其数据、模型、实验和评价范围。GRPO、GSPO 按用户指定的论文主贡献归入此类。
- **Law**：跨条件的经验关系或规律；记录被改变的变量、拟合或验证范围与失效/未知条件。Scaling laws 按用户指定归入此类。
- **Science**：从明确假设、结构或动力学出发，推导现象、预测或边界。Hyperball 按用户指定作为 Science 主贡献的示例，同时保留理论假设与经验验证的边界。

同一篇论文可以同时含算法、局部实验、经验关系和理论推导；论文主标签只是组织方式，具体主张及其范围才是核查单位。分类不代表价值高低，也不将形式推导自动等同于在现实大模型中成立。`review_level` 区分元数据、摘要、指定章节和全文：不以搜索结果或摘要冒充精读。

## 覆盖与限制

这是持续扩充的研究图谱，不假定一个有限的经典论文清单涵盖三个开放领域的“所有工作”。每个方向应记录检索范围、缺漏及无法公开下载的条目。概念条目与论文条目分开，避免把一个概念等同于唯一论文或把引用量视为分类依据。公开网络可能更新或失效，下载成功及 PDF 结构核验也不等于内容与分类已完成独立复现。
