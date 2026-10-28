# 文献覆盖、证据和后续扩展协议

这份资料库覆盖可解释性、强化学习和预训练三个开放领域。当前条目来自针对经典论文、代表性概念、一手参考文献与团队研究重点的定向检索和引文追踪；它不是在已定义完整总体上完成的系统综述，也不能证明已包含所有经典工作。最新计数与文件状态请看 [总表](catalog.md) 和 [机器审计](audit.json)；具体缺口与阅读程度分别见 [interp](interp_coverage.md)、[RL](rl_coverage.md)、[pretrain](pretrain_coverage.md)。

## 核验单位

1. 论文记录保留一手来源、版本或标识、具体主张、用户约定的 K/L/S 分类、适用条件、原文定位和阅读层级。一个论文包含多个贡献时可采用复合标签；分类不是研究价值排名。
2. 概念记录保留定义、分类、scope 和一手来源。概念、形式定义、经验现象和定理不能因名称相同而混为一谈。
3. PDF 下载、文件结构／页数、首页身份、摘要阅读、定向阅读正文、完整审读、独立复现分开登记。哈希和可解析性不证明分类正确；自动首页标题匹配只是启发式，扫描件可用基于哈希的人工确认。
4. 出版 PDF、作者稿、许可 HTML 转录、原生网页和作者海报分别标记。海报或摘要不能算作已下载论文全文；许可转录也不能冒充原版 PDF 或其分页。网页保留本地 HTML 不表示其图片、脚本和外部资源都能离线使用。从作者原始 PostScript 转换的 PDF 应保留原文件、来源和两种文件的哈希，注明转换过程，不称为新取得的出版 PDF。
5. 同一工作可能同时属于多个领域，不把跨领域重复记录当成互相独立的论文或证据。新条目应按 DOI、arXiv ID 和标题／作者检查重复。

## 向更完整覆盖扩展

- 先固定每一期的截止日期、学科边界、会议/期刊、arXiv 分类、关键词和纳入/排除规则，再保存检索式和检索时间。
- 对每个子方向建立从基础理论到现代方法、反例和复现的种子集，通过参考文献与被引文献双向追踪；对缺文、访问失败和无法确认归属的条目保留独立状态。
- 对高影响论文，从摘要初筛推进到关键定理、条件、公式、实验和反例的可定位阅读；对想用于论文的 S 主张，区分严格证明、形式假设、现实映射和尚未证明的边界。
- 用至少两名标注者对一批含有争议与混合贡献的记录独立分类，公开分歧、判断理由和修订；完成采样前不报告领域中 K/L/S 的比例。
- 为未覆盖的历史路线、多模态、系统、分布变化及失效案例分配后续批次；新增加的条目仍需下载核验，不能仅因存在引文元数据就算已审阅。

## 当前明确的文件缺口

Kool 等 2019 年《Buy 4 REINFORCE Samples, Get a Baseline for Free!》的作者页面与摘要可访问，但作者原论文链接及 OpenReview PDF 当前未取得；Google Drive 接口返回 404，不能判断是删除还是账号无权限。作者公开海报以补充材料单独保存，**不是**论文全文。Rummery 与 Niranjan 1994 年 Cambridge TR 166《On-line Q-learning using connectionist systems》的作者上传原文现可通过在线 PDF 文本读取；已核对封面、§3.2 式 (9) 与 SARSA 命名脚注、§3.3 及 §4 Figure 1（印刷页 5–8），阅读级别已由摘要升级为指定原文章节。普通本地下载仍返回 403，文件缺口保留；实验与附录未全面审计，也不把方法说明算作一般神经网络收敛定理。访问记录见 `papers/rl/rummery_remote_read_2026-10-28g.json`。R2D2《Recurrent Experience Replay in Distributed Reinforcement Learning》的主出版版 OpenReview PDF 仍返回 403；已取得并定向核对 18 页匿名审稿稿，独立记为 supplemental，不与最终版或主 PDF 缺口混同。论文和概念的阅读等级已更新为审稿稿方法定向核对，全部实验与附录仍未审计。其他缺口以各方向覆盖记录和审计结果为准。

平均回报批次新增两个主 PDF 缺口：Schwartz 1993《A Reinforcement Learning Method for Maximizing Undiscounted Rewards》的作者上传稿可远程阅读，但本地普通 GET 返回 403；Tesauro 1995《Temporal Difference Learning and TD-Gammon》已保存声明获得 ACM 许可的 HTML 转录，仍未取得出版 PDF。Tesauro 转录独立标为 `licensed_transcription`，不冒称原生出版网页、原始分页或完整离线图片。Mahadevan 1996 的本地 PDF 则来自作者公开 PostScript 的转换，两份文件均保留哈希和来源；这是可读作者稿，未声称与出版排版逐字节相同。具体访问路径及阅读位置见 RL 索引和覆盖记录。

POMDP 基础链补录另有三个主 PDF 缺口：Smallwood–Sondik 1973《The Optimal Control of Partially Observable Markov Processes over a Finite Horizon》、Sondik 1978《The Optimal Control of Partially Observable Markov Processes over the Infinite Horizon: Discounted Costs》、Lovejoy 1991《Computationally Feasible Bounds for Partially Observed Markov Decision Processes》。已核对出版社身份与摘要，但没有以 Monahan 综述或后继论文替代原始证明；三条的 Science 标签均明确为**摘要级候选、条件待核**，不表示定理已经审计。它们与上面的五项构成此前 8 个已声明主 PDF 缺口。后续稳健决策批次另增 Iyengar 2005《Robust Dynamic Programming》和 Blackwell 1962《Discrete Dynamic Programming》两项，共 10 个。控制基础批次再增 Witsenhausen 1968《A Counterexample in Stochastic Optimum Control》：已核对官方摘要并在非作者镜像转录中定位主张，原公式与条件尚未核验，保留为 Science 候选；没有把镜像转录当作已取得原文 PDF。当前共 11 个主 PDF 缺口；已有网页、海报或其他版本不会自动消除主版本缺口。

Iyengar 的 2004-05-04 作者修订报告已通过远程原文定向核对矩形性、Bellman 分解与折扣条件，本地 PDF 仍缺失，也未比较 2005 最终版；Blackwell 仅核到官方原摘要，仍是条件待核的候选。Lovejoy 的 Stanford 1988 工作论文 No.1024 是新发现的独立版本入口，公开下载流程持续返回 HTML，本轮未取得 PDF，不升级 1991 记录的阅读等级。ERIC 搜索命中的另一份旧报告实际为 Wollmer 1976，不填充 Sondik 的缺口。新来源尝试及错身份排除见 `papers/rl/pomdp_recovery_followup_2026-10-28.json`。

Altman 1999《Constrained Markov Decision Processes》仅取得 26 页出版方预览，作为 resource 单列；其前言、目录和导论不等于已读占用测度、可行性及随机化的定理章节。Howard–Matheson 1972 采用后出卷中的完整 14 页重印，重印容器日期与论文日期分别保留；它与教材预览的材料完整性不同。

旧版与勘误也分别留存。Smith–Simmons 2005 的 point-based POMDP 分析采用作者公开更正版为主文，2007-02-14 勘误与未更正 arXiv 版为补充，分类条件按修正后的折扣可达性权重记录。RLACE 以含作者凸性更正的 arXiv v8 为主文，旧 ICML 版单列对照。Shapley 1953 使用 1988 Roth 卷中的完整重印章节，明示容器与抽取页，不冒称已下载1953原排版。上述版本和材料区别不改变按具体主张分类的口径。
