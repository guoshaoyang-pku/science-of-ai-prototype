# RL 论文与概念覆盖地图（2026-10-28）

本文件是当前批次的研究索引，不是对所有 RL 论文的穷尽检索。[结构化条目](rl.json)记录每篇论文的主要主张、层级、scope、定位、已阅读程度与 PDF 状态；多主张论文另外记录定理层面的 `secondary_claims`。具体科学性和证据强度不随论文名自动确定。

## 当前覆盖

- **基础理论与算法**：Sutton 的 TD、Watkins–Dayan 的表格 Q-learning、Williams 的 REINFORCE、Sutton 等策略梯度、Konda–Tsitsiklis actor-critic、Silver 等确定性策略梯度、Ng 等势函数奖励塑形。
- **深度值函数与策略梯度**：DQN 2013 与 2015 Nature、Double DQN、Dueling、Prioritized Replay、Rainbow、A3C、DDPG、TRPO、PPO、GAE、TD3、SAC、IMPALA。
- **探索、目标、离线与世界模型**：RND、HER、C51、CQL、IQL、AlphaZero、MuZero、Dreamer 及其后续。
- **语言模型 RL 与对齐**：人类偏好、摘要 RLHF、InstructGPT、DPO、RLAIF、REINFORCE++、RLOO、DeepSeekMath/GRPO、R1、DAPO、Dr. GRPO、GSPO 和 PPO 的实现细节。
- **经验范围与测量**：奖励模型过优化经验规模关系、直接对齐过优化、LLM RL 算力与表现、Deep RL 可重复性、统计不确定性及 PPO/TRPO 实现差异。
- **概念**：模型与价值、Bellman、TD、策略梯度、离策略校正、采样基线、PPO/GRPO/GSPO、奖励与反馈、探索、离线偏移、世界模型和统计评估，当时共 107 条；当前见文末更新，每条有独立定义、主张类型、适用范围与来源。

此前版本有 **118 条论文记录，其中 116 条已下载原论文 PDF，2 条未取得**；本轮最新统计见文末。PDF 检验包括 PDF 文件签名、`pdfinfo`、文件大小及 SHA-256；这不等于独立复现实验或验证所有公式。下载目录 `papers/rl/` 已由项目根目录 `.gitignore` 忽略，不应提交大体积原文。

- **本轮扩展**：新增 MBPO、BEAR、AWAC、Decision Transformer、D4RL、QR-DQN、IQN、CPO、MADDPG、QMIX、COMA、GAIL、DRQN、MOPO、ICM 和 Agent57，涵盖离线/模型式 RL、分布式价值、安全约束、多智能体、模仿学习、部分可观测、探索及游戏基准。按具体算法或实验主张暂标 Knowledge；其中的理论结果尚待逐项核验假设和证明，不因论文含定理而整篇自动标 Science。

- **本次追加**：取得 Retrace、VDN、Ape-X、D4PG、ACER、UNREAL、SAC 温度调节 7 篇原论文；把先前 12 个摘要级条目升级为正文定向核对。新增 7 条概念。分类按主张标记，未以算法含定理便将整篇列为 Science。

- **本轮追加**：DAgger、过程监督 PRM、伪计数探索、soft Q-learning、MPO、神经情节控制、自奖励迭代 DPO，以及两篇使用不同推理度量研究 RLVR 的论文，共 9 篇原论文和 9 个概念。新增文献首页、文件身份及选定方法／导论片段已核对，尚未完成证明或全部实验审读。

- **本批补充**：新增 True Online TD、Bootstrapped DQN、Option-Critic、MAXQ、FeUdal、Go-Explore、随机先验函数、POMCP 和 Sutton 的稀疏粗编码／SARSA 实验，共 9 篇与 9 条概念；已核对 PDF 首页身份和指定方法／引言片段。Sutton 1995 论文不是 SARSA 最初来源，Rummery–Niranjan 1994 等仍待补。
- **适用边界**：true online TD 的前向视图等价限论文给定线性逼近和在线视图；MAXQ 的递归最优不等于全局最优；POMCP 需要生成式模拟器；Go-Explore 的模拟状态恢复、确定化和领域知识不能默认现实环境可用。上述定理证明与完整实验均未审计。

- **本批追加**：Deep RL deadly triad、有限时间线性 TD、Reactor、自然策略梯度四份原文；Rummery–Niranjan 的 1994 年早期报告仅以作者公开页面摘要入库，PDF 未取得。新增五条概念，定理未逐行审核。

- **模型式控制与理论**：本轮新增 PETS、PlaNet、PILCO 与 UCRL2 及四项概念；四份 PDF 已核对结构、首页、页数及 SHA。PETS、PlaNet、PILCO 已定位方法；UCRL2 §2 Theorems 2、5 的陈述在有限状态动作、奖励有界和有限直径未折扣 MDP 内给出上界及相应下界，证明未逐行核对。

## 本轮：教材基础概念定位

使用已核验的 Sutton–Barto 第二版，指定核对 12 个此前未独立核验的概念：有限 MDP、Bellman 最优算子、TD 预测、自举、同／离策略、探索／利用、状态与动作价值、Bellman 期望方程、Monte Carlo 回报、TD 误差、策略评价与改进、离策略重要性采样。各概念的书页、PDF 页和适用条件已写入 `rl.json`；与原论文来源不完全相同者明确保留原来源并注明此次核对的是教材，并未完整复核原始证明。

其中教材 §3.6 和 §4.4 给出有限 MDP 下的最优方程、唯一解与价值迭代；对于有限状态动作、有界奖励、0≤γ<1，标准无穷范数收缩界可由最优算子定义及最大值差界直接推导，本记录并未声称教材指定页给出完整收缩证明。策略改进的精确价值及逐状态不等式条件记录在 §4.2；不能直接推广到任意深度近似。教材本身并不对每个概念统一赋予 K/L/S 标签，这些标签是本文框架下的判断。

## 已定向读原文的位置

- **过程监督及 RLVR 度量**：Lightman 等 §2 明确用奖励模型进行 best-of-N，模型训练使用 MATH 测试集中的 4.5K 道题、在其余 500 题评估；不应称其直接证明通用在线 RL 受益。Yue 等用有限采样 pass@k 比较训练前后，Wen 等使用额外判定中间过程的 CoT-Pass@K；指标和判定机制不同，均不能单独证明所有推理路径的概率支持集包含关系。两篇列作有条件的 Knowledge，尚未逐项审查评价器、所有实验与理论。
- **DAgger、伪计数与最大熵方法**：DAgger §3 的数据聚合要求可查询专家；伪计数依赖密度模型的更新；soft Q-learning §3 的能量策略以近似采样实现；MPO §3 的 E/M 步是具体近似策略改进，不能把理想理论直接外推为一般深度策略的保证。Neural Episodic Control §3 用字典近邻检索价值；自奖励方法 §2 含人类标注种子，不是无外部信号的自我提升证明。


- **新增离策略算子与分布式结构**：Retrace §2–3 的迹系数为 λ min(1, π/μ)，§3.1 Theorem 1 的收缩结论针对固定目标策略和期望算子，控制还需额外条件；VDN §3 使用局部值函数的加和作为联合值函数近似；Ape-X Figure 1 和 Algorithms 1–2 区分多个 actor、共享 replay 和单 learner；ACER §3.1 使用 Retrace。科学主张只标记 Retrace 定理的给定假设，证明未独立逐行审核。
- **上一轮 12 篇方法定位**：BEAR §4.1 的误差与支持约束权衡；AWAC §IV 的隐式策略投影；D4RL §4 的轨迹拼接及数据特性；QR-DQN 与 IQN 的分位参数化；MADDPG、COMA 的集中训练；GAIL §3 的条件化占用分布等价；DRQN 的循环回放；MOPO §4 的模型不确定性惩罚；ICM §2.2 的逆／正向动力学；Agent57 §1–2 的探索策略组合。各自具体假设、位置及未复核内容记录于 JSON。
- **REINFORCE 与 TD**：Williams PDF 第 4–5 页 §3–4、Theorem 1 给出特定条件下期望更新与期望奖励梯度的关系；Sutton PDF 第 2–3、6–7 页 §1、§2.2–2.3 给出连续预测差与 TD(λ) 更新。证明未全面复核，不能将其无条件推广至深度神经网络。
- **DQN 2013、A3C、TD3、SAC、IMPALA**：分别核查 DQN §4 Algorithm 1；A3C §4–5；TD3 §4.2、§5.2–5.3 与 §6 消融；SAC §4 的表格软策略迭代及 Algorithm 1；IMPALA §4.1–4.2 的截断 V-trace 权重和固定点。有限截断 V-trace 的固定点一般不是原目标策略的值函数，SAC 表格理论也不自动保证神经网络收敛。
- **IQL、DPO 和人类偏好 RL**：核查 IQL §4.1–4.4（只在数据动作上评价 Q 的损失、expectile 和策略提取）；DPO §4 的偏好模型与 KL 正则假设；Christiano 等 §2 的片段比较与奖励预测；InstructGPT Figure 2 和 §3 的三阶段流程；Anthropic helpful/harmless 论文 Figure 1–2 的偏好模型与 PPO 流程。定理证明和完整实验仍待核查。
- **世界模型与棋类**：MuZero §3、Figure 1 的隐状态模型预测奖励、价值和策略，不要求重建图像；Dreamer §2–4 在潜在动态中想象并训练行为；DreamerV2 §2.1–2.2 使用离散潜变量；DreamerV3 Figure 2–3 包含多领域实验；AlphaZero 在已知规则棋类中用自博弈训练策略/价值网络和 MCTS。结果受各自任务及计算条件约束。
- **新增四篇正文定位**：MBPO PDF 第 1–2 页短模型轨迹、Decision Transformer PDF 第 1–4 页回报条件化自回归生成、CPO PDF 第 1–2 页 CMDP 及期望成本约束、QMIX PDF 第 1–2 页单调价值混合和局部贪心一致性。安全约束不等于每次轨迹零风险；QMIX 的单调结构有表达限制。
- **Dyna**：PDF 第 1–5 页 §1–2、§4–5 的 Dyna-PI、学习的世界模型与 Dyna-Q 已定向核对；证明和实验未完整复核。
- **离线与时间抽象**：新增 BCQ §3–4 的生成候选动作及受限扰动、TD3+BC 第 6–8 页的行为克隆正则和 D4RL 结果，以及 options 作者论文 §2 的启动集合、策略、终止条件。Options PDF 字体映射导致文本提取乱码，其标题和第 4、7–8 页以页面图像核对；定理证明仍待审核。
- **对齐及缩放**：TL;DR RLHF、RLAIF、DeepSeek-R1、DAPO、REINFORCE++ 与 PPO 实现细节论文的目标和方法已查原文片段。直接对齐过优化论文 §3.1–3.2 将 GPT-4 评判胜率用于既有拟合形式，不能写成直接人类奖励；RL 算力论文 Figure 2–3 的 sigmoid 指同分布验证通过率，拟合的渐近值不是无条件任务上限。
- **深度 RL 谱系及评估**：Dueling、优先回放、Rainbow、DDPG、HER、RND、C51，以及实现/复现和梯度诊断研究均已定位原文方法或图表；分布式控制算子的反例不应与固定策略评估混同。
- **GRPO**：DeepSeekMath PDF 第 12–14 页 §4.1.1–4.1.3、式 (3)、outcome/process 监督；记录组内奖励基线，不把两种监督混为一谈。
- **GSPO**：PDF 第 2–5 页 §4.1、式 (5)–(7)、§5.1。其权重是**完整序列新旧似然比的 1/输出长度 次方**，不是未经归一化的完整序列重要性比。Qwen3-30B-A3B 的稳定性和效率属于该实验设置。
- **Q-learning**：PDF 第 2–4 页，定理在 PDF 第 4 页／期刊页码 282；有界奖励、状态动作访问及各状态动作步长级数条件已视觉核对；证明没有完整复核。
- **Baird 的 star problem**：作者原文 PDF 第 2–3 页 §2–3 和 Figure 1；有限状态与线性函数逼近下的直接更新可不稳定，不能泛化成所有函数逼近方法都会发散。
- **策略不变奖励塑形**：PDF 第 3–4 页 §3 Theorem 1；无 MDP 先验的普遍策略不变性中，势函数形式的必要性指其他形式**存在**反例，并非每个 MDP 都失败。
- **策略梯度**：Sutton 等 PDF 第 2–3 页 §1 Theorem 1；Silver 等 PDF 第 4 页 §3.2 Theorem 1，需保留各自的 MDP 和可微性假设；证明没有完整复核。
- **TRPO、PPO、GAE**：TRPO §3 Theorem 1 与 §4 实际算法近似要分开；PPO §3 clipping 不自动等于硬 KL 约束；GAE §2 Proposition 1 中 γ-just 是相对于折扣梯度，无折扣目标可能已有偏差。
- **RLHF 多采样和偏置**：RLOO 文献 §2.3 将 Kool 等 2019 列为先例；Dr. GRPO §3.1–3.2 对响应长度及题目级标准差归一化提出偏置分析，具体效果依赖论文的 Qwen/MATH 设定。
- **奖励过优化**：Gao 等 PDF §1–2、§3.2 使用合成 gold reward model，并分别拟合 PPO 与 best-of-n 的形式，不应说成直接测量真实人类偏好。
- **其他定向片段**：2015 DQN 的 49 游戏结果；Double DQN 的选择/评价分离；CQL §3.1 Theorem 3.2 的期望值及采样误差边界；RLiable 的区间、性能分布与 IQM 建议。

以上是对特定段落或定理**陈述**的核查，不是对整篇论文、证明或实验的全面复核。彼时 116 份已下载原论文已定向核对方法、定理陈述或实验的部分正文，但尚未对整篇论文、全部证明或完整实验逐项审核；Kool 与 Rummery–Niranjan 1994 原文未取得，仍为摘要级。具体程度以每条 `review_level` 和 `evidence` 为准。彼时 107 条概念中，除明确标记的原文片段外，定义与对应来源仍待逐条精读。

## 访问未完成的来源

1. **Rummery 与 Niranjan (1994)**：[On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems)，Cambridge 报告 CUED/F-INFENG/TR 166；作者上传页面摘要可访问，原始 Cambridge PDF 因 TLS 连接失败，ResearchGate PDF 返回 403，CiteSeerX 被重定向至无法连接的归档。现仅依据作者摘要初筛，不称已取得 PDF。Cambridge 原链接为 https://mi.eng.cam.ac.uk/reports/svr-ftp/auto-pdf/rummery_tr166.pdf 。
2. **Kool 等 (2019), Buy 4 REINFORCE Samples, Get a Baseline for Free!**：作者页面及摘要已确认；OpenReview PDF 返回 403；作者页指向的 Google Drive 只读元数据查询返回 NOT_FOUND / HTTP 404，无法判断文件删除还是当前账号无权限。已另取作者公开的 [会议海报](https://wouterkool.github.io/pdf/poster-buy-4-samples-free-baseline.pdf)，确认标题与作者并作为补充材料收录；这不是原论文，论文 PDF 仍标记未取得。
3. **Sutton 1990 Dyna 论文**：已从 [HTTP 公开镜像](http://www.derongliu.org/adp/adp-cdrom/refs/sutton19900216.pdf) 获取 9 页 PDF；首页题名、作者与 ICML 1990 会议信息和 ML Anthology 核对一致。该来源 HTTPS 证书不受信任，HTTP 下载本身不提供传输完整性认证。
4. **Sutton–Barto 教材第二版**：作者站 PDF 的 TLS 连接失败；已从丹麦技术大学课程镜像取得并核验 2018/2020 第二版（548 页、73,129,769 字节；第 3–4 页确认作者、标题与 CC BY-NC-ND 许可），仅确认书目和文件身份，未全文审读。教材单列资源，不对整本书赋单一分类。文件超过 50 MB，位于被忽略的 `papers/`，不得直接提交。

Baird 的 HTTPS 作者站曾遇 TLS 错误；后续通过作者站原文的 HTTP 版本取得论文，核对了标题、作者和正文；同站 1995b2 为会议幻灯片，不作为原论文替代。

## 重要缺口与下一批

- 经典来源：SARSA 早期 Rummery–Niranjan 报告已建摘要条目但 PDF 未取得；eligibility traces 的原始理论仍需查找合法可访问原文并逐篇定位；Dyna 已取得原文并定向核对，options 的定理及证明未全面审读。
- 理论：Bellman 与策略梯度收敛的必要条件、函数逼近反例与离策略不稳定的正式假设；当前 Science 多为充分条件或局部结果，不能冒称完整边界。
- 算法谱系：分布式强化学习与分布 RL 的其他重要算法、层级 RL、安全 RL 与多智能体 RL 仍不完整（已补入 CPO、MADDPG、QMIX、COMA）；BCQ/TD3+BC 已补入但其余离线算法谱系仍不完整。
- 大模型 RL：偏好模型与可验证奖励、不同 group normalization/length reduction、过程奖励、离策略和 token/sequence 归一化的原始论文与后续相互反例仍需补充。
- 证据完成度：对定向读过的文章补齐实际图表、训练条件、消融、反例与可复现条件；对 Law 逐轴查外延，对 Science 区分充分条件、必要条件和真正闭合边界；根据同一规则对双人标注分歧进行复核。

## 随机与对抗 bandit 及蒙特卡洛规划

本批新增 UCB1、EXP3、UCT 与 Thompson sampling 教程 4 份原文、4 条概念；核对了方法或指定定理陈述，未逐行审核证明。UCB1 的对数 regret 界依赖固定分布、有界奖励与各臂间隙；EXP3 的 weak regret 参照最佳固定臂；UCT 的有限时域结论不自动适用所有神经引导搜索。Thompson sampling 依赖明确先验、后验与反馈。新增原文均经首页标题、页数和 SHA-256 核对。

## 本批：LSPI、AlphaGo 历史与奖励可利用性

新增 LSPI 与有限样本 LSTD/LSPI 分析、2016 AlphaGo、2017 AlphaGo Zero 作者接收稿、以及 reward hacking 的条件化理论；5 份论文 PDF 分别核验标题、页数与 SHA-256，并读指定方法或定理陈述。2016 AlphaGo PDF 的文字提取报告 Invalid Font Weight 警告，正文可提取。区分 2016 人类棋谱监督与 2017 自博弈；后者仍使用围棋规则及模型／搜索结构。奖励可利用性论文 v2 Theorem 1 的结论仅限非平凡奖励对与包含开集的策略集合，不能解释为任意优化都会作弊。LSPI 理论界的泛化条件区分平稳 β-mixing 和后续策略迭代额外假设。证明、完整实验与现实映射均未独立复核。
## 本轮：策略更新与概念来源核对

新增 REPS、保守策略迭代（CPI）及 Path Consistency Learning 共 3 篇原论文和 3 条概念：REPS 的约束针对状态–动作联合分布，不能误写为每个状态的硬 KL 约束；CPI 以重启分布、近似贪心选择器及新旧策略混合为设定；PCL 的软一致性理论以熵温度及论文 MDP 假设为条件。PCL 的定理陈述、REPS 的方法已定位，CPI 仅视觉核对首页摘要和引言；其旧字体使文本提取乱码，已请求以文件 SHA-256 绑定手工身份核验，不声称完整证明已审阅。

另外将 13 条概念从未独立核对推进到指定论文的方法或定理陈述：REINFORCE、优势函数、GAE、TRPO、PPO clipping、经验回放、actor–critic、Double Q、优先经验回放、分布式 RL、HER、自博弈、模型式／无模型。分别保留 γ-just、TRPO 实践近似、PPO 比率不是硬约束、优先重放权重指数、分布 Bellman 评估／控制区别等范围；这些是局部核对而非全文审读、证明审计或独立复现。数据当前 121 篇、110 概念；Kool 与 Rummery 两篇原论文仍缺。


## 本轮：探索、PAC 与离策略稳定性复核

新增 R-max、有限 MDP 的 PAC 分析和 emphatic TD 三篇原文与三条概念。R-max 的保证以已知最大回报、有限显式状态与混合时间等条件为前提，比较的策略类有明确限制；PAC-MDP 界依赖论文设定的有限折扣 MDP 和参数，不能推及所有深度 RL；emphatic TD 原文论证期望更新稳定，明确没有声称该文给出了随机算法的完整收敛证明。三份 PDF 核对首页标题、页数、SHA-256；定理的完整证明和实验尚未复核。

本次数据为 **124 篇论文、113 条概念；122 篇原论文 PDF 已取得，2 篇未取得**。对 14 个关键或混合主张的定位、原分类、判决和局限见 [RL 审计记录](../survey/rl_adjudication.md)。

## 本轮：示范学习、策略更新和梯度时序差分

新增 DQfD、PPG、V-MPO、GTD（2008）及 GTD2/TDC（2009）5 篇原论文与 5 条对应概念，所有原论文 PDF 均检查首页题名、有效页数及 SHA-256。DQfD 的示范预训练、PPG 的策略／辅助阶段、V-MPO 的前 50% 优势和两类 KL 约束是具体方法而非普遍效果定律。GTD Theorem 4.1、GTD2/TDC Theorems 1–2 是有条件的 Science 候选：限线性策略评价、i.i.d. 样本、有界二阶矩、矩阵非奇异和指定步长；TDC 还要求两时间尺度。GTD 的子采样定理另有行为策略覆盖等假设。这里只查定理陈述与指定方法，没有逐行审计证明或复现实验；2008 NeurIPS 网页标题与 PDF 首页措辞次序不同，以 PDF 身份、作者及主办方来源交叉核对。

10 条既有概念补核对原文定位：奖励模型过优化、奖励作弊、离线分布偏移、跨种子与任务统计不确定性、结果奖励、过程奖励、KL 正则、序列概率比、组内优势归一化及最大熵强化学习。Gao 等的 gold 奖励由合成模型提供；GSPO 使用完整序列比值的长度倒数次方；DeepSeekMath 的过程优势按其具体后续步骤奖励定义。此轮后共 **129 篇论文、118 条概念**，其中 7 条概念仍标记未独立核对；Kool 与 Rummery–Niranjan 的原论文 PDF 仍未取得。完整论文、证明和实验的系统审核以及后续谱系扩展仍待完成。

## 本轮：值函数表示、批量拟合与条件化收敛

新增 Tsitsiklis–Van Roy 线性 TD 分析、Ernst 等 fitted Q iteration、Even-Dar–Mansour 的 Q-learning 步长有限时间分析、Dayan 后继表示、UVFA、Barreto 等 successor features 与广义策略改进，共 **6 篇原论文、6 个概念**。六份 PDF 均核对首页、结构、页数及 SHA-256。原文方法或定理陈述已定位，完整证明、全部实验和独立复现均未完成。

必须保留几个范围差异：线性 TD 的在线轨迹收敛不意味着任意神经函数逼近或离策略控制收敛；Even-Dar–Mansour 的 1/t 慢速下界来自特定构造与采样模型；FQI 的收敛、回归精度与策略最优是不同结论；Dayan 的状态占用依赖动力学与策略；UVFA 的“universal”不证明所有新目标可泛化；广义策略改进界要求所有状态动作上的统一价值误差控制，successor features 的迁移设定还需固定动力学与共享奖励特征。条件化定理标为 Science 候选，并不等同于已经刻画全部可应用与失效边界。

当前 **135 篇论文、124 个概念；133 份原论文 PDF 已取得，2 份未取得**。搜索 Cambridge 技术报告和 Kool 作者页面后仍未找到可访问的两份原论文 PDF；已有作者摘要及 Kool 海报不能替代原文。

## 本轮：资格迹、离策略、规划与回放

新增 9 篇原论文和 9 条概念：Singh–Sutton replacing traces、Precup 等离策略策略评价与逐决策重要性采样、GQ(λ)、Q(λ) with Off-Policy Corrections、Kearns–Mansour–Ng 稀疏采样、Kearns–Singh E³（ICML 1998 短版）、Never Give Up、类别分布 RL 分析和经验回放基础研究。PDF 均已检查结构、页数、大小、SHA-256 及首页身份；其中 replacing traces 为扫描件无文字层，已渲染首页目视核对标题与两位作者，须在总审计的手工身份清单中按文件哈希记录。ICML 1998 短版不冒称已核验 2002 年期刊完整版。

选定方法及定理陈述已定位，完整证明、全部实验与独立复现尚未完成。需要保留的边界：replacing trace 的首篇理论分析限定离线 TD(1) 与无折扣吸收马尔可夫链；逐决策重要性采样需目标策略的行为策略覆盖；Q(λ) Theorem 1 在固定目标与行为策略、表格更新等假设下，以 λε<(1−γ)/γ 给出一个充分条件，不能把边界外说成一定发散；稀疏采样需要任意状态动作的生成模型，状态数无关的单次规划界仍依赖时域、动作数和精度。类别投影的结果属于论文固定策略和指定距离下的分析，不能迁移为深度控制的普遍收敛保证。

经验回放论文 Definition 3 将 replay ratio 定义为每环境转移对应的**梯度更新次数**；不同批量下不能与采样 transition 数/环境转移混用。其不同容量、最旧策略年龄和性能的观察标为经验关系候选，未宣称跨所有任务已充分释放约束。当前 RL 清单为 **144 篇论文、133 条概念**；Kool 和 Rummery–Niranjan 两篇原论文 PDF 仍未取得，其他来源的 PDF 已取得。R2D2 原始 OpenReview PDF 此次直接下载返回 403，未伪装成已下载。

## 本轮：多智能体基础、离策略评价及悲观模型

新增 **9 篇论文记录与 9 个概念**：Tan 1993 的独立／合作学习、Littman 1994 的 Markov game 与 minimax-Q、Jiang–Li 的序贯双重稳健评价、Thomas 等的高置信离策略评价、QTRAN、QPLEX、MAPPO/IPPO、MOReL 与 R2D2。前八篇取得原论文 PDF，均核对文件结构、首页标题及作者、页数、大小和 SHA-256；其中四篇 arXiv 论文还核对首页编号。PDF 原文方法或定理陈述已定向阅读，证明、全部实验与独立复现未完成。

不能混淆如下边界：Littman 的 minimax-Q 以两人零和 Markov 博弈为对象；QTRAN 理论给出其形式条件下局部与联合贪心一致的充分条件，近似优化不保证这些条件精确成立；QPLEX 的架构一致性不等于实际训练可找到真最优价值。Jiang–Li 双重稳健评价保留目标／行为策略覆盖、概率比和价值估计独立性条件，模型或环境随机性仍可导致高方差；高置信离策略评价的下界依赖统计假设，不能承诺单次执行安全。MOReL 的实际性能受数据覆盖、模型与不确定区域判别影响；MAPPO 的实验结果只对应论文所测合作任务和配置。

R2D2 已按用户要求作为独立论文和概念记录入库：作者网页摘要可访问，但 OpenReview PDF 两种直链均返回 403，**只有摘要初筛，未取得本地原论文 PDF，也未将网页或补充材料当作原文**。Kool 2019 的 OpenReview PDF 本轮复查仍 403；Rummery–Niranjan 1994 的 Cambridge 原链接仍出现 TLS 连接失败，均未取得 PDF。RL 当前为 **153 篇论文记录、142 个概念**，其中 150 份 PDF 可用、3 篇原论文 PDF 未取得；跨全项目的最新版审计由主线程统一生成。



## 本轮：同策略收敛、Double Q 与乐观规划

新增 **7 篇论文和 7 个概念**：Tsitsiklis 异步随机近似与 Q-learning、Singh 等单步同策略收敛、van Hasselt 的原始 Double Q-learning、Expected SARSA、UCBVI 的有限时域遗憾界、Prioritized Sweeping 的预印本，以及 Cramér 距离与有限样本 Wasserstein 梯度偏差。七份 PDF 的签名、页数、大小和 SHA-256 已核验；除同策略收敛论文外，首页标题可提取且核对。该论文是无文字层扫描件，已渲染首页核对标题、作者及摘要，其他定理条件与证明尚未审阅；其对应哈希需总审计人工身份表登记。

定向核对了 Double Q 的两表更新与 Theorem 1、Expected SARSA 的 Algorithm 1 和 Theorem 1、Tsitsiklis 指定随机近似假设与折扣 Q-learning 应用、UCBVI 的 MDP 假设和两种置信上界变体、Prioritized Sweeping 的前驱队列，以及 Cramér 论文的 Bernoulli 反例与距离性质。**证明、完整实验和独立复现均未完成。** Double Q 的双估计可低估，不等于有限样本一般无偏；Expected SARSA 的方差优势与收敛都依赖指定条件；UCBVI 的遗憾上界以有限时域、有限状态动作及作者规定的奖励与转移模型为前提。Prioritized Sweeping 的这份 PDF 首页写 *Less Real Time*，与期刊标题的措辞有差异，明确按取得预印本记录。

重新检索 R2D2 的 OpenReview 原 PDF 哈希直链仍返回 403；Rummery–Niranjan 的 CiteSeerX 索引指向归档但实际下载发生 TLS 失败。Kool 仍只有作者网页与海报，三篇原始 PDF 尚未取得，不把搜索索引或海报冒充全文。本轮 RL 清单现为 **160 篇论文记录、149 个概念**，其中 157 份原论文 PDF 已取得、3 篇未取得；全项目审计由主线程统一生成。

## 本轮：逆强化学习、随机价值探索与模型规划（2026-10-28）

新增 9 篇原论文与 9 个概念：Ng–Russell 2000 的逆强化学习、Abbeel–Ng 2004 学徒学习、Ziebart 等 2008 最大熵 IRL、Kaelbling 等 1996 RL 历史综述、Osband 等 RLSVI、Reanalyse/MuZero Unplugged、TD-MPC、TD-MPC2，以及 Arcade Learning Environment。9 份 PDF 均核验签名、页数、大小与 SHA-256，并定位指定方法或理论陈述；并未逐行审计所有证明或独立复现实验。

- Ng–Russell 的 Theorem 3 在**已知有限 MDP、完整确定性策略与状态奖励**下，给出指定策略最优的奖励集合的必要且充分条件；零奖励是退化解，故它并没有识别唯一的真实奖励。该原文是 Cambridge 课程扫描镜像，无文字层；已视觉核对第 1–3 页标题、作者、定义和定理陈述，须依哈希记录人工身份。ML Anthology 对这篇论文列出的 DOI `10.2460/AJVR.67.2.323` 与该文无关，未采用该 DOI。
- 学徒学习中匹配特征期望的近似保障依赖已知有界特征、线性奖励、折扣与规划假设；不会自动识别专家真实奖励。最大熵 IRL 所得全球归一化路径模型针对文中的特征和环境假设，随机转移推理另有近似。取自作者更正版，作者站注明已修正原发表算法中的 off-by-one。
- RLSVI 的 Theorem 1 是指定 Dirichlet 先验、有限时域表格特征及参数条件下的 **Bayesian 期望遗憾界**，不是对任意深度网络或所有 MDP 的频率学最坏情况保证。原论文的数学文字抽取有字符失真，定理页已目视核对，PMLR PDF 有注释目的地警告但结构有效。
- Reanalyse 对保存经验重新生成规划训练目标；TD-MPC 和 TD-MPC2 用学习的潜空间动力学与终端价值进行短时域规划。它们的实践结果受模型误差、规划和环境决定，不自动保证全局最优。ALE 是具体 Atari 模拟与评测平台，游戏、随机性和帧数统计不同可导致分数不可直接比较。

本轮后 RL 清单为 **169 条论文记录、158 个概念，其中 166 份论文 PDF 已取得、3 篇未取得**。本次重试 R2D2 官方 OpenReview PDF、内容哈希链接和公开 API 均返回 403；Kool 作者 PDF 链接未发现可访问原文，Rummery 的 Cambridge 服务器连接仍失败。这些状态不改变原有缺口。

## 本轮：两篇扫描版理论论文的定理陈述核对（2026-10-28）

对此前仅核对摘要的两篇原文进行视觉定向阅读，并更新论文、概念和具体主张：

- Singh 等《Convergence Results for Single-Step On-Policy Reinforcement-Learning Algorithms》PDF 第 3、5–8 页：核对 GLIE、表格 SARSA(0)、Theorem 1 的有限状态动作、折扣、步长及奖励方差条件。原文的 Lemma 1 不重述既有证明；这里没有逐行独立审计证明，也不将充分条件写成必要条件或推广至神经网络。
- Kakade–Langford《Approximately Optimal Approximate Reinforcement Learning》PDF 第 4–6 页：核对策略混合式、Theorem 4.1、Corollary 4.2 与对目标分布的覆盖因子。性能改进界针对指定重启分布，目标分布近最优性还依赖访问分布比值；不能视作任意实际 PPO 实现的保证。证明尚未逐行审计。

两份扫描文本提取失效，因此以上范围来自直接查看渲染页；PDF 身份以手工视觉记录和 SHA-256 绑定。

## 本轮：探索、元强化学习、延迟奖励和策略更新（2026-10-28）

新增 **12 篇论文与 12 个概念**：NoisyNet、PEARL、RUDDER、Off-PAC、RL²、DIAYN、VIME、Q-Prop、参数空间进化策略、episodic curiosity、VariBAD 和 PGQL。12 份原始 arXiv PDF 均核验签名、pdfinfo、首页题名和 arXiv ID、页数、大小及 SHA-256；所标方法、定理陈述或勘误已定向阅读，未逐行审计全部证明或复现实验。

- **RUDDER** 的返回等价和保持最优策略针对论文定义的序列 Markov 过程；其理想奖励重分配以指定策略和精确期望为条件。近似 LSTM 与有限训练不自动达到这些理想条件，也不能推断任何任务均加速。
- **Off-Policy Actor-Critic** v5 的 Theorem 3 写有 λ=0、i.i.d. 转移、行为策略正覆盖、线性 critic、投影及双时间尺度等条件；PDF 第 18 页 Appendix B Errata 明确将现有理论缩至表格策略表示，并指出近似梯度稳定性及证明问题。本记录保留受勘误限制的条件化 Science 候选，不声称神经策略的一般收敛或全局最优。
- **PGQL** 的熵正则固定点关系在文中表格条件下推导；作者明确一般函数逼近下对应关系不必成立。算法、条件化理论及论文实验分别判断，不能将固定点关系直接推广到任意深度策略。
- PEARL 与 VariBAD 的任务后验是由指定模型近似推断，RL² 的隐状态在同一任务 trial 的 episode 之间保留；这些设计不等于任意任务的贝叶斯最优保证。DIAYN 无需外部任务奖励，但仍以技能判别器构造内在学习信号。
- NoisyNet、VIME、episodic curiosity、参数空间 ES 和 Q-Prop 均按具体方法及条件化实证归 Knowledge；噪声、内在奖励、可达性模型、平滑目标和数据采样条件不得混为普遍探索／收敛规律。

本轮后 RL 清单为 **181 篇论文记录、170 个概念，其中 178 份论文 PDF 已取得、3 篇未取得**。Kool 与 R2D2 的 OpenReview 原文重试仍返回 403；Rummery–Niranjan 的 ResearchGate 页面显示作者上传原文及摘要，但 PDF 下载仍返回 403，未把页面或搜索缓存当成本地全文。


## 本轮：正则化 MDP、安全改进、动作基线争议、多智能体通信与偏好优化（2026-10-28）

新增 **12 篇论文与 12 个概念**：正则化 MDP、SPIBB、RIAL/DIAL、CommNet、因子化动作相关基线及其反例分析、Expected Policy Gradients、受破坏奖励通道、KTO、ORPO、SimPO、ΨPO/IPO。12 份原始 arXiv PDF 均检查 PDF 签名、首页题名及 arXiv 版本、页数、字节数和 SHA-256；实际阅读限于记录所列方法及定理陈述的相关页面，未完整审查证明、实验或独立复现。

- **正则化 MDP** 的 Bellman 收缩与不动点属于有限、折扣和文中正则器等条件下的形式性质；不能因此声称深度网络优化收敛。**SPIBB** 的安全改进保证是高概率的 ζ-近似界，允许相对基线出现界内损失，指定有限表格 MDP、经验模型和策略约束；神经近似及其他变体不能自动继承。
- 动作分量基线与 *The Mirage of Action-Dependent Baselines* 保留为成对争论：前者依赖分量因子化与对应基线条件，后者在具体对照和环境下指出收益可能消失。两者不构成“动作基线一律有效”或“一律无效”的普遍结论。EPG 的动作期望计算与可行性也依赖参数化。
- **受破坏奖励通道** 的不可能性依赖有限均匀离散奖励、最坏情形遗憾定义及完整真实奖励／破坏函数族；有结构的函数类并不自动落入该结论。**ΨPO 命题 1** 的最优策略等价依赖 Bradley–Terry 偏好和 log-odds 变换，不能推到有限数据训练结果或偏好模型失配。以上理论标为带条件 Science 候选，证明尚未独立审计。
- RIAL/DIAL 和 CommNet 按具体多智能体架构、通信信道和协作任务记录；KTO、ORPO、SimPO 是具体偏好优化目标，保留数据结构、参考模型与归一化条件，不能直接称为通用对齐定律或在线环境 RL。

本轮后 RL 清单为 **193 篇论文记录、182 个概念，其中 190 份原论文 PDF 已取得、3 篇未取得**。再次查询 R2D2 的 OpenReview 原文可见搜索缓存摘要，但直接访问仍遇验证或 403；Kool 的作者页面所指原文不可获取；Rummery–Niranjan 的 Cambridge 来源连接失败、ResearchGate 下载受阻。搜索缓存、海报、摘要和其他替代材料均未作为原论文 PDF 入库。

## 本轮：既有低审阅等级概念的原文定向复核（2026-10-28）

逐条复核原先标为 `not_independently_reviewed` 或 `source_abstract_only`、且本地已有原论文的 **20 个概念**，覆盖 19 篇不同论文。现已按各概念指定实际读到的 PDF 页码及方法、设定或相关实验，补充定义与适用范围，并统一标记为 `targeted_method_or_empirical_sections_proofs_and_experiments_not_fully_checked`。这是所列段落的定向阅读，**不是全文或全部定理证明的独立审计，也不是实验复现**；没有仅凭元数据升级。既有分类维持不变，分类应继续按具体主张判断。

- PPO 截断发生于代理目标，而非硬性限制所有概率比或 KL；GAE 的 γ-just 性质针对指定折扣梯度，不能混为未折扣目标的无偏性。
- CMDP 的成本是指定分布下的期望累计成本，不保证每条轨迹安全；QMIX 单调分解与局部贪心相容，受可表示联合价值的单调性约束；COMA 的反事实基线不自动构成一般因果识别。
- MuZero 的潜在模型预测规划所需的价值、策略和奖励，而不是要求重建观测；MOPO 理论下界需要模型误差上界等前提，实践的不确定性代理不自动满足这些前提；GAIL 需要当前策略与环境交互。
- RND、ICM、奖励模型及可验证奖励均是具体代理信号；其数值或判定不自动等于任务成功、真实偏好或完整推理正确性。Decision Transformer 的目标回报也不是保证达成的回报。
- 计算效率、环境样本效率与墙钟时间分别计量；ScaleRL 曲线限定于论文的具体模型、任务、配方和拟合，不能据此概括为所有 RL 的定律。

本轮未新增论文。另有 Rummery–Niranjan 1994、R2D2 的 **2 个原文缺失概念**仍只按摘要记录；Kool 2019 另有 1 个以二手方法材料核对、原文待取的概念，均未冒充读到原文。本轮搜索作者页面：R2D2 作者所列链接仍指向此前受 403 限制的 OpenReview PDF；Kool 作者页面列出的材料为报告资料，不直接视作论文原文；未发现可确认的新原文。当前 RL 论文清单仍为 193 篇、概念 182 个，其中 190 份 PDF 已取、3 篇原 PDF 未取。


## 本轮：离策略评价、离线策略学习与后验探索（2026-10-28）

新增 **11 篇原论文与 11 个概念**：PSRL、FQE 与约束批量策略学习、DualDICE、正则化拉格朗日 OPE、离散 BCQ 的 Atari 基准、OPE 实证比较、一步离线策略改进、深度 OPE 基准、GenDICE、WDR 与 MAGIC、PEVI 悲观价值迭代。11 份 arXiv 原论文均检查首页题名和版本、PDF 结构、页数、文件大小及 SHA-256；记录的特定方法、条件或定理陈述已定向核对，未逐行审计证明，也未复现实验。

- **PSRL** 的 Theorem 1 是对 MDP 先验平均的 Bayesian 期望遗憾界，属于固定 episode 长度及有限状态／动作设定；不应当作任意深度强化学习的频率学最坏情况界。
- **FQE** 保持目标策略固定并用 Bellman 目标拟合价值；DualDICE 的行为策略未知不等于目标占用可以超出离线数据支持，原文 Assumption 1 仍要求覆盖。GenDICE 的目标链正则性、正则化 DICE 的固定点与比值条件，以及有限数据函数近似误差均需分别记录。
- **WDR** 使用归一化权重并可有有限样本偏差；Theorems 1 和 2 各有不同充分条件给出强一致性。**PEVI** 的高概率次优上界依赖真正满足 Definition 4.1 的不确定性量化，其数值由最优策略轨迹上的量化误差决定；不能用“悲观”一词替代假设或保证小误差。
- OPE 与离线 RL 的基准、一步策略改进、离散 BCQ 的结果均按方法和实验设置分类为 Knowledge；Regret@k 度量候选集内的策略选择遗憾，与直接回报预测误差不同。

当前 RL 清单 **204 条论文记录、193 个概念，其中 201 份论文 PDF 已取得、3 篇未取得**。再次核查 R2D2 官方 OpenReview，公开搜索可读到摘要缓存，但原文访问仍要求浏览器验证，不等于已下载原文。Kool 和 Rummery–Niranjan 原文仍未取得；原有三个条目的阅读级别与缺口保持不变。

## 本轮：层级控制、无监督技能与目标条件克隆的边界（2026-10-28）

新增 **9 篇原论文与 9 个概念**：1992 年 Feudal RL、h-DQN、HIRO、HAC、Laplacian eigenoptions、VIC、DADS、GCSL，以及 *Imitating Past Successes can be Very Suboptimal*。9 份 PDF 均核对标题、作者、结构、页数、大小与 SHA-256；8 份 arXiv PDF 首页核对编号及版本，Feudal 使用 NeurIPS 官方会议 PDF。局部方法和指定定理陈述已定位，未全面审计证明和实验或独立复现。

- Feudal 的奖励隐藏对应上层对子任务的奖励；h-DQN 用内在目标奖励训练低层，同时高层优化环境奖励，层级优势以具体任务、目标及基线为范围。HIRO 按当前低层动作概率重标记历史高层目标，HAC 则将实际到达状态作为后见高层动作；二者不保证学习过程完全消除非平稳性或低层已达到最优。
- Eigenoptions 源自状态表示的谱方向：原文 Figure 6 也报告少量 options 可使探索变慢，不能只记录探索加速。VIC 衡量选项与终止状态间的互信息；DADS 衡量给定当前状态时技能与下一状态的互信息，并在技能动力学模型上规划。两种优化目标、状态分布和终止机制不同，均不能保证覆盖所有任务或状态。
- GCSL Theorem 3.1 的下界含新旧策略最大 TV 距离项，Theorem 3.2 要求确定性动力学、满支持数据收集策略，以及全状态—目标—剩余时域的近似控制，**不等于**实际神经训练在任意随机环境单调改进。Eysenbach 等的 OCBC Proposition 3.4 是特定结果重标记及 Bayes 最优克隆设定中的存在性反例；其比较对象是任务条件行为策略。反例不意味着全部目标条件模仿失败，也不直接推翻条件不同的 GCSL 定理。两篇论文的证明尚未逐行审计，相关 Science 标签是类型与候选，非独立验证结论。
- 检索时发现 arXiv:1805.08180 已由作者撤回，arXiv 页面注明其为重复稿并指向 arXiv:1712.00948 的 *Learning Multi-Level Hierarchies with Hindsight*；本库收录后者，不把重复稿计作不同论文。

本轮后 RL 清单 **213 篇论文记录、202 个概念；210 份论文 PDF 已取得、3 份仍缺**。原缺的 Kool、Rummery–Niranjan、R2D2 不因本轮补录而改变状态；此处不冒充所有 RL 经典论文已完成覆盖。

## 本轮：三个原文缺口与 R2D2 审稿稿（2026-10-28）

本轮没有新增论文或概念。**取得 R2D2 的 18 页匿名审稿稿，但主出版版本 PDF 仍未取得**；为避免版本混同，该文件在 `rl.json` 中独立记为 `supplemental`，不改变既有主 PDF 的缺失统计。当前仍是 **213 篇论文、202 个概念、210 份主 PDF 已取、3 个主 PDF 缺口**，另有 R2D2 原始审稿稿和 Kool 海报两份明确标记的补充 PDF。

- **R2D2**：作者主页指向的主 PDF 本轮标准请求仍返回 HTTP 403 / `text/html`。公开文献链接所列 [OpenReview reference PDF](https://openreview.net/references/pdf?id=Hy7PKCFCQ) 正常返回 HTTP 200 / `application/pdf`，已保存为 `papers/rl/r2d2_2019_review_draft.pdf`：18 页、7,448,938 bytes、SHA-256 `3caaa2a7fd381d017c022dd3fa5a4a52647dcb5d555a3a80931b7dfbfaecd87e`。首页明确为匿名 ICLR 2019 审稿稿；实名作者依据 [Will Dabney 官方论文页](https://willdabney.com/publication/r2d2/)核对，不能说匿名 PDF 印有实名作者。
- **定向阅读**：R2D2 审稿稿 PDF 第 2–5 页 §2.3–3 与 Figure 1 已核对。Stored state 可能因参数变化而陈旧；burn-in 用前缀展开循环网络，损失更新从后续序列开始；Q-value discrepancy 比较相同当前网络权重作用于不同循环状态时的输出。原文对此的 DMLab 消融是有限设置下的经验证据，不是一般收敛、完整历史恢复或所有任务普适性的保证。相应论文及概念现标为审稿稿方法定向核对；最终版、全部实验及附录仍未审计。
- **Rummery–Niranjan 1994**：新增核对 [G. A. Rummery 于 1994-10-04 发布的原始报告公告](https://groups.google.com/g/comp.ai.neural-nets/c/GtWN2rmxvME)，题名、两位作者、TR 166、摘要及公开 FTP 文件名一致。原公告的 FTP 主机无法解析；Cambridge HTTPS PDF 返回 `SSL: UNEXPECTED_EOF_WHILE_READING`，公开 HTTP PDF 和 PS.Z 均为 `RemoteDisconnected`。报告正文仍未取得，不能升级为公式或实验核验。
- **Kool 2019**：ICLR workshop 官方录用列表和作者页均链接同一 Google Drive 文件，web 访问返回 404；这不能判断文件已删除还是权限不可用。UvA 作者记录只链接 OpenReview，本轮原 PDF 仍为 HTTP 403。作者 GitHub 完整文件树只有对应海报和网页，没有此原论文 PDF；未以海报、课件或别的论文替代原文。

逐项来源、实际错误和版本身份保存在 [本轮访问记录](../papers/rl/missing_original_access_2026-10-28c.json)。全部使用正常公开访问和默认 TLS 校验，未绕过登录、验证码、证书或访问限制。没有改动论文或 panel。

## 本轮：安全、风险、鲁棒性与部分可观测基础（2026-10-28）

新增 **11 篇论文及 11 个概念**，先按标题、来源编号和现有条目核查 `rl.json`、`pretrain.json`、`interp.json`，没有重复收录。本轮是沿经典综述及原论文参考链的定向补缺，不能称这些分支已经穷尽。

| 分支 | 本轮原文 | 已核对的位置 |
|---|---|---|
| 安全 RL 分类入口 | García–Fernández 2015, *A Comprehensive Survey on Safe Reinforcement Learning* | PDF 第 1、5–7、9–10 页，目标修改、探索修改与模型不确定性分类 |
| 命中风险与约束控制 | Geibel–Wysotzki 2005 | PDF 第 3、6–9 页，Definition 4.1、Proposition 4.1、§4.2–5.2 |
| 参数鲁棒 MDP | Nilim–El Ghaoui 2005 | PDF 第 2–4 页，矩形不确定性、固定与时变模型区别、Theorem 1 |
| 分布鲁棒 MDP | Xu–Mannor 2010 | PDF 第 2–4 页，按状态独立分布、嵌套集合、Theorems 1–2 |
| 静态 CVaR 规划 | Chow 等 2015，arXiv v1 | PDF 第 3–8 页，风险定义、乘性扰动预算、扩充状态与插值 |
| 相干风险梯度 | Tamar 等 2015，arXiv v2 | PDF 第 2–5 页，静态/动态风险、Assumptions 2.2/4.1、Theorem 4.2 |
| Lyapunov 约束策略 | Chow 等 2018，arXiv v1 | PDF 第 3–6 页，暂态 CMDP、可行 baseline、Theorems 1–2、Proposition 1、函数逼近区别 |
| RCPO | Tessler 等，arXiv v3 / ICLR 2019 | PDF 第 3–5 页，原约束与指导惩罚、三时间尺度、Theorem 2 |
| POMDP 系统整理 | Kaelbling–Littman–Cassandra 1998 | PDF 第 7–10 页 §3.1–3.4；第 34–36 页参考文献 |
| POMDP 反应式策略改进 | Jaakkola–Singh–Jordan，NIPS 1994 / 论文集 1995 | PDF 第 2–5 页，平均回报、Monte Carlo 评价、局部改进 Theorem 1 |
| 感知混叠下的策略边界 | Singh–Jaakkola–Jordan，ICML 1994 | 作者 PS 转换 PDF 第 2–4 页，Facts 1–4、Figure 1、遍历性及观测级 TD；第 3 页另做图像目检 |

**采用的参考链。** [García–Fernández 的 JMLR 原文](https://www.jmlr.org/papers/v16/garcia15a.html)按优化准则及探索过程分支提供检索入口，其中 §3 与 §3.1.2 指向 Geibel–Wysotzki、Nilim–El Ghaoui 及早期风险敏感/鲁棒控制。Xu–Mannor、CVaR 和相干风险梯度补充这些问题的分布与时间结构；Lyapunov 与 RCPO 是后续 CMDP 方法，不冒称被 2015 综述引用。已有 Kaelbling 等 1996 综述的 PDF 第 31–33 页 §7，尤其第 32 页对 Jaakkola 等局部随机策略优化的引用，连接早期 POMDP 学习；[1998 年作者原文](https://people.csail.mit.edu/lpk/papers/aij98-pomdp.pdf)的 [55]–[59] 则连接无记忆策略和 Smallwood/Sondik 源头。其 [55] 使用的标题与实际 ICML 作者稿首页不同，本库以实际原稿题名和作者为准，并记录对应页码 284–292。

**Scope 的关键区别。** Geibel 的风险是错误状态命中概率，论文的学习部分明确是受限策略类内的启发式方法；低方差不等于低失败概率。Lyapunov 的可行性约束是初始状态上的期望累计成本，基线、准确算子及局部约束都重要；每次策略更新可行不代表每条随机轨迹无违规。RCPO 的渐近结论要求指导惩罚的所有局部极小点满足原约束，不能省略这个假设后说“任意奖励惩罚都能安全收敛”。

Nilim–El Ghaoui 的有限时域结果针对逐时可变且按状态行/动作可分的不确定性；固定但未知模型可有不同值。Xu–Mannor 还要求参数分布按状态独立、嵌套集合等条件。静态总成本 CVaR、动态嵌套风险、模型参数不确定性和轨迹随机性各自不同；本轮没有把它们合并为一个无条件“安全”保证。POMDP 的正确 belief 具有充分性，并不意味着单个观测、RNN 隐状态或任意窗口已恢复真实状态；局部无记忆随机策略优化也不等于完整 POMDP 最优控制。Singh 等反例中的任意大绝对回报差依靠放大奖励尺度，不能推广成固定奖励界和折扣下仍无界。

**版本与核验。** 10 份 PDF 直接来自期刊、会议、作者机构或 arXiv；Singh 1994 来自 [作者 MIT 页面所列原始 PS](https://people.csail.mit.edu/tommi/papers/ML94.ps)，保留 220,718-byte 原文件及 SHA，使用 Ghostscript `-dSAFER` 生成 9 页 PDF，单列原文件和派生 PDF 的 SHA。Poppler 渲染该旧字体文件时有 Type 3 bounding-box 警告，第 3 页图像可读，已核对负号、反例文字和图示；未伪称这是出版社 PDF。Kaelbling 1998 的 Cassandra 作者站返回 406 后改取 MIT 共同作者站；Singh 的 Michigan 作者页遇 TLS 证书链错误后改取 MIT 共同作者的正常公开 PS，没有跳过证书校验。全部新 PDF 核验 `%PDF-`、`pdfinfo` 页数、字节数、SHA、首页题名/作者和可见版本；上述段落为定向阅读，完整证明和全部实验并未审计，也未独立复现。

**仍未覆盖。** Altman 的 *Constrained Markov Decision Processes*、Howard–Matheson 的指数效用风险敏感 MDP、Iyengar 的 *Robust Dynamic Programming*、Smallwood–Sondik 1973 和 Sondik 1978 原论文仍未在本轮取得并审读；它们已在已读参考链中出现，不能以综述转述代替原文核查。非矩形鲁棒性、严格逐轨迹/几乎必然安全、shielding 与控制屏障函数、现代 POMDP 样本复杂度也未系统覆盖。原有 Kool、Rummery–Niranjan、R2D2 最终版三项缺口本轮未重复尝试受限入口。

本轮后 RL 清单为 **224 篇论文、213 个概念；221 份主文 PDF 可读、3 个主 PDF 缺口**，另保留 R2D2 审稿稿与 Kool 海报两份明确标记的补充 PDF。新增文件都在 `.gitignore` 覆盖的 `papers/` 内，最大新增 PDF 约 1.6 MB；未编辑 panel、冻结论文或共享总表。

## 本轮：模型式学习、线性控制与探索可学性的条件（2026-10-28）

新增 **11 篇论文、13 个概念**：11 个概念对应新论文，另从已有原文补 TD3 与 Dreamer 方法概念。同时为既有的 `PPO clipping` 和 `Importance ratio clipping` 补充 PPO 原论文来源；没有把它们再增加成新概念。当前为 **235 篇论文、226 个概念；232 份主 PDF 已取、3 个主 PDF 缺口**。本轮优先补基础理论链，不重复已有 Dyna 1990、E³、R-max、UCRL2、UCBVI、PILCO、PETS、MBPO 等条目。

| 本轮原文 | 核对位置 | 所列主张的关键条件 |
|---|---|---|
| Recht, *A Tour of Reinforcement Learning*，2018 v2 / 2019 综述 | PDF 第 13–16、24–28 页 | 控制问题与参考链的组织，Knowledge；示例比较不是普遍优劣定律 |
| Dean 等，*On the Sample Complexity of the LQR*，arXiv v3 | 第 3–5 页，Propositions 1.1–1.2 | 可重置为零的独立轨迹、全状态、高斯输入/扰动、有效模型误差界、robust SLS |
| Fazel 等，LQR policy gradient，ICML 2018 | 第 2–7 页，Lemma 3、Theorems 7/9 | 稳定初始策略、非退化初始状态二阶矩、规定步长/oracle；模拟版另要求有界初始状态 |
| Abbasi-Yadkori–Szepesvári，COLT 2011 | 第 3–8 页，A1–A4、Theorems 1–2 | 在线 LQ；给定参数集合、已知界、次高斯 martingale 噪声和 A4 的统一闭环收缩/增益条件 |
| Simchowitz 等，*Learning Without Mixing*，arXiv v4 | 第 2–4 页，Theorem 2.1 | 无输入、全状态、单轨迹、各向同性高斯噪声、谱半径不超过 1 与 Gramian 条件 |
| Mania–Tu–Recht，certainty equivalence，NeurIPS 2019 | 第 2–4 页，Assumptions 1–2、Theorems 1–2 | 足够小模型误差、稳定最优闭环与 Riccati 扰动控制；所核对特化使用定量可控性 |
| Jin 等，linear MDP / LSVI-UCB，完整 arXiv v2 | 第 3–8 页，Assumptions A/B、Theorems 3.1–3.2 | 已知特征同时线性表示转移核和奖励；在线 episodic 访问；失配用统一转移 TV/奖励误差 |
| Yang–Wang，linearly additive features，ICML 2019 | 第 2–6 页，Definition 1、Assumptions 1–2、Theorems 2–3 | 任意状态动作生成模型访问；最优样本速率另要求 anchor 凸组合条件；奖励不必在同一转移特征内 |
| Du 等，representation sufficiency lower bounds，ICLR 2020 / arXiv v4 | 第 3–8 页，§3.3、Theorems 4.1–4.2 | 存在性最坏情况；已知转移但未知奖励亦可困难；近似价值/逐坐标转移误差不同于 TV |
| Jiang 等，Bellman rank / OLIVE，ICML 2017 | 第 3–7 页，Definitions 2–5、Theorem 1 | 环境与函数类共同低秩、存在有效函数、有限动作；基线为类内最优有效函数，计算效率另论 |
| Sun 等，witness rank，COLT 2019 | 第 2–10 页，Definition 1、Theorems 2/6、Assumptions 1–2 | 模型 realizability、Bellman domination、统计复杂度；分离结论限定 G-profile 信息接口 |

**采用的一手参考链。** [Recht 综述](https://arxiv.org/abs/1806.09460)的 §4.1.1 和 references [27]/[70] 直接连接 Dean 的 Coarse-ID 与 Simchowitz 的系统辨识，[2] 指向 Abbasi-Yadkori–Szepesvári。[Fazel 原文](https://proceedings.mlr.press/v80/fazel18a.html) §1.3 进一步把这三条模型式路线与直接策略优化相对照；[Mania 原文](https://papers.neurips.cc/paper_files/paper/2019/hash/5dbc8390f17e019d300d5a162c3ce3bc-Abstract.html) §2.1 明确比较 Dean 的鲁棒合成，后者是时间更晚的后续分析，不冒称被 2018 综述引用。离散探索链以 [Jin 的完整原文](https://arxiv.org/abs/1907.05388) §1.2 对 Yang–Wang、Jiang 的比较及 [Du 原文](https://arxiv.org/abs/1910.03016) §2/§4 对 Jin/Yang、Bellman rank、witness rank 的整理为入口，再返回各原论文核对假设。Dyna 源头已在库，本轮不重复计数。

**边界核对。** 多条独立可重置实验、单条被动观测轨迹、在线自适应交互和任意状态动作查询是四种不同信息条件。Dean 的控制性能界不能直接解释成采集过程中每一步安全；Simchowitz 的辨识可激发性也不能解释成不稳定系统更好控制。Fazel 的“全局”从稳定初始策略出发，并有非退化性及梯度/模拟访问条件。Mania 的二次成本界属于模型误差足够小的区域，且阈值和常数依赖系统；不能只比较 O(ε) 与 O(ε²) 上界就宣称任意粗模型下 certainty equivalence 都更好。

Jin 的 linear MDP 不是只要求最优 Q 函数线性；Yang–Wang 的 near-optimal sample complexity 还需 generator 与 anchors。Du 的下界是存在难实例而对所有算法成立，并不是每个表示、每个任务都失败；逐坐标转移误差小也不等于分布总变差小。OLIVE 的一般比较基线是 `V*_F`，额外 realizability 条件才把它接到环境最优值；Bellman rank 依赖所用函数类。Sun 的 model-free 下界按原文 Definition 1 限定为只能通过 `G-profile` 访问上下文，并取 `G=OP(M)`；更丰富表示可能解除信息损失，不能把该定理扩大成全部模型式方法胜过全部无模型方法。Bellman/witness 的样本界也不解决通用规划或函数枚举的计算成本。

**已有概念来源修补。** 两个 clipping 条目均新增 `primary_source=https://arxiv.org/abs/1707.06347` 与原文 v2 PDF 第 3 页 §3、式 (7)、Figure 1 的 `primary_evidence`；保留 Implementation Matters 的来源和实现边界证据，并在 `sources` 列表列出两篇原文供既有工具关联。原式是取未截断/截断代理目标的较小值，不能把它写成概率比或 KL 的硬约束。TD3 从已有 v3 PDF 第 4–6 页核对双 critic 最小值、延迟更新和目标平滑；Dreamer 从已有 v3 PDF 第 3–4 页核对潜在想象轨迹、重参数梯度与 λ-return。二者新增概念均为 Knowledge，不从算法定义推出普遍收敛，也不把学得隐状态直接视为真实环境充分统计量。

**文件、版本及审阅强度。** 11 份原文均核对 `%PDF-`、`pdfinfo` 页数、字节数、SHA-256、首页题名/作者和可见版本。会议官方 PDF 与完整 arXiv 稿分开标记；Jin 的公开发表索引是 COLT 2020 extended abstract，而实际存档/阅读的是 28 页 arXiv v2。Fazel、Yang、Jiang 存档是会议正文，未假装其补充证明已完整审计。上述 Science 标识对应条件化定理主张类型及待进一步审计的候选，不能当作独立验证了完整证明；也未独立复现实验。三库按标准化标题、arXiv 编号及已有 SHA 复核，无新增重复；新增概念名不重复，TD3/Dreamer 也检查了既有同义方法概念。最大新增 PDF 1,243,396 bytes，全部位于已忽略的 `papers/`。

**尚未覆盖。** 经典非线性自适应控制、dual control、未知观测模型/部分观测 LQG 的完整界、系统辨识到安全采集的设计、精确 Q* realizability 与 Bellman completeness 的更细分界、现代 misspecified linear bandit/RL 的紧阈值均未系统完成；当前理论链也不是全部 model-based RL 的穷尽分类。三项原有受限主 PDF 未重复访问，仍独立记录缺口。未编辑 panel、冻结论文或共享生成物。

## 本轮：已收经典论文的概念反向关联修补（2026-10-28）

本轮以 `concept_coverage_gaps.md` 当时的 **32 个 RL 待关联条目**为入口，处理其中 **20 个**：**修补 9 个已有概念，新增 4 个概念，新增论文 0 篇**。当前为 **235 篇论文、230 个概念、232 份主 PDF、3 个主 PDF 缺口**。新增关联使用规范 `sources` 列表，保留原来源，逐来源补方法位置与适用范围；没有为每篇论文机械生成一个概念。关联通过仅表示已有明确来源连接，不等于该论文所有贡献已覆盖或完整证明已核验。

| 概念处理 | 本轮解除的待关联论文 | 本次原文定位 |
|---|---|---|
| 修补 Experience replay | DQN 2013；DQN 2015；Rainbow | 2013 v1 第 4–5 页 §4/Algorithm 1；2015 Nature 第 6–7 页 Methods/Algorithm 1；Rainbow v1 第 2–4 页组件与 Integrated Agent |
| 修补 Actor and critic | A3C / Asynchronous Methods | v2 第 3–4 页 §4、n-step advantage 和熵正则；保留 Konda–Tsitsiklis 来源 |
| 修补 Reward model | Learning to summarize from human feedback；InstructGPT；Helpful and Harmless Assistant；RLAIF vs. RLHF；N+ Implementation Details | 摘要 v3 第 3–5 页；InstructGPT v1 第 6、8 页；HH v1 第 4–5 页；RLAIF v3 第 3–4 页；N+ v1 第 3–4 页 |
| 修补 Model-based vs model-free | Sutton 1990 Dyna | 第 2–5 页 §2–5、Figures 1–2、Dyna-Q 式 (3)–(4)；保留 MuZero 来源 |
| 修补 Offline RL distribution shift | D4RL | 第 3–4 页 §3–4、Figure 2；保留 CQL 来源 |
| 修补 CTDE | MADDPG / Multi-Agent Actor-Critic | 第 4–5 页 §4.1–4.2、Figure 1、式 (4)–(8)；保留 QMIX 来源 |
| 修补种子与任务不确定性 | Deep Reinforcement Learning that Matters | v3 第 2–5 页 Experimental Analysis、Random Seeds and Trials、Tables 1–3、Figure 5；保留 RLiable 来源 |
| 修补 RLOO | Back to Basics / REINFORCE Style Optimization | v2 第 5–6 页 §2.2–2.3、式 (6)–(8)、RLOO 的 i.i.d. 多样本公式；Kool 原主 PDF 仍缺 |
| 扩展已有 TD3 概念为“TD3 与离线 TD3+BC” | A Minimalist Approach to Offline RL | 第 3–4 页问题/实现讨论及第 6 页 §5、式 (3)–(5)；保留 TD3 原论文 |
| 新增 Dueling 双流与聚合 | Dueling Network Architectures | v3 第 4–5 页 §3、式 (7)–(9)；另关联 Rainbow 的逐 atom 聚合，Rainbow 不重复计数 |
| 新增确定性策略梯度与 DDPG | Deterministic Policy Gradient Algorithms；Continuous control with deep RL | DPG 第 3–5 页 §3.1–4.3、Theorem 1 与式 (15)；DDPG v6 第 3–4 页 §3 |
| 新增 IQL expectile 与优势加权克隆 | Offline RL with Implicit Q-Learning | v1 第 3–6 页 §4.1–4.4、式 (5)–(7)、Algorithm 1、Lemma 1、Theorem 3 |
| 新增 DPO 参考策略对数比目标 | Direct Preference Optimization | v3 第 4–6 页 §4–5.1、式 (4)–(7)、Definition 1、Theorem 1 陈述 |

**修补中保留的范围区别。** DQN 2013 与 2015 的游戏集合和训练设计不同；2015 的周期复制目标网络不能倒写成 2013 原始 Algorithm 1 的同一实现。Rainbow 的六组件组合和以分布 KL 损失设优先级也不等于普通均匀回放。Dueling 去均值聚合后两流有相对原始 V/A 定义的常数偏移，不可直接当作真实状态价值/优势分解。A3C 的神经异步更新不能继承经典线性 critic 的全部收敛条件。

DPG 的精确 Theorem 1 与行为状态分布下的近似式 (15) 分开；原文明确丢弃一项，近似 critic 也未必给出真实上升方向。DDPG 的回放和目标网络是实际方法，不能仅凭公式升为 Science。IQL 的 expectile 是对数据动作 Q 值的条件统计，随机转移仍通过均方目标取期望；其 τ→1 结论仅比较数据动作支持内最优值，不给有限数据神经实现全局最优保证。TD3+BC 克隆项是软正则，不是硬支持约束。

偏好奖励的共同概念保留各来源差异：早期轨迹片段、TL;DR 摘要、API 指令、HH 两种标注方向、AI 软标签及 TL;DR 复现分别有自己的数据/标注者/模型范围。Canonical RLAIF 训练独立 RM，而 direct-RLAIF 直接让 LLM 评分；没有把后二者混成同一管线。DPO 的理想重参数化与偏好分布拟合不保证恢复唯一效用或真实对齐。MADDPG 的合作/竞争 critic 与 QMIX 的合作单调分解不能无条件互换。Henderson 的有限种子敏感性观察不推出所有 RL 不可复现，也不给通用最低种子数。

**核验。** 对已解除的 20 个队列项逐一验证本地 PDF 签名、页数、字节数和 SHA 与既有登记一致；新增概念的名称及定义已检索同义项，无新增重名。4 个新概念全部按方法/表示定义列为 Knowledge，理论陈述仅保留明确条件和未全面审计标识；没有把实验自动升为 Law。9 个旧概念均保留旧来源并补新来源，原论文记录及论文总数未变。未新增下载文件，未访问三个旧受限入口，未编辑反向队列、panel、冻结论文或统一生成物。

**剩余 12 个队列项。** DAPO；Understanding R1-Zero-Like Training；REINFORCE++；DreamerV2；DreamerV3；A Closer Look at Deep Policy Gradients；Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms；On the Theory of Policy Gradient Methods；Agent57；D4PG；ACER；UNREAL。本轮停在上述授权的 20 项修补，保留剩余工作供下一轮。

## 续轮：完成剩余 12 项关联并核对 TR166 远程原文（2026-10-28）

上述 **12 项均已建立原文定位的概念关联**，没有以缺乏独立概念为由机械造 12 个条目。队列部分修补 **8 个已有概念、新增 3 个独立概念**；另按根线程提供的新来源，修补已有 **Modified Connectionist Q-Learning** 的原文证据。因此本次合计 **9 个已有概念修补、3 个新概念、0 篇新论文**。当前 **235 篇论文、233 个概念、232 份本地主 PDF、3 个本地主 PDF 缺口**。

| 队列论文 | 概念处理 | 实际核查位置 |
|---|---|---|
| DAPO | 复用 Group relative policy optimization | v2 PDF 第 4–7 页 §3.1–3.4、式 (8)–(13)、Algorithm 1 |
| Understanding R1-Zero-Like Training | 复用 GRPO 与 Group advantage normalization | v2 第 6–8 页 §3.1–3.3、Figure 4、Listing 1、Figure 5 |
| REINFORCE++ | 复用 Group advantage normalization，显式对照全 batch 统计 | v9 第 2–4 页 §2.1–3.4、式 (4)–(8)、Algorithm 1 |
| DreamerV2 | 扩展 Dreamer 潜在想象中的行为学习 | v4 第 3–6 页 §2.1–2.2、Algorithms 1–2、式 (1)–(6) |
| DreamerV3 | 同一 Dreamer 概念，保留版本差异 | v2 第 3–6 页 World model/Critic/Actor learning、式 (1)–(7) |
| A Closer Look at Deep Policy Gradients | 复用 Gradient estimator variance | v4 第 3–6 页 §2.1–2.3、Figures 2–5 |
| Direct Alignment Reward Overoptimization | 复用 Reward model overoptimization | v2 第 4–6 页 §3.1–3.3、式 (5)、Figures 1–3 |
| On the Theory of Policy Gradient Methods | 复用 Natural policy gradient；新增条件性全局收敛/分布失配概念 | v5 第 9 页问题设定，第 13–15、18、22–23、28–30、33–34 页：Lemma 4.1、Theorems 4.1/5.1/5.3/6.1/6.2、Corollary 6.1 及对应假设 |
| Agent57 | 新增策略族的自适应探索分配 | 第 3–5 页 §2–3.2、§4 的 episode bandit 与评价口径 |
| D4PG | 复用 Distributional RL | 第 3–5 页 §3、式 (5)–(8)、Algorithm 1 |
| ACER | 复用 Off-policy importance sampling | 第 3–5 页 §3–3.3、式 (3)–(12) |
| UNREAL | 新增辅助控制/奖励预测共享表征 | 第 2 页 Figure 1、第 3 页辅助策略用途，第 4–6 页 §3.1–3.4、式 (1)–(2) |

**方法与经验边界。** DAPO 的组内标准差、动态过滤与总 token 分母，不等同 Dr.GRPO 去标准差和固定分母；REINFORCE++ 的全 batch 统计更不能写成组内 GRPO，基础 k=1 与带 Baseline 版本也各有 KL/基线设计。有限 batch 无偏性、通用稳定性和全部附录证明均未据方法公式宣称成立，三个配方维持 Knowledge。

Dreamer 的共同概念只覆盖潜在想象框架：V1 使用动力学重参数梯度，V2 在 Atari 用 REINFORCE、在连续控制用动力学反传，V3 对两类动作均用 REINFORCE。模型内 Markov 状态和改善不证明真实环境充分状态或改善。Closer Look 的千万状态动作对参考梯度仍是估计，余弦一致性也不是协方差本身。D4PG 的回报分布 critic 与分布式 actor 分开，优化期望回报不自动变成风险敏感控制。ACER 的精确截断修正恒等式只对应其行为状态分布下的边际梯度对象，实际神经 Q/Retrace 近似不保证真实目标无偏；本次未覆盖其连续动作 §5。

**过优化的来源差异。** Gao 的合成 gold 与直接对齐论文的 GPT-4 胜率是两种评价；后者只将已核查的 TL;DR/Pythia/DPO/IPO/SLiC 网格纳入范围，未把附录 Gemma/HH 当成已读证据。原文承认在 `d=sqrt(KL)` 上二次拟合误差相近，所以保留竞争拟合解释，不把曲线唯一性或过优化必然性写成定律。原有 Law 分类仅保留为所列经验域的关系候选。

**PG 理论分层。** 表格概率投影法依赖分布覆盖系数；表格 softmax 的 Theorem 5.1 是严格正 μ、规定步长下的渐近价值收敛；精确表格 NPG 的 Theorem 5.3 使用均匀初始化与同一初始/评价分布，所得迭代界不含分布失配系数，不等于有限样本无需探索。函数逼近的 Q-NPG/NPG 另要求特征或 log-policy 光滑性、有界方向、统计误差、向比较策略分布的迁移误差与相对协方差条件数，结论是对指定比较策略的期望最佳迭代界。将 on-policy 拟合误差换成迁移误差需额外控制分布失配。Science 仅归于这些明确条件的定理主张，完整证明未独立审计；旧 Policy gradient theorem 梯度恒等式概念未混入全局收敛论断。

**TR166 远程原文进展。** [Niranjan 上传的原 PDF](https://www.researchgate.net/profile/Mahesan-Niranjan/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems/links/5438d5db0cf204cab1d6db0f/On-Line-Q-Learning-Using-Connectionist-Systems.pdf)经 web PDF text 可读 21 页，地址保存在 `remote_pdf_source`。核对首页身份，以及物理第 7–9 页/印刷第 6–8 页 §3.2 式 (9)、脚注 3、§3.3、§4 Figure 1。paper 与 MCQ-L 概念从摘要初筛升级为原文指定章节核对，仍为 Knowledge；一般神经收敛未获证明。标准本地 GET 仍 403，无本地字节或哈希，原 `pdf_status`、旧失败证据和三个主 PDF 缺口均保留。根线程另存本次远程访问记录；本分工未重试下载。

**核验与未覆盖。** 12 份既有 PDF 均重新验证签名、字节数、页数与 SHA-256 和索引一致；关联源以规范 `sources` 列表保存并保留旧 URL，三个新概念已检索 RL/pretrain/interp 名称和近义项，无新增重复。另修正 D4PG paper 的“分布式价值”歧义及 PG paper 的精确参数化/步长/历史开放问题口径，共修改 3 个既有 paper 记录（含 TR166），论文总数不变。本队列未保留未关联条目；关联完成不表示所有贡献、完整证明和实验均已覆盖。未新增论文或二进制，未修改 panel、冻结稿、反向队列或构建器。

## 续轮：平均回报与早期控制／棋局学习源头（2026-10-28）

本轮新增 **9 篇论文、9 个概念**，修补已有 **Actor and critic** 的 1983 原始来源。当前 **244 篇论文、242 个概念、239 份本地主 PDF、5 个主 PDF 缺口**。7 份新增本地主 PDF 中，6 份是作者／机构公开的论文 PDF，1 份由作者原始 PostScript 转换；2 个新增缺口已有可读原文或许可转录，仍不计为 PDF 已取得。旧有 Rummery TR166、R2D2 和 Kool 三个访问缺口未重复尝试。

**参考链。** 从 Sutton–Barto 第二版物理第 278–280 页（印刷 256–258）§10.3 的书目说明，追到 Schwartz 1993、Mahadevan 1996、Tsitsiklis–Van Roy 1999；继续核对平均代价 Q-learning 的 Abounadi–Bertsekas–Borkar 作者公开原文，以及平均回报策略梯度的 Marbach–Tsitsiklis、Baxter–Bartlett。后两篇的原始参考文献连接已有 policy-gradient 和 actor–critic 理论。教材物理第 37–44 页 §1.7 历史回顾明确连接 Barto–Sutton–Anderson 1983、Samuel 1959 和 TD-Gammon；Barto §IV 与 Tesauro 原文参考 [10] 再连接 Samuel。教材与综述只提供线索，分类证据定位回各篇原文。

| 新增原论文 | 原文与核查位置 | 概念／主张边界 |
|---|---|---|
| Schwartz 1993，ICML R-learning | 作者上传远程 8 页；第 2–5 页 §3–5、式 (9)–(10) | 平均回报／差分价值与 R-learning；K。明确无一般收敛证明 |
| Mahadevan 1996，Average Reward RL | 作者 PS 转 PDF 38 页；第 4–10 页 §2.1–2.2、第 19–21 页 §2.7–3.2 | 最优性层级、链结构与 R-learning 反例；K，不整篇升为 S |
| Tsitsiklis–Van Roy 1999，Average Cost TD | 作者站期刊 PDF 10 页；第 2–3 页 Assumptions 1–3、Theorem 1 | 固定不可约非周期链、线性特征下 TD 收敛；条件 S |
| Abounadi–Bertsekas–Borkar 2001 | 作者站期刊 PDF 18 页；第 2–6、9–11、13–14 页 | RVI 和 SSP 双时间尺度分开建概念；各自条件 S |
| Marbach–Tsitsiklis 2001 | 作者站期刊 PDF 19 页；第 2–5、7–8 页 Propositions 1–4 | 再生周期与逐步更新条件不同；梯度趋零不等于全局最优 |
| Baxter–Bartlett 2001，GPOMDP | JAIR 镜像期刊 PDF 32 页；第 8–11、17–20 页 | 固定策略／固定 β 的估计极限与 β→1 偏差极限分开 |
| Barto–Sutton–Anderson 1983 | 作者站原期刊扫描 13 页；第 5–9 页 §V–VIII、式 (1)–(7) | 修补 Actor and critic；已知状态与 162 格编码，方法／实验 K |
| Tesauro 1995，CACM TD-Gammon | ACM 许可 HTML 转录；Learning Methodology、Complexity 两节 | 新增神经 TD 自博弈概念；后期手工特征、搜索与 doubling cube 分开；K |
| Samuel 1959，Checkers | Berkeley 课程原期刊扫描 21 页；第 2、6、9–10 页 | 新增棋局记忆／评估函数自举；相关统计不是现代 TD 梯度公式；K |

**本轮最关键的范围校正。** 原始 R-learning 的“收敛即 T-optimal”不能作为一般保证：Mahadevan §2.7.1 的三状态 unichain 固定点例子无法区分 gain 相同而 bias 不同的动作。这不等于否定所有平均回报算法的 gain 收敛，也不能把 Abounadi 的 RVI/SSP 定理倒写成原始 R-learning 的证明。平均回报、差分价值、bias 与 Blackwell 最优性分别建边界；communicating、所有策略 unichain、共同再生状态、固定链不可约且非周期不能混用。

平均代价 TD 的定理只给固定链的线性投影解；RVI 要保留归一化和有界性证明条件；SSP 的慢／快步长比、投影与参考状态不可省略。Marbach 的结论是性能梯度趋零，逐步版额外要求统一回访。GPOMDP 的 Theorem 5 是固定 θ、β 的估计一致性，而 Theorem 2 是随后 β→1 的偏差极限；二者不自动推出联动训练、全历史 POMDP 最优性或任意神经策略全局收敛。5 个新 Science 概念都限定在这些条件定理，不称完整证明已验证。其余 4 个新概念与早期方法实验均为 Knowledge；没有由多任务结果自动升 Law。

**版本与访问缺口。** Mahadevan 原作者稿为 `mlj-rl-si-paper.ps.Z`，保留原件（248,207 bytes）及 SHA；用 `uncompress` 和 Ghostscript 转换，得到 38 页可检索 PDF。稿头写 159–196，与出版社 159–195 的页码元数据不同，已显式记为作者稿而非假称出版社最终版。Samuel 21 页包含开头棋盘插图，物理第 2 页才是印刷第 210 页；Barto 是 1983 原期刊扫描，不是 1988 文集重印。Baxter 使用 2001 JAIR 论文，不将后来的 arXiv 上传年当发表年。

Schwartz 的 [作者上传原文](https://www.researchgate.net/profile/Anton-Schwartz/publication/221346025_A_Reinforcement_Learning_Method_for_Maximizing_Undiscounted_Rewards/links/5e72421aa6fdcc37caf4cf4b/A-Reinforcement-Learning-Method-for-Maximizing-Undiscounted-Rewards.pdf)可经 web 阅读 8 页，标准本地 GET 为 403；保留远程页数和定位，没有本地哈希。Tesauro 的 CMU PDF 404，UW 课程入口转到 Shibboleth，CACM 正文入口返回 403；停止受限入口，保存 [bkgm 的许可原文转录](https://www.bkgm.com/articles/tesauro/tdl.html)（56,176 bytes），`html_kind: licensed_transcription`、`source_type`、转录 SHA 和原 PDF 缺口分别登记。它不是原生网页或出版社 PDF，离线文件未包含全部链接图片与原版分页。同名一页 ICGA 通知和 Schwartz 同年不同题名 CogSci 文均未用作替代。

**核验与未覆盖。** 新增 7 份 PDF 已核对标题、作者、版次／期刊页、PDF 签名、页数、字节数与 SHA-256；扫描稿用逐页渲染/OCR，关键理论页另作目检。三库按标准化标题、现有 DOI/arXiv 及 SHA 去重通过，9 个新论文均有规范 `sources` 概念关联；概念名无重复。完整证明和实验均未独立复现。最大新增 PDF 13,014,024 bytes，均放在已忽略的 `papers/`，没有引入超过 50 MB 的文件。访问与材料核验详见 `papers/rl/average_reward_classics_access_2026-10-28.json`。

经典 Blackwell/Howard/Veinott 原证明、Jalali–Ferguson 自适应平均代价控制、multichain 状态相关 gain、一般非线性平均回报逼近与统一有限样本理论仍未系统覆盖。根线程两状态例子已独立核算：均值 A=0.2、B=0.5，折扣偏好在 γ=5/8 反转，两策略均不可约非周期；这只展示固定折扣与长期目标的差异，不否定接近 1 的 Blackwell 稳定性。本轮未修改 panel、冻结论文、共享构建器或总表，至此交还 RL 索引写权限。


## 2026-10-28：POMDP 经典规划、belief 压缩与更正版恢复

本轮从中断后保留的原 PDF 恢复入库，新增 **11 篇论文、11 个概念**，并补充既有「POMDP belief state 与正确模型的充分统计量」的 Monahan 来源。RL 索引从 244/242 增至 **255 篇／253 概念**。其中 8 篇已有本地主 PDF，3 篇只有出版社身份与摘要；3 个缺文的 Science 标签均明确标为**摘要级候选、条件待核**，只识别理论主张类型，不宣称证明验证。

| 原论文 | 本轮实际原文证据 | 分类与范围 |
|---|---|---|
| Smallwood–Sondik 1973，有限时域 | INFORMS 原摘要、题名、作者、卷期页 | S 候选；有限时域 PWLC 主张，完整假设与原证明待核；主 PDF 缺失 |
| Sondik 1978，折扣无限时域 | INFORMS 原摘要、书目信息 | S 候选；有限暂态策略／凹包改进；不声称一般无限时域有限 α 精确表示；主 PDF 缺失 |
| Lovejoy 1991，Computationally Feasible Bounds | INFORMS 原摘要、书目信息 | S 候选；网格上下界及策略损失；不与同年算法综述混同；主 PDF 缺失 |
| Monahan 1982 | 带封面的 17 页期刊扫描；物理 pp.1–5，重点 pp.4–5／期刊 pp.3–4 | 综述 K；正确有限 POMDP 的 belief 更新／充分性条件关系 S，原始引用证明未追审 |
| Cassandra–Littman–Zhang 1997，Incremental Pruning | 8 页 UAI 扫描；pp.1–4，§2–4、式 (6)–(10)、Figures 1–3 | 精确交叉和／purge 恒等关系 S，方法效率 K |
| Pineau–Gordon–Thrun 2003，PBVI | 作者 6 页原稿；pp.1–4，p.3 图像目检 | 覆盖稠密度误差界 S；随机 belief 扩展配方 K |
| Smith–Simmons 2004，HSVI | 8 页作者所存 UAI 正式排版；pp.1–5 | 指定初始化与搜索下 regret／终止条件 S；实现 K；详证所引技术报告未审 |
| Kurniawati–Hsu–Lee 2008，SARSOP | RSS 正式 8 页；pp.1–4，§II–III-C | 算法 K；小最优可达域与已知 proper cover 的区别为所引条件 S，原复杂度证明未审 |
| Roy–Gordon–Thrun 2005 | 本地 JAIR 40 页；p.1、pp.4–9、11–13、19–21 | E-PCA 压缩与近似规划 K；重构精度、值误差、策略质量分开 |
| Poupart–Boutilier 2002 | NeurIPS 正式 8 页；pp.1–5，p.4 图像目检 | 奖励包含与共同不变子空间的线性充分条件 S；有损搜索 K |
| Smith–Simmons 2005 | 作者更正版 8 页 pp.1–5，另读 1 页作者勘误；旧 arXiv 单列补充 | ρ^(−p) 范数下条件界 S；HSVI2 实现 K；采用 2007-02-14 修正版 |

**关键 Scope。** PBVI 的覆盖量是指定 belief 域到样本集的最大 L1 距离，不能换成平均重构误差；原文明确固定 belief 集时，迭代次数增加不必收敛。HSVI 的 ε 是指定初始 belief 的 regret，其他 belief 需看各自 gap；有效、uniformly improvable 的界和规定探索规则不能省略。SARSOP 引用的困难性结果说明，仅有小最优可达集合仍不保证易求解；给定 proper cover 是额外输入，不是启发式样本集自动具备的性质。

**压缩不是同一种保证。** Roy 的 E-PCA 用指数 link 与未归一化 KL，正值不等于归一化，非线性压缩后原 α-vector 的凸值结构不保留；文中实验 γ=1，不直接套用折扣收缩结论。Poupart 的线性充分条件保留奖励相关不变空间，但该空间可能全维，策略执行语义也须保留；本轮不宣称已经审核其非线性一般化、完整有损界或实验。

**作者勘误已闭合。** [Smith 的作者页面](https://longhorizon.org/trey/pubs/b2hd-smith05hsvi.html)明确标注更正版及 [errata](https://longhorizon.org/trey/papers/smith05_point_based_pomdp_errata.pdf)。式 (5) 及加权范数的 ρ^p 改为 ρ^(−p)；原 arXiv 上传仍保留旧符号，只作版本溯源。此处不再登记为“未解决证明错误”。ρ=γ^L 使用最短可达深度而非路径概率，作者明确指出替换成 discounted occupancy 后其证明不成立；HSVI2 的工程改进也不是由该分析推出。

**取得与来源。** 7 篇主 PDF、作者勘误及旧 Smith 上传已通过标准 HTTPS 重新比较字节哈希，和对应公开来源相同。第 8 篇 Roy 的本地 40 页期刊 PDF 已核对首页与正文，但中断前的原下载 URL 未从持久日志恢复；本轮 CMU JAIR／Gordon 标准请求超时或断连，arXiv 同题名版本字节不同，故不冒充出处。索引以 `reference_pdf_url` 保留对照入口，并显式标记 `source_provenance_status`；有效本地 PDF 不等于下载来源已重证。未重试此前三条出版社 403，也未用错误标为 Sondik 1978、实际为 1973 的 ResearchGate 文件替代。

全部新增本地文件均检查 PDF 签名、pdfinfo 页数、大小、SHA-256；概念有规范原始来源链接，三库按规范标题与 arXiv 标识检查无重复。每批正文读取不超过 8 页；公式乱码处查看了原页图像。完整证明、全部实验与运行时比较未独立复现。文件保留在已忽略的 `papers/`；未修改 panel、冻结稿、共享构建器或生成总表。详见 `papers/rl/pomdp_classics_access_2026-10-28.json`。

本轮未闭合：三个原出版主 PDF、HSVI 技术报告详证、SARSOP 所引 Hsu 2007 原始复杂度证明、线性 belief 压缩后继重审、一般连续／无限时域 POMDP 和目标可达性版本。不能将本批已登记等同 POMDP 或 RL 学科全覆盖。

**Roy05 来源补核（本批追加）。** 对两个新公开作者来源进行了标准 HTTPS GET，均返回有效 PDF，但均与本地 857,192 bytes、SHA `f99b7e669db5ada361a5f333a2deb50b9fe9e10e8550edcc125403dfe6a3f5a8` 不同：[Thrun 所存 Roy04a.pdf](https://robots.stanford.edu/papers/Roy04a.pdf) 为 751,737 bytes；[Roy 团队 JAIR05.pdf](https://groups.csail.mit.edu/rrg/papers/jair05.pdf) 为 857,538 bytes。后者虽仅差 346 bytes，未做内容／像素对照，不能宣称只有元数据差异。保留现有已读主 PDF，来源未核状态维持；未替换版本、未重试 CMU、未改 `rl.json`。访问 JSON 已保存两份远端字节哈希与结果。

**Roy05 可追溯主版本已闭合（继上条）。** 已将 [MIT RRG 作者团队公开 PDF](https://groups.csail.mit.edu/rrg/papers/jair05.pdf) 另存为 `papers/rl/roy_gordon_thrun_2005_belief_compression_mit.pdf`（40 页，857,538 bytes，SHA `48e23c8ce49e53d3666132a9b2de1fb40ebd8b2a5b7e127946c9abc6ff4b596c`）。原文件未覆写，已转为 supplemental，并保留其下载出处未恢复的历史说明。

两个文件按每批各 4 页、共 8 页抽取，分 10 批逐页比较原始 `pdftotext` 输出与 NFC／空白规范化文本。37 页逐字一致；p.1 差异为卷号 23→22、提交日期 05/04→05/05 及版权行，p.37–38 仅参考文献大小写及 `space→sepace` 拼写差异。已有证据 pp.4–9、11–13、19–21 的文本和物理页码完全一致；p.1 科学题名、作者、摘要及正文一致。未进行全篇像素或图像比较，不声称两份 PDF 同字节。

新主稿首页已渲染目检。卷号／提交日期**与期刊元数据不一致，属于作者网站收录稿，非已核实最终出版版**；不能由正文相同就认定是出版社排版笔误。`rl.json` 已更新主路径、来源、大小、SHA、版本说明和概念证据表述，Knowledge 分类不变。旧件原来源仍未恢复，但当前主文件有可追溯作者来源；`roy_2005_version_comparison_2026-10-28.json` 保存全部逐页比较哈希与差异，访问 JSON 保留历次尝试并追加闭合记录。论文与概念数量不变。


## 2026-10-28：风险、矩形不确定性与控制目标经典补录

本批新增 **8 篇论文、6 个概念、1 本书预览 resource**，并补读既有 Nilim–El Ghaoui 2005；RL 现为 **263 篇／259 概念／2 resources**。253 篇有本地主 PDF，10 个缺口（原 8 个，加 Iyengar 与 Blackwell）；缺文均有明确 `pdf_path`，没有捏造本地 SHA 或页数。分类只针对列出的主张，正文选读不等于完整证明核验。

| 原作 | 本轮实际证据 | 分类与边界 |
|---|---|---|
| Howard–Matheson 1972 | 原文重印抽取14页，重点§2–6与Appendices A–B | 条件 S；指数效用/财富无关风险偏好、有限状态动作；§4渐近式另用不可约非周期。出租车例子 K |
| Artzner–Delbaen–Eber–Heath 1999 | 1998-07-22作者稿，pp1,3–8,13–16 | 条件 S；有限终端状态、静态净值与正参考回报下相干风险表示，非动态一致性保证 |
| Rockafellar–Uryasev 2000 | 作者1999-09-05稿pp1,3–8，Theorems1–2 | 条件 S；连续损失CDF与阈值优化，联合凸性另需损失凸/可行集凸 |
| Rockafellar–Uryasev 2002 | 作者2001-11-28稿pp1,3–6,8–10,12–16 | 条件 S；绝对可积损失、分位点原子分割及Theorem10，不能随意换成超过VaR的条件均值 |
| Epstein–Schneider 2003 | 作者2002-09稿pp1–16，§§2–3 Definition3.1/Theorem3.2 | 条件 S；给定有限信息树、CP/MP/RP/DC/FS公理与闭凸全支持先验。矩形性相对于信息过滤 |
| Ruszczyński 2010 | 作者2009-04-29稿pp1–16，Theorems1–4 | 条件 S；嵌套风险、Markov风险映射、紧性/连续性/可积性；无限折扣另需平稳及有界成本 |
| Iyengar 2005 | 作者上传的2004-05-04 CORC报告网页原文，pp3–10及27–28 | 条件 S，远程原文已选读；本地PDF403。并非仅摘要，但未比对最终2005刊版 |
| Blackwell 1962 | 官方原作目录所录原摘要及书目 | S摘要级候选，条件待核；有限S/A与β→1问题已确认，原定理/证明仍缺失 |
| Nilim–El Ghaoui 2005（已有） | 原刊PDF本轮补pp4–8，重点Theorems3–4及式19–35 | 深化既有鲁棒递推概念；无限折扣矩形模型下静态/时变对手可等值，不新增paper |
| Altman 1999（resource） | 出版社/发行商26页preview，版权ISBN、目录及导论pp1–9 | 不计paper，不整书升层。第3章折扣、第4章平均成本及§3.5/§4.4随机化定理正文尚未取得 |

**概念复用。** 6 个新概念分别为指数效用与确定性等价增益、静态相干风险表示、CVaR阈值优化、离散CVaR原子边界、相对信息过滤的矩形先验、嵌套Markov风险DP。Iyengar接入已有「矩形转移不确定性下的鲁棒 Bellman 递推」，Blackwell接入已有「平均回报、差分价值与最优性层级」且单列摘要来源状态，没有为每篇机械造概念。

**矩形性不是静态性的反面。** Iyengar作者报告§2 Assumption1、式(2)–(3)允许所有跨历史/时期的条件核拼接；式(15)还用动作间可分选择。时序是决策者选π、对手观察π后选路径分布，当前所选核不能限制未来选择。§3区分重访同一(s,a)时重新选核与固定核，AppendixA的同一urn共享参数例子显示矩形性会失效。Nilim的有限时域静态与时变问题需区分，但其无限折扣Theorems3–4在矩形集合等条件下给等值结论。跨行共享参数导致的失效应明确归于耦合，不能只说‘因为模型静态’。

**风险目标的层级边界。** 静态相干公理不自动保证动态一致；动态一致的风险也未必是Markov风险。Ruszczyński先定义确定性历史选择策略，随后才给其类内Markov化和折扣方程，不能泛称所有随机策略无需随机化。Rockafellar–Uryasev2000稿明确用连续CDF；2002稿才处理原子，保留‘阈值＋超额期望’公式但改变尾部定义。Altman导论明确分期望约束与样本路径约束，目录中的随机化次数/可行性定理尚未读，未把它们写成已证结论。

**版本与身份。** Howard主文件来自公开重印卷物理417–430页，重印883–896，对应Management Science18(7):356–369；实际论文首页两位作者已渲染核对。源URL文件名写1983，但卷版权页为1984，索引保留这一差别；目录仅列Howard不能覆盖论文首页作者。主文件14页SHA `a213f801bc043a083bd132d86775f12de4a88df407cbe24c4ecd5478d4e74c12`，原卷492页另存supplemental。其余作者稿日期与出版年份分别记录。Epstein文是JET113(1):1–31，不是同卷pp32–50的IID companion。Altman HAL `inria-00074109`实为1995 RR-2574报告119页，保存为发现材料，不顶替1999书；本地book仅为26页preview。

**有界访问与核验。** 已失败的入口不重复：Iyengar作者站证书主机不匹配/替代域名DNS失败，ResearchGate作者PDF403，但作者上传网页的原文定向可读；Blackwell/Veinott的Euclid下载返回HTML，Blackwell镜像403。没有跳过TLS校验，也没有将HTML伪存成PDF。每批正文抽取≤8页；6份新增主PDF及resource/重印容器核对PDF签名、SHA、字节数和页数，三库题名去重、8篇来源关联、概念名唯一性自检通过。最大新增源卷16,011,046 bytes；所有材料在已忽略的`papers/`、`.cache/`内，没有加入训练产物或大二进制Git追踪。本轮未修改panel、冻结稿、共享构建器或总表。访问与版本记录：`papers/rl/risk_control_access_2026-10-28.json`。

**仍开放的后继链。** Veinott1969原文、Blackwell1962完整证明、Altman1999第3–4章及随机化/可行性条件、Iyengar本地主PDF及最终刊版比对，均未闭合。相干风险在无限Ω的拓扑条件、随机化风险控制、无折扣瞬态风险及风险约束规划可后续分别追原作；本批没有宣称控制/决策理论全覆盖。

**根线程并行POMDP补核（照实转记）。** 根线程确认Stanford Lovejoy1988 WP1024入口公开HTML可取，但Download路由及正常cookie/referer跳转均只回Downloading HTML，未得PDF；不升级1991论文的摘要级核验。ERIC ED124161的2页实际是Wollmer1976 USC报告77，不是Sondik1978，已排除错身份；详见根维护`papers/rl/pomdp_recovery_followup_2026-10-28.json`。本分工未重试上述来源。

根线程新构造的矩形性反例脚本`tools/rectangular_uncertainty_counterexample.py`本轮只读检查并运行通过；独立核算固定联合集合下值为(1−δ)/2、逐行矩形包给0，uniform δ下误差>ε的测度为max(0,1−2ε)（ε≥0）。这属于本库独立有限模型推导，不计论文或新概念，也不是两篇原作中的具体例子。至此交还RL索引写权限。


## 2026-10-28：状态估计、双重控制与信息结构原作补录

本批新增 **8 篇论文、6 个概念**，补强既有「POMDP belief state 与正确模型的充分统计量」；RL 达到 **271 篇／265 概念／2 resources**。其中 **260 篇有本地主 PDF、11 个主 PDF 缺口**。新增缺口只有 Witsenhausen 1968；其 Science 标签仅表示已定位理论主张的候选类型，原文公式和证明条件未核验。此前 10 个缺口保持实际状态，没有用镜像转写制造本地页数或哈希。

| 原作 | 实际阅读与版本 | 分类及关键 Scope |
|---|---|---|
| Kalman 1960 Contributions to the Theory of Optimal Control | IIT Bombay 所存许可重印；PDF p3 身份、pp4–6/8–18 模型和条件，扫描 OCR 加关键页目检 | 条件 S；精确状态、连续线性系统、二次成本。有限时域解、无限时域极限、闭环稳定性使用不同条件 |
| Kalman 1960 A New Approach to Linear Filtering and Prediction Problems | NTNU 所存双页扫描重印；PDF p3 右页身份、pp4–6（原36–41）Th1/2/3 与递推 | 条件 S；最佳线性估计与联合高斯下条件均值分开；矩阵逆需条件，渐近收敛未由本文证明 |
| Kalman–Bucy 1961 | Rutgers 所存期刊扫描；PDF pp1–3、6–8，原95–97、100–102 | 条件 S；有限时域、无限过去和稳定性分别依赖 Th1/3/4；§9 全部证明未审 |
| Åström 1965 | Lund 机构原作存档，两页封面；PDF pp3、6–18 对应原174、177–189 | 条件 S；有限状态/观测、有限时域、已知模型、紧控制集和连续性；belief 充分性不是点估计充分性 |
| Ho–Chu 1972 Part I | 课程站所存 IEEE 原刊扫描，作者 Harvard 书目交叉核实；8页，重点 pp2–6 | 条件 S；部分嵌套信息结构下静态化；联合高斯和正定二次共同成本进一步给仿射最优。纯策略，不替换为任意动态博弈 |
| Feldbaum 1960 I | Math-Net 俄文原刊10页；pp4–9 定义与八项设置，p10 英文摘要 | K；问题定义与机制主张，不能因题名含 Theory 就把整篇认作已证一般理论 |
| Feldbaum 1960 II | Math-Net 俄文原刊12页，分批选读设置、后验、风险递推与末页摘要 | 条件 S；有限时域、无记忆对象、已知 Bayesian 模型；明确假定每步 argmin 取得，未独立审核一般可测选择条件 |
| Witsenhausen 1968 | SIAM 官方摘要/书目；非作者 Paperzz OCR 镜像仅定位 §§2–6 主张，原 PDF 未得 | **S 候选，条件待核**；存在非线性优于全部仿射的反例主张，不表示所有参数均如此，也不表示已核出全局最优解 |

**三组不能合并的边界。** Kalman 1960 调节文的有限时域 Riccati 构造不以完全可控为必要条件；原文无限时域极限用完全可控，稳定性再加一致完全可观/可控与 Q/R 的上下界。本轮保留原文条件，不用现代较弱的 stabilizable/detectable 表述替换它们，也没有从精确状态反馈推出一般 LQG 分离或模型失配鲁棒性。重印 Eq6.5 的可见反馈符号与后文闭环式不一致；已目检而未与原刊校勘，索引不抄该式，不断定为作者错误或重印排版错误。

Kalman 1960 滤波文 Th2 将联合高斯和限制线性估计的情形分开：非高斯时平方损失投影是线性类最优，未必是所有估计器最优。其 Problem I 写 y=Mx，观测噪声可通过状态扩展处理；不能把现代模型形式当原文公式。Kalman–Bucy 1961 再区分有限起点最优、无限过去协方差解、稳定与收敛；§4(e) 对白噪声形式化的自我限制、§9 未逐行审核均已记录。

Åström 的充分状态是完整后验，允许紧连续控制集；它既不是实际隐状态，也不是条件均值或任意 RNN 隐向量。行动仍影响未来信息。Feldbaum I 提出双重作用，II 才在所列设置和最小值取得假设下给后验与风险递推；其 neutral 情况仍可学习，只是学习规律不依赖控制幅度。Ho–Chu 的部分嵌套要求信息包含对所有策略成立；联合高斯原始变量可相关，不能把应用中独立噪声条件错误提升为整个定理的必要条件。部分嵌套是充分条件，其失败不自动证明最优策略非线性。

**版本与页码。** Kalman 控制文原刊为 Boletín de la Sociedad Matemática Mexicana 5:102–119；所存文件前2页为后来的导读，PDF p3–20 是重印149–166。滤波文原刊 ASME82(1):35–45；8张 PDF 每张两页，p3右=原35/重印169，邻篇和导读不计滤波原证据。Åström 机构题名有 I，原刊页头未印 I，索引明确保留这一差别；PDF p3=原174。Ho–Chu 文件首尾含邻篇与 Part II 开头，只登记 Part I。Feldbaum I/II 原刊分别为1960年21(9):1240–1249和21(11):1453–1464；英语译本的1961翻译日期与俄文发表年分开，译本未阅读、未另计论文。

**来源与核验。** 7份新增主 PDF 均通过签名、字节数、SHA-256 和 pdfinfo 页数检查；最大9,593,823 bytes。扫描身份由指定原页目检，根线程维护哈希身份例外。本轮正文提取或 OCR 每批不超过8张 PDF 页；页码与选读范围写入 paper/concept evidence。三库规范化标题、规范来源 URL 去重通过，6个新概念名唯一且来源均能关联本批原作。原 belief 概念改以 Åström 为主链接并保留 Kaelbling/Monahan 历史证据。

Witsenhausen 的 NTUA 原文入口与公开卷扫描 GET 均发生 TLS EOF，Paperzz 下载入口403，出版社 PDF 路由转到摘要；没有关闭 TLS 校验、反复重试或将 HTML 保存为 PDF。Paperzz 是**非作者转写镜像**，公式有 OCR 缺字；paper、concept、claim annotation 均标出候选和未核条件，保留明确但尚不存在的 `papers/rl/witsenhausen_1968_counterexample.pdf`，确保共享 audit 报缺失。根线程的新源检索未找到可核原刊，未提升阅读级别。

**未计入与后续。** Bellman1957 仅取得原 p679 的远程首页，未读其余定理条件；原首页写第5期而站点书目写第4期，待核，不补造论文条目。Ho–Lee1964 有作者书目和 NTIS 报告摘要；2页报告记录不是7页期刊论文，未登记。Feldbaum III/IV、有记忆系统推广、英语翻译版本、现代 LQG 分离和鲁棒性边界均留作后继原作链。两份 Kalman 只是独立重印章；已见选集目录不等于取得包含 Witsenhausen 的整书容器，未继续整书扩读。此批不宣称控制学科、RL 或所有经典已覆盖。

访问、阅读范围与校验记录保存在 `papers/rl/control_foundations_access_2026-10-28.json`；本地材料均位于 `.gitignore` 覆盖的 `papers/`、`.cache/`，没有新增 Git 追踪或超过50MB文件。未修改 panel、冻结稿、共享构建器或生成总表。RL 索引与 coverage 完成后交还写权限。
