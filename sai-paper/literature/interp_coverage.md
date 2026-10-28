# 可解释性经典论文与概念：覆盖与证据地图

截至 2026-10-28，当前登记 **317 项论文或作者原生文章、302 个概念**：299 项有本地主 PDF、18 项为本地 HTML。每项在 [interp.json](interp.json) 中记录原始来源、主张、K/L/S 暂定分类、范围、证据位置、审读级别、下载路径与 SHA-256。PDF 均经 `pdfinfo`、首页题名匹配核验；HTML 保存页面本体，其脚本、样式或图片可能仍需联网。本清单是代表性检索，不是完整普查。

分类针对写出的具体主张，K/L/S 不评判方法是否有价值。Network Dissection、Sanity Checks、Attention is not Explanation 、Refusal Direction、Space and Time、Geometry of Truth、Word2Vec-style Vector Arithmetic、Induction Heads 和 Function Vectors 的实证主张已由 Law 候选调整为 Knowledge，因当前证据局限于指定条件；这不是对其研究价值的评价。“L 候选”表示跨条件关系有研究证据，但是否已充分穷尽目标条件和边界尚未审核；“Science（部分理论）”表示有条件化构造或定理，不等于已给出现实模型上的完整边界。PDF 取得不代表整篇已读或论文结论被复现。

## 已覆盖的研究脉络

| 脉络 | 已纳入的代表工作 / 概念 | 尚需注意 |
|---|---|---|
| 视觉解释与归因 | Zeiler-Fergus、Deep Visualization、Network Dissection、Grad-CAM、DeepLIFT、SmoothGrad、Integrated Gradients、SHAP、TCAV、显著图 sanity checks、LIME | 从刺激响应或更好看的图到机制忠实性的区分；归因基线及反事实分布 |
| 语言表征与探针 | 结构探针、BERT attention、BERTology、control tasks、selectivity、Tuned Lens、INLP、概念瓶颈 | 可解码不等于被模型自然使用；控制任务与探针容量 |
| 电路与归因 | Circuits、Transformer Circuits、induction heads、IOI、Greater-Than、ACDC、edge pruning、attribution patching／AtP*、Tracr | 数据／指标／替换方法、真值电路、完整性和反例 |
| 稀疏特征与 SAE | Toy Models of Superposition、Towards Monosemanticity、SAE 特征、Scaling and Evaluating SAEs、Claude 3 Sonnet SAE、Sparse Feature Circuits、Transcoder | 玩具模型与真实模型；稀疏性、重建、忠实性和特征不完整 |
| 因果干预与限制 | ROME、causal tracing、activation patching、patching 最佳实践、subspace patching illusion、Hydra self-repair、因果抽象、causal scrubbing | 有干预不等于唯一机制；比较替代路径和干预分布 |
| 分布式表征与控制 | 线性表示假说、真值几何、时空、Othello world models、function vectors、activation steering、refusal direction、多模态神经元 | probe、方向和控制有范围；跨任务和外部模型的延伸仍是待验假说 |
| 隐性知识与评价 | CCS、quirky 模型、ELK 不可能性、TruthfulQA、RAVEL、解释忠实性综述 | 已验证的是形式化或受限 benchmark，不能推出通用诚实性或不可解释性 |

## 已定位原文的关键证据

以下 **69 项为局部章节／图表核对**，未宣称全文精读：

1. **Induction Heads**：PDF 第 1–3 页，作者区分小型 attention-only 模型中的直接干预与较大含 MLP 模型的间接／相关证据。
2. **Toy Models of Superposition**：PDF 第 1–4 页，稀疏特征与小 ReLU 玩具模型；作者明确真实网络外推未知。
3. **Grokking 进展量**：PDF 第 1–3 页，mod 113、单层 ReLU Transformer 与 AdamW 的具体设置；不能直接泛化至所有 grokking。
4. **IOI circuit**：§4、Figure 2 和 §4.1；26 个注意力头与特定 logit-difference 度量的 87%，不是任务准确率。
5. **ROME**：§2.1/Figure 1 和 §3.4/Figure 5；定位与编辑的关系还须对照下方反例。
6. **定位不等于编辑**：§4.2–4.3/Figure 4；在 GPT-J 和筛选出的 652 个 CounterFact 例子中，指定层两种指标相关系数为 −0.13。
7. **Probe control tasks**：§3.5/Figure 2；ELMo 词性任务 MLP 示例 97.3% 对控制任务 92.8%，selectivity 为 4.5 个百分点。
8. **Sparse Autoencoders Find Highly Interpretable Features**：§3.2 解释指标比较及 §4.1–4.2/图 3 的 IOI 干预。
9. **Scaling and Evaluating SAEs**：§3.1.1、§3.1.4 和 Figures 3–4；作者指出不同 latent 数下的信息瓶颈不同，MSE 直接比较存在限制。
10. **ACDC**：§3/Figure 2 中的 patching 和删边过程；GPT-2 small 的 Greater-Than 回路找回结果依赖人工基准。
11. **Subspace Patching Illusion**：§3.2–3.3、式(2)–(3) 的三维线性构造性反例；不说明所有 patching 都失败。
12. **Hydra**：§2/Figure 1 的消融和 §3.3/Figure 6 的 self-repair / erasure 玩具模型。
13. **有限群 universality**：§4 的群表示算法及 §5.4/Figure 5 的读出分解；仅限作者所测网络及任务。
14. **Tuned Lens**：§2–4 的层间表示变化、仿射探针和因果忠实性设计；模型内部读出不是现成的唯一机制。
15. **Function Vectors**：§2.3/式(3)–(4) 和 §3.1/Table 2；主要结果按 10-shot ICL 原本答对的子集统计，不能把它当作无条件总体。
16. **RAVEL**：§2.2 的干预指标及 §4/Table 2；MDAS 的 60.1%/65.6% 为基准 disentanglement 分数，不是全网络准确率。
17. **ELK 不可能性**：PDF 第 8–10 页、§5.1/Theorem 5.2 与 Limitations；定理在特定 CID、训练分布和评估器假设下给出边界，附录证明未逐条审核。

18. **LRP 原文**：第 7 页 relevance conservation，第 20 页式 (55)；守恒是归因分配条件，不保证因果忠实性。
19. **Sanity Checks 原文**：§2 区分参数随机化与数据标签随机化，§3–4 分别测试；方法敏感性并不直接等于解释忠实性。
20. **Revisiting Sanity Checks**：§1–4；作者针对模型随机化提出任务选择混杂的论证，并在自造视觉任务比较 guided 与 vanilla backprop；并非作废一切随机化检查。
21. **Network Dissection**：§2.1–2.2，Broden 标签及 IoU > 0.04 的检测器阈值；不在标签集中的概念无法得到相应高分。
22. **A is for Absorption**：NeurIPS 2025 版 Figure 1、§3 和 §7；层级概念诱发的特征吸收及作者承认的测量限制，未逐条核对证明。
23. **Logit Lens 原文**：overview、the logit lens、logits；GPT-2 1558M 的中间层读出案例；作者明确这是部分视角。
24. **Building Blocks**：How Are Concepts Assembled?、Design Space、Challenges；组合可视化和归因，作者指出神经元语义一致性等可信性问题。
25. **BLIP Causal Tracing**：§3–4、Table 1 和 Figures 4–5；具体追踪采用 COCO-QA 颜色识别，200 个样本和每个样本 10 次噪声运行；不能推及全部视觉问答。
26. **Large Multi-modal Models Can Interpret Features**：Figure 1 的 LLaVA-NeXT 单层 SAE、自动解释与干预流程；尚未逐项验证其结果。

27. **LIME**：§3.1–3.4、式 (1)–(2)、Figure 3；可解释表示、采样扰动和加权距离核决定局部替代模型，局部拟合不等于全局因果忠实。
28. **Integrated Gradients**：§2–4、式 (1)、Proposition 1；固定基线与正则性假设下的完整性为形式性质，不足以证明解释是真实机制；未独立审核证明。
29. **TCAV**：§3.1–3.5、Figure 1；使用概念和对照例子训练线性方向，再计算方向导数和样本统计，结果依赖概念数据及选定层。
30. **Structural Probe**：§2.1–2.3、§3、Table 1；训练线性变换拟合树距离／深度，评估 BERT、ELMo 与 Penn Treebank 英语；可解码不等于模型因果使用。
31. **Attention is not Explanation**：§2–4、Figure 1、Table 1；BiRNN 等指定任务中的替代注意力现象，不能作全架构否定定理。已从 Law 候选改为 Knowledge。
32. **Attention is not not Explanation**：Figure 1、§3–4；均匀权重、随机种子和端到端对抗训练补充诊断；不证明注意力总能忠实解释。
33. **SHAP**：§2–3、Theorem 1/式 (8)；唯一性属于规定解释类与公理下的形式主张，不解决特征缺失分布或实际因果贡献；未独立审核证明。
34. **Sparse Feature Circuits**：ICLR 2025 §2、式 (1)–(4)、Figure 1；保留 SAE 重建误差项并使用间接效应的一阶及积分近似；实证限于作者模型和任务。
35. **Transcoders**：§3.1–3.2、式 (3)–(5)、Figures 1–2；稀疏近似 MLP 输入输出，固定权重不意味着激活或行为对输入不变。
36. **Grad-CAM**：§3、Figures 1–2；按类别梯度加权卷积特征，得到粗粒度热图，类别定位不等于精确因果贡献。
37. **Feature Visualization**：Optimization Objectives、The Enemy of Feature Visualization、The Spectrum of Regularization；无约束优化可能产生高频伪影，解释受到先验和目标影响。
38. **Towards Monosemanticity**：Using Sparse Autoencoders、The (one-layer) model we're studying、Feature Splitting、How much of the model does our interpretation explain?；作者承认尚不清楚多大比例的模型行为被解释。

39. **DeepLIFT**：§3.1–3.3、式 (1) 和 Figure 2；差异归因的加和性质依赖具体规则和参考输入，并非普遍因果忠实。
40. **SmoothGrad**：§2.2、§3.1–3.2、Figure 3；高斯扰动下对敏感图平均，示例采用 50 次采样；视觉更平滑不能证明更忠实。
41. **Concept Bottleneck Models**：§1–3、Figure 1；将输出限制经过预测的概念层，并可干预概念；作者指出模型形式可表征因果关系但不要求如此。
42. **Activation Patching 最佳实践**：§2、§3.1–3.2、§4–5、Figures 1–2；破坏方法、度量与恢复窗口影响定位，高斯扰动分布外机制是初步假说。
43. **Refusal Direction**：§2.2–2.5、Table 1、Figures 1–2；13 个模型，用每类 128 训练、32 验证样本选方向，跨层跨位置消融；现调整为 Knowledge，不能推断普遍唯一机制。

44. **Causal Abstraction**：PDF 第 6–8 页、§2.1 Definitions 5、9–11；形式框架明确聚焦确定性模型，定理证明和具体应用尚未逐项核对。
45. **BERT Attention**：§2–3、Figure 1；英语 BERT-base 的 144 个头和语言探针案例，相关模式不证明因果使用。
46. **Tracr**：§1、Figures 1–2；RASP 程序编译到结构已知的 Transformer，编译案例不是自然训练模型的机制证据；§3–5 尚未逐项审核。
47. **Contrastive Activation Addition**：Figure 1、Table 1；成对激活差在指定层和 token 处注入，示例是 Llama 2 7B 第 13 层；可控性不证明唯一语义方向。

48. **Zeiler–Fergus**：§1–2 的 deconvnet 投影和遮挡思路；像素图展示条件化响应，不是完整机制。
49. **Deep Visualization**：§1–2 的实时激活与输入优化可视化工具；图像先验和正则会影响可视化。
50. **Deep Inside CNNs**：§1–3 的类别图像优化和单次反向传播显著图；单 CNN 上的定位结果不保证因果忠实。
51. **FFN Key–Value Memories**：§1、Figure 1 的第一、第二矩阵与激活系数组合；代数描述与人类语义可解释性的实证结论应分开。
52. **Greater-than Circuit**：§1–2、Figures 1–2；GPT-2 small 年份任务，包含 tokenization 过滤，作者不将其视为完整数学能力。
53. **Othello World Representations**：§1–2 的 Othello 下一步预测与非线性探针；合成博弈模型不代表一般语言模型。
54. **Othello Linear Representations**：§1、Figure 1；使用当前玩家／对手相对颜色重新定义状态后线性探针可用；不把过去绝对颜色探针失败解读成普遍非线性。
55. **Influence Functions**：§2.1–2.2、式 (1)–(3)、Figure 1；局部公式依赖二次可微与严格凸等假设，非凸网络应用是近似。

56. **Contrast-Consistent Search**：§1、Figure 1；反命题一致性作为无监督探针训练约束，一致不充分保证为真。
57. **Quirky Language Models**：§1、Figure 1；Bob 触发的人为错误设置，89% 是真假上下文的 AUROC 差距恢复比例，不是准确率。
58. **Space and Time**：§1、Figure 1；空间和时间可被线性读取，但作者强调并非动态因果世界模型；已调整为 Knowledge。
59. **Geometry of Truth**：§1、Figure 1；LLaMA-2-70B 上真假陈述的 PCA 与后续探针、干预研究，不保证所有真假判断都有稳定方向；已调整为 Knowledge。
60. **Linear Representation Hypothesis**：§1–2.2、Figure 1、Definition 2.1、Theorem 2.2；形式关系依赖指定反事实和表征假设；实证外推未知。
61. **Word2Vec-style Vector Arithmetic**：§1；首都、大写、过去式等关系任务的机制报告，作者区分预训练检索与局部上下文检索；已调整为 Knowledge。

62. **Edge Pruning**：§1；梯度优化选取连边，作者提出多种模型和任务案例，尚未核验全部比较。
63. **How to use and interpret activation patching**：§1.1–1.3；不同输入的缓存激活替换与零消融及其解释区别。
64. **Attribution Patching**：§1；以两次前向和一次反向传播对单个输入对近似边的影响，全面性能比较待核验。
65. **AtP***：§2.1–2.2、式 (1)–(5)、Figures 1–2；目标函数和所用 clean/noise 输入对决定因果归因的含义；论文所称形式界尚未核实。
66. **INLP**：§1、Figure 1；迭代线性分类和零空间投影抑制线性属性读取，不保证所有非线性信息被消除。
67. **Causal Mediation for Gender Bias**：§1、Figure 1；在定义的干预与中介下区分直接／间接效应，不推广为所有偏差的普遍规律。
68. **Activation Addition**：§1 的方法和作者例子；成对提示差的可控性受模型、层、系数和提示条件约束。
69. **Representation Engineering**：§1、Figure 1；表征层读取和控制的研究框架，其安全相关案例不保证完整透明。

此处列出前 69 项；另有 15 项已于下一轮核对指定位置。当时共 84 项局部核对、19 项摘要／页面级初筛；具体实验、定理和全篇尚待阅读，最新统计见文末。两篇近似的方法不应仅凭算法介绍就归入 Science，初筛“Law 候选”也尚不足以证明用户要求的充分外延。

## 本轮新增：表征比较、功能重用与忠实性边界（18 项）

此批 18 篇中，14 篇现已局部核对具体章节或图表，4 篇仍为 **PDF 首页与摘要初筛**；尚未完成方法、实验或证明的系统审读。对具体发现的表述均保留原作者所用模型和评估条件。对应论文和 15 个概念的链接与本地文件见 [interp.json](interp.json)。

- 注意力头与事实表征：Knowledge Neurons（2104.08696）、FFN 词表投影（2203.14680）、Voita 等头剪枝（1905.09418）、Michel 等头剪枝（1905.10650）。
- 探针与跨网络比较：探针编码成本（2003.12298）、跨网络特征收敛（1511.07543）、CKA（1905.00414）、SVCCA（1706.05806）。其中 CKA 的形式限制在满行秩等假设下列为条件化 Science；定理证明尚未独立逐行审查。此处编码长度只是探针评价工具，不是 SAI 分类的核心原则。
- 解释可靠性：Ghorbani 等解释脆弱性（1710.10547）、Alvarez-Melis 等局部稳健性（1806.08049）、模型解释调试测试（2011.05429）、思维链不忠实（2305.04388）。一个反例不证明所有方法失效，局部稳定性也不能代替因果忠实性。
- 回路跨条件与控制：copy suppression（2310.04625）、70B Chinchilla 多选回路分析（2307.09458）、ITI（2306.03341）、回路跨任务重用（2310.08744）。在单个模型或有限任务上找到机制及复用，仍不等于穷尽其外延。
- 特征是否唯一：跨模型 SAE 空间相似性（2410.06981，下载为 arXiv v4，题名为 *Quantifying Feature Space Universality Across Large Language Models via Sparse Autoencoders*）和跨种子稳定性（2606.12138）。前者测量空间相似，后者区分特征与子空间重现；都不能单独证明真实模型存在唯一的 SAE 分解。后者是近期工作，尚不宜称为经典。

## 本轮：定向核查 15 篇与概念标签一致性

这 15 篇检查了指定原文位置，没有逐行审查整篇论文、定理证明或复现实验；在 `interp.json` 各条的 evidence 保留具体位置。

- 知识神经元（§2–3.1）、FFN 词表投影（§2–3）：区分神经元归因、词表线性投影与真实因果机制。
- Voita 注意力头（§3–4）、Michel 头掩码（§2.3–3）：区分单头遮蔽与联合剪枝，结果依赖翻译或 BERT 任务与所测模型。
- MDL probing（§1–2）和 SVCCA（§1–2）：分别依赖探针/编码方案和有限输入样本、保留子空间；都不能直接证明因果使用。
- CKA（§2.1 Theorem 1）：其定理需 $X,Y\in\mathbb{R}^{n\times p}$ 且两者满行秩。论文另讨论列中心化，中心化矩阵秩至多 $n-1$；因此不把该满秩命题原样套到中心化矩阵。表征相似性仍不等于机制相同。
- 解释脆弱性（Figure 1）与局部稳定性（Definition 2.1、Figure 1）：结果受扰动、邻域和模型限制，稳定也不自动证明忠实。
- 思维链不忠实（Table 1）与 ITI（§1–2）：分别区分偏置信号被遗漏、可探测的真假相关性和实际模型机制。
- Copy suppression（Figure 1）、跨任务回路复用（§1–2.1）、Chinchilla 多选电路（Figure 1）：明确 GPT-2 Small 指定头、GPT-2 Medium 的两任务及 Chinchilla 的答案字母选择范围。
- 解释对称性（§2.2–2.3、Proposition 2.3）：精确群平均在假设内具备不变性；Monte Carlo 是近似，形式性质不自动保证因果忠实。

**概念与论文一致性。** 显著图随机化、attention 权重、induction head、拒答方向与 function vectors 均改为 Knowledge，避免把有限实证称为已充分扩展的规律。Tracr 构造与 causal scrubbing 方法也不再仅凭构造或定义标为 Science。因果抽象概念作为框架暂列 Knowledge，具体经核查的条件定理可另做 Science 主张；线性表示及对称性的条件结果保留限定标签。

## 关键仍欠缺项，建议下一批补齐

- 音频、视频、视觉语言模型的多模型多任务机制研究仍不充分；新增 BLIP、LLaVA-NeXT、SAE-V 和 Prisma 涵盖不同方向，并不代表系统覆盖全部多模态。
- SAE 的特征分裂／吸收已有专门原文，但不同训练种子、字典非唯一性、分布外失效与跨模型复现还需扩展；不能把 SAE latent 视为自然唯一单位。
- 各方法应继续补查实验细节、相反证据和限定域；目前没有双人复核，未独立复现。部分数学推导与其对应的实证主张应分别分级。
- 该批 18 篇 PDF 均成功保存；其中 4 篇摘要之外的章节仍待核对。作者原生 HTML 仅存页面本体，不保证图片和交互组件离线可用。

## 下载及校验

`papers/interp/` 是本地资料目录，`tools/fetch_interp_*.py` 和 `tools/retry_interp_official.py` 保留下载方式。引用的原生网页均已保存 HTML 文件；无需 PDF 的 10 项不应被误报为“PDF 下载失败”。单个文件以 JSON 中的 hash 与大小为准；会议版和 arXiv 版可能是不同版本，`pdf_source_url` 记录实际下载来源。单个 PDF 首页标题核对的目的是避免错配，不是科学内容复现。

## 新增批次：归因评价、训练样本影响与路径定位

本批新增 11 篇一手论文和 11 条概念，覆盖 attention rollout、representer points、TracIn、影响函数脆弱性、自解释网络、解释正则化、ROAR、path patching、事实回忆信息流、SAEBench 及归因不忠实度。全部 11 份 arXiv PDF 已下载并以首页题名、标识、页数和 SHA-256 核验。7 篇定向查看指定方法或讨论章节；其余 4 篇仅首页及摘要初筛。局部阅读不表示全文审读、证明逐行复核或独立复现。

重要区分：ROAR 衡量移除后重新训练的模型，不能直接当作原模型单次干预；不忠实度依赖扰动分布，而低敏感度本身可由无信息的常数解释达到；SAEBench 的代理指标表现不保证任意应用上的评价一致。Path patching 的定位结论始终以计算图、输入和反事实分布为条件。新增条目主贡献暂记 Knowledge；如需对个别形式命题记 Science，须逐条核查其假设、证明和对应边界。
## 本轮：反事实、rationale 与特征稳定性

本轮补充 5 篇原论文与 5 个概念：Concept Whitening、ERASER、Wachter 等反事实解释、Rationalizing Neural Predictions 和反事实因果可行性约束。5 份 PDF 均已用 PDF 签名、pdfinfo 页数、SHA-256、首页标题核验；其中 4 份首页可见对应 arXiv ID，Wachter 的版本首页未显示 arXiv ID，标题及作者已核对。Rationalizing Neural Predictions 全文文本抽取报告 xref 重建警告，已核查的方法页可读取，原文仍需留意该限制。

另外将 8 篇已有资料从首页／摘要推进到指定方法或定义：Convergent Learning、Debugging Tests、跨模型 SAE 特征空间、跨种子 SAE 稳定性、Representer Points、TracIn、自解释网络和 Right for the Right Reasons。TracIn 的理想求和以单样本训练更新为前提，检查点梯度内积是实用近似；Representer Point 的分解依赖线性输出、正 L2 正则和驻点条件。二者都不能被无条件视为实际删除训练样本后重训得到的精确因果效应。

目前 119 项中 **103 项已核对具体章节、方法或图表，16 项仍为摘要／页面初筛**。各条 evidence 保留实际定位；无逐行证明审核或独立实验复现。跨模型与跨种子 SAE 的近期工作只是相关新研究，不按“经典”定性。

## 本轮：原生网页的框架和边界核查

将 5 篇页面初筛推进到指定原文章节：Zoom In、A Mathematical Framework for Transformer Circuits、Causal Scrubbing、Circuit Tracing 和 On the Biology of a Large Language Model。目前 119 项中 **108 项已核对指定正文、方法或图表，11 项仍在摘要／首页／页面初筛**；均未声明全文或证明逐行审核。

Transformer Circuits 在主要推导中忽略 MLP、层归一化和偏置；一层模型 QK/OV 路径展开是明确架构下的形式分解，未据此声称深层现实模型解释边界已闭合，论文主标签调整为 Knowledge。Causal Scrubbing §2–3 将模型图、解释图和对应映射组成假说，递归使用假说允许的数据等价类重采样，并以 scrubbed 行为对照原行为；作者明确指出该检验不足以排除所有错误假说，主标签记 Knowledge。两者都不因使用数学语言而自动记 Science；如果后续找到满足判据的具体条件定理，再逐条标注。

Circuit Tracing 的局部替代模型冻结注意力和归一化量、用跨层 transcoder 近似 MLP，再通过原模型扰动检验部分机制。Biology 文中特别区分筛选出的成功案例、替代模型和原模型，不应将这些案例推广为全面理解；Zoom In 的跨模型 universality 是作者研究主张而非已验证的普适定律。

## 本轮：剩余摘要级条目的正文定位

将 10 篇已有资料由摘要／首页推进到指定正文：Doshi-Velez 与 Kim 的评价框架、BERTology、探针综述、TruthfulQA、Claude 3 Sonnet SAE、NLP 忠实性综述、机制解释开放问题、Prisma、SAE-V 以及 Wachter 等反事实解释。证据位置及具体范围写入对应 `interp.json`；它们属于限定条件的研究框架、方法、基准或案例，均未因形式化或跨案例研究自动归为 Law/Science。TruthfulQA 的 817 道题及 38 个类别是其设计，不代表对所有真实场景的真实性评测。SAE-V 的筛选方法依赖文本和视觉 token 表征相似度，不能由此保证真实因果解释。反事实的输入距离最小不自动意味着可行或具因果效力。

现共 119 篇资料、84 个概念，其中 118 篇已定位指定正文、章节、定理陈述或图表；一篇 *Multimodal Neurons in Artificial Neural Networks* 仍为页面级：本地 HTML 主要正文由外部交互内容加载，未将静态页面当作完整正文。指定位置阅读不等于全文阅读、证明逐行核对或实验复现。

## 本轮：输入不变性、扰动归因与信息论探针

新增 7 篇原论文及 7 个概念：输入不变性和基线选择（Kindermans 等），随机遮罩 RISE，有意义扰动，极值扰动，Lipton 可解释性概念分析，Anchors 条件规则，以及语言表征信息论探针。7 份一手公开 PDF 均检查签名、`pdfinfo` 页数、字节数、SHA-256 及首页标题；五份 arXiv 首页对应标识可见，另外两份分别为 AAAI 论文和 ACL Anthology 版本。每篇定位了指定正文、方法或定义，并未完整审计证明或复现实验。

Anchors 的规则精度和覆盖是针对论文设定的扰动分布与置信水平，并非对任意输入分布的无条件保证。输入平移反例和基线选取提示归因结果可能改变；它们不证明所有归因方法都不可靠。随机遮罩、插入／删除和极值扰动取决于扰动分布；分布外输入与因果解释之间的联系需另行论证。信息论探针的可解码性和控制函数不自动证明语言模型在决策时使用该特征。

当前共 126 篇资料、91 概念：125 篇已有指定原文定位，Multimodal Neurons 网页仍仅页面级。新增论文的主贡献暂记 Knowledge；这不排除论文中存在应单独核查的条件定理或形式性质。

## 本轮：事实知识探测、扰动攻击与忠实性

新增 7 篇论文与 7 个概念：LAMA、提示敏感性、MEMIT、多模态 ViT／CNN 表征对比、LIME/SHAP 对抗攻击、自然语言解释忠实性测试，以及忠实性与可信貌似性之区分。7 份 arXiv PDF 均已检查签名、pdfinfo、页数、大小、SHA-256、首页标题和 arXiv 标识；均仅定位指定方法／定义或图表，完整证明和全部实验未审计。

Slack 等攻击依赖解释扰动与真实输入分布可区分，不能推广为任何解释都失效；LAMA 和提示改写的不同结果表明评估与提示相关，不能据此穷尽模型能调用的事实。MEMIT 的批量编辑以具体模型、层和评价协议为条件；自由文本解释的反事实测试存在作者指出的输入修改混杂；ViT/CNN 的 CKA 或线性 probe 结果不是完整因果机制。

此前仅页面初筛的 *Multimodal Neurons*，本轮核对了 OpenAI 配套研究文章的 Multimodal neurons in CLIP、Absent concepts 和 Fallacies of abstraction 等节（https://openai.com/index/multimodal-neurons/），故记录为配套文章局部核对。它与 Distill 原论文区分：本地 Distill 静态 HTML 的交互正文仍未完整取得，不能视为原论文全文审读。

当前 133 篇资料、98 个概念；其中 132 篇已定位原文指定章节或图表，另 1 篇仅核对配套文章的局部文字。分类均为对应主张的工作判断，不等于证明审计或独立复现。


## 本轮：表征擦除、因果对齐与机制解释基准

新增 8 篇一手论文和 8 个概念：早期表示擦除、Amnesic Probing、主谓一致因果中介、因果抽象、分布式对齐搜索 DAS、交换干预训练 IIT、LEACE 线性概念擦除与 MIB 机制解释基准。8 份 PDF 均检查格式、页数、SHA-256、首页标题和 arXiv ID；每篇已核查指定正文或定理陈述，未逐行审读所有证明或复现实验。MIB 是近期基准，不因收录而称其历史经典。

Amnesic Probing 的 INLP 针对线性可读性，不能推断所有非线性属性信息消失；DAS 与 IIT 的对齐、干预和高层模型均须按具体任务解释。IIT 的全部输入对条件不能由有限采样的高准确率直接推出。LEACE 的线性 guardedness、条件均值与协方差的等价建立在指定线性预测器、凸损失与分布条件下，不证明非线性信息消失；已将条件化定理另列 Science，主方法与有限实验列 Knowledge。因果抽象论文的线性构造展示“可解码而不影响输出”的特定反例，不是所有探针均无效。

当前 141 篇资料、106 个概念；完整原文与配套网页的证据边界见各条 review_level，未完成独立复现。


## 本轮：概念完整性、人类验证与不可辨识性

新增 6 篇论文及 6 个概念：ACE、概念完整性与 ConceptSHAP、Data Shapley、人类解释验证任务、概念瓶颈潜在信息泄漏，以及 Locatello 等无监督解缠不可辨识性。所有本地 PDF 经签名、pdfinfo、页数、SHA-256 和首页题名核对；其中 5 份首页显示对应 arXiv 标识，Data Shapley 为 PMLR 会议版，首页无 arXiv 标识，题名及作者与 PMLR 一手页面吻合。所有条目只核对指定方法、实验或定理陈述，未全面审计证明与复现实验。

ACE 的分割与聚类再由 TCAV 给出方向敏感性，分割分布和作者选择的模型限制其外推；ConceptSHAP 的完整性依赖所选概念、恢复函数和目标输出，不等于现实系统的完整因果解释。Data Shapley 的价值依赖学习算法、训练数据组合、性能指标与测试分布；人类验证研究仅支持所测决策集和合成验证任务。OAI/CUB 上联合概念瓶颈的连续输出可能携带额外任务信息，不构成所有概念瓶颈失效证明。

Locatello 等 §3 Theorem 1 在潜变量维数大于 1 和先验有因子分解密度等设定下，给出保持分布的纠缠变换，构成条件化 Science 不可辨识结果；作者指出结果不能排除通过模型、数据归纳偏置或额外监督在具体情形中获得可辨识性。定理陈述已核对，附录证明尚未逐行审计；论文的实验结果单独保持 Knowledge。

## 本轮：争议标签的非盲复核与两项补充

本轮在知悉原标签后，对 14 项涉及 SAE scaling、因果干预、线性概念擦除、不可辨识与归因性质的主张重新核对原文与条件，详见 [主张复核表](../survey/interp_adjudication.md)。这不是盲法双人标注，不支持一致率估计。关键修订：

- DeepLIFT 的 summation-to-delta 在 §3.1 是所要求的定义性质；§3.2.2 的链式一致性单列条件化 Science 候选，附录证明未逐行审阅。
- Causal Abstraction 的 Proposition 40 是给定 alignment 的结构分解；形式框架本身不等于真实网络的完整机制边界。
- 影响函数的逆 Hessian 公式补明最优点处 Hessian 正定假设；严格凸不自动保证处处可逆。
- LEACE 补明线性预测器、凸损失、类别正概率和矩条件；Locatello 不可辨识的陈述限定在所述一般模型类，定理只要求 factorized density，不额外宣称密度连续。
- 新收录 Deep Taylor Decomposition（1512.02479）与 Concept Activation Regions（2209.11222），两份 PDF 已验证结构、首页标题及 arXiv ID、页数、SHA；前者按方法和实验记 K，后者的径向核/双射等距不变性记 S 候选，证明和完整实验未审。

## 本轮补充：探针、归因分歧与解释目标（2026-10-28）

新增九篇作者原始论文，均有本地 PDF；全部通过 `pdfinfo`、首页题名和 arXiv 编号核对，并在 JSON 中登记 SHA-256。以下均为**局部阅读**，未全文精读、复核全部实验或独立复现。分类对应所摘主张而非给整篇论文定永久等级。

| 来源 | 主张及范围 | K/L/S |
|---|---|---|
| [Conneau et al.](https://arxiv.org/abs/1805.01070) | 受控的十类句子表示 probing；依赖 BookCorpus 构造数据和所用编码器及探针。可解码不等于在原任务中被使用。 | Knowledge |
| [Tenney et al.](https://arxiv.org/abs/1905.05950) | 在 BERT 和 edge-probing 任务上，层权重的相对位置与部分经典 NLP 管线顺序相似；这不是对因果执行顺序的证明。 | Knowledge |
| [Liu et al.](https://arxiv.org/abs/1903.08855) | 在所选上下文表示、十七项英语探针及迁移任务上分析语言信息和可迁移性；不能自动扩张到别的语言。 | Knowledge |
| [Jain et al.](https://arxiv.org/abs/2005.00115) | FRESH 的预测器只接收被选片段；构造性忠实主要关乎这一预测器，不保证忠实解释原始全输入模型。 | Knowledge |
| [Krishna et al.](https://arxiv.org/abs/2202.01602) | 在四个数据集、六种解释方法及所选模型与指标中研究分歧，并结合访谈与用户研究；分歧本身不证明哪个解释错误。 | Knowledge |
| [Gevaert et al.](https://arxiv.org/abs/2202.12270) | 多种图像归因质量指标可能衡量不同性质，跨数据集结果也可能变化；结论限于所测八个数据集与方法。 | Knowledge |
| [Ancona et al.](https://arxiv.org/abs/1711.06104) | 梯度归因的统一表述与 Sensitivity-n；论文含条件等价陈述，目前仅读概述，假设和证明尚待核查，暂不把等价命题登记为 Science。 | Knowledge |
| [Murdoch et al.](https://arxiv.org/abs/1801.05453) | 标准 LSTM 的上下文分解可分析短语交互，实验为 Yelp/SST 情感任务；分解依赖其特定规则。 | Knowledge |
| [Bastings and Filippova](https://arxiv.org/abs/2010.05607) | 对模型开发者要寻找相关输入词这一目的，论文论证使用 saliency 方法；这是限定目标下的立场，不是所有 saliency 均优于 attention 的定理。 | Knowledge |

**尚缺的核查**：Ancona 等方法的形式等价假设及附录证明、FRESH 选择机制可能的标签信息泄漏、各归因方法在不同干预基线和质量指标下的具体失效反例，以及句子／层级探针的跨语言与因果介入证据。覆盖清单仍是抽样而非所有论文的穷举。


## 本轮补充：特征移除、归因定义与充分性（2026-10-28）

新增六篇论文及六个概念，均已获取作者原始公开 PDF，核对首页题名、arXiv 编号、PDF 页数和 SHA-256。每篇仅定位所列正文，未逐行审查证明或复现实验；它们均按所写的限定方法和分析暂记 Knowledge，论文中其他形式命题需要另行审读才能分类。

- Janzing 等区分按观测条件分布与按边缘分布移除特征（§3，式 (3)–(5)），并在特定模型解释语义下支持后者；这不是一般情况下哪种分布唯一正确的定理。
- Covert 等的 removal-based 框架用移除方式、被解释的模型行为和影响汇总方式组织 26 种方法（Figure 1）；其信息论联系依赖指定条件，不能将框架当作所有解释方法的穷尽。
- SAGE 以模型、损失和数据分布定义特征子集的预测力，再做 Shapley 汇总（§2.1）；缺失特征使用条件边缘化，所得全局归因不等于因果效应。
- SIS 的充分性相对选定的 mask、模型和决策阈值而言，文章以反向选择近似求解（§3）；不保证贪心结果全局最小，也不保证任意替换未选特征仍保持决策。
- Many Shapley Values 区分同一 Shapley 算子配合不同集合函数、基线或分布时的归因：对于给定博弈的唯一性不能决定该用哪一个博弈（§1–2.1）。
- Integrated Hessians 对积分梯度再次归因以刻画成对交互（§2）；分析依赖参考基线与可微性，对 ReLU 采用 SoftPlus 平滑时分析的是平滑后的函数。

## 本轮：统计解释、人类因素与模型类边界

新增 7 篇原文与 9 个概念，涵盖 ICE 与部分依赖曲线、ALE、Rashomon 集合与 model class reliance、内在可解释模型、房价预测中的人机评估、社会科学解释综述及结构因果对比解释。7 份 arXiv PDF 均已核验首页标题及 arXiv ID、页数、大小和 SHA-256，并定向查看每篇所列章节；未全文审读、逐项复现实验或审核所有证明。

三个容易混淆的范围：ICE/PDP/ALE 描述给定预测函数，不直接证明现实中的因果效应；PDP 和 ICE 可能涉及数据支持之外的特征组合，ALE 的条件分布构造避免该类外推问题不意味着自动识别因果关系。MCR 是给定模型类、参考损失和容差下近优模型的变量依赖范围，不是所有可能模型或数据生成机制的范围。Poursabzi-Sangdeh 等的用户研究以纽约房价及普通参与者为条件；透明度与人类表现的关系不能不加限定地推广。Rudin 的观点性文章和 Miller 的综述按 Knowledge（观点或综述）标注，不把规范性建议或跨领域综述当作已验证普适规律。


## 本轮：反事实、可行动性与再决策的因果边界（2026-10-28）

新增 9 篇论文、9 个概念：Actionable Recourse、DiCE、FACE、最小因果干预、因果结构不完全的概率 recourse、预测多重性、模型变化鲁棒性、执行噪声下概率失效率及相关综述。9 份 arXiv 原文 PDF 经 pdfinfo、首页题名和 arXiv ID、页数、大小与 SHA-256 核验。条目均标明实际局部阅读位置；未全文审读、复现实验或逐行审计所有证明。

- Ustun 等在固定线性分类器下用整数规划表示个人行动约束；DiCE 同时优化翻转、距离和多样性，其基础优化设定是静态可微二分类器。二者的形式化行动不自动保障现实可实施性。
- FACE 在数据加权图中寻找满足目标类与可行性条件的路径；其路径受数据覆盖、图和条件函数决定。高密度路径与干预后可达性不是同一个结论。
- Karimi 等区分特征向量的最邻近替代与结构因果模型内的最小干预。另文的 Example 1 构造观测与干预分布相同而个体反事实不同的两个模型，Proposition 2 明确写有后代集为空的例外，因而只将其作为**条件化 Science 反例**；没有逐行审查证明。
- Pawelczyk 等比较数据支持与稀疏反事实在预测多重性下的成本和可迁移性。数据支持是稀疏候选的附加约束时，前者最小距离不小于后者；作者其他边界定理尚未逐行核验，形式贡献列 Science 候选。
- ROAR 指定模型变化集合后优化最坏情形，而 PROBE 在固定模型下研究行动实施噪声的概率失效率；二者的不确定性对象不同，其假定的变化集合或噪声分布不必等于真实部署环境。

## 本轮：数据归因与 Shapley 假设和交互（2026-10-28）

新增 8 篇论文及 8 个概念，覆盖实例特征选择 L2X、因果 Shapley、条件与边际价值函数争议、Datamodels、TRAK、Faith-Shap、FastSHAP 和特征相关性下的条件 Shapley。8 份原始 PDF 均经签名、pdfinfo、首页题名与 arXiv ID、页数、大小及 SHA-256 核验，并定位所记载的方法或定理陈述；未完整审读所有证明或独立复现实验。

- L2X 使用变分互信息下界选择输入子集；信息性依赖数据分布、被解释模型、特征数及变分近似，不自动成为唯一因果解释。数据模型固定学习算法、目标输入和训练子集分布；TRAK 使用随机投影、核近似与线性数据模型分数，分数只衡量指定训练集修改的预测相关性。
- 条件、边际和因果 Shapley 定义了不同的价值函数；价值函数中的背景分布、干预与因果图决定所要回答的问题。相关特征下条件归因可能把贡献分给代理变量，而独立替换可能得到偏离数据支持的组合；这些是条件化讨论，不证明所有 Shapley 方法不可信。
- Faith-Shap 的 Science 主张限于论文定义的 Faith-Interaction 指数类：v3 Theorem 19 对四项交互公理给出唯一性，定理陈述已核对但证明未逐行审计；形式唯一性不保证现实因果交互。FastSHAP 摊销解释器训练，单次前向估计不排除训练成本、surrogate 误差或分布外误差。
- FastSHAP 原 PDF 为 52,298,264 字节，保存在项目忽略的 \`papers/\` 目录，不应加入普通 Git 追踪；若未来必须版本化请用外部存储或 Git LFS。


## 本轮：补足已有概念的范围合同（2026-10-28）

对 19 条此前仅写模板范围的核心概念，重新定位一手原文的指定章节并改写 scope 与证据：单神经元对齐、特征可视化、积分梯度、TCAV、探针和 selectivity、IOI 回路和 faithfulness、叠加、多义神经元、SAE 及其规模关系、activation/subspace patching、Hydra、ACDC、transcoder 和 ELK 反馈识别限制。范围分别写出模型、输入或数据、指标、干预及假设；ELK 定理注明训练外评价者可出错与训练策略无差别假设。这个更新是**局部原文核对**，不构成全文审读、逐行审计证明或独立复现实验。

随后对其余 17 条也定位对应原文或作者配套文章，补写 Othello 状态与相对坐标、steering、Tuned Lens、INLP、中介分析、定位与编辑、SHAP、DeepLIFT、Grad-CAM、SmoothGrad 等概念的条件与限制。原先 36 条完全相同的范围占位已清除；其中多模态神经元依赖作者配套文章的局部核对，Distill 交互正文并未全部取得。所有更新仍属局部核对，不等于理论证明完整审查或复现。

## 本轮：思维链忠实性与运行时监测（2026-10-28）

新增七篇原始论文和七个概念，涵盖思维链截断／注错测试、符号求解器、提示显露率、代理编码任务中的推理监测、推理必要性与规避、可读性／覆盖度评分，以及反事实任务对照。七份 arXiv PDF 已核验首页题名与编号、PDF 结构、页数、大小及 SHA-256；均仅定向核对指定章节，未复现实验。

- [Lanham 等](https://arxiv.org/abs/2307.13702) 对生成链做截断、注错等干预以观察答案变化；单项干预无法测出全部内部推理的忠实程度。
- [Lyu 等](https://arxiv.org/abs/2301.13379) 将问题翻译为符号步骤，再由确定性求解器输出答案。答案由程序计算不意味着自然语言注释、问题翻译或生成步骤均正确。
- [Chen 等](https://arxiv.org/abs/2505.05410) 通过有无提示的题目对估计提示使用，再统计链中是否显露提示；这只测所定义提示与判定标准，不代表对所有隐含原因的完全观察。
- [Baker 等](https://arxiv.org/abs/2503.11926) 在所测代理编码环境中比较推理链监测与仅看动作／输出，并报告强优化监测奖励时出现意图隐藏；其效果受任务、监测器、模型和训练协议限制。
- [Emmons 等](https://arxiv.org/abs/2507.05246) 区分事后解释和完成行为所需的文本计算，研究提高任务难度及有辅助的规避；相应必要性和监测效果并非对所有危险行为的普遍定理。
- [Emmons 等](https://arxiv.org/abs/2510.23966) 用自动评分器评估文本可读性与覆盖度。作者明确将它作为代理指标，与针对性规避测试互补，而非替代人类判断。
- [Wu 等](https://arxiv.org/abs/2307.02477) 用 11 类规则变化的反事实任务评估模型迁移；表现下降本身不能区分记忆、计算不足及多种其他机制。

上述条目按具体方法和实验记 Knowledge，不将论文提出的评价框架、动机假设或局部经验差异包装成普适 Law 或第一性 Science。

## 本轮：图神经网络解释、评价与鲁棒性（2026-10-28）

新增 9 篇论文和 9 个概念，覆盖 GNNExplainer、PGExplainer、GraphLIME、SubgraphX、CF-GNNExplainer、PGM-Explainer、ZORRO、GraphFramEx 和预测保持攻击 GXAttack。9 份 arXiv PDF 均核验签名、pdfinfo、首页题名及 arXiv ID、页数、字节数和 SHA-256；仅定向查看条目所列方法或定理，未完整审读证明和实验、也未独立复现。

- GNNExplainer 对指定预测优化图及特征掩码；PGExplainer 将边分布生成参数化并在实例间共享。GraphLIME 在节点的局部邻域选择特征，SubgraphX 使用 MCTS 和近似 Shapley 搜索子图；其输出均受模型、候选集、扰动和评估规则限制。
- CF-GNNExplainer 搜索可使所测预测改变的删边集合；局部优化所获解释不保证全局最少删边，也不保证现实干预可行。
- GraphFramEx 将指定模型预测或现象标签下的充分／必要 fidelity 分开；这些分数不是现实世界因果充分／必要性的自动证明。ZORRO 关注未选输入被扰动时的解释稳定性，结论依赖指定扰动分布。
- PGM-Explainer v1 Theorem 1 假设总体分布存在完美图，并在其定义的成对依赖集合上给出包含 Markov blanket 的结果；将该特定理论标为条件化 Science 候选，附录证明未逐行审计。该结论不直接保证有限样本统计检验、因果方向或真实机制。
- GXAttack 在所测白盒和图结构修改预算内构造预测保持而解释变化的例子；不把特定攻击成功外推为一切图解释都不可信。

## 本轮：稀疏字典方法、评估及非唯一性（2026-10-28）

新增 8 篇论文与 8 个概念，覆盖 Gated SAE、JumpReLU SAE、BatchTopK、Matryoshka SAE、任务监督字典评价、SAE stitching／meta-SAE、SAGE 评价框架与 SAEBench 指标可靠性审计。8 份原文 PDF 的文件签名、首页题名与 arXiv ID、页数、大小及 SHA-256 已核验；只阅读所列方法、图表和设定，未全文审读、逐行审计证明或独立复现。最后一篇 2026 年研究是重要的近期评价对照，不称其已成为经典。

- Gated SAE 把门控与幅值估计分开，JumpReLU 用阈值和直通估计器训练稀疏目标；其重建与解释性表现受模型激活、层、训练预算和指标限制。
- BatchTopK 训练时按批选择 \(n k\) 个激活，个体样本激活数可不同；其推理以估计阈值替代批级选择，不把训练期的固定 batch 总数误称推理期每样本固定 \(k\)。
- Matryoshka 用嵌套字典前缀的多重重建改善所测任务中的层级特征与吸收指标。Leask 等用不同宽度字典的 stitching 与 meta-SAE 检查缺失与可分解性；两项工作均不构成对所有 SAE 表征唯一性或原子性的普遍定理。
- Makelov 等的监督字典和 SAGE 的任务属性截面均为**任务相关、近似**参照，不等于模型所有机制的无条件 ground truth。Chanin 的 SAEBench 审计在固定 SAE 重复随机种子、合成字典与训练快照下揭示评价噪声和可区分性问题，范围受模型、指标、top-N 与样本设计限定。

每篇按具体方法、观察或评估记为 Knowledge；文中可能存在其他理论主张，但其假设及证明未单独审计，不将其自动标成 Science。当前本方向共 **212 条论文／原生文章记录、179 条概念记录**；最终项目统计以主线程重建的审计为准。

## 本轮：复核初筛与摘要级解释性概念（2026-10-28）

对 20 条原先标记为「初筛」或「摘要级」的概念，分别回到各自一手原文的具体方法、定义或实验段落，补充 scope、原文定位及审阅等级；未将关联论文的已读状态替代逐篇核查。此次未新增论文或 PDF。这是局部阅读，不表示阅读全文、逐行审计证明或独立复现；论文条目原本的审阅状态保持不变。

- 表征比较中，线性 CKA 对正交变换与各向同性缩放不变，但不对任意可逆线性变换不变；RBF CKA 的尺度不变还依赖带宽选择。SVCCA 的 SVD 截断依赖所保留子空间，不应把 CCA 的一般不变性无条件传给整个方法。
- Representer points 的原定理限定末层线性结构、正的末层 L2 正则和目标驻点，给出预测的训练样本项分解；TracIn 先定义沿训练轨迹的逐步损失变化，再引入一阶梯度及 checkpoint 近似。两者都不能直接等同于删点重训效应。
- 复制抑制的 76.9% 是 GPT-2 Small 的 L10H7 在作者定义指标下可解释的头效应比例，不是模型准确率；回路重用的约 78% 是 GPT-2 Medium 的 IOI 与 Colored Objects 中所定义的回路内注意力头重叠，不是普遍跨任务保证。
- 思维链忠实性、可视化调试、ITI 和特征稳定性均写明所测对象或操作；反事实输出的最小距离依赖所选距离与约束，本身不能证明现实可行动性或因果效应。现有归类不变，仍需按具体主张而不是按论文整体判断 Knowledge、Law 和 Science。

剩余限制：本轮未补查各篇全部实验与证明，不对新的预印本宣称经典地位；其他已标为局部审读的概念、论文条目与跨领域统一分类仍需进一步覆盖。最终总数以主线程汇总审计为准。

## 本轮：视觉决策线索、捷径及对抗特征（2026-10-28）

新增 8 篇论文和 8 个概念，补充模型究竟依赖哪些视觉证据及其测量边界。8 份 arXiv 原文 PDF 已核对文件签名、pdfinfo、首页题名与编号，并记录页数、文件大小、SHA-256 与版本。仅定向阅读记录中所列章节／图表；没有逐行审计证明或独立复现实验。

- Geirhos 等以风格迁移制造形状与纹理类别相冲突的图，比较 ImageNet CNN 与人类的选择；Burgert 等的近期反驳采用线索抑制，其变换也会影响其他线索。两者测量不同，不能单凭任一标题判定 CNN 天生依赖或不依赖纹理。后者是有用的近期对照，不标为经典定论。
- Lapuschkin 等展示 PASCAL VOC 中 Fisher-vector 马分类器对来源标签的依赖，并使用删除／添加标签的局部干预；所示微调 DNN 在相同删除示例中与该分类器不同。BagNet 说明局部 patch 分类证据聚合可在 ImageNet 有效，但不能由此断言普通 CNN 与其内部计算相同。
- Shortcut learning 是有关跨测试条件失败的概念框架；Xiao 等的前景／背景合成实验提供受控设置下的背景敏感性证据。具体替换会改变输入分布，因而 scope 必须包含构造方式。
- Ilyas 等将有用但非鲁棒的特征相对于指定分布和扰动集合形式化，并构造特定训练数据支持其实证假说。Li 等考察跨监督、自监督和生成范式的迁移，采用更严格的“跨范式有用性”定义；其结果不否认原先监督任务中可预测性的观察。

上述条目按各自写出的实验、方法或概念归为 Knowledge；文献中其他理论结果不因论文整体被归类而自动被判定。当前本方向共有 **220 条论文／原生文章记录、187 条概念记录**；全项目最终统计以主线程统一重建审计为准。

## 本轮：参考文献抽查发现的 11 项缺口（2026-10-28）

本轮补入抽查队列中 11 个原始来源和 11 条对应概念：10 份 PDF 已通过 pdfinfo、首页题名与 arXiv ID、版本、大小及哈希核验；OpenAI 2023 文章保存原生 HTML、含正文的 JS bundle 及其模板 JS，正文通过 bundle 中 Methods 和 Limitations 定向核对，其他资产与完整离线交互尚未检查。审阅深度均见 JSON；取得原文并非全文审读、实验复现或证明审计。

覆盖 CCE、线性分类探针、神经加性模型、神经元因果干预、BERT 语料解释错觉、自然图像比较、自动神经元解释、端到端字典及两篇综述。观察到自然图像在论文指定二选一任务中的优势，不代表一切视觉解释任务；e2e 字典作者明确指出输出一致仍可能通过不同下游路径得到。

**需重点保留的理论审计问题。** Adler–Shavit v3 的 Theorem 4.1 / A.1（PDF 第9、13页）声称，在允许一般变长比特串的参数驱动模型里，几乎所有函数需要至少 log₂|F| 比特；其附录证明将“不能把所有 k-bit 消息都编码为少于 k bit”误推为“短于 k bit 的只能是可忽略比例”。原文 §2.1（PDF 第5页）规定参数为 {0,1}*，未给出前缀自由或自定界要求；T 对该有限函数族统一，不要求在所有不同函数族之间共用。令 |F|=2^k，U={0}，V={0,…,2^k−1}，F 为 2^k 个输出不同元素的常值函数，给 2^k−1 个函数分配长度 0 到 k−1 的全部比特串，剩下一个用长度 k 的串；固定 T 根据码字的长度字典序编号输出对应常值。这满足 §2.1 的可变长度参数模型，却使短于 k 位的比例趋近 1，反驳该精确量词。标准计数可给出至少一个码长 ≥k，以及对任意正整数 c，少于 k−c 位的至多占约 2^(−c)；这不验证或否定论文其他理论结果及其构造。该项为 Science 类型的**有明确反例的理论主张**，不是经审计的有效定理；不外推至真实训练模型。

当前可解释性方向 231 条论文／作者原生文章与 198 个概念，最终总数以项目统一审计为准。

## 本轮：综述引文追踪——功能模块、解释评估与解耦（2026-10-28）

沿 Sharkey 等《Open Problems in Mechanistic Interpretability》v1 参考文献 PDF 第 40–44 页，选取与可解释性直接相关、在本轮开始时三个领域论文目录均未按标题和 arXiv ID 命中的 12 项，补入 12 篇原文与 12 个概念。本次为目的性选样，不是对文献表的穷尽、随机抽样或领域覆盖率估计。12 个 arXiv PDF 已检查 PDF 签名、pdfinfo、首页题名和 arXiv 版本，保存大小及 SHA-256；只定向阅读所列方法／图表，未全文审读、逐行审计证明或独立复现。各分类只覆盖条目指定主张。

- LIB 从层间 Jacobian 构造局部交互基与积分梯度图；作者报告它在所测小型语言模型上没有显著可解释性改善。功能权重掩码、梯度路由、DiffMask 分别对已有权重、训练梯度和输入选择施加掩码，不能把它们的干预对象混为一谈。
- SNAFUE 自动检索用于复制粘贴攻击的自然图像 patch；另一工作以植入木马重发现和用户判断来评价解释方法。已知触发器的测试提供可控验证，但不等于一切自然错误的真值。可视化反审计的正常／异常模型对照同样受异常构造和评估规则约束。
- 语言神经元相关性分析、静态嵌入空间参数投影，以及模型审计的访问层级均按具体方法／评价框架记为 Knowledge。后者是论证性综述，不证明单凭白箱访问就能完成充分审计。
- 对抗输入可在模型输出近似不变的情况下操纵所测解释图；论文几何界的定理及证明未在本轮独立审计，所以只记录实证／方法主张，不泛化为任意模型的结论。
- β-TCVAE 工作在因子化先验和指定聚合后验下给出 ELBO 的条件化 KL 分解；形式分解列 Science 类型，算法和经验实验列 Knowledge。只核对了公式陈述，未逐行审计证明；降低总相关性不保证无监督语义解耦。

本方向目前 243 篇论文／原生文章、210 个概念，最终项目统计以主线程汇总审计为准。

## 本轮：四项 Science 主张定向质量复查（2026-10-28）

依次核对四篇原始 PDF 指定定理／反例、scope 与索引；这是定向质检，不是完整论文审阅或实验复现。

1. **Kornblith 等，CKA**：v4 PDF 第 2 页 Theorem 1 与第 12 页 Appendix A。仅要求相似度对**第一参数**的可逆线性变换不变，X、Y 都为同形 n×p 满行秩矩阵，从而对相容的 Z 得到相同相似度。已逐步核对原证明中补全矩阵及构造可逆变换的步骤，修订条目明确量词；中心化数据的秩最多 n−1，不能直接套用该满行秩命题。其他附录和实验未审核。
2. **Koh–Liang，影响函数**：PDF 第 2 页 §2.1 式 (1)–(2)。在原文给出的经验风险可微、严格凸与最优点处 Hessian 正定的条件下，公式是 ε=0 的局部导数；将 ε 设为 −1/n 预测有限删点属于一阶近似。已修正概念条目，避免把严格局部导数与有限删除效应混同；附录证明与非凸网络实验未独立审核。
3. **Lundberg–Lee，SHAP**：PDF 第 4–5 页 §3–4、Theorem 1、式 (5)–(8)。唯一性针对给定简化输入映射的加性解释模型，及局部准确性、缺失性、一致性；条件期望／缺失特征建模选择不由唯一性确定。现有条目没有发现须修正的范围错误；底层合作博弈论证明及实证未独立审核。
4. **Locatello 等，解缠不可辨识**：v4 PDF 第 3 页 §3、Theorem 1 与脚注 3。原文假设 d>1、潜变量先验具有因子分解密度，构造保分布的纠缠双射并讨论同观测分布的生成模型；不能据此否认附加监督或结构假设下的可辨识性。现有条目保留了这些前提，未发现须修正的范围错误；附录证明与实验未逐行审计。

## 本轮：综述引文追踪——多维特征、稀疏探针与性能证书（2026-10-28）

沿 Sharkey 等《Open Problems in Mechanistic Interpretability》v1 参考文献 PDF 第 45–49 页继续目的性抽查，补入 10 篇此前三个领域目录均未按标题／来源编号命中的直接相关原始论文，并配套 10 条概念记录。这是参考文献追踪的一个区间，不是穷尽清单或领域覆盖率。8 份 arXiv、2 份 ACL Anthology 原文 PDF 均核验 PDF 文件签名、pdfinfo、首页题名／编号、版本、页数、字节数和 SHA-256；阅读仅涵盖 JSON 明列章节／公式／图表，未全文审读、独立复现或完成证明审计。经典源头与较近期关键工作并列存档，不把所有 2023–2025 年结果都宣称为经典。

- **版本与表征范围。** [Engels 等的多维特征论文](https://arxiv.org/abs/2405.14860) 当前下载为 v3，题名明确为 *Not All Language Model Features Are One-Dimensionally Linear*；应使用这个首页题名，而非综述所列旧题名。圆形特征仍位于低维线性子空间，不能把该工作写成否定一切线性表征。星期／月份干预包含 PCA、圆形探针及其他方向的均值消融，作用范围应连同操作一起说明。
- **SAE 残差。** [Dark Matter](https://arxiv.org/abs/2410.14670) v2 对 Gemma／Llama 的具体 SAE 和数据考察线性可预测残差；部分层的剩余 FVU 在已测宽度范围近似不变，进一步外推到无限宽度是作者明确使用的假设。条目只列 Knowledge，没有据幂律图自行升级为 Law。
- **可读出与实际使用。** [Ettinger 等](https://aclanthology.org/W16-2524/) 的早期组合语义诊断与 [Gurnee 等](https://arxiv.org/abs/2305.01610) 的 k 稀疏探针分别补齐任务控制与神经元定位线索。两者都要求关注负样本和捷径；读出成功不是内部因果机制的自动证据。[几何因果探针](https://arxiv.org/abs/2307.15054) 提出反事实信息指标及额外潜概念因果假设，不能把因果图设定当作观测已经确认的事实；其附录互信息论证留待独立核查。
- **归因对象。** [Neuron Shapley](https://arxiv.org/abs/2002.09815) 固定联盟价值函数和均值消融、不重训，近似算法包括提前截断；所得责任依赖该游戏。[Grosse 等](https://arxiv.org/abs/2308.03296) 只对所测预训练模型的 MLP 参数估计影响，目标与 PBRF、阻尼曲率及 EK-FAC 近似相连，不等于真实删点重训或后训练助手回答原因。两篇的方法／实验均按 Knowledge 记录，未审计理论部分不自动升为 Science。
- **Science 的条件。** [ISAN](https://arxiv.org/abs/1611.09434) 在输入切换仿射递推及线性 logit 读出内提供精确展开，固定序列下的加和不代表整个网络对离散输入线性，也不直接给 token 删除的因果效应。[Compact Proofs](https://arxiv.org/abs/2406.11779) v14 提供指定 Max-of-K 模型的下界证明方案；正文注明只给正确性论证 Q，未附计算 C 的运行 trace。条目将这些特定条件化理论与经验观察分开，并明确附录正确性、浮点处理和代码未独立审计。
- **解释的失败条件。** [Feng 等](https://aclanthology.org/D18-1407/) 的输入缩减工作在固定模型和数据上观察到短而不可理解的输入仍保持预测及高置信；它限制“预测保持即可证明解释充分”的说法，不是对所有解释方法的否定。

**未解决与下一步。** 第 45–49 页只选取与当前解释性主线直接相关的缺口，尚未逐条处置其余参考文献；候选包括 Erhan 等 2009 可视化技术报告、Mamba 的 IOI 电路、SAE 辅助遗忘、无需生成 token 的知识估计、深层表征与早期词向量属性分析。第 50 页及其后参考文献尚未由本轮追踪。近期条目的经典地位、全文论证和实验可复现性都不在本轮核验范围，不能用当前数量宣称“所有 paper 已覆盖”。

本方向现为 **253 篇论文／原生文章、220 个概念**；最终全项目总数以主线程统一重建审计为准。

## 本轮：综述引文追踪——表征源头、诊断探针与解释检验（2026-10-28）

继续沿 Sharkey 等《Open Problems in Mechanistic Interpretability》v1 参考文献 PDF 第 50–55 页追踪，先与三领域目录按标准化题名、来源 URL 和 arXiv ID 去重，再补入 **12 篇原文与 12 个概念**。选择侧重基础工作及测量边界：1981 年形状表征、2012 年无标签视觉特征、2015 年多语言词嵌入诊断、2018 年诊断分类器，以及 OSDT、置换外推、解释操纵、可证伪框架、MILAN、神经元解释评价、Backpack 和 Backward Lens。仍是目的性选样，不是对这六页全部条目或整个领域的穷尽覆盖。

12 份原文 PDF 均检查文件签名、pdfinfo、页数、大小及 SHA-256；电子原文核对首页题名与编号，Hinton 作者网站扫描本以首页目检和 IJCAI 官方书目确认身份。仅阅读 JSON 指定段落；该扫描本额外用 OCR 核对 PDF 第 3、6 页，没有把扫描文本不完整写成全文已读。文件总大小 56,499,776 字节，最大单文件为 MILAN 的 36,681,711 字节，均位于已有忽略规则覆盖的 papers/。

**版本与引文处理。** Hupkes 在综述里是 IJCAI 2018 的扩展摘要，本轮沿溯至同题 JAIR 正式长文 arXiv 1711.10203v2（含 Sara Veldhoen 的三作者版本），只计一个原始工作。Hernandez 下载题名是 *Natural Language Descriptions of Deep Visual Features*，保留其中 Visual。OSDT 下载为 1904.12847v6，明确记录其 2023 修订版而不冒充 2019 原始版本。综述第 52 页 Jain 条目中误混有 Backward Lens 书目内容，而第 53 页又有正式条目；TCAV 的 2018a／2018b 也重复出现，均不重复计入。此前已有的 RAVEL、AtP*、Concept Bottleneck、Tracr、Convergent Learning 和 Chinchilla 回路研究没有重加。

- **观察与选择。** Le 等的无标签视觉特征证据受模型和评估选择约束：81.7% 是最佳神经元及 20 个候选阈值的所报结果，不能称为预先固定分类器的独立盲测。Köhn 的静态词向量诊断和 Hupkes 的线性诊断分类器测量可读出信息，不能仅靠解码准确率唯一确认网络实际执行的算法。
- **干预究竟改变什么。** Heo 等直接微调参数来操纵解释图，保留分类损失不保证函数处处不变；Hooker 等区分固定模型审计与真实关系推断，相关输入的无条件置换可能迫使模型外推。Huang 等把自然语言解释分成观察预测和指定任务的 interchange intervention 检验，未找到任务内中介作用不等于排除所有任务的作用。
- **语言描述与可控接口。** MILAN 实际使用有限 exemplar、近似描述模型和加权 PMI 搜索；作者明确把跨模型 BERTScore 评价视为 sanity check，而非完整可靠性证明。Backpack 的固定 sense 向量贡献有结构化接口，但上下文权重网络仍是非线性的，额外参数和计算不能略去。
- **条件化 Science。** OSDT 的界针对有限二值输入、二分类及“训练误分类率＋叶子正则”的搜索空间，不等于真实分布或因果意义的最优；只核对了指定界的陈述，完整证明与代码未审计。Backward Lens 的外积和给出指定线性层梯度 rank≤n 的上界，本轮核对了该代数步骤；它不自动适用于批次／多步累积或优化器变换后的更新，也不自动验证词表投影的语义和后续机制假说。
- **方法论不是已证明的普适边界。** Hinton 的形状／属性绑定方案和 Leavitt–Morcos 的可证伪研究框架，按其具体概念／方法论主张记录为 Knowledge；历史影响或标题中的科学诉求不构成自动晋级。

**本区间尚未处置。** 仍有直接相关候选尚未补入并逐篇审读，包括原生文章 *Understanding RL Vision*、线性情感表征、知识遗忘痕迹、ICL developmental landscape、InversionView、程序任务微调机制、Modular Networks、attention-output SAE、Gemma Scope、DPO 与 toxicity 的机制研究，以及若干 python-docstring／多义注意力头／SAE 数据依赖原生文章。其余医学应用、一般架构、风险讨论和法律文献未因出现在综述里就自动纳入；本轮没有逐条完成全部引用的纳入／排除裁决。第 56 页及之后参考文献不属于本轮追踪范围。

本方向现为 **265 篇论文／原生文章、232 个概念**；最终全项目数量以主线程统一审计为准。


## 本轮：综述引文追踪——表示源头、可视化与测量边界（2026-10-28）

沿 Sharkey 等《Open Problems in Mechanistic Interpretability》v1 参考文献 PDF 第 56–61 页继续目的性追踪。本轮按题名、arXiv ID 与来源 URL 同三个领域的当前目录去重，补入 **12 篇原文与 12 个配套概念**；另经同义概念检查，为既有 **LIME、ROME、结构探针**补入 **3 个独立概念及其原始来源／局部定位**。后 3 项不是新论文，没有重复下载或改变原有论文对象。新增后本方向为 **277 篇论文／原生文章、247 个概念**，最终全项目统计以统一重建审计为准。

新增来源为 Skeletonization（1988）、k-Sparse Autoencoders、Linguistic Regularities、表示反演、DGN-AM、Multifaceted Feature Visualization、L0 门控稀疏化、Dissecting Contextual Word Embeddings、Compositional Explanations、General Pitfalls、Circuit Faithfulness Metrics、Missed Causes and Ambiguous Effects。12 份 PDF 均检查文件签名、pdfinfo、首页题名／来源身份、实际版本、页数、字节数及 SHA-256；合计 **42,649,832 字节**，最大单文件 **12,825,142 字节**，均在已有 papers/ 忽略规则下。逐项实际阅读位置记录于 JSON；不是全文精读、实验复现或完整证明审计。

**版本及重复引文。** Miller 等在综述所列会议题名为 *Transformer Circuit Evaluation Metrics Are Not Robust*，本次实际下载为 **2407.08734v1**，首页题名仍为 *Transformer Circuit Faithfulness Metrics Are Not Robust*。按下载文件身份保存，没有将其冒称题名已改的最终会议稿；后续若需要该正式版本，应另取文件并对照。Nguyen 等的 Multifaceted 在参考文献中出现 2016b／2016c 两项，本轮只计一个工作；Mu–Andreas 采用 arXiv v2 长文，附录未完整审读。较近期测量边界工作作为关键原文纳入，不自动称作经典。

- **删除与稀疏化。** Skeletonization 将训练误差的有限删点差异用门控导数近似；一阶近似不是删除无害的保证。k-Sparse Autoencoder 的训练截断 k 与下游 αk 编码不同，不应与现代 SAE 的语义解释主张混同。Hard-concrete 门控下的期望非零参数计数关系按限定形式列 Science，稀疏化方法列 Knowledge；CDF、可重参数化及门控支持条件写清楚，完整附录推导和全局优化保证未审。
- **图像反演与偏好刺激。** 表示反演通常不唯一，图像先验会改变重构。DGN-AM 搜索被生成器表达空间限制，作者按可辨识图像质量选择设置且未观察到活动值与可辨识性相关；本轮未审其 §3.5 的先验控制实验。Multifaceted 的初始化来自有限样本聚类，不能把多个可视化面向当成所有响应类型的完整枚举。
- **表征中的规律。** Mikolov 等的向量偏移是特定词向量和类比集上的受测假说；Peters 等在 LSTM、Transformer、CNN 中观察到探针任务随层变化，跨架构证据仍未解除语言、语料、模型与解码器约束，因此没有仅凭三种架构提升为 Law。Mu–Andreas 的逻辑概念解释受概念库、阈值、最大长度和束搜索约束，IoU 上升不等于唯一或因果解释。
- **测量对象。** Molnar 等明确区分模型使用哪些特征、特征的泛化贡献以及数据生成规律；条件化解释可能更换估计对象。Miller 等给出消融六元组，IOI 中 node／edge、均值／重采样、token 位置及指标聚合都影响分数，超出 0–100% 的恢复率不应当概率。Mueller 讨论局部反事实依赖的非传递与冗余原因漏检，并给出 Pythia 70M 中间目标案例；引用的 Halpern 原始定理证明未审，不能把方法论论证或发生频率猜想写成已核验的普适边界。

**既有论文概念补链。** LIME 复核 v3 PDF 第 3–4 页的表示、局部核、K-LASSO 与局部非线性失败说明；ROME 复核 NeurIPS 2022 PDF 第 3–5 页的因果追踪、关联记忆假说与秩一编辑，并与目录中的定位／编辑分离工作配对，未审 Appendix A；结构探针复核 NAACL 2019 PDF 第 2–3 页的树距离、深度、半正定／伪度量限定及 PTB 设置。新增概念与 GraphLIME、MEMIT、probe selectivity 等既有条目分开，避免把后继方法来源代替经典来源。

**本区间仍待处置。** 尚未逐项审读并裁决的直接相关候选包括 Mante 等的前额叶循环动力学、AlphaZero 棋类知识获取、稀疏 crosscoder 原生文章、Seeing Is Believing 的模块训练、程序合成辅助解释、依赖特征的条件子群方法、*The Quest for the Right Mediator*、*From Insights to Actions*、Plug and Play Generative Networks、Early Vision in InceptionV1、Transformer 解释向 RNN 的迁移，以及若干 SAE／logit lens／attribution patching 原生文章。KAN、一般剪枝、科学假说生成、遗忘、法律／风险或医学应用文献没有仅因出现在参考文献里就自动纳入。本轮不是六页参考文献的穷尽纳入／排除表，也不估计领域覆盖率；第 62 页及之后不属本轮范围。


## 本轮：反向来源队列的经典概念补链（2026-10-28）

本轮处理 `concept_coverage_gaps.md` 中 interp 的 38 条书目未关联记录。先检查现有 247 个概念的名称、定义与原始来源，再定向核对本地原文，解决 **18 篇既有论文／原生文章的显式概念关联**：修改 **5 个已有概念**，增加 **11 个独立概念**，**没有新增论文或下载新版本**。当前本方向 **277 篇、258 个概念**；用当前构建器的 URL／arXiv 关联规则在内存中核查，interp 未关联队列由 **38 降至 20**。统一生成文件由主线程重建，本轮不写面板、论文或构建器。

**已关联的具体队列项。**

| 原始来源 | 处理 |
|---|---|
| Understanding Neural Networks Through Deep Visualization | 复用“特征可视化”，增加 1506.06579v1 第 3–4、6 页证据与输入优化／实时显示范围。 |
| Zoom In: An Introduction to Circuits | 复用“电路 / circuit”，增加原生文章 Claim 2、曲线电路与 Claim 3 的来源、假说限定和忽略中间 ReLU 的细节。 |
| Towards Monosemanticity | 复用“稀疏自编码器 SAE”，增加单层 MLP 试验、损失恢复率与完整性疑问的原生段落。 |
| Scaling Monosemanticity | 复用“稀疏自编码器 SAE”，增加已存 2605.29358v1 第 4–6 页的中层残差、约 1M／4M／34M 字典和代理损失范围。 |
| How to use and interpret activation patching | 复用“Activation patching”，增加 2404.15255v1 第 2–3 页的探索／确认、粒度及 noising／denoising 区分。 |
| Attribution Patching Outperforms Automated Circuit Discovery | 复用“Attribution patching 与假阴性”，增加 BlackboxNLP 2024 正式 PDF 第 2–3 页 EAP 的近似、top-k 排序、零梯度和比较例外，未混为 AtP* 定理。 |
| Visualizing and Understanding Convolutional Networks | 新建 Deconvnet 反投影与灰块遮挡概念；v3 第 2–3 页方法及第 7 页 Figure 7。 |
| Deep Inside Convolutional Networks | 新建输入梯度显著图概念；v2 第 2 页 §2–3 的局部 Taylor 近似与类别分数／概率区别。 |
| Progress measures for grokking | 新建模加法傅里叶机制与 restricted／excluded loss 概念；v3 第 3–4、6、8 页，按局部机制列 Knowledge。 |
| A Toy Model of Universality | 新建有限群 GCR 概念；v2 第 3–4 页区分有限群精确表示的条件代数与训练所得回路观察，原附录证明未逐行审计。 |
| Sparse Feature Circuits | 新建 SAE 特征与误差节点电路概念；ICLR 2025 正式 PDF 第 3–4 页，区分 AtP／IG、阈值与聚合。 |
| Finding Transformer Circuits with Edge Pruning | 新建可微边掩码概念；2406.16778v3 第 3–4 页，保留 clean/corrupt 混合、稀疏约束和优化界限。 |
| A Mathematical Framework for Transformer Circuits | 新建 QK／OV 分解与固定注意力路径概念；核对四个原生方法段落，形式结论限定为简化架构，不声称全部真实模型边界已知。 |
| Circuit Tracing | 新建跨层 transcoder 局部替代／归因图概念，核对 Architecture、Local Replacement Model、机制忠实性及选定附录段落。 |
| On the Biology of a Large Language Model | 关联到同一 CLT 方法概念，但以独立来源注释记录 Dallas→Texas→Austin 案例、干预验证和选择偏差；未替代其他机制案例的独立概念。 |
| Language Models Represent Space and Time | 新建时空坐标线性读出概念；ICLR 2024 正式 PDF 第 2–3 页的数据、ridge 目标与指标。 |
| The Geometry of Truth | 新建真假陈述线性分离与方向错位概念；v3 第 3–6 页的模板数据、patching、PCA 与否定方向错位。 |
| Discovering Latent Knowledge Without Supervision | 新建 CCS 对比一致性概念；v2 第 3–4 页的两个损失、方向歧义及实验设置。 |

5 个复用概念保留原 `source`，并增加 `sources` 与逐来源 `source_annotations`；新增证据及范围也写进通常展示的 `evidence`／`scope`。CLT 方法文与 Biology 应用文用两条来源注释分开。所有 277 个论文对象、其他 242 个旧概念对象及 18 个本地原文件的哈希均保持不变。只核对上述页码和原生段落，不声称重新全文审读；原生 HTML 的图片、脚本和交互资产完整性不由文本核对保证。Zeiler v3 第 7 页文本提取出现 xref 修复提示后成功返回图注；没有据此修改原 PDF，也没有把文本提取当作图像目检。

**新增的判据边界。** CCS 的无标签损失不能确定哪一个簇叫“真”；正文评估取两种标签方向中较高的测试准确率，附录无监督消歧方案本轮未审。归因图的局部替代模型在原提示上输出一致，不保证扰动机制一致；Biology 的 constrained patching 中干预层以前部分直接效应由构造保证，不能全部当成独立验证。SAE 损失恢复和代理目标不等于完整解释比例；Templeton 当前登记的是 2026-05-28 的 arXiv v1，不冒称原生文章最初发布时的 PDF。GCR 的 Science 限于有限群与精确表示；QK／OV 的 Science 限于给定架构简化和固定注意力的代数关系，完整附录证明未审。其余新增观察／方法未因经典地位、规模或标题被提升为 Law。

**仍未关联的 20 项。** BERTology、BERT attention、Probing Classifiers、Rigorous Science、Faithful NLP Survey、Open Problems、Social Sciences explanation、RAVEL、Representation Engineering、Activation Engineering、TruthfulQA、Word2Vec-style Arithmetic、Attention is not not Explanation、FFN vocabulary promotion、Sixteen Heads、Robustness、Chinchilla circuits、Large Multi-modal Models feature interpretation、Prisma、SAE-V。

其中 BERT attention 与“Attention 权重与解释”、Activation Engineering 与“Activation steering”、Sixteen Heads 与“多头剪枝与功能特化”、FFN vocabulary promotion 与键值记忆／词表投影、Robustness 与“解释脆弱性”只是**同义或相邻概念候选**：本轮没有定向重读这些原文，未添加 URL 来假装完成审阅，也未裁定应合并还是另建。其余综述、评价框架、benchmark 与多模态方法仍待分别审查。20 是书目未关联数量，不是 20 个已确认的语义缺漏，更不是领域覆盖率。


## 本轮：逐项处置剩余 20 条反向来源队列（2026-10-28）

本轮审视上一轮留下的 20 篇既有论文，核对本地文件签名、首页身份／实际版本和 SHA-256，并定向读取下表所列原文位置；**没有新增论文、下载新版本或重写论文对象**。17 篇完成显式来源关联：**8 个既有概念补链、9 个独立概念新增**。本方向由 **277 篇／258 个概念**变为 **277 篇／267 个概念**；另 250 个旧概念完全不变。按当前构建器的 URL／arXiv 规则在内存核对，未关联数由 **20 降到 3**。这 3 篇已逐项给出处置理由，不能把书目链接数直接称为语义缺漏数或领域覆盖率。

### 已补链的 17 篇

| 既有论文／版本 | 处置 | 本轮实际阅读位置 |
|---|---|---|
| Towards A Rigorous Science of Interpretable Machine Learning；1702.08608v2 | 新建解释性评价三类；这是评价设计，不因题名中的 Science 晋级。 | PDF pp. 2–5，Figure 1、§1–3.3 |
| What Does BERT Look At? An Analysis of BERT's Attention；1906.04341v1 | 新建 BERT 注意力矩阵的语言关系探针；与注意力作为解释及删头重要性分开。 | PDF pp. 3–5，§3–4.2、Figures 3–5 文字／图注 |
| Probing Classifiers: Promises, Shortcomings, and Advances；2102.12452v4 | 复用线性 probe；补充探针模型、原模型与两套数据的联合依赖。 | PDF pp. 2–3，§2、Figure 1 |
| Representation Engineering: A Top-Down Approach to AI Transparency；2310.01405v4 | 新建 RepE/LAT 读出与控制；明确相关、操纵、终止、恢复是不同证据。 | PDF pp. 8–12，§3.1–3.2.1、Algorithm 1 |
| Steering Language Models With Activation Engineering；2308.10248v5 | 复用 Activation steering；单独记录 ActAdd 的提示对、层、强度及对齐。 | PDF pp. 3–4，Table 2、Figure 1、§3 Algorithm 1 |
| TruthfulQA: Measuring How Models Mimic Human Falsehoods；2109.07958v2 | 新建 TruthfulQA 输出真实性／信息量评价；不与真假表征探针合并。 | PDF pp. 2–5，§1.1、§2–3.2 |
| RAVEL: Evaluating Interpretability Methods on Disentangling Language Model Representations；ACL 2024 正式 PDF，pp. 8669–8687 | 新建 RAVEL 的 Cause／Isolation；换目标属性与保留其他属性分开评分。 | PDF pp. 3–4，印刷 pp. 8671–8672，§2.1–2.2 |
| Towards Faithful Model Explanation in NLP: A Survey；Computational Linguistics 2024 正式 PDF，DOI 10.1162/coli_a_00511 | 复用忠实性与可信貌似性的区分；作为二级综合来源。 | PDF pp. 2–3，印刷 pp. 658–659，引言与 Table 1 |
| Language Models Implement Simple Word2Vec-style Vector Arithmetic；2305.16130v3 | 新建 FFN 关系向量迁移；与静态词嵌入类比分开。 | PDF p. 3 关系／模型范围可见段落、p. 4 §3–4.1 及脚注 |
| Attention is not not Explanation；EMNLP-IJCNLP 2019 正式 PDF，pp. 11–20 | 复用 Attention 权重与解释；补入模型一致性、种子和诊断对照。 | PDF pp. 2–4，印刷 pp. 12–14，Figure 1、§2–3.1 |
| Large Multi-modal Models Can Interpret Features in Large Multi-modal Models；ICCV 2025 正式 PDF | 新建多模态 SAE 自动文字描述与干预评价；区域 IoU 是代理。 | PDF pp. 2–4，印刷 pp. 3651–3653，§2.1–2.4、§3.1 |
| SAE-V: Interpreting Multimodal Models for Enhanced Alignment；2502.17514v2 | 新建 SAE-V 跨模态激活相似度数据排序；与文字解释管线分开。 | PDF pp. 3–4，Algorithm 1、§2.2、§3.1.1 |
| Transformer Feed-Forward Layers Build Predictions by Promoting Concepts in the Vocabulary Space；2203.14680v3 | 复用 FFN 键值存储；补入 value sub-update 的词表投影与人工标注。 | PDF pp. 2–3，§2–4、Eq. (1)–(2) |
| Are Sixteen Heads Really Better than One?；1905.10650v3 | 复用多头剪枝与功能特化；Michel 的 mask 梯度与 Voita 的 L0 剪枝分开。 | PDF pp. 2–3、5–6，§2.3、§3.1–3.4、§4.1–4.2 |
| On the Robustness of Interpretability Methods；1806.08049v1 | 复用解释脆弱性；增加连续／离散局部稳定性及估计预算限定。 | PDF p. 2 Definition 2.1/Eq. (1)–(2)、p. 4 §3.3–3.4 文字 |
| Does Circuit Analysis Interpretability Scale? Evidence from Multiple Choice Capabilities in Chinchilla；2307.09458v3 | 新建 Chinchilla 答案到字母的局部电路；保留外延失败与直接效应限定。 | PDF pp. 3–6，贡献总结、§2.1–3.1 |
| Explanation in Artificial Intelligence: Insights from the Social Sciences；1706.07269v3 | 复用对比解释的事实与替代项；支持人类解释动机，不替代 SCM 定理源。 | PDF pp. 5–7，§1.1–1.3，尤其 §1.2 第 1 项 |

8 个复用概念保留原 `source`，新增 `sources` 和逐来源 `source_annotations`。原来源注释保留既有判据，并明确它们本轮没有重新审读；新来源注释写入本轮版本、主张、scope、实际页码与审阅深度。新增的 9 个概念也保留独立版本／来源注释。通常展示的 scope 和 evidence 同时补上新增来源范围，避免只挂 URL。全部 20 个本地原文件哈希不变，论文的版本和 claim_annotations 等原有字段一律不改。

**需要保留的测量区分。** BERT attention 的依存关系预测是相关诊断；Michel 的头重要性是 mask 导数代理与删头实验；探针准确率是表征可读出性；TruthfulQA 是输出真实性；RAVEL 是指定干预下的属性改变／隔离；Prisma 是工具支持。这些不能只因为都叫 interpretability 就合成同一种证据。Attention is not not Explanation 的设计质疑也不构成 Transformer 注意力天然忠实的证明。

**新增范围与边界。** 所有本轮新增概念及新来源注释按具体方法／评价／观察列 Knowledge，没有仅因经典地位、论文标题或多个模型而提高为 Law。RepE 明确读出相关性不等于控制成功，四类评价提案不代表每个示例均完成必要／充分性检查。TruthfulQA 的目标模型过滤与无信息拒答计真会影响解释；早期模型的 inverse scaling 不外推为当前所有模型的规律。FFN 词表投影暂略 normalization 的条件与 softmax 的相对性保留；多模态分割 IoU 的目标由模型产生，不能写作独立人工真值。SAE-V 的重构与稀疏指标不直接验证单义性。Chinchilla 的固定 RMS、直接／总效应和随机化字母分布中的部分解释均明写。

### 已审视但不机械补链接的 3 篇

- **A Primer in BERTology: What we know about how BERT works（TACL 2020 正式 PDF，vol. 8 pp. 842–866）**：不需独立概念；按本轮目标已审视，书目链接仍保留为未关联。BERTology 在语法、语义与世界知识研究间做综合；局部读到的 probe、attention 和 MLM 诊断已各有一手概念来源。本轮不为这篇综合综述重复创建 BERT knowledge 总括概念，也未声称全部综述主题均已覆盖。 证据位置：TACL 2020 正式 PDF 第 2–3 页（印刷 pp. 843–844），§3.1–3.3 与 Figures 1–2 的文字／图注；未重新审查所引全部实验。
- **Open Problems in Mechanistic Interpretability（2501.16496v1）**：不需独立概念；作为 fan-out 入口和研究议程保留，书目链接仍未关联。第 19–20 页提出模型生物、基准和探针验证问题；这些是对多个既有方法的综合与开放问题，不需要一个空泛的 mechanistic interpretability challenges 概念来消除链接空缺。其他章节仍未全面审读。 证据位置：2501.16496v1 PDF 第 19–20 页，§2.1.4b、§2.2.1–2.2.3，模型生物局限与 probe 相关／因果区别。
- **Prisma: An Open Source Toolkit for Mechanistic Interpretability in Vision and Video（2504.19475v3）**：不需单列工具名为独立机制概念；工具能力已定向审视，初步机制观察仍待独立判定。Prisma §3 是 HookedViT、缓存、hook、消融和 sparse-coder 训练／可视化支持，不能把 API 支持当作解释已验证。§5 的视觉 SAE 密度和替换后损失下降是待另读的初步经验候选；本轮不把 denoising 假说当确定机制。未安装运行或验证作者所报 75+ 模型支持。 证据位置：2504.19475v3 PDF 第 2–3 页 §2.3–2.5、§3.1、§4 与 §5 开头；此版本页脚为 CVPR 2025 MIV workshop。

这 3 篇仍会出现在构建器生成的书目未关联队列里；这里的处置不修改构建器匹配逻辑。BERTology 和 Open Problems 不需为消除计数额外造一个总括概念；这也不是断言其每个主题都已经覆盖。Prisma 的 §5 视觉 SAE 密度、替换后损失变化及去噪解释属于**尚未闭合的内容候选**，本轮仅看过章节开头，后续需要定向核对实验设置和对照后再决定是否形成独立概念。作者所述软件能力没有运行验证。

**阅读深度。** 本轮是指定章节核对，未声称全文审读、实验复现、完整证明审计或图像目检。RepE v4 的 pp. 8–10 文本提取出现 xref 修复提示后成功返回正文；保留原 PDF，不把文本返回当作文件资产或插图完整性保证。一次 Word2Vec-style p. 3 输出部分被截断，因此只采用实际可见的范围段落，并完整核对 p. 4 方法与脚注；没有使用未显示部分为结论背书。此前参考文献 pp. 56–61 尚未处置的候选及 pp. 62 之后的扩展任务未在本轮完成。


## 本轮：回补 pp. 56–61 明确候选并推进参考文献 pp. 62–67（2026-10-28）

沿 **Open Problems in Mechanistic Interpretability，2501.16496v1** 的原始参考文献推进；本轮按两页一批读取，单批未超过 8 页。先复查 pp. 56–61 上轮留下的明确候选，再逐项记录 pp. 62–67 的可见条目。**新增 12 篇／项一手原文、9 个独立概念，并给 3 个既有概念补逐来源证据**；本方向从 **277 篇／267 个概念**变为 **289 篇／276 个概念**。这里的 12 项包括 11 篇 PDF 论文与 1 篇原生 HTML 研究更新；不把研究更新伪装成会议论文。

三库按规范化题名、arXiv 和 DOI／来源 URL 交叉去重后再纳入。全部 277 个既有论文对象不变；3 个复用概念为“线性 probe”“Attention 权重与解释”“因果中介分析”，保留旧 source／sources／source_annotations，增加本轮实际阅读证据。其余 **264 个旧概念对象不变**。12 个新来源均有明确概念链接；内存运行同一 URL／arXiv 匹配规则后，书目未关联仍为 BERTology、Open Problems、Prisma **3 项**，不机械造概念消除这 3 个计数。未重建共享输出，交由根线程统一处理。

### 新增来源、版本与实际阅读位置

| 综述参考页 | 新增原文 | 本地登记版本／总页数 | 本轮实际阅读 |
|---|---|---|---|
| 57 | Context-dependent computation by recurrent dynamics in prefrontal cortex | Nature 503, 78–84 (2013) 正式排版；本地 PDF 含方法／扩展图共 19 页；19 页 | PDF 第 3–5 页（印刷 pp. 80–82），Figures 2–5 图注、Recurrent network model of selection and integration、A novel mechanism of selective integration；核对模型／数据差异及猴 F 例外。 |
| 57 | Acquisition of Chess Knowledge in AlphaZero | 2111.09259v3，2022-08-18；69 页，非本轮核验的 PNAS 最终稿；69 页 | v3 PDF 第 2–3 页 §1.1–1.2、第 6–7 页 §3.1.1/Figure 1、第 9–10 页 §4.2–4.3.2，Eq. (6)–(12)，固定棋局集、两类对照和指标定义。 |
| 56 | Sparse Crosscoders for Cross-Layer Features and Model Diffing | 原生研究更新，页面标注 2024-10-25；本轮保存 HTML 快照；原生 HTML | 原生 Introduction 的 Research Update 声明、§1.2、§2–2.1、§4.2–4.3；核对联合编码／重构、L1-of-decoder-norms、因果限制及 Sonnet 案例选择说明。 |
| 58 | The Quest for the Right Mediator: A History, Survey, and Theoretical Grounding of Causal Interpretability | 2408.01416v1，2024-08-02；保留综述引用时题名，未替换为改题的 v3；32 页 | v1 PDF 第 3 页 Eq. (1)、第 5–7 页 Figure 2、§4–4.2，特别是非基对齐方向的外加系数与粗粒度干预副作用。 |
| 62 | Learning to Deceive with Attention-Based Explanations | ACL 2020 正式 PDF，pp. 4782–4793；12 页 | 正式 PDF 第 2–3 页（pp. 4783–4784）§3–4 和注意力惩罚；第 7 页（p. 4788）§5.2–5.3、Table 5，受试者数量与置零 attention 后的信息流对照。 |
| 62 | Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance? | EACL 2021 正式 PDF，pp. 3363–3377；15 页 | EACL 正式 PDF 第 2–6 页（pp. 3364–3368），§3、§4.1–4.3、Tables 1–3，固定属性、词项隔离、下采样、词向量及 probe 容量；§5 合成实验仅看开头，详细结果未审。 |
| 63 | The mechanistic basis of data dependence and abrupt learning in an in-context classification task | 2312.03002v1，2023-12-03；OpenReview PDF 返回 403 后取得 arXiv，非已核验 ICLR 最终稿；13 页 | v1 PDF 第 3–6 页 §2–3、Eq. (1)–(6)、Figures 2–5 文字；ICL/IWL 测试构造、六 seeds、progress measures、固定 β2=0 的缩减对照。 |
| 63 | The importance of mixed selectivity in complex cognitive tasks | Nature 497, 585–590 (2013) 正式排版，作者网站 PDF；6 页 | PDF 第 2–3 页（pp. 586–587），Figures 1–3 图注、Mixed selectivity／Information encoded／Neural representations are high-dimensional 段落。 |
| 64 | FIND: A Function Description Benchmark for Evaluating Interpretability Methods | NeurIPS 2023 Datasets and Benchmarks 正式 PDF，28 页；28 页 | 正式 PDF 第 3–6 页 §2–3.1，Table 1、Figure 2；函数构造、噪声／局部 corruption、Vicuna 可靠性和代码／语言评分规则。 |
| 65 | Counterfactual Explanations Can Be Manipulated | NeurIPS 2021 正式 PDF，14 页；14 页 | 正式 PDF 第 3–6 页 §2–4、Eq. (1)–(4)、Definition 2.1、Lemma 3.1、Table 1；核对搜索族、两阶段目标、数据和模型。 |
| 65 | Grokking Group Multiplication with Cosets | 2312.06581v2，2024-06-17；27 页 | v2 PDF 第 2–7 页，§3–6.3、Figures 1–5；第 3 页另行渲染目检确认换位表述；重点 §4 架构、§5 coset 构造及 §6 干预，未完整复现实验。 |
| 66 | Codebook Features: Sparse and Discrete Interpretability for Neural Networks | 2310.17230v1，2023-10-26；非本轮核验的 ICML 最终稿；30 页 | v1 PDF 第 2–4 页 §2–3、Figures 1–3；top-k cosine 编码、两项损失、梯度估计、k 和 TokFSM 设置；大模型性能的完整结果未审。 |

11 个 PDF 均核对了文件签名、pdfinfo 页数、首页题名／作者及可见版本／正式版页脚，记录 SHA-256 与字节数。Crosscoders 保存原生 HTML，核对正文题名、作者、2024-10-25 日期和指定段落；并非 PDF，其外部图像、脚本和交互资产不保证完整离线。12 个新原文件合计 **41,151,693 字节**，最大 **10,595,838 字节**，没有大于 50 MB 的新增文件；下载目录由现有 `papers/` 忽略规则覆盖。未修改旧下载文件，不新增任何 git 追踪或提交。

**版本不混用。** AlphaZero 本地是 2111.09259v3（2022-08-18，69 页），其首页作者表与 PNAS 最终条目并不完全相同；登记 DOI 用于识别同一工作，不代表已核验最终稿。Quest 保留综述引用时的 2408.01416v1 题名和内容，未替换为后来改题的 v3。Reddy 的 OpenReview PDF 返回 403，实际取得 2312.03002v1；Codebook 是 2310.17230v1，均不冒称已核验会议最终稿。Cosets 登记 2312.06581v2，不沿用综述参考条目中混杂的会议／年份说法。Gated SAE 在本轮参考链核对中发现正式题名与已收 arXiv 题名不同；通过正式论文作者及摘要匹配到 2404.16014，作为已在库的别名处理，没有重复纳入或改写旧版本。

### 按具体主张分类及 scope

本轮新增主张均列 **Knowledge**。它们给出有限设置内的机制证据、方法、评价协议或综述性定义；经典地位、多模型或一个形式化公式本身不满足用户的 Law 判据。没有因题名或局部推导把整篇论文升为 Science。

| 原文 | 本轮登记主张 | Scope 与边界 |
|---|---|---|
| Context-dependent computation by recurrent dynamics in prefrontal cortex | 通过群体状态空间轨迹和训练 RNN 的固定点附近动力学，提出上下文依赖的松弛方向可选择并积累相关感觉输入；无关输入并非必须在进入 PFC 前被过滤。 | 两个猕猴的运动／颜色选择任务与对应训练 RNN。模型局部线性化和近似线吸引子用于解释该模型；不等于已识别大脑唯一机制。猴 F 的颜色信号不足以接受或拒绝早期选择模型，神经数据与 RNN 的瞬态输入响应也有差异；补充推导未审。 |
| Acquisition of Chess Knowledge in AlphaZero | 在固定人类棋局位置集上，为各层与训练 checkpoint 拟合稀疏线性／logistic 探针，以可读出分数绘制人类棋类概念何时、何处变得可预测。 | 指定 AlphaZero ResNet、Stockfish／自定义棋类概念、ChessBase 去重位置、探针正则和训练／验证／测试拆分。输入回归与随机概念是对照；分数测量计算受限读出，不证明模型以人类概念作为唯一因果算法，也不意味着产生新的 Shannon 信息。重复训练图与开局行为的全部结果未复核。 |
| Sparse Crosscoders for Cross-Layer Features and Model Diffing | Crosscoder 从多层或多模型激活联合编码共享稀疏特征，并分别解码到各位置；按层 decoder 范数加权的稀疏惩罚可显露共享与局部特征，用于追踪持续特征及 model diffing。 | 作者明示为初步研究更新。基础版本可非因果，strictly causal 版本又不能直接覆盖 attention 计算；共享编码与电路简化不保证原模型因果机制等价。Sonnet 基础／微调对比只给初步、部分挑选案例，多数模型独有特征未立即可解释；不把所有跨架构比较设想当已验证结果。 |
| The Quest for the Right Mediator: A History, Survey, and Theoretical Grounding of Causal Interpretability | 以中介和干预值为单位统一组织局部反事实效应，区分整层／子模块、基对齐神经元集合和非基对齐子空间，讨论粒度、数量与外加表征模块的权衡。 | 方法综述及操作性框架。效应依赖输入、替换值、响应指标和中介定义；外加 probe／autoencoder 改变抽象单位，粗粒度定位也可能包含多种角色。不是对各方法完整性的统一定理，未复核全部历史归属或所引实验。 |
| Learning to Deceive with Attention-Based Explanations | 训练目标可惩罚指定 token 的 attention mass，同时保留预测性能与对相关信息的依赖；循环编码可把信息转移到邻近 token 表征，因而低注意力不直接证明信息未被使用。 | 四类二分类和四类 seq2seq 任务的方法设定；本轮重点核对 BiLSTM 干预与职业分类的三名研究生受试者小型实验。改变的是模型训练目标，不是仅给冻结模型替换一个任意 attention 分布；用户相信程度不外推到所有人群或模型。 |
| Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance? | 通过将训练任务中的时态／主语数／宾语数固定，并匹配样本规模，观察探针仍能从表示解出这些属性；预训练词向量是部分信息来源，但随机初始化控制并未消除该现象。 | MultiNLI 的英语启发式分组、CBOW／BiLSTM 与另设探针数据；探针本身是 200 隐单元 MLP，不能冒称本文只用了线性探针。固定属性自身不能区分主任务标签，不代表排除了全部条件关系或一切因果使用；不能从一项反例断言所有探针无用。 |
| The mechanistic basis of data dependence and abrupt learning in an in-context classification task | 在受控分类序列上区分 in-weights 与 in-context 解法，以位置配对、目标配对和上下文标签准确率跟踪训练，再用三参数／二参数 induction-head 模型与定向消融检验突变来源。 | 两层各单头 attention＋深分类器、有限类别与标签、人工 burstiness／频率／类内噪声。降维模型假定正交 content/buffer 子空间；复现部分数据依赖与曲线不证明所有 LLM 的突变都由同一机制引起，也未完成后续现象学推导审计。 |
| The importance of mixed selectivity in complex cognitive tasks | 单元对任务因素的非线性混合响应可形成高维群体表示；去除单因素平均选择性后，任务因素仍可从群体活动被解码，单神经元平均调谐不能穷尽分布式信息。 | 两个猕猴对象序列记忆任务的 237 个 PFC 神经元、伪同时群体及重采样。去除 classical selectivity 是数据变换，不是体内因果消融；线性分类容量作为维度代理有噪声与样本条件。补充材料的容量证明和行为相关性分析未完整审计，不宣称所有人工网络均依此机制。 |
| FIND: A Function Description Benchmark for Evaluating Interpretability Methods | 用数学函数、字符串程序与指令控制的合成神经模块评估解释器；分别测试生成代码的行为一致性及自然语言描述能否帮助评价模型辨识输入输出，并加入局部 domain corruption。 | 人工函数族、指定采样与有限测试集。数值 NMSE<0.1、字符串十个测试输入和语言三选一协议不等于全域函数等价；Vicuna 合成模块只是近似执行给定规则，其意图与真实输出不能无条件视为同一真值。该基准成功不自动证明真实网络机制已解释。 |
| Counterfactual Explanations Can Be Manipulated | 通过联合训练预测模型与小扰动，让指定 hill-climbing 反事实搜索在原输入返回较高成本解，在特定群体的扰动输入返回更低成本解，从而操纵基于搜索结果的行动成本审计。 | Wachter、稀疏变体、DiCE、prototype 方法及两种表格数据；成本由人工距离代理，搜索局部最优不同于全局最低可行行动。文中明确不含 MACE/FACE 类不同求解协议。Lemma 3.1 的隐式导数还需平滑、局部稳定分支与可逆 Hessian 等条件，附录证明本轮未审；不据它宣称全部反事实方法必可操纵。 |
| Grokking Group Multiplication with Cosets | 在 fully grokked 单隐层 ReLU 的 S5／S6 乘法模型中，按左右 pre-activation 在子群陪集上的方差识别 coset 单元，并通过阈值消融、左右 embedding 交换和符号改变检验机制。 | 有限群、指定单层架构与训练所得模型；观测分解和消融结果并不唯一证明所有群任务采用该算法。不得把任意 Fourier 对齐当作另一机制的充分证据。v2 §3.1 把换位分解长度写成不变量是错误，实际仅奇偶性不变；已目检原页，未沿用该表述。完整群论附录与 §7 机制争论未审。 |
| Codebook Features: Sparse and Discrete Interpretability for Neural Networks | 以余弦相似度选择 top-k code 向量之和替代子层激活，用原任务损失与重构项联合微调；通过切换 code 检验 FSM 状态与生成主题的可控性。 | codebook 加入 attention／MLP 输出、保留残差连接；会改变并微调网络，不是冻结原模型的无损解释。使用 straight-through 梯度、重构 stop-gradient 和有限 FSM／语言模型设置，离散 code 仍不能保证完全单义或跨域机制唯一。 |

**单独记录的原文错误。** Cosets v2 PDF 第 3 页 §3.1 将一个置换分解为换位的“数量”说成不变量，并声称所有分解都有相同 k；原页已经渲染目检，根线程也独立核对。反例为 `(13) = (12)(13)(23)`，右侧三个换位互异；正确不变量是奇偶性，而非长度。同页“唯一正规子群”为 A_n 的说法需要补“非平凡真子群”等限定（平凡子群和全群也正规）。这些背景表述没有被沿用，**仅凭它们不能推翻 §5–6 的 coset 观察与干预结果**；完整群论附录和 §7 的机制争论本轮未审。目录中的核心主张仍限定为所测单隐层 ReLU 的 S5／S6 实验。

**推导与测量限制。** Mante 的局部线性化解释训练 RNN，猴 F 的颜色任务保留不能裁决早期选择的例外。Rigotti 的移除 classical selectivity 是数据变换，不能称体内消融。Ravi 的探针是 200 隐单元 MLP，来源附到更广的“可读出不等于使用”主张，未说本文仅用线性 probe。FIND 的有限样本／阈值匹配不等于全域函数等价。Counterfactual Explanations Can Be Manipulated 的隐式导数还依赖局部稳定、平滑与可逆 Hessian 等条件；附录证明未审，不据此宣布所有反事实求解器均可操纵。Codebook 改变并微调网络，与冻结模型的事后解释分开。Crosscoder 的初步 model diffing 案例与所有跨模型设想分开。

### pp. 56–61：本轮重新打开的明确候选

这是从前轮未解决项中目的性回补的清单，**不是该区间全部参考文献已穷尽**。

| 参考页 | 条目／候选 | 本轮处置 | 说明 |
|---|---|---|---|
| 56 | Sparse Crosscoders for Cross-Layer Features and Model Diffing | 本轮纳入 | 新建跨层／跨模型共享字典概念，保留初步研究与因果限制。 |
| 56 | Seeing Is Believing（2305.08746） | 后续已补 | 2026-10-28第二批已补；实际题名、原文位置和边界见下方“跨架构、程序抽取与研究实践”。  原待核备注的自动特征解释归类已按BIMT原文纠正。 |
| 57 | Context-dependent computation by recurrent dynamics in prefrontal cortex | 本轮纳入 | RNN 选择／积分与局部线吸引子；不等于大脑唯一机制。 |
| 57 | Acquisition of Chess Knowledge in AlphaZero | 本轮纳入 | what–when–where 的稀疏探针；不把读出当使用。 |
| 57 | Opening the AI Black Box: Program Synthesis（2402.05110） | 后续已补 | 2026-10-28第二批已补；实际题名、原文位置和边界见下方“跨架构、程序抽取与研究实践”。 |
| 58 | Conditional subgroup permutation feature importance（DOI 10.1007/s10618-022-00901-9） | 待核 | 条件分组／置换估计对象与可识别性尚未审。 |
| 58 | From Insights to Actions（2406.12618） | 待核 | 解释用于干预／改进模型的实际证据尚未审。 |
| 58 | The Quest for the Right Mediator（2408.01416v1） | 本轮纳入 | 复用因果中介概念，新增中介粒度和非基对齐子空间范围。 |
| 58 | Attribution Patching 原生文章 | 待核 | 已有 EAP 等论文不替代原生文章的优先权／边界核对。 |
| 59 | Plug & Play Generative Networks（1612.00005） | 待核 | 与激活最大化／生成先验相关，但本轮未读。 |
| 59 | Fact Finding 原生文章 | 待核 | 已有事实编辑／电路工作不自动覆盖该原始文章。 |
| 59 | Logit lens 原生来源 | 待核 | 已有概念不意味着原始网页已完成定向核对。 |
| 59 | Features as linear representations 原生来源 | 待核 | 线性表征假设的原始定义与限制仍待核对。 |
| 60 | Early Vision in InceptionV1 | 后续已补 | 2026-10-28第二批已补；实际题名、原文位置和边界见下方“跨架构、程序抽取与研究实践”。 |
| 61 | Can Transformer Interpretability Transfer to RNNs?（2404.05971） | 后续已补 | 2026-10-28第二批已补；实际题名、原文位置和边界见下方“跨架构、程序抽取与研究实践”。 |
| 61 | Weight-Based Decomposition of Bilinear MLPs（2406.03947） | 后续已补 | 2026-10-28第二批已补；实际题名、原文位置和边界见下方“跨架构、程序抽取与研究实践”。 |

### pp. 62–67：逐项纳入／已在库／待核账本

该批最初登记时，可见参考段落手工列出 **59 行**：**8 行当批纳入、20 行已在三库、31 行待核**。这不是 59 篇独立论文：Simonyan 2014a／2014b 是同一工作的重复书目行；末尾 Gemini Team 是跨页未完的作者列表，题名在 p. 67 以后，本轮未读到，不把它计为完整工作。20 个“已在库”是身份／别名匹配，不表示本轮重读了这些论文。31 个待核项也不能直接叫已确认的概念缺漏；综述、工具、议程和历史文献有不同优先级。

| 参考页 | 可见条目 | 本轮处置 | 依据／未解决原因 |
|---|---|---|---|
| 62 | Dissecting Language Models: Machine Unlearning via Selective Pruning | 后续已补 | 第三批已补，v2 §3–5定向核对；伪代码比值方向与Definition 2不一致已记录，代码未审。 |
| 62 | Fine-tuning Enhances Existing Mechanisms: A Case Study on Entity Tracking | 后续已补 | 第三批已补；静态7盒子任务、全部MLP保留、72头及微调新增组件的边界见下方。 |
| 62 | Learning to Deceive with Attention-Based Explanations | 本轮纳入 | 新论文入库，复用已有概念并补逐来源证据 |
| 62 | Language Models are Unsupervised Multitask Learners | 已在库 | pretrain：Language Models are Unsupervised Multitask Learners |
| 62 | Direct Preference Optimization: Your Language Model is Secretly a Reward Model | 已在库 | rl：Direct Preference Optimization: Your Language Model is Secretly a Reward Model |
| 62 | A Practical Review of Mechanistic Interpretability for Transformer-Based Language Models | 后续已补 | 第三批核对实际v4指定章节；未恢复综述所引2024版本，复用既有概念。 |
| 62 | Improving Sparse Decomposition of Language Model Activations with Gated Sparse Autoencoders | 已在库 | interp：Improving Dictionary Learning with Gated Sparse Autoencoders；NeurIPS 正式题名与 arXiv Dictionary Learning 为同一作者／方法工作，不重复计数；保留现有 arXiv 版本。 |
| 62 | A Category-free Neural Population Supports Evolving Demands during Decision-Making | 后续已补 | 第三批以作者稿核对5只鼠PPC的PAIRS/解码/方差对齐，未检出不等于全称无类别。 |
| 62 | Toward Transparent AI: A Survey on Interpreting the Inner Structures of Deep Neural Networks | 后续已补 | 第三批核对v6 Scope/Taxonomy及图1；复用既有概念。 |
| 62 | Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection | 已在库 | interp：Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection |
| 62 | Linear Adversarial Concept Erasure | 后续已补 | 2026-10-28 后续已补入；版本、实际审读与边界见下方“经典源头、早期 SAE 与版本更正”。 |
| 62 | Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance? | 本轮纳入 | 新论文入库，复用已有概念并补逐来源证据 |
| 63 | The Mechanistic Basis of Data Dependence and Abrupt Learning in an In-Context Classification Task | 本轮纳入 | 新论文入库并增加独立概念 |
| 63 | Why Should I Trust You?: Explaining the Predictions of Any Classifier | 已在库 | interp：Why Should I Trust You?: Explaining the Predictions of Any Classifier |
| 63 | Finding Sparse Linear Connections Between Features in LLMs | 后续已补 | 2026-10-28第二批已补；实际题名、原文位置和边界见下方“跨架构、程序抽取与研究实践”。 |
| 63 | The Importance of Mixed Selectivity in Complex Cognitive Tasks | 本轮纳入 | 新论文入库并增加独立概念 |
| 63 | Steering Llama 2 via Contrastive Activation Addition | 已在库 | interp：Steering Llama 2 via Contrastive Activation Addition |
| 63 | The Principles of Deep Learning Theory: An Effective Theory Approach to Understanding Neural Networks | 待核 | 教材/跨域资源，须选定与解释直接相关章节；本批不把整书算新实验paper。 |
| 63 | Coup Probes: Catching Catastrophes with Probes Trained Off-Policy | 后续已补 | 2026-10-28第二批已补；实际题名、原文位置和边界见下方“跨架构、程序抽取与研究实践”。 |
| 63 | A Primer in BERTology: What We Know about How BERT Works | 已在库 | interp：A Primer in BERTology: What we know about how BERT works |
| 63 | High-Resolution Image Synthesis with Latent Diffusion Models | 已在库 | pretrain：High-Resolution Image Synthesis with Latent Diffusion Models |
| 63 | A Unifying Review of Linear Gaussian Models | 待核 | 已定位Roweis/Ghahramani作者源和摘要，但PCA/FA/Kalman统一框架与当前MI直接桥梁未核，暂不纳入。 |
| 63 | Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead | 已在库 | interp：Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead |
| 63 | Mechanistic? | 后续已补 | 2026-10-28第二批已补；实际题名、原文位置和边界见下方“跨架构、程序抽取与研究实践”。 |
| 63 | Are Emergent Abilities of Large Language Models a Mirage? | 已在库 | pretrain：Are Emergent Abilities of Large Language Models a Mirage? |
| 63 | Large Language Models Can Strategically Deceive Their Users When Put under Pressure | 待核 | 2311.07590行为风险候选；需核内部机制证据或转RL，不能仅按题名归入MI。 |
| 63 | Bridging the Human-AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero | 后续已补 | 2026-10-28 后续已补入；版本、实际审读与边界见下方“经典源头、早期 SAE 与版本更正”。 |
| 64 | FIND: A Function Description Benchmark for Evaluating Interpretability Methods | 本轮纳入 | 新论文入库并增加独立概念 |
| 64 | Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization | 已在库 | interp：Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization |
| 64 | Toward Verified Artificial Intelligence | 待核 | 形式验证议程；未读具体命题与假设，不因标题verified判Science。 |
| 64 | A Multimodal Automated Interpretability Agent | 后续已补 | 2026-10-28 后续已补入；版本、实际审读与边界见下方“经典源头、早期 SAE 与版本更正”。 |
| 64 | A Value for n-Person Games | 后续已补 | 2026-10-28 后续已补入；版本、实际审读与边界见下方“经典源头、早期 SAE 与版本更正”。 |
| 64 | A Technical Note on Bilinear Layers for Interpretability | 后续已补 | 2026-10-28 后续已补入；版本、实际审读与边界见下方“经典源头、早期 SAE 与版本更正”。 |
| 64 | Current Themes in Mechanistic Interpretability Research | 后续已补 | 2026-10-28第二批已补；实际题名、原文位置和边界见下方“跨架构、程序抽取与研究实践”。 |
| 64 | Taking Features Out of Superposition with Sparse Autoencoders | 后续已补 | 2026-10-28 后续已补入；版本、实际审读与边界见下方“经典源头、早期 SAE 与版本更正”。 |
| 64 | Locating and Editing Factual Associations in Mamba | 后续已补 | 2026-10-28第二批已补；实际题名、原文位置和边界见下方“跨架构、程序抽取与研究实践”。 |
| 64 | Towards Understanding Sycophancy in Language Models | 待核 | 偏好/行为与RL候选，需读机制或训练证据；跨域归属待核。 |
| 64 | Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer | 已在库 | pretrain：Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer |
| 64 | Observations on the Scratch-Reflex in the Spinal Dog | 待核 | Sherrington1906历史生理背景；本轮无直接MI桥梁，不按综述中的引用计数占位符生成当前引用量。 |
| 64 | Model Evaluation for Extreme Risks | 待核 | 2305.15324风险评测框架，未审其与解释方法的直接证据，保持候选。 |
| 64 | Hypothesis Testing the Circuit Hypothesis in LLMs | 后续已补 | 2026-10-28 后续已补入；版本、实际审读与边界见下方“经典源头、早期 SAE 与版本更正”。 |
| 65 | Language Models are Better than Humans at Next-Token Prediction | 待核 | 2212.11281行为比较，不直接等于内部解释；本批暂缓，未视为正文已核。 |
| 65 | Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps [2014a] | 已在库 | interp：Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps |
| 65 | Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps [2014b] | 已在库 | interp：Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps；综述 2014a/2014b 是重复引文。 |
| 65 | Fooling LIME and SHAP: Adversarial Attacks on Post Hoc Explanation Methods | 已在库 | interp：Fooling LIME and SHAP: Adversarial Attacks on Post hoc Explanation Methods |
| 65 | Counterfactual Explanations Can Be Manipulated | 本轮纳入 | 新论文入库并增加独立概念 |
| 65 | Beyond Model Interpretability: Socio-structural Explanations in Machine Learning | 后续已补 | 第三批核对v1社会结构论证与案例来源；招聘toy和借用医疗实证分开。 |
| 65 | The Strong Feature Hypothesis Could Be Wrong | 后续已补 | 2026-10-28 后续已补入；版本、实际审读与边界见下方“经典源头、早期 SAE 与版本更正”。 |
| 65 | Feature Is Overloaded Terminology | 评论原文已核 | 指定commentId正文已核，作为既有概念补充源；不伪造独立paper。 |
| 65 | Neural and Conceptual Interpretation of PDP Models | 后续已补 | 2026-10-28 后续已补入；版本、实际审读与边界见下方“经典源头、早期 SAE 与版本更正”。 |
| 65 | Deep Unsupervised Learning Using Nonequilibrium Thermodynamics | 已在库 | pretrain：Deep Unsupervised Learning using Nonequilibrium Thermodynamics |
| 65 | Dropout: A Simple Way to Prevent Neural Networks from Overfitting | 已在库 | pretrain：Dropout: A Simple Way to Prevent Neural Networks from Overfitting |
| 65 | Grokking Group Multiplication with Cosets | 本轮纳入 | 新论文入库并增加独立概念 |
| 65 | Axiomatic Attribution for Deep Networks | 已在库 | interp：Axiomatic Attribution for Deep Networks |
| 65 | Attribution Patching Outperforms Automated Circuit Discovery | 已在库 | interp：Attribution Patching Outperforms Automated Circuit Discovery |
| 65 | Intriguing Properties of Neural Networks | 后续已补 | 2026-10-28 后续已补入；版本、实际审读与边界见下方“经典源头、早期 SAE 与版本更正”。 |
| 66 | Going Deeper with Convolutions | 已在库 | pretrain：Going deeper with convolutions |
| 66 | Codebook Features: Sparse and Discrete Interpretability for Neural Networks | 本轮纳入 | 新论文入库并增加独立概念 |
| 66 | Gemini Team long-author entry（题名跨页至p.69） | 后续已补（pretrain） | 已确认Gemini: A Family of Highly Capable Multimodal Models，2312.11805；pretrain收录官方90页©2025修订稿，定向读pp1–8、73–75，非已取得2023v1或全文审计。 |

上一轮结束时未闭合的优先候选包括：**Linear Adversarial Concept Erasure（RLACE）**、Shapley 1953 原始章节、Smolensky 1986 原始 PDP 章节、Szegedy 的 Intriguing Properties、早期 Taking Features Out of Superposition 原生文章、bilinear 技术笔记／分解、MAIA、Hypothesis Testing the Circuit Hypothesis、AlphaZero 概念发现／向人迁移、Strong Feature Hypothesis 反例文章，以及前表的早期视觉电路、程序综合、跨 RNN 迁移。它们需要回到各自原文，不能由相邻概念或综述被引用事实替代。

**阅读深度与覆盖声明。** 本轮全部新记录都有指定页码／原生段落的实际阅读及版本，未声称全文精读、实验复现或完整证明审计。PDF 文本读取不自动算插图目检；只有上述 Cosets p. 3 因疑似原文错误另做了渲染目检。pp. 62–67 的可见书目行已经逐项登记处置，但待核仍开放，pp. 68 及以后尚未推进；这不是该综述引用链或解释学科的全覆盖完成。

## 2026-10-28 后续补录：经典源头、早期 SAE 与版本更正

本轮新增 **10 项论文／原生文章、10 个概念**，使 interp 目录达到 **299 项、286 个概念**；其中 285 项有本地主 PDF、14 项有本地 HTML。新增主材料是 8 份 PDF 与 2 份原生 HTML，另保存 RLACE 旧会议版 PDF 作版本对照。这里包括历史源头与近年重要方法／批评，不把“本轮纳入”当作它们全部已获经典地位的证明。

| 来源 | 主材料与实际核对位置 | 本轮分类与主要边界 |
|---|---|---|
| Linear Adversarial Concept Erasure | arXiv 2201.12091v8，pp.2–5，§2.3脚注4、§3.3脚注6；保留 ICML 2022 旧版作补充 | Knowledge：擦除方法与有限实验。作者更正凸–凸博弈及 max/min 顺序，交替优化没有一般收敛保证；不沿用旧版凸凹保证。 |
| Shapley, A Value for n-Person Games | 1953原章的1988正式重印，抽取章内pp.1–6（印刷31–36）；pp.3、5另作目检 | Science：给定合作博弈及原始三公理的唯一性。ML集合函数、基线、缺失分布与因果语义仍需另定；不混同SHAP 2017的三公理。 |
| Smolensky, Neural and Conceptual Interpretation of PDP Models | PDP Volume 2 Chapter 22扫描，pp.1–5 OCR、首页目检 | Knowledge：神经／数学／概念三层映射的建模区分；神经解释与概念解释独立，后续动力学变换尚未完整审计。 |
| Szegedy et al., Intriguing Properties | 1312.6199v4，pp.3–5，§3–4.2 | Knowledge：随机方向的语义刺激和近似优化攻击；不是全局最小扰动证明，敏感性不等于数学不连续。 |
| Taking Features Out of Superposition | Conjecture 2022-12-13原生 interim report，Methods、MMCS、真实激活诊断、Conclusion | Knowledge：指定玩具数据的方向恢复；真实LM诊断不同，MMCS和字典一致性不保证真值或机制完整。 |
| A Technical Note on Bilinear Layers | 2305.03452v1，pp.2–4，式(1)–(4) | Science：指定无额外激活层的二次张量恒等式；逐输入槽线性不等于整个网络线性，性能和安全外推未确立。 |
| MAIA | 2404.14394v2（2025修订，原工作2024），pp.3–6，API与Figure 3评价流程 | Knowledge：指定视觉单元和生成式评价；测试提示依赖候选描述和生成器，不能当作任意自然分布上的完备因果验证。 |
| Hypothesis Testing the Circuit Hypothesis | 2410.13032v1，pp.5–7，§3.1–3.2与Table 1；p.7目检 | Knowledge：检验程序依赖任务、消融及参考分布。v1表注与原假设方向有冲突，不转录通过计数；未拒绝原假设也不等于证实机制。 |
| AlphaZero Concept Discovery and Transfer | 2310.16410v1，pp.9–10、18–19、50–51；p.9目检 | Knowledge：有限原型与四名棋手的前后题集比较；已写齐次约束存在零解缺口，代码和最终版未核，不能由此宣布实验无效。 |
| The Strong Feature Hypothesis Could Be Wrong | Lewis Smith 2024-08-02作者原生文，Weak/Strong LRH、Monosemanticity、Explicit vs Tacit、Conclusions | Knowledge：方法论论证与思想例子，不是实际大模型的正式反证；弱存在性与强完备性分开。 |

**材料身份。** Shapley 本地文件是从 Roth 编辑的338页重印卷 PDF 第39–48页抽取的10页章节，容器 SHA 与抽取文件 SHA 分开保存；首页明确标出1953原章来源，不能把重印页码写成1953原版页码。Smolensky 是 Volume 2 Chapter 22，不是 Volume 1 的 Harmony Theory。Strong Feature 的个人主页入口实际返回 Alignment Forum 正文，HTML canonical 与页面作者／日期已核对，索引以 canonical 为主链接并保留入口；网页本体不保证图片和样式全部离线。

**三个版本／方法细节。** RLACE v8 作者更正足以撤回旧版凸凹游戏与可交换极值顺序的表述，但不抹掉有限任务上的方法效果。电路检验 v1 Table 1 的“✓表示拒绝原假设”与理想检验的原假设方向以及 synthetic circuit 正文相冲突，代码与后续版待核；本次只录入方法，不反向猜表格数据。AlphaZero v1 的目标是最小化L1、约束为非严格齐次排序，所以零向量可行且目标为0；这是所写公式无法确定非零概念方向的具体问题，未核对实现或最终版，不推断实际实验采用零向量。人类研究仍作为独立的有限前后观察记录，前后题目难度／质量差异与AZ走法作为答案标准均保留。

全部新增材料重验字节数、SHA-256、PDF签名与pdfinfo页数，并按标题／arXiv检查跨三库重复；最大新增主PDF为MAIA的28,511,381 bytes，均在已忽略的papers/内，未新增超过50 MB文件。每次PDF抽取不超过8页；指定图像目检只覆盖上表明确列出的位置。未全文精读、复现实验或独立审计全部证明；没有新增Law标签，主张层级不由论文名气、是否有代码或样本数量直接决定。

此前pp.62–67候选中的这10项已补录；**Mamba事实编辑、早期视觉电路／程序综合／跨RNN迁移等仍待核，综述pp.68及以后也未在本轮推进**。已登记参考行不等于引用链或整个可解释性学科覆盖完成。

## 2026-10-28 后续第二批：跨架构、程序抽取与研究实践

本批新增 **10项文献／原生文章、10个概念**，目录现为 **309项、296个概念**：291份主PDF、18份本地HTML。本批6份PDF和4份原生文均保存一手正文、版本、hash和字节数。9项的已审主张列Knowledge，双线性权重分解拆成条件Science与实例Knowledge；没有由跨几个模型或若干任务自动升为Law。

| 来源 | 实际阅读位置 | 判据与Scope |
|---|---|---|
| Locating and Editing Factual Associations in Mamba | 2404.03646v2，pp.3–8，§3–6、式(7)–(8)、Figures2–7 | Knowledge：400事实定位、前2000 CounterFact编辑以及所列模型；IE恢复比例、编辑评分和线性读出不同。均值消融切向所有后续位置，与单条attention边干预粒度不等同。 |
| Finding Sparse Linear Connections | 2023-12-09原生文，模型／函数、MSE差阈值、L1连接、Please Help | Knowledge：Pythia-70M五层及两端SAE上的拟合；作者承认0.001阈值任意，拟合误差小不等于全域线性或已证因果连接。 |
| Current Themes | 2022-11-16原生文，开头、Barriers beyond superposition、Study simpler models／model systems、causality | Knowledge：讨论框架；原文明确不是完整综述或研究议程。模型系统的开放性、种子与轨迹、易研究性／典型性是建议，非充分保证。 |
| Coup Probes | 2023-11-17原生文，toy setting、probe训练、格式与对抗变化、附录强L2推导 | Knowledge：Llama2-7B-Chat的盗窃建议代理任务；真实takeover检出率预期非实验结果，强L2渐近近似不能写成任意有限正则的精确擦除或完整因果识别。 |
| Mechanistic? | 2410.09087v1，pp.1–3，四种定义、§2.1–2.2、§3开头 | Knowledge：区分技术定义与社群用法；不把内部观察自动当因果机制，也不按社群身份赋予证据层级。全部历史归因未独立核查。 |
| Early Vision in InceptionV1 | Distill 2020-04-01，开头、Playing Cards、Caveats、Presentation of Neurons／Circuits、Conclusion | Knowledge：1056个非瓶颈单元的人工工作分类。家族可低置信、边界模糊；权重乘积示意有忽略中间ReLU的近似，不能当全模型精确机制分解。 |
| Opening the AI Black Box / MIPS | 2402.05110v1，pp.2–3、5–7；架构搜索、符号回归、62任务筛选及失败模式 | Knowledge：RNN训练成功后提取程序的限定方法；有限测试100%不证明所有序列正确，离散输入不自动推出任意长度FSM。作者也报告连续计算导致失败。 |
| Does Transformer Interpretability Transfer to RNNs? | 2404.05971v1，pp.2–7，§2–4.1 | Knowledge：CAA／状态干预及tuned lens迁移；跨层择优和未归一化方向限制比较，状态与残差效果非简单可加。quirky／ELK结果本轮未纳入已审证据。 |
| Weight-Based Decomposition: A Case for Bilinear MLPs | 2406.03947v1，pp.2–4，§3–4.2、式(2)–(3)、§5.1 | Science仅为固定输出方向的二次型谱分解；语义实例仍为Knowledge。符号与简并自由度、交互稀疏与激活稀疏之别、跨层展开指数增长均保留。 |
| Seeing Is Believing / BIMT | 2305.08746v3，pp.2–4，§2三步骤、Connectivity graphs、§3/Table1 | Knowledge：几何连接代价、L1及神经元交换促进小型任务模块化。视觉稀疏不等于严格零权重；不是原先待核备注误写的“自动特征解释”工作。 |

**材料和分类核验。** 所有新PDF首页题名／作者／arXiv版本已核对；逐份验证签名、pdfinfo页数、字节数和SHA-256，最大新文件20,008,967 bytes，均在已忽略papers/内。跨三库标准化题名与arXiv去重通过，10个新概念各有明确原文source。原生HTML保存正文，外部图片、脚本、交互、代码仓库或链接数据均未声称完整离线；文章现时归档正文不能当作发表当天快照。每次PDF抽取不超过8页，本轮未进行全部图像目检、训练复现或完整证明审计。

**未闭合的理论外延。** Coup Probes的有限token／序列代理实验与真实部署风险不同，BERT弱比较也不证明所有外部分类器失败。MIPS程序可读与其数学功能等价是两个问题，已生成代码仍须逐程序验证。双线性分解的精确代数明确了可成立的模型域，却不证明人类语义基唯一。以上边界直接进入paper和concept的scope，而非仅在总报告附注。

**待核队列。** pp.62–67原59个可见书目行现在为：8行前批纳入、20行此前已在库、15行后续补录、16行仍待核；其中有重复引文和跨页未完作者列，仍不是59篇独立论文。Roweis–Ghahramani的[作者稿](https://mlg.eng.cam.ac.uk/zoubin/papers/lds.pdf)及[出版版作者存档](https://cs.nyu.edu/~roweis/papers/NC110201.pdf)已定位，摘要说明PCA/FA/Kalman/ICA等统一框架；本批未完成其与具体解释方法的原文桥接，不强行新增或宣布已覆盖。pp.56–61余下置换重要性、From Insights to Actions、Attribution Patching原生文、Plug & Play、Fact Finding、Logit Lens、Features as Linear Representations仍待核。综述pp.68及以后本批没有推进。

## 2026-10-28 后续第三批：综述定位、选择性遗忘与早期字典／视觉源头

本批新增 **8 篇论文／综述、6 个概念**，interp 达到 **317 项、302 个概念**：299 项有主 PDF、18 项为主 HTML。8 份新增主材料均为 PDF；另保存 Lewis Smith 的指定评论 HTML 作为既有概念补充来源，不计一篇独立论文。两篇综述复用“从行为解释到机制解释的范式”并添加逐来源注释，没有用综述数量制造新概念。本批所记主张均为 **Knowledge**；未新增 Law 或 Science。论文价值与本轮证据级别分开。

| 来源 | 实际材料与核对位置 | 主张边界 |
|---|---|---|
| [Dissecting Language Models](https://arxiv.org/abs/2403.01267) | v2，PDF pp.1–6；p.3 已渲染目检 | 忘却/保留激活重要性剪枝；unlearning 是指定数据集能力下降，不是训练样本影响已删除或知识不可恢复。Definition 2 与 Algorithm 1 行2–4的比值方向不一致，代码未审，不能据此否定全部实测。 |
| [Fine-tuning Enhances Existing Mechanisms](https://arxiv.org/abs/2402.14811) | v1，PDF pp.3–9，§3–6/Table 1/CMAP | 7个静态盒子、无状态变化、单token物体；72头回路保留全部MLP。faithfulness为准确率比；微调模型完整表现还需额外头，Structure Reader功能未完全解释。 |
| [A Practical Review](https://arxiv.org/abs/2407.02646) | 实际首页 v4，2025-10-13，61页；p.1、pp.5–6、pp.8–10 | 特征/电路/普适性研究路线。请求v1 URL实际返回v4，明确登记实际版本；未恢复2024初稿，不能将新增内容倒推给2501.16496v1引用时版本。 |
| [Category-free Neural Population](https://doi.org/10.1038/nn.3865) | Stanford作者资料存档中的HHS/PMC作者稿，33页；首页、pp.5–10 | 5只大鼠PPC、指定选择/模态任务、PAIRS与PCA/解码；未检出多数类别不是绝对不存在类别。Variance Alignment比较协方差维度而非轨迹；不直接推出人工网络规律。 |
| [Toward Transparent AI](https://arxiv.org/abs/2207.13243) | v6，2023-08-18；首页、pp.2–4，Scope/Taxonomy/Figure 1 | 权重/单神经元/子网络/潜在表示与内在/事后方法的综述分类；明确主体范围，不是全部解释方法的穷尽或因果忠实性保证。 |
| [Beyond Model Interpretability](https://arxiv.org/abs/2409.03632) | v1作者稿，PDF pp.2–9，§2–5 | 社会结构解释的方法论主张；招聘例子是toy，医疗案例引用已有研究。本轮未审原医疗数据；不能写成新实证或普遍必要性定理。 |
| [Transformer Visualization via Dictionary Learning](https://aclanthology.org/2021.deelio-1.1/) | ACL DeeLIO 2021出版PDF；首页、pp.2–5，§2/§3/式(1)/Table 2 | 12层冻结BERT，共享跨层非负稀疏字典、top-1000均值和left词性例子；稀疏线性叠加是工作假说，重要性曲线不是因果贡献。 |
| [Object Detectors Emerge](https://arxiv.org/abs/1412.6856) | v2/ICLR2015，PDF pp.2–6，§2–3.3 | ImageNet/Places同架构对照、随机遮挡RF、AMT top-60标注；类别分布用precision>75%的筛选单元，不是完整/唯一机制证明。 |

**指定评论的身份和离线边界。** [Feature Is Overloaded Terminology](https://www.lesswrong.com/posts/9Nkb389gidsozY9Tf/lewis-smith-s-shortform?commentId=fd64ALuWK8rXdLKz6) 已从作者主站获取正文，内嵌数据将目标内容映射到 `fd64ALuWK8rXdLKz6_contents`，而不是只读整个Shortform的标题。它区分输入属性、模型使用的表征与SAE字典元素，并建议后者称SAE latent；仅作术语补充，追加到“弱／强线性表示假说与显式／隐式计算之别”的source_annotations。HTML路径、字节数与SHA-256在该注释内；仅页面本体与可读正文落盘，未镜像全部资源或验证离线交互。

**来源身份例外。** Raposo保存的是标明2014期刊信息、2015-06-01可在PMC取得的作者稿，页码使用作者稿PDF页码，不冒称Nature排版终版；首页标题与作者可直接提取匹配，不需新增扫描身份豁免。Practical Review保留请求v1与实际v4偏差。其他7份PDF的实际身份/版本与主张定位均在JSON；所有单次PDF提取均不超过8页，取得PDF不等于全文已读。

**旧队列仍未闭合。** pp.62–67的6篇本批原文和1条指定评论已完成该层面的补录；教材、线性Gaussian统一综述、行为风险/偏好、形式验证、历史神经生理等仍按下表理由开放。Gemini长作者列表于p.69落题，现由根代理补入pretrain，限定为官方修订稿指定章节核对；该跨库收录不增加interp条目数。pp.56–61原有7项待核仍开放。本批没有把“分类后暂缓”当作原文核对完成。

### pp.68–75：到达书目终点后的有界身份核对

本批仅提取一次PDF pp.68–75（8页）。p.68与p.69前部仍是Gemini作者列表，p.69末落题为2312.11805；参考文献在p.74结束，p.75已是Appendix A。以下从pp.70–74手工选择44项与解释方法、表征或机制相关的候选做三库身份核对；并非把44篇全文读过，也不是该页所有条目的穷尽。下表为2项本批原文已补、12项已在库、30项待核原文。既有条目的身份匹配不升级其既有阅读深度。

| 参考页 | 条目 | 状态 | 身份／后续核对重点 |
|---|---|---|---|
| 70 | [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://arxiv.org/abs/2605.29358) | 已在库（本批未重读） | interp：Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet；已有原生文章，不在本批重读 |
| 70 | [BERT Rediscovers the Classical NLP Pipeline](https://arxiv.org/abs/1905.05950) | 已在库（本批未重读） | interp：BERT Rediscovers the Classical NLP Pipeline；探针/跨层顺序 |
| 70 | But Is It Really in ROME? An Investigation of the ROME Model Editing Technique | 待核原文 | QL7J9wmS6W2fWpofd；原生批评，已有ROME论文不能替代 |
| 70 | TracrBench: Generating Interpretability Testbeds with Large Language Models | 待核原文 | vNubZ5zK8h；生成可验证测试床 |
| 70 | Do Sparse Autoencoders Find True Features? | 待核原文 | QoR8noAB3Mp2KBA4B；原生讨论，须读明确的特征/恢复定义 |
| 70 | [Function Vectors in Large Language Models](https://arxiv.org/abs/2310.15213) | 已在库（本批未重读） | interp：Function Vectors in Large Language Models；已纳主张边界仍有效 |
| 70 | [Steering Language Models with Activation Engineering](https://arxiv.org/abs/2308.10248) | 已在库（本批未重读） | interp：Steering Language Models With Activation Engineering；按既有条目核对 |
| 70 | [Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting](https://arxiv.org/abs/2305.04388) | 已在库（本批未重读） | interp：Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting；解释忠实性 |
| 71 | [Investigating Gender Bias in Language Models Using Causal Mediation Analysis](https://arxiv.org/abs/2004.12265) | 已在库（本批未重读） | interp：Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias；因果中介 |
| 71 | The System Model and the User Model: Exploring AI Dashboard Design | 待核原文 | 2305.02469；解释界面与使用者模型 |
| 71 | [Information-Theoretic Probing with Minimum Description Length](https://arxiv.org/abs/2003.12298) | 已在库（本批未重读） | interp：Information-Theoretic Probing with Minimum Description Length；探针定义；纳入来源不把MDL改成SAI主轴 |
| 71 | [Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned](https://arxiv.org/abs/1905.09418) | 已在库（本批未重读） | interp：Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned；头剪枝与功能 |
| 71 | Neurons in Large Language Models: Dead, N-gram, Positional | 待核原文 | 2024.findings-acl.75；神经元工作类型 |
| 71 | Visualizing Weights | 待核原文 | visualizing-weights；原生Distill权重解释 |
| 71 | Grokked Transformers Are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization | 待核原文 | ns8IH5Sn5y；grokking机制 |
| 71 | Loss Landscape Geometry Reveals Stagewise Development of Transformers | 待核原文 | 2JabyZjM5H；跨域训练动力学，需核版本身份 |
| 71 | [Interpretability in the Wild: A Circuit for Indirect Object Identification in GPT-2 Small](https://arxiv.org/abs/2211.00593) | 已在库（本批未重读） | interp：Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small；IOI |
| 71 | Knowledge Editing for Large Language Models: A Survey | 待核原文 | 10.1145/3698590；综述候选，不自动新增概念 |
| 72 | Gradient Based Feature Attribution in Explainable AI: A Technical Review | 待核原文 | 2403.10415；技术综述候选 |
| 72 | Mapping the Semantic Void: Strange Goings-on in GPT Embedding Spaces | 待核原文 | c6uTNm5erRrmyJvvD；原生嵌入空间观察 |
| 72 | Conceptual Challenges for Interpretable Machine Learning | 待核原文 | 10.1007/s11229-022-03485-5；概念论证 |
| 72 | Understanding Intra-Class Knowledge inside CNN | 待核原文 | 1507.02379；视觉概念分解 |
| 72 | Thinking Like Transformers | 待核原文 | weiss21a；RASP/程序表达 |
| 73 | Unifying and Verifying Mechanistic Interpretations: A Case Study with Group Operations | 待核原文 | 2410.07476；形式验证与有限经验须拆分 |
| 73 | Interpretability at Scale: Identifying Causal Mechanisms in Alpaca | 待核原文 | nRfClnMhVX；因果抽象 |
| 73 | ReFT: Representation Finetuning for Language Models | 待核原文 | fykjplMc0V；表征微调 |
| 73 | A Reply to Makelov et al. (2023)'s Interpretability Illusion Arguments | 待核原文 | 2401.12631；同一争论的独立回复原文 |
| 73 | Decomposing the QK Circuit with Bilinear Sparse Dictionary Learning | 待核原文 | 2ep6FGjTQoGDRnhrq；原生双线性QK字典，不能以MLP双线性替代 |
| 73 | Exploring the Whole Rashomon Set of Sparse Decision Trees | 待核原文 | WHqVVk3UHr；多重解释/可解释模型 |
| 73 | The Unreliability of Explanations in Few-Shot Prompting for Textual Reasoning | 待核原文 | 2205.03401；自然语言解释忠实性 |
| 73 | GPT-2's Positional Embedding Matrix Is a Helix | 待核原文 | qvWP3aBDBaqXvPNhS；原生位置嵌入观察 |
| 73 | Jump to Conclusions: Short-Cutting Transformers with Linear Transformations | 待核原文 | 2024.lrec-main.840；层间线性捷径 |
| 73 | [Understanding Neural Networks Through Deep Visualization](https://arxiv.org/abs/1506.06579) | 已在库（本批未重读） | interp：Understanding Neural Networks Through Deep Visualization；视觉源头 |
| 73 | Mechanistic Understanding and Mitigation of Language Model Non-Factual Hallucinations | 待核原文 | 2024.findings-emnlp.466；幻觉机制与干预 |
| 73 | Characterizing Mechanisms for Factual Recall in Language Models | 待核原文 | 2023.emnlp-main.615；事实提取机制 |
| 74 | [Transformer Visualization via Dictionary Learning: Contextualized Embedding as a Linear Superposition of Transformer Factors](https://aclanthology.org/2021.deelio-1.1/) | 本批原文已补 | interp：Transformer Visualization via Dictionary Learning: Contextualized Embedding as a Linear Superposition of Transformer Factors；本批已读原文 |
| 74 | [Visualizing and Understanding Convolutional Networks](https://arxiv.org/abs/1311.2901) | 已在库（本批未重读） | interp：Visualizing and Understanding Convolutional Networks；视觉源头 |
| 74 | Instilling Inductive Biases with Subnetworks | 待核原文 | 2310.10899；子网络先验 |
| 74 | Interpretable Deep Learning Under Fire | 待核原文 | interpretable deep learning under fire；解释攻击，USENIX2020 |
| 74 | [Object Detectors Emerge in Deep Scene CNNs](https://arxiv.org/abs/1412.6856) | 本批原文已补 | interp：Object Detectors Emerge in Deep Scene CNNs；本批已读原文 |
| 74 | Do Feature Attribution Methods Correctly Attribute Features? | 待核原文 | 10.1609/aaai.v36i9.21196；归因评测 |
| 74 | How Well Do Feature Visualizations Support Causal Understanding of CNN Activations? | 待核原文 | vLPqnPf9k0；可视化因果理解评测 |
| 74 | [Representation Engineering: A Top-Down Approach to AI Transparency](https://arxiv.org/abs/2310.01405) | 已在库（本批未重读） | interp：Representation Engineering: A Top-Down Approach to AI Transparency；表征干预 |
| 74 | Improving Alignment and Robustness with Circuit Breakers | 待核原文 | 2406.04313；表征干预与鲁棒性 |

未列入上表的上下文／跨域条目仍有：Provably Safe Systems (2309.01933)、Mass-producing Failures of Multimodal Systems、Automating Psychological Hypothesis Generation (10.1057/s41599-024-03407-5)、AI Sandbagging (2406.07358)、参数-函数映射简单性、Honesty Is the Best Policy、Neural Network Acceptability Judgments、Watanabe 2009教材、Emergent Abilities、Chain-of-Thought、Deep Learning Is Singular、Formal Methods for Autonomous Systems、Model Soups、Universal and Transferable Adversarial Attacks (2307.15043)，以及英国AI战略/安全机构报告、美国版权法资料。它们未在本批逐项回到正文；书籍、政策/法律与技术论文不能仅因同在书目就当作同类paper计数。本批不就法律现状作判断。后续按研究对象与跨域归属选取，不把暂缓当作证据闭合。

截至第三批及Gemini跨库补录，旧pp.62–67的59行状态为：8行原批纳入、20行既有、22行后续已补（含pretrain的Gemini）、1行指定评论原文已核、8行待核。该59行仍包含重复引文和跨页条目，不是59篇独立paper。当前已到达书目末尾，但多项原文尚待核，不宣称该综述引用链或学科已全覆盖。
