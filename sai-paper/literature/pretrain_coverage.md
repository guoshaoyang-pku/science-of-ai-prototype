# 预训练文献覆盖与核验记录

更新时间：2026-10-28。配套数据为 [pretrain.json](pretrain.json)，PDF 位于 `../papers/pretrain/`。这是一批可扩充的种子集，不是完整的系统综述或已完成逐篇精读的分类。

## 当前覆盖

共 **344 篇论文／技术报告**、**305 个概念条目**。当前分类与文件状态以 [总表](catalog.md)、[分层目录](by_level.md) 和 [机器审计](audit.json) 为准；混合贡献可跨层出现，不把条目比例当作领域总体的比例。论文不是单一真值主张：模型、算法和实验可以并存；标签不能直接推断作者全部论断的证据成熟度，也不评价工作的重要性。AdamW 的解耦算法仍为 Knowledge；索引对应主张为 §2 中条件性的等价与不等价命题，故归 Science。FlashAttention 的特定 IO 与复杂度分析归 Science，算法与硬件速度实测另属 Knowledge。对涌现度量分析，索引主标签降为 Knowledge：一部分实验反例不能证明所有涌现均为度量假象，条件性数学示例另列 Science。

| 方向 | 当前代表性内容 |
|---|---|
| 语言预训练历史和目标 | 反向传播、LSTM、神经概率语言模型、RNNLM、seq2seq/attention、word2vec、ELMo、ULMFiT、GPT-1/2/3、BERT、XLNet、RoBERTa、T5 |
| 模型家族和架构 | Transformer、Gopher、PaLM、LLaMA 1–3、Qwen2.5、DeepSeek-V3、GPT-NeoX、MoE、Switch、GLaM、Mixtral、Mamba/SSM |
| 组件和训练方法 | BPE、SentencePiece、RoPE、ALiBi、Xavier/He 初始化、Dropout、DropConnect、ResNet、随机深度、BatchNorm、LayerNorm、RMSNorm、Pre-LN、DeepNorm |
| 优化与动力学 | Robbins–Monro、AdaGrad、Adam、AdamW、AMSGrad、Shampoo、Adafactor、Lion、Muon、Hyperball、梯度裁剪、学习率预热、批量大小、SGDR、edge of stability、μP |
| 数据 | Pile、去重、RefinedWeb、Dolma、FineWeb、DataComp-LM、RedPajama、C4 审计、DoReMi、数据混合定律 |
| 视觉自监督与多模态 | CPC、SimCLR、MoCo、BYOL、SimSiam、CLIP、ViT、MAE、BEiT、DINOv2、Flamingo、BLIP-2、LLaVA；视频 VideoMAE；仍非全面覆盖 |
| 音频与跨语言 | wav2vec 2.0、HuBERT、Whisper、XLM-R、mT5；任务和监督方式各异，仍有大量遗漏 |
| 数值与分布式系统 | 混合精度、Megatron 层内模型并行、ZeRO 状态分片、FlashAttention；只代表主要节点 |
| 经验规律 | Kaplan、Chinchilla、早期及多模态缩放、数据受限、Kaplan/Chinchilla 分歧、double descent、指标与涌现 |
| 可解理论 | NTK、lazy training、Tensor Programs、Pre-LN 初始化分析、Hyperball 的几何与均方分析、部分理论 scaling |
| 测量 | BIG-bench、HELM、Paloma，按其指定评测对象记录，不把整个 benchmark 强行视为一个科学主张 |

## 已核验和未核验

- 此前 150/150 文件从一手 arXiv、作者或出版方公开 PDF 获取，均通过 `%PDF`、`pdfinfo`、SHA-256 检验。下载过程曾将随机深度错配为 DenseNet 的 arXiv ID；身份校验拒绝写入，现改为 `1603.09382` 并下载验证。
- 146/150 的首页文本与记录题名规范化后完全匹配；SGDR、Dolma、反向传播和 LeCun 1998 因多栏／扫描未自动匹配，已通过 PDF 首页图像或原文文本核对题名，确认均为目标论文。
- GPT-1/2 已查看指定方法章节；Hyperball 查看了指定理论假设、定理和实证章节；Hyperball 反思论文查看了摘要与径向／切向分解；Kaplan 与 Chinchilla 查看了摘要及指定章节；Muon 和 lazy training 查看了摘要；Tensor Programs IV 已查看指定定理陈述。**这些均不是全文逐段审读。**
- 此前 150 篇均核查过指定正文、图表或定理陈述；具体位置及边界见 [原文阅读卡](pretrain_reading_cards.md) 和索引每条的 `review_level`。这不是全文逐段审读。不能把一手元数据核验记成摘要精读，也不能把摘要精读记成全文审读。
- 概念的归类均为初步映射；所有 Law 的经验范围、Science 的假设及其对现实系统的映射仍需逐条审核。

## Hyperball：本轮核对的具体边界

按用户指定，将 Hyperball 的几何／优化理论主张归 **Science**；将其具体算法及一定模型预算下的加速结果另记为 Knowledge。[Hyperball 论文](https://arxiv.org/abs/2606.16899) v2 §4.2 的假设 4.1–4.2 包括无限历史、独立同分布各向同性高斯梯度和理想化的基优化器；其径向递推写在 \(E\lVert W_t\rVert_F^2\) 上，角度计算还涉及期望构造的代理量。不能据此声称任意现实 LLM 的每条随机轨迹都严格收敛到一个确定范数。论文的预训练实验覆盖所报告的 Qwen3 风格模型，不能自动扩展为所有架构与任务的效果。

[Hyperball May Not Be a Free Lunch](https://arxiv.org/abs/2607.22444) 将更新分为径向和切向分量，并用特定调度干预探讨表现差异。其对当前一步的直接效应分析不能排除径向更新对后续优化的间接影响；实验结论也不代表已经识别所有场景的唯一机制。这是有边界的后续研究，并不直接反驳原论文在自身假设内的数学关系。

## 仍需补齐

1. **历史主干**：虽已补统计神经语言模型、LSTM、seq2seq，早期统计 n-gram、经典词表示、语言模型迁移及数据谱系还需系统补齐。
2. **预训练机制与工程**：初始化、残差、dropout、warmup、数据混合已加入，但课程学习、数值精度、分布式训练与并行、训练稳定性仍覆盖不均。
3. **多模态和非语言预训练**：已有 ViT、图像自监督和 CLIP，但音频／视频、跨模态基础模型及其他科学领域预训练仍明显不足。
4. **开放/闭源模型时间线**：已补 GPT-4 技术报告，但架构和数据等训练细节未公开；其他有代表性的模型、数据报告仍缺，不应以一条 LLaMA/Qwen/DeepSeek 线替代整个领域。
5. **理论与反例**：最优化、泛化、特征学习、loss-to-capability、缩放关系破坏与不同训练协议的反例；应追加具体定理、图表和必要／充分边界，而非仅按题名分级。
6. **系统化检索和同行复核**：需要设定截止日期、数据库与检索式，追加按引用图谱滚动检索及独立编码；现有 150 项不构成“全部经典论文”的证明。

若继续扩充，优先补上述缺口，并为已经下载的高影响论文逐条定位定理、图表、模型／数据边界、反例和证据等级；新增论文、概念和原文均应保持可追溯。

## 本轮跨模态与语言目标扩展

新增 24 篇与 15 个概念：文本去噪／替换检测、多语言与混合目标、语音（wav2vec 2.0、HuBERT、Whisper）、视频（VideoMAE）、视觉自监督、图文连接器、长序列与注意力。24 份 PDF 已核对首页题名、格式、页数和 SHA-256；仅局部查看摘要，索引统一保守记录为元数据与主张初筛，方法章节、对照条件和实验还需逐项核查。弱监督 Whisper 与无标签音频自监督分开；线性核注意力、随机特征近似和普通 softmax 注意力也不混同。

## 第四轮定向复核

定向核查 20 篇此前仅元数据初筛的文章，包括归一化、位置编码、稀疏专家、状态空间模型、多模态音频与语言目标。仅查看索引所列方法位置，并非逐项独立复现或证明审计。修正了 DeepSeek-V3 同时采用补充序列级辅助平衡损失这一容易遗漏的条件；SSD 定理 3.5 单列条件 Science，而 Mamba-2 的模型和运行结果保持 Knowledge。将涌现指标的具体实验概念由 Law 改为 Knowledge，独立假设下的数学例子另列 Science；FlashAttention 的 IO 分块标明定理中的 SRAM 范围及硬件模型。详情见 [第四轮阅读卡](pretrain_reading_cards.md)。


## 第五轮定向复核

本轮新增 20 篇指定章节核查，预训练清单仍为 150 篇、67 个概念；当前 100 篇已核查指定正文或定理，3 篇查看摘要／首页，47 篇仍是元数据初筛。覆盖包括词向量、子词、优化、深残差、grokking、VQ-VAE 和长文档注意力。详细主张与限定见 [第五轮阅读卡](pretrain_reading_cards.md)。定向核查不等于完整证明审计或独立实验复现，新增和既有条目依旧需要对关键边界和反例继续评议。对可解 scaling 模型的现实映射已改为待验证的作者理论动机。

## 第六轮定向复核

本轮将 21 篇由元数据初筛推进至指定正文、方法或图表核查；当前共 150 篇、67 个概念，其中 121 篇定向核查、3 篇摘要或首页、26 篇仍为元数据初筛。新增核查覆盖 Shampoo、Lion、Weight Normalization、LAMB、warmup、线性 batch 缩放、ALBERT、XLM-R、mT5、BYOL、SimSiam、线性注意力、Performer、S4、BEiT、VideoMAE、Flamingo、UL2、BLIP-2、DINOv2 与 LLaVA。重点区分机制假说、有限基准实测和形式结论：BYOL 的目标仍允许坍缩解；Performer 的核估计与归一化注意力输出不能混为一谈；mT5 的 101 种语言受 cld3 的识别与计数口径约束；LLaVA 数据生成教师只看到文字化的 caption 和框。详情见 [第六轮阅读卡](pretrain_reading_cards.md)。定向核对不是逐行证明审计或实验复现。

## 本轮：理论与跨规模研究核查

将 18 篇从元数据或摘要初筛推进至指定方法、公式或定理陈述，包括四篇 Science／Law（Tensor Programs IV、Robbins–Monro、Rosenfeld 跨规模拟合、Henighan 多模态 scaling）及 14 篇语言模型、数据和词表示研究。范围、章节与未完成的证明审计分别见 JSON 和阅读卡；上述核查不等于阅读全文或复现实验。Rosenfeld PDF 提取全文时出现 xref 重建提示，文本可读但文件仍需谨慎处理；Mikolov 2010 PDF 的部分流有解析警告，仍为元数据初筛，未冒称已完成方法核查。

## 本轮：补足剩余初筛条目的指定正文

本轮将此前 9 篇元数据初筛及 GPT-1/2 两篇摘要级记录推进到指定正文。覆盖 GPT-1 的预训练与监督微调、GPT-2 的 WebText 与零样本、Llama 3 两阶段配方、优化器公平比较、BIG-bench/HELM 评估范围及 BLOOM 的 ROOTS 语料。1986 反向传播、1997 LSTM 和 1998 文档识别 PDF 字体抽取乱码，已直接查看各自指定的渲染页：分别核查梯度与动量更新、原始 CEC 与输入/输出门、文档识别的整体可微模块。Mikolov 2010 的 PDF 仍有 endstream 解析警告，但第二页的 RNN 更新、softmax 和 τ=1 训练文字可读，已仅据该页补充方法；文件其他页面仍须谨慎处理。优化器论文 PDF 提取有字体类型警告，具体位置仍可读。

**150 项均有定向位置，不表示完成全文精读。** 所有模型的完整数据、证明与实验均未逐项审核，定理证明未逐行审计，实验未独立复现。更广泛的经典文献和概念仍未收全。

## 本轮：优化几何、经验规模与训练系统补缺

新增激活重计算、彩票假说、尖锐度重参数化反例、NMT scaling、neural collapse、SAM、GPT-2 训练数据提取和 FlashAttention-2，共 8 篇论文与 8 个概念。新原文均从 arXiv 获取，首页标题、arXiv 编号、pdfinfo 页数、大小与 SHA-256 已核验；已阅读摘要及 PDF 第 2 页，未完整审读证明或所有实验。Dinh 的条件化反例不排除其他与参数化无关的几何度量；NMT 的经验 Law 限于作者研究的模型和语料；训练数据提取不意味着任意样本可被恢复。

## 本轮：八篇原文深化及长上下文补缺

对激活重计算、彩票假说、尖锐度反例、机器翻译 scaling、Neural Collapse、SAM、GPT-2 训练数据提取和 FlashAttention-2 共 8 篇，从摘要/首页推进至索引所标的方法、图表或定理陈述；其完整证明和实验仍未审查。修正两项易混淆条件：剪枝彩票需要反复训练与重置，不能把发现彩票直接当作免费训练加速；SAM 的实际每批近似更新与全数据精确内最大化并非同一过程。Dinh 的尺度变换只提供指定参数化与度量下的反例。

新增位置插值、Lost in the Middle、StreamingLLM 和 QLoRA 四篇原论文及四个概念，均定位方法或实验设置并核验文件结构、大小、SHA 和首页标题与 arXiv ID。StreamingLLM 维护有限缓存，作者明确指出并不扩大可访问上下文；位置插值的窗口延长亦不保证对所有历史的有效使用。所有新增条目和上述 8 篇均未逐行审计完整证明或独立复现实验。
## 本轮：微调、涌现原始论述与视觉生成

补入 LoRA、Prefix-Tuning、FLAN、Emergent Abilities 原文、MQA、GPTQ、DALL-E 和 iGPT，共 8 篇及 8 个概念。前 7 篇来自 arXiv，第 8 篇来自 PMLR 官方 ICML 论文页；PDF 格式、页数、大小、SHA-256 与首页标题均通过核验，7 篇 arXiv ID 对应首页。已定位各自的关键方法或经验论述，未完整审读所有证明、图表或独立复现。

将 Wei 等涌现原始经验主张按 Knowledge 记录，并与先前收录的指标与采样批评成对阅读：已观察到的表观突变不自动构成普适 Law，也不能由指标反例推断全部真实涌现不存在。DALL-E 的两阶段方法区分 32×32 图像 token 与文本 token，且论文中的加权目标只在 β=1 时对应所写的证据下界；iGPT 也含 VQ-VAE 和低分辨率输入变体，不一概称作对所有实验的原始 RGB 序列建模。LoRA 的可合并矩阵、前缀长度、FLAN 任务重叠、MQA 硬件与 GPTQ 校准均为相应范围的一部分。
## 本轮：提示微调、图文自监督与数据边界

新增 T0、Barlow Twins、ALIGN、SigLIP、Reformer、Universal Transformer、递归生成数据模型崩塌和 phi-1 教材质量数据共 8 篇与 8 个概念。PDF 均通过文件结构、页数、哈希和首页标题核验；7 篇首页显示匹配的 arXiv ID，SigLIP 首页无 ID，标题及作者与一手来源吻合。已查各篇 PDF 第 2–4 页的指定方法或研究设定，未完整审计证明和实验。

范围上，ALIGN 使用简单过滤而非完全无过滤图文数据；SigLIP 没有 softmax 全局分母不表示没有负样本；Reformer 的 LSH 注意力是近似；Universal Transformer 的图灵完备性依赖理论假设且本轮未审核证明。模型崩塌论文中的递归采样与微调设置不能推出所有合成数据训练必然失败；phi-1 的训练质量、数据构成及比较设置不能单独确定唯一因果因素。全部新增主标签暂按所列算法或有限实验归 Knowledge，不以标题或摘要强加 Law/Science。


### 一手主张复核补充（2026-10-28）

针对 12 项既有主张和 2 篇新增论文，逐项记录原标签、裁定、原文章节、作用域和未决事项，见 [预训练主张复核](../survey/pretrain_adjudication.md)。NTK 理论在附录中使用逐层依次趋宽的证明，数值实验的 ReLU 不满足定理 2 的二次可微条件；索引与阅读卡已据此更正。补入自适应优化与 SGD 泛化对比、真实数据与随机数据的记忆行为两篇论文和概念，均仅对所列章节定向阅读，不代表全文精读或复现实验。


## 补充：架构、激活、表征与优化隐式偏置

新增 8 篇原始论文和 8 个概念：AlexNet、VGG、DenseNet、ELU、GELU、SimCSE、模型校准与温度缩放，以及线性可分数据上梯度下降的隐式偏置。8 份一手 PDF 均已下载，确认 PDF 可解析、篇名一致并记录页数、字节数和 SHA-256；arXiv 版本核对了标识，AlexNet 取 NeurIPS 原版。索引给出各自主张、范围和正文页码；方法及实验结论仍限于原文所述 setting，不能推广为普适定律。

Soudry 等的分类依据其条件性定理，非全部论文或优化方法均为 Science：结论针对有限线性可分数据、齐次线性预测器、满足特定损失条件及足够小的固定步长的梯度下降。更强的有界残差结论额外限定为几乎所有数据集。已核对定理陈述，尚未独立逐步审查证明。其他 7 篇按索引中的具体方法及实验主张归 Knowledge；未全面复核实验表或全文。

此外，Hyperball May Not Be a Free Lunch 已从摘要加单一节定位推进到 PDF 第 4–8 页的角有效学习率、径向／切向分解与学习率匹配设计；单步影响较小属于原文特定设置下的数值观察，不排除长期间接作用。尚未逐步审计推导或复现实验。

## 补充：生成建模、波形和流水并行

新增 OPT、GPipe、Mistral 7B、变分自编码（AEVB）、DDPM、潜空间扩散、PixelRNN、DDIM、WaveNet 和分数 SDE 十篇及十个概念。十份原文均来自 arXiv，PDF 结构、首页题名和编号、页数、大小及 SHA-256 已核对；已核对索引中指定的正文和公式，但未全文审读、逐行审计证明或复现。

分数 SDE 将精确边缘分数下的反向时间过程作为条件性 Science 候选，神经网络拟合与有限步采样则是 Knowledge；反向 SDE 公式引用 Anderson 1982，尚未独立审核完整正则条件及证明，不能保证实际生成模型精确采样。DDPM 的简化损失经过重加权；DDIM 的少步采样有质量与离散误差权衡。GPipe 批末同步梯度，但批归一化使用微批统计，不能声称所有模型与整批训练逐步完全等价。上述音频和图像生成模型是预训练谱系中的相关工作，不代表对每个子领域的完整覆盖。

## 补充：扩散、分数匹配、对抗生成与可逆流的理论谱系

新增 12 篇原始论文和 12 个概念，补上 Anderson 1982 反向时间扩散、Hyvärinen 2005 分数匹配、Vincent 2011 去噪分数匹配，及 2015 年扩散生成、2019 年噪声条件分数网络、GAN/WGAN、NICE/Real NVP/Glow、Flow Matching 与 Rectified Flow。PDF 的结构、首页身份、页数、文件大小与 SHA-256 均已核验；索引记录所查正文或定理位置，均未逐行审计完整证明或独立复现实验。Anderson 正文取大学公开教学站镜像并比对出版方题名与卷页；镜像首页文字经扫描提取，公式需结合原页查阅。

科学主张只在原文模型和假设内成立：Anderson 定理依赖概率密度及联合过程对应偏微分方程的存在、光滑与唯一性，状态依赖扩散的漂移修正包含扩散矩阵导数。Hyvärinen 的目标变换依赖可微、边界和积分条件；Vincent 的等价对应指定平滑分布和目标，不把其视作对任意原始分布的等价。GAN 的全局最优性依赖分布层面的理想最优判别器，WGAN 的连续性定理与有限网络训练过程须分开。

对于连续流，Flow Matching 的目标梯度等价要保留 Theorem 2 的正密度等假设；Rectified Flow 的凸位移代价不增依赖可 rectifiable 耦合与精确流，且论文指出直路径一般不是最优运输的充分条件。NICE 的加性耦合层体积保持不意味着附加缩放后的完整模型必然体积保持；Real NVP 和 Glow 的精确似然指模型密度，不证明模型精确拟合真实数据。这些有限模型、数值求解与现实系统的映射均需独立检验。


## 补充：训练数据选择、并行与早期预训练

新增 13 篇论文与 13 个概念，涵盖 DSIR、SemDeDup、领域／任务适配继续预训练、Megatron 组合并行、8-bit 优化器状态、序列并行与选择性重计算、Ring Attention、数据剪枝与 scaling、unigram 子词随机化、受控评估数据污染、分层词概率、早期逐层预训练，以及课程学习。所有原文均经 PDF 签名、pdfinfo 页数与 SHA-256 核验，首页题名匹配；9 篇 arXiv 首页另核对其编号，数据剪枝论文采用可正常解析的 NeurIPS 正式 PDF。所列页码为定向核查位置，不代表全文审读、所有证明审计或实验复现。

数据剪枝论文单列两个贡献：在所给教师—学生感知机和 replica 分析中，完美探针对应的 Pareto 包络需要随数据规模改变保留比例；固定保留比例时最终回到幂律。原文把理论映射到视觉模型，真实性能、筛选成本和理论假设是否适用于其他训练需分别评估；理论推导尚未逐行审计。SemDeDup 判断的是所选表征与阈值中的相似性，图文实验只使用图像来识别重复；DSIR 匹配的是给定降维特征分布，不保证完整数据分布一致。Ring Attention 的通信完全隐藏依赖计算时间覆盖通信时间，不能将硬件相关性能表述为无条件零成本。

## 补充：蒸馏谱系、片段预训练与对比表示理论

新增 12 篇论文与 12 个概念：Hinton 蒸馏、序列级知识蒸馏、DistilBERT、TinyBERT、Born-Again Networks、Deep Mutual Learning、Mean Teacher、SpanBERT、MASS、DeBERTa，以及对比学习的对齐／球面均匀性和潜在类别理论。12 份 arXiv 原文均核验 PDF 结构、首页题名、arXiv ID、页数、大小和 SHA-256，并定位索引所列方法、公式或定理陈述；未逐项复现实验或完整逐行审计证明。

十项方法或指定实证主张暂记 Knowledge。Born-Again Networks 的同架构学生不是压缩方案的必然形式；Deep Mutual Learning 是同步对等训练而非固定预训练教师；Mean Teacher 的教师参数是学生参数的移动平均。DistilBERT 所报告的速度和保留比例与其指标、任务、设备和训练配方相关，不能外推为任何模型压缩都达到同一比例。SpanBERT 的片段边界目标与 MASS 的编码器—解码器掩码片段自回归重建是不同目标；DeBERTa 的内容／位置分离不等于证明语义因子可辨识。

另外两篇的条件化理论记为 Science／Knowledge：Wang–Isola 2020 的定理在固定温度、单位球面和无限负样本极限下分析对比损失，并在完美均匀编码器存在时刻画均匀项最小值；不能据此保证有限负样本训练必然获得完美均匀表示。Arora 等 2019 的界建立于潜在类别采样、函数类和损失假设，讨论的是平均二分类任务及均值分类器，并包含类别碰撞与样本复杂度条件；不是任意数据或下游任务的性能保证。两项均仅定位定理陈述，附录证明未逐行审计。

## 补充：视觉预训练架构、增强与优化（2026-10-28）

新增 14 篇原始论文和 14 个概念，补充 DCGAN、BigGAN、StyleGAN、Group Normalization、AdaDelta、SELU 自归一化、RAdam、mixup、CutMix、RandAugment、Inception、Wide ResNet、EfficientNet 和 Swin Transformer。14 份 arXiv 原文均核对 PDF 签名、首页标题和编号、页数、大小及 SHA-256；定向查看索引列出的正文及公式，未完整审读、逐行审计证明或复现实验。

DCGAN、BigGAN 与 StyleGAN 分别研究卷积生成对抗架构、类条件大规模训练与截断采样、中间潜变量风格调制，不把它们的生成质量当作跨模型普遍规律。mixup 的凸组合标签和 CutMix 的面积权重是训练构造，不证明每个混合样本具备对应真实语义；RandAugment 缩小搜索空间，但仍需设置操作数和增强幅度。Group Normalization 的组内统计不跨批次并不保证其在所有小批次任务都优于 BatchNorm；Swin 在固定窗口大小下的空间复杂度表述不等于所有通道、硬件与窗口配置的端到端延迟线性。

SELU 论文中的前向激活均值和方差映射被单列为条件化 Science 候选：其标准化权重矩及正态近似下，映射在 \((0,1)\) 的 Jacobian 范数小于 1；指定映射的局部结论不保证任意训练模型、注意力架构或反向梯度稳定。已核对论文的映射和局部稳定陈述，未逐行独立审核证明；作者模型与实验则按 Knowledge 单列。RAdam 对 warmup 的方差解释保留为论文所述机制，不排除其他机制。

## 补充：序列训练、稀疏注意力、词元与专家分配（2026-10-28）

新增 12 篇论文与 12 个概念，覆盖输入／输出嵌入绑权、scheduled sampling 与其理论批评、Professor Forcing、ByT5 字节级训练、BigBird、Sparse Transformer、BASE、GShard、GNMT、AWD-LSTM 和标签平滑。12 份 arXiv 原文均核验 PDF 签名、首页标题与 arXiv 编号、页数、大小和 SHA-256；定向查看条目所列方法及定理陈述，不代表全文审读、逐行证明审计或实验复现。

- Scheduled sampling 方法按 Knowledge 标记；Huszár 的分析按 Science 候选记录，严格限定为长度为二的模型前缀采样与作者的理想化总体目标。真实变量独立时，边缘分布乘积就是联合分布；论文自己说明 argmax 变体更难分析。此处未独立审计 KL 推导与采样中停止梯度实现的对应，不能断言所有有限训练变体必定失败。
- BigBird 的 Theorem 1 规定连续函数、固定长度有界域、包含星形图的稀疏 Transformer 与积分 Lp 距离，得到存在性近似结论；既不保证有限模型可由梯度下降找到，也不等同逐点误差保证。其图灵完备性需额外的任意精度等假设。具体注意力方法与实验另记 Knowledge。
- BASE 训练时按批次使用平衡的专家分配；测试时为避免泄露未来 token，逐 token 选择最高分专家，因而不保证推理负载严格均衡。GShard 使用组内 top-2 路由、容量限制、辅助损失及随机第二专家；溢出 token 可经残差传递。
- ByT5 使用 UTF-8 字节作为离散符号而非完全没有 token，其序列长度和计算、语言与扰动类型均影响效果。标签平滑混合的是 one-hot 与指定先验分布，并不保证模型恢复真实类别概率。稀疏注意边数或词表参数减少也不能直接等同端到端速度提升。

## 本轮：Hyperball 与 scaling 锚点的概念范围（2026-10-28）

针对用户指定的分类锚点，将 Hyperball 几何优化、固定权重衰减平衡半径、角向学习率、Kaplan scaling 和 Chinchilla 计算最优共 5 条概念从初步来源映射升级为指定原文章节核对。Hyperball 使用本地 v2 §4.2：无限历史、独立各向同性高斯输入和理想化基优化器是模型假设；递推针对范数平方的期望，不能改写为每条轨迹趋于固定半径。对照论文 v1 的小角度有效学习率还依赖参数与更新的夹角。Kaplan 的参数、数据和计算受限拟合分别有条件；Chinchilla §3.2 的 0.49／0.51 是等 FLOPs 方法，不能与其他拟合方法混用。

上述更新核对了定位、方法或定理陈述，尚未逐行审计证明、复拟合曲线或复现实验。截至当时，其余 **40 条**预训练概念仍明确标为初步来源映射，不能因为关联论文已下载就视为概念已经独立核查。


## 本轮：原有 40 条初筛概念的一手来源定向复核（2026-10-28）

对此前 `review_level=provisional_concept_mapping` 的 40 条概念逐项打开一手原文指定正文，对概念定义、范围及证据位置作独立核对并更新条目；当前该标记剩余 0 条。范围涵盖自回归与排列目标、BPE/SentencePiece、注意力与 MoE、Mamba、归一化、噪声尺度、数据重复与去重、双下降、grokking、RNN/LSTM、视觉／语音／多模态预训练和长序列架构。

关键修正包括：SentencePiece 的可逆性相对规范化后文本；XLNet 排列的是因子化顺序；ELECTRA 的生成器采样恰等于原 token 时判别标签为真实；BYOL 损失允许常量坍缩，避免坍缩仍是作者机制假说；Performer 对软最大核的随机特征估计性质不自动传递为归一化输出无偏；原始 LSTM 的 CEC 与后来带遗忘门变体不能混为一谈。

方法或定理陈述仅作指定页面定向核查，不是全文审读、独立逐行证明审计或实验复现。RNNLM PDF 的第 2 页提取时出现 endstream 解析警告但文本可读；LSTM 的字体编码失真，直接渲染第 3、5、7 页核对。多来源概念（MoE、数据过滤、离散潜变量）分别查看各引用论文的相关正文，证据位置已写入各概念。


## 2026-10-28 补充：检索增强与外部记忆

新增 8 篇公开 arXiv 原文和 8 个概念，全部按具体方法主张暂列 Knowledge，未把单篇基准性能或工程配方提升为普遍 Law。每篇 PDF 已核对首页题名与 arXiv 编号、页数、字节数和 SHA-256，并定向阅读 JSON 中列明的方法章节；不表示全文精读、重新运行或系统综述。

- ORQA：逆完形任务初始化检索器，弱监督 QA 联合学习；DPR 采用已标注问答段落与负例训练双编码器；REALM 把潜文档检索引入掩码语言模型预训练。三者的监督和训练阶段不同。
- kNN-LM 是已有模型的推理期近邻分布插值；RETRO 则在训练及评估中接入分块文本检索与交叉注意力。RETRO 文中默认训练检索库为 600B token，评估检索库约 1.75T token；具体成本、去重与检索库设置须随结果一起报告。
- RAG 的序列级与词元级边缘化不同，原实验训练时固定文档编码器和索引；FiD 在解码器融合分别编码的多个段落；Atlas 结合检索器和 FiD 并研究小样本适配。索引可更新不等于事实正确性得到保证。


## 2026-10-28 补充：生物序列与分子图预训练

新增 10 篇原论文与 10 个概念：Rives 等的蛋白语言模型、ProtTrans、ProGen、MSA Transformer、蛋白突变零样本预测、DNABERT 及 DNABERT-2、图神经网络预训练策略、GROVER 和 GraphMVP。10 份 PDF 均检查签名、首页题名、页数、大小、SHA-256，并定位各条方法章节；仅是定向核查，未全文审读、独立核验理论或复现实验。Rives 和 DNABERT 的期刊论文 PDF 取自公开镜像，条目的 `url` 分别指向期刊 DOI，`pdf_source` 记录实际镜像地址；这与从出版社直接下载不同。ProtTrans 的 PDF 首页题名与 arXiv 摘要页题名有版本变化，索引采用实际下载 PDF 首页的题名。

蛋白质单序列掩码、含功能标签的条件生成和多序列比对建模使用不同输入与监督信息：MSA Transformer 在接触任务上使用少量已知结构拟合逻辑回归读出，因此其语言模型自监督预训练不意味着整个接触预测流程完全无监督。突变预测的零样本定义不排除预训练集出现相关蛋白家族。DNABERT 的重叠 k-mer 可从未掩码邻居泄露片段信息；DNABERT-2 的 BPE 及具体基准改进也不是对所有基因组任务的最优性保证。图预训练策略的图级监督阶段可使用标签，GraphMVP 的 3D 信息存在于预训练阶段，并不能从二维拓扑唯一确定三维构象。

这 10 个方法及限定实验暂归 Knowledge；生物序列、分子和科学预训练领域的经典工作仍有明显缺口，包括 ESM-2/ESMFold、更多结构数据与独立测试切分，以及蛋白序列模型与任务监督关系的系统比较。


## 2026-10-28 补充：蛋白结构预测、逆折叠与生成设计

新增 9 条论文与 10 条概念：AlphaFold 1、AlphaFold 2、AlphaFold-Multimer、AlphaFold 3、ESM-2/ESMFold、ESM3、RoseTTAFold、ProteinMPNN、RFdiffusion；ESM-2 预训练与 ESMFold 折叠头分别建概念。9 份 PDF 均检查签名、首页题名、页数、字节数与 SHA-256，并定向查看各条指定章节；未全文逐行审核或复现实验。AlphaFold 1 与 ESM-2/ESMFold 的 PDF 为公开教学镜像；RoseTTAFold 为 NSF 收录的作者稿；ESM3 文件是 2024-06-25 作者预览稿而非期刊最终版本。其余来源分别为 Nature、bioRxiv 或作者机构；每条登记实际下载地址和主要一手来源。

AlphaFold 1 用距离分布构造势函数及搜索；AlphaFold 2 利用 MSA／残基对表示和结构监督；Multimer 明确处理多链；AlphaFold 3 使用 Pairformer 与原子坐标扩散。它们不应被称为从第一性原理完整推导蛋白的折叠动力学，因此所录具体模型和实验为 Knowledge。ESM-2 使用序列掩码预训练，ESMFold 折叠头还使用结构信息与教师预测；单序列推理不等于整个训练流程没有结构监督。ESM3 的序列、结构、功能轨道可以联合掩码，作者标题中的演化时间是一种基于序列相似度和模型的比较，不是实际运行了五亿年进化。ProteinMPNN 从给定骨架设计序列，RFdiffusion 产生骨架，二者的输入、输出和后续验证不同；序列恢复或预测置信度不能直接等同实验功能成功。

蛋白结构与设计只是预训练图谱相邻的科学应用路线，不证明已有目录涵盖全部生物基础模型；还需扩展结构数据、评估基准、数据泄漏及独立实验验证方面的关键论文。


## 2026-10-28 复核：条件理论、方法定义与双下降来源

本轮**新增 1 篇论文、1 条概念；复核原有 10 篇论文、9 条概念**。新增和复核分别计数，原有记录不算新增。预训练目录更新为 **289 篇论文、207 条概念**。

新增原文为 Belkin 等的 [双下降论文](https://arxiv.org/abs/1812.11118)，来自 Nakkiran 等的引用链。已按题名、作者、编号、版本、PDF签名、23页、1,171,814字节和SHA-256核验；本地为 arXiv v2（2019-09-10，首页排版日期2019-09-12），不是PNAS排版版。索引另记PNAS题名别名和DOI。新增概念“随机标签记忆与泛化解释”使用已有Zhang原文；已有“双下降”概念补入Belkin来源，不重复造概念。全部11份涉及的原文重新核对页数、字节和hash均与索引一致。

| 原有复核论文 | 本次定位与收紧之处 |
|---|---|
| Glorot–Bengio 初始化 | PDF第5–6页：2/(fan_in+fan_out)是前后向方差条件的折中，不等宽时不能同时严格满足；训练中独立性和线性近似会失效。 |
| He 初始化 | 第3–5页：区分2/fan_in与2/fan_out；ReLU输出二阶矩不等于其方差，反向还有门控/梯度独立假设。 |
| NCE | 第2–3页：把二分类算法定义与一致性定理分开；正确指定、全局最大化、支持覆盖、一致收敛及满秩条件不可省略。会议稿省略证明。 |
| AdaGrad | 第2–3、5–6、9–12页：区分对偶平均和镜像下降的条件；定理5位于§3，推论6的简化界需要有界距离和指定步长，不是非凸预训练保证。 |
| Lazy training | 第3–5页：定理2.2针对归一化目标的连续梯度流、零初始输出、固定有限时间及大缩放；局部尺度判据为充分条件，不是所有网络的必然范围。 |
| Tensor Programs V | 第3–8页：表3/定义4.1是参数化规范；宽度极限理论与有限模型近优超参数迁移实证分开，未重新证明前作或附录J。 |
| FlashAttention | 第4–6页：补条件理论与方法/实测两条主张。定理2的范围为d≤M≤Nd；命题3只排除对所有M均更优的算法，不是每个M的点态最优性。固定M、d时仍随N二次增长。 |
| AdamW | 第2–3页：标准无动量SGD的等价换算与非标量预条件器分析单列为条件理论；算法定义及具体任务收益单列Knowledge，不泛化到任意动量实现或调度。 |
| Zhang随机标签 | 第1–2、4–5页：CIFAR-10可零训练误差，ImageNet所报随机标签top-1为95.20%；随机化/记忆实验不是双下降曲线证据。 |
| Nakkiran深度双下降 | 第2–3页：模型/epoch曲线与部分样本非单调性是经验结果；EMC定义与明确非形式化的广义假说不能当作普适边界证明。 |

Belkin新增条目定向阅读第1–10页（分批处理）。随机特征使用最小系数范数解，网络使用SGD，树实验在插值阈值两边分别调叶数与树数，并省略bootstrap重采样；这些控制变量不同，不能把曲线当作单一无条件规律。小范数/平滑性解释与经验曲线按不同主张登记，附录A的特定无噪声定理未独立审计。

上述均为指定原文位置、主张及假设核对，**没有全文逐行证明审计或实验复现**。主分类仍是主张层的暂定标签；方法有公式不自动成为Science，跨若干设置重复出现也不表示已穷尽可外推范围。本轮未改panel、冻结论文及任何生成总表。


## 2026-10-28 补充：学习曲线、插值风险与数据重复的经典来源链

本轮**新增11篇论文、12条概念**，目录从289篇/207条增至 **300篇论文、219条概念**。12条概念中11条来自新增论文，另1条为已有Nakkiran原文中的“有效模型复杂度（EMC）”。本轮没有把用于追踪引文的既有论文计为新增，也没有新增一轮完整复核数；上一节10篇/9条复核数保持独立。

候选沿一手代表作的参考文献追踪，并与已有目录按标题、编号、PDF hash及概念名去重：

| 已有一手来源 | 补入的源头/分支 |
|---|---|
| Belkin 1812.11118参考文献[1]；Nakkiran 1912.02292参考文献 | Advani–Saxe高维训练动力学；Belkin–Hsu–Xu弱特征双下降；Hastie等ridgeless风险；Bartlett等benign overfitting；Mei–Montanari随机特征风险 |
| Bahri 2102.06701参考文献[10][19][28][32][59][73] | Sharma–Kaplan流形维数；Bordelon、Canatar核学习曲线；Hastie、Mei；Sollich–Halees GP学习曲线 |
| Hestness 1712.00409参考文献 | Banko–Brill 2001大语料消歧 |
| Scaling Data-Constrained 2305.16264参考文献[40] | Hernandez等重复数据 |
| 新增Bordelon §1.1、Canatar参考文献[34] | 交叉确认Sollich–Halees与Bordelon来源链 |

11份新增PDF均核对签名、首页题名与作者、实际版本、pdfinfo页数、字节数和SHA-256；具体hash写入JSON。arXiv文件下载后按首页所示版本登记，JMLR和ACL使用出版方PDF。Sollich采用cond-mat/0105015v1（2001作者预印本，期刊2002），Advani采用2017两作者v1，不能与后来的期刊作者列表混用。

| 新增原文 | 本地版本 | 页数 | 具体主张暂定分类 |
|---|---|---:|---|
| Two models of double descent for weak features | 1903.07571v2 [cs.LG], 10 Oct 2020; PDF排版日期13 Oct 2020 | 11 | Science |
| Surprises in High-Dimensional Ridgeless Least Squares Interpolation | 1903.08560v5 [math.ST], 7 Dec 2020 | 68 | Science |
| Benign Overfitting in Linear Regression | 1906.11300v3 [stat.ML], 29 Jan 2020 | 44 | Science |
| The generalization error of random features regression: Precise asymptotics and double descent curve | 1908.05355v5 [math.ST], 11 Dec 2020; PDF排版日期14 Dec 2020 | 76 | Science |
| High-dimensional dynamics of generalization error in neural networks | 1710.03667v1 [stat.ML], 10 Oct 2017; 与2020年三作者期刊版区分 | 32 | Science |
| Learning curves for Gaussian process regression: Approximations and bounds | cond-mat/0105015v1 [cond-mat.dis-nn], 1 May 2001; Neural Computation期刊版为2002年 | 25 | Science |
| Spectrum Dependent Learning Curves in Kernel Regression and Wide Neural Networks | 2002.02561v7 [stat.ML], 25 Feb 2021 | 22 | Science |
| Spectral Bias and Task-Model Alignment Explain Generalization in Kernel Regression and Infinitely Wide Neural Networks | 2006.13198v6 [stat.ML], 4 Feb 2022; Nature Communications 2021的arXiv修订稿 | 60 | Science |
| Scaling Laws from the Data Manifold Dimension | JMLR 23(9):1–34, 2022; submitted 10/20, revised 4/21, published 2/22 | 34 | Knowledge |
| Scaling to Very Very Large Corpora for Natural Language Disambiguation | ACL 2001 proceedings, pp.26–33; publisher PDF, 8 pages | 8 | Knowledge |
| Scaling Laws and Interpretability of Learning from Repeated Data | 2205.10487v1 [cs.LG], 21 May 2022 | 23 | Law |

各条证据位置、假设和预测量已写入JSON；下面是本批最重要的跨论文边界。

- **有限样本精确 vs 渐近。** Belkin–Hsu–Xu的高斯弱特征模型给有限样本期望风险，插值附近的“发散”是期望发散；Hastie的各向同性公式是p/n→γ≠1的高维极限。Mei–Montanari还需N/d、n/d均趋于正有限常数，定理中的非线性真值部分是指定各向同性随机GP；作者对任意确定非线性真值的观察不是同一证明。其λ→0等极限在比例极限之后，不能随意交换。
- **指标不同。** Belkin–Hsu–Xu和本条Advani训练曲线的测试响应MSE含新样本噪声；Hastie、Mei、GP后验方差、Bordelon/Canatar针对无噪声均值或目标；Bartlett是相对最优线性预测器的超额风险。Banko的10组消歧平均准确率与Hernandez的语言模型测试损失又是不同指标，不能直接叠成同一条scaling曲线。
- **良性过拟合需要谱条件。** Bartlett的结论来自协方差尾部有效秩及线性/次高斯/满秩条件，包含高概率上界与期望下界；p>n本身不充分。Advani的明确时间曲线依赖固定数据上的连续全批梯度流和线性高斯teacher/student，零特征值模态保留初始化；不能直接视为现代神经预训练的精确动力学。
- **谱公式的近似范围。** Sollich–Halees有正确GP先验下的风险恒等式，也构造同谱但不同学习曲线的例子；LC/UC依赖近似，UC不具有任意问题的普遍上界保证。Bordelon命题3明确位于PDE近似下；Canatar使用热力学极限、高斯特征与副本对称处理。有限曲线拟合得好不能消去这些条件，或推翻一般有限样本的同谱反例。
- **α≈4/d是解释猜想。** Sharma–Kaplan的toy分块光滑近似给条件上界，作者另将神经网络映射称为conjectural theory，并明说bound而非equality。主标签因此为Knowledge，条件toy上界单列Science；任务/表示维数代理不是一个无条件决定训练指数的第一性原理量。
- **数据量与重复量要分开。** Banko只说明指定任务和语料范围内的规模效应及算法排序变化。Hernandez区分独特子集大小、训练曝光占比和重复次数，在固定100B token预算下给经验曲线；“effective model size”按基线测试损失换算，与Nakkiran按期望训练误差定义的EMC不同。诱导头损伤或容量占用仍属特定机制证据/解释，不是已穷尽的因果边界。

**本批Science标签只表示明确假设内的理论主张，或注明近似模型内的推导；不表示证明已独立审核。** 本轮定向阅读每条列出的原文位置，未全文逐行精读、重做证明或复现实验。Law也不表示已经穷尽所有可外推设置。仅改预训练数据文件、覆盖说明与原文目录，未编辑panel、冻结论文或共享生成物。

下一批仍应补：数据质量/去重的早期测量与采样理论、统计学习曲线的更早经典（本批从GP和大语料源头切入）、以及非高斯/相关特征下的精确风险边界。当前300篇是工作目录的覆盖量，不代表所有预训练工作已分类。


## 2026-10-28 补充：PAC-Bayes、稳定性、平坦极小值与梯度噪声

本轮**新增11篇论文、14条概念**：11条概念来自新增论文，另外3条来自已有Adam、BatchNorm和word2vec两篇原文。目录从300篇/219条增至 **311篇论文、233条概念**。已有4篇只做概念所需位置的定向核对，不重新计为新增，也不称为完成全文审计。

候选均先与pretrain/interp/rl三库按编号、规范化题名及PDF hash去重。引文链为：

| 已有/新增一手入口 | 本轮补入 |
|---|---|
| 已有SAM §5与参考文献 | Dziugaite–Roy、Neyshabur等、Hochreiter–Schmidhuber |
| 已有Dinh参考文献、Keskar参考文献 | Bousquet–Elisseeff、Hardt–Recht–Singer、Hochreiter–Schmidhuber |
| 新增Neyshabur参考文献[15]；Seeger参考文献1999b | McAllester Some PAC-Bayesian Theorems（下载1999期刊扩展版，区别于1998会议版） |
| 新增Dziugaite–Roy回引Langford–Seeger 2001；Seeger 2002第8页明确承接此版本 | Seeger零一损失PAC-Bayes与GP分类界 |
| 已有McCandlish参考文献[MHB17][SL17]与附录C | Mandt–Hoffman–Blei、Smith–Le |
| Smith–Le参考文献；Jastrzębski §2.2 | Jastrzębski等、Zhu等 |

11份PDF均核对首页题名、作者和版本、PDF签名、pdfinfo页数、大小与SHA-256。来源为arXiv、JMLR、Springer出版方和作者机构JKU；Flat Minima为38页的March 1996作者稿（含期刊省略的伪代码），虽然首页列1997期刊42页引文，不能写成下载了42页期刊排版。已有4份方法原文的hash和字节数也与索引一致。

| 新增原文 | 实际本地版本 | 页数 | 主张暂定分类 |
|---|---|---:|---|
| Some PAC-Bayesian Theorems | Machine Learning 37:355–363, 1999; Kluwer出版PDF, 9页；扩展自COLT 1998, 非5页会议排版 | 9 | Science |
| PAC-Bayesian Generalisation Error Bounds for Gaussian Process Classification | JMLR 3:233–269, October 2002; publisher PDF, 37 pages | 37 | Science |
| Computing Nonvacuous Generalization Bounds for Deep (Stochastic) Neural Networks with Many More Parameters than Training Data | 1703.11008v2 [cs.LG], 19 Oct 2017 | 14 | Science |
| Stability and Generalization | JMLR 2:499–526, March 2002; publisher PDF, 28 pages | 28 | Science |
| Train faster, generalize better: Stability of stochastic gradient descent | 1509.01240v2 [cs.LG], 7 Feb 2016; PDF排版日期9 Feb 2016, ICML 2016前稿 | 32 | Science |
| Flat Minima | 作者March 1996稿, 38页, 含期刊省略的伪代码；首页列Neural Computation 9(1):1–42 (1997), 非42页期刊排版 | 38 | Knowledge |
| Exploring Generalization in Deep Learning | 1706.08947v2 [cs.LG], 6 Jul 2017 | 19 | Science |
| Stochastic Gradient Descent as Approximate Bayesian Inference | 1704.04289v2 [stat.ML], 19 Jan 2018; JMLR 18 (2017) author typeset, 35 pages | 35 | Science |
| A Bayesian Perspective on Generalization and Stochastic Gradient Descent | 1710.06451v3 [cs.LG], 14 Feb 2018; ICLR 2018 | 13 | Law |
| Three Factors Influencing Minima in SGD | 1711.04623v3 [cs.LG], 13 Sep 2018; PDF日期14 Sep 2018, ICANN 2018版本 | 14 | Law |
| The Anisotropic Noise in Stochastic Gradient Descent: Its Behavior of Escaping from Sharp Minima and Regularization Effects | 1803.00195v5 [stat.ML], 10 Jun 2019; ICML 2019 camera-ready | 14 | Science |

本批记录的关键边界：

- **PAC-Bayes有两层随机性。** 置信事件主要对iid训练样本S，风险内部还平均随机预测器Q。McAllester本篇使用按固定先验条件于集合U的混合；后续一般KL形式不能反写成该篇原定理。Seeger区分Gibbs、均值输出分类与投票分类；Dziugaite–Roy另控制Monte Carlo估计误差δ′，并明确没有给确定均值网络同一证书。先验须独立于被界定的训练样本；调先验尺度需要union bound等有效处理。
- **稳定性是泛化差距的条件。** Bousquet删除式β与替换式2β不同，高概率是对样本抽取；Hardt定理2.2界定的是|E gap|，不是E|gap|或逐次运行保证。其凸、强凸、非凸分支的平滑性、Lipschitz性、值域和步长条件不同。较好的gap界不等于训练误差低，也不证明实际训练越短越好。
- **Sharpness需要固定比较方式。** Flat Minimum Search用轴对齐扰动盒；Neyshabur强调输出尺度、margin和PAC-Bayes的KL项。单独Hessian或盒体积不是参数化无关的泛化排序。Smith–Le的精确evidence须把先验测度随参数换元一起变换；原文脚注也承认Laplace近似可能改变。
- **不同噪声理论预测不同量。** Mandt给局部OU稳态及KL(q||posterior)最优调节，未保证离散SGD精确采后验或测试最优。Jastrzębski的η/S关系需要连续近似和合适时间尺度，局部E[L]=(η/4S)tr(H)还假设C=H。Smith–Le的g=ε(N/B−1)和McCandlish的B_noise并非同一量；标量强度未包含完整方向几何。
- **局部离开指标不是首次逃逸。** Zhu的escaping efficiency定义为小时间期望损失增长。OU式(13)可精确计算该量，但Markov不等式只给事件概率上界，不能由较大期望推出更高逃逸概率或更短首次离开盆地时间。SGD噪声与Hessian对齐的具体网络证明另有窄模型条件，泛化实验单列为Knowledge。

已有方法的概念映射补充：

| 新概念 | 原有来源与本轮位置 | 不能混淆的对象 |
|---|---|---|
| Adam的一阶矩与二阶原始矩更新 | 1412.6980v9，第2–3页Algorithm 1与§2–3 | v_t为二阶原始矩，不是中心化方差；初始化校正不消除全部非平稳偏差；方法不等于收敛定理 |
| Batch Normalization的批统计与推理统计 | 1502.03167v3，第3–4页Algorithm 1–2 | 训练依赖同batch样本，推理使用固定统计；定义不证明原论文提出的机制是唯一原因 |
| word2vec的CBOW、Skip-gram与负采样目标 | 1301.3781v3第4–5页；1310.4546v1第2–4页 | CBOW与Skip-gram预测方向不同；NEG是替代二分类目标，不能直接等同softmax似然或NCE一致性 |

**Science只限明确条件内的具体主张；本轮均未独立审核完整证明，也未复现实验。** 其中OU与连续扩散是理想化模型，实际离散神经训练是否适用另行判断；Law不表示已穷尽可外推范围。所有阅读程度和定位已写入逐条记录。未改panel、冻结论文或共享生成总表。


## 2026-10-28 补充：已有经典论文的概念入口（21项）

本轮**新增论文0篇，新增概念15条，补充既有概念4条**，从311篇/233概念变为 **311篇论文、248条概念**。新概念中14条为Knowledge、1条为条件Science；21篇关联论文主标签保持原样，额外范围修订见后文。按paper.url是否出现在任一concept.sources的同口径检查，预训练库无概念关联的条目从74条降至 **53条**。这里解决的是来源到概念的缺口，不把来源数当成独立机制数，也不宣称完成全文审计。

先检查同义概念，再合并GPT-1、RoBERTa、ULMFiT与视觉自蒸馏的已有入口；T5/mT5与两篇早期seq2seq分别共享新入口。新增的ALBERT两个概念分别针对参数化和训练目标，不把整个模型作为单一结论。

| 原有论文 | 实际本地版本；本轮核对位置 | 概念关联 |
|---|---|---|
| GPT-1 | OpenAI 2018，12页；PDF第3–4页§3.1–3.3、式(1)–(5) | 补入既有「自回归下一 token 预测」 |
| ULMFiT | 1801.06146v5，2018-05-23；第2–5页§3.1–3.3 | 新「ULMFiT的分层学习率、斜三角日程与逐层解冻」；补入既有「领域与任务自适应继续预训练」 |
| ELMo | 1802.05365v2，2018-03-22；第2–4页§3.1–3.4 | 新「ELMo双向语言模型的上下文层混合」 |
| RoBERTa | 1907.11692v1，2019-07-26；第3–5页§4.1–4.3、表1–2 | 补入既有「掩码语言模型」 |
| T5 | 1910.10683v4，2023-09-19；第9–14页§2.4末、§3.1、图2 | 新「文本到文本与sentinel片段去噪」 |
| BART | 1910.13461v1，2019-10-29；第2–3页§2–3、图1–2 | 新「BART的完整序列去噪重建」 |
| ALBERT | 1909.11942v6，2020-02-09；第3–5页§3.1–3.2 | 新「ALBERT的嵌入分解与跨层参数共享」「句序预测SOP与跨文档NSP的区别」 |
| mT5 | 2010.11934v3，2021-03-11；第2–4页§2–4 | 与T5共用「文本到文本与sentinel片段去噪」 |
| Sutskever seq2seq | 1409.3215v3，2014-12-14；第2–3页§2、式(1) | 新「固定向量的循环编码器—解码器」 |
| Cho RNN Encoder–Decoder | 1406.1078v3，2014-09-03；第2–4页§2.2–2.3、§3.1 | 与Sutskever共用固定向量seq2seq概念，注明短语重评分、门控单元及上下文注入差异 |
| Bahdanau attention | 1409.0473v7，2016-05-19；第3–4页§3.1–3.2、式(4)–(6) | 新「解码状态条件下的软对齐注意力」 |
| Transformer-XL | 1901.02860v3，2019-06-02；第3–5页§3.2–3.3 | 新「Transformer-XL分段缓存与相对位置」 |
| ResNet | 1512.03385v1，2015-12-10；第3–4页§3.1–3.4 | 新「残差参数化与快捷连接」 |
| Identity Mappings | 1603.05027v3，2016-07-25；第3–4页§2–3、式(1)–(8)、脚注1 | 新「恒等残差链的前向与梯度加性分解」，仅此新概念为Science |
| ViT | 2010.11929v2，2021-06-03；第3–4页§3.1–4.1 | 新「ViT的图像patch序列表示」 |
| SimCLR | 2002.05709v3，2020-07-01；第2–4页§2–3.1、Algorithm 1 | 新「SimCLR的双视图、投影头与批内对比目标」 |
| MoCo | 1911.05722v3，2020-03-23；第3–4页§3.2–3.3、Algorithm 1 | 新「MoCo的队列字典与动量键编码器」 |
| CPC | 1807.03748v2，2019-01-22；第2–3页§2.1–2.3、式(1)–(4) | 新「CPC的未来潜在预测与InfoNCE目标」 |
| DINO | 2104.14294v2，2021-05-24；第2–4页§3.1、图2、Algorithm 1 | 补入既有「自蒸馏目标网络」，单列EMA教师与分布匹配子型 |
| SimSiam | 2011.10566v1，2020-11-20；第2–3页§3–4.1、图2 | 补入同一机制家族，明确共享权重、stop-gradient、预测头且无EMA |
| DINOv2 | 2304.07193v2，2024-02-02；第3–6页§3–4 | 补入同一机制家族，明确DINO+iBOT、分离投影头、SK、KoLeo及额外数据构建 |

这些PDF均已在库，本轮无下载。21份用于概念关联的原文与下述额外范围修订原文，共 **22份**重新核对PDF签名、pdfinfo页数、字节数与SHA-256；均与索引匹配。首页题名/版本及表中实际局部原文已核对，版本和阅读定位写入记录。

本轮保留的主张边界：

- **训练目标与迁移协议分开。** GPT-1的自回归预训练之后包含监督微调，不能从GPT-2/3的提示使用方式倒推所有自回归方法都不更新参数。ELMo分别训练两个方向的LM，下游使用任务层混合；这不是每个token同时条件于左右的MLM。
- **去噪目标各有输出范围。** T5以sentinel串联删除片段，BART重建完整原文，既有MASS的单连续片段又不同。T5的基线随机token删除与后续主动span选择消融分开。mT5扩到多语言改变语料、词表和采样，不能只归因于同一目标。
- **方法组件不直接成为通用因果解释。** RoBERTa的NSP实验联动输入格式；ALBERT减少独立参数不保证计算同倍下降；Transformer-XL停止缓存梯度且评测加速依赖对照协议；attention权重不自动是因果解释。
- **残差恒等式是有条件的Science。** h=f=identity且维度一致时，前向残差和及Jacobian的I加残差导数分解成立；投影或加法后激活破坏该直接前提。I项仍可被抵消，不能把代数恒等式写成“永不梯度消失”或全局收敛定理。未独立审核论文全部证明。
- **视觉自监督家族要保留子型。** SimCLR用批内负例，MoCo用历史队列和动量键编码器；BYOL/DINO用EMA教师，而SimSiam无EMA。DINO的分布匹配、BYOL的表示回归与DINOv2的组合目标分别注明。线性探测、全量微调、图像分割可视化和数据筛选不合并成同一性能保证。
- **InfoNCE方法与理论分开。** CPC本轮只录局部方法定义和正负采样条件，没有将互信息下界证明标成已审核Science，也不将分类目标直接等同完整生成似然。

### 额外修订：Reconciling Kaplan and Chinchilla（不计入21项或新增论文）

原有2406.12907记录把scope只写成模拟，漏掉本地 **v3（2024-11-21，17页）§4的实际训练**。根线程已核对第1–9页，本分支又核对第4–9页，更新claim、scope、evidence、source_version并拆成三条claim_annotations；整篇分类修为Knowledge＋Science（给定经验损失模型内的条件推导），三条均明确局部原文核对等级：

1. **Science条件推导：** §2.3–3假设6ND、固定损失形式及总参数与非embedding参数的关系，推出一般非单幂律的最优曲线与局部斜率。未独立审核全部证明。
2. **Knowledge条件模型模拟：** 用已假定的Chinchilla经验损失形式和拟合参数合成20个约790–1.58B非embedding参数模型，换统计口径和范围后得到约0.74–0.78的局部指数。这是模型内的模拟检查，不是独立经验Law；本轮未重新运行模拟。
3. **Knowledge微型实测：** BookCorpus上的五个0.8M–4.6M总参数模型，GPT-2词表50,257、context 16，按总参数/总算力得a≈0.49，按非embedding参数/算力得a≈0.74。四种学习率/预算方案的总参数指数分别0.58、0.49、0.46、0.49。

这些小规模训练不能称完整复现Kaplan/Chinchilla，也不能排除大规模优化器或日程效应。本轮未改panel、冻结稿、构建器或共享生成总表。

## 2026-10-28 第七轮：优化/理论与scaling的反向概念关联

从 **311篇、248个概念、53篇尚未关联概念** 出发，本轮为 **25篇既有论文** 建立来源映射；**新增20个概念、修订2个既有概念**。另经授权补SOAP原论文 **1篇/1份PDF**（其概念已包含在20个内），当前为 **312篇、268个概念，未关联队列剩28篇**。没有把既有PDF复核计成新增论文。

修订的既有概念为「基于 token、参数和计算的 scaling law」及「Batch Normalization的批统计与推理统计」。前者保留Kaplan Law锚点，以逐来源scope区分Hestness数据学习曲线、Rosenfeld联合曲面和Henighan多模态交叉熵；后者补Santurkar等的机制区分实验，没有把有限方向测量写成全局光滑性。

| 既有来源 | 本轮实际阅读定位 | 关联概念 |
|---|---|---|
| Shampoo: Preconditioned Stochastic Tensor Optimization (1802.09568) | v2 PDF第2–3页§1.1、Algorithm 1、矩阵存储/求根复杂度。 | Shampoo的张量轴预条件 |
| Adafactor: Adaptive Learning Rates with Sublinear Memory Cost (1804.04235) | v1 PDF第2–3页§3、Lemma 1/Algorithm 2；第5–6页§6–8。 | Adafactor的因子化二阶矩与相对步长 |
| Symbolic Discovery of Optimization Algorithms (2302.06675) | v4 PDF第2页Program 1、第3–4页搜索方法与适用限制。 | Lion的符号方向与双动量系数 |
| Muon is Scalable for LLM Training (2502.16982) | v1 PDF第2–4页§2.1–2.2、式(1)–(4)、Lemma 1与参数分组说明。 | Muon的动量正交化与参数分组 |
| Fantastic Pretraining Optimizers and Where to Find Them (2509.02046) | v3 PDF第3–4页§1–2与Figure 1；PDF首页版本2026-10-25。 | 优化器比较的终点预算与token等价效率 |
| On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima (1609.04836) | v2 PDF第3–6页§2.2、Metric 2.1、式(3)–(4)、实验配置与表。 | Keskar的参数盒锐度与大批量观察 |
| SGDR: Stochastic Gradient Descent with Warm Restarts (1608.03983) | v5 PDF第2–4页，§3式(5)位于PDF第4页。 | SGDR的余弦衰减与学习率重启 |
| Deep Information Propagation (1611.01232) | v2 PDF第2–5页§2–3.1、式(1)–(13)、dropout相关递推。 | 随机网络均值场的信号传播深度 |
| Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks (1602.07868) | v3 PDF第2–3页§2–2.1、式(2)–(4)。 | Weight Normalization的尺度方向重参数化 |
| How Does Batch Normalization Help Optimization? (1805.11604) | BatchNorm v3（2015-03-02）PDF第3–4页§3–3.1/Algorithm 1–2（既有核对）；新增Santurkar等v5 PDF第3–5页§2.1–3.1、Definition 2.1、Figure 2–4。 | Batch Normalization的批统计与推理统计 |
| On the Convergence of Adam and Beyond (1904.09237) | v1（ICLR 2018论文的2019 arXiv版）PDF第3–5页§3、Theorem 1–3与Algorithm 2；未审计全部正向收敛证明。 | Adam反例的量词与AMSGrad长记忆 |
| Large Batch Optimization for Deep Learning: Training BERT in 76 minutes (1904.00962) | v5 PDF第3–6页§2–4、Algorithm 2、Theorem 3。 | LAMB的逐层信赖比缩放 |
| Analyzing & Reducing the Need for Learning Rate Warmup in GPT Training (2410.23922) | Goyal v2 PDF第2–3页§2.1–2.3；GPT warmup v1 PDF第2–3页§2–4、Figure 1。 | 学习率warmup的批量近似与初期更新 |
| Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour (1706.02677) | Goyal v2 PDF第2–3页§2.1–2.3；GPT warmup v1 PDF第2–3页§2–4、Figure 1。 | 学习率warmup的批量近似与初期更新 |
| Deep Learning Scaling is Predictable, Empirically (1712.00409) | Kaplan等v1 §1.2式(1.1)–(1.3)、图1、§3/§6（保留既有核对）；本轮补Hestness v1 PDF第2–4页§1–3，Rosenfeld v2第3–6页§3–6/式(1)–(5)/Table 1，Henighan v2第3–4页§1/式(1.1)–(1.2)/图1–2。 | 基于 token、参数和计算的 scaling law |
| A Constructive Prediction of the Generalization Error Across Scales (1909.12673) | Kaplan等v1 §1.2式(1.1)–(1.3)、图1、§3/§6（保留既有核对）；本轮补Hestness v1 PDF第2–4页§1–3，Rosenfeld v2第3–6页§3–6/式(1)–(5)/Table 1，Henighan v2第3–4页§1/式(1.1)–(1.2)/图1–2。 | 基于 token、参数和计算的 scaling law |
| Scaling Laws for Autoregressive Generative Modeling (2010.14701) | Kaplan等v1 §1.2式(1.1)–(1.3)、图1、§3/§6（保留既有核对）；本轮补Hestness v1 PDF第2–4页§1–3，Rosenfeld v2第3–6页§3–6/式(1)–(5)/Table 1，Henighan v2第3–4页§1/式(1.1)–(1.2)/图1–2。 | 基于 token、参数和计算的 scaling law |
| Reconciling Kaplan and Chinchilla Scaling Laws (2406.12907) | Pearce–Song v3 PDF第4–9页§2.3–4；Porian v4 PDF第2–4页§1–2.3/Figure 1。 | 计算最优scaling的参数口径与训练协议 |
| Resolving Discrepancies in Compute-Optimal Scaling of Language Models (2406.19146) | Pearce–Song v3 PDF第4–9页§2.3–4；Porian v4 PDF第2–4页§1–2.3/Figure 1。 | 计算最优scaling的参数口径与训练协议 |
| Explaining Neural Scaling Laws (2102.06701) | v2 PDF第3–5页§1.3–2.2.3、Theorem 1–3。 | 方差受限与流形分辨率受限的scaling界 |
| A Solvable Model of Neural Scaling Laws (2210.16859) | v1 PDF第17–22页§2、式(35)；第28–29页§3.2的大维数极限；第50–51页式(163)–(165)；第68–69页§4.4（正文页码比PDF小1）。 | 潜在Gaussian随机特征模型的解析scaling |
| Learning representations by back-propagating errors (rumelhart1986backprop.pdf) | Rumelhart等Nature原文PDF第2–3页（534–535）、式(3)–(9)/Figure 5；LeNet作者预印本PDF第5页式(4)及模块链式法则；扫描页已目检。 | 反向传播的链式梯度与共享权重累积 |
| Gradient-Based Learning Applied to Document Recognition (lecun1998gradient.pdf) | Rumelhart等Nature原文PDF第2–3页（534–535）、式(3)–(9)/Figure 5；LeNet作者预印本PDF第5页式(4)及模块链式法则；扫描页已目检。；46页作者预印本PDF第5–6页§II–II.A及LeNet-5卷积/下采样说明；已目检扫描页。 | 反向传播的链式梯度与共享权重累积；卷积局部连接、权重共享与下采样 |
| On the importance of initialization and momentum in deep learning (sutskever2013momentum.pdf) | ICML 2013/PMLR 28 PDF第2–3页§2–2.1、式(1)–(4)。 | 经典动量与Nesterov前瞻梯度 |
| A Stochastic Approximation Method (robbins1951stochastic.pdf) | 1951原文PDF第2–7页（400–405），式(4)–(8)、(26)、(30)、(33)–(35)、Theorem 2；PDF第1页为JSTOR封面。 | Robbins–Monro的一维随机寻根充分条件 |

### 新增SOAP原文与核验

- 原文：[SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321)，固定下载 https://arxiv.org/pdf/2409.11321v2；v2为2025-01-31，首次提交2024-09-17。官方元数据与PDF首页题名及8位作者匹配。这不是题名添加“for Language Modeling”、作者表也有变化的ICLR版本。
- 本地 papers/pretrain/2409.11321.pdf：20页、1,300,562字节，SHA-256 d6f33bb71eab0f046d911b60800846fd3332f1dcf5f2f1434ab444ab89d0b22a；三库arXiv编号、规范化题名和PDF hash去重。核对PDF第4–8页§4–6.2、Algorithm 1–4/Claim 1和Figure 2。
- Claim 1限定数据期望、精确特征基、1/2幂与trace修正。其证明未包含算法框显示的ε；本库只记录无阻尼且逆存在时的代数等价。实际SOAP的EMA、周期/近似基、Adam逐坐标二阶矩与有限语言模型实验另列Knowledge，不声称实际SOAP与原始1/4幂Shampoo恒等。
- 效率比较基于360M/660M、所列batch和短cosine日程的终点拟合，时间由单H100加梯度累积的吞吐换算。论文将完整分布式比较留待后续，不能把该结果当任意硬件加速。

### 本轮保留的范围与纠错

- **方法定义与定理分开。** Shampoo轴向统计不是精确Hessian；Adafactor省状态需不额外保存完整一阶动量；Lion的sign分量不含权重衰减部分；LAMB正向理论的β1=0、无衰减、batch和有界缩放等条件不能省略。
- **已修正两处现有字段。** Muon的embedding和输出头虽是矩阵仍用AdamW，不能写成所有矩阵都用Muon。SGDR余弦式(5)的定位由§2改为§3、PDF第4页，重置学习率不重置权重，也不同于warmup。
- **sharpness与机制证据有限。** Keskar的参数盒/投影与近似最大化依赖坐标；BN噪声与方向测量不是全部网络的必要机制。Adam是存在反例，不是所有问题失败；AMSGrad定义本身不是无条件收敛保证。
- **scaling各有对象。** Hestness的样本曲线、Rosenfeld的联合测试误差曲面和Henighan的交叉熵不能共用单位。拟合L∞解释为熵需相应无限数据/模型极限与表达能力。Pearce–Song仍为条件代数Science、合成模型Knowledge及tiny实训Knowledge；Porian的多训练规律保留自身Law范围，不将两文归结为排他的单一原因。
- **理论边界显式写出。** Bahri的插值流形上界不是精确幂指数，随机插值点P不自动等于参数数；Solvable的Gaussian/随机特征/ridge模型及固定比例大维极限不包含真实特征学习或有限计算训练。随机网络均值场的传播尺度不等于所有有限网络训练深度上限。Robbins–Monro记录均方与依概率结论，不扩大为无条件几乎处处收敛。
- **基础结构保持原义。** Backprop是链式梯度计算，不是全局优化保证；卷积给有边界条件的平移等变，不是分类器严格平移不变；LeNet原下采样不是max pooling。经典/Nesterov动量的梯度求值位置分开，凸加速率不套用随机非凸训练。

本轮25份既有PDF加SOAP共 **26份**重新核对签名、pdfinfo页数、字节数和SHA-256，均匹配；首页题名/版本及上述局部原文已核对。扫描的backprop和LeNet页已目检；LeNet明确使用46页作者预印本页码。Science标签仅针对指定条件主张，不表示逐行独立证明审核或实验复现。未改panel、冻结稿、构建器或共享生成总表。

## 2026-10-28 第八轮：剩余28篇来源的概念归属

本轮从 **312篇、268概念、28篇未关联** 出发，**0新增论文、0新增PDF、8个新概念、7个既有概念修订**。28项全部关联，当前 **312篇、276概念、未关联来源0项**。这是来源映射完成，不是全部论文主张都已审核、全部工作覆盖或每篇全文精读。

同类技术报告接入共享自回归目标和开放训练资源，MoE、过滤、去重和冻结模态连接保留子类型。数据/评测新增概念同样跨来源组织，没有为每个模型、数据集或benchmark品牌机械建概念；每篇的决策与理由写入concept_link_review_2026_10_28h。

新增8个概念：Transformer FFN的GLU门控变体；GloVe的加权全局共现回归；多来源预训练语料的混合与构建记录；多任务评测的场景、适配与指标协议；按领域评估语言模型fit与perplexity可比性；LLaVA的视觉投影与指令监督阶段；SSM的半可分矩阵表示与SSD对偶；DeepNorm的残差缩放与初始化。

修订7个既有概念：自回归下一 token 预测；专家混合 MoE；选择性状态空间模型；预训练语料去重；数据质量过滤；跨模态冻结骨干与查询连接；开放预训练的模型、数据与训练轨迹。其中原「OPT 开放预训练与训练日志」扩为「开放预训练的模型、数据与训练轨迹」，保留OPT来源并补NeoX/Pythia/OLMo。

| 来源 | 下载版本 | 定向阅读位置 | 概念归属 |
|---|---|---|---|
| Scaling Language Models: Methods, Analysis & Insights from Training Gopher | 2112.11446v2 [cs.CL], 21 Jan 2022; PDF heading 2021-12-08 | v2 PDF第6–7页§3.1–3.4/Table 1–2。 | 自回归下一 token 预测 |
| PaLM: Scaling Language Modeling with Pathways | 2204.02311v5 [cs.CL], 5 Oct 2022 | v5 PDF第5–7页§2–3/Table 1–2。 | 自回归下一 token 预测 |
| GPT-NeoX-20B: An Open-Source Autoregressive Language Model | 2204.06745v1 [cs.CL], 14 Apr 2022 | v1 PDF第2–3页§2–3。；v1 PDF第2–3页引言及§2–3。 | 自回归下一 token 预测；开放预训练的模型、数据与训练轨迹 |
| LLaMA: Open and Efficient Foundation Language Models | 2302.13971v1 [cs.CL], 27 Feb 2023 | v1 PDF第2–3页§2.1–2.3/Table 1–2。 | 自回归下一 token 预测 |
| Llama 2: Open Foundation and Fine-Tuned Chat Models | 2307.09288v2 [cs.CL], 19 Jul 2023 | v2 PDF第5–7页§2.1–2.3/Table 1。 | 自回归下一 token 预测 |
| The Llama 3 Herd of Models | 2407.21783v3 [cs.AI], 23 Nov 2024; title page date July 23, 2024 | v3 PDF第4–5页Figure 1、§3–3.1.1。 | 自回归下一 token 预测 |
| Qwen2.5 Technical Report | 2412.15115v2 [cs.CL], 3 Jan 2025; PDF heading 2025-01-06 | v2 PDF第2–3页§1–3.1/Table 1。 | 自回归下一 token 预测 |
| DeepSeek-V3 Technical Report | 2412.19437v2 [cs.CL], 18 Feb 2025 | v2 PDF第8–9页§2.1.2/式(12)–(20)。 | 专家混合 MoE |
| GLaM: Efficient Scaling of Language Models with Mixture-of-Experts | 2112.06905v2 [cs.CL], 1 Aug 2022 | v2 PDF第4–5页§4–5/Table 4。 | 专家混合 MoE |
| Mixtral of Experts | 2401.04088v1 [cs.LG], 8 Jan 2024 | v1 PDF第2–3页§2.1/公式、Table 1。 | 专家混合 MoE |
| Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality | 2405.21060v1 [cs.LG], 31 May 2024 | v1 PDF第23–27页§7–8/Figure 6；只读子审查。；v1 PDF第4页Definition 2.2、第7–9页式(3)–(4)/Lemma 3.3/Proposition 3.4/Theorem 3.5；第16–17页§5.1–5.3/Corollary 5.1；只读子审查定向核对。 | 选择性状态空间模型；SSM的半可分矩阵表示与SSD对偶 |
| GLU Variants Improve Transformer | 2002.05202v1 [cs.LG], 12 Feb 2020 | v1 PDF第2–3页§2式(5)–(6)、§3.1–3.3、Table 1。 | Transformer FFN的GLU门控变体 |
| DeepNet: Scaling Transformers to 1,000 Layers | 2203.00555v1 [cs.CL], 1 Mar 2022 | v1 PDF第4–8页§4.1–5、Theorem 4.2–4.3、Table 1/Figure 6–7；第15–18页附录A–C与实验设置；只读子审查定向核对。 | DeepNorm的残差缩放与初始化 |
| The Pile: An 800GB Dataset of Diverse Text for Language Modeling | 2101.00027v1 [cs.CL], 31 Dec 2020 | PDF第2–3页§1.1–2/Table 1。 | 多来源预训练语料的混合与构建记录 |
| The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only | 2306.01116v1 [cs.CL], 1 Jun 2023 | v1 PDF第8–9页§4.3/Table 5与limitations。；v1 PDF第7–9页§4–5，重点Table 5。 | 预训练语料去重；数据质量过滤 |
| Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research | 2402.00159v2 [cs.CL], 6 Jun 2024 | v2 PDF第4页§4.1–4.2。；v2 PDF第4–5页§4.2–5.3/Figure 1–2。；PDF第3–5页§3–5.2，尤其第4页§4.1–4.2。 | 预训练语料去重；数据质量过滤；多来源预训练语料的混合与构建记录 |
| RedPajama: an Open Dataset for Training Large Language Models | 2411.12372v1 [cs.CL], 19 Nov 2024 | v1 PDF第6–7页§4–4.1.2；第3–4页区分V1。；PDF第3–4页§3–3.1/Table 2；第6–7页§4区分V2。 | 数据质量过滤；多来源预训练语料的混合与构建记录 |
| Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models | 2206.04615v3 [cs.CL], 12 Jun 2023 | PDF第6–9页§2–2.3.1、Table 1；未复核全部任务与结果。 | 多任务评测的场景、适配与指标协议 |
| Holistic Evaluation of Language Models | 2211.09110v2 [cs.CL], 1 Oct 2023; TMLR 08/2023 | PDF第4–6页Figure 3、Standardization、Adaptation及Caveats；未重评30模型。 | 多任务评测的场景、适配与指标协议 |
| Paloma: A Benchmark for Evaluating Language Model Fit | 2312.10523v2 [cs.CL], 7 Dec 2024 | v2 PDF第3–6页§2/Table 1、§3五条指南及控制、Table 2和perplexity定义。 | 按领域评估语言模型fit与perplexity可比性 |
| Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus | 2104.08758v2 [cs.CL], 30 Sep 2021 | v2 PDF第6–7页§5–5.3及讨论；不将源文讨论段的笔误方向当结论。 | 数据质量过滤 |
| Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling | 2304.01373v2 [cs.CL], 31 May 2023 | v2 PDF第2–3页§2–2.3/Table 1。 | 开放预训练的模型、数据与训练轨迹 |
| OLMo: Accelerating the Science of Language Models | 2402.00838v4 [cs.CL], 7 Jun 2024 | v4 PDF第2–3页§2–2.3/Table 1–2。 | 开放预训练的模型、数据与训练轨迹 |
| BLOOM: A 176B-Parameter Open-Access Multilingual Language Model | 2211.05100v4 [cs.CL], 27 Jun 2023 | v4 PDF第15–16页§3.2.2–3.2.3/Figure 5。 | 自回归下一 token 预测 |
| GloVe: Global Vectors for Word Representation | EMNLP 2014 proceedings, pp.1532–1543; ACL Anthology D14-1162, 12-page PDF | EMNLP 2014原文PDF第3–4页（1534–1535）、§3式(1)–(9)/Table 1。 | GloVe的加权全局共现回归 |
| GPT-4 Technical Report | 2303.08774v6 [cs.CL], 4 Mar 2024 | v6 PDF第2–3页§2–3.2/Figure 1–2。 | 自回归下一 token 预测 |
| Flamingo: a Visual Language Model for Few-Shot Learning | 2204.14198v2 [cs.CV], 15 Nov 2022 | v2 PDF第4–5页§2.1–2.2/Figure 3–4。 | 跨模态冻结骨干与查询连接 |
| Visual Instruction Tuning | 2304.08485v2 [cs.CV], 11 Dec 2023 | v2 PDF第3–5页§3/Table 1、§4.1–4.2、式(1)–(3)/Table 2。 | LLaVA的视觉投影与指令监督阶段 |

本轮重点边界：

- **共享目标不等于共享配方。** Gopher的精度与规模变化、PaLM并行层质量的尺度外推、NeoX两套LN的实际实现、各Llama尺寸的预算/GQA差异、Qwen dense/API-MoE区别分别保留。GPT-4未披露参数规模、完整架构、硬件、算力与数据构建，不补造。BLOOM的小规模消融不直接视为176B受控验证。修正LLaMA旧scope：1T/1.4T预算不同，但Table 1明确使用相同来源采样比例。
- **开放资源不保证复现成功。** OPT、NeoX、Pythia、OLMo的公开范围与设计控制分开；本轮核对论文原文，不宣称全部资源现时链接已逐一验证或已复现训练。
- **MoE存在不同路由。** GLaM隔层top-2，Mixtral每层8选2，DeepSeek-V3另有共享专家、sigmoid affinity和路由偏置；偏置不进入专家输出权重且仍有序列级辅助平衡项。激活参数、总参数和墙钟不能互换。
- **数据集与消融分开。** Pile字节权重/重复轮数不是token配比；Dolma局部干预限1.2B/最多150B token，且有性能与数据量取舍；RedPajama-V1有原配方描述缺口，V2是轻处理信号池。RefinedWeb缺本语料独立去重组且过滤收益依来源；C4方言统计基于模型预测，不能当作者身份或真实人口占比。
- **评测有协议范围。** BIG-bench任务自定义preferred metric与HELM同场景多指标不是同一评估；统一提示不保证对每个模型最优，也不控制各模型训练预算。Paloma按领域PPL不等于领域能力，词表/预处理会改变指标；其指定代码来源和短段落并未全部去污染。
- **跨模态训练阶段不同。** Flamingo冻结两端骨干训练resampler及门控连接；LLaVA第二阶段解冻LM，教师生成数据只看caption/boxes，不能当直接看图获得的无误标签。
- **SSD只升条件代数。** 给定A/B/C与零初始状态的递推—N-SSS对应列Science；A_t=a_tI下才有相应掩码核注意力且无softmax。输入依赖系数不使整体块固定线性；实际Mamba-2架构/通信/质量为Knowledge。未录入v1尚未厘清口径的全部复杂度数字。
- **DeepNet近似界尚未确立Science。** 附录用d=1量级代理、方差可加、一阶Taylor和η趋零的单步SGD。式(8)把(α+vw)/sqrt(α²+v²w²)近似为1，正v,w时该因子大于1，未给误差随深度控制；不得转录为任意深度的严格有限更新界。理论注释classification为Knowledge（条件近似分析），未确立Science的理由放scope，不让分类解析误计。DeepNorm只缩放指定FFN/V/O初始化，实际Adam训练另列。

28份既有PDF的签名、pdfinfo页数、字节数和SHA-256均重新核对并与索引匹配；无下载、无论文增量。SSD/DeepNet由有界只读子审查核对上述原文，其余由本分支直接读；每次抽取不超过8页。Science标签仅标限定主张，经典引理及全证明未独立逐行审核。本轮未复现实验，未改panel、冻结稿、共享构建器或生成总表。

## 2026-10-28：恢复早期无监督预训练的原始文献链

本轮新增 **11 篇论文、9 个概念**，并扩充既有「逐层无监督预训练与优化初始化」概念。合并后 pretrain 为 **323 篇论文 / 285 个概念**（本分支交接时的快照；根线程后续补入 CD 理论与 Boltzmann 前史会继续增加）。这 11 份 PDF 已在前次中断前保存；本轮重新读取原文、验证身份并入库，没有把下载记录自动当成已审读。

入口来自既有 Erhan 2009 AISTATS 原文第 8 页参考文献，包括 DBN、降维、greedy layer-wise、早期 DAE、Ranzato 与 Lee；再沿 DBN 的前置来源补 wake-sleep、Helmholtz 与 CD，并保留 Vincent 2010 和 Erhan 2010 的完整 JMLR 后继文章。Erhan 2010 JMLR 与同年 AISTATS 短版、既有 2009 AISTATS 文章不是同一份材料；Lee、Bengio、Ranzato 的会议年份与次年卷册引用年份分开登记。

| 原文 | 保存版本 | 实际核对位置 | 主张分类 |
|---|---|---|---|
| A Fast Learning Algorithm for Deep Belief Nets | Neural Computation 18:1527–1554 (2006); 28-page publication PDF on author site | PDF第8–12页，§3式(3.2)及参数依赖说明、§4式(4.1)–(4.3)、第12页脚注4；未完整审计附录A。 | Science（条件变分界）＋Knowledge |
| Reducing the Dimensionality of Data with Neural Networks | Science 313:504–507, 28 July 2006; 4-page author-hosted publication PDF, first page also contains preceding article tail | PDF第1页下部本篇标题；第2–4页、式(1)–(2)、Figures 1–4文字与正文；插图未逐图独立目检。 | Knowledge |
| Greedy Layer-Wise Training of Deep Networks | NIPS 2006, volume 19; commonly cited as 2007 proceedings publication | PDF第2–3页§2、以及第6–7页§4/Table 2/§5；完整附录和所有实验未复核。 | Knowledge |
| Extracting and Composing Robust Features with Denoising Autoencoders | ICML 2008 proceedings, pp.1096–1103; 8-page official PDF | PDF第2–3页§2.1–2.4、式(1)–(5)；另浏览第4–5页理论动机，但未独立审计全部推导或实验。 | Knowledge |
| Stacked Denoising Autoencoders: Learning Useful Representations in a Deep Network with a Local Denoising Criterion | JMLR 11:3371–3408 (2010), 38-page final article | PDF第7–13页§2.3–3.5，重点第9页方法、第11–12页损坏类型/加权目标/堆叠；其余实验未完整审核。 | Knowledge |
| The Wake-Sleep Algorithm for Unsupervised Neural Networks | Author manuscript dated 3 April 1995, 12 pages; published Science 268:1158–1161 (1995), not the final journal pagination | 作者稿PDF第2–5页式(1)–(8)、wake/sleep过程、模型幻想分布与factorial后验限制；完整实验未复现。 | Knowledge |
| The Helmholtz Machine | Author manuscript dated 14 December 1994, 19 pages; published Neural Computation 7:1022–1037 (1995) | 作者稿PDF第3–7页，式(5)及§3式(6)–(16)、§4；附录A/B与完整实验未审核。 | Science（条件变分分解）＋Knowledge |
| Training Products of Experts by Minimizing Contrastive Divergence | Neural Computation 14:1771–1800 (2002), 30-page publication PDF; distinct from 2000 technical report | PDF第4–6页§3式(3.1)–(3.3)、五步采样与脚注5；没有复核§10全部模拟。 | Knowledge |
| Efficient Learning of Sparse Representations with an Energy-Based Model | NIPS 2006, volume 19; commonly cited as 2007 proceedings publication | PDF第2–4页§2、式(1)–(7)、§3实际三步在线更新；完整实验与Poisson极限论断未审。 | Knowledge |
| Why Does Unsupervised Pre-training Help Deep Learning? | JMLR 11:625–660 (2010), 36-page article; distinct from existing 2009 AISTATS paper and 2010 AISTATS short version | PDF第8–11页§5设置；第18–21页§6.4–7.3、Table 1/Figures 7–9的正文解释；未全审全部图或独立复现。 | Knowledge |
| Sparse deep belief net model for visual area V2 | NIPS 2007, volume 20; commonly cited as 2008 proceedings publication | PDF第2–7页，§3式(1)–(4)/Algorithm 1及脚注3、§4、§5.1–5.2、§6；未独立复现生物统计或全图目检。 | Knowledge |

**分类边界。** DBN 的 Science 条目只覆盖条件变分界：初始化须使旧下界紧，冻结下层生成项和识别项，以真正的期望最大似然改进上层。原文第 12 页明确说改成 CD 会取消保证；脚注 4 又限定期望变化。下界提高不保证每个中间步骤的实际似然都提高。实际 CD / up-down、不同宽度和有限实验单列 Knowledge。

Helmholtz 的 Science 条目只给证据对数、变分下界与 KL 差距的概率分解。确定性实现还有均值替代、忽略跨层相关的近似；随机 wake-sleep 变体的两阶段没有共同单调代价保证。Wake-sleep 作者稿第 4 页还明确区分模型生成的幻想数据和真实训练数据，以及 factorial 识别分布无法一般表达 explaining-away。不能把变分分解的成立直接变成这些实现的训练保证。

Hinton 2002 中完整 CD 目标的导数与短链学习更新分开：第 6 页式 (3.3) 的重构分布参数导数被忽略，有限模拟支持的近似不提升为通用无偏或收敛定理。根线程另维护两篇 CD 理论文与可精确核算的边界案例；本分支没有重复添加它们。

DAE 的去噪训练、干净输入编码和下一层训练是三个不同步骤；2010 的 full emphasis 仅计受损维度，是可选目标。对流形的解释和重构表现没有被转写成对任意污染、语义解耦或普遍下游收益的保证。Bengio 对输入边际不能揭示标签条件分布的情况保留边界。Erhan 的 apparent local minimum 没有用完整 Hessian 验证，400 种子不能覆盖仅一个种子的 InfiniteMNIST；小网络负结果、优化与正则化解释可并存也保留。

Ranzato 的固定代码、线性滤波器子问题为凸二次，不证明包含非线性代码的联合问题全局可解；其稀疏性是单元跨样本的活动性质。Lee 的实际 mini-batch 稀疏项只调整偏置，不能当全部参数的无偏梯度；V2 比较还依赖刺激对比度校准、中心定位和单元筛选，统计相似不证明皮层实现同一网络。以上均维持 Knowledge。

**材料核验。** 11 份 PDF 分别对官方会议/JMLR 或作者网页的 HTTPS URL 做了新 GET，字节与现有文件逐一完全相同，均检查 PDF 签名、pdfinfo 页数、SHA-256 和首页身份。最大文件为 1,392,544 bytes；都在已忽略的 papers/ 中。Hinton–Salakhutdinov 2006 的第 1 页同时包含上一篇论文尾部，本篇标题在下部，不能依据首页开头的光学文字错认。Wake-sleep 是 1995-04-03 的 12 页作者稿；Helmholtz 是 1994-12-14 的 19 页作者稿，分别保留 1995 发表信息，不假称最终期刊分页。Hinton 2002 是 30 页正式期刊版，不是 2000 技术报告。

三库标准化题名与现有 SHA 去重通过；11 个新论文均有明确概念来源 URL。来源关联不是语义正确性的自动证明。每次 PDF 抽取不超过 8 页；本轮没有全部论文全文精读、完整附录证明审计或训练复现，也未逐图目检。尚未闭合 Erhan 2009 的全部引用链，例如 Weston 的 semi-supervised embedding、Ranzato 后继 sparse feature learning、相关早期 auto-associative memory 等，未将本轮称为该分支全覆盖。

根线程的 CD 案例另作独立有限模型核查：4 状态 Gibbs 核给出 P11=153/320、最大似然更新 0、CD 更新 7/320；一可见一隐藏 RBM 的混合偏导差为 9/128。脚本实际运行通过；2010 原文第 6 页式 (19) 与式 (20) 的符号不一致由正确映射 f=(1−λ)W+U 修复后能证明指定闭球内固定点存在，不推出唯一性或训练收敛。该核查没有修改根线程案例文件。

本轮未修改 panel、冻结论文、共享构建器或生成总表；至此归还 pretrain 索引及 coverage 写权限。

## 能量模型源头与 CD 理论补充（2026-10-28，根线程）

在上轮 11 篇早期预训练来源之后，另补 **4 篇论文、4 个概念**：Hopfield 1982、Ackley–Hinton–Sejnowski 1985、Carreira-Perpiñán–Hinton 2005 与 Sutskever–Tieleman 2010。当前 pretrain 索引为 **327 篇论文、289 个概念**；这不是该方向全部经典文献已覆盖的声明。

| 来源 | 实际阅读与核查 | 分类边界 |
|---|---|---|
| Hopfield 1982 | Caltech 课程公开的 6 页扫描含一页 JSTOR 封面；正文物理 pp.2–6 逐页 OCR，p.2 身份和 p.4 能量式另目检。机构仓库原下载重定向失败，保存的是另一公开课程版本。 | 对称、零自连接、异步二元更新的能量论证列 Science；原文未单列零场等号分支，这里补充保持状态的约定。正确回忆与有限 N 的容量观察另列 Knowledge。 |
| Ackley–Hinton–Sejnowski 1985 | Hinton 作者站 23 页期刊扫描。物理 pp.1、3–10、11–14、17–21 定向阅读；p.9 式(9)–(10)目检。 | 两相平衡下的 KL 梯度恒等式列 Science；有限退火、带噪夹持和固定符号步长的编码实验另列 Knowledge。完整推导与实验未独立审计。 |
| Carreira-Perpiñán–Hinton 2005 | PMLR 官方 8 页 PDF，读 pp.1–6、8。官方页面 2021 是重刊日期，原论文年为 2005。 | 指定有限模型的固定点差异列 Science；小模型中的小偏差、ML 初始化收益列 Knowledge，不扩为任意模型。 |
| Sutskever–Tieleman 2010 | PMLR 官方 7 页 PDF，读 pp.1–6；物理 p.6/印刷794式(19)–(20)目检。 | 非梯度场和正则化固定点存在性列条件 Science；非凸正则循环的数值构造另列 Knowledge。式(19)符号问题有明确修复，不据此否定定理。 |

四份 PDF 均核对签名、页数、字节数和 SHA；扫描身份与版本保留专门记录。精确核算另见 [CD 边界案例](contrastive_divergence_scope_case.md)：在指定 VBM2 的 ML 固定点处 CD₁ 更新为 7/320；指定 RBM 的混合偏导差为 9/128。预训练审查 agent 独立枚举并运行脚本，核算一致，同时核对修正后的 Brouwer 自映射只给存在性。这个独立检查不替代两篇完整论文的证明与实验审计。

来源链已补到能量下降、平衡学习、有限步近似及其偏差；Hopfield 引用的更早联想记忆/感知机/局部学习源头、连续 Hopfield 模型及后继容量理论还需各自回到原文，不能用本轮四篇替代。未重生成或修改 panel、冻结论文。

独立选读复核另确认 Hopfield 的物理 pp.3–4 和 Ackley 的 p.9、pp.10–14 与上述分类一致。Hopfield 零场保持是本目录明确补充的约定，已区分于原文只列严格大于／小于的更新式；没有把它冒称为原文单列条件。Ackley 有限退火后假定已平衡与精确梯度所需平衡之间的差异保留。

## 2026-10-28：阈值单元、局部学习与关联记忆的经典补录

沿 Hopfield/Boltzmann 前史中的联想记忆、阈值单元与感知机来源，另补 **8 篇原始论文、7 个概念**；本分支合并后的快照为 **335 篇论文 / 296 个概念**。Kohonen 1972 与 Anderson 1972 共用「外积关联记忆的信号与串扰分解」概念，避免把同一代数按作者拆成两个概念。连续分级响应与既有异步二元网络的条件不同，另设动力学概念。三库 pretrain/rl/interp 的标准化题名与 PDF SHA 去重通过，没有重合并上轮缓存或根线程四篇记录。

| 原文 | 保存版本与实际选读位置 | 本轮授予的主张层次 |
|---|---|---|
| McCulloch–Pitts, A Logical Calculus of the Ideas Immanent in Nervous Activity (1943) | 19 页原刊扫描，非 1990 重刊；PDF pp.1、3–7，重点印刷117–121的模型公理与 Theorems I–III。 | Science：无环形式网络与受限制时态命题表达式的构造性对应。 |
| Rosenblatt, The Perceptron (1958) | Psychological Review 65(6):386–408，23 页；PDF pp.1、3–8、10–15、17–21，重点组织、Table1、环境与测试定义、pp.18–19的外部模拟报告。 | Knowledge：指定随机网络、强化规则与环境模型的表征/泛化主张；不据此评定整篇所有理论的最高层次。 |
| Novikoff, On Convergence Proofs for Perceptrons | 8 页卷册扫描，pp.615–622；1962 年会议报告、1963 年卷册出版分开。OCR 选读物理 pp.1–6；p.1身份、p.2条件/更新、p.4范数界目检。 | Science：有限严格可分集、正阈值加法更新下有限次修正；后续连续推广未完整审核。 |
| Oja, A Simplified Neuron Model as a Principal Component Analyzer (1982) | J. Math. Biology 15:267–273，7 页；PDF pp.1–5，重点 p.3式(5)–(8)、p.4 Theorem与主成分说明。 | Science：条件平均 ODE；Knowledge：离散随机归一化学习与小步近似。 |
| Hopfield, Neurons with Graded Response Have Collective Computational Properties like Those of Two-State Neurons (1984) | PNAS 81:3088–3092，6 页含一页PNAS封面；物理 pp.2–5选读，p.2身份、p.4式(7)–(10)目检。 | Science：对称连续网络的条件 Lyapunov 耗散；非零外输入项有明确的原式符号修复说明。 |
| Kohonen, Correlation Matrix Memories (1972) | IEEE Trans. Computers C-21(4):353–359，7 页原刊扫描；物理 pp.1–5，主要核对 pp.1–2式(1)–(6)及残缺key限制。 | Science：完整线性矩阵存取的信号/串扰恒等式，等范数正交key的精确恢复。 |
| Kohonen, Self-Organized Formation of Topologically Correct Feature Maps (1982) | Biological Cybernetics43:59–69，11 页；物理 pp.1–5、9–10，Eq(1)–(3)、Simulations1–4、Table1与§4.1–4.5。 | Knowledge：特定输入、阵列及邻域更新的映射模拟，含失败模式。 |
| Anderson, A Simple Neural Network Generating an Interactive Memory (1972) | Mathematical Biosciences14:197–220，24 页；物理 pp.1–13分批选读，重点线性投影、局部乘积、矩阵叠加及噪声近似。 | Science：规定线性模型中的存储代数；MN/K噪声预测未作为一般容量定理录入。 |

**形式可实现性与学习不是同一个主张。** McCulloch–Pitts 的全或无、固定阈值、固定结构、绝对抑制和离散延迟必须保留；无环部分的命题表达式还有全假输入仍为假的限制。构造性表示能力不等于学习机制、生理解释或任意布尔函数。Rosenblatt 1958 则明确区分原刺激回测与新刺激类别测试：无结构的 ideal environment 不提供超机会泛化，differentiated environment 的预测取决于类内外共同激活等参数。正文的概率近似与另文待报 IBM704 模拟不能变成所有任务的保证，1962 年书也没有被当作已经选读。

**有限纠错与随机渐近分开。** Novikoff 要求存在 y 使每个 w_i 都满足 (w_i,y)>Θ>0，只有 (v,w_i)≤Θ 时才作 v←v+w_i。有限向量集给出单次范数增长的统一界；沿分离方向累积的投影给出二次下界，故修正不能无限持续。公平呈现用于最终满足全部训练约束，不推出非可分数据收敛、最优间隔、泛化或墙钟时间界。Oja 的 Science 只限固定半正定二阶矩、单重正最大特征值和非零初始主投影的 ODE；本目录显式排除退化零谱。零均值时才是协方差 PCA。本文对随机逼近技术条件明确略去细节，小常步长时又忽略统计波动，因此没有把这些表述转为任意随机训练的精确收敛定理。

**连续网络不能照搬二元假设。** Hopfield 1984 的连续能量模型允许自连接；零自连接、零外输入和高增益是后面与离散模型作对应分析的附加条件。原刊物理 p.4/印刷3090 Eq(7)外输入项确实印为 +ΣI_iV_i，但 Eq(5)、(8)–(9)要求负号。索引最稳妥的直接主张取 I=0；保留常量输入时，明确标示把该项修为 −ΣI_iV_i 的代数修复，不伪装成原文无误。耗散不证明唯一吸引点、全局最优或指定模式容量；延迟、非对称和噪声系统另有范围。

**串扰恒等式与噪声容量预测分开。** Kohonen 的完整矩阵 M=cΣxqᵀ 对任意给定向量有精确存取分解；等范数正交 key 才使统一 c 得到精确回忆。随机连接采样需要独立性，残缺 key 还需检查保留部分的相关性。Anderson 的局部乘积给出线性叠加构造，但噪声近似依赖等功率零均值迹、相关性、固定连接度和大样本条件；仅有“不相关”不足以自动给出高阶乘积因式分解或全部 CLT 条件。这里没有把 MN/K、编码最优性或生物类比作为普遍 Science。

**SOM 的边界是主张的一部分。** 原始模拟给出邻域竞争与归一化下的排序，也展示邻域太短的 pinch、太长的 collapse，以及弱相互作用/不当归一化的 focusing。局部适应的解释和有限阵列实验没有给出任意输入拓扑、初始化及边界下的全局保序保证，维持 Knowledge；未因经典地位或题名而提升为 Law。

**材料与未闭合来源。** 8 份原文均检查 PDF 签名、页数、字节数和 SHA-256，合计 9,661,159 bytes，最大 2,301,161 bytes，均位于已忽略的 papers/。下载和失败记录见 papers/pretrain/early_neuron_access_2026-10-28.json；身份依据分别登记，不把公开课程/研究镜像叫成作者或出版社。Novikoff 无可用文本层，采用 OCR 加关键页目检；Hopfield 含封面，物理页码与印刷页码分开。每次提取最多8页；这批是实际选读及有限公式核查，不是全文精读、所有证明独立审核或实验复现。

Little 1974《The Existence of Persistent States in the Brain》仍为**原文获取缺口**：SZTAKI镜像下载403，另一个公开旧镜像的网页读取返回502；未反复重试受限入口，也未根据摘要伪造主张记录。Rosenblatt 1962《Principles of Neurodynamics》作为待选章节的书籍资源候选保留，不机械加进论文数。后续可进一步核查 Oja–Karhunen 的完整随机算法证明、早期 associative memory 后继与容量理论；本轮不声称经典链已穷尽。

本批未改 panel、冻结稿、共享构建器或生成总表。完成本批后，pretrain 索引及 coverage 写权限交还根线程。

## 2026-10-28：统计可学习性、样本复杂度与计算边界

本批新增 **8 篇原始论文、7 个概念**，交接快照为 **343 篇论文 / 303 个概念**。VC1971与Blumer1989合并到同一个统计可学习性概念；未按论文数机械造概念。三库pretrain/rl/interp的标准化题名和PDF SHA去重通过，原有335篇论文和296个概念的记录保持不变。本批Science都指向下表指定的条件主张，不是对整篇论文、模型定义或现实大模型的评级。

| 原文 | 保存版本与选读位置（物理PDF页） | 主张及关键边界 |
|---|---|---|
| Valiant, A Theory of the Learnable (1984) | CACM27(11):1134–1142，9页；pp.1–6，TheoremA及协议。 | 固定k-CNF的删子句算法保证；原始EXAMPLES只返回正例，部分赋值的语义与通常现代二元带标签PAC表述分开。 |
| Vapnik–Chervonenkis (1971) | 17页B.Seckler英文译本，264–280；pp.1–11，Theorems1–3及固定P熵定义。 | 可测事件类的iid一致频率界；多项式增长是分布统一的充分条件，不把无限VC等同于每个固定P都不可学。 |
| Blumer–Ehrenfeucht–Haussler–Warmuth (1989) | JACM36(4):929–965，37页；pp.1–8、12–14、23–26。 | 非平凡well-behaved类的有限VC刻画；统计样本保证与proper多项式一致假设搜索的计算条件分开。 |
| Ehrenfeucht–Haussler–Kearns–Valiant (1989) | 作者站16页未标日期排印稿；pp.1–8。发表版本为I&C82(3):247–261，不能用稿件页数冒充期刊分页。 | 最坏分布下Ω(d/ε)及置信项下界；即使无限计算也有信息限制，不代表每个分布困难。 |
| Haussler (1992) | I&C100(1):78–150，73页正式期刊版；pp.1–5、9–30。 | 有界损失、iid联合分布、类内近似经验优化加一致偏差控制；目标是接近inf_H风险，优化算法的成功另证。 |
| Littlestone, Winnow (1988) | MachineLearning2(4):285–318，34页；pp.1–3、13–17。 | WINNOW1对固定单调析取的O(k log n)在线犯错界；任意顺序但无噪反馈，假阳性清零与WINNOW2区分。 |
| Kearns–Valiant (1994) | JACM41(1):67–95，29页；pp.1–8、11–15、19–21。 | 明确RSA/QR/Blum假设下的计算归约；允许换假设表示但须可高效求值，不是无条件不可能性。 |
| Haussler–Kearns–Littlestone–Warmuth (1991) | I&C **95**(2):129–161，34页扫描重印；pp.1–14 OCR，p.1身份和p.12引理目检。 | Lemma3.3单/双例子oracle的条件等价；regular₂、已知s、随机且必停机、转换会改变分布。 |

**定义不是定理，样本不是计算。** Valiant的Science落在TheoremA：对固定k枚举有限候选子句，从正例删除不满足的子句，保留目标子句，因此不产生假阳性，并以概率界控制漏掉的正例质量。k增长时成本条件改变。Blumer的Theorem2.1则以可实现目标、iid数据及well-behaved可测性建立统计刻画；正文明确不以此保证计算性。其§3.1的proper高效学习还需表示/求值条件及多项式一致假设寻找算法。原始PAC定义、VC维定义本身不据此升级成Science。

**分布量词必须保留。** VC1971的增长函数界控制集合类的频率偏差；后面固定P的熵条件不是同一个量词。Ehrenfeucht的Theorem1用被打散点集构造困难P，在ε≤1/8、δ≤1/100、d≥2时给m≥(d−1)/(32ε)。它允许随机化和不受目标表示限制的假设，说明某些目标/分布的样本需求，不能用来断言每个数据集或大语言模型必须服从相同样本门槛。

**有噪标签与无界损失不能混为一谈。** Haussler1992允许任意固定P(x,y)，因而不要求存在确定且无误的类内目标；但所选读框架要求损失在[0,M]内，并有permissible可测性条件。Lemma1区分经验优化精度与估计误差。平方loss需额外输出界；负对数loss须限制预测概率远离零。有限类和覆盖数界不保证SGD找到近似经验最优，也不等于相对任意决策规则的Bayes最优。

**在线顺序与测试风险分开。** Winnow1的初始权重为1，假阴性时乘α、假阳性时将激活属性清零。取α=2、θ=n，Theorem7给2k(log₂n+1)+1犯错界。这个保证允许对手安排顺序，却要求每次得到固定目标的正确标签；它不承诺错误何时出现，也不能跳过转换而把总犯错次数当终止模型的PAC风险。原文1988期刊年已与1987收稿/会议年份区分。

**计算障碍来自指定假设。** Kearns–Valiant把明确的密码问题归约到分布无关弱学习，Theorems4–6的目标类与大小/深度参数均需保留。记录允许任意多项式可求值的假设表示，避免误写成仅proper学习困难；也没有把密码假设压成P≠NP、任意单向函数或现实Transformer的普遍限制。正文§4.1的Blum整数为两个各自3 mod4素数之积，未沿用摘要中容易误读的简略括注。

**接口等价也有Scope。** 1991论文Lemma3.3的正向转换使用等权混合分布；逆向转换收集两类样本，罕见类不足时输出空集或全集。原文regular₂包含可高效求值、可测性与空/全/单点假设。由于转换改分布，不能将此等价无条件移到预先固定的分布对。完整模型等价图另涉及未知目标复杂度s时的概率停机，原文明确这种停机甚至不保证有限期望时间；本轮没有声称逐条审完图中所有转换。正确卷号为95，已对照首页与出版商元数据。

**材料与缺口。** 八份PDF均检查签名、pdfinfo页数、字节数及SHA，合计17,009,806 bytes，最大4,242,560 bytes；均在已忽略的papers/。下载/失败记录见papers/pretrain/statlearning_access_2026-10-28.json。每次提取不超过8页，所列阅读为定向选读；没有全部全文精读、全部证明独立审计或实验复现。1991重印扫描无可用文本层，已OCR并对身份及引理关键公式目检，人工身份登记交由根线程维护。

Blumer等1987《Occam's Razor》保留为**本轮原文获取缺口**：所试作者入口TLS失败、文本镜像403，未继续重复受限入口或用二手讲义替代阅读；本轮按授权以1991协议等价原文替换，没有把Occam1987加入论文计数。该来源值得后续补录，但不改变用户以Knowledge/Law/Science和Scope为核心、而非以MDL为主线的定位。Littlestone的WINNOW2、1991完整等价图、Haussler1992神经网络专门界及相关噪声学习/boosting文献仍未闭合。

本批未修改panel、冻结稿、共享构建器或生成总表。完成核验后交还pretrain索引、coverage和readingcards写权限。

## 2026-10-28：Gemini 原始模型族报告补录

由 interp 综述引文追踪发现三库未纳入 Gemini《Gemini: A Family of Highly Capable Multimodal Models》（arXiv:2312.11805）。本次取得 Google DeepMind 官方90页修订稿（©2025，27,110,234 bytes），原2023 arXiv v1下载未完成，未伪装成已取得历史初版，也未断言官方文件与arXiv v5逐字节一致。定向读取pp1–8、73–75，目检Table2和Figure9，按具体设计与评测归Knowledge。

新增1项文献、2个概念：原生多模态联合预训练、验证集阈值路由的CoT聚合；当前pretrain为344/305。MMLU 90.04%对应32条CoT与逐模型验证集阈值路由，普通CoT@32为84.99%、ScoreEval为83.96%；Table2中的5-shot 83.7%是另一个协议。原文所列数据污染、评测设置、模型版本及推理预算边界均保留，不从最高分推断全条件优势。未复现实验、核全部多模态结果或重建完整训练配方。来源与获取记录见 `papers/pretrain/gemini_access_2026-10-28.json`。
