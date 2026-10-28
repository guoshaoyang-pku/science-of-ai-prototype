# 预训练经典原文阅读卡

更新时间：2026-10-28。这里记录对原文指定位置的检查，不表示通读全文，也不表示独立复现。分类的对象是下列具体主张，不是对整篇论文定级；标为 Science 的推导只在其模型与假设中成立。详细一手来源、PDF 哈希和独立的分类记录见 [索引](pretrain.json)。

1. **神经概率语言模型 — Knowledge。** [Bengio et al. (2003)](https://jmlr.csail.mit.edu/papers/volume3/bengio03a/bengio03a.pdf) §4.2、表 1，学习分布式词表示并建模词概率。在 Brown 实验里，表中 MLP10 与 trigram 混合模型的困惑度为 252，比较的 class n-gram 为 312；252 不能写成纯 MLP 的成绩，数据集与比较设置约束了外推。

2. **Xavier 初始化 — Science（方差分析）。** [Glorot and Bengio (2010)](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) §4.2.1、式 (4)–(16)，考察初始化时前向与反向传播的方差，并折中给出以输入、输出宽度为参数的初始化。分析依赖独立权重、对称激活函数在零附近近似线性等条件；它不是一般训练收敛或泛化保证。经验表现另属具体实验 Knowledge。

3. **He 初始化 — Science（特定方差推导）。** [He et al. (2015)](https://arxiv.org/abs/1502.01852) §2.2、式 (6)–(10)，在作者的 ReLU 卷积网络及独立性、对称性等假设下得到保持方差的条件，采用方差 2/n。图 2–3 中 22 层案例两种初始化最终准确率相近，而在所测 30 层案例中一种停滞、He 初始化收敛；不能把它推广成所有架构均占优。

4. **Dropout — Knowledge。** [Srivastava et al. (2014)](https://jmlr.csail.mit.edu/papers/volume15/srivastava14a/srivastava14a.pdf) §2–3、§5，训练时随机丢弃单元，测试时的权重缩放用于近似多种瘦网络的预测。单个线性单元的期望等式不意味着整个非线性网络的测试输出精确等于集成平均；0.5 的保留概率是启发性建议，并非证明得到的普遍最优值。

5. **AdaGrad — Science（限定条件的 regret 界）；算法本身可另记 Knowledge。** [Duchi et al. (2011)](https://jmlr.csail.mit.edu/papers/volume12/duchi11a/duchi11a.pdf) §1 与定理 5，在论文指定的凸损失、可行域与有界条件下分析自适应梯度方法的 regret。该界不能直接当作非凸大语言模型训练的收敛保证。

6. **噪声对比估计 — Science（条件性统计性质）。** [Gutmann and Hyvärinen (2010)](https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf) §2.3、定理 1–3，通过区分数据样本与噪声样本构造非归一化模型的估计，并陈述一致性与渐近性质。核对时应保留噪声覆盖数据支持、相关一致收敛与秩等条件；会议稿没有给出证明，因此这里只核验了命题和条件，不能称已独立核验完整证明。

7. **梯度裁剪 — Knowledge。** [Pascanu et al. (2013)](https://arxiv.org/abs/1211.5063) §3.2 算法 1、§4.2 表 2，对超过阈值的整体梯度按范数缩放，保持当前梯度方向；所测字符预测实验中测试熵从 1.50 降至 1.42 bits。论文也指出裁剪不能解决梯度消失；指标与效果依赖该网络和任务。

8. **深线性训练动力学 — Science（可解模型）。** [Saxe et al. (2013)](https://arxiv.org/abs/1312.6120) §1–2，函数虽为线性，分层权重的学习动力学仍非线性；作者在特定数据与初始化、连续时间小步长极限下分析学习阶段与深度。真实非线性 Transformer 是否满足对应条件，须另作检验。

9. **学习率预热机制 — Knowledge，机制为待检验解释。** [Kalra and Barkeshli (2024)](https://arxiv.org/abs/2406.09405) §1–3，在多个网络、数据和优化器设置下研究 warmup 与可承受学习率及预条件化曲率的关系。论文提出、分析并测量相关机制，但这些结果并不证明所有训练过程都以同一种机制受益。

10. **DoReMi 数据混合 — Knowledge。** [Xie et al. (2023)](https://arxiv.org/abs/2305.10429) §2 算法 1、§4，用相对参考模型的域级超额损失训练代理并调整域权重，再以所选混合训练更大的模型。作者报告在其任务和设定中 8B 模型以约 2.6 倍速度达到基线精度；代理大小的效果并非简单单调，该数字不保证所有语料域、模型、完整成本口径均成立。

11. **数据混合定律 — Law。** [Ye et al. (2024)](https://arxiv.org/abs/2403.16952) §3.2、附录 B，拟合训练数据不同域的混合比例与域损失关系，并用它选择混合；所用域、数据、模型族与预算构成经验范围。不能凭一个拟合式推断跨任意域或分布都成立。

12. **可提取记忆与重复 — Law。** [Carlini et al. (2022)](https://arxiv.org/abs/2202.07646) §3.2–4.3、图 1，考察指定模型和语料下训练文本重复与可提取性的关系。评估依赖选取的前缀及 50-token 后缀等操作性定义；不能将其等同于全部记忆或全部隐私风险。

这些卡片并未解决“Law 已充分释放所有限制”和“Science 知道边界”的一般判别问题：本语料提供的是可核查的候选案例。实际论文可能同时包含算法、经验现象、理论模型三个主张；应逐主张标注 scope，再讨论它在该范围内的成熟程度。

## 核心预训练模型与优化：第二轮定向阅读

13. **BERT — Knowledge。** [Devlin et al. (2018)](https://arxiv.org/abs/1810.04805) §3.1、§5.1 表 5：选中 15% 的 WordPiece 位置预测，其中按 80%、10%、10% 分别用 MASK、随机词、原词替换；同时训练 NSP。消融限于作者的数据、输入构造和任务，不能将所有双向预训练都当作 BERT 原始目标。

14. **RoBERTa — Knowledge。** [Liu et al. (2019)](https://arxiv.org/abs/1907.11692) §4.1–4.4、尤其表 2：比较动态遮掩、输入格式、批量与预训练时长等。其无 NSP 对照同时变化文本组织，因而不能仅从该表归因“任何场景去掉 NSP 都有益”；结果限于所测训练方案及任务。

15. **T5 — Knowledge。** [Raffel et al. (2019)](https://arxiv.org/abs/1910.10683) §2、§3.1.4、§3.3.4、表 7：统一输入输出为文本，比较去噪任务。基线独立抽取 15% token、合并连续被删除位置为带 sentinel 的片段；§3.3.4 另试主动选取连续 span，平均 span 长度 3 在大多数所测非翻译任务中略优于基线。不能把个别方案说成所有任务最优。

16. **GPT-3 的上下文 few-shot — Knowledge。** [Brown et al. (2020)](https://arxiv.org/abs/2005.14165) §2、图 2.1 与 §3–5：few-shot 指推理时在上下文提供示例、无需更新权重；作者也研究 one-shot 和 zero-shot。表现受任务、示例和提示、上下文长度、污染与模型大小影响；论文 §5 亦记录其缺点。

17. **Adam 与原有收敛论证 — Knowledge（算法）。** [Kingma and Ba (2014)](https://arxiv.org/abs/1412.6980) §2 算法 1、§3–4：利用一阶和二阶矩指数平均与初始偏差校正。不能把原文所写一般收敛界直接作为成立的保证：[Reddi et al.](https://arxiv.org/abs/1904.09237) §1–3 指出原有证明的问题，并构造特定凸问题反例；这也不等于 Adam 在所有场景不收敛。

18. **AdamW 的解耦分析 — Science（明确更新形式下的关系）。** [Loshchilov and Hutter (2017)](https://arxiv.org/abs/1711.05101) §2 命题 1–2、算法 2：在文中标准 SGD 更新下，L2 惩罚与权重衰减经系数换算等价；一般自适应预条件更新下，该等价关系不再成立。AdamW 是基于此引入解耦衰减的算法，具体效果仍是有设置边界的经验判断。

19. **Muon 扩展 — Knowledge。** [Liu et al. (2025)](https://arxiv.org/abs/2502.16982) §2.2、§3.1–3.2、表 2 与图 3：对矩阵更新进行正交化及形状相关尺度调整，并用 AdamW 处理非矩阵参数。其报告 Muon 拟合曲线约使用 AdamW 的 52% 训练 FLOPs 匹配损失；比较为指定密集 Llama 风格模型、超参数及预算，并非通用墙钟加速率。

20. **Pile — Knowledge。** [Gao et al. (2020)](https://arxiv.org/abs/2101.00027) 摘要、§2–4：由 22 个子集构成约 825.18 GiB 英文语料，在指定语言模型和评测中比较跨域困惑度与下游表现。多来源设计及经验结果不自动证明多样性在所有模型、语言或任务均有因果收益。

21. **FineWeb 与 FineWeb-Edu — Knowledge。** [Penedo et al. (2024)](https://arxiv.org/abs/2406.17557) §3.1、§3.4、§4：通过 1.71B 参数模型消融数据提取、过滤及去重；FineWeb-Edu 用 Llama-3-70B-Instruct 合成标注训练教育内容过滤器。其论文中 MMLU 和 ARC 的结果依赖指定数据、基准与阈值，不能由此推断任何领域都优于其他数据。

22. **DataComp-LM — Knowledge。** [Li et al. (2024)](https://arxiv.org/abs/2406.11794) §3、§4.2–4.4、表 2–4：基准控制训练和评测配方以研究数据构造；在所测 1B-1x 筛选对照中，fastText OH-2.5+ELI5 得到表内最高 CORE。此结论限定于该语料池、对照和预算，不构成跨域最优证明。

23. **NTK — Science。** [Jacot et al. (2018)](https://arxiv.org/abs/1806.07572) §4.1–4.2 定理 1–2、§5：在其全连接网络无限宽极限，初始化 NTK 收敛至确定核；在额外激活光滑性和训练方向积分有界条件下，核在有限训练区间保持不变。该定理不能直接覆盖有限宽 Transformer 的一般特征学习。

24. **Lazy training — Science。** [Chizat et al. (2018)](https://arxiv.org/abs/1812.07956) §2.1–2.2、定理 2.2–2.3：对特定缩放下的梯度流，给出其与初始点线性化动力学之差的界。零输出初始化、局部光滑性和有限时间范围是相关条件；作者实验也讨论此极限对视觉模型性能的影响，不应断言所有训练都是 lazy。

25. **最大更新参数化与 μTransfer — Science / Knowledge 分开。** [Yang et al. (2022)](https://arxiv.org/abs/2203.03466) §2–4、§6.1、图 4 与表 1：参数化及宽度极限的解释属于条件性理论主张；小模型向大模型转移部分超参数的效果是经验主张。原文指出正则化超参数不能简单迁移、初始化尺度跨深度和 pre-/post-LayerNorm 条件存在限制，不能推广为全部超参数可迁移。

## 第三轮：预训练目标、数据、规模及系统

- **目标与架构。** [Transformer](https://arxiv.org/abs/1706.03762) §3.2–3.5 将缩放点积、多头注意力、位置及前馈模块组合用于编码与解码；所报优势限定所测任务。[XLNet](https://arxiv.org/abs/1906.08237) §2.2–2.3 变化的是概率**分解顺序**而非输入序列顺序。[ELMo](https://arxiv.org/abs/1802.05365) §3.2 用双向语言模型的各层加权构建任务相关表示。[Skip-gram](https://arxiv.org/abs/1310.4546) §2.2 的负采样从 NCE 得到启发，但不是同一个目标。均按具体方法和实验归 Knowledge。

- **节省优化器内存。** [Adafactor](https://arxiv.org/abs/1804.04235) 算法 2 对矩阵梯度平方使用行列统计来近似二阶矩，空间收益有形状依赖；这不是完整二阶矩的精确重现。归 Knowledge。

- **去重与数据配方。** [Lee et al.](https://arxiv.org/abs/2107.06499) §4–6、图 2 区分近重复文档、相同子串和验证集是否重复，结果受阈值及模型影响。[RefinedWeb](https://arxiv.org/abs/2306.01116) 表 5 指出过滤效果并非对每种语料一致，且其流水线不能独立实施 RefinedWeb 的去重阶段，不能把整体提升全部归因去重。[Dolma](https://arxiv.org/abs/2402.00159) §4.2、表 1 的数据消融限定于作者语料、模型与评价方式。这些数据集及方法归 Knowledge。

- **数据受限的规模关系。** [Muennighoff et al.](https://arxiv.org/abs/2305.16264) §3–6、图 1–4 研究在 GPT-2 架构和 C4 子集上重复数据的回报衰减；报告约 4 个 epoch 在所测条件下损失变化很小，不能把 4 当作普适阈值。分类为有范围的 Law。

- **Kaplan 与 Chinchilla 的差异。** [Pearce and Song](https://arxiv.org/abs/2406.12907) §2.3、图 1–5 是基于给定损失模型、参数换算和计算近似的模拟解释；[Porian et al.](https://arxiv.org/abs/2406.19146) §3、表 1 使用两种网页数据、训练运行和配置对照，研究末层 FLOPs、warmup 及优化器调参。两者均提供受方法限制的解释，不应合并成已排除全部其他机制的结论。经验关系暂归 Law。

- **初始化理论与 BatchNorm 机制。** [Deep Information Propagation](https://arxiv.org/abs/1611.01232) §2–4 的随机网络均值场深度尺度归条件性 Science，不能视作所有有限残差网络的深度界。[Santurkar et al.](https://arxiv.org/abs/1805.11604) §2.1 的噪声干预、§3 的曲率分析支持在所测设置中重新审视 BatchNorm 原有解释；不证明所有网络只有单一机制，实验主张归 Knowledge。

- **视觉自监督。** [SimCLR](https://arxiv.org/abs/2002.05709) §2、图 2 用双增强视图、投影头和对比损失；[MoCo](https://arxiv.org/abs/1911.05722) §3.2 用队列和动量教师构造对比字典；[CLIP](https://arxiv.org/abs/2103.00020) §2.3、图 3 以图文配对批次对比训练双编码器；[MAE](https://arxiv.org/abs/2111.06377) §3–4、图 5 遮挡图像块并重建像素，其约 75% 遮挡是特定实验结果；[ViT](https://arxiv.org/abs/2010.11929) §4.3、图 3 显示预训练数据量影响大型 ViT 效果；[DINO](https://arxiv.org/abs/2104.14294) §2、图 3–4 提供自蒸馏特征与注意力可视化。均归具体模型与实验的 Knowledge。

- **混合精度及并行系统。** [混合精度训练](https://arxiv.org/abs/1710.03740) §3 区分 FP16 前后向、FP32 主权重和损失缩放；[Megatron-LM](https://arxiv.org/abs/1909.08053) §3、图 3–4 通过层内矩阵切分及通信构造张量并行；[ZeRO](https://arxiv.org/abs/1910.02054) §4.1 分片数据并行的模型状态，峰值内存还涉及激活与缓冲。上述工程方法归 Knowledge，开销与收益均依具体硬件、实现和网络。

- **FlashAttention。** [Dao et al.](https://arxiv.org/abs/2205.14135) §3.1–3.2、定理 1–2 对指定 SRAM/HBM IO 模型给出分块注意力的数学结果和访存复杂度，归 Science；硬件加速实验归 Knowledge。数学上精确计算同一注意力不意味着浮点输出必须逐位相同。

- **GPT-4 报告。** [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) §2–4 报告指定能力评测和尺度预测，但明确未披露模型大小、硬件、训练算力或数据集构造。将所报告结果归 Knowledge，不能虚构其训练配方。

## 十项 Law／Science 分类边界复核

本节只核查列出的关键原文位置，未逐行审阅完整证明或全文。每项的主标签是索引对一个选定主张的初步分类；同文不同主张在 `claim_annotations` 中分开。

1. [Kaplan scaling](https://arxiv.org/abs/2001.08361) §1.2 式 (1.1)–(1.3)、§3、§6：**Law**。参数、数据、计算三条拟合各有不同瓶颈条件；参数量不包含嵌入参数，数据受限结果用早停。拟合指数不是无条件常数。计算分配的建议可另视为特定设置的 Knowledge。

2. [Chinchilla](https://arxiv.org/abs/2203.15556) §3.2、图 3、表 2：**Law**。等训练 FLOPs 下损失谷底拟合得参数和 token 的指数约 0.49、0.51；依赖作者的训练日程、数据与模型族，不包括部署与推理的总成本。具体模型评测另属 Knowledge。

3. [Explaining Neural Scaling Laws](https://arxiv.org/abs/2102.06701) §2.1 定理 1、§2.2 定理 2–3：**Science**。前者以网络输出集中和损失的正则性为前提；后者以紧致数据流形、i.i.d. 采样、光滑性和插值为前提，给的是上界。作者在 §2.2.3 对更强速率作预期，不能把它改写成已证明的真实 LLM 精确幂律。实验中的拟合另属 Law。

4. [A Solvable Model of Neural Scaling Laws](https://arxiv.org/abs/2210.16859) §1、§3.1–3.2、§4.1：**Science**。潜变量数据生成、随机特征与解析线性回归组成可解模型，其中有限潜空间规模可使规模关系失效。原文脚注强调解析最优相当于不受训练计算约束；模型与实际 Transformer 的对应是待检验的推测。

5. [μP 和 μTransfer](https://arxiv.org/abs/2203.03466) §2–4、§6.1、表 1、图 4：参数化的宽度极限论证暂归 **Science**，理论基础还引用前作；具体超参数迁移结果属 **Knowledge**。作者指出正则化超参不直接迁移，某些跨深度迁移依赖初始化或 pre-LN 设置。

6. [NTK](https://arxiv.org/abs/1806.07572) §4.1–4.2 定理 1–2：**Science**。定理 1 是初始化无限宽极限；定理 2 还要求二次可微、二阶导有界和训练方向积分随机有界，结论作用于指定有限时间段。有限宽实验与现实模型对应另属 Knowledge。

7. [Edge of Stability](https://arxiv.org/abs/2103.00065) §2–3、图 3–5：**Law 候选**。二次模型中固定步长的局部稳定阈值是条件性数学结论；论文在神经网络全批量梯度下降中观察到 sharpness 附近的非单调训练，不能把它套成 Adam 或任意 minibatch 的定律，也尚未穷尽失效边界。

8. [Gradient Noise Scale](https://arxiv.org/abs/1812.06162) §2.2–2.4、§3：**Law 候选**。在独立采样及有限协方差下，小批量梯度的方差缩减关系是条件性数学事实；预测临界批大小及训练计算权衡还依赖局部二次近似、学习率选择与实测训练任务，不保证泛化或全局优化最优。

9. [Deep Double Descent](https://arxiv.org/abs/1912.02292) §2、§5–7、图 4 和图 9：经验现象暂归 **Law 候选**。作者将广义机制写为非形式化假说；噪声常放大误差峰，部分无噪声设置可有峰，也有平台。实验支持多种设置中的非单调性，但不能当作所有模型都会出现双下降的证明。

10. [Emergent Abilities: A Mirage?](https://arxiv.org/abs/2304.15004) §2–5、§7：将所选实验主张从 Law 调整为 **Knowledge**；独立 token 条件下严格序列准确率的数学示例可另记 **Science**。若干任务的表观突变随度量或采样变化，并不证明真实涌现不存在——作者在 §7 明确说明这一点。

## 第四轮：20 项方法、结构与适用边界

以下是对所列位置的定向核查；每项都仍非全文精读或独立实验复现。

- [BatchNorm](https://arxiv.org/abs/1502.03167) §3、算法 1：使用 mini-batch 均值和方差、可学习尺度和偏置；原文以内部协变量偏移解释训练改善，但这一机制不能视为已经独占成立。
- [LayerNorm](https://arxiv.org/abs/1607.06450) §3：统计量按单个样本的一层神经元计算；训练与推理计算相同，循环网络逐时间步归一化。
- [RMSNorm](https://arxiv.org/abs/1910.07467) §3–4：以均方根缩放而不减去均值；关于重新中心化是否必要是作者假说，epsilon 和模型结构影响精确不变性。
- [RoPE](https://arxiv.org/abs/2104.09864) §3.2–3.3：对查询、键按绝对位置旋转，点积中体现相对位置；这一代数关系不保证任意长度外推成功。
- [ALiBi](https://arxiv.org/abs/2108.12409) §3、图 1 和图 3：分头固定斜率对注意力分数施加距离偏置；外推表现受模型、语料和长度影响。
- [稀疏门控 MoE](https://arxiv.org/abs/1701.06538) §2、图 1：每个输入只选少数前馈专家；总专家参数量不是每个 token 的激活参数量。
- [Switch Transformer](https://arxiv.org/abs/2101.03961) §2.1–2.2、图 3：将稀疏路由简化为每 token 一个专家，仍需考虑专家容量及负载均衡。
- [GLaM](https://arxiv.org/abs/2112.06905) §4、图 2：每个 MoE 层每个 token 选两个专家；模型族、活跃参数和完整成本应分开比较。
- [Mixtral](https://arxiv.org/abs/2401.04088) §2.1–2.2：每个 token 在八个专家中选两个，专家并行仍需通信与负载均衡。
- [Mamba](https://arxiv.org/abs/2312.00752) §2–3、图 1：部分 SSM 参数依赖输入，采用硬件感知的 scan；线性扩展依赖状态维度等量固定，性能取决于实现与任务。
- [SSD／Mamba-2](https://arxiv.org/abs/2405.21060) §3.1–3.3、定理 3.5：论文定义的 SSM 算子与对应阶数的半可分矩阵表示相同。输入依赖的系数不意味着整个从输入到输出的映射是一个固定线性算子；定理证明未独立逐行审计。
- [GQA](https://arxiv.org/abs/2305.13245) §2.1–2.2、图 2：将查询头分组，每组共享一对键和值头；转换、质量和解码速度受具体训练与系统条件约束。
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) §2.1.2，式 (16)–(17)：专家偏置参与选择但不直接乘入专家输出权重；完整模型还使用补充的序列级平衡辅助损失，不能将其描述为没有任何辅助损失。
- [LLaMA](https://arxiv.org/abs/2302.13971) §2.1、表 1：模型训练来自多种公开来源的混合数据；不同规模的 token 和数据配方应分别记录。
- [Transformer-XL](https://arxiv.org/abs/1901.02860) §3.2–3.3：相邻片段缓存隐藏状态并采用相对位置；跨片段梯度截断、缓存长度有限。
- [BART](https://arxiv.org/abs/1910.13461) §2、图 2：双向编码破坏后的文本，解码器自回归重建；文本填空和句子置换是不同破坏方式。
- [ELECTRA](https://arxiv.org/abs/2003.10555) §2：判别器识别生成器替换的位置；生成器以最大似然训练，不是 GAN 式对抗训练。
- [wav2vec 2.0](https://arxiv.org/abs/2006.11477) §3.3、式 (2)–(3)：对被遮掩时间步区分量化目标与同一话语的干扰项；下游识别微调另需带转录的数据。
- [HuBERT](https://arxiv.org/abs/2106.07447) §3：离线聚类产生语音伪标签，模型预测遮掩区域；标签不是预先给定的自然语言语义真值。
- [Whisper](https://arxiv.org/abs/2212.04356) §2.1、§3.8：以配对网络音频与转录进行弱监督学习，训练按 30 秒分段；它不是无标签语音自监督。

**概念修正。** “涌现测量和非线性指标”中的有限模型实证归 Knowledge；论文 §2 在 token 独立等假设下构造的数学示例可另列条件化 Science，并不证明所有涌现均为测量伪影。FlashAttention 的 Science 主张应限于原文 IO 模型：定理 2 假定 \(d \le M \le Nd\)，得到 \(\Theta(N^2d^2/M)\) 的 HBM 访问复杂度；模型不直接给出任意硬件的墙钟收益，定理证明尚未独立逐行审计。


## 第五轮定向复核：早期语言模型、优化与视觉架构

本轮将 20 篇论文从元数据初筛推进到所列章节或图表的定向核查。只核实明确位置，不等于全文精读、证明逐行审计或独立复现。

- [Word2Vec](https://arxiv.org/abs/1301.3781) §3.1–3.2、图1：CBOW 用上下文预测中心词，skip-gram 用中心词预测上下文；类比表现是指定词表、语料和任务中的观察。
- [BPE for NMT](https://arxiv.org/abs/1508.07909) §3.2：以字符或子词合并建立可变长度词单元；实验限于作者翻译数据与分词设置。
- [SentencePiece](https://arxiv.org/abs/1808.06226) §2–3.1：BPE 与 unigram 模型可直接从原始句子训练；编码—解码的逆关系以规范化后的文本为准。
- [ULMFiT](https://arxiv.org/abs/1801.06146) §3：通用语言模型预训练、目标语言模型微调、分类器微调三个阶段；不能以有限英语分类实验推断适合所有 NLP 任务。
- [GLU Variants](https://arxiv.org/abs/2002.05202) §2–3、表1：通过缩小隐层比较参数量和算量匹配的 T5 FFN；不能把特定实验优势推广为门控结构处处恒优。
- [Transformer LayerNorm](https://arxiv.org/abs/2002.04745) §3.3、定理1：理论分析明确使用单头、零初始化查询和键、隐藏状态集中等条件；初始梯度结论不等于所有训练无需 warmup。
- [DeepNet](https://arxiv.org/abs/2203.00555) §2、图2：DeepNorm 缩放残差连接及指定权重初始化；具体深度和稳定性结论受架构、初始化及实验约束。
- [Large-Batch Generalization Gap](https://arxiv.org/abs/1609.04836) §1–2.1：作者在特定设置下观察大 batch、泛化差距和锐度的关系；实验关联不能独自确定唯一因果机制。
- [SGDR](https://arxiv.org/abs/1608.03983) §2、式5、图1：余弦衰减与周期性学习率重启；重启日程不等于重新初始化模型参数。
- [Deep Learning Scaling is Predictable](https://arxiv.org/abs/1712.00409) §1、§3–4：跨作者六个模型及四个应用领域拟合数据量学习曲线；经验指数依赖应用，非普适常数。
- [Rethinking Generalization](https://arxiv.org/abs/1611.03530) §1.1：指定视觉网络能够拟合随机标签和随机像素；记忆能力不否定真实标签任务中的泛化。
- [Grokking](https://arxiv.org/abs/2201.02177) §2–3、图1–2：小型 Transformer 在离散运算表上出现训练后延迟泛化，受到任务、训练比例和优化配置影响。
- [ResNet](https://arxiv.org/abs/1512.03385) §1、图2：F(x)+x 是残差学习构造；易优化的动机及作者实验不构成对所有损失的保证。
- [Sequence to Sequence](https://arxiv.org/abs/1409.3215) §1–2：编码 LSTM 将变长输入压缩为固定向量，另一个 LSTM 自回归解码；论文还反转输入顺序。
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) §1–3：解码时对输入隐藏状态作随目标位置变化的软对齐；对齐可视化不自动证明因果使用。
- [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) §1、图1：在编码潜空间利用上下文预测未来并用对比目标训练；表示能力受模态、负样本和任务影响。
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) §2–3、式3–5：直接传播的推导要求恒等 skip 和恒等加法后映射，尺寸变化单元不满足严格条件。
- [Stochastic Depth](https://arxiv.org/abs/1603.09382) §1–2：训练时随机跳过残差分支，测试使用完整网络；结果受所测视觉网络和协议限制。
- [VQ-VAE](https://arxiv.org/abs/1711.00937) §3、式3：离散码本通过直通估计、码本与 commitment 损失训练；直通估计不是离散操作的精确梯度。
- [Longformer](https://arxiv.org/abs/2004.05150) §1–3、图2：滑动窗口与任务设定的全局注意力组成稀疏图；它不逐项等价于完整自注意力。

**锚点修正。** [A Solvable Model of Neural Scaling Laws](https://arxiv.org/abs/2210.16859) 的现实外推描述改为作者提出的可能解释与待检验映射；不将理想化模型对真实 Transformer 的适用性写成已验证事实。

## 第六轮定向阅读：优化与多模态

本轮仅核对下列具体页码与位置，不代表全文审读、逐行证明核查或实验复现。分类指所选主张，scope 约束结论。

- [Shampoo: Preconditioned Stochastic Tensor Optimization](https://arxiv.org/abs/1802.09568)（Knowledge）：Shampoo 对张量梯度各维分别维护累积二阶统计矩阵，矩阵情形左右预条件器分别作用于梯度。 范围：算法及深网实测属于 Knowledge；论文的理论分析以在线凸优化为背景，不能据此声称非凸神经网络全局收敛。 依据：PDF 第 2–3 页 §1.1、Algorithm 1；矩阵情形的存储与计算复杂度。
- [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675)（Knowledge）：以程序搜索发现 Lion：用梯度和动量的插值符号更新参数，只维护动量，并用另一插值更新动量。 范围：作者的视觉、扩散和语言训练及特定超参数设定；符号更新需要不同学习率与权重衰减口径，部分大规模语言结果差异无统计显著性。 依据：PDF 第 2–3 页 Program 1 与引言贡献和限制。
- [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks](https://arxiv.org/abs/1602.07868)（Knowledge）：Weight normalization 用 w=g v/||v|| 将权重向量方向与尺度重参数化，直接优化 g 和 v。 范围：对非零 v 定义；无动量的直接梯度更新方向与 v 正交，不能无条件推广到动量、权重衰减或任意优化器。 依据：PDF 第 2–3 页 §2–2.1，式 (2)–(4)。
- [Large Batch Optimization for Deep Learning: Training BERT in 76 minutes](https://arxiv.org/abs/1904.00962)（Knowledge）：LAMB 将逐层自适应缩放用于大批量优化，并在论文设置中评估 BERT 训练。 范围：76 分钟是指定硬件、训练方案、批量和评测协议下的实验结果；不能等价为任意模型均能达到该速度，论文非凸收敛分析也依赖额外条件。 依据：PDF 第 2–3 页贡献和相关工作；具体非凸证明及所有实验未逐项核查。
- [Analyzing & Reducing the Need for Learning Rate Warmup in GPT Training](https://arxiv.org/abs/2410.23922)（Knowledge）：在 GPT 训练中分析 AdamW 初期更新、层间更新和梯度信噪比，并提出缓解 warmup 需求的优化器修改。 范围：主要实验为 124M GPT-2、OpenWebText、batch 480、序列 1024；特征变化导致问题的解释是作者假说，不能推广为所有架构都无需 warmup。 依据：PDF 第 2–3 页 §2–3、Figure 1；摘要外延仍需检验。
- [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677)（Knowledge）：研究同步大批量 SGD，提出批量扩大时按比例缩放学习率并配合初期 warmup。 范围：论文 ResNet-50/ImageNet 和部分检测任务及实现协议；线性缩放依靠相邻更新中梯度近似稳定，不保证任意 batch、任务或优化器有效。 依据：PDF 第 2–3 页 §2.1–2.2，式 (3)–(4) 与 warmup 动机。
- [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)（Knowledge）：ALBERT 用分解词嵌入与跨层参数共享降低参数量，并引入句序预测目标。 范围：指定 BERT 风格架构与语言任务；参数减少不等于 FLOPs、内存或墙钟时间在所有配置下等比例减少。 依据：PDF 第 2–3 页引言及 §2.2–2.3；实验未全面核对。
- [Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116)（Knowledge）：XLM-R 使用多语言掩码语言模型，仅以单语文本训练跨语言表征，并考察规模与语言资源差异。 范围：基于论文的 Common Crawl 数据、语言采样、模型容量和 XNLI 等评估；跨语言迁移并非证明所有语言不会发生容量稀释。 依据：PDF 第 2–3 页 §3 Model and Data 及语言数据 Figure 1。
- [Bootstrap your own latent: A new approach to self-supervised Learning](https://arxiv.org/abs/2006.07733)（Knowledge）：BYOL 用在线网络预测目标网络对另一增强视图的表征，目标网络参数来自在线网络的慢移动平均，无需显式负样本。 范围：损失仍允许常量坍缩解，作者关于预测头与移动平均避免坍缩的机制是针对实测设置的假说，不能当成一般证明。 依据：PDF 第 2、4–5 页引言、§3.1–3.2 和 Figure 2；关于坍缩的解释以作者假说标注。
- [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)（Knowledge）：线性 Transformer 用可分解的特征映射构造注意力，并利用结合律及因果累计状态得到随序列长度线性的计算。 范围：具体核函数和特征维度决定注意力形式与成本；该确定性特征映射不自动等价任意 softmax 注意力。 依据：PDF 第 2–3 页 §2.2–3；复杂度按论文运算模型，不等于任意设备上墙钟优势。
- [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794)（Knowledge）：Performer 的 FAVOR+ 使用正交正随机特征近似可核化注意力，并研究 softmax 核的近似。 范围：核估计、归一化注意力输出及随机特征近似误差须区分；有限特征数不是精确 softmax，速度和精度取决于实现与任务。 依据：PDF 第 2–3 页 §2–2.1 和引言的理论主张；证明尚未逐行审计。
- [mT5: A massively multilingual pre-trained text-to-text transformer](https://arxiv.org/abs/2010.11934)（Knowledge）：mT5 延续 T5 的编码器—解码器文本到文本和 span corruption 目标，用 mC4 扩展到多语言。 范围：mC4 的 cld3 语言识别和过滤有测量误差；101 个语言计数与书写变体口径、语料分布及训练采样共同限定结果。 依据：PDF 第 2–4 页 §2–3.2、Figure 1 与 Table 1；语言采样指数按作者设置。
- [Exploring Simple Siamese Representation Learning](https://arxiv.org/abs/2011.10566)（Knowledge）：SimSiam 用共享编码器、预测头和 stop-gradient 匹配两个图像增强视图，无需负样本或动量编码器。 范围：作者设定中 stop-gradient 的消融有重要作用；不证明该操作是任意自监督系统避免坍缩的普适必要条件。 依据：PDF 第 2–3 页 Algorithm 1、§3；实验与理论解释未全面复核。
- [BEiT: BERT Pre-Training of Image Transformers](https://arxiv.org/abs/2106.08254)（Knowledge）：BEiT 将输入图像切成 patch、遮蔽部分 patch，并预测由预训练离散视觉 tokenizer 生成的 token。 范围：目标是离散视觉 token 而非直接恢复原始像素；效果依赖 tokenizer、mask、模型与下游数据。 依据：PDF 第 2–3 页 Figure 1 和 §2.1。
- [Efficiently Modeling Long Sequences with Structured State Spaces](https://arxiv.org/abs/2111.00396)（Knowledge）：S4 通过结构化状态矩阵及其分解，高效在状态空间模型的循环与卷积表示之间计算。 范围：效率复杂度依赖论文指定参数化、算法与实现；指定长序列基准的表现不证明优于所有 Transformer 或所有任务。 依据：PDF 第 2–4 页 Figure 1、§2.1–2.3；理论证明和基准未全部核查。
- [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602)（Knowledge）：VideoMAE 对视频采用高比例管状遮蔽，并使用非对称编码器—解码器重建被遮蔽内容。 范围：90%–95% 遮蔽及避免静态区域时序泄漏是论文设计与有限基准观察，不能推广为所有视频和任务的最优比例。 依据：PDF 第 2–3 页 Figure 1、引言与方法动机。
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198)（Knowledge）：Flamingo 利用冻结视觉编码器和语言模型，训练 Perceiver Resampler 以及插入语言模型层间的门控交叉注意力。 范围：视觉／文本交错输入、特定数据混合和 few-shot 基准；冻结骨干不意味着新增连接模块不训练。 依据：PDF 第 4–5 页 §2、Figure 3–4；定量结果未独立复现。
- [UL2: Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131)（Knowledge）：UL2 以混合去噪目标统一不同 span corruption 与前缀式训练形式，并用模式标记区分目标。 范围：混合策略、模型、训练数据和任务限定比较；统一输入—目标形式不等于不同目标训练梯度或结果普遍等价。 依据：PDF 第 8–10 页 Figure 2–3、§3.1.1–3.1.2；消融与证明未全面核查。
- [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)（Knowledge）：BLIP-2 冻结图像编码器与语言模型，用 Q-Former 分阶段进行视觉语言表征学习和视觉到语言生成学习。 范围：第一阶段联合对比、图文匹配和图像条件文本生成；第二阶段训练桥接模块，冻结两端不意味着整个系统没有可训练参数。 依据：PDF 第 2–4 页 §3.1–3.2、Figure 2；全部实验未核查。
- [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)（Knowledge）：DINOv2 结合图像／patch 级自监督特征学习与数据过滤、去重和检索，构建 LVD-142M 并训练视觉编码器。 范围：数据由人工整理来源扩展及无标签网络图像组成，迁移性能限所测特征、模型和任务；自监督目标不代表数据分布没有筛选偏差。 依据：PDF 第 2、4–5 页 §3、Figure 3；实验与独立复现未完成。
- [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)（Knowledge）：LLaVA 将 CLIP 视觉编码器连接 Vicuna 语言模型，并用生成的多模态指令数据训练。 范围：原始数据构造将图像 caption 和边界框交给仅接收文本的 GPT 教师；合成回答可能继承上下文遗漏与教师错误。 依据：PDF 第 2–4 页贡献、§3、Table 1；实验未全部核查。

## 本轮：理论与经验规律的原文边界

- [Feature Learning in Infinite-Width Neural Networks](https://arxiv.org/abs/2011.14522)（Science，条件化）：v3 §3 Theorems 3.3–3.8、Corollary 3.9 在稳定且非平凡的 abc 参数化类中描述无限宽极限的特征学习／核动力学二分；论文 §2 说明训练时间不随宽度增长，固定输入维度。参数化定义与 SGD、初始化和损失均须按论文保留；Word2Vec 与 Omniglot 结果是有限实验，不证明任意 Transformer 或优化器符合该二分。证明未逐行审计。
- [A Stochastic Approximation Method](https://www.columbia.edu/~ww2040/8100F16/RM51.pdf)（Science，条件化）：§3 Theorem 2 使用均值函数非递减、目标处导数为正、观测一致有界及论文的 1/n 型步长，得到均方误差趋零并因此依概率收敛；论文没有声称此结果为全局深度学习训练保证。扫描件 PDF 第 2–7 页可提取文字但部分公式 OCR 不清；定理陈述已核对，证明未逐行审计。
- [A Constructive Prediction of the Generalization Error Across Scales](https://arxiv.org/abs/1909.12673)（Law，候选）：§3 Table 1、§6 Figure 3–4 在指定架构缩放策略、语言与图像数据上拟合参数／数据二维误差曲面；拟合与有限外推不能无条件推广到新任务、优化器或其他架构。PDF 抽取出现 xref 重建提示，相关文字可读；未独立复现实验。
- [Scaling Laws for Autoregressive Generative Modeling](https://arxiv.org/abs/2010.14701)（Law，候选）：§1–2 和 Table 1 对不同模态的自回归 Transformer 损失拟合幂律加常数；语言计算曲线借用先前 GPT-3/Kaplan 数据，且 §6 明确讨论数据与计算规模拟合的潜在不一致。不同编码和数据配置下观测到接近的最优模型规模指数，不等于任何模态或训练协议上的严格普适性。

## 2026-10-28：统计学习基础的选读卡

以下八项是实际定向阅读，不代表全文或全部证明审计；Science只给所列条件主张。完整保存版本、SHA和Scope在pretrain.json。

- [Valiant1984](https://courses.grainger.illinois.edu/ece544na/fa2014/valiant84.pdf)，PDFpp.1–6：固定k-CNF的TheoremA及删子句证明。原始正例EXAMPLES、部分赋值和不产生假阳性的保证须保留；PAC定义本身不授予Science。
- [VC1971](https://courses.grainger.illinois.edu/ece544na/fa2014/vapnik71.pdf)，PDFpp.1–11：增长函数控制iid集合频率的一致偏差，可测性是条件。多项式增长给分布统一充分保证；无限VC不能直接推成每个固定P都不一致。
- [Blumeretal1989](https://trhvidsten.com/docs/classics/Blumer-1989.pdf)，PDFpp.1–8、12–14、23–26：非平凡well-behaved类的有限VC刻画、可实现一致学习上界。统计样本复杂度与§3.1的proper高效一致假设搜索分开，Borel单个集合不足以替代整个类的可测性。
- [Ehrenfeuchtetal1989](https://www.cis.upenn.edu/~mkearns/papers/lower.pdf)，16页作者稿pp.1–8：Theorem1在小ε/δ范围构造最坏分布的Ω(d/ε)下界。允许随机学习器和不同假设表示；不是每个分布的困难性或计算时间下界。稿件页数与1989期刊15页分开。
- [Haussler1992](https://bpb-us-e1.wpmucdn.com/sites.ucsc.edu/dist/4/821/files/2017/07/Decision-theoretic-generalizations-PAC-model-neural-net-learning-applications.pdf)，PDFpp.1–5、9–30：有界损失、固定联合P的iid样本、Lemma1和有限类/覆盖数界。目标接近inf_H，经验优化质量另须保证；任意标签分布不等于允许无界损失。
- [Littlestone1988](https://www.cs.yale.edu/homes/aspnes/pinewiki/attachments/WinnowAlgorithm/winnow.pdf)，PDFpp.1–3、13–17：WINNOW1对固定单调析取目标的O(k log n)犯错界。可对抗顺序但正确标签；假阳性清零、假阴性乘法提升。没有用WINNOW2更新冒充原Algorithm3，也没把犯错总数直接写成终止假设泛化保证。
- [Kearns–Valiant1994](https://www.cis.upenn.edu/~mkearns/papers/cryptojacm.pdf)，PDFpp.1–8、11–15、19–21：RSA样本模拟及Theorems4–6条件归约。假设表示可更换但需多项式可求值；分布无关例子模型的密码困难性不等于无条件不可学或成员查询模型困难。
- [Haussleretal1991](https://www.cis.upenn.edu/~mkearns/papers/equivmodels.pdf)，扫描PDFpp.1–14 OCR、p.1及p.12目检：Lemma3.3单/双例子oracle等价。须保留regular₂、已知s、随机且必停机和分布无关；混合/条件采样改变分布，固定分布版本不能直接照搬。原刊I&C95(2):129–161。

Occam's Razor1987未取得原文，不计入本批；1991协议等价论文为有明确引用链的替代。后续可以补取原文，但不能把本轮当作Occam定理已核验。

## Gemini 1.0：模型设计与聚合协议

**Gemini — Knowledge。** [官方修订报告](https://deepmind.google/gemini/gemini_1_report.pdf)（90页，©2025；非已下载的2023v1）pp3–6 §2–4描述多模态联合预训练及32k上下文，pp6–8和73–74说明MMLU评测。Figure9中Ultra的ScoreEval、CoT@32、阈值路由CoT@32分别为83.96%、84.99%、90.04%；gpt-4-0613相应为84.21%、87.29%、87.29%。阈值逐模型在验证集选择，一致性较低时回退无CoT最大似然选择。此例要求将模型、采样数、阈值、聚合与比较版本共同纳入Scope；同采样数不等于同FLOPs，也不证明该聚合在任意任务上必然改善。Table2的5-shot 83.7%不与Figure9的ScoreEval混用。已目检两处表图，未审原始响应或独立复现实验；完整多模态评测与其余章节未审。
