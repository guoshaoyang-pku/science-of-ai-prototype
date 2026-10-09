# r015：固定噪声 draw 扩大真实网络相对切线的 test 差距

第15轮（D7第2轮）执行第7轮留下的唯一小问题。24/24配对cell、48条优化轨迹完成，累计训练4.859595秒。P1通过4/4函数×宽度单元，12/12初始化配对的D为正。条件全部为development；没有封存OOD。原r007未训练与两次提交失败记录保留，不把本轮执行改记为第7轮完成。

## 条件、预测与提交时序

d5 standard Gaussian，train32/test256，mixed_sine与radial，含bias的两层hidden SiLU，width8/32，初始化11/29/47，data-seed7331、noise-seed733107。clean目标按train均值与标准差归一化；只切换训练标签sigma0/1，加噪后不再归一化。test始终干净。真实网络和完整初始Jacobian切线共享初始输出，full-batch SGD η=.05/mom0/nodecay，float64，512步。

定义Hσ=第512步干净test半MSE(真实)−半MSE(切线)，逐初始化配对计算D=H1−H0。原P1要求至少3/4函数×宽度单元mean(D)≥.02且各至少2/3 seed D>0；本轮没有修改阈值或排除seed。三个seed仅检查初始化稳健性，不算三种独立recipe。

原合同与两份pinned源码逐字匹配历史提交20b241232e9cc5c45df76aba03ad894b5f79e467。本轮恢复合同与额外分析入口先提交为ead3790fe2e1ced36fe0121bb1e516c67f612c22（2026-10-06T20:39:20+08:00），20:39:36核验同提交字节和hash，20:39:48启动首cell。首cell分析通过之后才续跑23cell；首cell的JSON与NPZ hash未变。无须修改权限，也没有改git配置。

## P1结果与配对区间

所有loss均为半MSE。区间为三个同初始化配对的95% Student-t区间（df=2），只描述固定数据和固定噪声向量下的初始化差异；不是跨数据或噪声的置信结论。

| 函数×宽度 | mean(H0) | mean(H1) | mean(D) [95% seed区间] | D>0 | P1单元判据 |
|---|---:|---:|---:|---:|---|
| mixed_sine×8 | .116602 | .674110 | .557508 [−.228023, 1.343039] | 3/3 | 通过 |
| mixed_sine×32 | .052857 | .853843 | .800986 [.498462, 1.103510] | 3/3 | 通过 |
| radial×8 | .016454 | .318227 | .301773 [−.175092, .778637] | 3/3 | 通过 |
| radial×32 | −.231632 | .055460 | .287093 [.106738, .467447] | 3/3 | 通过 |

两个width8单元的区间跨零，原成功判据不要求区间排除零。各seed原值、H0/H1区间和两模型加噪test变化均在[summary](../studies/r015_noise_interaction_resume/summary.json)；实际结果位于r007 study的results/。

## 相对差距与继续训练分别测量

以下是同一模型第512步减第128步的test半MSE；正值表示晚期test上升。对应train变化和配对区间见summary的late_units。

| 函数×宽度 | sigma | 真实test变化均值 | 真实test上升seed | 切线test变化均值 | 切线test上升seed |
|---|---:|---:|---:|---:|---:|
| mixed_sine×8 | 0 | .098034 | 3/3 | −.020969 | 0/3 |
| mixed_sine×8 | 1 | .580723 | 3/3 | −.017528 | 1/3 |
| mixed_sine×32 | 0 | .032336 | 2/3 | .036104 | 3/3 |
| mixed_sine×32 | 1 | .780441 | 3/3 | .057090 | 3/3 |
| radial×8 | 0 | .012496 | 1/3 | .004285 | 1/3 |
| radial×8 | 1 | .184392 | 3/3 | .055639 | 3/3 |
| radial×32 | 0 | −.274327 | 0/3 | −.043800 | 0/3 |
| radial×32 | 1 | .017206 | 1/3 | .128544 | 3/3 |

两个具体分歧：mixed_sine×32 seed11 clean的终点H=.020497，但真实test变化=−.001370；radial×32 seed29 noisy的H=.012378，但真实test变化=−.029205。这些是描述诊断，没有另设预测。终点相对切线劣势不能代替同模型晚期test上升。

## 验证、解释与交接

原analysis核验了24cell的合同/输入/NPZ/保存commit、sigma配对、损失和完整切线train递推。恢复analysis从保存参数重建真实网络初始及终点输出，最大绝对误差0；以einsum复算切线所有train/test检查点，最大误差分别6.89e−15/6.33e−15。全部保存数组finite。原analysis的NumPy matmul仍发出divide/overflow/invalid警告；原文保存在executed/original_analysis_log.json，根因未定，原pinned源码不改。独立复算证据见[核验文件](../studies/r015_noise_interaction_resume/executed/saved_evidence_verification.json)。

本轮测到的是该固定噪声干预对两种学习路径的test风险影响不同。真实网络对带噪train标签的终点loss更低，却不保证更接近clean train目标。clean的相对损害仍在，噪声不是必要原因；噪声拟合、有限样本下目标方向变化、隐式偏置与学习速度差仍不能按比例拆分。仅一个data/noise draw，不外推新数据、噪声强度、optimizer、LN或深度，也不能把D>0写成噪声后所有seed的H1>0（radial×32 seed11仍H1<0）。

下一轮只问一个新development小问题：保持同d5/train32/test256/函数/宽度/初始化/SGD合同，换一个事先固定的新noise seed但保留data seed与sigma0/1，检验本轮4/4放大是否仍至少3/4单元mean(D)≥.02且各至少2/3 seed D>0。先冻结新噪声seed、数值预测和源码并成功commit，再训练；不要重跑或覆盖本轮24cell。
