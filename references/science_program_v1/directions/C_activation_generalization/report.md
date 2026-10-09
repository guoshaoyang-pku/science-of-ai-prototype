# 小尺度学习通路与噪声风险

小尺度SiLU不是统一变慢：它的奇通路严格为z/2，偶通路≈z²/4；统一总特征RMS后，线性目标仍快，二次/交互通路却按scale²被压低。C02在事前封存的八维Uniform/Laplace交互目标上，仅恢复偶通路就将test MSE .9526→.00428、1.1365→.01499，两个分布四seeds全部改善。

![真实OOD干预](../../studies/C02_parity_ood/parity_ood.svg)

这条解释有明确边界。C03放开hidden学习后，小尺度SiLU部分逃出瓶颈，但初始固定核谱失准；较大scale=.3时SiLU learned test .00681胜过ReLU .1392，小scale=.03时仍输。因此旧KB的activation胜率应附条件，不宜替换成普遍ReLU或SiLU偏好。

噪声实验发现另一种条件依赖：train误差一直下降，但信号拟合收益与噪声拟合增长不同步。固定特征MSE的谱模型能分解期望test风险为signal bias与noise variance。C05对新六维Uniform、新目标、新noise、width96/train64条件在训练前保存整条风险预测，24recipes最佳时间17/24精确、24/24在factor2内；48/48风险检查在3MonteCarloSE内。

![事前完整风险预测](../../studies/C05_risk_ood/risk_ood_forecast.svg)

## 证据入口

| 问题 | 已完成实验 | 报告 |
|---|---|---|
| 纯幅度还是目标通路？ | 等RMS，linear/quadratic，LN/skip；216recipes | [C01](../../studies/C01_activation_scale/report.md) |
| 干预偶能量能外推吗？ | 新dim/distribution/interaction，192recipes | [C02](../../studies/C02_parity_ood/report.md) |
| hidden学习是否破坏固定核解释？ | frozen/learned、非对称输入；128recipes | [C03](../../studies/C03_hidden_boundary/report.md) |
| 何时出现U型？ | train/test、signal/noise分解；64recipes | [C04](../../studies/C04_noise_generalization/report.md) |
| 新条件能提前预测风险吗？ | 单独forecast seal后执行24recipes、16noise repeats | [C05](../../studies/C05_risk_ood/report.md) |

总计624recipes、1136head轨迹；其单位不是1136个独立任务。C01的LN严格尺度不变预测失败，C04的最优时间推迟只通过中位数判据；保留失败与单seed反例。没有新sol评测。交接和全部source/data/result核验见[handoff](handoff.md)与[verification](verification.json)。

## 已核验相关来源与贡献边界

Ramachandran、Zoph、Le的[Searching for Activation Functions](https://arxiv.org/abs/1710.05941)提出Swish=x·sigmoid(βx)，报告其在部分深网络上的经验优势；这里的β=1即SiLU。本文的Taylor及奇偶分解直接由该函数定义推出，不把分解本身当作新发现。

Rahaman等的[On the Spectral Bias of Neural Networks](https://arxiv.org/abs/1806.08734)讨论深ReLU的频率偏好；本研究测的是固定随机特征核的目标谱与奇偶通路，不能据此声称验证了该论文所有频率机制。固定特征风险公式属于线性动力学/bias-variance计算，尚未完整审查其相关工作；当前可声称的是具体干预、量化边界和保存于训练前的OOD预测。尚未形成可宣称新颖性的论文结论。
