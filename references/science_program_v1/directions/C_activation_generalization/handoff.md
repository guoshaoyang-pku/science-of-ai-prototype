# C方向交接：激活通路与U型风险

五个已启动study全部完成、分析及图已保存。当前无实验进程，无模型API/solver调用；收到新目标后再决定后续研究。proposal.json是待协调者登记的短KB提案，未直接修改旧主线KB。

## 结果和开销

| Study | Recipe runs | 实际拟合轨迹 | 进程秒/训练秒 | 可读报告 |
|---|---:|---:|---:|---|
| C01 activation scale | 216 | 216 | 7.10/6.79 | [C01](../../studies/C01_activation_scale/report.md) |
| C02 parity OOD | 192 | 192 | 15.97/15.32 | [C02](../../studies/C02_parity_ood/report.md) |
| C03 hidden boundary | 128 | 128 | 12.06/11.88 | [C03](../../studies/C03_hidden_boundary/report.md) |
| C04 noise generalization | 64 | 192 | 14.53/14.07 | [C04](../../studies/C04_noise_generalization/report.md) |
| C05 risk OOD | 24 | 408 | 3.17/3.11 | [C05](../../studies/C05_risk_ood/report.md) |
| 合计 | 624 | 1136 | 52.83/51.17 | 无新增sol涨分 |

C04每recipe同时拟合total/signal/noise三个heads；C05每recipe一个clean head和16noise heads。轨迹不等于独立研究实例。C01首个recipe在第一次checkpoint调用约.032秒，表中7.10秒是后续恢复完整调用；所有时间仅本机轻量训练，不含agent编写与分析时间。

小尺度SiLU的慢通路取决于目标。在对称零bias固定特征下，odd=z/2，even≈z²/4；统一总RMS后，二次/交互目标的核能量被scale²压低，但线性目标不慢。C02无标签偶通路匹配在封存新维度/分布/目标上将MSE .9526→.00428、1.1365→.01499，均4/4seed改善。C03放开hidden后部分逃出瓶颈；scale .03/.3使SiLU learned的交互test从 .9247变 .00681，ReLU .1392。

C04测得train持续下降而期望clean-test风险U型，late噪声拟合超过signal收益；初始谱给出target/noise相关时间，统一U阈值不成立。C05训练前预测完整新条件风险，48/48预定检查在3MonteCarloSE内；其中5次重复，实际43个不同recipe×checkpoint。最佳doubling-grid检查点17/24精确、24/24在factor2内，5个最佳点处于预算末端。预测使用clean train/test标签，不能直接用于无test-label部署早停。固定谱是已知理论的局部计算模型，C03已经显示它不能覆盖learned-hidden全程。

## 失败预测与OOD时间线

C01 P1的LN严格尺度不变容差失败：长期curve差 .01585>1e−4；无LN及linear-skip误差2.22e−16，LN初始核能量近似不变也不能掩盖原判据失败。有限epsilon是可疑机制，但未做独立epsilon干预。C04 N4只通过事前中位数判据；一个small-SiLU seed最优step16，比ReLU32更早。其余原判据按报告保留。

时间均为UTC（北京时间+8）：C02配置在2026-10-05 18:09:05.850555封存，第一result18:09:13.641631。C05配置18:24:56.494544，最后数值预测18:24:57.199951，全部forecast hash seal18:24:57.200402，第一训练result18:25:08.345665。C05 forecast/train为独立进程；seal记observed_training_results=0。源码和数据contract检查把实际训练绑定事前预测。

C02/C05都是既定任务recipe的OOD实验；C05允许解析模型查看新数据及clean函数值，预测训练结局。它不是未知目标盲预测。观测结果后的这些条件成为development，下一次修改需要新的封存条件。尚无多层/分类/Adam的泛化、原benchmark反转或sol收益。

## 离线核验与恢复

在repo根目录使用已装NumPy/PyTorch/Matplotlib的Python：

~~~sh
python directions/C_activation_generalization/verify.py
python studies/C01_activation_scale/analyze.py
python studies/C02_parity_ood/analyze.py
python studies/C03_hidden_boundary/analyze.py
python studies/C04_noise_generalization/analyze.py
python studies/C05_risk_ood/analyze.py
~~~

verify.py已核验16执行源码、16data NPZ（逐数组hash/shape）、624result NPZ（hash/finite）、624source/data/cell contracts、C01实际checkpoint恢复和两次OOD封存。五个analyze.py从repo外执行得到byte-identical summaries，4.81秒，不训练/联网。verification.json保存指纹；所有图人工看过，无被裁文字。analysis可重绘图，不覆盖成功训练。

需要复用训练时运行各study/run_study.py；C05按forecast→train顺序。C01执行前要求state/current.json的first_chain_complete=true，这是早期并行授权gate。现有成功cell须匹配当前源码/data合同，复用后不重写；源码改变会拒绝复用，应开新study。C01 recovery_audit.json首cellhash和原mtime证明真实恢复复用1cell、新增215cell；移机后mtime可变，内容hash可核验。

目录内current.json的旧pid是执行历史，status显示terminal，不代表仍存活。run_study.py每次持有单worker文件锁；线程1，禁止并行覆写同一study。public paths均相对repo。初始KB/source reports仅作development动机，development_sources.json保留原report hash而无个人路径；不把旧报告当前版本当独立复现。

## 后续尚未处理的具体问题

原benchmark recipe的一般性尚未验证；主张先保持对称/固定特征范围。可学习hidden下需可测的动态特征/Jacobian坐标，非零bias、非对称分布、多层传播都能打破严格奇偶分离。论文需要核验相关工作与新贡献，不能把Taylor分解或bias/variance公式称为首次提出。用户的新目标到来后再安排实验；当前先由协调者开源、登记与交接。
