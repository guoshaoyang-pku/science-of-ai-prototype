# r029：降低学习率后，最低点比值是否进入两倍范围

固定 D2 的 n=64、ReLU width128、零初始化线性头、float64、MSE、无动量/衰减、σ²=.25/.5/1 和 seed411–414。只将 η=.3 降为 .1，B=8/32 保留相同 batch 随机流；窗口由 768 改为 2304 步以保持 ηT=230.4。预注册及执行/分析/核验源码先提交 5e0f67c，然后训练。新增 48/48 cell：24 个小批量 cell（每个 8 条轨迹）与两个 η 各 12 个全批量对照；训练累计 10.779505 秒。旧 24 个小批量 cell 只读复用。

先审查发现旧比较口径不同：小批量分子固定一次标签噪声 ε，全批量分母则是 signal_bias+σ²·variance_unit，对标签噪声取期望。新合同在训练前为两个分母分别注册数值预测。旧数字和失败记录保留；旧比值不再解释为只有 batch 的效应。

P1（旧期望风险分母）被推翻，20/24 cell 落入 [.5,2]。B=32 比值 .272277–2.525483，B=8 为 .519802–4.759777。P2（同固定标签的全批量分母）也被推翻，22/24 通过。匹配口径结果如下，seed 是配对稳定性重复，不是独立 recipe：

| B | σ² | η=.1 比值范围 | η=.3 比值范围 | η=.1 通过数 | 偏离 1 减小数 |
|---|---|---|---|---|---|
| 32 | .25 | .960165–1.149686 | .905660–1.202479 | 4/4 | 2/4 |
| 32 | .5 | .932458–1.433803 | 1.101695–1.485714 | 4/4 | 3/4 |
| 32 | 1 | .624183–1.122449 | .480392–1.976190 | 4/4 | 4/4 |
| 8 | .25 | .870879–1.206289 | .848797–1.211321 | 4/4 | 4/4 |
| 8 | .5 | .845070–1.375972 | .923729–1.794805 | 4/4 | 2/4 |
| 8 | 1 | 1.206349–2.784314 | 1.225490–3.750000 | 2/4 | 3/4 |

两个匹配口径反例均在 B=8、σ²=1：seed411 为 105/49=2.142857，seed412 为 852/306=2.784314。它们的条件 batch-bootstrap 95% 描述区间分别为 [.734694,5.653061] 和 [.346405,3.062092]，都跨过 2。因此失败的是注册的八轨迹均值曲线判据；没有证明精确 batch 期望的最低点必然越界。

降低 η 后，18/24 配对 cell 的 |比值−1| 点估计减小，每个 batch 各 9/12。但 24 个配对变化的 bootstrap 区间全部跨零；同标签两倍内通过总数两档 η 都是 22/24，B=32 为 11→12，B=8 为 11→10。不能写成全部 seed 单调改善，也不能独立归因于离散稳定性边界。P3 支持：48 个新 cell 风险有限；小批量均值最低点及全批量两种风险最低点均在窗口内部。

## 核验与边界

预注册 pin 与源码一致，48 个 cell 的 receipt、NPZ、数据与提交合同核验通过。130 个旧文件 hash/mtime 不变。恢复调用只核验/跳过，不覆盖成功 cell。最终 head 的 audit 风险与独立 einsum 复算误差 ≤1.276756e−15，24 个全批量期望风险与独立谱复算误差 ≤3.108624e−15；一条小批量前 16 步的独立 einsum 重放误差 ≤1.110223e−16，保存比值与 bootstrap 区间复算一致。

运行和原核验中出现 matmul divide/overflow/invalid 警告，实际保存数组全部有限且上述独立复算通过；根因未定，原日志和 pinned 源码不改。旧核验源码及其预注册 pin 曾在收尾后改写，而旧训练 metadata 保留原 pin；此历史时序限制已记录，不能声称旧核验源码从训练前起未变。

仅 development，6 个 batch×variance recipe、4 数据/特征 seed、每小批量 cell 8 个 batch 抽样重复；没有新标签噪声抽样、没有 sealed OOD。没有检验连续 η 阈值、其他 n、learned features、动量、Adam、无限训练或 train-only 早停。窗口匹配 ηT，但步数和样本曝光量不同；所有区间仅描述固定数据与 ε 条件下 batch Monte Carlo，不是跨 recipe 的置信区间。

## 证据

- [冻结合同](../studies/r029_lower_eta_paired/preregistration.json)、[执行源码](../studies/r029_lower_eta_paired/executed/run.py)、[分析](../studies/r029_lower_eta_paired/analysis.py)、[汇总](../studies/r029_lower_eta_paired/summary.json)。
- [原核验](../studies/r029_lower_eta_paired/executed/verification.json)、[独立核验](../studies/r029_lower_eta_paired/executed/independent_verification.json)、[旧文件与中心状态快照](../studies/r029_lower_eta_paired/executed/input_audit.json)、[恢复记录](../studies/r029_lower_eta_paired/executed/resume_verification.json)。
- 预注册提交 5e0f67c；科学收尾以 [最终提交核验](../studies/r029_lower_eta_paired/executed/final_commit_verification.json) 为准。
