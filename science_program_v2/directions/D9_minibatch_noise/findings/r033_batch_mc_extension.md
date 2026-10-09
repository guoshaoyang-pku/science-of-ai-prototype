# r033：两个最低点反例在补足抽样后是否保留

固定 D2 n=64、d=4、ReLU width128、float64 MSE、零初始化含 bias 的线性头、无动量/无衰减 SGD，η=.1、B=8、σ²=1、seed411/412、T=2304。只增加 batch 抽样轨迹：旧 r=0–7 的风险、最终 head 和 batch seed 原样保留，各追加 r=8–63 的 56 条轨迹；不重抽数据或 ε，不延长窗口。同标签全批量最低点仍为 49/306 步。全部 development。

预注册、执行/分析/核验源码和输入快照先提交 b046352，再训练两个 cell。新增 112 条轨迹、保留 16 条，总计 128 条；复用两个 full 控制，没有新 full 训练。只有 1 个 recipe、2 个配对数据 seed，轨迹不算独立 recipe。训练累计 5.121093 秒。

| seed | 旧 M=8 的 t*/t*_full | M=64 的 t*/t*_full | 旧 95% bootstrap 区间 | 新 95% bootstrap 区间 | 新/旧区间宽度 |
|---|---|---|---|---|---|
| 411 | 105/49=2.142857 | 57/49=1.163265 | [.734694,5.653061] | [.938776,1.489796] | .112033 |
| 412 | 852/306=2.784314 | 235/306=.767974 | [.346405,3.062092] | [.660131,1.415033] | .277978 |

P1 支持 2/2：两个点估计均进入 [.5,2]。P2 支持 2/2：两个区间宽度都不超过旧值的 50%，并分别落入注册包络 [.5,3]、[.5,2.5]。P3 支持：风险有限、均值最低点位于 (0,2304)、旧前八条逐元素一致。新两个区间都不跨 2；2000 次 bootstrap 比值超过 2 的比例分别为 .0015/0，只是固定数据和 ε 下重抽轨迹的描述比例，不是精确期望越界概率或假设检验 p 值。

八轨迹反例未在本次补足后的点估计中保留，支持 batch Monte Carlo 误差是旧反例的候选解释。不能由此证明精确 batch 期望必在两倍内，也不能把补足后两个点与其他 22 个仍为八轨迹的 cell 拼成“统一 24/24 通过”。旧 P1/P2 失败与原始八轨迹记录仍有效，M=8 是其判据的一部分。M=8 是 M=64 的前缀，二者相关；本轮不做独立样本差异显著性检验。

旧期望标签噪声分母仅作描述：新 seed411/412 比值为 57/202=.282178、235/179=1.312849。口径混杂仍保留；两个学习率、固定 ε 与有限窗口不足以识别连续 η 临界、稳定性机制、其他 n/优化器、learned features、无限训练或 train-only 早停。

## 核验与历史限制

241 个旧文件 hash/mtime 未变；旧科学收尾 ea1a3e8 中核心合同、源码、汇总与本轮输入字节一致。旧 report 链接的 final_commit_verification.json 不存在，本轮保存实际审计，并将方向报告证据链接改为该审计；不改旧研究文件。新 cell receipt/NPZ/数据/hash/冻结提交匹配；恢复只核验并跳过两个 cell，全部结果 hash/mtime 未变。独立 einsum 最终风险与首 16 步重放误差均 ≤1.110223e−16，bootstrap 加权均值独立复算与保存 t*、区间完全一致。

新训练再次出现 matmul divide/overflow/invalid 警告；保存数组均有限且上述独立复算通过，警告根因未定。保留原日志和 pinned 源码。

下一小问题：固定这两个数据/ε/优化器/窗口及同标签分母，预注册一个独立 64 条 batch 流块 r=64–127，检验 [.5,2] 点估计和区间上界 <2 是否保持；旧 8/64 条不覆盖。

## 证据

- [预注册](../studies/r033_batch_mc_extension/preregistration.json)、[执行源码](../studies/r033_batch_mc_extension/executed/run.py)、[分析源码](../studies/r033_batch_mc_extension/analysis.py)、[汇总](../studies/r033_batch_mc_extension/summary.json)。
- [旧输入/提交审计](../studies/r033_batch_mc_extension/executed/input_audit.json)、[独立核验](../studies/r033_batch_mc_extension/executed/verification.json)、[恢复核验](../studies/r033_batch_mc_extension/executed/resume_verification.json)、[训练日志](../studies/r033_batch_mc_extension/executed/run.log)。
- 冻结 b046352；科学收尾以 [最终提交核验](../studies/r033_batch_mc_extension/executed/final_commit_verification.json) 为准。
