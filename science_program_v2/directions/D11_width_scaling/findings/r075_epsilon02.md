# ε=.02 端点评价（零新训练）

本轮只问：复用 trace=1 条件下保存的 66 条 ε=.01 损失曲线，把评价阈值改为 ε=.02 后，
`T/(d h_d)` 是否仍接近 `log(50)/2`（middle）与 `log(50)`（endpoints）。预注册提交
`b36d7d1` 早于任何评价；输入为 r073 的 133 个结果文件和 summary，评价源码在同一提交冻结。

## 结果

只读 runner 从每条 loss 数组取首次 `loss/loss[0]≤.02` 的整数步数。66/66 cell 完成，
无删失、无参数更新。独立核验重新读取每个 NPZ，66/66 首次命中与 JSON 一致；r073 输入
hash 全匹配，462 个既有 D11 文件的 hash/mtime 未变。

| 目标 | 预测阈值（全网格） | 实测最大绝对系数误差 | d≥128 最大绝对误差 | 判定 |
|---|---:|---:|---:|---|
| middle，`log(50)/2` | 3% | 1.244504% | 0.022762% | supported |
| endpoints，`log(50)` | 5% | 4.771486% | 0.205409% | supported |

误差按 `|T/(d h_d)/C−1|` 计算，`C` 分别为 `log(50)/2=1.956011502714073` 和
`log(50)=3.912023005428146`。三坐标 seed 的步数逐维一致；这只检查坐标置换稳健性。

## 边界

结果仅适用于固定特征、n=8192、非零样本数 d、`λ_i=1/(i h_d)`、旧目标与坐标 seed、
float64 全批量 half-MSE SGD η=.5/mom0/nodecay 的 development 条件。没有新增训练，
不外推独立数据、learned width、lazy↔rich、连续 ε 或 OOD。

## 证据

- [预注册](../studies/r075_epsilon02/preregistration.json) 与 [冻结源码](../studies/r075_epsilon02/executed/run.py)（提交 `b36d7d1`）。
- [评价汇总](../studies/r075_epsilon02/summary.json)、[独立核验](../studies/r075_epsilon02/executed/independent_verification.json)、[结果收据](../studies/r075_epsilon02/results/receipt.json)。
- [输入清单](../studies/r075_epsilon02/executed/source_manifest.json) 与 [既有文件基线](../studies/r075_epsilon02/executed/baseline_manifest.json)。
