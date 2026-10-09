# r087：epsilon=.05 保存曲线端点评价

## 问题与预注册

本轮只评价 r073 的 66 条 trace=1、固定特征 loss 曲线，把相对损失阈值设为 epsilon=.05。预注册预测 middle 的 `T/(d h_d)` 对 `log(20)/2=1.4978661367769954` 的全网格误差不超过 3%，endpoints 对 `log(20)=2.995732273553991` 不超过 5%，且 d>=128 时两者均不超过 .3%。评价属于 development，0 新训练；不连续外推 epsilon，不运行 r043 失效草稿，也不重跑 r073/r075。预注册、评价源码、分析源码和输入清单在读取曲线前提交。

## 结果

只读 runner 从每个保存 NPZ 的 loss 数组取首次 `loss/loss[0] <= .05` 的整数步数，66/66 cell 均命中。

| 目标 | 参考系数 | 全网格最大绝对相对误差 | d>=128 最大绝对相对误差 | 判定 |
|---|---:|---:|---:|---|
| middle | `log(20)/2` | 1.743710% | .065304% | supported |
| endpoints | `log(20)` | 4.814219% | .257302% | supported |

独立核验重新扫描 134 个 r073 输入文件，hash 与 mtime 全部匹配；66 个端点首次命中、22 个 d×target 的三 seed 一致性检查全部通过。r073/r075 结果未被覆盖。

## 判据与边界

P1（66/66 保存曲线可读且无新训练）supported。P2（全网格 3%/5%，d>=128 .3%）supported。结果仅适用于 n=8192、非零样本数 d、`lambda_i=1/(i h_d)`、旧目标与坐标 seed、float64 零初始化全批量 half-MSE SGD eta=.5/mom0/nodecay 的固定特征条件。三个 seed 只改变坐标置换/符号，不是独立数据重复；epsilon=.05 是一个离散端点，不能由本轮推出连续 epsilon 规律，也不能外推 learned width、lazy/rich、独立样本或 OOD。

## 证据

- 预注册与冻结源码：[preregistration](../studies/r087_epsilon05/preregistration.json)、[run.py](../studies/r087_epsilon05/executed/run.py)、[analysis.py](../studies/r087_epsilon05/analysis.py)，提交 `e6369c5`。
- [输入清单](../studies/r087_epsilon05/executed/input_manifest.json)，固定 134 个文件，manifest SHA256=`8217a2a968372e594c075a4399a4e7c2b045ac09b9d02b8c58be14fea52210a4`。
- [汇总](../studies/r087_epsilon05/summary.json)、[独立核验](../studies/r087_epsilon05/executed/independent_verification.json)、[receipt](../studies/r087_epsilon05/results/receipt.json)。
