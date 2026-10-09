# 同 Rayleigh 目标的三数量级联合维数标度

本轮只问：固定特征 λᵢ=i⁻¹ 的 n=d 联合网格中，近一次预算幂律是否伴随 Rayleigh 重新充分。d=8、16、…、8192，覆盖 log₁₀(8192/8)=3.010300 个数量级；不同时扫描 p 或网络架构。方向此前仅有 task.md，没有既有 report、findings 或 inbox。

## 合同与执行

预注册、实际参数 SGD 源码和分析源码先在 3804aa6 提交，随后核验同提交字节、源码 hash、结果目录为空，才开始参数训练。训练前做过两次已知谱模态能量算术审查，已保存其整数参考；这不是封存预测或新理论，不计作训练证据。早期未执行 runner 原型也曾采用能量递推，预注册前改为真实逐参数更新。一次写源码的临时 shell heredoc 因字符串语法错误未执行，修正后只做 ast.parse，再提交合同。

实际特征为固定满秩的 signed permutation：X[i,perm[i]]=√(dλᵢ)·sign[i]。中间目标 w_(d/2)=1；首尾目标 w₁=1/(d−1)、w_d=(d−2)/(d−1)。同 d/seed 配对的 X、谱、m₁=2/d、L₀=.5、初始梯度范数平方=2/d、优化器一致；m₂ 与目标参数范数未固定。NumPy float64、CPU 单线程、全批量 half-MSE SGD η=.5、mom0/nodecay，θ₀=0。

22 条件×3 坐标 seed=66 cell、33 配对。seed 只检查置换/符号坐标，不算独立数据，不做置信区间。每 cell 到首次相对损失≤.01 后多执行一步；5d 上限内全部达到阈值，没有删失。首 cell 核验通过后完成其余 65 个，累计实验计算 3.607685292 秒。成功 cell 再次执行时全部跳过，133 个 results 文件 hash/mtime 均未改变。

## 预测与结果

| 判据 | 预注册预测 | 保存结果 | 状态 |
|---|---|---|---|
| P1：已知递推审计 | 实际曲线误差≤1e−10、完整谱整数预算一致、参数与更新可复核 | 66/66 通过；曲线最大误差 4.385381e−15、闭式参数最大误差 8.668622e−13 | supported；不作新发现登记 |
| P2：全网格幂律 | 每目标各 seed β∈[.98,1.04]、最大拟合误差≤6% | 中间 β=1.003160561、c=2.249734987、误差 .868262%；首尾 β=1.005094515、c=4.438350206、误差 2.528432% | supported |
| P3：标量反例 | 首尾 Rayleigh 误差绝对值≥35%；中间准确；d≥256 配对比∈[1.95,2.05] | 首尾低估 48.571429%–50%；中间 33/33 准确；18 大维数配对比 1.998302–2.000000 | supported |

实际中间/首尾预算依 d 为 18/35、36/72、73/146、147/293、294/588、589/1177、1178/2356、2357/4714、4715/9430、9431/18861、18862/37724；三坐标 seed 的整数预算全部一致。同 Rayleigh 配对差为 0，初始损失最大配对差 1.665335e−16。单指数近似使用实测预算作误差分母，不能把首尾/预测的倍数写成相对误差。

## 解释与边界

增大 n=d 没有使两个同 Rayleigh 目标的预算合并。首尾目标的慢模态能量趋近 1，中间目标的模态 λ=2/d；已知递推给出渐近系数 log(100) 和 log(100)/2，符合实测约两倍比值。幂律稳定和标量充分是不同性质。首尾在最小相邻区间的指数 1.040642 稍高于 1.04，全网格判据仍通过；保留这一有限网格偏差。

这里 d 是人为固定特征的参数数，n 与其联动，谱尾、trace 和目标索引也一起延伸。目标参数范数平方为 d/2 与 d−1，未匹配；不能把预算差独立归因某一个中介。无随机特征集中、真实网络 hidden width、核漂移、lazy↔rich 边界、泛化或 sealed OOD。训练前解析已检查本次网格，不宣称网格外前瞻。

下一轮建议只固定 n=8192，保持 d 网格与同目标谱规则，前 d 个样本承载按 √n 缩放的特征/目标，其余样本为零；先写明确配对误差预测和 pinned 源码并 commit，再验证预算是否与本次一致。该控制只拆开 n=d 联动，不等于真实网络宽度机制。

## 证据

- [预注册](../studies/r027_dimension_rayleigh/preregistration.json)、[解析审查](../studies/r027_dimension_rayleigh/executed/pre_execution_analytic_review.json)、[训练前时序](../studies/r027_dimension_rayleigh/executed/experiment_start.json)。
- [执行源码](../studies/r027_dimension_rayleigh/executed/run.py)、[分析](../studies/r027_dimension_rayleigh/analysis.py)、[汇总](../studies/r027_dimension_rayleigh/summary.json)、[逐 cell receipt](../studies/r027_dimension_rayleigh/results/receipt.json)。
- [首 cell 核验](../studies/r027_dimension_rayleigh/executed/first_cell_verification.json)、[恢复审计](../studies/r027_dimension_rayleigh/executed/resume_audit.json)、[独立 sample-space 复算](../studies/r027_dimension_rayleigh/executed/independent_verification.json)。
- [中心更新审计](../studies/r027_dimension_rayleigh/executed/central_update_audit.json)确认只改 D11 last_round/next_question；[方向报告](../report.md)按要求以公式、符号定义与实测边界开头。
