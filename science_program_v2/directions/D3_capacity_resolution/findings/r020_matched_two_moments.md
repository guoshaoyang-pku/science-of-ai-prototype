# r020：匹配前两阶谱矩后，1% 拟合预算仍分离

程序第 20 轮、D3 第 3 轮完成 **6/6 cell、3 个同 seed 配对**。两目标的一阶谱矩、二阶谱矩、初始损失、初始梯度范数和实际第一步损失下降匹配，1% 阈值仍为 41 与 319 步。步数差为 278，配对比为 7.780487804878049，P1–P3 均通过。全部条件为 development。

## 权重可行性与冻结合同

只检验一个小问题：同固定谱和优化器下，匹配目标 Rayleigh 商 m₁ 与二阶谱矩 m₂，是否仍允许不同的 1% 阈值预算。先用四个可用模态 1、2、3、9 的三矩 Vandermonde 零空间向量正负部分构造非负权重，不运行训练、不扫描训练结果。A 的权重为 w₁=1/55、w₃=54/55；B 为 w₂=2048/4235、w₉=2187/4235，其余为 0。精确分数审查确认 Σw=1、m₁=7/55、m₂=1/33。旧单模态共同值 m₁=1/9、m₂=1/81 会强制谱方差为零，不能构造非平凡配对，因此本轮采用新的共同矩值。

条件固定为 n=d=9、λᵢ=i⁻²、float64、零初始化、全批量 SGD η=.5/mom0/nodecay、1000 步、L₀=.5、ε=.01；同 seed 固定同 X。seed0/1/2 只旋转参数坐标，不是独立数据重复，不报告置信区间。共两个目标条件、一个配对设计。

已知谱递推预测 A=41、B=319 步，注册 P2 为全部配对比≥7，P3 为各 seed 精确整数端点 41/319。P1 检查谱曲线、保存参数/损失及 SGD 更新误差≤1e−10，所有配对匹配量差≤1e−12且 X 完全相同。解析公式和可行性计算不是新测量。

[预注册](../studies/r020_matched_two_moments/preregistration.json)、[固定执行源码](../studies/r020_matched_two_moments/executed/run.py)、[固定分析源码](../studies/r020_matched_two_moments/analysis.py) 与[可行性审查](../studies/r020_matched_two_moments/executed/feasibility.json) 先提交为 `f6fc2a99f6ea91a0152eeca490c89c75d2c9cc21`。[训练前记录](../studies/r020_matched_two_moments/executed/experiment_start.json) 记录 0 新 cell，逐字节核验该提交内全部合同和源码后才运行。首 cell 实测 41 步、谱曲线误差 2.775558e−16，通过[首 cell 核验](../studies/r020_matched_two_moments/executed/first_cell_audit.json) 后补余 5 cell。

## 保存结果与数值核验

| 量 | A：模态 1/3 | B：模态 2/9 | 配对差或误差 |
|---|---:|---:|---:|
| m₁ | 7/55 ≈ .1272727273 | 同左 | 最大 3.053113e−16 |
| m₂ | 1/33 ≈ .0303030303 | 同左 | ≤1.179612e−16 |
| 实际 L₀−L₁ | .05984848484848487 | .05984848484848482–.05984848484848487 | 最大 5.551115e−17 |
| T₁%（3 seed 范围） | 41–41 | 319–319 | 278–278 步；比 7.780488–7.780488 |
| 目标参数范数平方（未匹配） | 487/55 ≈ 8.854545 | 2407/55 ≈ 43.763636 | 34.909091 |

[summary](../studies/r020_matched_two_moments/summary.json) 从保存 NPZ 复算各整数端点、同 seed 比/差、eigh 谱及权重。独立谱曲线最大误差 8.881784e−16；注册谱曲线最大误差 9.436896e−16；每步参数更新最大误差 4.435471e−16；保存参数复算 loss 与记录一致。所有数组有限、无删失，参数数/trace/有效秩/L₀/m₁/m₂/初始梯度范数平方/L₁/第一步下降的最大配对差为 3.053113e−16。累计实验计算 .146258333 秒。

[旧证据核验](../studies/r020_matched_two_moments/executed/previous_evidence_audit.json) 确认原 18 个成功 cell 的 metadata/NPZ hash 与 r012 收尾 `03525ec57ecc725d91dde031c0f34ec60c7093ea` 字节一致；旧 study/findings 全部只读。[恢复核验](../studies/r020_matched_two_moments/executed/resume_audit.json) 先检查所有成功 cell 的 hash，再调用执行器，新增 0 cell，所有结果及 receipt hash 完全不变。执行使用指定 Python，本机 CPU，无网络、GPU、评测、常驻进程。

[独立标量核验](../studies/r020_matched_two_moments/executed/independent_verification.json) 用 math.fsum 逐项重算保存数组，6/6通过，损失误差≤8.326673e−17、SGD更新误差≤5.031783e−16、Gram误差≤6.661339e−16。该附加审查只读取训练结果，不改预注册分析或参数。[中心更新审计](../studies/r020_matched_two_moments/executed/central_update_audit.json) 确认只改本方向 last_round 和 next_question，round 与 rounds_done 均保持。

## 科学解释与边界

本受控对照支持：相同固定谱、L₀、目标前两阶谱矩、初始梯度尺度和第一步实际损失下降，仍不足以唯一确定 1% 阈值预算。完整目标加权谱可按已知递推预测该差异。B 的慢模态 λ₉ 能量为 .5164108619，A 不含该模态；两目标的高阶矩和目标参数范数也不同。本轮没有独立区分慢模态质量、高阶矩及目标参数范数这些竞争解释。

全部谱满秩，两目标最终都可插值。结论只适用于注册权重、η、阈值和固定特征，不能称为永久表示容量差；没有测试 learned features、架构轴、泛化、空间分辨率或封存 OOD。旧矩阵乘法警告与未定根因保留；新源码用逐元素构造和 einsum，本轮未出现该警告，不声称已找到旧警告根因。

下一轮建议只加一个控制：保持本轮谱、维度、η、L₀与 ε，采用至少五个可用模态，先解析审查两个非负目标同时匹配 m₁、m₂ 和 ∑wᵢ/λᵢ（目标参数范数平方）的可行性，再写新数值预测与 pinned 源码并 commit，匹配提交后才训练。不覆盖或重跑本轮 6 个成功 cell。
