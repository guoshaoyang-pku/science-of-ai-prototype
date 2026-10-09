# r012：相同 Rayleigh 商仍有不同的拟合预算

程序第 12 轮、D3 第 2 轮恢复 r004 的原问题，完成 **18/18 cell**。同谱配对的参数数、trace、有效秩、初始损失和目标 Rayleigh 商相同，达到初始损失 1% 的步数仍可不同：p=1 为 13 与 38，p=2 为 41 与 364。原 P1–P3 均通过。

## 合同与提交时序

本轮只回答一个问题：保持固定特征、谱、初始损失和目标 Rayleigh 商，只改变目标模态权重，1% 阈值预算是否改变。沿用 n=d=9、λᵢ=i⁻ᵖ、p=0/1/2、float64、零初始化、全批量 SGD η=.5/mom0/nodecay、1000 步。第三模态目标与首尾混合目标的 Rayleigh 商均为 λ₃，初始损失均为 .5。

r004 的条件、预测与 pinned 源码未改。本轮新增 [r012 恢复预注册](../studies/r012_matched_rayleigh_resume/preregistration.json)，成功提交 `7fefbda52c519e069d6af83babdacab4de428b20` 后核验提交中的合同与源码字节，再训练。[训练前记录](../studies/r012_matched_rayleigh_resume/executed/experiment_start.json) 记录当时为 0 cell；首 cell 的 hash 与保存 X/y 谱核验通过后，补余下 17 cell。成功 cell 未覆盖，计算累计 .089859 秒。

上一轮的提交失败记录保留。[第一次失败记录](../studies/r004_matched_rayleigh/executed/preregistration_commit_attempt.json) 保存 p=2 下限 ≥10 的原草稿；训练前解析审查将下限改为 ≥8，原合同随后由 supervisor 提交。本轮评估修订后的 ≥8，不能把原 ≥10 草稿称为实验失败预测。本方向没有 inbox.md。

## 保存结果与预测

| 谱 p | 第三模态 T₁% 范围 | 首尾目标 T₁% 范围 | 配对步数差范围 | 首尾/第三模态比范围 | 注册判据 |
|---|---:|---:|---:|---:|---|
| 0 | 4–4 | 4–4 | 0–0 | 1–1 | 两目标均 4 步，通过 |
| 1 | 13–13 | 38–38 | 25–25 | 2.923077–2.923077 | 全部 ≥2.5，通过 |
| 2 | 41–41 | 364–364 | 323–323 | 8.878049–8.878049 | 全部 ≥8，通过 |

每行含 3 个同坐标 seed 配对；共 6 个谱×目标条件、9 个配对。seed 仅检查坐标旋转下的数值稳健性，不是独立数据重复，不报告置信区间。全部阈值在 1000 步内达到，无删失。

P1 是已知谱递推的执行核验：保存损失与从 X/y 独立 eigh 复算曲线的最大绝对误差为 1.665335e−15，保存解析曲线误差为 8.881784e−16；9 配对描述量最大差为 4.440892e−16。各 cell 实测与预测整数步相同。P2 平谱负对照与 P3 衰减谱预算比均为 supported。P1 的公式核验不登记为新发现。

[r012 summary](../studies/r012_matched_rayleigh_resume/summary.json) 保存逐 cell 数字、配对范围与提交/hash 证据；[analysis.py](../studies/r012_matched_rayleigh_resume/analysis.py) 只复算 r004 保存结果并标明本轮执行。原 NPZ、metadata、receipt 留在 [r004 results](../studies/r004_matched_rayleigh/results)。首 cell 核验另存为 [first_cell_audit.json](../studies/r012_matched_rayleigh_resume/executed/first_cell_audit.json)。

执行器在特征矩阵构造处输出 divide/overflow/invalid 的矩阵乘法警告，原始输出保存在 [execution_log.json](../studies/r012_matched_rayleigh_resume/executed/execution_log.json)。根因未确定，保存数组均为有限值。[独立核验源码](../studies/r012_matched_rayleigh_resume/executed/verify_saved.py) 以标量算术复算保存结构、Gram、1001 步损失及终点残差；[核验记录](../studies/r012_matched_rayleigh_resume/executed/saved_evidence_verification.json) 为 18/18 通过，最大 Gram 误差 6.661338e−16，未发现保存结果受影响的证据。

## 结论边界与下一轮

测量支持一个有限结论：在注册的 p=1/2 固定谱与 SGD recipe 下，匹配参数数、trace、有效秩和目标 Rayleigh 商不足以唯一确定 T₁%。慢模态目标质量是可区分的竞争解释；完整目标加权谱可按已知递推预测本例。预算描述量同时指定目标、优化器和阈值。所有谱满秩，两个目标都可最终插值，不能据此声称永久表示容量不同。

原合同称初始梯度范数未固定，措辞不准确：本线性模型中 `||g₀||²=(||y||²/n)×Rayleigh`，匹配初始损失与 Rayleigh 商会自动匹配初始参数梯度范数，保存数据标量复算也通过。原预注册保留，不改预测和条件。目标参数范数和实际第一步损失下降仍未匹配。跨 p 的 trace 也不同，反例只取各 p 内配对。本轮未测试 learned features、width/depth/激活/LN、泛化、空间频率分辨率或封存 OOD。

复算时先用指定 Python 执行 `studies/r012_matched_rayleigh_resume/analysis.py`，再执行该 study 的 `executed/verify_saved.py`；后者补回独立核验与合同措辞纠正。两步均只读原成功 cell，写分析产物，不训练。

下一轮只检验一个更严对照：固定 p=2 谱与原 SGD recipe，构造同时匹配目标 Rayleigh 商和二阶谱矩的两个目标，使实际第一步损失下降也匹配，再比较 T₁%。须先解析审查权重可行性、写新数值预测与 pinned 源码并 commit；本轮成功结果不重跑。
