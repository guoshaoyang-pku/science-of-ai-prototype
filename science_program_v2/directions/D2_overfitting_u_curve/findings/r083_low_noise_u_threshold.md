# r083：降噪后，风险回升未达到 .05 操作性阈值

仅检验一个问题：保持 n32、seed412 的 population 标签、原输入、W、特征及优化器条件，复用保存的 signal_bias 与 variance_unit，将 σ²=.125 减半到 .0625，终点风险减最低风险是否仍至少为 .05？完成 **1/1 新评价 cell、0 新训练**。差值为 .02215220201337556，落入事前区间 [.01,.04]，P1 supported；原操作性 U 型判据不再满足。最低点仍在内部，晚期风险仍回升，不能称 U 型形状完全消失。全部为 development。

## 先读与先提交

按序读取 GOAL、AGENTS、中心状态、task、全部 findings/report；inbox 不存在。报告已符合公式区及六节骨架，无需先重排。旧科学收尾 f61b6fe 的 56 个 git 对象与记录 hash 一致；预注册 ba83884 至少早于八个 cell 21.615900 秒。旧独立核验、76 项输入 pin、229 个历史文件和 33 个成功文件的 hash/mtime 均通过。

本轮对所有既有 study 共 281 个文件记录 hash/mtime，并核对 HEAD 字节；未运行失效草稿，未重跑或覆盖旧 60/8/8 个成功 cell。独立子任务只读核验旧证据；另一个子任务提供新独立核验源码，不作训练。所有写入均在本目录。

预注册及执行、分析、独立核验源码在 2026-10-07T12:13:53+08:00 一次提交为 86f82ac7bfeca40fee8005609798de736d4fdbf3。12:14:03 核验唯一提交绑定、四个冻结文件及 13 项来源 hash，12:14:17 才首次组合 .0625 风险，提交早于新评价 24.078096 秒。合同未改。

## 预测来源与配对结果

条件按旧 .125 差值仅 .051388874 的已知余量选择，属于 development。注册前未计算 .0625 的整条风险、端点或最低点。区间 [.01,.04] 只由旧差值和降噪的粗略判断给出；没有假设差值精确减半，也不是封存 OOD 或盲发现。仅注册差值区间与操作性判据失败；新最低点位置作描述。

| σ² | 首个全局最低点（步） | 最低风险 | 第 16384 步风险 | 终点减最低点 | 达到 .05 判据 |
|---:|---:|---:|---:|---:|---|
| .125（保存对照） | 313 | .1043385144583564 | .15572738849959528 | .051388874041238874 | 是 |
| .0625（新评价） | 461 | .07359224980542978 | .09574445181880534 | .02215220201337556 | 否 |

这是一个 recipe 配对、一个 seed，无跨 seed 统计区间。新最低点延后 1.472843450 倍；差值配对减少 .029236672027863314，最低风险减少 .030746264652926625，终点风险减少 .05998293668078994。风险单位为归一化标签平方，步数单位为更新步。

## 解释与边界

降噪同时降低最低风险和终点风险，并延后最低点。原 .05 阈值把旧条件计为 U 型、新条件计为不满足；新条件仍有内部最优点和正的风险回升。该例说明操作性幅度阈值的结论与是否存在回升须分别报告，不能把阈值未通过解释成没有过拟合形状。噪声项随 σ² 线性缩放，但风险最低点随 σ² 重选，因此终点减最低风险不能直接按噪声比例缩放。分解公式是已知理论，不登记新解析发现。

只改变指定条件中的噪声方差。没有识别输入、经验谱或目标投影的独立因果作用；没有测连续噪声临界值。oracle 使用 clean train/audit 标签，不是 train-only 早停。不外推其他 seed/n/width、learned hidden、Adam、小批量、CE、无限训练或 sealed OOD。旧 P2 refuted 40/44、P3 supported 57/60、共享标签归一化反例 201/62=3.241935、减半噪声反例 324/87=3.724138，以及原 NumPy bool 错误和失效轮记录均保留。

## 复算与交接

新评价只组合保存的 B(t)+.0625V(t)，保存所有整数步 0..16384，取首个全局最小点；耗时 .025457 秒，stderr 为空。分析从新 NPZ 复算，未调用旧执行器。独立从原数据重新构造 K/eigh，以 log1p/expm1 响应复算全部风险；信号、单位方差、风险最大误差分别为 8.770762e−15、7.094325e−14、1.283695e−14，差值误差为 6.189493e−15，最低点同为 461。另用 60 位 Decimal 对保存浮点数复算，差值与判据一致，最大 float64 组合舍入误差 5.030699e−17；这是数值核验，不是统计置信区间。

281 个旧 study 文件及三个新成功文件 hash/mtime 不变。恢复返回 0 新 cell、1 复用 cell、0 训练，成功结果及 summary 四个文件 hash/mtime 未改。中心状态仅更新 D2 的 last_round 与 next_question，round=83 和 rounds_done=5 保持；计数由 supervisor 管理。

下一小问题可仍用零训练：只改为预指定的 n32/seed413，将其保存 .125 曲线评价于 .0625，检验这一次 .05 判据跨 seed 是否同样失败。先核对本轮收尾、冻结新来源 hash、源码和数值预测，再组合新风险；不预先计算未注册条件，不扩为连续噪声阈值。

## 证据

- [预注册](../studies/r083_low_noise_u_threshold/preregistration.json)、[唯一提交与来源绑定](../studies/r083_low_noise_u_threshold/executed/execution_binding.json)、[历史只读审计](../studies/r083_low_noise_u_threshold/executed/prior_closeout_audit.json)与[独立旧证据审计](../studies/r083_low_noise_u_threshold/executed/independent_prior_closeout_audit.json)。
- [新评价 NPZ](../studies/r083_low_noise_u_threshold/results/n32_seed412_var0.0625.npz)、[metadata](../studies/r083_low_noise_u_threshold/results/n32_seed412_var0.0625.json)、[receipt](../studies/r083_low_noise_u_threshold/results/receipt.json)、[可复算 summary](../studies/r083_low_noise_u_threshold/summary.json)与[分析源码](../studies/r083_low_noise_u_threshold/analysis.py)。
- [独立核验](../studies/r083_low_noise_u_threshold/executed/independent_verification.json)、[独立源码](../studies/r083_low_noise_u_threshold/executed/independent_verify.py)、[执行记录](../studies/r083_low_noise_u_threshold/executed/evaluation_run.json)、[分析记录](../studies/r083_low_noise_u_threshold/executed/analysis_run.json)与[恢复核验](../studies/r083_low_noise_u_threshold/executed/resume_verification.json)。
