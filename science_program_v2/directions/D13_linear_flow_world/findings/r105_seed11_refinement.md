# r105：seed11 固定时钟 A 的 5% 步长检验

## 结论

P1 支持 1/1。固定 seed11 的数据、目标和所有初始层矩阵，仅 eta=.00125→.0003125，reference tau_A 从 0.023834804503907975 增至 0.024400284818224063。按旧值计变化 2.3724982272179767%，低于事前 5%。新增 1 个训练 cell 与 1 个 reference A 拟合，没有重跑或覆盖任何旧成功结果。

这一结果增加了一个已选初始化的两档离散步长证据。与 seed22 的 4.329963539479303% 分别通过同一 5% 判据，但没有完成三 seed 的新步长检验。旧跨 seed 10% 预测被推翻、CV=.276924>.25 与 A/B 偏移均保留。尚无从架构与初始化预测相变时间或有效深度的标定式。

## 小问题、预测与冻结

本轮只问预指定 seed11 的 reference A 读数是否在旧值 5% 内。点预测 0.024363632886567045 来自该 seed 已见 eta=.005→.00125 增量的四分之一。事前判据区间为 [0.022643064278712575, 0.025026544729103376]；它是操作范围，不是置信区间。已知旧两档与 seed22 结果，属于非盲 development 外推。新读数比点预测高 3.665193165701755e-05，没有把事后结果改写为新预测。

唯一冻结提交 a559d79681e7ba3ebc7f8d0907eedf133b75ccf5 包含合同、训练/分析/独立核验三源码和全部 103 个历史文件 hash/mtime，早于首训练 10.66754698753357 秒。冻结前只读检查旧条件和旧数组，未运行新训练或求新 tau。已有报告符合八节骨架；inbox 不存在，无处理标记。git add -A 纳入已有 supervisor 日志、state、prompt/output，未科学使用其他方向。

历史核查确认 r001 的 12 个训练、r101 的 6 个训练与 9 个后续 fit、r102 的 1 个训练与 1 个 fit、r103 的 4 个 fit、r104 的 2 个 fit全部保留。seed11 新 eta 未与这些条件重叠。103 个历史文件包含已有的 ignored pyc；本轮全部 Python 用 -B，未覆盖它。

## 条件与结果

无 skip、L4、hidden32、n128d4 固定高斯输入、4×4 正交目标、sigma=.8、seed11 原初始化、float64 CPU 全批量 GD、mom0/nodecay。input/target seed 为 20261008/20261007。新数据与初始层矩阵逐位等于旧 seed11 NPZ；数据/矩阵字节 hash 事前固定。沿用 g=2err/n，实际优化 J=4×MSE。仅改学习率及保持 tau=60 所需步数，新 T=192000。

reference 时钟 s=eta×step/.02；A 对全部 192001 点等权拟合，自由上下平台、原单起点、bounds、clip 和三项 1e-12 精度不变。生成器、梯度更新、fit_a/fit_b 函数与旧源码逐字一致。B 只读公共 tau=.02 网格的 11 点平滑峰，不新增 legacy A 拟合或另一条预测。

A success=True、归一化 RMSE=0.0009057780288794508、R²=0.997317958934518，通过旧形状门槛。B 仍在第 6 点、tau=.12；A/B=0.20333570681853386<.5。final MSE=2.7750242503034184e-30。训练 6.794929027557373 秒，A 拟合与 B 读峰合计 0.17409706115722656 秒。5% 步长判据通过仍不证明协议一致或 B 真峰收敛。

## 核验、失败与边界

开头只读核验上一轮科学收尾 753a19e 的 17/17 Git blob、四份 passed 回执、90/90 原历史 pin 与 4/4 恢复结果 pin。旧 state 的提交 blob 匹配；当前 state 由 supervisor 推进到 105，不以当前字节否定旧收尾。独立子审查全程 0 训练、0 拟合、0 文件写入。

本轮 103/103 旧文件集合/hash/mtime保持。12 个保存 checkpoint 及最初 2 次局部更新用独立 einsum 与 math.fsum 重构，loss 最大误差 1.1102230246251565e-16，更新最大误差 2.7755575615628914e-17。冻结独立核验用 expit 与高精度求和重算 RMSE/R² 差均 0，B 峰与 P1 判定一致。另一份只读审查的 RMSE 差为 1.0842021724855044e-19，R²/tau 差均 0；各自回执数字分开保留。恢复调用新增 0 训练/0 fit，reused cell/fit 各 1；NPZ、receipt、fit、summary 与独立回执共 5 个文件 hash/mtime 不变。

新训练再次发出 matmul 的 divide-by-zero、overflow、invalid RuntimeWarning，完整 stdout/stderr 保留。保存 loss 和全部 checkpoint 权重有限，独立重构通过；根因未定，不重训成功 cell。分析 stderr 为空。冻结前首次 Python heredoc 写稿命令有未闭合字符串 SyntaxError，解析前失败，未写文件或运行训练/拟合；修正编排转义后完成，未改冻结字节。

旧 P2 refuted 2/3、最细共同 eta 的 CV=.276924>.25、B tau=.12 第6点及 A/B 偏移、tc≥1 文字/源码≥0偏差、r102 purpose“不代替主P1”/实际 reference A 主判据冲突均保留。旧 matmul 未定日志、根编排三次 JS 解析失败及独立审查一次无 shell 执行记录不变。

最终只读审查通过报告数字保全、57 个证据链接、116 条旧 KB 保留及 state 仅两字段改变。独立链接扫描发出一次 FutureWarning，随后 plain string 扫描通过；无写入、训练或拟合。

仅单个已选 seed 的两档离散 eta、有限 tau=60、单起点局部拟合；不证明连续步长极限、seed 总体、参数可识别或全局最优。不进入 skip/deff，不做新 OOD、其他谱/sigma/宽度/目标/optimizer/泛化/无限梯度流外推。

## 下一小问题

下一小问题仅新增尚未执行的L4 seed33 eta=.0003125/T192000/tau60，逐位复用其原数据目标层矩阵，固定reference时钟/全点/原单起点模型bounds clip精度；检验tau_A相对该seed旧eta=.00125基准0.04093862880339804是否变化≤5%。先核查全部历史cell，冻结新点预测与明确5%数值区间（披露旧差值非盲外推）、NPZ/metadata/合同hash与三源码唯一commit，再训练/fit；不预先计算新tau。 本轮未运行 seed33 新条件。

## 证据

- [预注册](../studies/r105_seed11_refinement/preregistration.json)、[训练源码](../studies/r105_seed11_refinement/executed/run_experiment.py)、[分析源码](../studies/r105_seed11_refinement/analysis.py)、[独立源码](../studies/r105_seed11_refinement/executed/verify.py)。
- [原始结果](../studies/r105_seed11_refinement/results/)、[reference 拟合](../studies/r105_seed11_refinement/fits/)、[summary](../studies/r105_seed11_refinement/summary.json)。
- [独立回执](../studies/r105_seed11_refinement/executed/independent_verification.json)、[恢复回执](../studies/r105_seed11_refinement/executed/recovery_verification.json)、[执行审计](../studies/r105_seed11_refinement/executed/execution_audit.json)。
- [训练 stdout](../studies/r105_seed11_refinement/executed/training_stdout.log)、[原始警告](../studies/r105_seed11_refinement/executed/training_stderr.log)、[分析 stdout](../studies/r105_seed11_refinement/executed/analysis_stdout.log)、[分析 stderr](../studies/r105_seed11_refinement/executed/analysis_stderr.log)。
- [seed11 旧基准](../studies/r101_lr_protocol_offset/fits/L4_s11_eta00125.json)、[旧 NPZ](../studies/r101_lr_protocol_offset/results/L4_s11_eta00125.npz)、[上一轮收尾回执](../studies/r104_A_midpoint_start/executed/final_commit_verification.json)。
