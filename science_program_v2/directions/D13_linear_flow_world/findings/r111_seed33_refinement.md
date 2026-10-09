# r111：seed33 固定时钟 A 的 5% 步长检验

## 结论

P1 支持 1/1。固定 seed33 的数据、目标和所有初始层矩阵，仅 eta=.00125→.0003125，reference tau_A 从 0.04093862880339804 变为 0.04147141342752652。按旧值计变化 1.301422738624447%，低于事前 5%。

这使三个已选初始化 seed11、seed22、seed33 都完成同一两档离散步长检验，三者均在 5% 内。该结果仍不建立连续 eta 极限、seed 总体概率、协议不变量或有效深度公式。seed33 的 reference A/B 比为 0.3455951118960543，仍低于 0.5；B 的读数为 tau=.12、第 6 个公共网格点。

## 小问题、预测与冻结

本轮只问 seed33 的 reference A 读数是否相对 eta=.00125 旧值变化不超过 5%。预注册点预测为 0.041452492222264194，事前操作区间为 [0.03889169736322814, 0.04298556024356794]。点预测来自已见 seed33 的 reference A eta=.005 与 .00125 差值的四分之一，属于非盲 development 外推，不是置信区间或解析界。新读数比点预测高 1.8921205262326102e-05。

唯一预注册提交 f23ea04bdda776867ff81932eb9d1c982779b587 早于训练。冻结时未执行 seed33 新 eta 训练或拟合。合同、训练/分析/核验三源码以及 119 个历史输入的 hash/mtime 在冻结前固定。新条件此前不存在于历史结果集合。

## 条件与结果

条件为无 skip、L4、hidden32、n128d4 固定高斯输入、4×4 正交目标、sigma=.8、seed33 原初始化、float64 CPU 全批量 GD、mom0/nodecay，优化 J=4×MSE，eta=.0003125，T=192000，tau=60。输入、目标和四个初始层矩阵逐位复用 eta=.00125 基线。保存 192001 个 MSE 点和 12 个权重检查点。

reference A 使用 s=eta×step/.02 的统一时钟、全部点、自由上下平台、原单起点有界 local least_squares、原 bounds/clip 与 1e-12 精度。A 成功，归一化 RMSE=0.0012790417775864755，R²=0.9965349072347274。reference B 在公共 tau=.02 网格第 6 点，tau_B=.12。训练耗时 6.927286863327026 秒。

## 核验与失败

独立核验通过。12 个 checkpoint 和最初 2 次局部更新的最大 loss/更新误差分别为 1.1102230246251565e-16 与 1.3552527156068805e-20；A 拟合 RMSE 重算误差为 2.168404344971009e-19，R² 差为 0；B 峰索引与预测重算一致。恢复调用新增 0 训练、0 拟合，复用 1 cell/1 fit。119 个历史文件 hash/mtime 未变。

训练再次出现 matmul 的 divide-by-zero、overflow、invalid RuntimeWarning。保存 loss 与 checkpoint 权重有限，独立重构通过；根因未定，原始 warning 记录保留。预注册后第一次启动传入短 hash，冻结校验拒绝且未训练；第二次传入无效长 hash，同样未训练。第三次使用完整冻结 hash 成功。冻结前两次 JS 编排解析失败和一次 heredoc SyntaxError 均未执行 shell、未写结果、未训练或拟合。

## 边界与下一问题

本轮只支持三个已选初始化在 eta=.00125→.0003125 两档离散步长下的 reference A 5% 判据。旧固定时间轴 P2 仍为 2/3，最细 eta 的 seed CV=.276924，A/B 协议偏移仍保留。单起点 local 拟合不是全局最优证明。未测连续 eta、其他 sigma/宽度/目标/优化器、skip、泛化、无限梯度流、seed 总体或 OOD。

下一轮优先 0 新训练：只读汇总三个 seed 在 eta=.0003125 的 tau_A 跨 seed 变异与旧 eta=.00125 的配对变化，明确是否仍超过 CV=.25；不改曲线、不重跑训练或拟合。

## 证据

- [预注册](../studies/r111_seed33_refinement/preregistration.json)、[训练源码](../studies/r111_seed33_refinement/executed/run_experiment.py)、[分析源码](../studies/r111_seed33_refinement/analysis.py)、[独立核验源码](../studies/r111_seed33_refinement/executed/verify.py)。
- [结果](../studies/r111_seed33_refinement/results/)、[拟合](../studies/r111_seed33_refinement/fits/)、[summary](../studies/r111_seed33_refinement/summary.json)。
- [独立核验](../studies/r111_seed33_refinement/executed/independent_verification.json)、[恢复核验](../studies/r111_seed33_refinement/executed/recovery_verification.json)、[执行审计](../studies/r111_seed33_refinement/executed/execution_audit.json)。
- [训练 stdout](../studies/r111_seed33_refinement/executed/training_stdout.log)、[原始 warning](../studies/r111_seed33_refinement/executed/training_stderr.log)、[分析 stdout](../studies/r111_seed33_refinement/executed/analysis_stdout.log)。
