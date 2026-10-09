# r103：固定曲线的四个斜率起点检查

## 结论

P1 支持 4/4。同一条 seed22 最细步长曲线只改变拟合初始斜率 k0=.25/.5/2/4，四个拟合均成功。tau_A 相对原 k0=1 读数的最大差为 6.375948340467023e-8，即 0.000006375948340467023%，低于事前 1%。没有新增训练，也没有重拟合成功的 k0=1。

本次四个 k 起点未造成超过 1% 的读数差。它不建立从架构与初始化读出有效深度的公式。它也不证明所有参数起点都稳定、参数可识别或全局最优。旧 A/B 偏移与跨 seed 稳健性失败仍保留。

## 小问题、预测与冻结

只问同一保存曲线的 reference A 是否对四个新 k 起点稳定。该检查来自上一轮指定建议，是 development 条件；已知 k0=1 成功读数，不称盲发现。四条点预测均为 .01940348189419637，区间均为 [.019209447075254404,.019597516713138335]。判据要求四个拟合均 success 且相对原读数的差≤1%。该区间不是置信区间。

唯一提交 cdad9d28c8279717c92108204193d9bfaf8ad062 同时冻结合同、执行/分析/核验三份源码、全部历史来源 hash/mtime，早于首新拟合 22.294275999069214 秒。冻结前没有运行四个起点或计算新参数。旧报告符合八节骨架，无需重排；不存在 inbox，因此无处理标记。按要求 git add -A 包含已有 supervisor 日志、state 与本轮 prompt/output，未科学使用它们。

## 条件与结果

只读复用无 skip L4、hidden32、128×4 固定高斯输入、4×4 正交目标、sigma=.8、seed22 原始矩阵、float64 CPU 全批量 GD/mom0/nodecay 优化 J=4×MSE 的保存结果。原 eta=.0003125、T=192000、末端 tau=60；输入/目标种子为 20261008/20261007。没有重建数据、没有训练。

reference A 时钟仍为 s=.0003125×step/.02，模型、全部 192001 点、均匀 step 权重、自由上下平台、原 bounds/clip 与三项 1e-12 精度不变。初值为 [loss0,final loss,k0,log(11)]。旧半下降位置 s=1.21875<10，原 max(10,half) 因而同样给 h0=log(11)。仅 k0 改变；每个结果独占创建。

| k0 | tau_A | 相对原读数差（无量纲） | nfev | P1 |
|---:|---:|---:|---:|---|
| .25 | .01940348159976962 | 1.5173913215512478e-8 | 17 | 支持 |
| .5 | .01940348123809112 | 3.3813789379069e-8 | 16 | 支持 |
| 2 | .019403483131352352 | 6.375948340467023e-8 | 14 | 支持 |
| 4 | .019403482026889962 | 6.838648493542767e-9 | 15 | 支持 |

四次拟合共 .8001992702484131 秒。归一化 RMSE 范围 [.0009849355772315331,.0009849355772315472]，R² 范围 [.9968988754894406,.9968988754894407]，4/4 通过旧形状门槛。这些是同一训练曲线的四个拟合 cell，不计作四个独立训练 cell 或 seed。

## 核验、失败与边界

开头核验上一轮科学收尾 9bec9c7 的 18/18 Git blob、独立核验与恢复回执。60 个历史文件及 4 个成功结果的 hash/mtime 匹配；旧 state 的提交 blob 正确，当前 state 由 supervisor 推进到 103。独立审查用 expit 与 math.fsum 复算旧 A 指标，RMSE 差 4.336808689942018e-19、R² 差 0。旧回执保留 12 checkpoint/2 次局部更新误差 2.220446049250313e-16/3.469446951953614e-18，未调用会写文件的旧 verify.main。

本轮 74/74 历史 study 文件的 hash/mtime 未变。四个保存拟合用 expit 和 math.fsum 独立重算，RMSE 最大差 2.168404344971009e-19、R² 与 tau 差均为 0，判据匹配。恢复调用新增 0 训练、0 拟合、只复用 4 个结果；4 个结果与 summary 共 5 个成功文件的 hash/mtime 不变。独立审查确认合同和三份源码等于唯一冻结提交字节，结果目录仅有四个预指定起点。

本轮拟合 stderr 为空。恢复编排工具输入两次 JS 引号解析错误，均未执行 shell 命令；修正调用后通过，未改冻结源码或重拟合结果。旧训练 matmul 除零、overflow、invalid 原始日志保留，根因仍未定。旧 tc≥1 合同文字与源码 tc≥0 偏差保留。旧 reference_protocol.purpose“不代替主P1”与该轮主判据为 reference A 的冲突保留冻结字节，本轮 purpose 明确写为主判据。

旧 P2 的 10% 步长收敛仅通过 2/3，最细三 seed CV=.276924>.25；本次起点稳定不替代这些判据。旧 B 仍 tau=.12、第 6 网格点，A/B=.1616956824516364，不称真峰收敛或协议一致。只检查已选 seed22 曲线的四个 k 起点；不进入 skip/deff，不外推新谱/sigma/宽度/目标/optimizer/泛化/无限梯度流/seed 总体或 OOD。

## 下一小问题与证据

优先 0 新训练：复用相同曲线，仅固定 k0=1 改 h0 为 log(2) 与 log(101) 两个未执行起点，其余平台/全点/模型/边界/精度不变。可预测两者均成功且 tau_A 距原 .01940348189419637 的 1% 内。先核查历史起点、冻结两条新数值预测、来源 hash 与源码唯一 commit，再运行。不重复 h0=log(11) 的旧成功拟合或本轮四个 k 起点，不称全局最优或参数可识别。

- [合同](../studies/r103_A_start_sensitivity/preregistration.json)、[拟合源码](../studies/r103_A_start_sensitivity/executed/run_experiment.py)、[分析源码](../studies/r103_A_start_sensitivity/analysis.py)
- [结果](../studies/r103_A_start_sensitivity/results/)、[summary](../studies/r103_A_start_sensitivity/summary.json)
- [独立核验](../studies/r103_A_start_sensitivity/executed/independent_verification.json)、[恢复核验](../studies/r103_A_start_sensitivity/executed/recovery_verification.json)、[执行审计](../studies/r103_A_start_sensitivity/executed/execution_audit.json)
