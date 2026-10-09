# r102：seed22 固定时钟 A 读数的 5% 步长检验

## 结论

预测 P1 通过 1/1。仅把 eta=.00125 降为 .0003125，固定时钟 A 时间从 0.018598187170700903 增至 0.01940348189419637。以旧读数作分母，变化为 4.329963539479303%，低于预注册 5%。这支持一个已选初始化、两档离散步长之间的读数稳定性；尚未建立从架构与初始化读出有效深度的标定公式。

## 小问题、预测与冻结

只检验 seed22 的 reference A 时间是否相对当前值变化不超过 5%。选取上一轮差值最大的 seed，是 development 条件选择；不是封存 OOD，也不是盲选。点预测 0.01938013891034445 来自旧 eta=.005→.00125 增量的 1/4。判定区间为 [0.01766827781216586,0.01952809652923595]，是事前判据，不是置信区间。新读数比点预测高 0.000023342983851919186。

合同、训练/分析/独立核验三份源码和全部历史输入 hash 在唯一预注册提交 18390942c15dbc88aeed95ab00c6070544c50363 冻结，早于首新训练 12.39551305770874 秒。冻结前没有生成或训练新 eta，也没有拟合新曲线。原报告符合规定骨架，无需重排；无 inbox 文件，无处理标记。

开头核验上一轮 scientific_closeout 的 34/34 git blob、独立核验回执及恢复回执。30 个原 study 文件和 21 个成功 NPZ/回执/fit 的 hash 与 mtime 均保持。旧 state 在原提交中的 blob 正确；当前 state 已由 supervisor 推进到 102，没有将此差异算成旧科学产物损坏。另独立只读复算旧 72 checkpoint、12 局部更新及已存拟合指标，不重训、不重拟合旧成功结果。

## 条件与结果

无 skip、L4、隐藏宽度 32、固定 128×4 高斯输入、4×4 正交目标、sigma=.8、初始化 seed22、float64、CPU 全批量 GD、无动量衰减。生成器的 target/input seed 分别为 20261007/20261008，逐位复用原初始化、数据与目标；全部输入数组 hash 在训练前固定。更新仍为 g=2err/n，实际优化 J=4×MSE，记录每步 MSE。新增唯一 cell 为 eta=.0003125、192000 步、末端 tau=60；训练 7.587523937225342 秒。

reference 时钟固定 s=eta×step/.02；A 使用全部 192001 个点、有自由上下平台的 log-time logistic。使用原单起点局部 least_squares，函数与上一轮逐字一致；没有另拟合 legacy A。reference B 在 tau=.02 公共网格使用原 11 点窗口读峰。

A 的归一化 RMSE=0.0009849355772315344，R²=0.9968988754894407，通过旧形状判据。B 仍在第 6 网格点、tau=.12，A/B=0.1616956824516364，低于 .5。5% 的 A 稳定性通过，没有消除 A/B 协议偏移；B 的零变化只反映粗网格不变，不证明真实峰值收敛。final MSE=3.719756368669228e-30。

## 核验、失败与边界

60 个历史 study 文件的 hash 与 mtime 未变。独立 einsum/高精度求和重构 12 个保存检查点和最初 2 次局部更新，loss 最大误差 2.220446049250313e-16，权重更新最大误差 3.469446951953614e-18。A 拟合 RMSE 重算误差 4.336808689942018e-19，R² 差 0；reference B 峰索引和 P1 判定重算一致。恢复调用 0 新训练、0 新拟合；NPZ/回执/fit/summary 共 4 个成功文件的 hash 与 mtime 不变。

训练再次发出 matmul 除零、overflow、invalid RuntimeWarning，原始 stdout/stderr 已保存。保存 loss 和检查点权重全部有限，独立重构通过；根因未定，不把告警称为已解决。预注册前源码草稿和收尾文档脚本的引号转义错误发生在解析阶段，没有新增实验计算。按要求 git add -A 保留已有 supervisor 异步日志、状态与 round 目录，未科学使用其他方向产物。

冻结合同的 reference_protocol.purpose 沿用旧“不代替主P1”措辞；本轮 predictions.P1、question 和分析明确以 reference A 为主判据。保留冻结字节并在此披露，不在结果后更改合同。旧 tc>=1 文字与实际 tc>=0 源码偏差保留。单起点局部拟合不是全局最优证明；仅本 seed 达到两档步长 5% 判据，不推导连续 eta 极限或 seed 稳健性。未做新 OOD、skip、新谱、新 sigma、新宽度、新目标、其他 optimizer、泛化、无限梯度流或有效深度公式外推。

## 下一小问题与证据

优先 0 新训练：仅复用本轮保存曲线，保持 reference A 全点/模型/边界/精度不变，新增 k0=.25/.5/2/4 四个未执行起点（其余 a0=loss0、c0=final、h0=log(11)，仍按原 clip）。检验四个局部拟合是否均成功，且 tau_A 都在当前读数的 1% 内。先冻结四个新数值预测与源码/输入 hash 单一 commit；不重拟合已成功的 k0=1 起点。不称多起点搜索全局最优。

- [预注册](../studies/r102_seed22_refinement/preregistration.json)、[训练源码](../studies/r102_seed22_refinement/executed/run_experiment.py)、[分析源码](../studies/r102_seed22_refinement/analysis.py)
- [结果](../studies/r102_seed22_refinement/results/)、[拟合](../studies/r102_seed22_refinement/fits/)、[summary](../studies/r102_seed22_refinement/summary.json)
- [独立核验](../studies/r102_seed22_refinement/executed/independent_verification.json)、[恢复核验](../studies/r102_seed22_refinement/executed/recovery_verification.json)、[原始告警](../studies/r102_seed22_refinement/executed/training_stderr.log)
