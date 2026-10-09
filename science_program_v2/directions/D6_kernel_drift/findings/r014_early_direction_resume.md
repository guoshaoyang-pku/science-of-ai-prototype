# r014：恢复 r006 的早期核漂移方向测量

## 小问题

在固定初始残差 r0=f0-y、匹配 trace(K_t) 与既定 d3/n32 Gaussian、SiLU、SGD recipe 下，step32 的目标相关核漂移方向是否保留到 step256？判据沿用冻结的 r006：每个 function×width 单元至少 2/3 seed 在两个 checkpoint 均超过 5% deadband 且方向相同；至少 3/4 单元通过。

## 预注册与执行门槛

当前 .git 已恢复可写。训练前核验历史提交 63ccb32 已冻结 r006_early_direction 的预注册和 pinned run.py、analysis.py；三份源码/合同 hash 与冻结内容一致。本轮交接记录提交为 9cbad6d。先执行 product_w8_s611 1 cell，再由分析脚本核验合同、输入和 NPZ hash，之后恢复余下 11 cell。未覆盖成功 cell。

## 结果

12/12 cell 成功，4/4 recipe 单元完整。P1 通过，3/4 单元满足方向持续判据：product×width8 为 2/3 seed；product×width32 为 2/3 seed；mixed_sine×width8 为 0/3 seed，未通过；mixed_sine×width32 为 2/3 seed。

配对的 late−early log(rho) 均值和 seed 95% t 区间分别为 1.4761（[-0.6805, 3.6328]）、1.0427（[0.8007, 1.2848]）、1.7725（[-0.6112, 4.1562]）、1.2140（[0.6408, 1.7872]），顺序同上。mixed_sine×width8 的三个 seed 均在 step32 落入 deadband 或与 step256 不同向，保留为明确反例。seed 只用于同一 recipe 的稳健性检查。

独立分析复算了全部 Jacobian kernel、谱、trace 匹配、固定 r0/移动 rt Rayleigh 比值、置换比值和 probe recurrence；12 个 NPZ 均 finite。执行时出现重复的 NumPy matmul overflow/invalid warning，尚未定位根因；由于保存数据和复算通过，本轮不据此删除 cell，也不新增 probe 解释。

## 边界

这是 development 测量，不是 sealed OOD 预测验证。固定 r0 的方向持续不等价于移动晚期残差的局部拟合速度，更不等价于 test 泛化。结论只适用于 d=3/n=32 Gaussian、product/mixed_sine、width8/32、SiLU 两 hidden layer、full-batch SGD eta=.05/mom0/nodecay、256 步和本合同的三个初始化 seed。不得外推到新函数、数据 seed、优化器、LN 或更深网络。

证据：directions/D6_kernel_drift/studies/r006_early_direction/summary.json、directions/D6_kernel_drift/studies/r006_early_direction/results/、冻结提交 9cbad6d。
