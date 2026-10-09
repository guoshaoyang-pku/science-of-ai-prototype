# 固定早期幅度公式的新初始化检验

本轮只问：在同一 d3/n32 Gaussian、两 hidden layer SiLU recipe 中，step32 固定 r0 的 log(rho) 加宽度或固定目标 y 的早期 Rayleigh log 比值，能否减少 step256 log(rho) 的预测误差？这是 development 检验，函数、宽度和输入 draw 均已用过，只换初始化 seed；旧 mixed_sine×width8 的 0/3 方向反例保留。

## 可行性、冻结预测与执行

先只读旧 12 cell，以 E=log(rho32)、Q=log((yᵀK32y)/(a32 yᵀK0y))、L=log(rho256) 做显式探索校准，保留负值和 deadband。设计矩阵 E/W/T 的 rank 为 2/3/3，条件数为 11.160/14.765/14.453。按初始化 seed611/629/647 整组留出，同初始化的四个 recipe 一起留出，MAE 为 .732049731/.721367713/.632144327；拟合内误差为 .524735774/.421174635/.427489291。T 的旧组留出改善 .099905404，W 只改善 .010682018，系数跨组不稳。以上是预注册前使用旧 development 结果的探索，不冒充前瞻验证。

三个冻结公式为 E 模型 Lhat=1.428713415+.436889464E；W 模型 Lhat=.986409212+2.032984577E+.587644437 I(width=8)；T 模型 Lhat=1.982238919+3.109145083E−3.152502126Q。新 seed701/719/737、product/mixed_sine×width8/32 共 12 cell，不重拟合、不删 cell。P1 预测 T 的 MAE≤.70 且相对 E 改善≥.08；P2 预测 W 的改善<.08。原有 data_seed260606、标签样本居中/RMS1、bias、float32 默认初始化后 .double()、halfMSE、full-batch SGD eta=.05/mom0/nodecay、256 步保持相同。

预注册与源码提交 f6dce72 早于全部新训练。先运行 product_w8_s701 并核验，再恢复余下 11 cell。每个 cell 在 step32 写入只包含 E/Q/冻结预测的 early.json，随后继续到 step256；全部早期时间戳和输出一致。累计 cell 执行时间 1.752172 秒，12/12 cell 成功，4/4 recipe 完整。新代码以 einsum 复算核，无新 matmul 警告；旧警告根因未定，旧代码和 26 份输入证据 hash 均保持不变。

## 结果与边界

P1 与 P2 均 supported。新 cell 的 E/W/T MAE 为 .550345073/.498007947/.441188817，自然 log 单位；W/T 的同 cell 平均绝对误差改善为 .052337126/.109156256。T 的 MAE 比 E 低 19.83415%，但仅 7/12 seed 配对的绝对误差更小，不能说逐 seed 都有效。T 的绝对误差范围为 .034352882–1.229845961；product×width8×seed701 的 T 预测为 1.435308868、实测 L=2.665154829，是最大误差样本，保留。

| recipe | E MAE | W MAE | T MAE | T 相对 E 的配对改善 [初始化 seed 95% t 区间] |
|---|---:|---:|---:|---:|
| product×width8 | .689108873 | .541057069 | .677781454 | .011327420 [−1.306143395, 1.328798235] |
| product×width32 | .353222750 | .333551589 | .202855763 | .150366987 [−.532222901, .832956874] |
| mixed_sine×width8 | .903600011 | .805660105 | .614199435 | .289400576 [−.112959917, .691761069] |
| mixed_sine×width32 | .255448658 | .311763026 | .269918617 | −.014469959 [−.410333884, .381393965] |

固定 y 的读数与固定 r0 不同；Q 的负系数只属于此冻结关联公式，不证明目标对齐越高越坏。四个 recipe 的改善区间均跨零，mixed_sine×width32 的均值反而变差。三初始化 seed 只测稳健性，同一输入 draw 与两种已用函数不支持新数据或函数外推；本轮结果不证明核漂移的因果机制，也不预测晚期 rt 拟合速度或 test 泛化。

全部36checkpoint保存参数重建的输出/J最大差均为1.110223e−15；初始化参数完全匹配，0新训练核验。

## 证据与交接

自包含执行和分析源文件、冻结合同与旧输入 hash 位于 [新 study](../studies/r022_early_amplitude/preregistration.json)；[校准记录](../studies/r022_early_amplitude/executed/calibration.json) 与 [summary](../studies/r022_early_amplitude/summary.json) 可复算全部数字；[成功 cell 与恢复核验](../studies/r022_early_amplitude/executed/saved_evidence_verification.json) 保存不可覆盖检查；[独立参数/J重建](../studies/r022_early_amplitude/executed/independent_verification.json) 核验全部36checkpoint。旧科学产物的真实收尾提交为 3dd7adf；旧状态的 03a3741 仅提交 output.log，本轮未改旧状态历史条目或旧结果。

下一轮建议只变 input data_seed，保留本轮冻结三公式和新初始化 seed，对同四 recipe 检验 T 的 .70/.08 判据是否保持。新 draw 属于 development；必须先写明确新预测和 pinned 源码、取得新预注册 commit，不得覆盖两批成功 cell 或按新数据重拟合。
