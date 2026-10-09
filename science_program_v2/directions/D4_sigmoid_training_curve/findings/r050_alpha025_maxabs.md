# r050：alpha=.25 的最大误差下界仍超过 .05

本轮只问一个问题：仅复用 alpha=.25 的三条保存曲线，在同一固定平台、参数范围、评价网格与阈值下，最大误差选解是否仍不足？3/3 新拟合 cell 完成，0 新训练。P1 与 P2 均 supported；该条件的不足没有因改变选解目标而消失。

## 条件与提交时序

程序第 50 轮、D4 第 5 轮。已按顺序读 GOAL、AGENTS、state、task、已有 findings/report；无 inbox。原报告已符合六节骨架，无须重排。旧 15/27 训练 cell 和 6 拟合 cell 仅只读核验，154 个历史 study 文件的 hash/mtime 保持不变。旧收尾与 receipt 均核验：原 completion 回执曾在后续核验提交中更新，按最终 receipt 字节完成审计，没有改变旧曲线或旧结论。

预注册及执行、分析、独立核验源码冻结于 7176d1e9e5e7293f38018994c24c00d9e6d6909e，提交时间 2026-10-07T06:53:14+08:00，早于全部新拟合。执行前核对 HEAD 内四份 pinned 文件与合同字节一致，并核对原 summary 及三个 NPZ/metadata 共 7 个输入 hash。原最小二乘（LS）参数从保存 summary 读取，没有重新 LS 拟合。

曲线来自 n=d=2、float64、固定特征半 MSE、零初始化、全批量 SGD eta=.1/mom0/nodecay、谱[.01,1]、L0=1、T10000、alpha=.25、seed0/1/2。平台固定1/0，H(0)=1，H(t>0)=1/[1+(t/t50_fit)^beta_fit]；t50_fit∈[.01,100000]步、beta_fit∈[.1,10]。原150点整数log网格含0、终点4997，各点等权。误差除以真实动态范围1，RMSE≤.03且maxabs≤.05才称足够。一个科学条件、三个坐标 seed；不作统计置信区间。

## 预测与保存结果

| 项目 | 三坐标 seed 范围 | 判定 |
|---|---:|---|
| 原 LS RMSE / maxabs | .04090645242066033–.04090645242066045 / .17433680592279444–.17433680641499260 | 原选解不足，保留 |
| 最大误差选解 RMSE / maxabs | .05041269241012166–.05041269241012175 / .08338322419501444–.08338322419501458 | 0/3 通过两阈值 |
| 配对新−旧 RMSE | +.009506239989461297–+.009506239989461332 | 描述性比较 |
| 配对新−旧 maxabs | −.09095358221997817–−.09095358172777990 | 描述性比较 |
| min maxabs 数值下界 / 可行上界 | .08338322414783761 / .08338322420604527 | 3/3 同界，宽5.820766091346741e−11 |

P1 预测全部三个 seed 的 min maxabs 数值下界>.05且选解仍不足，3/3 cell、1/1 条件支持。P2 预测新 maxabs∈[.07,.13]，3/3 支持。注册前只知旧 LS 数字及 alpha=0/.01 的最大误差结果，尚未算新 alpha=.25 拟合；旧数据与可行性代数本身不是盲发现。

最大误差选解的 t50_fit=6.106741398761212–6.106741398761244步、beta_fit=.6791496385404663–.6791496385404676。原 LS 最大残差位置是 step1，新选解为166；原精确连续半衰期5.1650409882770125步继续保留，不能将6.1067414步解释成真实半衰期。

## 核验与解释边界

沿用 epsilon 可行性二分：logit(H)=c−beta·log(t)，c=beta·log(t50_fit)。离散残差界转为 c 的线性上下界，成对比较得到 beta 可行区间，不调用外部 solver。每个 cell 保存完整二分 trace、上下界和参数见证。该代数是解析工具；本轮科学区分是同保存曲线换目标后仍有>.05的最大误差下界。

对maxabs≤.05另有保存点的直接矛盾见证：step1下界与step10上界要求beta≥.822477112273925，step152下界与step15上界要求beta≤.487369166379817；三seed均矛盾，60位Decimal复算通过。该核验只读保存数据，没有新拟合。

先保存一个 cell，独立 expit 重建预测并用 (beta,c) 多边形半平面裁剪核验下界不可行/上界可行，随后续跑两个 cell。三 cell 的相同核验全部通过；合同、输入、源码、数组与 receipt 固定 metadata hash 通过。恢复调用只核验3cell、新增0cell。两个拟合执行调用累计 .059152209 秒，远低于1200秒预算；第三调用没有新拟合。

在注册离散网格和参数范围内，独立数值证书支持固定平台单 logistic 无法达到 maxabs≤.05；所以不足不限于原 LS 选解。结论是 float64 数值可行性界，未使用严格区间算术，不宣称连续时间全局定理。选出的 RMSE>.03不证明整个参数族 RMSE必>.03。两时间尺度同时有能量导致单段形状受限是兼容解释，本轮没有隔离机制。

全部 development；不将结论外推 alpha=.5/.75、其他网格、自由平台、learned features、其他谱/optimizer、前段预测或 sealed OOD；不报告连续临界 alpha/谱比。原 alpha=0/.01 的最大误差通过、其 RMSE增幅预测失败、原 LS 的 P2失败与 .25/.5/.75 P3支持均保留。下一小问题可仅读保存 alpha=.5 三曲线，固定其余条件，先注册新数值预测和 pinned源码/输入hash，再检验最大误差下界是否仍>.05，0新训练。

证据：[阈值直接见证](../studies/r050_alpha025_maxabs/executed/independent_threshold_verification.json)、[summary](../studies/r050_alpha025_maxabs/summary.json)、[预注册](../studies/r050_alpha025_maxabs/preregistration.json)、[冻结提交核验](../studies/r050_alpha025_maxabs/executed/preregistration_commit_verification.json)、[旧收尾审计](../studies/r050_alpha025_maxabs/executed/previous_closeout_audit.json)、[首cell核验](../studies/r050_alpha025_maxabs/executed/first_cell_verification.json)、[独立证书核验](../studies/r050_alpha025_maxabs/executed/certificate_verification.json)。

科学收尾提交与逐文件核验记录见[final_commit_verification.json](../studies/r050_alpha025_maxabs/executed/final_commit_verification.json)；核验回执随后单独提交。
