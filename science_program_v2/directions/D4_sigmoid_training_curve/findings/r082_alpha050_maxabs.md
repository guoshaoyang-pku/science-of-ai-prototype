# r082：alpha=.5 的最大误差下界仍超过 .05

本轮只问一个问题：仅复用 alpha=.5 的三条保存曲线，在同一平台、参数域、网格和阈值下，min maxabs 数值下界是否仍>.05？3/3 新拟合 cell 完成，0 新训练；P1/P2 均 supported，独立证书与恢复核验通过。

## 条件与冻结

程序第82轮、D4第6轮。依次读取 GOAL、AGENTS、state、task、全部已有 findings/report；无 inbox。报告已符合公式区开头的六节骨架，无须先重排。上一科学收尾 811fdea4d439d14bc6c89ef5c236d7bfcbea8839 的23个 study 文件及 finding/report 与工作树一致，回执与798c4b19d0355add2ddcb724c6de1f5b8ab380c8一致；D4-009未改。178个旧study文件保存hash/mtime，旧r00215/r01827训练及r0386/r0503拟合cell只读，不重跑。

预注册、执行器、分析与独立核验源码一次冻结于6f7c3355e6936f7648156ba89c89d6a8f4edec3e，早于全部新拟合。七个输入hash及冻结源码字节一致才运行。执行器显式绑定该唯一提交，先扫描所有成功结果、回执、请求与文件集合，再允许新增拟合；不接受混合合同、孤立或额外结果文件。冻结后源码与合同未改。

输入来自 n=d=2、float64、固定特征半MSE、零初始化、全批量SGD eta=.1/mom0/nodecay、谱[.01,1]、L0=1、T10000、alpha=.5、seed0/1/2。一个科学条件、三个坐标seed，无统计置信区间。原LS参数与误差仅读r018保存summary，不重新LS拟合。

平台固定1/0，H(0)=1，H(t>0)=1/[1+(t/t50_fit)^beta_fit]；t50_fit∈[.01,100000]步，beta_fit∈[.1,10]。原150点整数log网格含0、终点4997、各点等权。以真实动态范围1归一化，RMSE≤.03且maxabs≤.05才足够。只改拟合目标与对应选解算法。

## 数值结果与预测

| 项目 | 三坐标seed范围 | 判定 |
|---|---:|---|
| 原LS RMSE / maxabs | .06028518718062050–.06028518718062059 / .08556858257109223–.08556858265219294 | 原不足保留 |
| 最大误差选解 RMSE / maxabs | .06037167065143022–.06037167065143029 / .08268412550007484–.08268412550007492 | 0/3同时过阈值 |
| 同seed新−旧 RMSE | +.00008648347080969510–+.00008648347080971591 | 描述性比较 |
| 同seed新−旧 maxabs | −.0028844571521180334–−.0028844570710173100 | 描述性比较 |
| min maxabs 数值下界 / 可行上界 | .08268412546021864 / .08268412551842630 | 三seed同界，宽5.820766091346741e−11 |

P1预测全部seed的数值下界>.05且最大误差选解仍不足，3/3 cell、1/1条件supported。P2预测新maxabs∈[.06,.09]，3/3 cell、1/1条件supported。注册前只知旧LS数字、alpha=.25已测数值界及alpha=0/.01目标对照，没有先算alpha=.5新可行性或残差；全部为development，不称盲发现。

最大误差选解 t50_fit=27.640329970560757–27.640329970560980步、beta_fit=.5361017690313967–.5361017690313976。原LS最大残差在1798步，新见证在1877步。真实连续延拓半衰期16.316431977857476步来自原保存summary；拟合中点仍不能当作真实半衰期。

## 复算与边界

沿用epsilon可行性二分：logit(H)=c−beta·log(t)，c=beta·log(t50_fit)。逐点残差约束转为c线性上下界，两两比较形成beta区间，不调用外部solver。每cell保存二分trace、数值界与参数见证。这个代数是解析工具，科学结果是受控更换选解目标后alpha=.5仍有>.05的数值下界。

先保存一个cell，独立expit重建与(beta,c)半平面多边形裁剪通过，再续跑两个。三cell下界多边形均空、可行上界均有3顶点。合同、输入、数组与receipt固定metadata hash通过；恢复只核验3cell、新增0cell。两次有新增拟合的调用累计.057776667032158秒，计入恢复的全部三调用.078930542018497秒，低于1200秒预算。178旧文件hash/mtime保持不变。

在注册有限网格与参数域内，float64独立数值证书支持固定平台单logistic无法达到maxabs≤.05，不足不限于原LS选解。证书未使用严格区间算术；不作连续时间全局定理。见证RMSE>.03不能证明整个参数族RMSE>.03。两时间尺度都有能量是兼容解释，没有隔离其机制。

不报告连续临界alpha/谱比，不外推alpha=.75、自由平台、其他网格/谱/优化器、learned features、前段预测或sealed OOD。原alpha=0/.01最大误差选解通过、配对RMSE增幅P2失败、原LS P2失败以及.25/.5/.75 P3支持均保留。下一小问题可只复用保存alpha=.75三曲线，先注册数值预测、pinned源码和输入hash并一次commit再拟合；不重跑本轮或历史成功cell。

## 证据

[预注册](../studies/r082_alpha050_maxabs/preregistration.json)、[提交核验](../studies/r082_alpha050_maxabs/executed/preregistration_commit_verification.json)、[旧收尾审计](../studies/r082_alpha050_maxabs/executed/previous_closeout_audit.json)、[首cell核验](../studies/r082_alpha050_maxabs/executed/first_cell_verification.json)、[summary](../studies/r082_alpha050_maxabs/summary.json)、[独立证书](../studies/r082_alpha050_maxabs/executed/certificate_verification.json)、[恢复核验](../studies/r082_alpha050_maxabs/executed/resume_verification.json)、[独立复算](../studies/r082_alpha050_maxabs/executed/independent_review.json)。

科学收尾提交与核验回执见[final_commit_verification.json](../studies/r082_alpha050_maxabs/executed/final_commit_verification.json)。
