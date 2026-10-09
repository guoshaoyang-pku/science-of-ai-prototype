# r038：alpha=0/.01 的拟合不足依赖选解目标

只选一个小问题：在同一保存曲线、固定平台、网格与阈值下，最大误差选解是否消除原最小二乘判定的不足？2 条件×3 坐标 seed，共 6/6 新拟合 cell 完成，0 新训练；原 15/27 训练 cell 与 122 个历史文件保持 hash/mtime 不变。

## 合同与提交

本轮为程序第 38 轮、D4 第 4 轮。已依次读取 GOAL、AGENTS、state、task、已有 findings/report；无 inbox。先重排 report 为公式区开头的六节骨架，保留旧数字与结论。r018 收尾 8f96d00/receipt bbfc9c5、原预注册 c07224e 及旧 r010 收尾 37ee424 已核验。

新预注册与 pinned 执行/分析源码提交 299f6cf8df38da4ea811596755fd8b1317007eff 先于任何新拟合。执行前核对当前 HEAD 内合同及源码字节一致，再核对 r018 summary SHA256=61ebbf59efb05adaea7785b7964984ff35036156eed3611ba10ab01f57ba6e47 及 6 输入 NPZ/metadata hash。原最小二乘参数直接读取保存 summary，不重新最小二乘拟合。

条件固定为 n=d=2、float64、固定特征平方损失、全批量 SGD eta=.1/mom0/nodecay、谱[.01,1]、L0=1、原 T10000 保存曲线、alpha=0/.01、seed0/1/2。H(0)=1、H(t)=1/[1+(t/t50_fit)^beta_fit]，平台1/0；t50_fit∈[.01,100000]步，beta_fit∈[.1,10]。原150点整数log网格含0、终点4997，各点等权；RMSE/maxabs除以真实动态范围1，≤.03/.05同时成立为足够。坐标seed不作统计置信区间，2条件不是6种独立科学条件。

## 预测与配对结果

| alpha | 原最小二乘 RMSE / maxabs | 最大误差选解 RMSE / maxabs | 配对 RMSE 差 / maxabs 差 |
|---:|---:|---:|---:|
| 0 | .01188040961 / .07462910449–.07462910489 | .01437779613 / .03702806721 | +.00249738653 / −.03760103768 至 −.03760103728 |
| .01 | .01022359488 / .06476095049–.06476095218 | .01227958828 / .03337965078 | +.00205599340 / −.03138130140 至 −.03138129972 |

P1 预测两条件新选解都通过 .03/.05，实际 2/2 条件、6/6 cell 通过，supported。P3 预测 alpha=0 maxabs∈[.025,.045]、alpha=.01∈[.020,.045]，6/6 cell 均落入，supported。P2 预测 maxabs 至少减少 .015 且 RMSE 至少增加 .003；实际最大误差降幅成立，但两个条件的 RMSE 增幅均<.003，6/6 cell 是反例，refuted。

## 方法、核验与边界

最大误差目标用 epsilon 可行性二分，不调用外部 solver。logit(H)=c−beta·log(t)、c=beta·log(t50_fit)，每个采样点产生 c 的线性上下界；两两比较给出 beta 可行区间，结合原参数范围，取区间中点作误差见证。该代数是解析工具，不登记发现；新发现是同曲线、只改选解目标的受控区分。

alpha=0 最大误差数值界 [.037028067163191736,.0370280672213994]；alpha=.01 [.033379650732968,.033379650791175663]，界宽 5.8207661e−11。t50_fit/beta_fit 为 3.02359036549步/1.54682735344 与 3.08045127144步/1.50913825752，不能当作实际半衰期。最大残差位置由原 step1 移到18/17。首cell保存/核验后续跑；恢复只核验6cell、新增0cell，不覆盖成功结果。拟合执行调用 .060657 秒。独立 expit 重建预测差≤1.11e−16，误差与原最小二乘 comparator 复核通过。

alpha=0/.01 原不足仅适用于注册最小二乘选解，不能说明整个固定平台logistic参数族都不足。该结论不修改r018原P2失败及P3支持，也不回答alpha=.25/.5/.75的最大误差选解。全部development；数值可行性界只适用于注册离散网格和参数范围，非连续时间全局声明；不报告连续临界alpha/谱比，不外推自由平台、learned features、其他谱/优化器或sealed OOD。

下一小问题：仅复用保存alpha=.25三条曲线，先提交新数值预测及hash/pinned源码，再检验最大误差选解是否同时通过原.03/.05；不新增训练。

证据：[summary](../studies/r038_maxabs_objective/summary.json)、[预注册](../studies/r038_maxabs_objective/preregistration.json)、[首cell核验](../studies/r038_maxabs_objective/executed/first_cell_verification.json)、[原收尾审计](../studies/r038_maxabs_objective/executed/previous_closeout_audit.json)、[独立复算](../studies/r038_maxabs_objective/executed/independent_saved_review.json)、[恢复核验](../studies/r038_maxabs_objective/executed/resume_verification.json)。
