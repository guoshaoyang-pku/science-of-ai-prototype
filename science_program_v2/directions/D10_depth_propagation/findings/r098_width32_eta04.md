# 第98轮：width32 学习率 eta=.04 的逐层冻结时间

本轮只改变学习率：固定 data_seed260071 的输入与标签、width32、SiLU、无 bias/LN/residual、depth1/2/4/6、init_seed1103/1129/1151、全部初始参数与规则、full-batch half-MSE SGD、momentum0、weight_decay0、T4096，将 eta 从 .02 升至 .04。问题是全核冻结候选是否仍在 12/12 cell 的两倍内。

## 预注册与执行

预注册及 pinned run.py、analysis.py、修正 checkpoint0 筛选的 verify.py 在训练前提交 9b4ee55。提交先冻结，训练后才产生结果。事前从已保存 eta=.02 checkpoint0 谱计算最大初始 eta·lambda_max=1.5426750550657522<2；全核候选预测为 1–8 步，最慢组候选为 35–306 步。这是注册前已知算术，不是盲发现。

12/12 cell 完成，训练累计 7.058 秒。每个深度是一个 recipe，三个 seed 是重复。保存每步 loss、初始/终点参数及 0/1/8/32/128/512/2048/4096 checkpoint 的 prediction、residual、Jacobian、kernel、eigenspectrum、residual energy 和 hidden RMS。旧 r026/r030/r034/r042 cell 未覆盖。

## 结果

P1 supported：最慢组 12/12 的候选/实际 T50 均大于 2，范围 17.5–60.5；四个深度各 3/3。实际 T50 按 seed1103/1129/1151 为 depth1 2/2/2，depth2 2/2/2，depth4 4/8/3，depth6 2/8/1。

P2 supported：全核 12/12 在 [0.5,2] 内，四个深度各 3/3。全核候选/实际比值按四深度分别为 depth1 1/1/1、depth2 1/1/1、depth4 .75/.875/1、depth6 1/1/1。

相对 eta=.02 的同 cell 实际时间新/旧比值为 .4–1，差值为 0 至 −7 步；depth6_seed1151 为 1→1，记录整数阈值影响。eta=.04 的全核比值范围与 eta=.02 的 .75–1 相同；最慢组仍全部超过两倍。

## 独立核验与边界

修正核验通过 12 cell、96 checkpoint，未产生新训练。最大 prediction/Jacobian/首步/final prediction/loss/eigen reconstruction/frozen fraction 绝对误差分别为 3.1086244689504383e−15、5.329070518200751e−15、2.275957200481571e−15、2.6645352591003757e−15、2.220446049250313e−16、1.4210854715202004e−14、6.938893903907228e−15；kernel、kernel_sum、residual_energy 误差为 0。最大初始 eta·lambda=1.5426750550657522。

本轮是 development。结论只适用于 width32、当前输入/标签、四个已测深度、三个初始化、eta=.02/.04 的配对及 full-batch SGD。未测其他学习率、宽度、深度、优化器、mini-batch、测试/泛化或 sealed OOD。核求和与固定核递推是已知解析关系；核漂移与残差方向变化未被独立干预。保留旧 width16 全核反例、跨 width 变慢 cell、历史不稳定分支差异和 NumPy matmul 警告。

## 证据

- [预注册](../studies/r098_width32_eta04/preregistration.json)，冻结提交 9b4ee55。
- [训练源码](../studies/r098_width32_eta04/executed/run.py)、[分析源码](../studies/r098_width32_eta04/executed/analysis.py)、[修正独立核验源码](../studies/r098_width32_eta04/executed/verify.py)。
- [逐 cell 汇总](../studies/r098_width32_eta04/summary.json)、[训练日志](../studies/r098_width32_eta04/executed/training.log)、[分析日志](../studies/r098_width32_eta04/executed/analysis.log)、[独立核验](../studies/r098_width32_eta04/executed/verification.json)。
