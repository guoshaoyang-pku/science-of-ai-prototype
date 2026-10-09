# 第34轮：宽度32下的逐层冻结时间

本轮只改变隐藏宽度：保留 data_seed260071、输入与标签、SiLU 无 bias/LN/residual、depth1/2/4/6、初始化 seed1103/1129/1151、fan-in 缩放、全批量 half-MSE SGD、eta=.005、momentum0、weight_decay0、T4096，将 width16 改为32。跨宽度沿用种子与缩放规则，但矩阵形状、随机数消费和初始函数改变，不是逐位相同的初始化。

## 执行与预测

预注册、run.py、analysis.py、verify.py 和旧证据审计在提交15be488冻结，源码门禁先核对提交内完整字节再训练；提交时间严格早于最早训练。指定 Python、CPU 单线程小张量，无网络/GPU/API。4个深度 recipe×3个初始化重复，共12/12 cell完成，训练累计7.105134秒；参数量为128/1152/3200/5248。成功结果保存每步损失、初始与终点参数，以及8个 checkpoint 的逐组J/K/谱/残差能量。无新封存OOD。

P1（最慢参数组的有限冻结时间在12/12 cell均高估实际半损失时间超过两倍）supported：比值18.333333–60.562500，四个深度均3/3高估超过两倍。P2（相加全核至少10/12在[.5,2]且每深度至少2/3）supported：12/12通过，四个深度均3/3；比值.666667–1.052632。预测高估通过与候选拟合失败是不同判定，原最慢组两倍内预测失败仍保留。

| depth | 实际T50（seed1103/1129/1151） | 最慢组候选 | 全核候选 |
|---|---|---|---|
| 1 | 12/15/18 | 437/275/331 | 12/15/18 |
| 2 | 15/11/17 | 385/339/766 | 15/10/16 |
| 4 | 27/59/24 | 1252/1564/873 | 25/55/20 |
| 6 | 16/57/6 | 969/2443/293 | 13/60/4 |

## 配对与边界

按depth1/2/4/6、各depth中seed1103/1129/1151的顺序，width32−width16实际步数差为−3/−26/−20、−55/−5/+1、−18/−154/+22、−74/−662/−16；新旧比值范围.079277–12。10/12加快，depth2_seed1151的16→17与depth4_seed1151的2→24变慢。旧两个全核反例在本配对变成55/59=.932203与60/57=1.052632；不撤销旧反例，不推宽度普遍加快或普遍提高精度。width32中最慢组在11/12为input，depth6_seed1129为最后hidden；head在候选内。

固定随机种子不能固定跨宽度矩阵或初始函数；参数量、初始残差方向与核均随宽度改变。仅development、当前数据/目标/种子/四深度和eta=.005，不独立归因宽度机制，不作深度单调、因果、泛化、其他优化器或OOD外推。

## 复算与保留

独立解析Jacobian与矩阵平方递推通过12个cell/96个checkpoint：Jacobian最大误差5.329071e−15，首步预测3.552714e−15，谱重建2.842171e−14，冻结过线占比8.459899e−14；核重建与相加误差为0。所有初始候选稳定，最大eta·lambda_max=.1928343819<2，没有无限、删失或发散。新分析统一unstable为两倍内失败并保留slowest聚合状态；不能把unstable计作P1有限高估支持。旧eta=.02谱分支缺失、eta=.005分支合同差异和旧matmul警告原样保留，新条件未触发这些分支。

恢复调用逐cell核对合同/hash并跳过12/12，0新训练，24个结果文件hash/mtime未变；两个旧study共71文件hash/mtime未变。旧科学收尾429d475和8903ba8的39/37条回执hash在各自提交内全部核验通过。中心仅更新D10的last_round与next_question，supervisor的round/rounds_done不改。

## 证据

- [预注册](../studies/r034_width32/preregistration.json)、[冻结源码hash](../studies/r034_width32/executed/manifest.json)、[训练源码](../studies/r034_width32/executed/run.py)、[分析源码](../studies/r034_width32/executed/analysis.py)、[独立核验源码](../studies/r034_width32/executed/verify.py)。
- [逐值汇总](../studies/r034_width32/summary.json)、[独立复算](../studies/r034_width32/executed/verification.json)、[训练前审计](../studies/r034_width32/executed/baseline_audit.json)、[执行与恢复审计](../studies/r034_width32/executed/execution_audit.json)。
- 预注册与源码提交15be488；无inbox，无待追加处理标记。
