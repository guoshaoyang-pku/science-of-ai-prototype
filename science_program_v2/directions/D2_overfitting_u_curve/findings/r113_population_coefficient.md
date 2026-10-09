# r113：population 归一化没有使聚合起点系数极差减半

在 n=32/64、n/σ²=128/256、seed411–414 的固定特征条件下，注册的“population 标签归一化使跨 n 起点系数极差至少减半”被推翻。样本归一化的两系数为 .351328125/.3355078125，population 为 .2860546875/.3601171875。极差从 .0158203125 增至 .0740625，比值 4.6814814814814865；注册区间 [0,.5] 未通过。32/32 曲线最低点均在窗口内部，0 新训练。

## 问题、定义与来源

只检验一个问题：在相同 n/σ² 格点上，共享 population mean=0/RMS=√1.25 是否使 n32/64 的起点比例系数极差减半。固定幂次 1、零截距，注册定义为 x=n/σ²、cₙ=.3∑(x t*)/∑x²，D=|c64−c32|；等权汇总四 seed 与两个格点。P1 支持要求 D_sample>0、D_population/D_sample∈[0,.5] 且全部最低点内部。这个定义不是发现，也不是旧自由指数留出模型的 .20–.33 系数。

条件沿用 d4 antithetic Gaussian、目标 x₀+.5x₀x₁、冻结 ReLU width128/scale.03、训练特征居中与总 RMS、含 bias 零 head、float64、全批量 GD η=.3/mom0/nodecay、audit2048、整数步0..16384。每个 n/seed 内只改 clean 标签 mean/RMS；跨 n 输入原本不同，不能独立归因样本量或谱。

population 的全部 t* 已保存且已见。协调者事后分析已保存另一散布度量（x=128 的平均 |log₂(t64/t32)|）：sample .7887045273、population .4777503736。它并非本轮固定幂次聚合系数极差，不能用于宣称本轮盲预测。新定义的系数在注册前没有计算；sample n32/σ²=.125 四条曲线在冻结后首次组合。32 个评价包含四条新噪声格点、28 条已有格点复算，来自八个输入/特征抽样、两种标签协议、两格点，四个 seed 是重复而非四种 recipe。

## 配对结果与机制边界

| seed | sample c64−c32 | population c64−c32 | 单 seed 极差是否缩小 |
|---:|---:|---:|---|
| 411 | −.27046875 | −.0459375 | 是 |
| 412 | +.0928125 | +.06609375 | 是 |
| 413 | −.136875 | −.01125 | 是 |
| 414 | +.25125 | +.28734375 | 否 |

sample 聚合系数接近，来自正负 seed 差抵消。三个 seed 的单 seed 极差缩小与聚合极差增大可以同时成立。P1 失败保留，不能改用事后选定的绝对差平均来救回主预测，不能排除 seed414。

已知精确分解写成 Rₙ(t;s)=Bₙ(t)+(s/n)Mₙ(t)，Mₙ=nNₙ。Aₜ 带 1/n 只隔离显式因子；谱与审计对齐仍依赖 n。在 population 的64..4096步、n64相对n32窗口内，M 相对最大差为 .2144157049724281–.4715511865908319，B 为 .2638168788890106–.6513063299158051；相邻步的 ΔM/ΔB 偏差为 .1673454630476911–.25897673488618245/.06705251392015041–1.4568330287010538。这些是注册的辅助描述，没有另加浓集判据。它们不支持忽略 n 依赖；未独立识别有限 n 谱的因果作用。

## 提交与数值核验

先读规定材料与 inbox，再提交报告六节重排、编辑性边界限定。全部旧数字与实测结论保留。新预注册、102 来源 pin、341 旧 study 文件集合/hash/mtime、四份执行分析源码和旧快照一次提交后才重加权。先1 cell独立核验，通过gate后补31；唯一冻结合同未改。

r112 唯一冻结早于评价47.519542秒，7个来源pin/3源码pin/合同匹配；成功结果hash/mtime与真实科学收尾一致。r112执行器未自动核对git中的合同，本轮直接核对确认实际未变。最终回执所指integration提交保留；协调者浓集分析实际使用sample曲线，不当作population结果。

另一次独立审查用有理数复算系数，sample为4497/12800、8589/25600，population为7323/25600、9219/25600；极差比精确为632/135。审查子任务原要求只读，但额外保存了自包含审查源码与回执；两文件为冻结后的验证补件，不属于执行源pin，未改合同/结果/旧文件、未计算未注册条件。

本轮独立用60位Decimal逐点组合保存B/N、独立重选最低点，全部argmin一致，风险舍入差最大0；28条同噪声保存对照最大差0；fsum系数复算差0。341旧文件集合/hash/mtime通过。恢复为0新cell、32复用cell、0训练，65结果文件加summary的hash/mtime不变。图包含全部seed和格点、n32/64系数线、逐seed极差、聚合极差和注册减半阈值。

准备期两次JS解析、一次patch上下文、两次数值保全regex解析失败均未运行风险计算，后已修正并核对旧数字全部保留。落盘时两次Python字符串转义SyntaxError未执行写入，改用分步脚本后完成。Matplotlib生成本study内字体缓存，成图检查后只删除自建缓存。源码未改、成功cell未覆盖。receipt是逐cell更新的恢复索引。

## 适用范围与下一轮

仅上述development条件，不给连续n/σ²阈值、跨seed概率、n128、learned hidden、Adam、小批量、CE、train-only早停、无限时间或sealed OOD。旧57/60留出预测、40/44折叠失败、seed414两个反例与seed413低噪声区间失败均保留；没有重拟合或取消旧结果。

下一轮先核验本轮科学收尾与最终回执，不覆盖32成功评价或任何旧训练。可选一个保存曲线机制反例检验：固定population seed414、x=256，只在n64风险表达式中以n32的M=nN替换原M、保留n64的B，检验起点与n32的237步差距能否至少减半。先预注册具体区间、B/M来源hash与pinned源码，唯一提交后才组合。这是代数反事实分解，不当作真实训练或输入/谱因果干预。

## 证据

- [预注册](../studies/r113_population_coefficient/preregistration.json)、[执行器](../studies/r113_population_coefficient/executed/run.py)、[分析](../studies/r113_population_coefficient/analysis.py)与[汇总](../studies/r113_population_coefficient/summary.json)。唯一冻结 edf3c7f；报告重排 c434c19。
- [r112审计](../studies/r113_population_coefficient/executed/prior_closeout_audit.json)、[历史快照](../studies/r113_population_coefficient/executed/prior_snapshot.json)、[执行审计](../studies/r113_population_coefficient/executed/execution_audit.json)、[首cell核验](../studies/r113_population_coefficient/executed/first_cell_verification.json)。
- [独立数值核验](../studies/r113_population_coefficient/executed/independent_verification.json)、[独立审查](../studies/r113_population_coefficient/executed/peer_review.json)、[恢复不覆盖](../studies/r113_population_coefficient/executed/resume_verification.json)、[点图](../figs/r113_population_coefficient.svg)。
