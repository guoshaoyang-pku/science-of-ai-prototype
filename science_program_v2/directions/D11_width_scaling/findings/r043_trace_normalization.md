# trace 固定为 1 的候选对照：准备阶段失效

本轮只选一个问题：保持固定 n=8192、d=8..8192 二倍网格、中间/首尾目标、三坐标 seed、η=.5/mom0/nodecay、L0=.5 与 ε=.01，仅将 λᵢ改为 i⁻¹/h_d，使 trace=1，检验 T/(d h_d) 是否接近两个渐近系数。既有报告已符合固定骨架，无需改动开头；没有 inbox。

## 失效与处置

只读审查子任务报告它写入并执行了 /tmp/calc.py（实际目录 /private/tmp），随后删除。这违反 GOAL 与 AGENTS 的“只写 science_program_v2 内”硬边界。本轮因此失效；发现后停止新实验，0 新训练、0 新拟合、0 新实验 cell，没有有效预注册实验提交。当前已核实该临时文件不存在。该子任务未回传确切执行时间或命令文本，记录只使用已知的路径、行为和删除确认，不补写细节。

候选合同以 invalid_draft_do_not_execute 状态保存，并明确 execution_allowed=false。executed/run.py 是永久执行锁，不能在后续恢复中运行这份草稿。analysis.py 只审计旧保存证据并生成失效汇总；没有参数更新。工具字符串语法与目录遍历错误均在训练前发生，纠正后不改变旧成功产物。

## 旧证据审计

r027 的 145 个 tracked study 文件与科学收尾 37d1ea6 逐字匹配；原目录没有 final_commit_verification.json，这一缺口保留，r031 的补核也仍有效。r031 的 147 个科学 study 文件与科学收尾 882fd01 相同，现有 final_commit_verification receipt 与补充提交 f143aa3 相同；当前 148 个 tracked study 文件与该补充提交匹配。两研究共 293 个文件的 hash、mtime 与 pinned 清单均匹配。

共 132 个旧 completed cell 的 metadata receipt hash、NPZ hash、request hash 与保存曲线整数阈值复算全部通过。r031 baseline manifest 的 145 个 r027 文件 hash/mtime 未变。现有收尾 receipt 的预注册时序与祖先关系通过；它记录的 153 个 study/handoff 历史快照检查保留，不把已被其它方向更新的中心文件错误要求为当前逐字相同。

r031 的旧 66 配对预算差仍为 [0,0] 步，归一化曲线最大差 4.440892098500626e−16。没有重跑、覆盖或删除旧成功 cell。两个原目录的未追踪 __pycache__ 文件不计入科学证据；审计不导入或运行旧训练脚本。

## 预测状态与交接

草稿 P1（已知递推执行审计）和 P2（trace 归一化误差）均为 not_evaluated，不能标 refuted 或登记新 claim。本轮只更新 KB 审查记录，新增 claim 数为 0。

注册前已知谱算术给出的中间/首尾整数参考依 d 为 49/96、124/245、298/593、698/1393、1601/3196、3609/7213、8036/16065、17705/35403、38678/77348、83893/167777、180859/361710。这些是已知理论算术，非本轮训练测量、非盲预测。草稿保留来源，并保存 T/(d h_d) 的算术复算。

下一有效轮可新建同问题研究，采用全网格相对误差中间≤3%、首尾≤5%，d≥128 两目标各≤.3% 的预注册候选；明确逐维数整数预测、来源与旧冻结 d 幂律对照。实际参数更新源码、分析与旧输入 hash 必须在匹配提交中先冻结，再开始训练；不可恢复失效执行锁。全部 development，非零样本仍 d，不外推独立样本增加、随机特征、learned width 或 lazy↔rich 边界。

## 证据

- [失效记录](../studies/r043_trace_normalization/executed/round_invalidation.json)、[锁定草稿](../studies/r043_trace_normalization/preregistration.json)、[执行锁](../studies/r043_trace_normalization/executed/run.py)。
- [只读分析](../studies/r043_trace_normalization/analysis.py)、[汇总](../studies/r043_trace_normalization/summary.json)、[旧证据核验](../studies/r043_trace_normalization/executed/prior_evidence_audit.json)。
- [中心更新审计](../studies/r043_trace_normalization/executed/central_update_audit.json)、[收尾提交核验](../studies/r043_trace_normalization/executed/final_commit_verification.json)。
