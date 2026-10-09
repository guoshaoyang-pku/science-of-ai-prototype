# r051：噪声方差减半后，最低点反例仍保留

仅检验一个问题：完全保留 population 标签归一化、输入、W、特征、epsilon、η=.3/mom0/nodecay 和 T=16384，仅把 n32/n64 的 σ² 从 .25/.5 同时减半到 .125/.25，seed414 的等 n/σ²=256 最低点比值是否仍在 (2,4]？完成 2 个 recipe×四 seed，共 8/8 cell。最低点为 87/324 步，比值 3.724137931；P2 supported。全部为 development。

## 先读、先核验、先提交

按序读取 GOAL、AGENTS、中心状态、task、全部 findings/report；无 inbox，无需处理标记。报告已符合六节骨架。旧 218 个 r003/r019 文件与当前 HEAD 一致，连同 r039 共 229 个旧文件已记录 hash/mtime；r039 收尾 1be2873 的 15 个对象匹配。失效记录、零训练与执行锁保留，未运行失效草稿或重跑旧 60/8 个成功 cell。新合同复核 r019 基线 n32→.25、n64→.5，九项控制数组与 76 个输入 pin。

预注册与新执行/分析源码于 07:03:37+08:00 提交为 ba838844d8f256f8476a1c6b4268cdb4e868eb99。07:03:47+08:00 核验 HEAD 字节与源码/输入 hash，之后才执行首 cell。首 cell 分析和恢复核验通过后完成余下七个；首成功 cell hash/mtime 未改。训练与保存累计 9.770942 秒，运行和分析 stderr 为空，固定源码未改。

准备阶段的 inline 指令先出现换行转义语法错误；随后审计误含未跟踪 __pycache__、又把空 results 目录误判为已有结果。均在训练前纠正并记录，未越过写入边界、未运行旧草稿、未改变科学合同。

## 预测来源与配对结果

数值不是盲预测：上一失效轮注册前已由已保存 development 的 signal_bias + 新 σ²×variance_unit 算出四 seed 的 331/278、313/383、230/212、87/324，本轮预注册完整披露。新实际 GD 与独立复算检验该受控条件；不把已知线性分解登记为新理论。P1 是执行核验，8/8 supported，不登记科学 claim。

| seed | 基线 t*：n32/.25 / n64/.5 | 减半 t*：n32/.125 / n64/.25 | 基线比值 | 减半比值 | 配对 log₂ 改变量 |
|---|---:|---:|---:|---:|---:|
| 411 | 144 / 152 | 331 / 278 | 1.055556 | .839879 | −.329749 |
| 412 | 184 / 185 | 313 / 383 | 1.005435 | 1.223642 | .283362 |
| 413 | 129 / 141 | 230 / 212 | 1.093023 | .921739 | −.245894 |
| 414 | 62 / 201 | 87 / 324 | 3.241935 | 3.724138 | .200051 |

最低点均为 87–383 步内部；8/8 满足终点风险比最低点高至少 .05，差值范围 .051388874–.177381316。各 cell 最低点延后 1.403226–2.298611 倍。新比值范围 .839879–3.724138，3/4 落在 [.5,2]；一组 recipe 配对和四 seed 重复，不是四组独立 recipe。只为 seed414 注册科学预测，其余 seed 完整描述。

## 解释与边界

噪声方差减半使这八个条件的最低点均延后，但未恢复等 n/σ² 的严格折叠。seed414 两端延后幅度不同，超两倍反例从 3.241935 变为 3.724138。因每个 n/seed 的九项数据固定，可把本条件内的变化归于噪声方差减半；跨 n 仍改变原始输入抽样、经验谱与目标投影，未识别它们的独立作用。

旧 P2 refuted 40/44、P3 supported 57/60、population seed414 201/62=3.241935 与旧 NumPy bool 错误记录均保留。oracle 使用 clean train/audit 标签，不是 train-only 早停；不外推 learned hidden、Adam、小批量、CE、其他 width、连续噪声阈值、无限训练或 sealed OOD。

## 复算与交接

保存风险等于 signal_bias+σ²×variance_unit；新旧 signal_bias/variance_unit 全步逐字节相同。谱信号误差最大 1.088019e−14，直接最终信号风险误差最大 4.690692e−15；train MSE 最大相邻增量 −2.053337e−9。独立保存证据核验重新 eigh，并以 log1p/expm1 闭式响应复算；全步信号/单位方差/期望风险误差分别 ≤9.769963e−15/5.235812e−13/7.038814e−14，显式终点风险误差 ≤6.566970e−14，三 head 闭式误差 ≤4.092283e−12，noisy=clean+noise 误差 ≤5.417889e−14，均低于 1e−9。229 个旧文件和 33 个新成功文件 hash/mtime 未改；再次恢复为 0 新 cell，不训练、也不调用执行器 measure。

下一轮先核验本轮收尾和独立核验，禁止覆盖旧 60/8 和新 8 个成功 cell。候选小问题：复用 n32/seed412 已保存信号与方差曲线，只把 σ²=.125 再减半至 .0625，检验“终点高于最低点至少 .05”的操作性 U 型判据是否仍满足。注册新数值区间、来源 hash 与源码后才计算；无需新训练，不预先算新端点再称盲预测。

## 证据

- [新预注册](../studies/r051_noise_halving/preregistration.json)、[训练前提交核验](../studies/r051_noise_halving/executed/pretraining_commit_verification.json)、[静态核验](../studies/r051_noise_halving/executed/static_validation.json)与[历史只读核验](../studies/r051_noise_halving/executed/prior_closeout_audit.json)。
- [首 cell 运行](../studies/r051_noise_halving/executed/first_cell_run.json)、[首 cell 分析](../studies/r051_noise_halving/executed/first_cell_analysis.json)、[首 cell 核验](../studies/r051_noise_halving/executed/first_cell_verification.json)、[恢复运行](../studies/r051_noise_halving/executed/resume_run.json)与[完整分析](../studies/r051_noise_halving/executed/full_analysis_run.json)。
- [可复算汇总](../studies/r051_noise_halving/summary.json)、[独立核验源码](../studies/r051_noise_halving/executed/independent_verify.py)与[独立核验](../studies/r051_noise_halving/executed/independent_verification.json)。
