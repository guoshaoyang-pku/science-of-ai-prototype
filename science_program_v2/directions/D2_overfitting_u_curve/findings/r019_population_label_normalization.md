# r019：共享 population 标签归一化后，最低点反例是否保留

只回答一个问题：保留原 n32/σ²=.25 与 n64/σ²=.5 的四 seed 输入、W、特征和 epsilon，仅把 clean 标签从 train mean/RMS 归一化改为解析 population mean=0/RMS=√1.25，原 seed414 的等 n/σ²=128 反例是否保留？完成 2 个 recipe×四 seed，共 8/8 cell。P2 supported：最低点为 62 与 201 步，比值 3.24193548 落在预注册 (2,4]，反例保留。全部为 development。

## 先审计与先提交

按序读 GOAL、AGENTS、中心状态、task、已有 findings/report；inbox 不存在，无需追加处理标记。现有 report 缺公式区且证据链接散在正文，已在新研究前重排，不改旧数字或结论。

r011 实际收尾为 11d6ba891b251340cb48bf9eb85aba818233ccaf，包含 60 个成功 cell、报告、D2-001/002 与中心更新；149 个旧产物逐字节匹配。冻结合同 d5fae2c 于 17:38:53 提交，首次旧训练在 19:14:34。中心旧记录的 final_commit=null 保留为历史值，实际提交见本次审计，不重跑或覆盖旧结果。

新预注册、执行与分析源码于 21:47:41 提交为 214a8b2104ea2711ec614c2ecb64b5b60dfc90e6。21:48:00 核验三份文件与 HEAD 逐字节匹配及 37 个旧输入文件 hash；首次新训练始于 21:48:40。先完成 n32/seed411 首 cell，再核验并恢复余下七个。成功 cell 未覆盖，训练与保存累计 10.115601 秒，两个运行 receipt 的 stderr 均为空。

## 合同、预测与配对

固定 d4 对称 Gaussian、目标 x₀+0.5x₀x₁、冻结 ReLU width128/scale .03、原训练特征居中/RMS、含 bias 零初始化 head、全批量 GD η=.3/mom0/nodecay、float64，整数步 0..16384 首个全局最低点。审计集 2048 行，是指定解析函数生成的 development 数据，不读 benchmark test。新标签为 raw/√1.25，噪声仍是原 √σ² epsilon，单位为归一化标签。跨 n 原始输入不同；只在每个 n/seed 内识别归一化干预。

P1 是执行核验，不登记科学 claim：要求 8/8 有限、六项控制数组不变、train MSE 增量≤1e−10，以及风险/谱/仿射复算误差≤1e−9。全部通过。P2 是唯一科学数值预测：seed414 的 n64-to-n32 最低点比值在 (2,4] 且两端内部；其余 seed 完整描述，不作为额外预测。

| seed | 原 t*：n32 / n64 | 新 t*：n32 / n64 | 原比值 | 新比值 | 新 log₂ 比值 |
|---|---:|---:|---:|---:|---:|
| 411 | 274 / 137 | 144 / 152 | .500000 | 1.055556 | .078003 |
| 412 | 174 / 190 | 184 / 185 | 1.091954 | 1.005435 | .007820 |
| 413 | 174 / 120 | 129 / 141 | .689655 | 1.093023 | .128324 |
| 414 | 64 / 180 | 62 / 201 | 2.812500 | 3.241935 | 1.696855 |

这是一组 recipe 配对、四 seed 重复。新比值范围 1.005435–3.241935，3/4 在 [.5,2] 内；比值的归一化干预 log₂ 改变量范围 −.119093–1.078003。逐 cell 新/旧 t* 比值为 .525547–1.175。8 个新最低点均为 62–201 步内部，全部满足终点风险高于最低点至少 .05 的原 U 型描述判据；未按 U 型筛选条件。

## 数字复算与执行异常

首 cell 分析的科学计算完成，但 summary 落盘因 numpy.bool_ 不能直接 JSON 序列化而失败。原 traceback 保存，冻结 analysis.py 未修改；新增 run_analysis.py 只把 NumPy scalar 转为 Python scalar，再用 runpy 执行同一分析。源码 hash 保存，首 cell 复核通过后续跑。该入口不改公式、条件、判据或成功结果，完整分析用时 1.411744 秒。

所有 cell 的信号风险从实际 head、谱统计及旧信号曲线的仿射关系独立对照。令 a=old_RMS/√1.25，b=old_mean/√1.25；因原训练特征居中且有 bias，pop 残差为 a×旧残差−b×.4ᵗ。全步仿射风险误差最大 8.74e−15，谱信号误差最大 1.09e−14；旧/新噪声曲线误差最大 6.24e−13，最终 direct head 信号误差最大 4.69e−15。train MSE 最大相邻增量 −5.33e−9。已知线性递推只作为核验。

独立保存证据核验另从最终噪声响应矩阵复算方差和期望风险，不训练、不调用原 measure。全步信号风险、全步噪声方差、显式终点期望风险最大误差分别为 1.29e−14、2.36e−14、1.27e−14。旧 60 cell 的 hash 由收尾审计核对，37 个旧输入 pin 和全部新结果的 receipt/metadata/NPZ/data hash 由新核验器再次复核，详见证据。

## 解释、边界与下一问

共享 population 标签归一化仍保留 seed414 的超两倍反例，样本标签 mean/RMS 差异不足以单独解释它。均值与 RMS 同时改变，不能拆分其作用；噪声保持归一化单位，不能解释为固定 raw 信噪比。经验谱、目标投影和跨 n 原始输入抽样仍可解释剩余差异，本轮没有独立机制干预。

原 P2 40/44 的 refuted 与原 P3 57/60 的 supported 均保留，不因新对照修订。oracle 使用 clean train/audit 标签，不是 train-only 早停；不外推 learned hidden、Adam、小批量、CE、其他 recipe、无限训练或封存 OOD。

下一轮先核验本轮收尾，禁止覆盖原 60 和新 8 个成功 cell。可只选一个新问题：保持 pop 归一化和本次全部输入/特征不变，仅把两 recipe 的 σ² 同时减半到 .125/.25，检验 seed414 的等 n/σ²=256 比值是否仍>2。新条件、数值预测和源码先提交，取得匹配 commit 后才训练。

## 证据

- [r011 收尾审计](../studies/r019_population_label_normalization/executed/r011_closeout_audit.json)。
- [新预注册](../studies/r019_population_label_normalization/preregistration.json)、[预注册提交记录](../studies/r019_population_label_normalization/executed/preregistration_commit_attempt.json)与[训练前核验](../studies/r019_population_label_normalization/executed/pretraining_commit_verification.json)。
- [首 cell 运行](../studies/r019_population_label_normalization/executed/first_cell_run.json)、[原序列化错误](../studies/r019_population_label_normalization/executed/first_cell_verification.json)、[首 cell 核验](../studies/r019_population_label_normalization/executed/first_cell_verification_after_serialization_fix.json)与[恢复运行](../studies/r019_population_label_normalization/executed/resume_run.json)。
- [可复算汇总](../studies/r019_population_label_normalization/summary.json)、[分析入口](../studies/r019_population_label_normalization/executed/run_analysis.py)、[全量分析记录](../studies/r019_population_label_normalization/executed/full_analysis_run.json)与[独立证据核验](../studies/r019_population_label_normalization/executed/independent_verification.json)。
- [完成核验](../studies/r019_population_label_normalization/executed/round019_completion.json)记录中心计数、原 KB 与固定源码未改。
