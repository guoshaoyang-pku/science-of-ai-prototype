# r092：a=0.1 越过固定 λ 的 2% 精度范围

在固定 data31001、对称 Gaussian d4/n128、width64 冻结单层 SiLU、共同加性 bias、训练列居中、seed101–106 条件下，新增 a=0.1、λ=−3/+3 的 12 个 cell 全部越过两项原 2% 判据。Q=R/a⁶ 相对解析极限 F 的误差为 0.045673857405–0.052255236441；与保存 a=0.0125/0.025/0.05 合成的四尺度 spread 为 0.047094464030–0.051410247227。两项原判据均为 0/12 通过。新注册的“误差及 spread 落在 [0.03,0.07]”预测均 supported 12/12。这给出离散尺度上的精度边界，没有推翻旧三小尺度结果或 a→0 的解析极限。

## 条件与事前预测

本轮为程序第92轮、D8第6轮，只检验扩大尺度后的有限精度。固定所有旧输入、W、u、root、居中算法和初始化 seed，仅新增 a=0.1。b=b*+λa²，b*=2.3993572805154675；输入反射时 b 固定。R=mean(even²)/mean(odd²)，Q=R/a⁶。F 保持旧矩与根导数给出的 0.015779531472–0.140811355279，未重拟合。两种 λ 是 2 个新 recipe 条件；六 seed 仅测固定输入下初始化稳健性。12 新 cell、36 只读旧对照、12 同 seed/λ 四尺度配对，0 新训练、0 标签/test/OOD。

候选尺度由已见旧结果选择，预测不是盲发现。把保存 a=0.05 的 signed(Q/F−1) 乘 4，假设有限尺度的首修正随 a²；绝对误差点预期为 0.046486298995–0.051631259199。逐 seed 的 Q、F、方向、区间与四尺度 spread 点预期在预注册前保存。P1 要求全部12个新 cell 的 abs(Q/F−1)∈[0.03,0.07]，λ=−3 为正、λ=+3 为负；P2 要求全部12个四尺度配对的 spread∈[0.03,0.07]。两个区间是事前允许范围，不是置信区间。原 0.02 判据及通过数单独报告。

## 结果与配对证据

| λ | 新 signed(Q/F−1) 六 seed 范围 | Q(0.1)/Q(0.05) 同 seed 范围 | 四尺度 spread 均值及95%初始化t区间 | 新减旧三尺度 spread 均值及95%初始化t区间 |
|---:|---:|---:|---:|---:|
| −3 | 0.050919364708–0.052255236441 | 1.037761578435–1.038846004608 | 0.050824847009 [0.050318307639,0.051331386380] | 0.038820083174 [0.038388275005,0.039251891344] |
| +3 | −0.048208700074–−0.045673857405 | 0.963630607797–0.965547322982 | 0.048331318440 [0.047291054750,0.049371582130] | 0.037024364093 [0.036226459100,0.037822269085] |

点预测的 Q 相对差为 0.000193522199–0.000984261413，仅作描述性误差，未另注册这一精度。两侧偏差方向满足 P1，负侧 Q 随尺度增大、正侧减小。F 的解析极限没有给出有限 a 的精确等式；当前结果只说明原 2% 近似在 a=0.1 不成立。跨格点不计算 pooled CI；表中区间仅描述每 λ 六个权重初始化，不能视为跨输入总体区间。

## 时序、恢复与旧证据

先按指定顺序读取入口、旧 findings/report 和 inbox。inbox要求六节报告顺序，已有独立失败节并入方法支持材料，所有正文与数字保留，实际重排单独提交 108ae25。第一次重排命令因字符串换行语法错误退出；后续提交 7e320f7 只误收已有 supervisor 运行文件，未改变报告或科学量。错误保留于本轮中心日志，重排在任何新测量前完成。

前置审计核验旧科学收尾 fb51e47 的 95 个 blob，旧806保护文件、旧72结果文件及首cell hash/mtime匹配。旧 independent_verification 的 new0/reused36 收据匹配，未运行其旧恢复脚本。旧失效研究的 failed/0claim 保留。四份历史 Cartesian 与414保存 request（72+216+90+36）全部完整；12新增请求零重叠，896旧 study文件受保护。旧36对照只读保存 odd/even，逐位复算 R/Q，并保留原 request/contract/hash；没有调用旧激活、训练或分析脚本。

唯一预注册与全部 pinned 源码/输入/审计提交 c994793d74a711a56ea7622caba8a45b385dd950 早于全部新测量。执行 gate 从预注册文件唯一历史提交取绑定，先检查所有源码字节、输入 hash、旧文件集合、历史重叠与本批全局保存合同，再允许新激活。先 limit1 并独立核验首cell，再复用它新增11cell；12/12 成功、实际累计测量 0.058717791 秒、0训练。

独立 NumPy 新激活重建最大绝对差 3.108624e−15；独立导数多项式与标量矩 F 相对差 8.015810e−14；fsum 对新旧保存数组的 R/Q 相对差 4.440892e−16。新旧配对判据独立复算为12/12。恢复 new0/reused12，24新结果文件与首cell hash/mtime不变；896旧文件hash/mtime不变。独立验证只重建新条件激活，旧对照的激活调用为0。

## 边界与交接

普通 bias 的 a²、精确根的 a⁶、近根混合与正负不对称结论均保留。旧 r048 的18重复测量导致90cell协议失效，不能补算为有效验证；matmul根因未定、首cell打印错误、pooled CI纠正和d00误用审计错误继续归档。没有未经测量的外推。两项新预测均通过，新增2条 measured claim；原2%有限精度失效与新预测 supported 分开登记，不能写成旧小尺度预测被反驳。

下一候选仅 development，保持所有条件只新增未测 a=0.075、λ=±3的12cell，检验原两项2%判据是否仍失败。先核验本轮 final_commit_verification 与 independent_verification、全部历史请求，再给新数值范围、来源hash和pinned源码并单一commit，匹配后才激活。所有旧成功 cell 直接读取原数组；不重新激活或覆盖，不启动旧失效脚本。当前只有离散尺度成功/失败，不能声称连续临界尺度、跨seed概率、目标相关核、训练或干预收益、随机bias、非对称输入、多层、learned hidden、CE、小批量或OOD。

## 证据

- [预注册](../studies/r092_scale01_boundary/preregistration.json)、[非盲数值预测](../studies/r092_scale01_boundary/executed/numeric_forecasts.json)、[只读配对对照](../studies/r092_scale01_boundary/executed/readonly_controls.json)、[summary](../studies/r092_scale01_boundary/summary.json)。
- [旧收尾核验](../studies/r092_scale01_boundary/executed/prior_closeout_audit.json)、[全部历史条件审计](../studies/r092_scale01_boundary/executed/condition_audit.json)、[首cell核验](../studies/r092_scale01_boundary/executed/first_cell_audit.json)、[独立与恢复核验](../studies/r092_scale01_boundary/executed/independent_verification.json)。
- [自包含测量](../studies/r092_scale01_boundary/executed/run.py)、[分析](../studies/r092_scale01_boundary/analysis.py)、[独立核验源码](../studies/r092_scale01_boundary/executed/verify.py)、[收尾提交核验](../studies/r092_scale01_boundary/executed/final_commit_verification.json)。
