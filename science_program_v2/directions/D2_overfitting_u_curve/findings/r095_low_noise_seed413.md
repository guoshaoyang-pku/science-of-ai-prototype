# r095：seed413 的回升幅度低于事前区间

本轮仅检验预指定 n32、seed413 的 population 条件：复用已保存 signal_bias 与 variance_unit，把 σ²=.125 降至 .0625，终点风险减最低风险是否仍至少 .05？完成 **1/1 新评价 cell、0 新训练**。差值为 .014188114079254938，低于事前区间 [.015,.045] 的下界 .000811885920745061。P1 refuted，不能因 .05 判据失败而把整个区间预测记为支持。最低点仍在内部，风险回升仍为正。全部 development。

## 先读、报告重排与旧证据

按序读取 GOAL、AGENTS、state、task、全部 findings/report，最后读取 inbox。优先执行 inbox 六节顺序，保留全文、全部数字与结论，把失败与反例归入方法与条件；重排独立提交 fcce095。已追加指定处理时间标记。

先核验上一轮科学收尾 c8a9e3a：12 个 study/findings 对象与当前文件匹配。旧 final_commit_verification.json 缺失，如实登记，不补造旧回执；真实科学提交存在。独立只读审查确认旧 4 份冻结文件、13 项来源 pin、281 项历史 hash/mtime 和 4 项成功结果/summary hash/mtime。旧风险误差、Decimal 与 peer review 一致。未运行失效 r039 草稿，未重跑或覆盖 r003 的 60、r019 的 8、r051 的 8、r083 的 1 个成功 cell。本轮另冻结 300 个历史 study 文件 hash/mtime。

## 预注册与预测来源

新合同与 4 份源码一次提交为 cdea62877315c3db9718a8569b15955509aa7393，绑定该唯一 commit 后才组合新风险。18 项来源 hash 同时匹配文件与该提交。提交早于新评价 37.962684 秒；条件、区间、源码均未改。区间 [.015,.045] 来自旧 .125 差值 .058105292352594984、粗略减半约 .029 和已见 seed412 的 .0625 结果 .02215220201337556。seed413 由上一轮预指定。注册前未读取该 seed 的曲线数值以计算 .0625 整条风险、端点或 argmin。属于 development、非盲发现，不是封存 OOD。

固定 d4 对称 Gaussian、目标 (x₀+.5x₀x₁)/√1.25、冻结 ReLU width128/scale .03、原训练特征居中/RMS、含 bias 零 head、float64、全批量 GD η=.3/mom0/nodecay、audit2048、整数步 0..16384。只改变噪声方差；九项原数据数组固定。一 recipe 配对、一 seed；逐步网格不是独立重复。

## 保存配对结果

| σ² | 首个全局最低点（步） | 最低风险 | 第 16384 步风险 | 终点减最低风险 | .05 判据 |
|---:|---:|---:|---:|---:|---|
| .125（旧保存对照） | 230 | .15135295688426875 | .20945824923686374 | .058105292352594984 | 满足 |
| .0625（新评价） | 433 | .11506551092432435 | .1292536250035793 | .014188114079254938 | 不满足 |

最低点延后 1.882608695652174 倍。差值减少 .043917178273340046，最低风险减少 .0362874459599444，终点风险减少 .08020462423328445。新差值对 .05 的余量为 −.035811885920745065；对事前下界的偏差不是浮点误差。噪声项线性缩放，最低点重新选取；不假定差值精确减半。

## 数值核验与执行记录

保存曲线组合耗时 .025099708 秒，stderr 为空。分析仅从新 NPZ 复算。独立从原 data 重构 K/eigh，以 log1p/expm1 响应复算全步信号/单位方差/风险，最大误差分别为 9.769963e−15/8.726353e−14/1.437739e−14；差值误差 6.467049e−15，argmin 同为 433。60 位 Decimal 的差值为 .014188114079254930677276291817179298959672451019287109375，最大组合舍入误差 4.553649e−17，P1 同样 refuted。数值核验不算新科学 claim。

恢复为 0 新 cell、1 复用 cell、0 训练；三个成功文件和 summary 共四个文件的 hash/mtime 不变。300 个旧文件 hash/mtime 通过。准备阶段出现 inline 换行 SyntaxError、apply_patch JS 引号 SyntaxError，均未执行风险计算。首次 git show 暂时返回路径不存在；git tree 与 cwd 复核确认文件存在，原准备脚本重试通过，根因未定。原日志与记录保留，未改科学合同。

## 结论边界与下一问题

两条预指定 seed412/413 保存曲线均在 .0625 下未达到 .05 回升幅度，且均保留内部最低点和正回升。本轮 P1 区间失败与上一轮区间支持分别保留。两 seed 不能给跨 seed 概率或连续噪声阈值；只作各自旧新噪声配对，不把两 seed 作为独立 recipe。不外推 train-only 早停、谱/输入因果、其他 n/width、learned hidden、Adam、小批量、CE、无限训练或 sealed OOD。旧 40/44 严格折叠失败、57/60 粗略幂律支持、201/62=3.241935、324/87=3.724138、NumPy bool 及失效轮均保留。

下一小问题可继续零训练：只改为预指定 n32/seed411，旧 .125 差值 .1773813164971647；评价 .0625 是否仍达到 .05。先核验本轮科学收尾与最终回执，冻结新数值区间、来源 hash 与源码，一次提交后才组合。该未测新条件未先算端点或最低点。

## 证据

- [预注册](../studies/r095_low_noise_seed413/preregistration.json)、[提交绑定](../studies/r095_low_noise_seed413/executed/execution_binding.json)、[历史审计](../studies/r095_low_noise_seed413/executed/prior_closeout_audit.json)与[独立旧证据审查](../studies/r095_low_noise_seed413/executed/independent_prior_review.json)。
- [summary](../studies/r095_low_noise_seed413/summary.json)、[新结果](../studies/r095_low_noise_seed413/results/n32_seed413_var0.0625.npz)、[metadata](../studies/r095_low_noise_seed413/results/n32_seed413_var0.0625.json)与[receipt](../studies/r095_low_noise_seed413/results/receipt.json)。
- [独立数值复算](../studies/r095_low_noise_seed413/executed/independent_verification.json)、[执行记录](../studies/r095_low_noise_seed413/executed/evaluation_run.json)、[分析记录](../studies/r095_low_noise_seed413/executed/analysis_run.json)与[恢复核验](../studies/r095_low_noise_seed413/executed/resume_verification.json)。
