# r108：data_seed260610 上冻结早期公式的负收益复核

## 小问题

仅将 data_seed 从 260609 改为预先指定的 260610，固定四个 function×width recipe、初始化 seed 701/719/737、d=3/n=32 Gaussian 输入、两 hidden SiLU、bias、float32 默认 Linear 初始化后转 float64、full-batch SGD η=.05/momentum0/weight_decay0、T=256，以及 data260606 校准的 E/W/T 三个冻结公式，检验冻结 T 相对冻结 E 的配对 MAE 改善 Δ=MAE_E−MAE_T 是否仍小于 0。

## 预注册与审计

预注册点预期为 Δ=-.20，来源是已见 260607/260608/260609 的 −.383048543/−.152998507/−.045043418；260606 为 +.109156256。预期只用于符号检验，不把点值当作幅度预测。唯一预注册及 pinned 源码提交为 f6e9aa1193cd4e7107fe8b80bd892c61e618f5e0，早于首个新 cell 48.375070 秒。训练前审计通过：r006/r022/r046/r078/r090 共 60/60 个旧成功 cell 完整；历史 200 个输入 hash、151 个旧文件 hash/mtime 记录一致；本批无孤立 early/NPZ/failure/tmp/未知或成功 cell 额外文件。

## 结果

12/12 cell、4/4 recipe 完成。E/W/T MAE 分别为 0.506008323/0.842219210/0.535851321，自然 log 单位；T 相对 E 的改善 Δ=-0.029842999，支持预注册 Δ<0。点预期 −.20 与实测相差 +.170157001，不称点幅度实现，也不追加旧 .70/.08 联合判据。T 仅 5/12 cell 误差更小，T 绝对误差范围为 .030577240–.999907087。

四 recipe 的 T 相对 E 改善均值及三初始化 95% t 区间为：product×width8 −.027799487 [−1.530861524, 1.475262549]；product×width32 −.080765284 [−.475478929, .313948362]；mixed_sine×width8 +.205592094 [−.759560449, 1.170744637]；mixed_sine×width32 −.216399317 [−.866375070, .433576436]。四区间均跨零；两个 recipe 均值改善、两个均值变差，因此不把 Δ<0 解释为每个 recipe 或每个 seed 的稳定方向。

相对上一批 260609，T MAE 增加 .050848465；product×width32 的 T 误差变化均值 +.347354694，区间 [−.135000674, .829710062]，其余 recipe 区间也跨零。相对 260606 的基线，T MAE 增加 .094662504；该比较只作描述，不把共同改变的输入、clean 目标和归一化拆分归因。

## 独立核验与边界

独立脚本从 36 个保存 checkpoint 的参数重建输出与 Jacobian，最大误差分别 8.881784e−16/8.881784e−16；初始化参数误差为 0。核、trace、固定 r0/y 比值、早期时间顺序、冻结预测、MAE 与 recipe 区间复算通过。恢复核验显示 36 个结果文件 hash/mtime 未变。结果仅适用于 data260610 的 development 条件，不估计跨 draw 概率，不作输入/目标/归一化独立因果归因，不外推新函数、优化器、LN、深度、移动 rt 拟合、test 或 sealed OOD。

## 失败预测与保留反例

本轮唯一符号预测支持；点预期幅度未实现。此前 mixed_sine×width8 的早期方向 0/3 反例、260606 原 draw 的 T 仅 7/12 更好和四区间跨零、260607 两项收益预测失败、260608 联合失效再现预测被推翻、260609 两 recipe 均值变差与最大误差 .907918090 均保留。旧 NumPy matmul overflow/invalid warning 根因仍未定。

## 证据

- 预注册与 pinned 源码：directions/D6_kernel_drift/studies/r108_data_seed_transfer/preregistration.json、executed/run.py、analysis.py、executed/independent_verification.py、executed/verify_saved.py、executed/audit_historical_inputs.py。
- 结果与复算：summary.json、executed/independent_verification.json、executed/saved_evidence_verification.json、results/。
