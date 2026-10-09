# r046：冻结早期公式对新数据 draw 的检验

本轮只问：固定四个 recipe、init701/719/737、全部三公式及 d3/n32/归一化/两 hidden SiLU/full-batch SGD eta.05/mom0/nodecay/T256，仅将 data_seed260606 换为预先指定的 260607，T 的 MAE≤.70 且相对 E 改善≥.08 能否保持。全部属于 development，不重拟合，无 test 或 OOD。

训练前核验旧 24 个成功 cell、旧方向科学收尾 3dd7adf，以及原幅度科学收尾 c0adba1 的回执与 50 个归档 blob；03a3741 仅日志。79 份旧 study/findings hash 与 mtime 固定。既有方向 report 在新研究前重排为固定六节，不改旧数字和结论。inbox 仅有已处理标记，无新科学指令。只读子任务完成源代码审查，未训练或写入；新 runner 在任何训练前拒绝孤立 early/NPZ/failure/tmp，分析删除所有校准重拟合。

预注册与四份 pinned 源码同提交 cb76a737365ef6135d8d9c323ef1e092fc257469，至少早于首 cell 12.340233 秒。首 cell 保存并核验后完成其余 11 cell；成功 cell 不覆盖。每 cell 在 step32 先写 E/Q/三公式预测再到 step256；累计 cell 时间 1.352800 秒。12/12 cell、4/4 recipe 完整。

P1 与 P2 均 refuted。E/W/T MAE 为 .882267851/.790145266/1.265316394 自然 log 单位；T 相对 E 改善 −.383048543，联合 .70/.08 两项均不满足；W 改善 .092122584≥.08，推翻“width 改善仍不足 .08”。

| recipe | E MAE | W MAE | T MAE | T 相对 E 改善 [seed 95% t 区间] |
|---|---:|---:|---:|---:|
| product×width8 | 0.896237229 | 0.746317453 | 1.119423672 | -0.223186443 [-1.870694819, 1.424321933] |
| product×width32 | 0.273858499 | 0.368165311 | 0.364690936 | -0.090832437 [-1.440104703, 1.258439828] |
| mixed_sine×width8 | 1.227636082 | 1.355985181 | 1.907532437 | -0.679896355 [-1.463134625, 0.103341914] |
| mixed_sine×width32 | 1.131339593 | 0.690113120 | 1.669618530 | -0.538278936 [-0.951519879, -0.125037993] |

T 只有 2/12 cell 误差更小，绝对误差范围 .187467119–2.263445083。mixed_sine×width8×seed737 的预测为 2.276896410，实测 L=.013451327，最大误差 2.263445083。四 recipe 平均改善都负，mixed_sine×width32 区间完全低于零，其余三个跨零；不能说逐 seed 失效，也不能将整体 W 改善外推到每 recipe。

相对原 draw，T MAE 增加 .824127577。三 seed 配对的新旧 T 误差变化均值/95% t 区间分别为 product×8 .441642218 [−3.128054945,4.011339382]、product×32 .161835173 [−.895298345,1.218968692]、mixed_sine×8 1.293333002 [−.966552951,3.553218955]、mixed_sine×32 1.399699913 [.850734023,1.948665803]；只有最后一个区间不跨零。

新旧所有 cell 初始参数逐位相同；输入 draw 确实改变，标签与样本居中/RMS 同时改变。这是冻结公式对数据 draw 敏感的反例，不独立证明输入几何、目标或归一化机制。三 seed 只测初始化稳健性；只测试一个新 draw，不估计跨 draw 成功概率。Q 负系数不作机制解释，不预测晚期 rt 拟合或 test。

全部 36 checkpoint 用保存参数独立重建输出/J：初始化误差0，输出/J 最大差 8.881784197e−16/1.110223025e−15；核、trace 比值、固定r0/y、loss、三预测、MAE、配对区间和时间顺序复算通过。补充脚本仅事后核验，从两批保存 Jacobian 独立重算旧新误差变化区间，未训练/拟合。12成功cell恢复跳过后36文件hash/mtime不变，旧79文件不变。新计算无旧matmul warning，旧根因未定。

原 mixed_sine×width8 的 0/3 方向反例、原新初始化 T 仅7/12更好、四区间跨零、mixed_sine×width32变差.014469959，以及product×8×701误差1.229845961全部保留，原两项预测仍supported。

下一轮可只更换预先指定 data_seed260608，固定全部三公式和训练 recipe，先注册 T MAE>.70 且改善<.08 的数值预测，再判断失效是否再现，不覆盖任何成功 cell。

证据：[预注册](../studies/r046_data_seed_transfer/preregistration.json)、[summary](../studies/r046_data_seed_transfer/summary.json)、[历史核验](../studies/r046_data_seed_transfer/executed/historical_evidence_verification.json)、[独立参数/J核验](../studies/r046_data_seed_transfer/executed/independent_verification.json)、[配对区间与恢复核验](../studies/r046_data_seed_transfer/executed/saved_evidence_verification.json)、[收尾回执](../studies/r046_data_seed_transfer/executed/final_commit_verification.json)。
