# r078：第二个新数据 draw 的冻结公式失效复核

本轮只问：固定四个 recipe、init701/719/737、三冻结公式和完整 d3/n32 Gaussian、两 hidden SiLU、bias、float32 初始化转 double、full-batch SGD eta.05/mom0/nodecay/T256，仅将 data_seed 改为预先指定的 260608，T MAE>.70 且相对 E 改善<.08 是否再次出现。全部属于 development，未重拟合，未使用 test 或 sealed OOD。

## 预注册与执行

开始先核验上一批科学收尾：r046 收尾提交 5740f208 的61个归档blob、79个旧文件与36个成功结果文件hash/mtime全部匹配，cb76a73为其祖先。r006/r022/r046各12个成功cell完整，没有孤立中间文件。报告已符合公式区起首的六节结构，无需重排；inbox无新科学指令，已追加规定处理标记。

预注册点预期为T MAE约1.20、改善约−.30，依据已看过的260607开发证据；注册前未生成260608数据。P1只有联合MAE>.70与改善<.08均满足才supported。预注册与四份pinned源码提交67f05e1d7ccb5d43f86fa294566adc7d054eb6d3，至少早于首cell 13.592260 秒。新runner在任何训练前扫描全部计划cell，拒绝无成功metadata的early/NPZ/failure/tmp，也拒绝成功cell的额外或缺失文件；先核验全部已有成功cell合同，再允许新训练。

首cell保存并分析通过后，恢复其余11cell。step32先逐cell落盘E/Q/三公式预测，再续训到step256。12/12cell、4/4recipe完整，累计cell时间1.250168秒；首cell与全部成功cell恢复时均未覆盖。

## 结果与预测判定

P1 refuted：T MAE=0.586472067≤.70，未再现绝对误差>.70；相对E改善-0.152998507<.08仍成立。E/W/T MAE为0.433473560/0.524427395/0.586472067自然log单位，W相对E改善-0.090953835。点预期1.20/−.30未实现，未据结果调整阈值。绝对门槛恢复不代表原联合“MAE≤.70且改善≥.08”通过：后者仍在改善项失败。

| recipe | E MAE | W MAE | T MAE | T 相对 E 改善 [seed 95% t 区间] |
|---|---:|---:|---:|---:|
| product×width8 | 0.900820617 | 0.981808086 | 0.906613866 | -0.005793248 [-0.485006348, 0.473419851] |
| product×width32 | 0.208168069 | 0.357236000 | 0.291868498 | -0.083700428 [-0.632785028, 0.465384172] |
| mixed_sine×width8 | 0.384832322 | 0.467043462 | 0.699104615 | -0.314272293 [-1.367503087, 0.738958501] |
| mixed_sine×width32 | 0.240073233 | 0.291622034 | 0.448301292 | -0.208228058 [-0.820915505, 0.404459389] |

T仅4/12cell误差更小；四recipe平均改善均负，全部T改善区间跨零，不声称逐seed失效。T误差范围0.152497957–1.332692609；product×width8×seed701预测1.619191152、实测L=2.951883761，误差1.332692609为本批最大值。W在mixed_sine×width8的改善区间[−.149328203,−.015094079]完全低于零，其余三个区间跨零。

同seed相对260607的T绝对误差变化均值及95% t区间为：product×8 -0.212809806 [-3.679293654, 3.253674041]；product×32 -0.072822439 [-0.487221843, 0.341576966]；mixed_sine×8 -1.208427822 [-1.943295012, -0.473560632]；mixed_sine×32 -1.221317238 [-2.323247728, -0.119386748]。两mixed_sine区间完全低于零，两个product区间跨零。总体T MAE相对260607减少0.678844326；相对260606则增加0.145283250，四recipe与260606的T误差变化区间均跨零。上述区间只描述三初始化的条件波动，不估计跨draw概率。

## 核验与边界

保存参数独立重建36checkpoint通过：初始化误差0，输出/J最大差8.881784197e-16/1.110223025e-15。核、trace、固定r0/y比值、loss、早期预测时间、三MAE、四recipe配对区间及两批历史draw对照均从保存数组复算。恢复跳过12/12cell，36成功文件hash/mtime不变；136份历史输入hash保持不变，79个旧文件与36个r046成功文件mtime仍匹配旧回执。没有新runtime warning，旧matmul根因未定。收尾审计脚本首个链接正则因转义报re.error；保存错误记录后改为子串检查，仅修复事后审计，未改注册、训练、分析或结果。

唯一实验因子data_seed共同改变输入、clean目标及样本居中/RMS尺度，不能独立归因其中之一；Q负系数只描述冻结关联。T在260607与260608均比E差，但260608的绝对误差门槛不同，不能将两个draw称为相同失效。原draw的两项supported判定、旧mixed_sine×8方向0/3、原T仅7/12改善与四区间跨零、mixed_sine×32变差.014469959、product×8×701误差1.229845961，以及260607两项refuted和最大误差2.263445083均保留。

下一轮可只将data_seed改为预先指定260609，保留四recipe、全部三公式和训练条件，注册更窄的“T相对E改善<0”预测及点预期；先提交新合同与pinned源码再训练。不能沿用本轮联合失效预测为已通过，也不能从两个draw推断成功概率或rt拟合/test/OOD机制。

证据：[预注册](../studies/r078_data_seed_recheck/preregistration.json)、[summary](../studies/r078_data_seed_recheck/summary.json)、[历史审计](../studies/r078_data_seed_recheck/executed/historical_input_audit.json)、[独立参数/J重建](../studies/r078_data_seed_recheck/executed/independent_verification.json)、[260606配对与成功恢复](../studies/r078_data_seed_recheck/executed/saved_evidence_verification.json)、[260607配对核验](../studies/r078_data_seed_recheck/executed/previous_draw_verification.json)、[收尾回执](../studies/r078_data_seed_recheck/executed/final_commit_verification.json)。
