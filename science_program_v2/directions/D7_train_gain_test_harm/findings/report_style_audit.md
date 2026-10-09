# D7 报告 L4 补审

审查日期：2026-10-08。审查范围仅为方向报告的文字、结构及数字保全。没有训练、生成新噪声或重算新科学结果。

## 处理结果

报告按 inbox 指定的六节保留：结论→Formulation→成立程度→方法与条件→未决问题→证据。“失败与反例”仍为“方法与条件”内的三级节。报告第一段直接回答方向问题：训练误差下降不能判定继续训练是否提高 test 风险；晚期风险差 C 需要查看 test，尚无只靠训练信息的前瞻信号。机制比例仍未识别。这些是已保存证据与旧报告边界的直述，没有新增机制结论。

结论节把 test、train、clean、切线、init、SGD、学习率、动量、data/noise、draw、recipe、sigma、bias、hidden、SiLU、float64 和 H/D/G/C 在首次使用处解释。公式或方法节解释 Jacobian、full-batch、cell、hash、finite、einsum、matmul、stderr、df2、mtime、pinned、OOD。长句按单一事实拆开；失败预测与正反例全部保留。

## 数字与公式保全

以修前全文和修后全文作数值 token 多重集比较。允许小数格式与负号归一。修前 791 个数值 token 均在修后保留，缺失数为 0。修后共有 804 个 token；新增 token 仅来自重复说明已有 C 判据时点、符号定义、自由度解释及三组既有 sigma=1 通过率，不是新增测量值。44 行表格逐字相同。Formulation 的三行公式逐字相同。

不修改任何已测数字与科学结论。σ=.5 的 radial×8 反例、noise733131 的1/4失败、sigma=1三条件的4/4、终点与晚期分歧、区间跨零和 matmul 根因未定均保留。全部原证据条目保留。

## 自查

- [x] 六节顺序遵从 inbox；结论在最前。
- [x] 结论直接回答 train 增益能否判断继续训练的 test 损害。
- [x] 各科学结论保留范围与数字；没有跨 draw 概率、机制比例、连续 sigma 阈值或缩放外推。
- [x] 首次行话就地解释，长句拆开。
- [x] 无未量化的程度形容词。
- [x] 结论与成立程度均无文件路径、commit 号或轮次号。
- [x] 原数值 token 全部保留；表格与核心公式逐字保留。
- [x] 失败预测与反例保留，未弱化。
- [x] 没有新增未经测量的 claim。

## 证据

- [方向报告](../report.md)
- [补审前后数值保全记录](report_style_audit.json)
- [报告风格规则](../../../central/style/REPORT_STYLE.md)
- [用户和协调者指令](../inbox.md)
- [半强度 noise draw 独立审查](../studies/r091_noise_draw_half/executed/independent_review.json)

修前报告为当前补审父提交内的方向报告。补审独立 commit 由本轮主代理完成；本审查不混入本轮新训练或结果。
