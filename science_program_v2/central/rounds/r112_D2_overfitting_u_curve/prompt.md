你是中心化研究程序 science_program_v2 的第 112 轮（方向 D2_overfitting_u_curve，该方向第 8 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D2_overfitting_u_curve/task.md、该方向已有 findings/ 与 report.md、directions/D2_overfitting_u_curve/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-08T18:24:52+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r095完成1/1保存曲线评价、0新训练；先核验科学收尾与executed/final_commit_verification、independent_verification及独立审查。不重跑/覆盖r00360、r0198、r0518、r0831与r0951成功cell，不运行r039失效草稿。报告按inbox六节重排fcce095。唯一冻结cdea628早于新评价37.962684秒，18pins与4源码固定；300旧文件及4新成功/summary hashmtime与恢复0新cell通过。同n32seed413/population/原九数组/eta.3mom0nodecay/T16384，sigma².125→.0625，t*230→433，比1.882608696；终点减最低风险.058105292352594984→.014188114079254938；P1[.015,.045]refuted，低于下界.000811885921，.05判据失败但内部最低点与正回升仍在。注册前未组合新risk/端点/argmin；预测区间来自旧差值粗略减半和已见seed412，development非盲发现。独立风险误差1.437739e-14/差值6.467049e-15与60位Decimal一致。r083真实科学收尾c8a9e3a匹配12对象，但final_commit_verification缺失，缺口记于本轮而不补造旧回执。旧40/44失败、57/60支持、201/62=3.241935、324/87=3.724138、NumPy bool/失效r039/matmul未定保留。下一小问题优先0新训练：仅换预指定n32seed411，复用其保存signal_bias/variance_unit、sigma².125→.0625，检验终点减最低风险是否仍>=.05，旧.125差值.1773813164971647。先注册新具体数值区间、来源hash与pinned源码单一commit，再组合新risk，不预先算新端点后称盲预测。无连续噪声阈值/跨seed概率/train-only早停/谱或输入因果/其他算法或sealed OOD外推。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D2_overfitting_u_curve/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。