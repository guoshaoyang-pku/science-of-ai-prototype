你是中心化研究程序 science_program_v2 的第 85 轮（方向 D9_minibatch_noise，该方向第 6 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D9_minibatch_noise/task.md、该方向已有 findings/ 与 report.md、directions/D9_minibatch_noise/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T12:35:30+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r074完成2/2精确batch期望cell、0新训练，冻结d8943a9早于矩计算；先核验科学收尾与final_commit_verification，不重跑/覆盖旧r025/r029/r033/r041或本轮成功cell。固定data/epsilon/eta.1/B8/sigma²1/seed411412/T2304/full49/306，训练span rank36，t*50/315，比值1.020408163/1.029411765，P1 supported2/2；独立64维raw二阶矩曲线差<=5.034862e-12，B=n协方差0/full误差<=1.526557e-14；285旧文件hashmtime、19本轮恢复文件不变。两M64块相对精确t* seed411+7/+6、seed412-80/+78，仅分别描述；旧M8失败仍有效。下一小问题优先0新训练，仅B8→4保持其他条件/窗口/epsilon/full分母；先注册两个新精确t*数值范围与pinned源码/输入hash并commit后计算，不预先算B4再称预测，不推连续B/eta或其他条件。已知矩递推不登记新发现，全部development；保留期望分母混杂、旧改pin历史与matmul根因未定。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D9_minibatch_noise/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。