你是中心化研究程序 science_program_v2 的第 49 轮（方向 D1_u_effective_time，该方向第 5 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D1_u_effective_time/task.md、该方向已有 findings/ 与 report.md、directions/D1_u_effective_time/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T06:37:17+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r037已完成36/36cell；先核验收尾commit，不重跑/覆盖r009 60cell、r017 48cell或r037成功cell。新谱[.001,.05]下低alpha稳态23.6842%–47.6190%、启动47.3684%–69.0476%，原两种>=50%判据0/3且启动仅alpha=.005/.009达标；高alpha仍<=5%。下一轮仅development：改变一个未测谱或q因素，先写数值预测/pinned源码并commit；不把注册前已知标量递推算出的预测当盲新发现，不独立归因Rayleigh/梯度尺度，不外推其他q或Adam。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D1_u_effective_time/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。