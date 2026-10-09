你是中心化研究程序 science_program_v2 的第 17 轮（方向 D1_u_effective_time，该方向第 3 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D1_u_effective_time/task.md、该方向已有 findings/ 与 report.md、directions/D1_u_effective_time/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-06T21:01:41+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：先核验r009产物与中心更新的收尾commit（当前沙箱拒写.git；原预注册7da7d8a已在训练前提交，不要重跑或覆盖60个成功cell）。新小问题：固定谱、η和初始loss，双模态目标的慢模态能量占比能否预测q=.01持续端点的U换算失效？先写新数值预测并commit；未取得新预注册commit不训练。原P4已在6/10条件失败，须保留，不能用启动公式普适外推。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D1_u_effective_time/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。