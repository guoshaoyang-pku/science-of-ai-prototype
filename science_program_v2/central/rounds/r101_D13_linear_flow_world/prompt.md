你是中心化研究程序 science_program_v2 的第 101 轮（方向 D13_linear_flow_world，该方向第 2 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D13_linear_flow_world/task.md、该方向已有 findings/ 与 report.md、directions/D13_linear_flow_world/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T21:38:21+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：固定无skip链的初始化谱与目标奇异值，降低学习率并比较协议A/B偏移；仅在tc对seed稳定后再进入skip与d_eff组合律。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D13_linear_flow_world/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。