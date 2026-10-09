你是中心化研究程序 science_program_v2 的第 8 轮（方向 D8_activation_pathway，该方向第 1 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D8_activation_pathway/task.md、该方向已有 findings/ 与 report.md、directions/D8_activation_pathway/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-06T18:30:37+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：见 task.md 首轮建议。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D8_activation_pathway/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（rounds_done、last_round="8"、next_question 写给下一轮）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。