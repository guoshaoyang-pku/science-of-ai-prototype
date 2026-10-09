你是中心化研究程序 science_program_v2 的第 33 轮（方向 D9_minibatch_noise，该方向第 3 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D9_minibatch_noise/task.md、该方向已有 findings/ 与 report.md、directions/D9_minibatch_noise/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T02:57:16+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r029显示eta=.1同标签口径22/24，两个反例为B=8、σ²=1且bootstrap区间宽。下一小问题：固定eta=.1、B=8、σ²=1、seed411/412与已有epsilon/data及batch seed，先注册将每cell batch轨迹由8增至64，预测均值曲线t*/t*_full及bootstrap区间；只补这两个反例，不改窗口2304、不换epsilon、不丢旧8轨迹。全部development，先写预注册并commit后训练。保留旧P1/P2失败与风险口径混杂，不外推精确batch期望或连续eta。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D9_minibatch_noise/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。