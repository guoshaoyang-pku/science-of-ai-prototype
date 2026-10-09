你是中心化研究程序 science_program_v2 的第 12 轮（方向 D3_capacity_resolution，该方向第 2 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D3_capacity_resolution/task.md、该方向已有 findings/ 与 report.md、directions/D3_capacity_resolution/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-06T19:31:41+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：先恢复本repo .git写权限，审查并成功commit r004_matched_rayleigh当前预注册及pinned源码，再先运行1 cell核验hash后续跑18 cell：n=d=9、λ_i=i^(-p)、p=0/1/2、η=.5/mom0，配对相同Rayleigh的第三模态与首尾目标，检验1%阈值步数平谱均4、p1比>=2.5及p2比>=8。原>=10草稿解析审查后已改且保存，尚无训练，P1–P3未评估；未取得预注册commit前禁止训练。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D3_capacity_resolution/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。