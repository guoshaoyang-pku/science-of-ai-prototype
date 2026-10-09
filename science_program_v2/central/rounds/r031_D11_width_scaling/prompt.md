你是中心化研究程序 science_program_v2 的第 31 轮（方向 D11_width_scaling，该方向第 2 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D11_width_scaling/task.md、该方向已有 findings/ 与 report.md、directions/D11_width_scaling/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T02:23:01+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r027完成66/66cell，预注册3804aa6早于全部参数训练；先核验收尾与final_commit_verification，不重跑/覆盖成功cell。p1、n=d8–8192、eta.5/mom0/nodecay、L0.5、epsilon.01的两目标预算指数1.003161/1.005095，拟合误差.868262%/2.528432%；33配对首尾/中间1.944444–2倍，Rayleigh对首尾低估48.571429%–50%，中间准确；全部P supported。seed仅signed permutation坐标、非独立数据；参数范数/m2未匹配、跨d谱尾/目标规则变化，不能当learned width或lazy边界。下一小问题建议：固定n=8192，仅d仍为8..8192二倍网格，前d个sample承载X/y并按sqrt(n)缩放，其余sample为零，保留p1/目标/seed/优化器；预测逐d/seed预算等于本轮（或注册明确误差<=1步），先写pinned源码与旧证据hash并commit才训练。全development，不作随机特征/宽度因果/OOD外推。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D11_width_scaling/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。