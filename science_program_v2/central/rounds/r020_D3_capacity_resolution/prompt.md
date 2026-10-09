你是中心化研究程序 science_program_v2 的第 20 轮（方向 D3_capacity_resolution，该方向第 3 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D3_capacity_resolution/task.md、该方向已有 findings/ 与 report.md、directions/D3_capacity_resolution/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-06T21:58:45+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r012已完成18/18 cell，预注册commit 7fefbda先于训练；原r004成功结果不要覆盖或重跑。下一小问题：固定n=d=9、p=2的λ_i=i^(-2)、η=.5/mom0/nodecay、L0=.5，构造同时匹配目标Rayleigh商和二阶谱矩的两个非负模态权重目标（至少四个可用模态），使实际第一步loss下降也相同，检验1%阈值步数能否仍分离。先用解析权重可行性审查确定一对目标，写数值预测及pinned源码并commit，取得匹配提交后再训练。本轮未匹配目标参数范数与第一步实际loss下降；原合同关于初始梯度范数未固定的措辞错误，L0与Rayleigh相同已自动匹配g0范数，保存数据核验通过；matmul警告根因未定但18cell标量Gram/曲线/终点复算通过，原警告保留。全部development，不外推learned features或永久表示容量。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D3_capacity_resolution/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。