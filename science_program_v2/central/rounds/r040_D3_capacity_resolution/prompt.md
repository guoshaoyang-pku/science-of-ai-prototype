你是中心化研究程序 science_program_v2 的第 40 轮（方向 D3_capacity_resolution，该方向第 4 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D3_capacity_resolution/task.md、该方向已有 findings/ 与 report.md、directions/D3_capacity_resolution/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T04:43:54+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r020已完成6/6cell，先核验收尾commit；预注册f6fc2a9早于训练，原r004/r01218cell和新r0206cell不要覆盖或重跑。A(w1=1/55,w3=54/55)与B(w2=2048/4235,w9=2187/4235)共同m1=7/55,m2=1/33、L0=.5与实际第一步下降=.05984848485，T1%=41/319，比7.780488，P1/P2/P3 supported；保存参数、eigh与标量复算通过。旧matmul警告根因未定保留，本轮einsum无该警告。下一小问题：保持n=d=9、p2谱lambda_i=i^-2、eta=.5/mom0/nodecay、L0=.5、epsilon=.01，采用至少五个可用模态，先解析审查两个非负目标同时匹配m1、m2与sum(w/lambda)=theta_star范数平方的可行性，再写新数值预测和pinned源码并commit，取得匹配提交后再训练。目标参数范数本轮仍不同(487/55 vs2407/55)，高阶矩与慢能量共同变化，不能独立归因。全部development，不外推learned features或永久表示容量；旧r004失败措辞纠正保留。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D3_capacity_resolution/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。