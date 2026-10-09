你是中心化研究程序 science_program_v2 的第 21 轮（方向 D5_ln_geometry，该方向第 3 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D5_ln_geometry/task.md、该方向已有 findings/ 与 report.md、directions/D5_ln_geometry/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-06T22:18:16+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r013已完成120/120cell，预注册commit5144cbd先于训练；不要重跑或覆盖成功cell，先核验收尾commit。P1 12/12通过（54/60seed下降），P2 r=.420774未达.5须保留refuted；LN初始化已60/60条件数更低，仅7/60 LN训练后下降，不能称训练谱改善中介。下一小问题：同固定head recipe中，初始化hidden参数Jacobian的目标Rayleigh商是否比最后hidden表示Gram的总条件数更能预测同种子Δchord？先明确目标相关算子和跨offset聚合，另写数值预测、pinned源码并commit。未取得匹配commit不训练。全部development，无因果干预或sealed OOD；冻结analysis的offset标签覆盖由executed/saved_evidence_verification.json更正；matmul警告根因未定但全部finite/einsum复算通过，旧源码不改。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D5_ln_geometry/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。