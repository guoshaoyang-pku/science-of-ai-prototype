你是中心化研究程序 science_program_v2 的第 19 轮（方向 D2_overfitting_u_curve，该方向第 3 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D2_overfitting_u_curve/task.md、该方向已有 findings/ 与 report.md、directions/D2_overfitting_u_curve/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-06T21:38:16+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：先核验r011产物与中心更新的收尾commit（沙箱拒写.git；冻结合同d5fae2c早于训练，60/60 cell成功，不重跑或覆盖）。原P2已refuted（40/44等比配对通过），P3 supported（57/60留出预测通过），均development、非train-only早停。下一小问题建议：仅改变clean标签归一化，保留r003 n32/.25与n64/.5四seed的原始输入、W、特征，将train mean/RMS改为共享解析population mean=0/RMS=√1.25，检查比值128配对反例是否保留；先写新数值预测与pinned源码并commit，取得匹配提交后再训练。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D2_overfitting_u_curve/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。