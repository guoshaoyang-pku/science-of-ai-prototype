你是中心化研究程序 science_program_v2 的第 14 轮（方向 D6_kernel_drift，该方向第 2 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D6_kernel_drift/task.md、该方向已有 findings/ 与 report.md、directions/D6_kernel_drift/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-06T20:12:25+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：先恢复science_program_v2/.git写权限，审查并成功commit r006_early_direction当前预注册与pinned源码；此前提交失败的合同副本已保存。提交后先运行1 cell并核验合同/NPZ hash，再恢复12 cell：d3 n32 Gaussian product/mixed_sine×width8/32、3 seeds、SiLU/full-batch SGD η=.05/mom0/nodecay、256步，检验固定r0且trace匹配后的step32/256漂移同向是否达到每unit≥2/3 seed、总≥3/4 unit。未取得匹配预注册commit前禁止训练；本轮0 cell，P1未评估。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D6_kernel_drift/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。