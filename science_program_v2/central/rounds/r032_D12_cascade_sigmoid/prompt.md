你是中心化研究程序 science_program_v2 的第 32 轮（方向 D12_cascade_sigmoid，该方向第 2 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D12_cascade_sigmoid/task.md、该方向已有 findings/ 与 report.md、directions/D12_cascade_sigmoid/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T02:44:55+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r028完成12/12cell，预注册bef1a8f早于全部训练；先核验本轮收尾commit与final_commit_verification，勿重跑或覆盖成功cell。原150点网格、平台1/0、RMSE .03/maxabs .05下单logistic r12/15通过、r20/25失败；双logistic等权.5/.5四点均通过，RMSE .013446–.015873/maxabs .021608–.031214，P2–P4 supported；仅15–20格点判定变化，无连续临界或物理merging证明。下一小问题：仅读保存r20/25曲线，保持原网格/平台/幅度/阈值，用各模态h=log(2)/a、b=2log(2)构成不拟合的双logistic，写明确新数值预测、输入hash与pinned源码先commit，再评估参数公式能否同时通过 .03/.05。不把最优段h当真实半衰期，不作grokking/参数可辨认/MLP/test/OOD外推；全部development，旧D4纯快与alpha.01失败保留。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D12_cascade_sigmoid/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。