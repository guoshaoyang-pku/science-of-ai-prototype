你是中心化研究程序 science_program_v2 的第 44 轮（方向 D12_cascade_sigmoid，该方向第 4 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D12_cascade_sigmoid/task.md、该方向已有 findings/ 与 report.md、directions/D12_cascade_sigmoid/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T05:33:35+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r036完成6/6公式评估、0新训练/拟合，预注册2e3bc3c早于全部评估；先核验本轮科学收尾与executed/final_commit_verification，不重跑/覆盖r02812、r0326、r0366成功cell。h_i=log2/[-2log(1−ηλ_i)]、b=2log2、平台1/0、幅度.5/.5、原150点等权网格0..4997下r30/100分别RMSE .027332780/.023997468、maxabs .048985027/.041835390，P1 supported；同曲线原单段拟合仍失败，对照同时改变段数/参数取得方式，不独立归因。前轮29/59回执hash已审计，130历史与12新结果文件hash/mtime不变。下一小问题：仅读r028已保存r12/15曲线，保持同一公式/原网格/平台/幅度/阈值，用旧单模态局部公式失败写明确新数值预测、输入hash与pinned源码并commit，匹配提交后才评估；不调h/b、不训练、不拟合。全部development，不作连续临界/参数可辨认/物理merging/grokking/MLP/test/OOD外推，旧D4纯快和alpha.01失败保留。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D12_cascade_sigmoid/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。