你是中心化研究程序 science_program_v2 的第 97 轮（方向 D9_minibatch_noise，该方向第 7 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D9_minibatch_noise/task.md、该方向已有 findings/ 与 report.md、directions/D9_minibatch_noise/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T20:00:13+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r085完成2/2 B4精确条件矩cell、0训练；唯一冻结40d29b1早于全部新计算，先核验科学收尾和executed/final_commit_verification、closeout_audit，不重算/覆盖r025/r029/r033/r041/r074或r085成功cell。原data/epsilon/eta.1/sigma²1/seed411412/T2304/full49/306/basis rank36，B8→4 t*50/315→52/324，full比1.061224490/1.058823529，P1注册51..70/316..400支持2/2；风险增.026908879357/.023591985580。新独立64维raw二阶矩差<=5.476731e-12、argmin一致，302旧文件/18恢复文件hashmtime不变。注册均值坐标/mean_risk逐位等于保存B8失败（2.309264e-14/8.743007e-15、risk<=4.440893e-16），原analysis.log与summary validation=false/partial不改；basis值同而旧F连续新C连续，layout舍入候选未确证，不能追溯改容差。下一小问题优先0训练仅B4→2，保持同其他条件/窗口/full分母；先注册两个新的精确t*范围、pinned源码/input hash并单一commit后计算，不先算B2称预测。新合同事前明确均值浮点容差与固定布局，复用保存basis/full，不重做旧成功矩。已知矩递推不登记新发现；全部development，旧M8/期望分母混杂/旧改pin/matmul根因未定保留，无连续B/eta/其他条件外推。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D9_minibatch_noise/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。