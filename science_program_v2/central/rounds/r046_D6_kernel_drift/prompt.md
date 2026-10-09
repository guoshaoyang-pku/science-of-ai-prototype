你是中心化研究程序 science_program_v2 的第 46 轮（方向 D6_kernel_drift，该方向第 4 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D6_kernel_drift/task.md、该方向已有 findings/ 与 report.md、directions/D6_kernel_drift/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T05:56:17+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r022已完成12/12cell，预注册f6dce72早于训练；先核验本轮收尾commit和final_commit_verification.json，勿覆盖旧r00612cell与新r02212cell。E/W/T冻结新seed MAE=.550345073/.498007947/.441188817，T改善.109156256达.70/.08，W改善.052337126<.08，P1/P2 supported；但T仅7/12seed更好、4recipe改善CI皆跨零、mixed_sine×width32均值变差.014469959、product×8×701绝对误差1.229845961须保留。旧方向反例mixed_sine×8为0/3不可改；旧真正科学收尾3dd7adf，03a3741只日志。下一小问题建议：同4recipe与seed701/719/737、d3/n32、归一化/架构/SGDeta.05/mom0/nodecay/T256和全部三公式，只换预先指定的新data_seed，检验冻结T MAE<=.70且比E改善>=.08能否保持；先写新development数值预测和pinned源码commit，匹配提交后训练。不重拟合系数，不当OOD/因果/rt拟合/test外推。26旧输入hash不变，新einsum无旧matmul警告，旧根因未定。源码review提醒孤立early恢复当前可重复未成功轨迹，必须审计后再恢复；成功cell不可覆盖。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D6_kernel_drift/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。