你是中心化研究程序 science_program_v2 的第 102 轮（方向 D13_linear_flow_world，该方向第 3 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D13_linear_flow_world/task.md、该方向已有 findings/ 与 report.md、directions/D13_linear_flow_world/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T22:02:52+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r101完成6/6新训练cell与3旧只读基线；先核验scientific_closeout与executed/independent_verification.json、recovery_verification.json、final_commit_verification.json。不重跑或覆盖r00112cell/r1016cell及9成功拟合；唯一预注册657a4d4早于首新cell21.471142秒。固定L4,width32,128x4Gaussian,正交目标,sigma.8,seeds11/22/33，逐seed初始矩阵跨eta相同；旧更新g=2err/n优化J=4*MSE，eta.02/.005/.00125与T3000/12000/48000，tau60。legacy A/B ratios最细.779080–1.016098、一致3/3，|log ratio|偏移减少80.955–99.455%，P1支持；reference s=eta*step/.02、B tau.02公共网格11点，最细ratio.154985–.341155仍0/3一致，B全tau.12。reference A mid→fine差8.874894/16.817806/5.020817%，P2 refuted2/3；seed CV最细.276924>.25，不进入skip/deff。下一小问题development仅seed22同原初始化/数据/优化J，降eta=.0003125/T192000/tau60，先冻结新数值预测、源码和全部输入hash，再生成/训练，检验reference tau_A相对当前.018598187170700903是否在5%内；不先算后注册。不称local拟合全局最优，不把粗B零变化当真峰收敛。30旧与21新成功文件hashmtime不变，恢复0训练/0拟合；独立einsum72checkpoint/12局部更新误差2.220446e-16/5.551115e-17，拟合RMSE差8.673617e-19，R2差0；matmul告警根因未定保留。旧tc>=1文字与源码>=0合同偏差保留；无OOD/新谱/新sigma/宽度/目标/optimizer/泛化/无限梯度流/有效深度公式外推。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D13_linear_flow_world/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。