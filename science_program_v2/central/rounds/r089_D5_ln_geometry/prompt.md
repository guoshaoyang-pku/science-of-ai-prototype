你是中心化研究程序 science_program_v2 的第 89 轮（方向 D5_ln_geometry，该方向第 6 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D5_ln_geometry/task.md、该方向已有 findings/ 与 report.md、directions/D5_ln_geometry/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T15:04:36+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r077完成40/40初始化K测量、0新训练，预注册95663c6早于测量；先核验科学收尾与executed/final_commit_verification，不重跑覆盖旧r013120/r02140/r04540或新r07740成功cell。共享input/三个hidden Linear weight/bias排除head/LN affine，float32逐样本J转float64 K=JJᵀ/256；eta.001/a3/A=I−.002K/t1与256。真实train20/20 ΔC<0（−.288059215～−.027001544），fixed t1/t256均20/20>0，方向匹配均0/20、改善0，P1/P2 refuted；最大eta lambda=.001323758，独立VJP/JVP/eigh/原始残差与444历史/80新结果hashmtime通过，恢复0新测量。下一小问题仅development：同数据/eta/a/T，只加入原训练可更新的LN affine初始化核项，复用保存共享Linear K与全部noLN K，只测20个新LN affine Jacobian，不重测旧成功核、不训练；先明确Ktotal=KLinear+Jaff Jaffᵀ/n及float64传播，冻结新数值预测、旧核/train预测hash与pinned源码commit后才测量。只区分参数子集遗漏，不能自动归因核漂移/小批量/decay，train/test分开；fixed-J fullbatch nodecay解析不是实际训练或test机制，无因果/sealed OOD。旧Gram/Rayleigh/coupling失败、初始化κ60/60更低和训练仅7/60下降以及未定matmul警告保留。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D5_ln_geometry/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。