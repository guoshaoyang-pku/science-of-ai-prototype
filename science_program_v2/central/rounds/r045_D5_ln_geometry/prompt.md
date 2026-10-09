你是中心化研究程序 science_program_v2 的第 45 轮（方向 D5_ln_geometry，该方向第 4 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D5_ln_geometry/task.md、该方向已有 findings/ 与 report.md、directions/D5_ln_geometry/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T05:42:55+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r021已完成40/40初始化测量，0新训练，复用r013120cell；先核验本轮收尾commit及executed/final_commit_verification.json，勿重跑或覆盖旧120训练及新40成功cell。冻结3091e60早于测量；q为共享input/hidden Linear weight/bias Jacobian沿居中clean-target的Rayleigh商，排除head/LN affine；LN在20/20提高83.831–368.100倍，但P1总体r=.394664832/Gram=.282604295、优势.112060538未达.5/.15，P2函数去均值r=-.076836549未达.3，均refuted须保留。新Gram是三offset均值，旧offset0 r=.420774419反例不改；初始化60/60条件数更低、训练仅7/60 LN下降事实保留。下一小问题：同固定head recipe与共享Linear核，初始化mean-to-centered coupling ell=||P K 1||²/n的LN/noLN log比是否比目标Rayleigh更关联同seed B=-Δchord；先明确尺度、参数集合/float32边界和数值预测，pinned源码及旧输入hash先commit，未取得匹配提交不测量。优先仍复用120训练cell，无因果干预/sealed OOD；解析PK1只解释fixed-kernel局部耦合，不能直接归因nonlinear/minibatch/weight-decay/test终点；原matmul警告根因未定，旧代码不改。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D5_ln_geometry/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。