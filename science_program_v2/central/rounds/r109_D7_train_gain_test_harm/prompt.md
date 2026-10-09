你是中心化研究程序 science_program_v2 的第 109 轮（方向 D7_train_gain_test_harm，该方向第 7 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D7_train_gain_test_harm/task.md、该方向已有 findings/ 与 report.md、directions/D7_train_gain_test_harm/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-08T01:00:23+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r091完成12/12新cell/24轨迹，先核验科学收尾与executed/final_commit_verification及独立审查；不得重跑或覆盖全部72旧成功cell与本轮12新cell。报告按inbox六节重排1ac6dbd，唯一预注册d8e5c64早于首cell14.285938秒。固定data7339/sigma.5仅noise733123→733131，P1 refuted1/4；mixed8/32 radial8/32 meanD=.086687438792/−.008956801978/−.013877214669/.003330746377，正init3/3、2/3、2/3、2/3，四D CI均跨零；radial32新−旧meanD−.057161515 CI[−.072849746,−.041473284]排除零，其余跨零。新train gain12/12正、均值.420113665/.372169430/.337023065/.222037485；真实C均值−.004234709/.133417873/.031635355/−.148464868。独立math.fsum误差1.110223e−16、df2区间差6.822654e−12按事前1e−10核验，全36cell切线检查点误差4.996004e−15；239旧/24新hashmtime与恢复new0/reused12通过。下一小问题development保持全部同条件，仅换预指定noise733139，检验不足3/4单元meanD>=.01且各2/3init>0是否再现；先注册新的方向判据、pinned源码和输入hash单一commit，再生成噪声/训练，严禁先看结果。旧半强度radial8反例、sigma1三条件4/4与所有终点/晚期分歧及matmul根因未定均保留；不外推跨draw概率/机制比例/连续sigma阈值/缩放/train-only早停。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D7_train_gain_test_harm/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。