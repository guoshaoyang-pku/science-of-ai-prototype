你是中心化研究程序 science_program_v2 的第 90 轮（方向 D6_kernel_drift，该方向第 6 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D6_kernel_drift/task.md、该方向已有 findings/ 与 report.md、directions/D6_kernel_drift/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T16:42:22+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r078完成12/12cell，预注册67f05e1早于首cell13.592260秒；先核验本轮科学收尾与executed/final_commit_verification，不重跑覆盖r00612/r02212/r04612或新r07812成功cell。仅data260608后E/W/T MAE=.433473560/.524427395/.586472067，T改善−.152998507，W改善−.090953835；注册MAE>.70且改善<.08再现P1 refuted（MAE项不成立/改善项成立），不说原.70/.08联合成功或逐seed失效。T仅4/12更好，四recipe改善负且CI跨零，product×8×701最大误差1.332692609；相对260607 T MAE降.678844326，mixed_sine×8/32变化−1.208427822 CI[−1.943295012,−.473560632]/−1.221317238 CI[−2.323247728,−.119386748]，product CI跨零；相对260606 T MAE增.145283250且四CI跨零。136旧输入hash、旧79/r04636mtime与新36成功hashmtime不变，36checkpoint重建通过，初始化差0。旧方向0/3、原7/12改善/CI跨零/混合32变差.014469959/1.229845961及r046两项refuted/2.263445083保留，matmul根因未定。下一小问题仅development同四recipe、d3/n32、归一化/两hidden SiLU/bias/float32初始化转double/fullbatch SGDeta.05/mom0/nodecay/T256、init701/719/737和三冻结公式，只改预先指定data260609，先注册更窄T相对E改善<0的数值预测/点预期与pinned源码commit，再匹配训练；任何训练前全局扫描全部cell，孤立early/NPZ/failure/tmp需审计，成功cell额外/缺失文件亦拒绝；不重拟合、无OOD/因果/rt拟合/test外推，data共同改变输入/clean目标/归一化不独立归因。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D6_kernel_drift/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。