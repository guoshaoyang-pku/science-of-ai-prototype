你是中心化研究程序 science_program_v2 的第 58 轮（方向 D6_kernel_drift，该方向第 5 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D6_kernel_drift/task.md、该方向已有 findings/ 与 report.md、directions/D6_kernel_drift/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T07:36:18+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r046完成12/12 cell，预注册cb76a73早于训练，先核验本轮科学收尾和executed/final_commit_verification.json；勿覆盖旧r00612/r02212和新r04612成功cell。固定三公式、init701/719/737，唯一data_seed260606→260607后E/W/T MAE=.882267851/.790145266/1.265316394；T改善−.383048543未达.70/.08、W改善.092122584≥.08，P1/P2均refuted；T仅2/12更好、四recipe均值均负，mixed_sine×32改善CI[−.951519879,−.125037993]，最大反例mixed_sine×8×737误差2.263445083保留。T跨draw MAE增加.824127577，mixed_sine×32同seed误差变化1.399699913 CI[.850734023,1.948665803]。旧mixed_sine×8方向0/3、旧7/12改善、四CI跨零、mixed_sine×32变差.014469959/product×8×701误差1.229845961保留，不追溯重判旧supported。79历史和36新成功文件hash/mtime不变、36checkpoint参数/J重建通过，孤立early/NPZ/failure/tmp在任何训练前拒绝需审计，旧matmul根因未定。下一小问题建议仅development同4recipe、d3/n32、归一化/架构/bias/float32初始化转double/SGDeta.05/mom0/nodecay/T256、相同3seed和三冻结公式，只改预先指定data_seed260608，先注册T MAE>.70且改善<.08是否再现失效的数值预测和pinned源码commit，匹配后训练；不重拟合、无OOD/因果/rt拟合/test外推。data_seed共同改变输入/clean目标/样本归一化，不独立归因。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D6_kernel_drift/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。