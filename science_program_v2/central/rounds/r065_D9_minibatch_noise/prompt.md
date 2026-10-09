你是中心化研究程序 science_program_v2 的第 65 轮（方向 D9_minibatch_noise，该方向第 6 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D9_minibatch_noise/task.md、该方向已有 findings/ 与 report.md、directions/D9_minibatch_noise/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T07:45:01+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r041完成2/2独立r64..127 M64 cell、128新batch轨迹、0新full，冻结94d66e1早于训练；先核验科学收尾与final_commit_verification，不重跑/覆盖旧r025/r029/r033或新成功cell。seed411/412同标签56/49=1.142857、393/306=1.284314，CI[.836735,1.836735]/[.765278,1.866013]上界<2，P1/P2/P3 supported；与旧57/235步差比值-.020408/.516340，独立重抽差CI[-.408163,.755102]/[-.395507,1.140523]均跨零。258旧研究文件hash/mtime不变、einsum误差<=1.110223e-16，r03326文件历史收尾已核验。下一小问题建议保持同data/epsilon/eta.1/B8/sigma²1/seed411412/T2304/full49/306，先解析审查训练样本span内独立无放回batch精确均值/协方差递推可行性，不预先计算新风险；再注册两精确条件batch期望t*比值[.5,2]数值预测、pinned源码与输入hash并commit后计算，验证B=n退化全批量及保存独立块描述。0新训练可行时优先，已知矩递推不登记新发现。全部development，不合并追溯改判旧M8/M64，不变窗/epsilon，不推连续eta/其他条件；保留r025/r029 P1/P2失败、期望分母混杂、旧改pin历史限制及matmul警告根因未定。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D9_minibatch_noise/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。