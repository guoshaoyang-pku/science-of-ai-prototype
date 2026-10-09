你是中心化研究程序 science_program_v2 的第 41 轮（方向 D9_minibatch_noise，该方向第 4 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D9_minibatch_noise/task.md、该方向已有 findings/ 与 report.md、directions/D9_minibatch_noise/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T04:58:10+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r033完成两个M64补足cell、112新轨迹+16保留、0新full；冻结b046352早于训练。seed411/412同标签57/49=1.163265、235/306=.767974，CI[.938776,1.489796]/[.660131,1.415033]，宽度旧值.112033/.277978，P1/P2/P3 supported。先核验本轮收尾和final_commit_verification，不覆盖旧8/64轨迹及241个旧文件；旧真实科学收尾ea1a3e8已审计、原final回执缺失。下一小问题：固定原data/epsilon、eta=.1/B8/σ²1、seed411/412、T2304与同标签full分母49/306，先预注册独立64条batch流块r64..127，预测各点仍在[.5,2]且95%条件bootstrap上界<2，再pinned源码commit后训练；与当前r0..63独立块配对描述，不替换/合并追溯改判旧结果。仅development，不改变窗口/epsilon，不外推精确batch期望或连续eta；保留r025/r029 P1/P2失败、期望分母混杂及matmul警告根因未定。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D9_minibatch_noise/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。