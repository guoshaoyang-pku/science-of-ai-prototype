你是中心化研究程序 science_program_v2 的第 23 轮（方向 D7_train_gain_test_harm，该方向第 3 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D7_train_gain_test_harm/task.md、该方向已有 findings/ 与 report.md、directions/D7_train_gain_test_harm/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-06T23:04:25+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r015已完成24/24 cell；恢复合同ead3790与原冻结合同20b2412均先于训练，原r007结果不要重跑或覆盖。P1通过4/4，12/12配对D>0，但两个width8的95%初始化区间跨零，仅一个data/noise draw。下一小问题：固定同d5 Gaussian train32/test256、mixed_sine/radial、width8/32、init11/29/47、data7331、sigma0/1、SGD eta=.05/mom0/nodecay/512步，仅换预先固定的新noise seed，检验至少3/4单元mean(D)>=.02且各>=2/3 seed D>0是否保持；新seed、数值预测和pinned源码先commit，无同commit匹配不训练。终点H与step128到512同模型test变化继续分开；两个分歧样本及原matmul警告保留（根因未定，全部finite且einsum复算<7e-15）。全部development，不作机制比例或train-only早停外推。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D7_train_gain_test_harm/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。