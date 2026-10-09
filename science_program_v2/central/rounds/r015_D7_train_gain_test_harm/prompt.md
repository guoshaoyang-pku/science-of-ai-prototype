你是中心化研究程序 science_program_v2 的第 15 轮（方向 D7_train_gain_test_harm，该方向第 2 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D7_train_gain_test_harm/task.md、该方向已有 findings/ 与 report.md、directions/D7_train_gain_test_harm/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-06T20:31:28+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：先恢复science_program_v2/.git正常写权限，审查并成功commit r007_noise_interaction当前预注册及pinned源码；未取得同commit匹配前禁止训练。随后先跑1 cell并用analysis核验hash，再续跑24 cell总计划：d5 mixed_sine/radial×width8/32、init11/29/47、data7331/noise733107、sigma0/1、SGD η=.05/mom0/nodecay/512步，检验D=(test真实−切线)noise−(test真实−切线)clean是否至少3/4单元mean>=.02且各>=2/3 seed正；另外区分终点test gap与step128到512同模型test变化。全部development，当前P1未评估。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D7_train_gain_test_harm/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。