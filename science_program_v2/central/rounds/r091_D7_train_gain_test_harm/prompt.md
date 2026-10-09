你是中心化研究程序 science_program_v2 的第 91 轮（方向 D7_train_gain_test_harm，该方向第 6 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D7_train_gain_test_harm/task.md、该方向已有 findings/ 与 report.md、directions/D7_train_gain_test_harm/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T17:03:47+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r079完成12/12 sigma=.5新cell与24轨迹，预注册652ea06早于训练；P1 supported 3/4。固定data7339/noise733123下D(.5)均值mixed_sine×8/.32、radial×8/.32=.125194/.229192/-.046567/.060492，正初始化3/3、3/3、1/3、3/3；radial×8为边界反例，四区间均保留跨零/窄区间。sigma=.5四单元train gain均值=.373420/.326799/.324617/.283633；真实C128→512=.019583/.367057/-.012013/-.100348。D(.5)均低于D(1)但不作线性/二次缩放。193旧文件与24新文件hash/mtime不变，新切线检查点最大4.440892e-15，df2独立区间差4.294343e-12以1e-10核验，恢复0新训练。下一轮可development更换预指定noise draw检验半强度判据；先注册再训练，不外推机制比例/跨draw概率/train-only早停。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D7_train_gain_test_harm/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。