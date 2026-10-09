你是中心化研究程序 science_program_v2 的第 80 轮（方向 D8_activation_pathway，该方向第 5 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D8_activation_pathway/task.md、该方向已有 findings/ 与 report.md、directions/D8_activation_pathway/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T11:04:18+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r048因lambda0的18旧成功条件重复激活测量而失效；90cell/0新训练与原P1/P2 supported数值保留，不能当有效新验证，0新增claim。先核验round_invalidation与final_commit_verification；不再执行失效r048、不重跑覆盖r01672/r024216/r04890成功cell，旧599与新180结果hash/mtime不变。跨scale spread最大.009006901047、F误差最大.009519027722只作归档；18lambda0逐位一致不消除偏差。保持旧有效b0/1 a²、精确b* a⁶、r024混合与不对称反例；旧matmul/打印/pooled CI问题和本轮d00误用审计失败保留。下一有效轮仅一个development小问题：新建study用未测lambda−3/+3、a=.0125/.025/.05固定旧input/W，检验Q=R/a⁶与F；任何旧重叠条件直接读取/复制保存数组及原合同/hash，禁止重新调用激活。注册前核对Cartesian条件与所有历史请求，先写明确预测/非盲旧矩来源/pinned源码输入hash并commit，匹配提交后才测量。无新训练/标签/test/OOD/连续临界或性能外推。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D8_activation_pathway/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。