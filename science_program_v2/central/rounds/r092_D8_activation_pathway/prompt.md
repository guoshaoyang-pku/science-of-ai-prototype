你是中心化研究程序 science_program_v2 的第 92 轮（方向 D8_activation_pathway，该方向第 6 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D8_activation_pathway/task.md、该方向已有 findings/ 与 report.md、directions/D8_activation_pathway/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T18:40:37+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r080完成36/36新lambda−3/+3×a.0125/.025/.05×seed101–106测量，预注册/pinned源码e3239cd先于全部激活；P1supported12/12，spread.011019256654–.012094422622；P2supported36/36，F误差最大.012907814800，F由已见旧u矩解析给出.015779531472–.140811355279，非盲预测且未拟合。先核验本轮final_commit_verification与independent_verification，不重跑/覆盖旧r01672/r024216/失效r04890或新r08036成功cell；806旧文件与72新结果hashmtime、首cell及new0/reused36恢复通过。r048重复18lambda0失效/0claim、旧matmul/打印/pooled CI与d00误用记录保留；b0/1 a²、b* a⁶与r024混合不对称结论保留。下一小问题仅development建议保持同input/W/居中/root/seeds，只加未测a=.1和lambda±3共12新cell，用保存三小尺度Q/F作只读配对对照，检验是否越过固定2% F或折叠判据；注册前核查全部历史Cartesian/request，写新数值区间、非盲来源、pinned源码与输入hash并commit，匹配后才激活。任何旧重叠必须直接读取/复制保存数组与原合同/hash，禁止再激活。无新训练/标签/test/OOD/连续临界或性能外推。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D8_activation_pathway/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。