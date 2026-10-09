你是中心化研究程序 science_program_v2 的第 37 轮（方向 D1_u_effective_time，该方向第 4 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D1_u_effective_time/task.md、该方向已有 findings/ 与 report.md、directions/D1_u_effective_time/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T03:47:33+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r017已完成48/48cell；不要重跑或覆盖旧r009的60cell及新r017成功cell，先核验收尾commit。P1/P2各3/3通过，但alpha=.01/.011仅描述，不能把alpha>q当充分条件；原P4 6/10失败保留，alpha=.1/.5启动仍比稳态更差。下一小问题：保留n=d=2、eta=.1、L0=1、lambda_slow=.001、alpha=.001/.005/.009/.02/.1/.5、mom0/.9、T40000与seed0/1/2，仅将lambda_fast从.5降至.05，检验低能量3条件的两种换算误差是否仍>=50%，与r017同alpha/seed配对。先根据已知单模态证据写明确新数值预测及pinned源码并commit，未取得匹配提交不训练。仅development，不独立归因Rayleigh/梯度尺度，不外推其他q或Adam。r009真正收尾提交为665d137；旧last_round_result.final_commit=b05588b错误指向D6日志，审计在r017 executed/r009_closeout_audit.json。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D1_u_effective_time/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。