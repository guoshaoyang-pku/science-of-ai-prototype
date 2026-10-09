你是中心化研究程序 science_program_v2 的第 84 轮（方向 D3_capacity_resolution，该方向第 6 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D3_capacity_resolution/task.md、该方向已有 findings/ 与 report.md、directions/D3_capacity_resolution/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T12:22:41+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r040完成6/6cell，预注册503ad59早于训练；先核验科学收尾与final_commit_verification，不重跑/覆盖旧r004/r01218cell、r0206cell和新r0406cell。固定五模态并集1/2/3/4/9，A用1/3/9与B用2/4，共同m1=109/964、m2=19/964、m_minus1=theta_star范数平方=3076/241、L0=.5、首步下降417/7712；T1%=135/68，差67，比1.985294，P1/P2/P3 supported。最大配对差2.131628e-14、谱曲线误差8.881784e-16；保存数组/eigh/标量与旧86文件hash/mtime核验通过。下一小问题建议同n=d9/p2/eta.5mom0nodecay/L0.5/epsilon.01/T1000，预先固定六可用模态1/2/3/4/5/9，先解析审查两个非负目标同时匹配m1/m2/m3/m_minus1的可行性，再写新development数值预测与pinned源码commit，匹配提交后才训练。最少五模态是支持集并集；高阶矩与慢能量仍共同改变，不独立归因；共同矩值不同，不把预算比缩小因果归于范数控制。旧matmul根因未定与r004梯度/草稿措辞纠正保留。解析预测非盲发现，不外推learned features或永久容量。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D3_capacity_resolution/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。