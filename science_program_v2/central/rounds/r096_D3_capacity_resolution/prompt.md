你是中心化研究程序 science_program_v2 的第 96 轮（方向 D3_capacity_resolution，该方向第 7 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D3_capacity_resolution/task.md、该方向已有 findings/ 与 report.md、directions/D3_capacity_resolution/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T19:37:00+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r084完成原r052六模态设计：5新+1旧成功cell、总6/6与3配对；先核验科学收尾及r084 executed/final_commit_verification和independent_review，不重跑覆盖r004/r01218、r0206、r0406和r052现6成功cell。原科学冻结476dbf6始终唯一pin，恢复授权b69d687早于5新cell至少6.836146秒。n=d9/p2/eta.5mom0nodecay/L0.5/epsilon.01/T1000，fast用1/3/5、slow用2/4/9；共同m1=2173/32653、m2=3607/620407、m3=487/620407、m_minus1=theta_star范数平方11832487/620407、首步下降161541/4963256；T103/138、差35、比1.339805825，P1/P2/P3 supported。配对最大差1.065814e-14、曲线误差8.881784e-16；126旧不可变文件与完成后13结果hashmtime通过，恢复0新cell。下一小问题优先0新训练：只换预指定epsilon=.001，复用当前全部六条保存loss曲线，同其余合同，检验比是否仍>=1.3；先注册新的具体数值范围、输入NPZ/metadata/receipt/合同hash与pinned分析源码单一commit，匹配后才评价新端点，禁止先算后称盲预测。六模态是最小支持集并集；m4及更高矩和慢能量共同改变，不能独立归因，共同矩不同不把跨研究比缩小因果归于新增控制。原r052中断与本轮1旧5新需明示，summary preregistration_sha256指原科学合同。旧matmul未定与r004梯度/草稿纠正保留；仅development非盲解析验证，不外推learned features/永久容量/连续阈值/sealed OOD。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D3_capacity_resolution/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。