你是中心化研究程序 science_program_v2 的第 107 轮（方向 D5_ln_geometry，该方向第 7 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D5_ln_geometry/task.md、该方向已有 findings/ 与 report.md、directions/D5_ln_geometry/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T23:40:50+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r089完成20/20 LN affine初始化Jacobian结果（首次调用保存1个，恢复调用新增19个）、0新训练；先核验科学收尾与executed/final_commit_verification；预注册99a9ed1早于所有cell，独立核重构、eigh传播、VJP及初始预测核验通过。不得重测/覆盖r013/r021/r045/r077旧结果或r089成功cell。下一小问题仅development：固定共享Linear+LN affine初始化核，只把全批量递推换成预先固定的batch64小批量递推，在eta=.001/a=3/T=256下方向匹配是否改善？先定义批次顺序与每批残差更新式，核对是否能从保存的seed和训练源码重建旧顺序；若不能则使用新声明的确定顺序，不称复现实际训练批次。预注册数值阈值、旧核/train预测hash与pinned源码单一commit后再计算，不训练。r089中真实train ΔC 20/20为负（−.288059215～−.027001544），共享Linear+LN affine固定核20/20为正（+.014690203～+.043216926），匹配0/20，P1 refuted；最大ηλ=.001330483。冻结verify.py存在读取NPZ缺失kernel_pin字段的错误，冻结字节已恢复并保留，不报告原脚本通过；独立核验通过。独立核重构误差0、eigh传播差4.662937e−15、VJP相对误差8.439118e−7。这里只区分固定核的小批量算子，不识别核漂移、weight decay、非线性训练/test动力学、因果或sealed OOD；旧失败与未定matmul警告保留。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D5_ln_geometry/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。