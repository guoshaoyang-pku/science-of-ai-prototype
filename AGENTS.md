# 工作约定

先读 docs/HANDOFF.md 与 docs/RESEARCH_DIRECTIONS.md。当前是 parked 快照，task/inbox/next_question 是历史材料或候选，不能自动视为新预算。

- 可并行任务使用 subagents，明确文件范围；只在本仓库提交。
- 保护原 ArchitectureIQ RL 代码、trainer、checkpoint、历史 run 和远程进程，不得修改或操作。
- 新实验写 AIQ_KB_DATA_ROOT/runs。v2 快照只读，运行 supervisor 先用 tools/prepare_science_run.py 创建外部独立研究 repo。
- 不因交接启动收费模型、solver 或研究轮；不读取最终 benchmark test，不重跑成功保存的解题。
- 凭据只在执行时读取 AIQ_KB_KEYS，不输出、复制、提交。CLI home、会话历史和模型原始记录不得入库。
- 题库引用 external/ArchitectureIQ 的固定 main commit；显式选 release 并核对 group/seed/variant 与历史暴露。新 run/seed 不是新数据。
- 一轮一个可检验问题；预注册 commit 早于实验，数值必须能从保存产物复算。
- 一条结论 = 一句主张 + 一句数字证据 + 范围词；显示短数，精度留 summary.json。定义不是发现，失败、反例和分母必须保留。
- 报告顺序：结论 → Formulation → 成立程度 → 方法 → 失败 → 未决 → 证据。解析事实、测量和假设分开。
- 看过的封存条件转 development；任务数、机理文字和字段完整不等于突破。
- Solver 输入冻结，答案只在发现阶段揭示；commit 保留 origin_epoch/applied_epoch 和 add/merge/revise/delete 谱系。
- 旧 soa_async_v1 和 publisher 保留原绝对路径；只用 inspect 或只读 renderer，不操作旧 Loop、Jobs、publish_once。
- 不增加主页 block，不重放 monitor queue。旧 v2 publisher 的 main 已禁用，避免覆盖最新 D2。

验证：PYTHONPATH=src python -m pytest -q。交接检查：python tools/verify_handoff.py。

## 模糊方向的文献调研约定

- 用户提出拟贡献的方向时，由 agent 完成支持或相关论文的调研，并在该方向 HTML 顶部提供调研入口。当前仅调研 D1，内容放在现有 D1 文献子页，不建跨方向汇总页，不自动扩展至其他方向。
- 调研按三列呈现：论文链接（标题、作者、年份）／提出的理论／理论的条件。理论列区分证明、近似分析、经验观察与仅相关背景。
- 优先核对原论文；写明已核对的章节或定理、检索日期及未核实部分。条件至少覆盖模型、数据、损失、优化方式和极限或近似假设中与结论有关的项。
- 明确论文结论与本方向的对应及缺口；相关性不等于支持，未检索到不等于新颖。文献调研不自动授权新实验或研究预算。
