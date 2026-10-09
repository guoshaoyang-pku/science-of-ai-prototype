你是报告修复会话（不是研究轮）。禁止：跑实验、新测量、预注册、git commit、写 directions/D7_train_gain_test_harm/ 之外的文件。

任务：把 directions/D7_train_gain_test_harm/report.md 自查并重写到统一标准。

步骤：
1. 读 central/style/REPORT_STYLE.md（结构+语言规则+自查 checklist）与 AGENTS.md 的 report.md 骨架节。
2. 读 directions/D7_train_gain_test_harm/report.md、directions/D7_train_gain_test_harm/findings/ 全部轮次记录、central/kb.json 中 direction 匹配的 claim。
3. 自查：按 checklist 逐项列出不符合点。
4. 重写 report.md：结构 = 结论 → Formulation → 成立程度 → 方法与条件 → 失败与反例 → 未决问题 → 证据；语言 = L4 给人汇报风格（结论先行、短句、行话首次出现即解释、数字代替形容词、无填充）。
5. 铁律：不得改动、新增或删除任何已测数字与科学结论。原报告的每个数字都必须在新文中出现（允许格式归一，如 .9526→0.9526）。拿不准的句子原样保留并放进合适的节。
6. 把自查发现与修复动作写入 directions/D7_train_gain_test_harm/findings/report_style_audit.md。
7. 若报告已完全符合标准，只做最小编辑。

最终回复只写一行：REPAIR_RESULT: <ok|noop|failed> | <一句话>