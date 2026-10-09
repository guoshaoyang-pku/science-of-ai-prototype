你是报告内容对齐重写会话（不是研究轮）。目标：directions/D3_capacity_resolution/report.md。

先读（按序）：directions/D3_capacity_resolution/task.md 的"用户指定核心问题"节、central/style/REPORT_STYLE.md、AGENTS.md 的骨架与核心问题直答规则、directions/D3_capacity_resolution/findings/ 全部文件、directions/D3_capacity_resolution/studies/ 各 summary.json 的数字（只读，不重跑）。

任务：重写 report.md，使其直接回答用户核心问题：
1. "结论"节第一段直接回答核心问题：能给公式就给公式（分段/列表），公式下一行逐符号一句话定义；随后逐条展开，每条 = 一句主张 + 一句数字证据。
2. 与核心问题无关的发现全部压缩进 support 节（方法与条件/失败与反例/未决问题/证据），不得占据结论区。
3. 人话：一句一事实；行话首次出现即解释；数字代替形容词；外行只读结论节能拿走 90% 图像。
4. 若回答需要新数字（如上升支指数、停点尾部验证），允许对**已保存曲线**做零训练分析：脚本与 summary 存 directions/D3_capacity_resolution/studies/rewrite_20261007/；所得公式必须标注"保存数据事后拟合（development）"。禁止新训练、新 cell、新 seed 实验。
5. 铁律：不改动、不删除任何既有已测数字与结论（可重组、改写表述、补 post-hoc 标注）；结论与成立程度两节不出现文件路径/commit/轮次号。
6. 禁止：写 central/、写其他方向、git commit、改 kb.json/state.json/PROGRESS.md。

自查后写 directions/D3_capacity_resolution/findings/report_rewrite_audit.md（核心问题是什么、原报告哪里跑题、重写如何对齐、人话检验结果）。
最终回复只写一行：REWRITE_RESULT: <ok|failed> | <一句话>