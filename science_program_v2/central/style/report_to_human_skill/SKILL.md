---
name: report_to_human
description: 面向人的 4 层汇报体系 v2（源自 Karpathy 2026-10 的输出阶梯）。L1 用 ASD-STE100 的 80% 严格性写散文，L2 用图标和图示展示 settings 与结构，L3 用交互式 HTML panel 作为主阅读面，L4 聊天框只留一句话结论和链接。v2 新增三条硬规则：极简（删掉副标题、KPI 卡、状态栏等读者用不上的元素），术语先解释后使用，panel 套用 Anthropic 配色模板与统一绘图风格；交付前用 scripts/panel_audit.py 自检。默认用中文汇报，术语保留英文。承袭 RL 训练 trial 分析全流程（reward/seqlen/clip 曲线 + 思维链演变抽查，360-2 集群 async GRPO/GSPO run，任何有 log_history.json 的实验）。agent 向人汇报结果、实验分析、进度或发现，或制作报告网页、HTML panel、dashboard 时使用。触发词：跟我汇报、汇报一下、report to human、4层汇报、做个 panel、做个网页汇报、ASD-STE100、STE 风格、HTML panel、settings 图标化、实验分析、trial 分析、reward 曲线、seqlen、截断率、思维链、CoT、checkpoint 对比、涌现、held-out 评测、dashboard、panel。
---

# report-to-human v2：4 层汇报体系

agent 给人看的所有产出，都按 4 层组织。内容放在上面三层。聊天框是最后、最薄的一层。

来源：Karpathy 2026-10-02 的输出阶梯：writing → diagrams → HTML → video（[原帖](https://x.com/karpathy/status/2105819303471976479)）。本 skill 把第 2 层落地为 settings 图标化，把第 4 层从视频换成聊天框入口。

## 三条硬规则（v2，优先于下文一切）

### 1. 极简：读者用不上的，一律删掉

删除测试：对每个元素问一句"删掉它，读者会少知道一件他需要的事吗？"答不上来就删。

默认删除，除非用户明确要：

- 标题下的副标题、tagline、"完整研究报告 · 逐轮原文与……"这类说明行。
- KPI 卡、大数字卡、指标看板、得分徽章。数字写进结论句或图里。
- 节编号（"0 · 实验线谱系"）和每节标题旁的灰色说明小字。
- 状态信息：PID、进程存活、"运行中 / 已归档"标签、快照时间、生成时间、SHA256、文件行数。最多在页脚留一行来源和日期。
- 页面谈论自己："本页是静态快照""阅读范围""如何使用本页"。
- 短页面的目录、跳转链接、"查看 / 下载 Markdown""打印"按钮。
- 所有值都相同的表格列（例如整列都是"归档"）。
- 重复正文的图注，重复图的正文。

首屏（第一个 h2 之前）只有三样：一个用平实语言写结论的标题，一句话结论，一行 settings。不放别的。

每个 h2 写成一个结论句（"回答长度在第 120 步附近快速下降"），不写成话题标签（"长度分析"）。

### 2. 术语：先解释，后使用

读者是懂本领域通用知识、但没参与这个项目的人。通用术语（token、loss、SGD、reward、KL）可以直接用。下面这些必须在第一次出现的地方用平实的话解释：

- 项目自造的概念和缩写（RGI、near/far、强命题、one-third 理论）。
- 公式符号（D、A、Q₂、η、g_B）。第一次出现写成"D（冻结难度：训练开始前的模型在这批数据上的 loss）"。
- 内部代号、run 名、轮次编号（soa_v1、luna_main、R55、e8、kb_0010）。正文用平实的名字（"第一版同步循环实验"），代号只作为等宽小标签放在后面，或者直接不出现。

规则：

- 标题和首屏不出现任何未解释的术语。做不到就换成平实说法。
- 解释紧跟术语，写在同一句的括号或冒号里。不要把解释放到页尾的术语表。
- 一个概念全文只用一个名字（STE 规则 1.11）。选定平实名字后，不再换回代号。
- 公式放在解释之后。先用一句话说清它在算什么，再给式子。

### 3. 好看：套模板，统一风格

- HTML panel 从 `assets/panel-template.html` 开始改，不从零写 CSS。模板用 Anthropic brand-guidelines 的配色和字体：背景 #faf9f5，正文 #141413，主强调色橙 #d97757（主结果），蓝 #6a9bcc 与绿 #788c5d 做对照，灰 #b0aea5 / #e8e6dc 做边框和网格。标题用 Poppins / 苹方，正文用 Lora / 思源宋体。正文单栏，宽 760px。
- HTML 里的 settings 图标用单色线性 SVG（模板里有示例），不用彩色 emoji。彩色 emoji 只在聊天和 markdown 里用。
- 数据图用 `scripts/panel_style.py`：同一套配色，去掉上框和右框，浅网格，无边框图例，主结果固定用橙色。导出 SVG（panel 用）和 2x PNG。图里不写标题，标题由 h2 承担。
- 选图类型和画图细节，按 Orchestra AI-Research-SKILLs 的 academic-plotting 规则：有 step 轴用折线图，N 个方法比 M 项用分组柱状图，不用饼图，突出主结果，超过 5 条线就拆图或换 Okabe-Ito 色盲安全配色。本机装了 research-skills 时，拉取 `academic-plotting/SKILL.md`。

## 语言

- 默认用中文写所有面向人的产出：报告正文、图内文字、HTML panel 界面、聊天消息。
- 如果用户用其他语言提问，就用用户的语言。
- 技术术语、指标名、标识符、代码、路径、命令保留英文原样，不翻译。
- 四层协议和 ASD-STE100 规则按英文规范执行，只改变产出语言。中文同样遵守句式纪律：短句、一句一事、术语一致、主动语态。

## 四层协议

### L1 文字层：ASD-STE100，80% 严格性

- 一个概念一个名字，全文一致。不要同义词轮换。
- 短句。英文描述句 ≤25 词，指令句 ≤20 词。中文一句只讲一个事实，不堆从句。
- 主动语态，主语具体，说清谁做什么。条件在前，动作在后。
- 用平实的词：use / start / before，不用 utilize / commence / prior to。中文不用"赋能、抓手、闭环"。
- 情态词只用 must / can，或者写出精确条件。不堆 should / might。
- 不用分号。每段一个主题，≤6 句。
- 数字、单位、不确定性、例外和限定词全部保留。不为缩短篇幅删事实，宁可拆句。
- 风险写成"注意"块：先写条件，再写后果。
- 完整规则和开源实现清单见 `references/asd-ste100-80.md`。

### L2 图标/图示层：settings 不靠散文交代

- 配置用一行"图标 + 名称 + 值"展示：模型、算法、数据、关键超参、日期。只列读者判断结论需要的项，一般不超过 6 项。
- 结构和流程用图示（Mermaid / SVG），不用成段文字描述。
- 曲线和指标一律出图。

### L3 HTML panel 层：主阅读面

- 项目已有 dashboard 就直接用，并给出链接。
- 没有时，复制 `assets/panel-template.html` 改写：单文件，数据内联，不需要构建，没有外部依赖。文件放在报告旁边，在浏览器里打开预览。
- 结构：首屏（标题、一句话结论、settings 行）→ 每节一个结论句 h2 + 一段话 + 一张图 → 需要时加样本浏览器（例如模型输出并排对照）→ 细节数字折叠进 `<details>` → 页脚一行来源。

### L4 聊天框层：只留入口

- 聊天消息 = 名称 + 一句话结论 + panel 链接 + 报告文件链接。
- 不在聊天里贴报告正文，不在聊天里堆表格和数字清单。

## 交付前自检（必须做）

1. 运行 `python scripts/panel_audit.py <panel.html> --allow <读者已知的通用术语>`。它按阅读顺序列出：多余元素、首屏块数、第一次出现时附近没有解释的术语。
2. 逐条处理：删除多余元素，给术语补解释或换成平实说法。确实不用改的，加进 `--allow`。
3. 重跑到没有你认为该改的条目为止。
4. 截图看首屏：只剩标题、一句话结论、settings 行，没有一个读者不认识的词。

## 领域示例：RL trial 分析

分析一次 RL 训练 run 时，四层这样落位（适用于有逐步日志的任何框架，例如 HF-Trainer 风格的 `log_history.json`）：

- **L4 聊天**：trial 名 + 一句话结论 + dashboard 地址（如果 dashboard 没有 URL 参数，说明要在下拉框里选哪个 run）+ 报告路径。panel 链接只放在聊天里，不写进报告。
- **L3 panel**：rlforge dashboard（360-2，端口 8871，下拉框选 run），或者用模板生成的静态 panel。
- **L1 报告**：先用一段散文讲指标走势（reward、回答长度、截断率、clip ratio、entropy/KL、独立测试探针）。然后是思维链分析：早期样本、中期长短样本对比、末期短样本，再加 think-vs-direct 对照。样本 ≤1000 字全文放出，更长的保留头、中、尾各约 500 字，并标明省略。分析结论最后写。
- **L2 图**：`scripts/plot_run_metrics.py` 从任何 `log_history.json` 画出 reward、回答长度与截断率、clip ratio 与 entropy/KL，已套用 panel 风格。
- **采样纪律**：生成思维链样本时，条件必须和训练一致：同样的 temperature/top_p、同样的 thinking 开关、同样的长度上限、独立测试题集。对照要选中期 checkpoint，那时思维链文本还会出现。

本机 360-2 集群操作手册（路径、dashboard 启动与隧道、SSH 抖动、vLLM 环境变量、checkpoint shim）与全部避坑清单：`references/rl-trial-analysis.md`，动手前必读。思维链样本用 `scripts/gen_cot_one.py` 生成，think-vs-direct 对照用 `scripts/ctrl_think_vs_direct.py`。

## 致谢与前人工作

- [Karpathy 原帖](https://x.com/karpathy/status/2105819303471976479)（2026-10-02）：本 skill 实现的输出阶梯。
- [ASD-STE100 Issue 9](https://www.asd-ste100.org/)：L1 背后的受控语言标准。
- [Anthropic skills · brand-guidelines](https://github.com/anthropics/skills)：panel 模板的配色和字体。
- [Orchestra AI-Research-SKILLs · academic-plotting](https://github.com/Orchestra-Research/AI-Research-SKILLs)：绘图规则。
- 调研过的 ASD-STE100 开源实现见 `references/asd-ste100-80.md`。公开版：github.com/guoshaoyang-pku/report-to-human。
