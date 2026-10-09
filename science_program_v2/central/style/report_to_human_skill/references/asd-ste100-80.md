# ASD-STE100 @ 80%：汇报用规则参考

基线：ASD-STE100 Simplified Technical English, Issue 9（2025-01-15，ASD 欧洲航空航天与国防工业协会）。
Karpathy 建议"80% of the way to ASD-STE100"：保留句式纪律，放宽词表锁定。80% 是风格选择，不是合规分数；宣称 strict 合规必须对照官方 Issue 9 规则与 ~900 词受控词典（官方免费申领但禁止再分发：asd-ste100.org）。

## 核心规则（80% 模式，报告写作够用）

**词与术语**
- 一个概念一个名字，全文一致（规则 1.11 / 9.4）。这是 AI 输出最常犯的错：先叫 worker、再叫 agent、又叫 executor。
- 平实动词替换官僚词：use←utilize、start←commence、before←prior to、about←approximately。
- 允许现代技术名词（API、cache、checkpoint、reward），但每个技术名词全文单一含义。
- 不把技术名词当动词用（不要 "to checkpoint the model" 之外再造新动词化用法时不加说明）。

**句子**
- 指令句 ≤20 词，描述句 ≤25 词（硬上限 25）。中文按同一精神执行：一句一个事实。
- 一句一条指令；两个动作仅当同时发生才可并列。
- 条件在前、命令在后，用逗号隔开（"If the run has a cap penalty, report seqlen first."）。
- 主动语态；描述句仅当动作者未知才允许被动。
- 不用分号；不用缩写式（don't→do not）；少用括号，括号内容计为一词。

**段落与结构**
- 每段一个主题，≤6 句；复杂材料转竖排列表。
- 句间用显式连接词，不靠读者脑补逻辑。
- 多词名词 ≤3 词；更长的先给全称再定义短形式。

**情态与精确性**
- must / can 或精确条件；should / might / could 要么删掉要么改成明确条件。
- 数字、单位、不确定性、例外、限定词一律保留。为长度删事实是违规，宁可拆句重组。
- 警示块：WARNING / CAUTION，先命令或条件，后风险后果。

**中文报告的映射**
STE 规则跨语言执行（JAICHANGPARK 的 STE pane 对任何语言的回答应用同样规则）：短句、单一术语、主动语态、一句一事、显式连接词、保留全部数字与限定。

## 注意

- Karpathy 推文附带的 cheat-sheet 图有错（颠倒了一条词典规则、批准了一个标准拒绝的动词）：max.nardit.com/articles/karpathy-understanding-llm-outputs 做了逐条 fact-check。不要拿那张图当规范，用 Issue 9 或下面的机器可读参考。
- 格式越漂亮越容易让错误显得可信；每种格式要回答"它让人能检查什么"。

## 开源实现清单（2026-10 调研）

| 项目 | 定位 | 可取之处 |
| --- | --- | --- |
| `0xpili/simplified-technical-english` | Agent skill，911 stars，早于推文（2026-08） | 规则 + 批准词表 + check 工具，README 本身用 STE 写成 |
| `ashryaagr/karpathy-output-style` | Codex/Claude 插件，四 skill 套装 | 与 Karpathy 四层一一对应：clear-writing / explain-diagram（Excalidraw MCP）/ explain-webpage / explainer-videos（Manim+ElevenLabs）；references/asd-ste100.md 有 Issue 9 规则映射和推文图片勘误 |
| `danyuchn/asd-ste100-skill` | Claude skill（Karpathy 回复区链接的那个） | Strict / STE-flavored 双模式；确定性 linter `ste-lint.py`（查分号、短语动词、名词化、营销形容词、被动语态、长句、同义词轮换）；`npx skills add danyuchn/asd-ste100-skill` |
| `JAICHANGPARK/ASD-STE100` | Claude Code 插件 + mod | `/asd` 命令、维修手册风格 STE pane、strict-100 vs 80% 对照表、`bin/ste.js check` 打分 CLI、四认知模态框架 |
| `prdmy/asd-ste100`（prithivrajmu） | 轻量 agent skill | 默认 80% 模式，规则精简 |
| asdste100foragents.shop | 机器可读参考站（非 skill） | 53 条规则的独立转述：`/llms-full.txt`（prompt-ready 全量）、`/rules.json`、`/prompt.txt`（drop-in 编辑提示词），可 curl 后内嵌 |

## 出处

- Karpathy 原帖：x.com/karpathy/status/2105819303471976479（2026-10-02）
- 官方标准：asd-ste100.org，Issue 9 PDF：asd-ste100.org/assets/files/ASD-STE100_ISSUE9.pdf
- Fact-check：max.nardit.com/articles/karpathy-understanding-llm-outputs
- 本文件规则条目转述自公开二手来源（上表），未再分发官方词典。
