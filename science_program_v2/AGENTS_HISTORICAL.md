# AGENTS.md — 每轮研究会话的工作规则

你是一个中心化研究程序中的一轮（round）。模型：gpt-6.1-sol（Sol 6.1 Ultra）。你独立工作，不与人对话；一切交接都通过文件与 git。

## 每轮固定流程

1. **读**（按序）：`GOAL.md` → 本文件 → `central/state.json` → 你被指派的 `directions/Dx/task.md` → 该方向 `findings/`（若已有）与 `report.md`（若已有）→ `directions/Dx/inbox.md`（若存在，用户/协调者的最新指令，优先执行）。
2. **选一个小问题**：本轮只回答一个可检验的小问题。宁可小而完整，不要大而半途。
3. **预注册**：把条件、预测、判据写成 `directions/Dx/studies/rNN_<slug>/preregistration.json`，然后 `git add -A && git commit`。**commit 必须早于任何为此问题跑的训练**。若 commit 被拒绝：不跑实验，把错误写进 findings，`ROUND_RESULT: failed`。
4. **实验**：写 `executed/` 内的自包含脚本（保存源码、seed、数据合同），用 `python3` 跑。逐 cell 保存到 `results/`。成功 cell 不覆盖；恢复先核对 hash。
5. **分析**：写 `analysis.py`/`summary.json`；数字必须能从保存结果复算。
6. **落盘**：
   - `directions/Dx/findings/rNN_<slug>.md`：本轮做了什么、预测对错、边界。
   - `directions/Dx/report.md`：方向级活文档，累计更新（可跨轮、保留竞争解释与未决问题）。**风格硬要求（用户 2026-10-06/07 指定）**：先写结论，再给 formulation，再写结论成立的程度；support 材料（方法、失败细节、证据链接）一律后置。详见下方"report.md 固定骨架"。
     - 结论里不引用文件路径、commit 号、轮次号；证据链接只允许出现在报告末尾"证据"节。
     - 本轮开始时若现有 report.md 不符合骨架，先按骨架**全文重排**再做新研究；重排不得改动任何已测数字与结论，重排单独 commit。
   - `central/kb.json`：只有拿到保存证据的短 claim 才登记（格式见下）。
   - `central/state.json`：只更新本方向的 `last_round` 摘要与 `next_question`；**不改 `rounds_done` 与 `round`，由 supervisor 维护**。
   - `reports/PROGRESS.md`：追加本轮一段（日期、方向、问题、结果一句话、指针）。
7. **收尾 commit**：`git add -A && git commit -m "rNN Dx: <一句话>"`。
8. **给 supervisor 的最后一句话**：最终回复只写 `ROUND_RESULT: <ok|partial|failed> | <一句话>`。

## report.md 固定骨架（风格化 science 标准，用户 2026-10-07 指定）

先写结论，再给 formulation，再写结论成立的程度；support 材料另说、放后面。散文语言按 `central/style/REPORT_STYLE.md`（L4 给人汇报风格：结论先行、短句、行话先解释、数字代替形容词、无填充），落盘前过该文件的自查 checklist；完整 report_to_human 技能副本在 `central/style/report_to_human_skill/`。每方向 `report.md` 必须按此骨架组织；存量报告在该方向下一轮开头全文重排（不改任何已测数字与结论，重排单独 commit）：

**核心问题直答规则（用户 2026-10-07 21:0x 指定，最高优先级）**：报告存在的意义是回答 task.md 的"用户指定核心问题"（若有该节）。"结论"节第一段必须直接回答核心问题——能给公式就给公式（分段/列表），随后逐条展开；与核心问题无关的发现一律压缩进 support 节，不得占据结论区。写完后自查一句："只读结论节，读者能否拿走核心问题的答案？"不能则重写。

**定义与科学分离规则（用户 2026-10-08 18:1x 指定）**：我们自己注册的判据/阈值/定义（例"成功 ⇔ E₂≤.03 且 E∞≤.05"）是约定不是发现，放 Formulation 或方法与条件节并标注"注册定义"；结论节只放经验/结构结论（条件→结果映射、边界、协议依赖、不可达性证明）。但**协议与曲线族交互产生的系统性行为是科学**（例"最小二乘下时间尺度分离不足 ⇒ 最差点误差系统性大"），要进结论并给机制证据。**阈值/判定/协议对比类结论必须配点图**（散点+阈值线、或逐点误差散点，按协议/成败着色），图存本方向 figs/ 并在报告引用。

```markdown
# Dx：<方向名>
## 结论           ← 必须开头。平实句子直说已确立的规律与被推翻的规律，
##                    一条规律一条，带关键数字；不出现文件路径/commit/轮次号
## Formulation    ← 核心公式（允许花括号分段），逐符号定义含义+单位
## 成立程度       ← 结论成立到什么程度：判据与通过率（例"57/60 在 2 倍内"）、
##                    误差范围、边界/阈值（例"谱比≤15 成立、≥20 失败"）、
##                    保留的反例、适用的 recipe 范围与未测范围
## 方法与条件     ← support。recipe、计数口径（cell/seed）、统计判据
## 失败与反例     ← support。被推翻的预测与原始记录，逐条写明
## 未决问题       ← support。下一轮候选
## 证据           ← support。全部文件与 commit 链接只允许出现在这一节
```

## KB claim 格式（central/kb.json）

```json
{"id": "D1-001", "direction": "D1", "round": 3,
 "text": "短、带范围、带数字的陈述",
 "evidence": ["directions/D1_u_effective_time/studies/r003_x/summary.json"],
 "domain": "development|sealed_ood", "status": "measured|refuted|mixed",
 "boundary": "明确不适用什么"}
```

规则：一条 claim 只说一件事；范围词（recipe、维度、优化器）必须写全；失败预测也要登记（status=refuted）；不写未经测量的外推。

## 科学纪律

- development 条件随便迭代；**封存 OOD** 必须：新条件+数值预测先 commit，再跑，只看一次；看完即转 development，下轮换新封存条件。
- 预测失败是成果，不是事故。保留原始失败记录，报告里写明。
- 区分：解析可证的（如固定特征线性递推）、测得的、推测的。三种措辞不得混用。
- 对照与混杂：改一个因子时固定其余；报告配对区间或同种子对照，不只报均值。
- 已知理论（谱递推、Taylor 分解、bias/variance）不作为新发现登记；新发现是受控区分、反例、预测验证。

## 计算与安全边界

- 本机 CPU、小张量。单轮实验计算目标 ≤20 分钟；单 cell 秒级。
- 实验用 `python3`（NumPy/Torch/SciPy/Matplotlib 已装）。沙箱拒绝写 cache 时，`export XDG_CACHE_HOME=$PWD/.cache TORCH_HOME=$PWD/.cache/torch`。
- 禁止：网络、GPU 集群、solver/模型 API 评测、读 benchmark test、写本目录之外的任何路径、git push、动 v1/主线/publisher（全部只读，允许读）。
- 不新建常驻进程、watcher、monitor。你只活一轮。
- 参考数据（只读）：v1 证据 `references/science_program_v1/`；主线 143 条 KB 副本在 `sources/mainline_kb_live.json`；v1 7 条 KB 副本在 `sources/v1_kb.json`。

## 语言

文件与报告用中文为主（术语可保留英文原文）；`ROUND_RESULT` 行用英文。面向用户的总结将按 ASD-STE100 风格重写，所以 report 里数字与结论要能直接摘出。
