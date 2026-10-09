# KB 科学实验线总览 · 2026-10-06 21:15

本页取代 panel_kb_science_lines.html（该 HTML 保留但不再维护）。数据来源：central/log.jsonl、central/kb.json、各方向 report.md、公开 kb_site index（built 2026-10-03）、v1/主线交接文档。

## 新线 science_program_v2（运行中）

🤖 gpt-6.1-sol（Sol 6.1 Ultra） · 🧠 ultra · ⏱️ 轮超时 7200s · 📊 预算 3 轮×8 方向 · 🖥️ 本机 CPU · 🔒 沙箱可写工作区+.git，无网络

现状一句话：16 轮完成、第 17 轮（D1）进行中；19:31 修复重启后 5 轮全部 ok；中心 KB 17 条 claim（15 measured/mixed + 2 refuted 保留）。

| 方向 | 问题 | 轮 | 当前发现一句话 | 报告 |
|---|---|---|---|---|
| D1 | 等效训练时间 U 何时可信（含 mom=0 异常） | 2/3 | 梯度范数只是单位换算，不改归一化轨迹（差 3e-15）；动量启动换算早期 4/4 准、晚期 6/10 失败（已登记 refuted） | [report](../directions/D1_u_effective_time/report.md) |
| D2 | 过拟合 U 型最低点、无标签早停 | 2/3 | "同 n/σ² 严格折叠"被 4 反例推翻；n/σ² 幂律粗预测 57/60 在 2 倍内通过；无标签早停未建立 | [report](../directions/D2_overfitting_u_curve/report.md) |
| D3 | 容量与分辨率的描述参量 | 2/3 | 匹配全部标量描述量（trace、有效秩、Rayleigh、初始损失）的两个目标，1% 损失预算仍差 2.9–8.9 倍；完整目标加权谱预测精确 | [report](../directions/D3_capacity_resolution/report.md) |
| D4 | 训练曲线的 sigmoid 描述与参数公式 | 2/3 | log-time 单 sigmoid 适用边界=谱时间尺度分离度：比值 1/3/10 通过、30/100 失败；拟合参数≠真实半衰期（偏差 −9%~+72%） | [report](../directions/D4_sigmoid_training_curve/report.md) |
| D5 | LN 同宽度收益机制 | 2/3 | LN 降条件数方向 12/12 一致，但降幅与收益相关 r=.42 低于注册 .5（refuted 保留）；条件数差在初始化已存在，机制未定 | [report](../directions/D5_ln_geometry/report.md) |
| D6 | 核漂移早期→晚期预测 | 2/3 | 前 32 步漂移方向在 3/4 recipe 保留到 256 步；mixed_sine×width8 反例 0/3 seed 保留 | [report](../directions/D6_kernel_drift/report.md) |
| D7 | train 增益 vs test 损害边界 | 2/3 | 固定加噪使真实网络对线性化网络的 clean-test 差距 4/4 单元扩大（12/12 初始化）；"每个 seed 晚期都受损"不成立 | [report](../directions/D7_train_gain_test_harm/report.md) |
| D8 | 激活通路结论在 bias 下的存活性 | 2/3 | 解析+72 cell 实测：普通 bias 只改系数、能量比仍按 a²（测得 ~4.0 倍）；二阶导零点 b*=2.3994 处变 a⁶（测得 ~64.0 倍）；P1–P3 全过 | [report](../directions/D8_activation_pathway/report.md) |

事故与修复（已完成）：① 沙箱默认拒写 .git，前 10 轮无法预注册 commit（8 failed、2 partial）；修复=writable_roots 加入本 repo .git，烟测+第 12 轮实证通过。② rounds_done 双重计数；修复=supervisor 从 log.jsonl 单源重建。修复 commit `205a9f4`；第 12 轮起每轮预注册 commit 早于训练（git log 可查 r012–r017）。

## 旧线整理（公开页：[kb_site index #runs](https://guoshaoyang-pku.github.io/blogs/kb_site/index.html#runs)）

| 线 | 模型 | 规模 | 结果一句话 | 状态 |
|---|---|---|---|---|
| soa_v1 | gpt-6.1-sol | e1–e13，KB 28 条 | kb_0010 val +6.83±2.13pt；反复选择后 val 非独立证据 | 归档 |
| soa_v2 | gpt-6.1-sol | 仅种子 KB | 训练对照诊断，无完整轮 | 归档 |
| soa_v3 | gpt-6.1-sol | e1–e10，KB 28 条 | kb_0008 val +5.67±1.84pt；被拒候选保留 | 归档 |
| soa_v4 | gpt-6.1-sol | 6.9 轮，KB 35 条 | e5 val +1.04±1.82pt，不能排除零增益 | 归档 |
| luna_main | gpt-6-luna | 12 轮，~$84/10h，40 题×3 采样 | noKB 36.7%；e8 峰值 44.2%（+7.5±4.4pt）后回落；增益主要来自筛选种子条目（361 引用仅 8 次指向新增） | 归档 |
| qwen_6ep_40eval | qwen3.8-max | 5 轮中断 | kb_0002 −6.9pt、kb_0004 −6.3pt，无增益 | 归档 |
| sol_tool | gpt-6.1-sol | e1–e4，15 题/轮 | 工具检索版初始 KB，训练均值 .47→.62 | 归档 |
| luna_b15 / luna_gate | gpt-6-luna | 试跑 1 轮 / 门控中止 | 试跑 −8.75±10.5pt；门控版无结果 | 归档 |
| smoke×3（qwen/opus） | qwen / opus-5.5 | 烟测 | opus 线 vapi 余额耗尽全部失败 | 归档 |
| **soa_async_v1（旧主线）** | gpt-6.1-sol | 15 epochs，143 条 KB，77 报告 | macro15（105 dev ranking×3）：KB 70.08% vs noKB 65.87%，**+4.21±2.37pt**；早期长 KB 曾 −8.1pt，改短 KB+工具检索后转正 | 完成，publisher 存活（PID 15910） |
| **science_program_v1** | 本机小实验 | 3 方向 12 研究，2652 cells | 7 条 KB（head 增长非必要、U 非普适、SiLU 偶通路 scale² 抑制、谱 forecast 风险曲线）；854MB 测量随 release | [开源 v0.2.0](https://github.com/guoshaoyang-pku/science-of-ai-kb/releases/tag/v0.2.0) |

旧线三点结论（2026-10-03 阶段验收，仍是当前共识）：闭环全自动跑通；有提升但不稳（单次评测噪声 ±5pt 量级）；提升主要来自筛选而非新科学——这是转向 v1/v2 受控小实验的原因。

## 文件索引

| 内容 | 路径（相对本目录） |
|---|---|
| 各方向科学报告（8 份，活文档） | `../directions/D*/report.md` |
| 逐轮进度流水 | `PROGRESS.md` |
| 中心 KB（17 条） | `../central/kb.json` |
| 每轮会话完整输出 | `../central/rounds/rNNN_Dx/output.log` |
| 操作手册（监控/停车/转向/预算） | `../OPERATIONS.md` |
| 目标与边界 | `../GOAL.md` · 每轮规则 `../AGENTS.md` |
| 启动报告 | `LAUNCH_REPORT_2026-10-06.md` |
| v1 交接 / 主线交接 | `~/Desktop/workdir/Ideas/handoffs/KB定向Science开源交接_2026-10-06.md` |
