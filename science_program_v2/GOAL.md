# GOAL — 中心化模糊方向自动研究（science_program_v2）

发起：用户，2026-10-06（北京时间）。承接 science_program_v1（12 项研究、7 条 KB、已开源 v0.2.0）。

## 用户目标（原话要点）

1. 每个方向的研究是中心化的：单一进程、单一状态、单一 KB、单一 git 历史。
2. 由一个 Codex 进程自动研究。模型：Sol 6.1 Ultra（= `gpt-6.1-sol`，reasoning effort `ultra`，经本地 relay，cctq 主路由、phybench 兜底）。
3. 模糊方向可以多一些。用户给出的种子例子：
   - **U 的研究**：包含 SGD+mom=0 导致等效 U 异常变小的现象。
   - **过拟合的研究**：U 型曲线。
   - **模型容量和分辨率的描述参量**。
   - **训练曲线能否用 sigmoid 描述**：sigmoid 的各参数或许有可描述的公式。
4. 交付物：技术报告 + 操作手册（OPERATIONS.md），以及每方向持续更新的 report。
5. 汇报语言纪律：面向用户的总结去掉 jargon，用 ASD-STE100 风格（短句、主动语态、平实词、术语必须解释）。

## 方向清单（8 个，前 4 个来自用户，后 4 个来自 v1 未决问题）

| ID | 模糊问题 | 主要依据（只读） |
|---|---|---|
| D1 | 等效训练时间 U 什么时候成立、什么时候误导（含 mom=0 异常） | v1 B01；主线 K1001/K1012/K1019 |
| D2 | 过拟合 U 型风险曲线：最低点在哪、能否不用 test 标签早停 | v1 C04/C05；K1197 |
| D3 | 模型容量与分辨率的可测描述参量 | v1 B01–B03、A03 |
| D4 | 训练曲线的 sigmoid 描述及参数公式 | v1 B01 精确递推 |
| D5 | LN 同宽度收益的机制（Jacobian/条件数中介） | v1 A03/A04 |
| D6 | 核漂移：何时特征学习改变目标相关结构、能否早期预测晚期 | v1 B02/B03 |
| D7 | train 增益 vs test 损害的边界条件 | v1 B02（8/8 train 受益、6/8 test 变差） |
| D8 | 激活通路结论在 bias/非对称/多层/CE 下的存活性 | v1 C01–C03；K1056 |

每个方向的 task.md 是入口。方向可以分叉出新子问题，但必须写回该方向的 findings 与 report，由中心 KB 登记。

## 中心（centralization）的含义

- 唯一 supervisor 进程（`central/supervisor.py`），串行调度轮次；同一时刻只有一个 codex 会话在研究。
- 唯一状态：`central/state.json`（轮次与方向状态）、`central/kb.json`（短 KB）、`reports/PROGRESS.md`（进度）。
- 唯一 git 历史：本目录是独立 git repo；预注册必须在跑实验**之前** commit（修复 v1 的时序见证缺口）。
- 记忆在文件里，不在会话里：每轮 codex 会话先读中心状态与该方向已有 findings，再干活，再把结果写回。

## 硬边界（继承 v1，违反即无效轮次）

1. 只做本机轻量计算（NumPy/PyTorch 小张量，秒~分钟级）。禁止 GPU 集群、禁止大模型训练。
2. 科学研究 0 模型 API/solver 调用；不评测解题、不动 benchmark test、不动 ranking 题池。
3. 只写 `science_program_v2/` 内部；v1 repo、soa_async_v1、kb_site、publisher 一律只读。
4. 不 push、不改 git config、不开网络服务。
5. 失败预测必须保留；development 与封存 OOD 分开；看过一次 OOD 结果后该条件转为 development。
6. 计数纪律：recipe/cell 是单位，seed 只测稳健性；不夸大数量。

## 完成判据

- 每方向至少完成配置的轮数预算，每轮有：预注册 commit、实验产物、findings、report 更新、中心状态与 KB 更新、收尾 commit。
- `reports/` 下有每方向技术报告；OPERATIONS.md 与实际行为一致。
- 用户可随时 STOP；停车状态必须可从文件完整恢复。

## 阶段 2：发车（用户指令，2026-10-06 23:26）

1. **预算扩大**：每方向 3→6 轮，总轮 24→80。当前循环按新预算持续研究。
2. **新增 4 个模糊方向**（从已测规律的自然边界生长出来，入口同样是各自 task.md）：
   | ID | 模糊问题 | 从哪来 |
   |---|---|---|
   | D9 | 小批量噪声如何改写 U/sigmoid/t* 规律 | D1/D2/D4 全部建立在全批量上 |
   | D10 | 深度增加时规律如何逐层传播 | D5/D8 未决 + v1 C 多层缺口 |
   | D11 | 描述量随 width/n 的标度律、lazy↔rich 边界 | D3 只在 n=d=9 测过 |
   | D12 | 单 sigmoid 失败后的级联结构与延迟跃迁公式 | D4 的谱比 ≥30 失败侧 + 10–30 空档 |
3. **报告标准升级为一等交付物**：每方向 report.md 必须达到"风格化 science 标准"——固定骨架见 AGENTS.md（开头公式区：核心公式+符号定义+适用阈值；结论段直说现象与解释、不引用文件；证据链接只进末尾"证据"节）。存量 8 份报告由各方向下一轮自行重排，不改数字。
4. **工作原则**（用户原话要义）：给定模糊方向，AI 有方向通常能给出好结果。task.md 只定问题、依据与"什么算进展"，不预定答案；具体小问题由每轮会话自选并预注册。

## 阶段 3：历史报告统一修复（用户指令，2026-10-07 17:0x）

1. 研究线暂停（STOP），`central/repair_reports.py` 串行给 12 个方向各开一个修复会话：自查 + 按 `central/style/REPORT_STYLE.md`（L4 给人汇报风格）重写 report.md；铁律不改数字；runner 做数字保全软校验并统一 commit。
2. 修复完成后 runner 自动清除 STOP 并重启 supervisor；`central/STOP_REPAIR` 可中止修复并保持停车。
3. report_to_human 技能全文副本入库（central/style/report_to_human_skill/），后续每轮报告都按此标准自查。
5. **D13 Linear Flow World**（用户 2026-10-07 群讨论新增）：深度线性网络（仅线性层+skip），三旋钮 {σ_l}/拓扑/{width_l}；验证"训练曲线 sigmoid → 相变时间 t_c → 有效深度 d_eff"链；skip 打破无 skip 情形的精确可解理论，是缺口。**特殊权限：仅此方向允许自造合成题目**（生成器入库带 seed）。论文种子标题：Measuring effective depth using phase transition time。
