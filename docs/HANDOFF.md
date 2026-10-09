# 学弟接手说明 · 2026-10-09

## 先读什么

1. [研究方向地图](RESEARCH_DIRECTIONS.md)：13 个问题、已有证据和候选下一问。
2. [最新 D2 报告](../evidence/d2_scaling_20261009/report.md)：扩大 n/噪声范围、跨任务、train/val 与图示。
3. [操作手册](OPERATIONS.md)：安装、取证据、只读复核和单轮研究准备。
4. 每方向 science_program_v2/directions 下的 task.md、report.md、findings/、studies/*/summary.json。

runtime 支持“解题→发现→整理”；v2 是单 supervisor 串行调度的模糊方向研究。两条线共享纪律，效果证据分别登记。当前没有新 solver 对照或泛化收益结论。

## 快照边界

v2 在 10-08 19:39 parked；旧源 HEAD 298b76434a03eddbff48d8a773bbf5c8e76c66fc、480 commits、128 claims（measured 96/refuted 29/mixed 3）。保留源端三个未提交停车账文件的快照，未将它们虚构成 commit。

轮账保留 **113 次启动 / 93 账面轮 / 92 按当前 history 退款规则重算**。D1 r93 已作废且 elapsed=0，但 rounds_done 仍为 7；日志仍有 130.8 秒。没有修掉差异，也不把预算计数称为科学成功轮。

runtime 来自 aiq_kb HEAD 2fb99ed6db92061dc3490db0129094b05efae454；v1 来自 c49964d3556e703aee59223446c0665565228085。每文件 source/export hash 在 evidence/manifests/，480 commit 索引在 evidence/history/v2_commits.json。原 Git 对象未打包；新库初始 commit 不能证明原预注册时序。

## D2 更新优先级

10-09 的 3240 轨迹与图表在 evidence/d2_scaling_20261009；research/ 提供源码和报告副本。旧 D2 report、INTEGRATED 和中心 KB 未追改，属于历史材料。新报告结论优先，旧 r113 聚合极差比 4.68 的失败仍保留。

R=B+σ²N 是风险恒等式，不推出最低点精确反比噪声。注册的 rise 是在窗口格点寻找并局部精化的全局最低点估计，不保证是所有任务第一次局部回升，也未证明遍历全部整数极值。这不改变扩域线性模型失败。

## 第一次操作

完成 README 安装与 verify_handoff.py，读 D2 的 summary 和图。复算前下载三个归档、校验并解压；先核对 hash 和分母，再决定小问题。

首次研究建议只开一个方向的一轮：用外部研究 repo、自己的 CLI 和明确预算；先 commit 预测，再算，再报失败/边界。D13 skip 未测，旧“不能进入 skip”是 agent 自设流程，可由负责人调整。

题库来自 AIQ main，但当前 CPU 模糊方向实验不依赖题目。新 solver 实验需另行确定模型、题池和对照合同；研究记录完整不能替代 solver 收益。
