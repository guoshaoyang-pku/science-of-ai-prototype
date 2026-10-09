# D5 — LN 同宽度收益的机制（v1 A 方向未决）

## 模糊问题

同 width 下 LN 稳定改善偏移形状拟合（v1 A03：8/8 配对区间<0），且固定 LN affine、只学 hidden weights 即可保留 ~99–111% 收益（A04）。**机制是什么？** v1 未能证明特定 Jacobian 中介。

## 已知起点（只读）

- v1 A01–A04（`../../../references/science_program_v1/studies/`，报告在 `directions/A_residual_features/report.md`）。
- 候选解释（v1 已排除/未排除）：head 增长必要性（已反驳）；LN affine 更新（不够）；width 混杂（已排除）；**LN 几何下 hidden weights 学习**（当前支持，但中介量未测）。

## 待检验的中介假设

1. 条件数/谱整形：LN 使 hidden 表示的 Gram 谱更均匀 → 慢模态加速 → chord 收益。可测：配对训练里逐 checkpoint 的条件数与谱衰减 vs chord。
2. 梯度对齐：LN 下 hidden 梯度与目标方向对齐度更高（Rayleigh/夹角）。
3. 优化路径：LN 使轨迹停留在更平坦/更稳的区域（loss 曲面局部曲率沿轨迹的分布）。

## 什么算进展

- 至少一个中介量在 LN/noLN 配对中方向一致、幅度与 chord 收益相关（开发条件），并在封存新函数/新 width 上前瞻复现。
- 若干预该中介量（如人工重整谱）能部分复制/消除 LN 收益，则机制证据更强。
- 失败也登记：若三个候选都不中介，写明并给出下一步区分设计。

## 首轮建议

复用 v1 A03 的配对设计（同 width LN/noLN，8 函数），逐 checkpoint 保存 Gram 谱与梯度对齐量，做中介相关性分析。参考 v1 executed 源码保证 recipe 一致。
