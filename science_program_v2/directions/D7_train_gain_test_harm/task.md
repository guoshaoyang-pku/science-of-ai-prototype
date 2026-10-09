# D7 — train 增益 vs test 损害的边界（v1 B02 未决）

## 模糊问题

非线性特征学习让 train 更好但 test 更差（v1 B02：8/8 封存新条件 train 受益、**6/8 test 反而变差**）。什么可测条件决定继续训练是收益还是损害？

## 已知起点（只读）

- v1 B02（`../../../references/science_program_v1/studies/B02_nonlinear_clock/`）：真实 MLP vs 初始 Jacobian 切线配对。
- v1 C04/C05：噪声下的 U 型风险与谱 forecast（干净目标与带噪目标是两种损害机制，要分开）。
- v1 KB B0002："distinguish target-directed kernel change and implicit bias before ranking generalization"。

## 待区分的机制

1. **噪声拟合**（C04 路线）：test 损害来自拟合标签噪声；干净目标下应消失。
2. **方向性过拟合**（干净目标也发生）：核漂移把能量放进对 train 样本特异的方向；B02 的 6/8 是在什么噪声条件下测的要先核对，再设计干净/带噪配对。
3. **隐式偏置变化**：切线 vs 真实网络的泛化差来自优化路径的偏置差异。

## 什么算进展

- 干净目标 vs 带噪目标的受控配对：把 B02 的 test 损害分解到机制 1/2/3 的比例（至少给出主导机制的判据）。
- 一个可测的"继续训练是否损害 test"的前瞻信号（train-side 或核-side），开发条件建立、封存条件验证一次。
- 与 D2 的分工：D2 管 U 型曲线与早停，D7 管损害机制归因；共用测量代码时注明。

## 首轮建议

先重读 B02 的 executed 配置与结果，确认其 test 损害的噪声条件；然后设计干净/带噪 × 切线/真实 的 2×2 配对小实验。
