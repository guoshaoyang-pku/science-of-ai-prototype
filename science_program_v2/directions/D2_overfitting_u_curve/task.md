# D2 — 过拟合与 U 型风险曲线

## 用户指定核心问题（2026-10-07 21:0x，报告必须最先回答，不许跑题）

1. **U 到多少时 test 风险开始上升**：给出上升起点 U_rise 的公式（用 U=η·t 表述，t* 律换算），带实测系数与通过率。
2. **上升之后公式如何**：给出上升支 L_test(U)−L_min 的增长公式（允许对保存曲线做事后拟合，标注 post-hoc/development），带指数/系数与跨条件通过率。
报告"结论"节按这两问逐条直接回答；n/σ² 折叠反例等其余内容压缩到 support。

## 模糊问题

test 风险随训练时间呈 U 型（先降后升）：最低点位置由什么决定？能否**不用 test 标签**预测最低点或安全早停？

## 已知起点（只读）

- v1 C04（`../../../references/science_program_v1/studies/C04_noise_generalization/`）：固定特征、带噪标签下 train loss 持续降、期望 clean-test 风险 U 型；晚期噪声拟合超过 signal 收益；统一 U 阈值不成立。
- v1 C05（`.../C05_risk_ood/`）：初始谱 forecast 在封存新条件上预测整条风险曲线，43 个 recipe×checkpoint 在 3 MC SE 内，最优 step 17/24 精确、24/24 在 2 倍内。**但 forecast 用了 clean train/test 标签**——不是无标签早停。
- v1 KB C0002；主线 K1197（晚期 test CE 变差、无 train CE 机理测量的旧记录）。

## 未决问题

1. learned hidden（特征学习）下 U 型与谱 forecast 还成立吗？C03 已显示初始固定核预测在 learned hidden 下失效。
2. 无 test 标签的早停信号：train-side 可测量（梯度范数、残差谱、噪声拟合速率）里有没有能代理风险最低点的量？
3. 噪声率、样本量、width 如何移动最低点？有没有可检验的标度关系？
4. Adam/小批量/CE 下 U 型是否变形（双下降等）？（范围允许时）

## 什么算进展

- 在 learned-hidden 小 MLP 上测出 U 型边界（哪些条件 U 型消失/保留）。
- 提出并检验至少一个 train-only 早停信号，与"oracle 最低点"比较（开发条件迭代，封存条件前瞻验证一次）。
- 噪声率×样本量的最低点标度律（哪怕只在固定特征范围内成立并写明边界）。

## 首轮建议

复算 C04/C05 的固定特征设置（参考 v1 executed 源码与保存的 data 合同），把"最低点 step"作为因变量，扫噪声率与 n，拟合标度关系；这是纯本地小张量实验。
