# KB 定向 Science prototype

从已有 KB 的矛盾出发，三个方向分别研究残差与特征学习、有效学习时间与容量、激活通路与泛化。现已完成12项研究，短 KB 有7条，每条关联可读报告、提前预测及原始测量。用户将提供新目标，当前停在开源交付边界。

| 方向 | 已完成研究 | 具体发现 |
|---|---|---|
| A：残差与特征学习 | A01–A04，1740次训练 | head增长和LN affine更新均非偏移收益必要条件；固定LN几何下hidden weights学习可保留收益。 |
| B：有效时间与容量 | B01–B03，360次optimizer训练，另72条解析probe | 目标相关谱决定固定特征拟合时间；晚期特征学习改变方向，train增益仍可能损害test。 |
| C：激活与泛化 | C01–C05，624个recipe、1136条拟合轨迹 | 小尺度SiLU的偶通路受抑制；初始固定谱能提前预测封存条件的噪声风险曲线。 |

计数对应保存的recipe/cell；C04/C05每recipe包含多个head，不能当作独立目标。研究新增solver调用为0。固定谱、Taylor奇偶分解与bias/variance使用已知理论；当前贡献是受控区分、反例和条件内的前瞻验证，不是统一训练理论。

先读 [HANDOFF.md](HANDOFF.md)、[kb.json](kb.json) 和各方向 handoff.md。sources/保存e16/e17成功测量及当前KB的development导入，sources/manifest.json核对冻结版本。每个study包含preregistration、实际执行源码、结果、分析与报告；OOD只在首次事前预测后使用。

公开版位于 science-of-ai-kb 的 programs/directed_science；原始数值证据由release附件提供。使用公共metadata变化映射区分原始身份与公开hash；不把缺少原始Git历史的clone称为原预注册历史。旧正式主线和publisher独立保留。
