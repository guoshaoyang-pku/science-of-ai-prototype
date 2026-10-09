# A：残差与特征学习交接

四项研究已完成，新增1740次训练，复用e16/e17及A02基线，无收费模型或solver调用。

| study | 新训练 | 已有结果复用 | 状态与主要结论 |
|---|---:|---:|---|
| A01_head_mediation | 360 | e16/e17基线 | head增长必要性预测失败；fixed head仍保留形状收益。 |
| A02_ood_residual | 960 | 0 | 四新函数OOD fixed-head收益4/4；优化器赢家10/12，二次函数±3失败。 |
| A03_ln_width | 240 | 240个A02cells | 同width下LN效应8/8配对区间<0，排除width混杂。 |
| A04_ln_affine | 180 | 60个A02cells | 固定LN affine、weights-only保留收益，affine-only不够；每项4/4。 |

A01、A02、A03和A04分别约65.9、123.7、29.1、27.9秒；计时口径见current.json与逐cellseconds。A01初始化与旧executor byte一致；7cell续接后hash未变。A02预注册commit1e3cfa8先于结果，seal.json冻结生成器/条件/预测；看过结果后这些条件已转development。A03和A04是development，不是新的OOD。

报告在每个study/report.md；A01主报告在本方向report.md。analyze.py读取逐cellJSON/NPZ核对hash，复算chord和配对区间。A01/A02历史入口用自己的executed/host/experiment.py，避免共享执行器变化失去contract。运行run.py会复用成功cell；应先核对离线release恢复契约，再允许新增训练。

独立审计在review.md与evidence/；原始A02的314个、A03的80个和A04的40个高MSE终点均保留。failure_probe只是工程保存/恢复检查，不计science。当前不能声称Jacobian中介已证明、Adam规律已统一或sol涨分。

用户将给新目标，当前停在可复现交付边界；不自行启动A05、solver或论文投稿。旧主线/publisher独立保存。
