# C：小激活、学习速度与泛化

当前问题：小预激活下 SiLU 变慢，究竟是整体梯度缩小，还是特定目标的可学习通路变弱？已有 K1056 只有 recipe 条件下的胜率；K1197 的后期 test CE 变差没有 train CE 或机理测量。K1031 已不在当前活跃/retired 列表，不能按旧 ID 推断内容。

第一轮 C01 固定随机特征、只训练线性 head。使用同一初始化、对称 Gaussian 数据和 minibatch-free 梯度下降，比较线性目标与二次目标；统一特征 RMS 后，尺度差仍存在才排除纯幅度解释。另比较 preactivation LayerNorm 和线性 skip。每个 seed 同时用于所有配对条件。

SiLU 的精确奇偶分解是 odd(z)=z/2、even(z)=z·tanh(z/2)/2；小 z 下 even≈z²/4。ReLU 的两部分为 z/2 与 |z|/2。由此提出可区分预测：整体 RMS 相同的小尺度 SiLU 保留线性学习，却压低二次目标通路；ReLU 的归一化特征随正尺度不变。局部导数均方本身不能区分目标，需测目标对齐的核能量。

第二轮根据 C01 证据选择干预：单独匹配奇/偶特征能量，验证是否恢复二次目标速度；在封存的新维度、输入分布或目标上检验预测。随后放开 hidden 学习，检查随机特征结论的边界。噪声小样本实验单独测 train/test 曲线和噪声拟合，不把学习速度与泛化混为同一指标。

所有新条件均在测量前保存 preregistration。sources/ 只读；原主线及 solver 不变。结果单位是目标/recipe，seed 仅测稳健性。方向只提交自己的 state、report、proposal 与 study 文件，由协调者登记 KB/commit。
