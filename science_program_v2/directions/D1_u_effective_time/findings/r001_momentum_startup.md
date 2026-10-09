# r001：预注册提交受阻，实验未启动

本轮选定一个问题：同学习率、同固定谱、同目标和初始残差时，mom=0 与 mom=.9 达到相同归一化函数误差的步数，能否由稳态 U 或恒梯度启动修正换算。已按顺序读取 GOAL、AGENTS、中心状态、task；此前没有 findings、report 或 inbox。

预注册计划为 n=d=8、全批量 SGD、float64、η=.1，固定谱 [.001,.01,.05,.1,.2,.3,.4,.5]。目标落在其中 5 个模态，初始梯度 L2 范数为 .003 或 .01，动量为 0 或 .9。3 个旋转种子仅用于坐标稳健性。计划 10 个配对条件、20 个条件/优化器单位、60 个 cell，各 40000 步。主阈值为 L(t)/L(0)≤.9，次阈值为 .1 和 .01；同时记录首次过阈值与持续至预算末端过阈值。

已保存 [预注册](../studies/r001_momentum_startup/preregistration.json)、自包含执行器和分析脚本。Python AST 解析与源码 SHA256 核验通过。这些是静态核验，不是实验结果。所有数值预测均未评估，保存 cell 数为 0，没有科学 claim 写入 KB。

`git add -A` 返回 exit 128：无法创建 `.git/index.lock`，报 `Operation not permitted`。当前权限将 `.git` 设为只读，且不允许权限升级。因为预注册没有 commit，本轮没有运行训练，也没有尝试绕过 git 权限。原始失败见 [提交尝试记录](../studies/r001_momentum_startup/executed/preregistration_commit_attempt.json)，本轮状态见 [summary.json](../studies/r001_momentum_startup/summary.json)。收尾 git 提交同样受此权限阻断。

下一轮须先恢复本 repo 的 git 元数据写权限，审查并提交预注册，再执行已准备的脚本。该计划仅检验固定特征单模态目标下的配对换算；不测 Adam，不判断相同 η/(1−μ) 下的动量额外收益，不给 K1019 的历史 bigram 结果作因果分摊。
