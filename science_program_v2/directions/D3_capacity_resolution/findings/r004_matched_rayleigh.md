# r004：相同目标 Rayleigh 商的配对拟合预算

程序第 4 轮、D3 第 1 轮没有训练结果。两次预注册提交都被 `.git/index.lock` 的写权限拒绝阻断；未执行 `run.py`，保存 **0/18 cell**，P1–P3 均为 `not_evaluated`。本轮结果为 `failed`。

## 本轮问题与合同

只检验一个小问题：固定特征和谱、保持初始损失及目标 Rayleigh 商相同，只改变目标的模态权重，达到初始损失 1% 的步数是否不同。Rayleigh 商指目标加权的平均特征值；本轮不把它称为通用容量。

模型为 n=d=9 的固定线性特征，特征值 `λᵢ=i⁻ᵖ`，p=0/1/2，各谱最大特征值均为 1。全批量 SGD 使用 η=0.5、无动量、无权重衰减、零初始化、float64、1000 步。同 p/seed 比较第三模态单目标与首尾模态混合目标；两者初始损失均为 0.5，目标 Rayleigh 商均为 λ₃。参数数、trace（特征值总和）与有效秩也保持相同。

共 6 个谱×目标条件，3 个参数坐标旋转 seed，计划 18 cell、9 个同 seed 配对。旋转 seed 仅检查数值稳健性，不是独立数据重复，不计算置信区间。

## 预测与解析工具

P1 要求实际归一化损失与保存 X/y 复算的目标加权谱递推误差不超过 1e−10，阈值步数一致，所有配对描述量差不超过 1e−12。P2 要求平谱 p=0 的两目标都在第 4 步达到阈值。P3 要求首尾目标/中间目标步数比在 p=1 时至少 2.5，在 p=2 时至少 8。任一阈值时间超过 1000 则保留删失记录，按合同判失败。

已知固定谱公式为 `L(t)/L(0)=Σwᵢ(1−ηλᵢ)²ᵗ`。第一版预算描述量为 `Tε(μᵧ,η)`：完整目标加权谱 μᵧ 在给定优化器下首次满足损失比例 ε 的整数步。公式只用于预测与执行核验，不是本轮的新发现。相同 Rayleigh 商也不保证实际第一步损失下降相同，因为离散更新包含二阶项。

首次提交前的草稿把 p=2 的 P3 写为至少 10 倍。静态解析审查发现该谱的预测预算比约为 8.88，因此在未训练、未查看数据时改为至少 8 倍；另补齐 P2 的两目标均为 4 步判据、强制 BLAS 线程为 1 并更新源码 hash。原草稿完整保存在第一次提交失败记录内，修订合同再次提交仍失败。这是执行前修订，不是测量得到的失败预测。

## 保存证据与边界

[预注册草稿](../studies/r004_matched_rayleigh/preregistration.json) 和 [执行源码](../studies/r004_matched_rayleigh/executed/run.py) 已保存。执行器要求合同及源码与本 repo 当前 HEAD 字节相同，核对 source/request/metadata/NPZ/receipt hash 后才恢复；成功 cell 不覆盖。源码通过 AST、JSON 和 hash 静态检查，未作数值执行验证。

[summary.json](../studies/r004_matched_rayleigh/summary.json) 由 [analysis.py](../studies/r004_matched_rayleigh/analysis.py) 读取空结果目录和提交失败记录生成，记录 0 cell 与未评估预测；分析没有调用训练函数或生成数据。[第一次失败记录](../studies/r004_matched_rayleigh/executed/preregistration_commit_attempt.json) 保留原草稿，[第二次失败记录](../studies/r004_matched_rayleigh/executed/preregistration_commit_attempt_002.json) 对应修订合同。[收尾记录](../studies/r004_matched_rayleigh/executed/final_commit_attempt.json) 保存最终 add/commit 尝试。

本轮未新增 KB claim。不能把解析算例或静态审查写成 measured/refuted；P3 尚未由保存训练结果检验。所有谱满秩，预算描述量只表示指定目标的优化时间，不是永久表示容量。本轮未测试 learned features、width/depth/激活/LN、泛化、空间频率分辨率或封存 OOD；目标参数范数和初始梯度范数也没有匹配。

下一轮须先恢复本 repo `.git` 写权限，审查并成功 commit 当前合同与 pinned 源码，再用指定 Python 执行一个 cell、核对恢复 hash 后续跑，最后复算配对步数与判据。未取得预注册 commit 前禁止训练。
