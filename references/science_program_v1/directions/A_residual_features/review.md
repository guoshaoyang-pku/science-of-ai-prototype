# A01–A04 独立复核：固定 head 收益与外推边界

360 次 A01 新测量支持一条具体修正：在三个已见函数和当前 recipe 中，head 权重增长不是 SGD 偏移收益的必要条件。固定完整 head 时 hidden 仍能拟合常数，也改善中心化目标；不能因此把中介直接命名为特征学习。预注册的“固定 head weight 至少消除 50% 收益”在三个函数均失败。后续 A02 的 960 次封存外推中，四个新目标都保留固定 head 收益，但旧赢家规则两项预测失败。

## 测量与来源核对

逐一核对 360 cell 的初始参数、四组输入/标签 tensor、初始 train/test 预测、256 步 minibatch stream 与 optimizer defaults，全部与对应 e16 保存条件一致。五个观测步中，被冻结的 head 参数均无梯度、无更新、无位移；全部终点 MSE、中心化分解和保存统计独立复算一致。1,510 个来源文件及 1,800 个 cell 执行 pin 一致；A01 无失败，也没有 cutoff=2 超限结果。

预注册 SHA 为 2e2892dbf0980120096df3c27bac39558420871b01e2b2e4e1383e3f72ff5be0，commit 7027db7 在首个成功记录保存前封存。首次 7 个成功 cell 的 14 个 JSON/NPZ 文件在续跑后 hash 均未变。新测量耗时合计 65.9 秒；没有新增 solver 评测。审计结果和当时实际执行器存于 [evidence/A01](evidence/A01/independent_audit.json)。

## 反例的强度与解释边界

C=[中心化误差(−3)+中心化误差(+3)]/2−中心化误差(0)。负值表示偏移收益；以下区间只针对固定函数/划分的 10 配对 seeds，未做多重比较校正。

|函数|原 SGD C|固定 head weight 的 C|保留收益|固定完整 head 的 C|
|---|---:|---:|---:|---:|---:|
|02036a|−.7339|−.4513 [−.4734,−.4292]|61.5%|−.4739|
|071e42|−.5725|−.6447 [−.6720,−.6174]|112.6%|−.6624|
|0852dd|−.5613|−.4730 [−.4967,−.4492]|84.3%|−.4903|

用同 seed 的 C_fixed−0.5 C_original 直接检验，三个 95% 区间分别为 [−.1025,−.0662]、[−.4076,−.3093]、[−.2367,−.1479]，均违反预设衰减方向。两种冻结模式的全部 30 个 SGD seed 都有负 C。完整固定 head 时，六个非零均值条件的终点均值残差平方为 .00155–.01601，全部满足预设 <.5。

“偏移仍改善 SGD”与“SGD 一定胜 Adam”是两个命题。固定 head weight 后，02036a 两个非零偏移的 SGD−Adam 均值虽为负，配对 95% 区间均跨零；完整固定 head 的六个非零均值比较则都为负且区间不跨零，seed 胜数为 7–10/10。不能沿用 e16 的每方向 10/10 到 A01。

冻结 head 同时删除它的优化器状态和 decay 更新；可训练 hidden 仍包含 Linear weight/bias 与 LN gain/bias。因此现有结果排除了 head 增长“必需”解释，仍容许 LN 参数、hidden bias、非线性 Jacobian 漂移共同起作用。位移或初步更新范数增加不是目标相关特征已学到的充分证据。

## 下一项区分实验与 OOD

最有信息量的开发对照是固定 head 后拆分 hidden：只训练 Linear weights、只训练 bias/LN gain 与 bias、训练全部 hidden；在相同初始化和 stream 下比较 C、中心化训练误差及函数更新。若去掉 bias/LN affine 仍保留大部分负 C，可否定“常数主要经 affine 参数拟合”解释；若仅 affine 已足够，则特征重塑不是当前证据要求的机制。参数组改变后的初始模型必须完全相同，不能用删 LN 代替冻结 LN。

更直接的理论对照是同一网络在初始化的完整线性化模型。固定 Jacobian、平方损失、同一 minibatch stream 的 SGD 输出关于标签 offset 为仿射，中心化误差 C 必非负；这个代数命题应在任何函数上成立。若原非线性网络有负 C、完整线性化对照没有，能把所需非线性路径与仅改变可训练容量分开。它并不提供 Adam 的仿射结论。

后续 OOD 必须另外提前保存新函数/recipe 的赢家方向及 C 预测，说明哪些条件是跨函数、跨输入分布、跨宽度/LN外推。对称性和固定特征的代数校验不能替代 SGD/Adam 赢家的 OOD 预测。失败条件全部保留；形成新解释后不能继续把同一批已测条件称为最终保留域。

## A02：四个新目标的首次封存外推

已独立复核全部 960 cell 的 contract/执行源码/NPZ、数据生成器和四组输入、配对初始化与 stream、冻结参数、统计和预测判定。预注册、生成源码与当时执行器在 1e3cfa8 封存；sealed_at=1791223588.628 早于首保存 1791223589.524。四个新函数使用 uniform[−1,1]^8 或 Gaussian12，训练集计算中心与尺度后同样用于实例内 test。全部有限完成，314 个 cutoff>2 结果保留。审计脚本与结果在 [evidence/A02](evidence/A02/independent_audit.json)，重跑审计不训练。

|新函数|LN010/w192，固定 head 的 SGD C [95% CI]|zeroLN/w64 的同项 C|
|---|---:|---:|
|trigonometric8|−.3642 [−.4179,−.3105]|−8.61e−6|
|quadratic8|−.0341 [−.0568,−.0114]|−2.74e−5|
|interaction12|−.1613 [−.1855,−.1371]|+5.31e−5|
|radial12|−.2330 [−.2649,−.2011]|+5.65e−5|

LN010/w192 下四个新函数均有负 C，20/20 条件内 seeds 同向。±3 的 SGD 赢家预测只命中 6/8 均值方向，mean0 的 Adam 预测 4/4；预设 all12 的整体规则失败。quadratic8 两个非零偏移反向但区间跨零，另 trigonometric8/radial12 的 −3 方向虽均值命中，区间也跨零。这说明 SGD 内部拟合收益比“SGD 必胜 Adam”稳定，不能以 10/12 包装普适赢家理论。

平移响应的16/16与固定特征SGD的8/8检验均通过；它们核对代数机制，赢家外推单独统计。预设 LN_boundary 的4/4只证实两个 recipe 的收益幅度相差很大：LN与width同时改变，不能判定 LN 必需。下一轮应补同width×LN全因子，同时拆 hidden 的 Linear weights 与 bias/LN affine；继续用A02拟合后它归 development，新修订必须在另一组未见条件确认。每函数固定一次数据划分，五个seeds不是五个新目标，当前对总体置信度仍有限。

## A03：同宽度对照排除 width 混杂

新增240次SGD训练、复用A02的240次成功记录，在width64/192分别补齐LN010与zeroLN。全部240个同宽度对照的输入和minibatch stream一致；shared Linear参数初始化逐tensor相同，新增的参数只有LN的gain/bias。新执行器已独立archive，预注册commit 3a43ef7早于首测量保存，80个cutoff>2结果保留。统计独立复算一致，见 [evidence/A03](evidence/A03/independent_audit.json)。

固定head时，8个“函数×宽度”的LN−zeroLN中心化C均为负，配对95%区间全部小于零；LN010/w64的C为−.2619/−.0178/−.0405/−.0721，而zeroLN/w192只有−4.80e−5/−1.81e−5/−6.60e−6/−2.35e−4。这次消除了LN与宽度的联合变化，可确认当前recipe中插入LN模块增强偏移收益。LN的forward归一化与新增可训练affine仍共同改变，下一步只冻结LN gain/bias可保留相同初始forward，区分这两者；不能立即改写为“归一化尺度解释已完成”。A03沿用四个已见A02目标，属于development。

## A04：LN affine 可冻结，hidden 权重路径已足够

独立核对180次新训练与60次A02复用的配置、源码与数组hash、初始全部参数和预测、数据、minibatch stream、optimizer defaults以及五步参数组记录；全部一致。预注册与实际host执行器在74bf2b4封存，commit时间1791224529早于首保存1791224530.643。40个cutoff>2结果全部保留，无失败，训练耗时合计27.4秒；审计脚本与结果见 [evidence/A04](evidence/A04/independent_audit.json)。

|A02已见目标|冻结LN gain/bias保留收益|只训练hidden Linear weights保留收益|只训练bias/LN affine的C [95% CI]|
|---|---:|---:|---:|
|trigonometric8|99.81%|103.48%|.00444 [−.00243,.01131]|
|quadratic8|99.55%|111.22%|.00106 [−.00236,.00447]|
|interaction12|99.16%|108.57%|.00048 [−.00142,.00238]|
|radial12|99.23%|99.95%|.00115 [−.00180,.00410]|

三项事前50%阈值预测均4/4通过，配对C_mode−0.5C_baseline的12个95%区间全部支持预设方向。head固定后，LN的trainable gain/bias不是主要收益所需条件；bias/LN affine单独训练不足以保留一半收益，hidden权重单独训练已足够。affine-only的C区间均跨零，不能证明严格没有效应。冻结同时去掉相应gradient/decay更新；四个目标在A02后已成为development，这次不构成新的OOD，也没有solver收益测量。

完整初始线性化对照仍是最直接的下一项区分：固定所有初始Jacobians时，平方损失SGD的关于标签偏移的预测是仿射，中心化C应非负；非线性网络的负C是否超出该对照，尚没有实测。A04支持hidden权重路径的因果充分性，未证明Jacobian漂移的具体方式，也未建立SGD/Adam赢家理论。

## 执行版本与失败记录的边界

train module的Model/build_optimizer被experiment.py运行时替换；executed/model.py和optimizer.py不能单独定义实际测量。cell contract已pin实际experiment.py，各轮现保留executed/host版本；A02/A04审计使用该immutable archive，根执行器改变不影响旧成功的核验。公开snapshot若不包含原私有Git历史，审计明确将chronology依据标为保存seal或原审计记录；不能声称在该snapshot重新验证了原commit时间。

原版本无法strict JSON保存Infinity的问题已修：executor_failure_probe保存failed=true、test_mse=null、空终点NPZ，首步非有限观测用null表示；没有以工程探针作science测量。verification.json声称复用失败未重训，但该probe的实际executor pin 4265414d…尚未找到字节archive，独立恢复证据仍不完整。更晚步骤非有限时原train最终指标取上一有限步，当前共享执行器的failure_reason可能被标成finite_threshold，需显式记录实际失败原因和step。A01–A04全部有限，结论不受这项限制影响。

理论定位参照 Chizat、Oyallon、Bach 的 [On Lazy Training in Differentiable Programming](https://proceedings.neurips.cc/paper/2019/hash/ae614c557843b1df326cb29c57225459-Abstract.html)，其缩放/初始线性化与 nonlinear training 的区别已核验原摘要和 arXiv/Semantic Scholar。A01 的可取贡献是可复核的标签偏移干预与 head 必需解释被反驳；尚未完成新机制或跨 recipe 预测理论，不宜把“发现 feature learning”作为新颖性主张。
