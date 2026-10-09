# B方向交接：有效时间与容量

B01/B02/B03全部完成，没有运行中的B worker。按用户最新要求只完成开源和交接，等待新目标；下一研究建议只是待接手问题，未启动B04。coordinator负责Git提交和短KB应用；此方向只写自己的文件。

## 实际计数与耗时

|study|保存cells|optimizer训练|另计解析轨迹|两阶段cell执行秒|
|---|---:|---:|---:|---:|
|B01_effective_time|192|192固定特征校准|0|10.52|
|B02_nonlinear_clock|72paired|72真实MLP＋72切线|0|13.16|
|B03_kernel_direction|24|24真实MLP|72线性kernel probes|7.43|

合计360次optimizer训练，其中96为真实MLP，72为训练了同optimizer的切线，192为解析校准。另72条解析kernel递推不算训练。无model API/solver calls。所有结果成功且有限；30秒左右是小张量累计cell计算，不包括写代码/分析/协调。研究产物第一至最后时间约33分钟。

B01发展测量中含checkpoint/restart，首至末约24.90秒；其余各phase约2.6–6.7秒。B02/3每cell秒是实验函数walltime；独立方向并行时不能把这些累加成全prototype walltime。

## 已得到的知识与未决问题

[总报告](report.md)串起三轮。固定谱理论和SGD启动校准精确；同名义U不同目标可以完全不在同一拟合阶段。Adam参数正交旋转改变了进度，不能导出坐标无关的SGD/Adam常数。

真实MLP与固定初始Jacobian切线模型第一步一致，后期分开。8/8事前封存新函数SGDunits获得非线性train增益；其中6/8test反而更差。B03同残差/同trace交换核，4个封存function/widthunits都保留方向收益，12个相关定向checks通过。soft_bump width8SE较大，不写成每seed都稳健。

没有被隐藏的预注册失败；存在明确边界：初始谱不能预测晚期，核漂移norm与曲线误差相关弱，train增益不保证test增益。文献只实时核对3篇原文摘要；未完成完整新颖性审核，不称论文级统一理论。没有新sol涨分数据。

## 封存顺序与恢复

以下UTC时间来自保存metadata；verify对B02/3以saved_at−cell_seconds重建开始，预测必须先于开始。

|OOD study|预测保存UTC|第一cell开始UTC|最后cell完成UTC|
|---|---|---|---|
|B01|2026-10-05 18:02:00|18:05:16|18:05:21|
|B02|2026-10-05 18:14:57|18:15:10|18:15:16|
|B03|2026-10-05 18:23:26|18:23:38|18:23:43|

所有OODconditions在各study首轮训练前封存；B02/3方向数值预测在开发后、OOD前另存ood_predictions.json。不再把已经使用的函数叫untouched。B01先训练3cells退出，再续跑，其metadata和arrays SHA完全未变，resume_verification.json证明未重复成功训练。

~~~sh
python directions/B_effective_time_capacity/verify.py
python studies/B01_effective_time/analyze.py --phase development
python studies/B01_effective_time/analyze.py --phase ood
python studies/B02_nonlinear_clock/analyze.py --phase development
python studies/B02_nonlinear_clock/analyze.py --phase ood
python studies/B03_kernel_direction/analyze.py --phase development
python studies/B03_kernel_direction/analyze.py --phase ood
~~~

verify只读已保存证据、不训练，独立用einsum复算谱/切线/核probe，最大loss误差<1e−12；B01原NumPy BLAS警告保留，独立einsum核验无警告。包需NumPy/Torch/SciPy/Matplotlib，所有研究模型源码保存在study/executed。

恢复某phase用原run.py --phase development或--phase ood：先核对prereg/source/arrayhash，成功cell直接复用；worker.lock避免同study并发。当前均terminal；不要为了验证再开训练。proposals在proposal.json，三个条目分别关联具体report，coordinator决定是否合并。

## 接下来的科学问题

固定feature精确U依赖目标谱，真实feature学习改变谱形状；能否用少量目标相关谱量预言充分拟合以及何时开始泛化损害，尚未解决。后续应保留初始/晚期和test方向信息，而非继续拟合全局g常数。容量/宽度、lr、动量需独立干预，且在新封存函数/recipe上预测；不能直接沿用B02/3作第二次OOD。
