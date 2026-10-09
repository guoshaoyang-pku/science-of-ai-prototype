# 偏移收益需要 hidden weights 学习，不需要 LN affine 更新

保留同一 LN010 forward，固定 head 后，再分别冻结或开放 hidden 参数。四个已见函数中，固定 LN gain/bias 仍保留 99.16–99.81% 的负 centered chord；只训练 hidden Linear weights 保留 99.95–111.22%。只训练 Linear biases 与 LN affine 则没有该收益，中心化误差的 chord 均值略为正。

## 区分实验

沿用 A02 的三角、二次、交互、径向函数，标签均值 −3/0/+3，配对 seeds100–104。模型为 d3w192/GELU/LN010/plain，SGD .001/momentum0，coupled wd1e−4，256 steps/batch64。输入、初始参数、完整 minibatch stream 相同；同一 frozen-head 基线的60个结果直接复用。

本轮180个新训练，约27.86秒，0失败；40个终点MSE>2全部保留。预注册提交74bf2b4早于执行。每个记录步核对被冻结参数无梯度、位移为零。这里的 centered chord 定义为两个非零offset的中心化残差MSE均值减去offset0的中心化残差MSE；负值表示大常数残差改善形状拟合。

| 函数 | 固定 LN affine | 只学 hidden weights | 只学 hidden bias/LN affine |
|---|---:|---:|---:|
| trigonometric8 | 99.81% | 103.48% | −1.22% |
| quadratic8 | 99.55% | 111.22% | −3.10% |
| interaction12 | 99.16% | 108.57% | −0.30% |
| radial12 | 99.23% | 99.95% | −0.49% |

数值为本干预的平均chord除以A02 frozen-head平均chord；负百分比表示收益反向，而不是训练失败。五个seed的配对区间保存在 analysis.json。三项事前判据各为4/4函数通过，不能把相关的12项判据算作12个独立目标。

## 当前机制与边界

A01反驳“head必须增长”；A03在相同width下区分LN有无；A04表明LN affine可训练性也不是必要因素。当前recipe中，固定LN几何加hidden weights学习足以保留收益，仅bias/affine拟合常数不够。这支持常数初始残差经hidden权重路径改变形状拟合，尚未证明某个特定Jacobian方向或完整理论。

本轮是看过A02后的development区分实验，没有新增最终OOD、Adam控制或solver调用。固定特征SGD的凸性与真实hidden学习的非凸offset响应仍是不同对象；四个函数内的seeds只测条件内变化。下一目标可以对完整初始切线模型、晚期目标相关核或未见recipe提前预测，不能把本轮作为其结果。

## 离线复算

~~~sh
python studies/A04_ln_affine/analyze.py
~~~

run.py使用 executed/host 中固定执行器；已保存cell会核对contract和NPZ hash后复用。开源归档的机器元数据与原始元数据若有变化，应由release映射明确标记，不能覆盖原始成功测量。
