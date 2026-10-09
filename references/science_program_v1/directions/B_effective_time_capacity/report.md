# U需要目标相关谱，容量改变也会改变方向

当前最清楚的结果是：把优化器预算、核总尺度或初始谱兑换成一个U，都不足以解释后期拟合。非线性特征学习在新函数上带来训练增益，却经常损害test；把学习后的核调到与初始核相同trace，方向收益仍保留。下一步应研究目标相关谱的演化和泛化，不能只再校准一个平均梯度常数。

## 三轮证据

|轮次|问题与实际测量|结果|
|---|---|---|
|[B01](../../studies/B01_effective_time/report.md)|192固定特征校准训练，d8开发/d16新谱|动量精确递推通过；相同U快模态45步充分拟合、慢模态1024步仍保留约74%loss；Adam坐标依赖。|
|[B02](../../studies/B02_nonlinear_clock/report.md)|72真实MLP＋72同初始化切线训练|初始谱解释SGD第一步，后期失败；8/8封存新函数SGDunits获train增益，其中6/8test更差。|
|[B03](../../studies/B03_kernel_direction/report.md)|24新MLP训练＋72解析核probe|相同残差、相同trace仍有方向收益；新d7/uniform/w64的4units定向预测通过，续训gain.114–.132。|

累计360真实optimizer训练，其中192为解析校准；72线性probe另计。无新增model API/solver calls。3轮源码、预测、全部数据/数组和hash已保存；独立verify从J/K复算预测，最大loss误差<1e−12。

## 可用公式与尚未得到的阈值

固定特征半MSE，普通SGD的目标模态eᵢ(t)=(1−ηλᵢ)ᵗeᵢ(0)。动量递推需保留启动和二阶状态；ηt/(1−μ)仅为慢变近似。只知道lr/T/m不能断言拟合充分，因为λ和目标对齐决定慢模态。

对函数空间瞬时kernel，目标残差Rayleigh R(r,K)=rᵀKr/(rᵀr)给出一阶局部进度；完整谱决定长期固定kernel收敛。B03显示特征学习改变了r相关谱，trace无法替代。Adam逐坐标预条件器也排除了坐标无关的固定SGD换算；历史|g|≈.01/.03仍只在其具体题库内是经验。

还没有得到跨MLP/Adam的精确U、最低充分U或普适容量N；尚未分辨一般feature学习的泛化收益/损害。当前短KB应保留这些条件，不以实验链已运行完成替代理论完成。

## 与已有理论的关系

[Jacot, Gabriel, Hongler：Neural Tangent Kernel](https://arxiv.org/abs/1806.07572)已描述GD的函数kernel动力学和无限宽固定kernel极限；[Chizat, Oyallon, Bach：On Lazy Training](https://arxiv.org/abs/1812.07956)说明线性化/惰性训练依赖scale而非仅参数量。这两条原文摘要已实时核对；固定谱与非线性差异不是本研究的新理论。

[Lewkowycz等：Catapult mechanism](https://arxiv.org/abs/2003.02218)原文摘要指出大/小lr动力学可定性不同。此轮没有扫到catapult phase，不借其名解释当前低lr现象。文献保存摘要页面与SHA，尚未做完整新颖性审核；论文撰写应先读全文与更近的target alignment工作。

## 下一轮的可区分问题

同trace实验否定“只是全局时间”；下一问题是能否用少量目标加权谱量预测何时进入稳定拟合及容量比较。拟在新recipe中测残差投影、慢模态权重、更新的函数方向和test谱，先检验局部Rayleigh与完整谱的差异，再冻结新外推。已有sine/radial/product等全部归development，不能重称untouched。

~~~sh
python directions/B_effective_time_capacity/verify.py
~~~

verify不训练、不调用模型，核对每份源码/数组hash、预测先于OOD产物、完整固定特征/切线递推及equaltrace核probe。独立结果在verification.json；coordinator负责原子KB应用和Git提交。
