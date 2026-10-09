# B/C 发布证据独立复核

B01–B03、C01–C05 的912份保存cell和数组均有限、数量完整。方向verifier已通过；另独立核对B的request/preregistration/源码字段、cell-id、数据生成和模型forward，并重算关键结果。审查未训练、未调用模型、未读最终benchmark test。完整记录在[evidence/release_audit.json](evidence/release_audit.json)，重跑[evidence/release_audit.py](evidence/release_audit.py)只读保存结果与解析计算。

## 数量和实测耗时

|Study|保存cell|实际轨迹单位|cell耗时合计|
|---|---:|---|---:|
|B01|192|192次固定特征optimizer校准|10.52s|
|B02|72|72次非线性MLP＋72次配对切线optimizer训练|13.16s|
|B03|24|24次MLP预训练；72条解析核probe另计|7.43s|
|C01|216|216条head轨迹|6.79s|
|C02|192|192条head轨迹|15.32s|
|C03|128|128条head/hidden联合或冻结轨迹|11.88s|
|C04|64|每recipe同时训练total/signal/noise三个head，共192条|14.07s|
|C05|24|每recipe一个clean＋16个noisy head，共408条|3.11s|

B合计360条optimizer轨迹：192条校准、96条MLP、72条切线；另72条解析probe。C合计624recipes、1136条head轨迹。cell/seed/噪声draw均不能直接当作独立任务数量。耗时是保存cell的运行时间合计，未包括分析、等待、恢复间隔或写作。B01开发从首cell到末cell跨24.90s，其中包含停止恢复间隔；不与10.52s训练合计混用。

## 复核后的具体发现

**B01是已知理论的校准。** 同d16谱、同SGD .05预算，快目标45步达到持续≤1%初始loss，慢目标1024步仍保留73.55%初始loss。独立Adam首更新公式误差≤2.74e−15；正交坐标会改变逐坐标预条件。该结果支持“U必须携带谱与目标对齐”的条件修正，没有提出新的二次优化定理，也没有预测Adam终点赢家。

**B02训练收益与泛化分开。** 新函数8个SGD function×width×recipe均比完整初始切线模型获得>5%初始loss的训练收益，其中6个test反而更差。保存初始化的手工NumPy forward误差≤5.56e−17，终点四组train/test MSE与数组一致。width变化同时改变初始核与非线性容量，不能称参数量单因素干预。

**B03确实区分了总尺度与方向。** 相同late residual、相同trace和共同lr下，late fixed kernel对四个封存OOD单位的512步probe额外相对收益为.114/.126/.130/.132；核的闭式谱轨迹独立重算误差≤6.78e−15，32置换读数全匹配。12个定向checks来自4个相关单位；soft_bump width8有一seed只得.0151收益，未达到.05阈值，单位三seed均值通过。这个干预交换固定核，不能写成真实非线性网络续训的实测收益，也未预测test普遍改善。

**C02有直接的机理干预。** 独立用奇部z/2、SiLU偶部z·tanh(z/2)/2和训练输入重建全部192cell的特征，最大差2.23e−15。干预系数未读labels，保存的未见交互test endpoint从Uniform .952600→.004278、Laplace 1.136482→.014986，每分布4/4seed改善。范围是antithetic输入、零bias、固定特征、已指定2048-step；不推广为普遍SiLU优势。

**C03/C04保留机制边界。** C03终点test数组复算支持hidden学习可以部分逃出small-SiLU瓶颈，scale .3下排序反转；初始head为零使第一步hidden梯度为零，不能以step1作hidden因果证据。C04终点signal/noise分解最大独立误差3.11e−15，属于固定线性特征bias/variance结构。C01的ReLU＋LN精确尺度不变P1预测失败（差.01585>1e−4）；C04 N4通过预注册中位数判据，但一seed最优step16早于ReLU step32。这些失败与反例都应随报告保留。

**C05是已知数据条件下的定量预测。** 独立NumPy重算完整forecast，最大误差1.30e−13；实际终点head/test数组与风险最大差1.20e−14。17/24最佳检查点精确，24/24在factor2内。原48/48风险检查有5个recipe的beststep与final均为8192，所以实际43个不同recipe×checkpoint全部在3MonteCarloSE内；5个最佳点也处于预算末端。最优时间只在doubling grid上比较，不能写成精确连续最优步数。forecast允许使用新train/test输入、干净train/test标签及noise variance，没有观察实际训练loss；因此验证的是条件风险公式外推，未验证隐藏任务的盲预测或可部署的无test-label早停器。

## 预测封存与公开复核的限制

|预测|保存时间|最早OOD执行起点|提前量|
|---|---:|---:|---:|
|B01条件与公式|1791223320.163|1791223516.660|196.50s|
|B02定向预测|1791224097.215|1791224110.318|13.10s|
|B03定向预测|1791224606.830|1791224618.506|11.68s|
|C02配置/预测|1791223745.851|1791223753.561|7.71s|
|C05全forecast seal|1791224697.200|1791224708.197|11.00s|

B使用JSON中的saved/started及duration；C02/C05从原worker日志恢复进程起点，并核对原文件mtime和duration。C02原数据文件mtime1791223753.564晚于seal；C05配置先封存，24forecasts于1791224697.108–4697.200保存后再seal，随后独立train进程运行。原私有Git中未找到这些B/C预测的事前commit；时间证据由保存记录支持，未经过外部时间戳见证。

公开复制会改变mtime，worker日志也因包含运行身份而排除。release_audit.json已保存原时间及来源，脚本在公开snapshot中明确只引用该审查记录，不能声称重新验证了已排除的日志。B既有verifier未逐cell核验request字段，C既有verifier不重算全部科学判据；本次补了这些request与核心数字，源码pin和完整624份C合同/数据/数组则由既有C verifier核验。全部本轮结论属于science测量；强模型KB收益、普适理论和论文新颖性仍未测得。
