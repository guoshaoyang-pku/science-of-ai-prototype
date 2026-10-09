# r085：仅 B8 降至 B4，精确期望最低点是否小幅延后

固定原 D2 n64/d4、冻结 ReLU width128+常数、零初始化 head、float64 MSE、SGD 无动量/无衰减、η=.1、σ²=1、seed411/412 原 data/ε、整数窗口0..2304及同标签 full 分母49/306。唯一变化为 B8→4；完成2/2新条件矩cell、0新训练，累计1.627101秒。直接读取两个旧B8的basis与full数组，未重算SVD、旧B8或full成功cell；1新batch recipe、2配对data seed。全部development。

事前先核验旧科学收尾26项历史字节及23项当前D9文件，报告原六节骨架符合要求，无inbox。预注册与四项source pin、14项input pin于40d29b1一次冻结，早于全部B4计算。预测依据是已见的B8最低点50/315及full49/306；γ由1/9增至5/21不能证明argmin延后，故注册有限数值范围检验该假设。注册前未算B4风险或端点，不称盲发现或封存OOD。

| seed | 注册 B4 t* 范围 | B8 t* | B4 t* | B4−B8 步数 | B4/full | 最低期望风险 | 最低点协方差风险 | 最近邻余量 |
|---|---|---|---|---|---|---|---|---|
| 411 | 51–70 | 50 | 52 | +2 | 52/49=1.061224490 | .277158450296 | .049492000902 | 5.913242543e−6 |
| 412 | 316–400 | 315 | 324 | +9 | 324/306=1.058823529 | .292429181526 | .043376178400 | 1.017070178e−8 |

P1支持2/2。两个最低风险相对B8增加.026908879357/.023591985580，最低时间变为1.04/1.028571429倍。该配对结果表明，在这两个固定data/ε条件，小批量减半后的条件期望最低点略晚、最低风险更高。风险新增项来自batch参数协方差；γ增强与协方差递推反馈共同变化，不能把风险或步数按15/7线性缩放。两倍内比值只是另报的描述，不是新注册的广网格保证。

## 方法核验失败与轮次判定

原分析要求B4均值坐标及mean_risk与保存B8逐位相等，此注册方法判据在两个seed均失败：坐标maxabs为2.309263891e−14/8.743006319e−15，mean_risk为3.885780586e−16/4.440892099e−16。原analysis.log报错及summary validation=false、round_result=partial全部保留；未改注册容差、源码或成功cell，也未重新生成summary。本轮因此为partial，不能宣称全部注册核验通过。

保存的旧basis为Fortran连续，新脚本读取后.copy()保存为C连续，basis值完全相等。内存顺序改变导致einsum舍入路径变化是候选解释；没有布局干预复算，根因未确证。解析均值递推不依赖B属于已知理论，不能据此把float64逐位失败改判为通过。

新B4独立64维样本坐标w=Xᵀα与raw二阶矩复算通过，未调用主矩递推或basis。seed411/412风险曲线差为5.476730180e−12/2.006728117e−12，最终均值head差≤6.813993814e−14，首argmin同为52/324；这些是实现差，非严格误差证书。数组有限、最低点窗内、协方差最小特征值非负；继承full数组逐位不变，其风险对保存full误差≤1.526556659e−14。旧B=n协方差0及枚举核验只读继承，没有重算。

302个旧study文件hash/mtime未变。全局results扫描与唯一冻结commit绑定通过；恢复仅核验/跳过2cell，18个既有本轮文件hash/mtime不变。新增执行与核验无运行警告。保留旧M8反例、旧期望标签分母混杂、旧核验训练后改pin历史及matmul根因未定。

边界为这两个已选固定data/ε、离散B4/B8、η=.1、σ²1、0..2304；不外推连续B/η、其他数据或标签抽样、n、优化器、learned features、无限训练、train-only早停。下一小问题优先0新训练，仅B4→2、其余条件和full分母不变，先注册新的最低点数值范围与pinned源码/input hash后才计算。均值对照须事前注册明确浮点容差与固定布局，不追溯放宽本次逐位判据，也不重算成功B4/B8/full。

## 证据

- [预注册](../studies/r085_batch4_exact/preregistration.json)、[执行源码](../studies/r085_batch4_exact/executed/run.py)、[分析源码](../studies/r085_batch4_exact/analysis.py)、[原汇总](../studies/r085_batch4_exact/summary.json)、[逐cell回执](../studies/r085_batch4_exact/results/receipt.json)；唯一冻结40d29b1。
- [旧证据及状态快照](../studies/r085_batch4_exact/executed/input_audit.json)、[原分析报错](../studies/r085_batch4_exact/executed/analysis.log)、[运行日志](../studies/r085_batch4_exact/executed/run.log)。
- [独立核验源码](../studies/r085_batch4_exact/executed/verify.py)、[独立核验](../studies/r085_batch4_exact/executed/verification.json)、[恢复核验](../studies/r085_batch4_exact/executed/resume_verification.json)、[收尾审计](../studies/r085_batch4_exact/executed/closeout_audit.json)、[最终提交核验](../studies/r085_batch4_exact/executed/final_commit_verification.json)。
