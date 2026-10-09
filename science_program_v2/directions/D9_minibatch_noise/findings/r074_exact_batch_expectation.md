# r074：精确 batch 期望的最低点是否仍在两倍内

固定 D2 n=64、d=4、冻结 ReLU width128+常数、零初始化 head、float64 MSE、SGD 无动量/无衰减、η=.1、B=8、σ²=1、seed411/412 原 data/ε、整数窗口0..2304和同标签 full 分母49/306。仅计算独立无放回 batch 的精确一、二阶矩，无新训练、无新 batch 流。预注册及执行/分析/核验源码冻结于 d8943a9，早于矩计算。2/2 精确 cell 完成，累计2.443217秒；1 recipe、2配对数据seed、2复用full控制、4分别描述的M64块共256旧轨迹。全部development。

预注册前只做解析闭合与输入span审查，未计算新风险。两个训练span均秩36，最小保留奇异值.020925/.034717，SVD截断阈值2.29e−13，重建误差≤1.55e−15。使用实际梯度系数2η。有限总体因子γ=(n−B)/(B(n−1))=1/9；线性更新的一、二阶矩闭合及均值与full相等为已知理论，本轮不把公式登记为新发现。

| seed | 注册t*范围 | 精确条件batch期望t* | 同标签full t* | 比值 | 最低期望风险 | 其中协方差风险 | 最低点相邻最小余量 |
|---|---|---|---|---|---|---|---|
| 411 | 25–98 | 50 | 49 | 1.020408163 | .250249570939 | .022759234690 | 1.749632435e−5 |
| 412 | 153–612 | 315 | 306 | 1.029411765 | .268837195946 | .019800149753 | 2.579671404e−8 |

P1支持2/2。精确期望指固定data/ε下对每步batch随机性积分，实际使用float64；没有Monte Carlo抽样区间，不是对ε积分，也不是跨数据统计保证。两个条件的期望最低点均不越界；旧M8注册反例继续有效，不追溯改判、不补成广网格24/24结论。

| seed | M64独立块 | 块t* | 块−精确t* | 块−精确比值 | 风险曲线RMSE | 风险曲线maxabs |
|---|---|---|---|---|---|---|
| 411 | r0–63 | 57 | +7 | +.142857 | .005280278 | .017178030 |
| 411 | r64–127 | 56 | +6 | +.122449 | .004715794 | .024000393 |
| 412 | r0–63 | 235 | −80 | −.261438 | .004135192 | .013354175 |
| 412 | r64–127 | 393 | +78 | +.254902 | .004195469 | .017044643 |

这些差值只描述两个固定条件下的两个独立块，未合并、未重新bootstrap、未作显著性检验。最低点相邻余量很小的seed412保留较大的块间步数差；这是平坦最低点与抽样变化共同出现的测量，未单独干预其因果作用。

## 核验与边界

开始时核验r041回执中的21个当前文件与科学收尾85dc1e79字节匹配，冻结94d66e1为收尾祖先。285个旧研究文件hash/mtime不变。新receipt、数组、数据与冻结源码pin一致；恢复仅核验/跳过两个cell，19个已有本轮文件hash/mtime不变。

独立实现使用不投影的64维样本坐标w=Xᵀα、raw二阶矩与单/双样本inclusion概率，未调用主矩递推或其basis。风险曲线最大差seed411/412为5.034861417e−12/2.164435298e−12，最终均值head差≤6.955547249e−14，两个首argmin完全一致；这些是独立实现差，不是严格误差证书。B=n的协方差严格为0，full曲线相对保存控制差≤1.526556659e−14。n4/B2确定性fixture枚举6batch，一步均值误差0、协方差误差6.938893904e−18。

新执行/核验使用einsum，无运行警告；预注册前span可行性检查的matmul仍出现旧类警告，输入和保存结果有限。旧r025/r029 P1/P2失败、固定ε与期望ε分母混杂、旧核验训练后改pin历史和matmul根因未定全部保留。不外推连续η、其他batch、n、优化器、learned features、无限训练或train-only早停。

下一小问题候选：保持data/ε、η=.1、σ²=1、seed411/412、T2304及分母49/306，仅B8→4，先注册两个精确期望t*的数值范围与pinned源码/输入hash再计算；本轮不预测或预先计算B4风险。优先0新训练。

## 证据

- [预注册](../studies/r074_exact_batch_expectation/preregistration.json)、[可行性审查](../studies/r074_exact_batch_expectation/executed/feasibility.json)、[旧文件与收尾审计](../studies/r074_exact_batch_expectation/executed/input_audit.json)。
- [执行源码](../studies/r074_exact_batch_expectation/executed/run.py)、[分析源码](../studies/r074_exact_batch_expectation/analysis.py)、[汇总](../studies/r074_exact_batch_expectation/summary.json)、[逐cell回执](../studies/r074_exact_batch_expectation/results/receipt.json)。
- [独立核验源码](../studies/r074_exact_batch_expectation/executed/verify.py)、[独立核验](../studies/r074_exact_batch_expectation/executed/verification.json)、[恢复核验](../studies/r074_exact_batch_expectation/executed/resume_verification.json)、[最终提交核验](../studies/r074_exact_batch_expectation/executed/final_commit_verification.json)。
