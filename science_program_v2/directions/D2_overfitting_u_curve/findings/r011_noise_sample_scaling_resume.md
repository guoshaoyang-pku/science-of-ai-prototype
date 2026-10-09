# r011：噪声与样本量折叠的反例及近似预测

程序第11轮、D2第3轮恢复r003的同一个小问题：固定ReLU特征下，条件期望clean-risk最低点步数能否按n/σ²折叠，并由单一幂律跨样本量预测？完成60/60 cell。严格折叠预测P2被推翻：40/44配对在两倍内。幂律预测P3达到原判据：57/60留出预测在两倍内，要求至少48/60。全部为development。

## 合同与提交时序

已按序读取GOAL、AGENTS、中心状态、task、旧findings与report；没有inbox。原合同、执行器和分析脚本由supervisor在2026-10-06T17:38:53+08:00提交，commit为 `d5fae2cdbf2eaf66f598e74f001684a4be98ba4d`。本轮19:14:20核验三文件与该提交及HEAD `37ee424dc87f206e0226f5c3b56a0a048cebdae2` 逐字节一致；首次训练始于19:14:34。未修改预测或pinned源码。合同中round=3和draft字段保留原值，本次执行轮为11。

`.git` 已有POSIX owner写位，但沙箱拒绝创建index.lock；本轮预注册阶段git add返回128。使用已有匹配提交恢复，没有绕过权限。旧失败summary另存为 `executed/summary_r003_blocked.json`，旧findings和提交失败记录保留。

固定d4对称Gaussian、目标x₀+0.5x₀x₁、冻结ReLU width128/scale .03、训练特征居中与RMS归一化、含bias零初始化head、float64、全批量GD η=.3/mom0/nodecay。扫描n32/64/128×σ².25/.5/1/2/4，共15 recipe×四seed411–414；逐整数步0..16384取首个全局最小点。跨n保持W seed，重新抽输入并重算clean标签RMS；不是只改变样本量的因果对照。

先1 cell核验，再恢复剩余59 cell，成功结果未覆盖。训练与保存累计21.277秒。NumPy matmul出现divide/overflow/invalid警告，原stderr保存在run receipt；所有保存数组有限。独立einsum逐项乘积与显式噪声响应方差复算的终点期望风险最大误差9.59e−14。该核验说明结果数值一致，不确定警告的底层原因。

## 原预测与配对结果

P1全部通过：noisy train MSE最大相邻增量−5.10e−9，实际clean-GD与谱信号风险的全步最大误差6.22e−15，直接终点clean-risk最大误差6.14e−15。谱递推与bias/variance分解是已知理论，不登记为新发现。60 cell最优点均在内部，范围8–555步；全部满足原U型描述判据：终点风险比最低点高至少.05。

下表为较大n的t*除以较小n的t*。范围含全部四seed；44配对对应11个recipe配对组，每组四次重复，不算44种独立recipe。

| n配对 | n/σ² | t*比值seed范围 | 两倍内 |
|---|---:|---:|---:|
| 32→64 | 128 | .5000–2.8125 | 3/4 |
| 32→128 | 128 | .5000–2.7500 | 3/4 |
| 32→64 | 64 | .7229–1.6957 | 4/4 |
| 32→128 | 64 | .6867–1.8043 | 4/4 |
| 32→64 | 32 | .6522–.7188 | 4/4 |
| 32→128 | 32 | .6250–1.6207 | 4/4 |
| 32→64 | 16 | .6842–.8667 | 4/4 |
| 64→128 | 256 | .4103–1.2802 | 3/4 |
| 64→128 | 128 | .5421–1.3583 | 4/4 |
| 64→128 | 64 | .9500–1.4571 | 4/4 |
| 64→128 | 32 | .8696–2.3500 | 3/4 |

四个反例保留：seed414、比值128时n32/.25的t*=64，n64/.5为180、n128/1为176；seed412、比值256时n64/.25为390、n128/.5为160；seed413、比值32时n64/2为20、n128/4为47。前两个共享同一n32结果，不是独立重复。全部log₂配对值见summary。

## 留出样本量的幂律预测

每次用另外两个n的40 cell拟合log(t*)=a+b log(n/σ²)，预测留出n的20 cell。留出n32/64/128各通过19/20，共57/60，P3 supported。三次斜率b为1.05557/.96928/1.09019，截距a为−.33541/.09033/−.39273；参数只描述本合同，不给出普适标度律。

| 失败cell | oracle t* | 预测步数 | 倍数误差 |
|---|---:|---:|---:|
| n32，seed411，σ²=.25 | 274 | 119.853 | 2.286 |
| n64，seed411，σ²=2 | 15 | 31.488 | 2.099 |
| n128，seed412，σ²=.25 | 236 | 606.811 | 2.571 |

P3容许部分误差，故其通过与P2失败可以同时成立。预测利用其他development条件的clean-risk最低点；不是train-only早停或封存OOD。

## 复算、解释与交接

原analysis.py逐步重新解压NPZ，首次全量分析被中断；结果未改。新增executed/run_analysis.py仅缓存读盘数组，通过runpy执行未改动的pinned analysis.py，6.544秒完成。中断记录、缓存入口hash和原始stderr均保存。60 cell的metadata、NPZ、数据合同、源码与receipt hash全部通过独立核验。

n/σ²给出多数cell的粗略预测，但不足以逐seed保证最低点折叠。经验谱、输入抽样和训练标签RMS变化均可解释反例，本轮未分离这些因素。oracle需要clean train/audit标签；不外推learned hidden、Adam、mini-batch、CE、无限训练或无test标签早停。

证据：[summary](../studies/r003_noise_sample_scaling/summary.json)、[训练前提交核验](../studies/r003_noise_sample_scaling/executed/round011_static_audit.json)、[首cell核验](../studies/r003_noise_sample_scaling/executed/first_cell_verification.json)、[全量核验](../studies/r003_noise_sample_scaling/executed/saved_evidence_verification.json)、[本轮记录](../studies/r003_noise_sample_scaling/executed/round011_completion.json)。

下一轮先核验本轮产物的收尾提交，不重跑或覆盖60个成功cell。建议只研究归一化对照：保留n32/.25与n64/.5的四seed原始输入、W和特征，仅将clean标签的train mean/RMS归一化改为共享解析population mean=0/RMS=√1.25，检查比值128反例是否保留。新数值预测与pinned源码先commit，再训练；该对照尚未执行。收尾提交因只读.git受阻，本轮记partial。
