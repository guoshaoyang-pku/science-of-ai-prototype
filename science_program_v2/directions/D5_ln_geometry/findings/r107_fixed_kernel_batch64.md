# r107：保存 batch64 顺序未修复固定核方向

只把全批量算子换为旧实际 batch64 顺序，方向匹配仍为0/20，改善0对。共享Linear+LN affine初始化核、eta=.001、a=3、T=256及无decay线性化保持不变。batch64的LN−noLN ΔC在20/20配对为正，范围+0.014394735835437978～+0.04279174328309669；保存真实train ΔC在20/20为负，范围−0.2880592148066915～−0.027001543968979136。P1预测匹配0/20得到支持。相对保存全批量ΔC，20/20的绝对变化≤.005，最大0.0006713233049124474；P2支持。这里“支持”指预注册的无改善预测成立，并非恢复了真实训练方向。

## 读取、旧收尾与预注册

依次读取GOAL、AGENTS、state、task、全部既有findings/report及inbox。inbox明确要求六节，优先于通用七节骨架。按其指令将失败与反例全文降为方法节小节；数字顺序、全部展示公式和表格行不变。重排单独commit f8dd6659f61a884b51195dceda52e192f2bb032e，并追加指定处理标记。

r089科学收尾8c11329dcc91aec9e82ab7b09938a71d692270d1与预注册99a9ed117c73bcd79f10e5b37deca5e900603442均为HEAD祖先。59个历史收尾对象与提交匹配，535个旧保护文件及40个r089成功结果文件hash/mtime不变。只读复算旧summary一致。冻结verify.py读取不存在的NPZ kernel_pin字段的错误保留，不报告它通过。此前独立核验通过核重构、eigh传播、VJP及初始预测核验。

本轮唯一预注册commit 4e151b4e16ec198308b579fccd1de2c87397e32b同时冻结合同、40个输入spec、593个历史文件hash/mtime与四份源码。首cell启动早于任何新结果分析、晚于提交5.647775秒。没有新传播pilot；预测来自已知全批量正ΔC、真实train负ΔC和小eta的未检验猜测。全部为development，非盲候选检验，不称sealed OOD。

预注册前冻结工具先遇到旧回执字段名KeyError，再遇到既有被git忽略的pycache。正确字段为r089_result_pins；扫描排除旧cache，文件未改。两次失败均在预注册及新传播前，原始失败另存。没有混用冻结提交，没有新训练或新核测量。

## 批次、公式与执行

旧训练120/120 cell保存完整256×64 int64 batch_indices。核对旧NPZ SHA、Linear初值、batch hash和源码重建全通过。顺序为torch.manual_seed(seed)→原Model构造→256次torch.randint(0,256,(64,))，每步有放回抽样。重复索引按次数累计。当前/历史PyTorch2.13.0。同函数同seed的LN/noLN与三offset顺序逐位一致，总共10个不同序列。本轮直接读取并冻结旧保存顺序，因此可称复用实际批次；仅构建初始化模型核对随机序列，没有forward/backward训练或重新测核。

K=JJᵀ/n，n256，b64。每批残差递推为r[t+1]=r[t]−(2eta n/b)K[:,B_t]r[t][B_t]。B_t是有重复的有序样本集合。offset传播令v0=1，v[t+1]=v[t]−.008K[:,B_t]v[t][B_t]，C=9*mean((v256−mean(v256))²)。一“步”指一个batch更新，256步不是256个epoch。LN核直接读取r089 total_kernel，noLN核读取r077 kernel；head固定，Linear与LN affine都包含在LN核中，核不居中、传播过程中不去均值，仅终点去均值算C。理想精确加性offset与同批次顺序下，该chord公式由线性递推解析导出；真实训练指标从保存的float32标签与终点预测另算，保留平移量化差别。

40/40评价cell完整，8个函数×recipe条件各5seed，20个配对只含四函数。复用120个旧训练cell和40个已保存核；0新训练、0新Jacobian/核测量。累计评价1.365258083秒。先1cell独立核验，再恢复新增39；完整后再次恢复新增0、复用40。成功cell不覆盖。每cell保存完整257×256 v轨迹、旧核、batch索引、旧train数组、全批量终点向量及源码/数据pins。

| 函数 | batch64 ΔC：均值 [配对seed t95] | batch64−全批量 ΔC：均值 [配对seed t95] | 方向匹配 |
|---|---:|---:|---:|
| trigonometric8 | +0.031636951 [+0.018082021, +0.045191880] | -0.000292725 [-0.000497017, -0.000088433] | 0/5 |
| quadratic8 | +0.025306186 [+0.018362451, +0.032249922] | +0.000021371 [-0.000132313, +0.000175054] | 0/5 |
| interaction12 | +0.023456081 [+0.016480307, +0.030431854] | +0.000040007 [-0.000474907, +0.000554920] | 0/5 |
| radial12 | +0.023599606 [+0.012741354, +0.034457857] | -0.000181732 [-0.000534971, +0.000171508] | 0/5 |

## 核验与边界

独立更新以bincount统计重复次数，再对256个唯一列求和，独立轨迹逐步传播；40cell的所有256步与保存轨迹最大绝对差4.996003611e-16。math.fsum独立chord误差1.040834086e-17，真实train chord误差2.220446049e-16，均低于预注册1e−12。旧核、旧train数组、实际batch与全批量向量逐位一致，summary从结果复算一致。593个旧文件及80个新结果+1个summary的hash/mtime在核验/恢复期间不变。

结论只区分这个固定初始化核下的batch算子。它否定“只换成旧实际batch64顺序即可修复当前固定核方向”的候选。它不识别核漂移、weight decay、非线性训练/test动力学、因果或sealed OOD。5seed区间覆盖初始化变化，不是函数总体概率。旧条件数/Rayleigh/coupling/一步多步/affine失败、冻结核验脚本缺陷与matmul根因未定均保留。

下一小问题优先0新训练：保持本轮全部保存核和实际batch，单独加入原SGD参数weight_decay=1e−4，比较方向是否仍0/20。先定义offset差的递推v[t+1]=(1−eta*wd)v[t]+eta*wd*1−.008K[:,B_t]v[t][B_t]，不能只乘衰减而漏常数项。先注册新数值区间、pins与单一源码commit，再计算。此候选仍不代表实际非线性训练或test机制。

证据：studies/r107_fixed_kernel_batch64/preregistration.json、summary.json、executed/report_reorder_audit.json、executed/precommit_audit.json、executed/precommit_failure.json、executed/input_manifest.json、executed/receipt.json、executed/first_cell_verification.json、executed/saved_evidence_verification.json、executed/recovery_verification.json。入口executed/run.py、analysis.py、executed/verify.py，指定Python -B。科学收尾与其核验回执将分别提交。
