# r088：保存慢损失占比 .25 的加权公式边界

固定 r=100 与局部 h/b，仅读保存曲线的初始慢/快损失权重 .25/.75 生成双段候选，三个坐标seed均 RMSE=.022740946≤.03，但最大误差=.053989623>.05。候选同时通过率0/3；“RMSE通过而maxabs失败”与新数值区间两项注册预测均通过。本轮新增训练0、拟合0。

唯一问题是：等权公式的通过是否能直接延伸到同谱比的 α=.25？候选为 C(t)=αH_s(t)+(1−α)H_f(t)，H_i(t)=1/[1+(t/h_i)^b]，H_i(0)=1；h_i=log2/[-2log(1−ηλ_i)]，b=2log2。α只从保存mode_weights取得，未拟合。t/h单位均为步，b与α无量纲。n=d=2、float64、固定特征half-MSE、全批量SGD η=.1/mom0/nodecay、λ=[.01,1]、L0=1、零初始化、T10000，平台1/0，原150点等权整数对数网格0..4997。

| seed | RMSE | maxabs | 最大残差步 | 两阈值同时通过 |
|---:|---:|---:|---:|---|
| 0 | .02274094575736451 | .053989622779292756 | 15 | 否 |
| 1 | .022740945757364756 | .053989622779293034 | 15 | 否 |
| 2 | .022740945757364902 | .05398962277929331 | 15 | 否 |

RMSE余量 .007259054，maxabs超出 .05 约 .003989623；跨坐标seed指标差≤5.56×10⁻¹⁶。1个recipe×3个cell，seed仅QR坐标数值稳健性，不能当3个独立样本，也不计算统计CI。P1要求3/3均RMSE≤.03且maxabs>.05，结果supported；P2要求3/3分别落在闭区间[.020,.030]/[.050,.060]，结果supported。区间来自既有development误差量级，未校准，非盲发现或封存OOD。候选失败是保存的描述反例，未发生注册预测失败。

同seed原单段LS RMSE=.04090645242066033–.040906452420660454、maxabs=.17433680592279444–.1743368064149926。公式减原单段LS的RMSE范围[−.018165506663295944,−.018165506663295427]，maxabs范围[−.1203471836356993,−.1203471831435014]；这是同曲线描述性对照，段数和参数取得方式同时改变，不能单独归因于段数或阶段机制。误差降低仍不满足双阈值。

开头按inbox把报告31个原段落全文重排，未改已测数字与结论，独立提交69502bb，追加指定已处理标记。预注册和19个输入hash、四份源码于 a4d15c2c8460143733358201405cc6d2426c8da3 冻结；执行先核验唯一commit、源码/输入hash及历史hashmtime，再评价。commit早于全部结果完成至少27.756991秒。首次执行.149907375秒，含合同与历史核验，未训练或拟合；源码独占创建每cell，不覆盖。全局恢复先扫描全部receipt与结果，混合commit、额外/孤立/不完整结果或冲突pin均拒绝；恢复只加载3个成功cell，新增评价0，6个新结果文件hash/mtime不变。

五份历史收尾回执59/29/29/30/31个提交blob全部匹配，原256历史与上一轮12新结果hashmtime通过，扩展283文件全程不变。独立标量加权公式与向量曲线差≤1.11×10⁻¹⁶、逐模态差≤3.34×10⁻¹⁶、指标差≤3.47×10⁻¹⁸；直接幂递推与保存曲线差≤8.89×10⁻¹⁶，初始逐模态损失归一化核验保存权重一致。谱递推与局部匹配推导是已知解析工具，本轮新证据是冻结描述候选的受控幅度边界。

没有重评或覆盖历史成功cell，旧r1局部公式与D4纯快/α=.01原LS失败保留。该结论只适用本recipe、α=.25、原网格及平台；不定位连续α或谱比临界、不把误差重叠当独立机制、不证明参数可辨认、物理merging或grokking，不外推MLP、learned features、momentum、mini-batch、test或OOD。下一小问题可只读保存α=.75三曲线，固定同合同、先注册新数值区间再评价；本轮未计算该条件。

证据：[预注册](../studies/r088_weighted_formula/preregistration.json)、[summary](../studies/r088_weighted_formula/summary.json)、[执行源码](../studies/r088_weighted_formula/executed/run.py)、[分析源码](../studies/r088_weighted_formula/analysis.py)、[独立核验](../studies/r088_weighted_formula/executed/independent_verification.json)、[恢复核验](../studies/r088_weighted_formula/executed/resume_verification.json)、[历史审计](../studies/r088_weighted_formula/executed/previous_closeout_audit.json)、[交付验证](../studies/r088_weighted_formula/executed/delivery_validation.json)、[收尾核验](../studies/r088_weighted_formula/executed/final_commit_verification.json)。
