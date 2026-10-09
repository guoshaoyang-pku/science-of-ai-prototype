# r106：慢损失占比0.75通过，失败预测被推翻

固定谱比100、局部h/b与原评价约定，仅用保存初始慢/快损失权重0.75/0.25生成双段候选。3/3坐标cell同时通过，RMSE=0.027824997、maxabs=0.049822548；最大误差余量仅0.000177452。预注册P1/P2均refuted。新增训练0、拟合0。

唯一问题：同一加权候选在α=0.75是否通过原0.03/0.05阈值，误差是否落入新数值区间？候选C(t)=αH_s(t)+(1−α)H_f(t)，H_i=1/[1+(t/h_i)^b]；h_i=log2/[-2log(1−ηλ_i)]，b=2log2。α只读保存mode_weights，不拟合。t/h单位步，b/α无量纲。n=d=2、float64、固定特征half-MSE、零初始化、全批量SGD η=0.1/mom0/nodecay、谱[0.01,1]、L0=1、T10000，平台1/0，原150点等权整数对数网格0..4997。一个recipe×三个QR坐标seed，不算独立数据重复或统计CI。

| seed | RMSE | maxabs | 最大残差步 | 同时通过 |
|---:|---:|---:|---:|---|
| 0 | 0.027824996906011187 | 0.04982254827281055 | 1583 | 是 |
| 1 | 0.02782499690601169 | 0.04982254827281064 | 1583 | 是 |
| 2 | 0.027824996906011888 | 0.04982254827281068 | 1583 | 是 |

P1预期3/3均RMSE≤0.03且maxabs>0.05，三个cell的maxabs≤0.05推翻预测。P2预期RMSE在[0.025,0.035]、maxabs在[0.050,0.060]，三个cell最大误差均低于下界0.000177451727189325–0.000177451727189450。RMSE部分通过不能把联合区间预测改判为支持。区间来自已见等权与α=0.25、旧单模态局部公式误差量级，非盲/封存OOD且未校准；注册前未计算本轮公式误差。

同seed公式减原单段LS的RMSE范围[−0.04191871668915578,−0.04191871668915522]、maxabs范围[−0.06492670954636501,−0.06492670931944299]。原LS RMSE/maxabs=0.069743714/0.114749258。两项改善同时改变段数和参数取得方式，不能独立归因。慢/快初始损失分配改变还伴随目标参数和初始梯度改变，不能证明独立动力学机制。

本轮按序读取必要文件；inbox要求六节，全文重排保留全部数字与结论，并将失败记录置于方法支持小节。重排提交2605e7b，按指定时间追加已处理标记。冻结前工具JS SyntaxError与重排Python字符串SyntaxError未写入文件或进行科学计算，纠正后完成重排；科学执行后一次文档工具JS SyntaxError未启动shell或写文件，修正后落盘。两只读子任务作历史提交/合同与本轮静态审查，无科学训练或旧条件重评。

唯一冻结232da3a19d8b43ce2ed421e67bb71535191e82c1包含预注册、20个输入hash及5份源码，早于每cell公式计算开始至少45.608635秒。执行先检查唯一creation commit、冻结源码/输入hash和历史hashmtime，然后逐cell独占保存。首次执行0.291165秒，0训练/拟合。六份旧收尾回执59/29/29/30/31/27个blob均匹配，原283历史/6新结果hashmtime通过，扩展305历史文件与81份旧成功cell合同全程通过。

独立标量log/exp/fsum复算本轮结果：曲线最大差2.220446049250313e−16，逐模态差3.3306690738754696e−16，指标差3.469446951953614e−18；直接幂递推与保存损失差≤1.9984014443252818e−15。初始模态损失归一化与保存幅度一致。恢复全局先扫receipt/结果，混合commit、冲突pin、额外/孤立/不完整结果均拒绝；本次仅加载3成功cell，新增评价0，6结果文件与305历史文件hash/mtime不变。

α=0.25同公式失败与八个等权格点通过、旧r1局部公式/纯快/α=0.01原LS失败保留，未重评或覆盖任何旧成功cell。本次通过余量小，仅此recipe/幅度/网格/平台；不推连续α或谱比阈值、参数可辨认、误差重叠独立机制、物理merging、grokking、MLP、learned features、其他优化器、mini-batch、test或OOD。下一小问题仅仅读保存α=0.9三曲线，保持同合同，先注册新具体数值预测与pins再评价，不先算误差。

证据：[预注册](../studies/r106_weighted_slow_formula/preregistration.json)、[summary](../studies/r106_weighted_slow_formula/summary.json)、[执行源码](../studies/r106_weighted_slow_formula/executed/run.py)、[分析源码](../studies/r106_weighted_slow_formula/analysis.py)、[独立核验](../studies/r106_weighted_slow_formula/executed/independent_verification.json)、[恢复核验](../studies/r106_weighted_slow_formula/executed/resume_verification.json)、[历史审计](../studies/r106_weighted_slow_formula/executed/previous_closeout_audit.json)、[只读审查及写作自查](../studies/r106_weighted_slow_formula/executed/independent_review.json)、[交付验证](../studies/r106_weighted_slow_formula/executed/delivery_validation.json)、[收尾核验](../studies/r106_weighted_slow_formula/executed/final_commit_verification.json)。
