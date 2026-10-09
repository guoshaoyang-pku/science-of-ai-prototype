# r097：仅 B4 降至 B2，期望最低点是否继续小幅延后

固定 D2 n64/d4、冻结 ReLU width128+常数、零初始化 head、float64 MSE、无动量/无衰减 SGD、η=.1、σ²=1、seed411/412 原 data/ε、整数窗口0..2304及同标签 full 分母49/306。只将每步独立无放回 batch 从4改为2。2/2新条件矩cell完成，0训练，主计算1.744321秒；1新batch recipe、2配对data seed。复用保存B4的basis与full，不重算B4/B8/full，不新建MC轨迹或bootstrap。

| seed | 事前 t* 范围 | B4 t* | B2 t* | 延后步数 | B2/full | 最低期望风险 | 相对B4最低风险增加 | 最低点协方差风险 | 最近邻余量 |
|---|---|---|---|---|---|---|---|---|---|
| 411 | 53–75 | 52 | 55 | 3 | 55/49=1.122448980 | .333980100133 | .056821649837 | .105835131584 | 5.426518285e−6 |
| 412 | 325–400 | 324 | 343 | 19 | 343/306=1.120915033 | .343054486608 | .050625305083 | .093934691234 | 5.596599068e−8 |

P1支持2/2。仅这两个固定条件中，batch再次减半后期望最低点继续延后，最低风险继续升高。两倍内比值仅作描述。γ从5/21变为31/63，比例31/15；协方差反馈随之变化，不能据此给出最低时间或最低风险的线性缩放公式。均值递推不随B改变是已知理论，不登记为新发现。

## 冻结、核验与历史限制

开工按inbox六节骨架重排报告：只将“失败与反例”降为方法下三级节，正文逐字与全部数字保留，追加指定处理标记后单独提交d644e3b。git add -A同时纳入supervisor已有日志/轮次记录，未更改其内容。旧科学收尾29项历史字节与5项冻结pin、302旧文件sha256/mtime均独立复核通过；原B4 summary.validation=false/partial与analysis.log原样保留。

新合同、4项源码/审计pin和18项输入pin在唯一提交94beabf冻结后才计算B2。预测来自已看过的B4/B8/full最低点，粗略按γ比例外推约55/343步后注册较宽的53–75/325–400范围；未事前计算B2风险/端点，不称盲发现或sealed OOD。准备命令一次字符串换行语法错误发生于解析阶段，无写入或数值计算；修正准备命令后静态编译通过，冻结后的源码未改。

新合同事前固定basis、q、qa为float64 C连续。新B2对保存B4的均值坐标和mean_risk绝对容差分别为1e−10/1e−12，rtol=0；实际两seed均maxabs=0。此处逐位相等是新结果描述，不追溯改判旧B4对B8的严格逐位失败，也未做布局因果干预。basis/full数组逐位继承，full对保存控制最大差≤1.526557e−14，所有数组有限、argmin在窗口内、协方差最小特征值非负。

独立64维样本坐标raw二阶矩与单/双样本inclusion概率只计算新B2，未调用主矩递推或basis。风险曲线maxabs为4.772848783e−12/2.434441537e−12，最终均值head差≤6.813994e−14，首argmin同为55/343。这些是实现差，不是严格数值误差证书。旧B=n/枚举证据只读继承，没有重算。

332个旧study文件集合、sha256与mtime不变。恢复仅运行冻结run.py，核验/跳过2cell、新增0cell；18个已有本轮文件hash/mtime不变，summary/verification各只写一次。冻结analysis.py/verify.py自身可覆盖输出，恢复编排不再次调用；这一实现限制明确保留，未事后改冻结源码。独立静态审查的写一次保护/旧文件集合扫描建议在冻结后收到，文件集合另作只读审计通过。所有新执行/分析/核验日志无运行警告。

收尾工具先因正则转义错误失败，保留closeout.log；修正的是未冻结的收尾脚本。随后.git不可见，现场确认元数据目录名为.git_disabled，其中HEAD仍为本轮冻结94beabf。未重命名目录或改config；显式GIT_DIR/GIT_WORK_TREE后原冻结合同与收尾审计通过，科学源码与结果均未改变。提交排除元数据目录，暂存前目录又恢复为.git，改用当前.git显式绑定；本会话未执行重命名。具体事件记录见末尾证据。

## 边界与下一问题

仅原D2数据/ε、n64/d4/width128、零head、float64 MSE、无动量无衰减SGD η=.1、σ²1、两个已选seed、离散B2/B4及0..2304窗。旧M8失败、固定ε与期望ε分母混杂、旧核验训练后改pin、matmul根因未定均保留。不推连续B/η、跨data概率、新标签实现、其他n/优化器/learned features、无限训练或train-only早停。

下一小问题优先0训练：只将B2→1，其余同条件和full分母不变。先注册两个新的精确最低点数值范围、固定布局/容差、输入hash与pinned源码并单一commit，再计算。先实现summary/verification写一次或核验跳过保护及完整旧文件集合扫描；不重算旧B2/B4/B8/full，不预先算B1风险后称预测。

## 证据

- [预注册](../studies/r097_batch2_exact/preregistration.json)、[执行源码](../studies/r097_batch2_exact/executed/run.py)、[分析源码](../studies/r097_batch2_exact/analysis.py)、[汇总](../studies/r097_batch2_exact/summary.json)、[逐cell回执](../studies/r097_batch2_exact/results/receipt.json)；唯一冻结94beabf。
- [输入与旧收尾审计](../studies/r097_batch2_exact/executed/input_audit.json)、[运行日志](../studies/r097_batch2_exact/executed/run.log)、[独立核验](../studies/r097_batch2_exact/executed/verification.json)、[恢复核验](../studies/r097_batch2_exact/executed/resume_verification.json)、[收尾审计](../studies/r097_batch2_exact/executed/closeout_audit.json)、[最终提交核验](../studies/r097_batch2_exact/executed/final_commit_verification.json)。

- [独立只读审查](../studies/r097_batch2_exact/executed/independent_review.json)、[Git可见性与收尾工具事件](../studies/r097_batch2_exact/executed/environment_incident.json)、[首次收尾失败](../studies/r097_batch2_exact/executed/closeout.log)、[显式Git目录收尾通过](../studies/r097_batch2_exact/executed/closeout_explicit_git.log)。
