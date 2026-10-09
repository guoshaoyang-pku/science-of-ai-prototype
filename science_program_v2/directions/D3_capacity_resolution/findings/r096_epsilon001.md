# r096：0.1% 阈值下，匹配前三阶矩与范数的预算仍分离

程序第96轮、D3第7轮只回答一个问题：同六模态匹配设计，ε=.01→.001 后 slow/fast 预算比是否仍 ≥1.3。完成6/6保存曲线评价、3个同seed配对，0新训练。fast=160步、slow=322步，差162步、比2.0125；P1/P2/P3均supported。两个目标仍匹配前三阶谱矩、插值参数范数、初始损失与首步下降。

## 预测与条件

只改变损失比例阈值，其他条件全部复用：n=d=9、满秩固定线性特征λᵢ=i⁻²、float64零初始化、全批量SGD η=.5/mom0/nodecay、L₀=.5、0..1000整数步。fast用1/3/5、slow用2/4/9，权重不变。共同m₁=2173/32653、m₂=3607/620407、m₃=487/620407、m₋₁=插值参数范数平方=11832487/620407，首步下降161541/4963256。

评价前注册fast[150,190]、slow[280,380]与比[1.5,2.6]。区间来自已见1%端点103/138与谱衰减尺度的粗略外推；注册前未读取NPZ或计算.001端点，不作阈值扫描。整条来源曲线已保存，因此属于非盲development预测，不称封存OOD或盲发现。

P1检查已知谱递推误差≤1e−10、整数步一致、同seed特征字节相同与来源hash/mtime不变。P2要求全部3配对比≥1.3。P3要求全部端点及比进入注册范围。三个判据均通过，无删失。seed仅参数坐标旋转，不是独立数据重复，不报置信区间。

## 执行与证据

inbox要求六节骨架；先把失败与反例完整并入方法与条件，原全文除标题层级外逐字相同，单独提交a85dbe9，并追加指定处理时间。此前字符串格式失败调用没有写入或科学计算。

先审查[r084收尾回执](../studies/r084_three_moments_resume/executed/final_commit_verification.json)和[独立审查](../studies/r084_three_moments_resume/executed/independent_review.json)。历史科学收尾9b0cfad的33项产物逐字节通过；原科学冻结476dbf6与恢复授权b69d687均为祖先。原第52轮中断时只保存1旧成功cell，第84轮补5新cell，总6/6；六个原cell只有唯一原科学pin，不能把本轮0训练写成新的6训练cell。

本轮[合同](../studies/r096_epsilon001/preregistration.json)、[执行源码](../studies/r096_epsilon001/executed/run.py)、[分析源码](../studies/r096_epsilon001/analysis.py)、[输入清单](../studies/r096_epsilon001/executed/input_manifest.json)与[旧证据审计](../studies/r096_epsilon001/executed/pre_execution_audit.json)在唯一f3194fe提交中冻结。输入清单固定155项全部旧study/findings的SHA256与mtime，包括原NPZ/metadata/receipt/合同。评价前逐字节核对提交与源码、原科学pin、request与所有结果hash。

非TTY执行遇到.git不可见，在第一步Git祖先校验停止；四个run调用和一个analysis调用未加载NPZ，0新评价。[失败记录](../studies/r096_epsilon001/executed/execution_failures.json)保留此限制。TTY/非登录shell/显式workdir原样运行冻结脚本后通过，未改合同或源码。环境根因未识别。

[summary](../studies/r096_epsilon001/summary.json)从六个逐cell评价与receipt复算；preregistration_sha256指本轮新阈值合同，source_preregistration_sha256指原r052科学合同。[时序核验](../studies/r096_epsilon001/executed/timing_verification.json)确认预注册早于全部评价至少129.336940秒，六cell计算合计0.005614999秒。谱曲线最大误差4.440892098500626e−16。fast相邻保存比例R159=.001020886418044211、R160=.0009804589925383558；slow为R321=.001012181660955561、R322=.0009997240962184122。

[恢复核验](../studies/r096_epsilon001/executed/resume_audit.json)重新检查全部成功结果后读取，新增0评价，六结果加receipt共7文件hash/mtime不变，155旧文件也不变。[独立核验](../studies/r096_epsilon001/executed/independent_verification.json)另行复算首次命中和标量谱曲线。未重跑或覆盖r004/r01218、r0206、r0406与r0526个训练cell。

## 结论边界与下一轮

保存结果支持指定0.1%阈值的受控反例：共同前三阶谱矩与插值参数范数仍不唯一确定预算。m₄及更高矩与慢能量共同改变，不独立归因。预算比1.339806→2.0125只描述同目标曲线的两个离散阈值；不外推连续阈值规律。不同研究共同矩不同，不把比缩小或扩大因果归于新增控制。

满秩两目标都能最终插值，预算不是永久表示容量。未测learned features、架构轴、泛化、空间分辨率、其他优化器或sealed OOD。旧matmul警告根因未定、原梯度措辞纠正和≥10草稿执行前改为≥8均保留。没有已测失败预测，环境失败不等于科学预测失败。

下一小问题优先0新训练：只换预指定ε=.0001，同其余合同，复用全部六条保存曲线，检验比是否仍≥2.0。先注册具体数值范围、NPZ/metadata/receipt/合同hash与pinned源码单一commit，匹配后才评价；不得先计算新端点后称盲预测。中心仅更新D3 last_round与next_question，round与rounds_done交给supervisor。
