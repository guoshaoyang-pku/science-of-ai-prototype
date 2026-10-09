# alpha=.75：最大误差目标下仍无法通过 .05 阈值

本轮完成3/3新拟合 cell，0新训练。注册有限网格与参数域内，alpha=.75 的 min maxabs 数值下界为 .09797454939689487，高于 .05。P1/P2均supported，计为1/1科学条件、3/3坐标核验，不把seed计成独立条件。

## 条件与预测

程序第94轮、方向第7轮。按序读取 GOAL、AGENTS、state、task、已有 findings/report、inbox。先执行inbox六节报告重排并单独提交020116c；仅改变旧失败节标题层级，正文和数字全部保留，已追加指定处理时间。r082科学收尾及独立审查通过。旧r00215/r01827训练和r0386/r0503/r0823拟合成功cell均不重跑覆盖。

合同为development：固定特征平方损失、n=d=2、float64、谱[.01,1]、全批量SGD eta=.1/mom0/nodecay、L0=1/T10000、alpha=.75三条已保存曲线。候选H(0)=1，H(t>0)=1/[1+(t/t50_fit)^beta_fit]，平台1/0；t50_fit∈[.01,100000]步、beta_fit∈[.1,10]。原150点整数化log网格含0、终点4997，各点等权，误差除以真实动态范围1；RMSE≤.03且maxabs≤.05为足够。原LS参数/误差只读保存summary，不重新LS拟合。

P1预测三seed的min maxabs数值下界均>.05且最大误差选解仍不足。P2预测maxabs∈[.07,.11]。依据仅为旧alpha=.75 LS数字与alpha=.25/.5已测数值界；冻结前没有新alpha=.75可行性、残差或参数计算，不称盲发现。预注册、源码和输入hash一次提交bd67bcf22d3bae14b35eb58a9a03d0501a13d5d1，早于全部拟合。唯一commit绑定及历史202文件hashmtime在拟合前核验。

## 保存结果

| 项目 | 三坐标seed范围 | 判定 |
|---|---:|---|
| 原LS RMSE / maxabs | .06974371359516697–.06974371359516725 / .11474925759225374–.11474925781917555 | 原不足保留 |
| 最大误差选解 RMSE / maxabs | .07083688813461789–.07083688813461816 / .09797454943632976–.09797454943633009 | 0/3同时过阈值 |
| 同seed新−旧 RMSE | +.0010931745394508335–+.0010931745394509168 | 描述性比较 |
| 同seed新−旧 maxabs | −.016774708382845796–−.01677470815592365 | 描述性比较 |
| min maxabs 数值下界 / 可行上界 | .09797454939689487 / .09797454945510253 | 三seed同界，宽5.820766091346741e−11 |

最大误差参数见证t50_fit=119.96164200684001–119.96164200684076步、beta_fit=.7412374677525078–.7412374677525105。真实连续延拓半衰期202.63117087422313步只读原summary，不把拟合中点等同真实半衰期。原LS最大残差在10步，新见证在1723步。

## 核验与边界

沿用epsilon可行性二分；将logit(H)=c−beta·log(t)的误差约束转为线性上下界。两两比较求beta可行区间，不调用外部solver。每cell保存二分trace、参数见证、原LS预测、误差界与执行metadata。实数变换是已知解析工具，浮点证书不是严格区间算术。

先保存1cell，独立expit重建与半平面裁剪通过后续跑2cell。3/3下界多边形为空、上界各3顶点。恢复全局扫描拒绝混合、孤立、缺失和额外结果文件；实际恢复0新cell、3复用。202旧文件和6新结果文件hash/mtime不变。两次新增拟合调用累计.06936995801515877秒；含恢复的三调用累计.09296287503093481秒，低于1200秒。

本结论支持注册有限网格/参数域内无maxabs≤.05的固定平台单logistic解，不限于原LS选解。参数见证RMSE>.03不证明整个族RMSE>.03。没有连续临界alpha/谱比，不外推自由平台、learned features、其他网格/谱/优化器、前段预测或sealed OOD。原alpha0/.01新通过、配对P2失败、原r018P2失败及.25/.5/.75的P3支持保留；.25/.5已有数值界保留。

静态准备中出现字符串转义解析失败、路径relative/absolute不匹配及旧合同缺grid键的核验读取错误。均发生在新拟合前，未生成训练或拟合结果；改正后静态审查通过。错误记录保留于static_validation，不改已冻结源码。预注册按要求git add -A，包含supervisor预先产生的中心日志/状态和当前轮prompt/output记录；它们不参与科学合同，中心计数未被本轮脚本修改。

下一小问题优先0新训练，仅复用alpha=.9三曲线，其余条件不变，先注册新数值预测与pinned源码/输入hash并单一commit后再拟合。

## 证据

[summary](../studies/r094_alpha075_maxabs/summary.json)、[预注册](../studies/r094_alpha075_maxabs/preregistration.json)、[提交核验](../studies/r094_alpha075_maxabs/executed/preregistration_commit_verification.json)、[旧科学收尾审计](../studies/r094_alpha075_maxabs/executed/previous_closeout_audit.json)、[首cell核验](../studies/r094_alpha075_maxabs/executed/first_cell_verification.json)、[独立证书](../studies/r094_alpha075_maxabs/executed/certificate_verification.json)、[恢复后独立复算](../studies/r094_alpha075_maxabs/executed/independent_review.json)、[恢复核验](../studies/r094_alpha075_maxabs/executed/resume_verification.json)、[收尾回执](../studies/r094_alpha075_maxabs/executed/final_commit_verification.json)。
