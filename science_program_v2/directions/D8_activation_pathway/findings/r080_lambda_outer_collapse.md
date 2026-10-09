# r080：未测 λ=±3 的有效折叠验证

程序第80轮、D8第5轮只检验一个 development 小问题：保持已保存输入与冻结权重，令 δ=λa²、λ=−3/+3，Q=R/a⁶ 是否在 a=.0125/.025/.05 跨尺度相差不超过2%，并接近固定 Taylor 极限 F。先依次读取研究入口、历史 findings/report 与 inbox；报告已符合公式开头六节骨架，inbox无新指令并追加指定处理标记。

## 条件、预测与时序

固定 data31001、对称Gaussian d4/n128、width64冻结单层SiLU、共同加性bias、训练逐列居中和seed101–106。6个λ×scale条件、36新测量cell、12同seed/λ三尺度配对；六seed只测初始化稳健性。0新训练、不读标签/test，无OOD。

F_s(λ)=mean([f‴(b*)λU₂/2+f⁗(b*)U₄/24]²)/(f′(b*)²M₂)。注册前只从旧保存u矩与函数导数计算12个F，范围.015779531472–.140811355279；点预期Q=F、spread=0，有限尺度预测P1为全部12配对spread≤.02，P2为全部36cell abs(Q/F−1)≤.02。解析式与旧失效数值事先已知，明确披露非盲来源，没有拟合或预览新bias激活。

先核验旧round_invalidation与final_commit_verification：旧收尾018b78d的205个文件blob与receipt全部匹配；旧599保护文件及失效study180结果文件hash/mtime不变。核对三份历史Cartesian与378保存request（72+216+90），本轮36请求零重叠，806旧study文件纳入保护。原d00误用审计错误、matmul警告、打印错误与pooled CI纠正保留。匹配预注册、输入、审计与pinned源码先提交e3239cde77b1e39786d3f7296a5fb2e1010faa28，之后才调用新条件激活。

## 结果与解释

P1 supported 12/12：spread=max(Q)/min(Q)−1=.011019256654–.012094422622。P2 supported 36/36：最大abs(Q/F−1)=.012907814800，发生在seed102/λ=−3/a=.05（Q=.083156693593，F=.082097000712）。两项均低于事前.02。有效验证不包含失效study的90cell。

λ=−3的六seed spread均值.012004764，95%初始化t区间[.011927115,.012082413]；λ=+3均值.011306954，区间[.011064572,.011549336]。a=.0125→.05的Q比值在负侧为1.011878275–1.012094423，正侧为.988477488–.989100844，显示两侧有限尺度修正方向相反；此方向观察是描述性结果，未另注册判据。固定λ使近根二次、四次偶项同阶，旧矩主项可近似描述有限尺度Q，但不是精确等式。跨格点误差只报描述统计，不套pooled CI。

## 执行与边界

先跑limit1，独立数组/hash/commit与标量能量检查通过，再复用首cell完成其余35cell。36/36成功，累计测量.198246秒，0训练。独立NumPy激活重建最大差3.108624e−15；独立longdouble sigmoid多项式导数与标量矩F相对差8.015810e−14；fsum复算R/Q相对差4.440892e−16。全部结果mtime晚于预注册提交，全部源/输入/合同hash通过。

恢复检查new0/reused36，72结果文件hash/mtime和首cell不变；806旧study文件hash/mtime不变。没有执行r048，没有重跑或覆盖旧72/216/90cell。本轮有效科学claim新增2条，旧失效数值与0新增claim记录保留。

适用范围仅为固定data31001、对称Gaussian d4/n128、width64冻结SiLU、共用bias、训练居中、seed101–106与λ±3/三尺度离散网格。总偶能量不等于目标相关核或训练收益；不外推连续λ/尺度临界、其他数据/seed、随机bias、非对称输入、多层、learned hidden、CE、小批量或OOD。下一小问题可只改未测a=.1、λ=±3，仅读保存小尺度结果作对照，先注册新数值预测与复用合同，不提前算新激活。

## 证据

- [预注册](../studies/r080_lambda_outer_collapse/preregistration.json)、[旧矩解析预报](../studies/r080_lambda_outer_collapse/executed/analytic_forecasts.json)、[summary](../studies/r080_lambda_outer_collapse/summary.json)。
- [历史条件审计](../studies/r080_lambda_outer_collapse/executed/condition_audit.json)、[旧收尾审计](../studies/r080_lambda_outer_collapse/executed/prior_closeout_audit.json)、[首cell审计](../studies/r080_lambda_outer_collapse/executed/first_cell_audit.json)、[独立与恢复核验](../studies/r080_lambda_outer_collapse/executed/independent_verification.json)。
- [自包含测量](../studies/r080_lambda_outer_collapse/executed/run.py)、[分析](../studies/r080_lambda_outer_collapse/analysis.py)、[独立核验](../studies/r080_lambda_outer_collapse/executed/verify.py)；原r048的失效记录保持不变。
