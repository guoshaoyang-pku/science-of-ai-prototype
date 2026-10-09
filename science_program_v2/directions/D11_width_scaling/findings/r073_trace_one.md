# trace 固定为 1 后的预算缩放

本轮只问：固定 n=8192、d=8..8192 二倍网格、旧中间/首尾目标和三个坐标 seed、全批量 half-MSE SGD η=.5/mom0/nodecay、L0=.5 与 ε=.01，仅把 λᵢ=i⁻¹ 除以 h_d 后，T/(d h_d) 是否在注册误差内保持 log(100)/2、log(100)。现有方向报告符合公式先行骨架，没有 inbox。

## 合同与执行

预注册、实际参数更新、分析与独立核验源码在 89a0f20 同提交冻结。训练前逐字核对该提交、源码 SHA256、302 个历史文件 hash/mtime 与空 results 后才运行。旧 293 个科学文件及 132 个 cell 的 metadata/NPZ/request hash、保存整数阈值通过；失效草稿永久锁定，未运行或改写旧训练/分析。

逐维数整数参考和误差算术在失效准备轮中已由已知谱递推算出，本轮用户交接也给出同表。全部为 development、非盲验证；它们不是新理论、独立实测或封存 OOD。原目录缺少的早期收尾 receipt 仍如实保留缺口。

直接复用每个旧 cell 的 modal_weights、target、permutation、signs 字节，仅将 eigenvalues 除以 h_d 并按 √(nλ) 构造特征。参数 θ 从零开始真实更新，不以谱曲线替代训练。22 个条件×3 个坐标 seed=66 cell、33 个两目标配对；seed 只测坐标稳健性，不是独立数据，不报置信区间。上限 ceil(5 d h_d)，首次阈值后多一步；66/66 完成，无删失。累计实验计算 34.852269 秒，单 cell .000756–5.844675 秒。

## 预测与结果

| 注册判据 | 保存结果 | 状态 |
|---|---|---|
| P1：已知递推执行审计；曲线/合同/保存更新≤1e−10、闭式 θ≤1e−7、整数参考一致、无删失 | 66/66 通过；曲线 maxabs 5.695999e−13，θ 最大误差 1.078604e−10，1101 检查点 loss 与 264 相邻更新误差均 0 | supported；不登记新理论 |
| P2：中间/首尾 T/(d h_d) 对 log100/2、log100 全网格相对误差≤3%/5%；d≥128 两者≤.3% | 全网格 2.126802%/4.124214%；d≥128 为 .029565%/.207098% | supported |

新实际预算依 d 为 49/96、124/245、298/593、698/1393、1601/3196、3609/7213、8036/16065、17705/35403、38678/77348、83893/167777、180859/361710。三个坐标 seed 整数预算一致。误差分母为相应渐近系数，不是实测预算；P2 的两个小维数最大误差都出现在 d=8。

描述性对照保持旧 d 幂律系数冻结，对新预算分别低估 63.027438%–89.515453% 与 62.619843%–89.475831%。新预算对 d 的等权 log-log 拟合为 4.888044 d^1.177950 与 9.616830 d^1.180201，最大相对误差 15.540384%/16.570413%，超出旧幂律 [.98,1.04]/6% 的条件；此项不是新增注册预测失败。中间/首尾同 Rayleigh 预算比 1.959184–1.999956，Rayleigh 对首尾仍低估 48.958333%–49.998894%，对中间准确；旧反例全部保留。

## 解释与边界

固定 trace 改变谱的整体时间尺度。λ_(d/2)=2/(d h_d)、λ_d=1/(d h_d)，已知谱递推解释两预算约随 d h_d 增长，h_d 近似 log d。原近一次 d 幂律依赖指定谱尺度，不能把其指数当成维数单独决定的普遍性质。trace=1、同 Rayleigh 仍没有合并两目标预算。

n=8192 只是零填充记账，非零样本数仍 d。构造固定特征不存在 learned width 或核漂移；不外推独立数据、随机特征、lazy↔rich、泛化或 OOD。归一化必然改变初始梯度与参数目标范数，两个目标高阶矩和范数未匹配，不能独立归因某个中介。下一小问题优先零新训练：复用本轮保存曲线，只把评价阈值改为 .02，先注册新数值区间、源码与来源 hash，再检验 d h_d 系数边界。

## 过程与恢复

四次源码/预注册工具字符串语法错误均在训练前阻止执行，未产生训练或外写；修正后合同 AST/compile 通过再提交。源码审查未见阻断问题。按要求 git add -A 时一并保存 supervisor 原有状态/log 和工作区并行出现的两张图，图不作本轮证据，归属见过程记录。成功运行后再次调用冻结 runner 新增 0 cell；133 个 results 文件及 302 个历史文件 hash/mtime 不变，不覆盖旧或新成功 cell。

## 证据

- [预注册](../studies/r073_trace_one/preregistration.json)、[参数更新](../studies/r073_trace_one/executed/run.py)、[分析](../studies/r073_trace_one/analysis.py)、[独立核验源码](../studies/r073_trace_one/executed/verify.py)、[训练前时序](../studies/r073_trace_one/executed/experiment_start.json)。
- [汇总](../studies/r073_trace_one/summary.json)、[首 cell 核验](../studies/r073_trace_one/executed/first_cell_verification.json)、[独立核验](../studies/r073_trace_one/executed/independent_verification.json)、[恢复审计](../studies/r073_trace_one/executed/resume_audit.json)。
- [旧输入清单](../studies/r073_trace_one/executed/baseline_manifest.json)、[旧证据审计](../studies/r073_trace_one/executed/prior_evidence_audit.json)、[过程记录](../studies/r073_trace_one/executed/session_audit.json)、[中心更新审计](../studies/r073_trace_one/executed/central_update_audit.json)、[收尾核验](../studies/r073_trace_one/executed/final_commit_verification.json)。
