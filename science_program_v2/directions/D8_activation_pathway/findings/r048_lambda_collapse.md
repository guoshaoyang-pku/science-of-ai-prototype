# r048：固定 λ 的折叠测量及协议失败

程序第 48 轮、D8 第 4 轮只研究一个 development 问题：固定 λ=δ/a² 后，同 seed 的 Q=R/a⁶ 能否在 a=.0125/.025/.05 上相差不超过 2%，并接近固定 Taylor 极限。本轮失效：λ=0 的 18 个 cell 与已有成功条件相同，却重新调用激活测量，违反用户本轮“旧成功 cell 不重跑”的要求。旧文件没有覆盖。原结果保留，不新增科学 claim。

## 预注册与条件

先依次读取研究入口、已有 findings/report 与 inbox；inbox 没有新指令，末尾指定处理标记已追加。报告在测量前重排为六节固定骨架，原数字与结论保留。上一轮科学收尾 454 个提交文件字节核验通过，旧 150 文件 hash/mtime 与 18 个 δ=0 重叠 cell 逐位核验通过；本轮保护清单含 599 旧文件。

固定 data31001、对称 Gaussian d4/n128、width64、seed101–106 的旧 input/W/u，仅改变共同 bias 与尺度：λ=−2/−1/0/1/2，δ=λa²，b=b*+δ。even 使用训练列均值居中，反射时 bias 固定；无标签、test、OOD 或新训练。15 个 λ×scale 条件、90 cell，六 seed 只测初始化稳健性。

极限 F_s(λ)=mean((f‴(b*)λU₂/2+f⁗(b*)U₄/24)²)/(f′(b*)²mean(u²)) 来自已知 Taylor 展开。注册前从旧 u 矩算出 F=0.0011876556204974424–0.09076095542175361，明确披露非盲来源，未拟合系数或预览新偏移激活。完整预注册、源码、输入与 hash 在 9fe16e6 提交后才测量。

## 保存结果（协议失效，不作为新验证）

| 原预测 | 原判据 | 原 summary 结果 | 有效性 |
|---|---|---|---|
| P1 | 全部30同seed/λ三尺度配对 max(Q)/min(Q)−1≤.02 | supported；范围 .000280769139–.009006901047 | 本轮协议失效，保留数值判定 |
| P2 | 全部90cell abs(Q/F−1)≤.02 | supported；最大 .009519027722 | 本轮协议失效，保留数值判定 |

最大跨尺度差与极限误差均在 seed103/λ=2；该 seed 的 Q=.009248793839/.009232217070/.009166234472，F=.009254326664。λ=−2/−1/0/1/2 的六seed配对差范围依次为 .007779793–.008082016、.003566358–.004010718、.000280769–.000904474、.004569616–.005590772、.007218185–.009006901。区间只按每 λ 的六 seed 算，跨格点不套置信区间。

## 执行与偏差

90/90 cell 保存，实际测量累计 .465821954 秒，0 新训练。先执行一 cell 并核验，再恢复 89 cell；首成功结果未覆盖。18 个 λ=0 新数组与旧数组逐位相同，但这种一致性不能消除重复测量偏差；其余 72 cell 条件此前未测。

独立 NumPy 激活重建最大差 3.552714e−15，独立导数与标量矩 F 最大相对差 2.442491e−14，保存 odd/even 的 einsum 复算 R/Q 最大相对差 4.729550e−14。恢复检查 new0/reused90，180 个新结果文件及 599 个旧文件 hash/mtime 不变。全部结果晚于预注册提交。

前置旧证据审计曾错把 d00 当 δ=0；原源码与 AssertionError 保留，随后从旧 bias_offsets 定位 d04 后通过。这个审计错误发生在新测量前，未改旧数据或科学计算。原 pinned 测量与分析源码、预测阈值和 summary 不改；另存 round_invalidation 明确本轮失败。

## 交接与边界

不再执行本失效 study，不重跑或覆盖旧 72/216 cell 及新 90 cell。下一有效轮只能新建 study 选一个未测 development 条件；任意重叠条件直接复制旧成功数组并记录原合同/hash，不能重新激活测量。候选为固定未测 λ=−3/+3 与 a=.0125/.025/.05，先写新数值预测、实际只读复用方案和 pinned 源码/输入 hash 并 commit，匹配后才测量。不能把已知解析预测或本轮失效结果当盲新发现。

结论范围仍以既有有效研究为准：普通 bias 的 a²、精确根的 a⁶ 与固定偏移网格的混合规律保留。不得外推连续临界、目标相关核、训练性能、随机 bias、非对称输入、多层、learned hidden、CE、小批量或 OOD。

## 证据

- [原预注册](../studies/r048_lambda_collapse/preregistration.json)、[原 summary](../studies/r048_lambda_collapse/summary.json)、[解析 forecast](../studies/r048_lambda_collapse/executed/analytic_forecasts.json)。
- [协议失败](../studies/r048_lambda_collapse/executed/round_invalidation.json)、[旧收尾审计](../studies/r048_lambda_collapse/executed/prior_closeout_audit.json)、[首 cell 核验](../studies/r048_lambda_collapse/executed/first_cell_audit.json)、[独立与恢复核验](../studies/r048_lambda_collapse/executed/independent_verification.json)。
- [测量源码](../studies/r048_lambda_collapse/executed/run.py)、[分析源码](../studies/r048_lambda_collapse/analysis.py)、[审计错误记录](../studies/r048_lambda_collapse/executed/prepare_failure.json)。
