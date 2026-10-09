# r024：近根 bias 的偶通路混合与不对称转折

程序第 24 轮、D8 第 3 轮，只检验一个 development 小问题：SiLU 二阶导数正零点附近，二次与四次偶项及其交叉项能否预测固定 bias 网格上的有限尺度能量比转折。已有报告先按公式区结构重排，保留旧测量结论；inbox 无新指令，已追加指定处理时间。

## 条件与预注册

复用旧 data31001 对称 Gaussian train128、d4、width64 冻结投影和 seed101–106，保持所有 bias/scale 的同 seed 输入和权重一致。b*=2.3993572805154675，δ=0、±1e−5、±1e−4、±1e−3、±1e−2，a=.0125/.025/.05/.1。反射输入时 bias 固定，even 按训练逐列均值居中。216 测量 cell、36 bias×scale 条件，0 新训练；不读取标签和 test。六 seed 仅初始化重复。

预注册预测由旧保存 u 矩与函数导数给出，未预览新 bias 激活。R₂₄=mean((c₂a²U₂+c₄a⁴U₄)²)/mean((f′au)²)，c₂=f″(b)/2、c₄=f⁗(b)/24。初始合同与 pinned 源码先提交 406cfe1；独立设计审查后补齐精确 cell 标签与 Cartesian grid 完整性审计，在任何新测量之前提交 22ebe0c。执行 manifest 记录此完整匹配提交。

| 预测 | 原判据 | 结果 |
|---|---|---|
| P1 | a=.025/.05 的全部 108 cell 相对误差 ≤.02 | supported；最大 .001192850 |
| P2 | δ=0/±1e−5 全部18配对增长 [60,68]；−.01 六配对 [5,8]；+.01 六配对 [1.5,3.5] | supported；三范围分别 62.813129–65.187934、5.476460–6.992584、1.959788–2.770145 |
| P3 | +.001 减 −.001 的同 seed 增长差均 ≥20，正侧至少2/6增长 >100 | supported；差 28.136243–125.024683，正侧 5/6 >100 |

## 现象与解释

混合公式在主尺度最大误差为 0.119285%，全四尺度最大误差为 0.591217%。去掉交叉项后的主尺度最大误差为 86.129939%，但这是描述性模型消融，没有新增成功判据。+.001 的增长 48.047861–155.318944，高于 −.001 的 19.911618–31.026926；同 seed 差均值 83.489843，95% 初始化 t 区间 [49.549244,117.430442]。seed104 正侧仅 48.047861，不能写成全部超过 100。

两项的分项等能量尺度由实际导数与旧矩解析确定，近根 a_eq²≈|δ|/κ，六 seed κ=1.088258–2.154200。|δ|=.001 时两侧 a_eq=.021544978–.030314131。正偏移的交叉项为负，可抵消偶项；有效增长指数可以高于 6 或低于 2，并非两纯幂之间的单调混合。δ=0/±1e−5 满足注册窗口，而 ±1e−4 全部越出，只能给固定网格边界，未测中间值。

## 执行与核验

216/216 cell 完成，实际测量耗时 1.059147 秒，0 head 训练。先运行 limit1，逐 hash/合同/finite/独立 NumPy 激活复算后恢复 215 cell；首 cell 保留。再运行恢复检查得到 new0/reused216，432 个结果文件 hash 与 mtime 全部不变；150 个旧证据文件 hash 与 mtime 不变，未覆盖旧 72 个成功 cell。18 个 δ=0、a=.025/.05/.1 的重叠 cell 与旧保存 odd/even 逐位相同。

分析首次首 cell 运行在 summary 成功写出后因打印空主尺度统计退出 TypeError，原错误、首 cell 核验摘要及 hash/mtime 全保留。完整分析随后退出0；原 pinned 源码不改。额外 run_analysis.py 仅条件处理这一打印错误，并删除跨108/216格点误差统计错误套用的 seed CI；R、导数、阈值、每seed配对结果不改。每 δ 的六seed配对区间与正负差区间保留。

独立 sigmoid 导数和标量多项式复算 R₂₄，与数组构造式最大相对差 3.663514e−12；独立能量比与 summary 差为0。独立 NumPy 激活重建最大绝对差 3.552714e−15（阈值2e−14），完整 grid 和全部合同核验通过。旧科学收尾为 3ad423e，原 state 的 final_commit=null 已用只读证据审计澄清，不改原旧结果。

## 边界与交接

结论仅适用固定 data31001、对称 Gaussian d4/n128、width64 冻结单层 SiLU、共同加性 bias、训练列居中、六seed及预定离散网格。所有是 development，不是新 OOD；公式 Taylor 本身不是新发现，新增证据是受控混合预测与两侧不对称。分项等能量不等于总能量转折或训练效果，不预测目标相关核、MSE、test、其他优化器、随机 bias、非对称输入、多层、learned hidden、CE或小批量。

下一轮可只选一个新 development 小问题：不新训练，保持输入与权重，仅以事先固定 λ=δ/a² 配对尺度，检验 R/a⁶ 是否近似折叠。需新数值预测与 pinned 源码先 commit；不得覆盖旧72与新216成功 cell。

## 证据

- [预注册](../studies/r024_near_root_crossover/preregistration.json)、[旧矩解析预测](../studies/r024_near_root_crossover/executed/analytic_forecasts.json)、[summary](../studies/r024_near_root_crossover/summary.json)。
- [首 cell 审计](../studies/r024_near_root_crossover/executed/first_cell_audit.json)、[独立复算](../studies/r024_near_root_crossover/executed/independent_verification.json)、[恢复核验](../studies/r024_near_root_crossover/executed/resume_verification.json)、[旧收尾审计](../studies/r024_near_root_crossover/executed/r016_closeout_audit.json)。
- [自包含测量源码](../studies/r024_near_root_crossover/executed/run.py)、[pinned分析](../studies/r024_near_root_crossover/analysis.py)、[输出与区间修正](../studies/r024_near_root_crossover/executed/run_analysis.py)。
