# r104：同一保存曲线的两个中点起点检查

## 结论

P1 支持 2/2。固定初始斜率 k0=1，只把 reference A 的中点初值 h0 改为 log(2) 与 log(101)，两个拟合均成功。tau_A 分别为 .019403481836800077 与 .01940348161806243；相对原 .01940348189419637 的最大差为 1.4231153982422837e-8，即 0.0000014231153982422837%，低于事前 1%。新增 2 个拟合、0 个训练，没有重拟合任何旧成功起点。

两个指定中点起点未造成超过 1% 的读数偏差。这不证明所有初值稳定、参数可识别或全局最优，也未建立从架构与初始化预测相变时间或有效深度的公式。旧 A/B 偏移与跨 seed 失败仍保留。

## 小问题、预测与冻结

只问同一保存曲线的 reference A 是否对两个新 h 初值稳定。已知旧 h0=log(11)、k0=1 读数及四个 k 起点结果，属非盲 development 检查。两条点预测均为旧基准，两条判据区间均为 [.019209447075254404,.019597516713138335]；它们是事前 1% 操作范围，不是置信区间。两者均须 success 且相对旧基准差≤1%。

唯一预注册提交 ae9671fe9c1c153dfeb80eb240591b75a6d65240 冻结合同、拟合/分析/独立核验三源码和全部 90 个历史 study 文件的集合/hash/mtime，早于首新拟合 32.71190905570984 秒。冻结前未运行两个新起点，也未计算新参数或 tau。独立只读静态审查时 results 为空；三源码语法与 hash 匹配。已有报告符合固定骨架，不需重排；inbox 不存在，无处理标记。按要求 git add -A 收入已有 supervisor 日志、state 与本轮原始 prompt/output，未科学使用这些日志。

历史核查覆盖四个 study：原深度网格 12 个 A 拟合、学习率研究 9 个配对拟合文件、seed22 细步长 1 个 reference A 和 4 个 k 起点。同本 NPZ 的旧起点只有 k0={1,.25,.5,2,4}，h0 全为 log(11)。新 h0=log(2)/log(101) 没有请求重叠。

## 条件与结果

只读复用无 skip L4/hidden32、n128d4 固定高斯输入、4×4 正交目标、sigma=.8、seed22 原初始化/数据、float64 CPU 全批量 GD/mom0/nodecay 优化 J=4×MSE 的保存曲线。旧 eta=.0003125、T=192000、末端 tau=60；input/target seed 为 20261008/20261007。未重建数据、未训练。

reference 时钟 s=.0003125×step/.02，tau_A=.02×expm1(h)。全部 192001 点、均匀 step 权重、自由上下平台、原模型/bounds/clip 与三项 1e-12 精度不变。初值为 [loss0,final loss,1,h0]，其中 loss0=1.2699456693495557、final loss=3.719756368669228e-30；仅 h0 改变。log(2)/log(101) 对应初始 s_c=1/100，原 log(11) 对应 10。

| h0 | tau_A | 相对原基准差（无量纲） | nfev | P1 |
|---|---:|---:|---:|---|
| log(2) | .019403481836800077 | 2.9580408760040104e-9 | 12 | 支持 |
| log(101) | .01940348161806243 | 1.4231153982422837e-8 | 17 | 支持 |

两拟合合计 .3458700180053711 秒。归一化 RMSE 分别为 .0009849355772315346/.000984935577231538，R² 均 .9968988754894407，2/2 通过旧形状判据。原成功 h0=log(11) 和四个成功 k 起点只读复用。这是同一曲线的两个拟合 cell，不增加独立训练 cell 或 seed 数。

## 核验、失败与边界

开头只读核验上一轮科学收尾 456d453 的 19/19 Git blob，四份回执均 passed；74/74 旧文件及 5/5 成功产物的 hash/mtime 保持。旧 state 提交 blob 匹配，当前 state 已由 supervisor 推进到 104；没有把这项预期差异判为旧证据损坏。独立审查重算四个旧拟合的 RMSE 差≤2.168404344971009e-19，R²/tau 差为 0，没有重拟合。

本轮 90/90 旧文件集合/hash/mtime 保持；合同与三源码等于唯一冻结字节。独立 expit 与 math.fsum 复算两保存拟合，RMSE 最大差 4.336808689942018e-19，R²/tau 差均 0，P1 判定一致。恢复执行与分析/核验各调用一次，新增 0 训练/0 拟合、reused 2；两结果、summary 与独立回执共 4 个文件 hash/mtime 未变。拟合 stderr 为空。

根编排工具有三次 JS 引号解析错误，均无 shell 执行；独立审查编排也保留一次解析错误。收尾链接扫描在根编排与独立审查分别触发一次 TypeError/AttributeError；修正未冻结扫描编排后通过，均未改冻结源码或结果。旧恢复工具两次解析失败仍保留。旧 matmul 除零、overflow、invalid 日志原字节保留，根因未定。旧 tc≥1 文字与源码 tc≥0 偏差、旧 reference_protocol.purpose“不代替主P1”与该轮主 reference A 判据冲突均保留披露。

旧 P2 的 10% 步长收敛仅通过 2/3、已被推翻；最细三 seed CV=.276924>.25。B 仍 tau=.12、第 6 公共网格点，A/B=.1616956824516364；不称真实峰收敛或协议一致。只检查两个指定 h 初值，不外推全部 p0/参数可识别/全局最优、seed 总体、其他谱/sigma/宽度/目标/optimizer、泛化、无限梯度流或 OOD，不进入 skip/deff。

## 下一小问题

回到跨初始化的步长稳定性。只新增未执行 seed11、eta=.0003125、T=192000，固定其原数据/目标/初始矩阵、reference 时钟、全点和原单起点拟合。检验 tau_A 相对该 seed 的 eta=.00125 旧读数 .023834804503907975 是否变化≤5%。先冻结新点预测与数值范围、输入 hash 和三源码的唯一提交，再训练；不提前计算新读数，不重复任何成功 cell/fit。不称连续步长极限或 seed 总体。

## 证据

- [预注册](../studies/r104_A_midpoint_start/preregistration.json)、[拟合源码](../studies/r104_A_midpoint_start/executed/run_experiment.py)、[分析源码](../studies/r104_A_midpoint_start/analysis.py)。
- [两个结果](../studies/r104_A_midpoint_start/results/)、[summary](../studies/r104_A_midpoint_start/summary.json)。
- [独立源码](../studies/r104_A_midpoint_start/executed/verify.py)、[独立回执](../studies/r104_A_midpoint_start/executed/independent_verification.json)、[恢复回执](../studies/r104_A_midpoint_start/executed/recovery_verification.json)、[执行审计](../studies/r104_A_midpoint_start/executed/execution_audit.json)。
- [拟合 stdout](../studies/r104_A_midpoint_start/executed/fitting_stdout.log)、[拟合 stderr](../studies/r104_A_midpoint_start/executed/fitting_stderr.log)、[来源 NPZ](../studies/r102_seed22_refinement/results/L4_s22_eta0003125.npz)、[旧基准](../studies/r102_seed22_refinement/fits/L4_s22_eta0003125.json)。
