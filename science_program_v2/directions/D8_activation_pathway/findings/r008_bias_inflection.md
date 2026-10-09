# r008：固定 bias 的 SiLU 偶通路尺度律

本轮为程序第 8 轮、D8 第 1 轮。预注册与自包含源码已保存，但预注册提交因无法写入 .git/index.lock 失败。未生成实验数据，未运行训练，保存 0/72 cell；P1–P3 均未评估。此状态是执行失败，不能写成预测被反驳。

## 问题与事前判据

仅检验一个问题：在固定加性 bias 下，SiLU 偶/奇能量比的小尺度指数是否在二阶导数为零处从 2 变为 6？沿用 C01 的四维对称 Gaussian、train128/test1024、宽64冻结特征、零初始线性 head、MSE、full-batch GD η=.8/mom0/nodecay、1024 步。数据 seed31001，特征 seeds101–106；bias 为 0、1、二阶导数的唯一正根；scales=.025/.05/.1/.2。每 cell 并行保存线性与二次目标的两条独立 head 轨迹，共 72 cell、24 个 bias×scale×target 条件、计划 144 轨迹；六 seeds 是条件内重复。

反射输入 x→−x 时保持 bias 不变。令 u=xW、f=SiLU，odd=[f(b+au)−f(b−au)]/2，even=[f(b+au)+f(b−au)]/2 减训练列均值。特征总均方根（RMS）归一化不改变 R=mean(even²)/mean(odd²)。常数项必须居中，否则 bias 的常数输出会被误算为偶通路能量。

| 预测 | 数值判据 | 本轮状态 |
|---|---|---|
| P1：b=0、1 时 R 保留 a² 领先阶 | a=.025→.05 的全部 12 个同 seed 配对增长在 [3.6,4.4] | 未评估 |
| P2：二阶导数零点处 R 的领先阶为 a⁶ | 同尺度全部 6 个配对增长在 [48,80]，事前期望 64 | 未评估 |
| P3：导数与保存 u 的矩可预测领先系数 | 最小尺度的全部 18 cell，abs(R/R_leading−1)≤.10 | 未评估 |

## 解析依据与执行证据

Taylor 展开给出 odd≈f′(b)au，普通 bias 的居中 even≈f″(b)a²[u²−mean(u²)]/2；当 f″(b)=0 且 f⁗(b)≠0，领先项改为 f⁗(b)a⁴[u⁴−mean(u⁴)]/24。因此能量比的领先幂次分别为 a²、a⁶。这是已知解析展开；有限尺度是否满足表中判据仍需测量。

无数据解析核验得到正根 b*=2.3993572805154675，f′(b*)=1.0998393201288668，f″(b*)≈−3.39e−17，f⁗(b*)=.15259173097310152。该核验在提交失败后进行，只用函数公式与求根条件；没有生成样本、测 kernel 或训练，不作为新科学 claim。证据在 studies/r008_bias_inflection/executed/analytic_design.json。

预注册 git add -A 返回 128：Unable to create science_program_v2/.git/index.lock: Operation not permitted。失败命令、时间与预注册 SHA256 保存在 executed/preregistration_commit_attempt.json。收尾提交的最终状态见 executed/final_commit_attempt.json。未绕过只读 .git；本轮没有获得事前 commit，故实验没有启动。analysis.py 已执行并从结果目录复算出 0 cell、0 秒训练；summary.json 将全部预测标为 not_evaluated。central/kb.json 新增 0 条 claim。

## 边界与恢复

本设计仅含对称输入、同单元固定加性 bias、冻结单层 SiLU、MSE 与全批量梯度下降。未测随机 bias、根附近 bias 容差、非对称输入、多层、learned hidden、CE、小批量或干预收益。总偶能量也不等于二次目标核能量；脚本分别保存两者，不能用指数直接推断终点拟合排序。

恢复时先由有权限的执行环境恢复本 repo .git 写权限，审查并提交 preregistration.json、executed/run.py 与 analysis.py；未取得匹配 commit 前禁止训练。再用指定 Python 执行 executed/run.py --limit 1，运行 analysis.py 核验首 cell。记录首 cell JSON/NPZ 的 hash 与 mtime 后运行剩余 cell，再分析并核对首 cell 未被覆盖。源码在生成数据之前检查 HEAD 中的预注册及 pinned 源码字节；恢复逐 cell 校验合同与 NPZ hash。有限结果必须保留，三项预测按原判据报告。
