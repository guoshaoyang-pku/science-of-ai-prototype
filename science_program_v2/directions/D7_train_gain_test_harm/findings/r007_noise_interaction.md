# r007：标签噪声是否扩大相对 test 损害，未进入训练

本轮完成 B02 来源复核，准备了干净／带噪 × 真实网络／初始切线的配对实验。预注册提交被 `.git` 写权限阻断，未训练，保存 0/24 cell、0/48 优化轨迹，P1 为 `not_evaluated`。没有新增科学 KB claim。

## B02 来源复核

v1 B02 的 `run.py:47–49` 直接计算确定性目标，再用干净 train 标签的均值和标准差处理 train/test；没有添加标签噪声。模型为含 bias 的两层 hidden SiLU 网络，d=5、train32、test256、width8/32，data-seed=7331、初始化 seed=11/29/47，全批量 halfMSE、float64、无 decay，训练 512 步。两种 SGD recipe 是 η=.05/mom0 和 η=.005/mom.9。

从已保存的 `ood_cells.csv` 按相同 seed 配对复算：8/8 函数×宽度×SGD recipe 单元的均值 train 增益超过 .05×初始 loss；6/8 单元的均值 test 差距为正。这里 train 增益是切线 loss 减真实 loss；test 差距是第 512 步真实 loss 减切线 loss。来源及源码 SHA256、逐 seed 数值、95% Student-t 区间保存在 `../studies/r007_noise_interaction/executed/b02_audit.json`；这属于旧证据复核，不是本轮新增条件验证。

| B02 单元 | SGDplain test 差距均值 [95% seed 区间] | SGDmomentum test 差距均值 [95% seed 区间] |
|---|---:|---:|
| mixed_sine，width8 | .11660 [−.01753, .25073] | .12324 [.00955, .23693] |
| mixed_sine，width32 | .05286 [−.10106, .20677] | .05295 [−.09966, .20556] |
| radial，width8 | .01645 [−.19623, .22913] | .02243 [−.17262, .21749] |
| radial，width32 | −.23163 [−.33550, −.12776] | −.22629 [−.32629, −.12630] |

标签噪声不是该 6/8 现象的必要原因。这个排除只涉及人为添加的标签噪声，不能排除有限样本效应，也不能区分目标相关核变化和隐式偏置。radial width8 的 SGDplain 有 1/3 seed test 差距为正，momentum 有 2/3；“6/8”不是所有 seed 同号，更不是六个单元的区间都排除零。终点相对切线劣势也不等于同模型继续训练使 test 上升。

## 预注册的小问题

固定 B02 的上述数据、函数、宽度、初始化，只用 SGDplain η=.05/mom0/nodecay。每个 seed 训练 sigma=0 和 sigma=1 两个 cell；每 cell 内真实网络与完整初始 Jacobian 切线共享初始输出与 optimizer，共 24 cell、48 轨迹、4 个函数×宽度单元。clean target 标准化之后加独立 noise-seed=733107 的 Gaussian 噪声，不再重新标准化；各 cell 共用同一噪声向量，test 始终干净。

P1 定义 `H_sigma=L_test(real,sigma)−L_test(tangent,sigma)`，`D=H_1−H_0`。预测至少 3/4 单元的 mean(D)≥.02，且各至少 2/3 初始化 seed 的 D>0。全部 cell 保存并核验之后才能判定；阈值不调整，缺失不视为预测失败。逐单元报告配对 seed 区间；另报两模型各自加噪影响、clean train 风险及 step128→512 的 test 增量。条件已在 B02 看过，全部标为 development。

## 执行证据与交接

2026-10-06T18:26:11.820225+08:00，预注册阶段 `git add -A` 返回 128：不能创建 `.git/index.lock`，`Operation not permitted`。当前沙箱把 `.git` 列为只读且禁止权限升级；没有绕过权限，没有运行训练。源码语法检查通过；指定 Python 运行 `analysis.py` 得到 `blocked_before_training`、0 cell、P1 `not_evaluated`。这是流程失败，不是噪声放大预测被反驳。

`executed/run.py` 的 commit gate 要求当前预注册与 pinned 源码在同一 commit 逐字匹配；成功 cell 恢复时核验合同、输入和 NPZ SHA256，不覆盖。`analysis.py` 从保存输出复算损失与配对指标，并核验初始参数/Jacobian 配对及切线递推。收尾提交记录见 `executed/final_commit_attempt.json`；两次原始错误和零训练状态纳入 `summary.json`。

下一轮先取得本 repo `.git` 的正常写权限，审查并提交现有预注册与源码。匹配 commit 出现后先运行 1 cell、执行 analysis 核验，再恢复剩余 23 cell。当前设计只估计固定 data/noise seed 的噪声敏感度差；不量化三种机制的比例，不推广其他 optimizer、样本量或噪声强度。

收尾提交在 2026-10-06T18:29:03.525100+08:00 再次因 `.git/index.lock` 写权限失败，`git add -A` 返回128，未产生任何本轮commit。所有产物和交接状态仅保存在工作树；`summary.json` 记录两次提交错误。
