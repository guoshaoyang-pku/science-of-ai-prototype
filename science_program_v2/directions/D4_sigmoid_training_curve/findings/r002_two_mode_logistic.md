# r002：预注册提交受阻，双模态实验未启动

本轮是程序第 2 轮、D4 第 1 轮。已依次读取 GOAL.md、AGENTS.md、central/state.json 和 task.md；此前该方向没有 findings、report 或 inbox。只选了一个问题：固定慢模态、初始损失权重与学习率，只增加快模态曲率时，对数时间轴的单 logistic 何时超过预先规定的拟合误差？

## 条件与预测

预注册为 n=d=2、float64、全批量随机梯度下降（SGD，实际使用全部样本）、学习率 η=.1、momentum=0。慢特征值 λslow=.01，快慢特征值比为 1、3、10、30、100；每模态初始损失 .5，总损失 1。3 个固定旋转 seed 只检查坐标稳健性，共计划 5 个条件、15 个 cell，每 cell 10000 步。为保持损失权重，目标参数随谱调整；参数范数和初始梯度没有固定。

候选为 H(t)=1/[1+(t/t50)^β]，H(0)=1，最终平台固定 0。它是对数时间轴上的 logistic；本轮不拟合原始线性时间 sigmoid。所有条件使用同一整数化对数网格，各点等权。以真实初始损失范围 1 归一化，均方根误差（RMSE）≤.03 且最大绝对误差≤.05 才称该窗口下拟合足够。P2 预测特征值比 1、3 通过；P3 预测 30、100 失败；10 是预先列出的描述性中间条件。

精确双指数是执行核验基线。它还给出连续半衰期 t50 和半高处局部斜率 βlocal，用来与拟合参数比较。谱递推、原始时间轴凸性和局部斜率匹配属于已知解析工具，不登记为新发现。

## 实际结果与证据

两次预注册提交尝试均在 `git add -A` 阶段返回 exit 128：无法创建 `.git/index.lock`，报 `Operation not permitted`。第二次命令包含 commit，但 shell 的 `&&` 阻止了 commit 执行。未取得预注册 commit，因此没有运行执行器，没有训练或拟合，保存 cell 数为 0，P1–P3 全部未评估。本轮没有 measured/refuted 科学 claim。

已保存[预注册草案](../studies/r002_two_mode_logistic/preregistration.json)、[自包含执行器](../studies/r002_two_mode_logistic/executed/run.py)和[分析脚本](../studies/r002_two_mode_logistic/analysis.py)。JSON、Python AST 和源码 SHA256 静态核验通过；另有只读理论与源码审查。只运行了分析脚本的“0 cell 且提交失败”分支来生成[summary.json](../studies/r002_two_mode_logistic/summary.json)，没有进入训练或拟合分支。这些核验不是实验测量。

原始失败记录保存在 `executed/preregistration_commit_attempt.json` 与 `executed/preregistration_commit_attempt_002.json`；原始草案 hash 保留。失败后补齐源码、hash 和恢复边界，全部仍为未提交草案；没有改动已冻结的实验。收尾提交的独立记录见 `executed/final_commit_attempt.json`。

## 恢复与边界

下一轮先恢复本仓库 git 元数据写权限，审查并提交当前预注册与两个脚本，再用指定 Python 先运行 `executed/run.py --max-new-cells 1`，随后续跑和分析。执行器在训练前核对本仓库 HEAD 中的合同和源码内容，不接受未提交版本。成功 cell 不覆盖；恢复核对合同、请求、源码、数组及 receipt 固定的 metadata hash。receipt 在每次调用末尾写入；异常中断留下缺少 receipt 的 cell 时会停止，需要另行审查，不能直接覆盖。

当前没有实测拟合质量、参数误差或计算耗时，也没有封存 OOD。计划仅适用于等初始损失双模态固定特征二次型与所列谱、优化器、网格；不能据此判断 learned features、动量、mini-batch、固定目标范数或前段外推。
