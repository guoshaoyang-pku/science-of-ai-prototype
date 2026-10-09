# r003：噪声与样本量能否预测风险最低点步数

本轮未得到实验结果。`git add -A` 无法创建 `.git/index.lock`，返回 `Operation not permitted`，因此预注册未提交。按先提交后训练的规则，未训练、未拟合，保存结果为 **0/60 cell**。P1–P3 均未评估；这是执行失败，不是科学预测被反驳。

## 已固定的问题与判据

本轮只检验一个问题：C04 固定 ReLU 特征 recipe 的条件期望风险最低点步数，能否按 `n/σ²` 折叠并用幂律跨样本量预测。固定四维 Gaussian 对称配对输入、目标 `x₀+0.5x₀x₁`、宽 128、scale 0.03、训练特征居中与总 RMS 归一化、零初始化含 bias 的线性 head、全批量 GD、学习率 0.3、无动量、float64。

扫描 `n=32/64/128` 与标签噪声方差 `σ²=0.25/0.5/1/2/4`，共 15 个条件；每个条件使用 411–414 四个数据/特征 seed，共计划 60 cell。同 n/seed 跨噪声复用输入、特征和 epsilon；跨 n 保持特征权重 seed，但按 C04 合同重新抽取输入并重新计算标签 RMS。跨 n 配对不等于只改变样本量的因果实验。

每个整数步 `0..16384` 记录期望风险，取首个全局最小点 `t*`。端点最低点单列为边界最优，在幂律判据中计为失败。44 个同 seed、同 `n/σ²` 的跨 n 配对必须全部拥有内部最优点，且步数之比在 `[0.5,2]` 内，才支持 P2。P3 用另外两个 n 拟合 `log(t*)=a+b log(n/σ²)`，留出第三个 n；60 cell 至少 48 个预测在两倍内才支持。P1 是训练单调性和风险复算的执行核验，不登记为新发现。

## 已保存与未执行的内容

- [预注册草稿](../studies/r003_noise_sample_scaling/preregistration.json) 保存数据合同、数值判据、参考源码 hash 与自包含源码 hash。它仍未提交，不具备事前 commit 见证。
- [提交失败记录](../studies/r003_noise_sample_scaling/executed/preregistration_commit_attempt.json) 保存首次尝试时的预注册 hash、命令、返回码和原始错误。之后只完成未运行的脚本与草稿审查；首次尝试的 hash 不被改写。
- [执行器](../studies/r003_noise_sample_scaling/executed/run.py) 在生成训练数据前核验本 repo HEAD 中的预注册、执行与分析源码；逐 cell 保存 JSON/NPZ 和 receipt，恢复核对源码、数据、结果 hash，拒绝覆盖成功结果。
- [分析入口](../studies/r003_noise_sample_scaling/analysis.py) 已在没有结果的情况下运行，只生成 [summary.json](../studies/r003_noise_sample_scaling/summary.json) 的未评估状态，没有运行数值训练或拟合。
- [静态检查](../studies/r003_noise_sample_scaling/executed/static_validation.json) 记录 JSON、AST、hash 与条件计数检查；执行器未做数值验证。
- [收尾提交记录](../studies/r003_noise_sample_scaling/executed/final_commit_attempt.json) 保存相同 `.git/index.lock` 写入拒绝；预注册与收尾 commit 均未完成。文件交接已保存，git 见证仍缺失。

## 边界与下轮交接

固定特征谱公式和 bias/variance 分解是已知理论，不能登记为新发现。`n/σ²` 折叠与幂律只是可反驳的经验假设；单模态基线在额外假设下可产生对数关系。最低点计算需要指定函数生成的 clean train/audit 标签，不是无 test 标签早停，也没有封存 OOD。

下一轮先恢复本 repo 的 `.git` 写入能力，审查并成功提交当前预注册和 pinned 源码。然后用 `python3` 执行 `executed/run.py --max-new-cells 1`，核对首 cell 与恢复 hash 后续跑，再运行 `analysis.py`。不得因草稿存在、supervisor 后补提交或此前失败记录而声称本轮预测已得到验证。
