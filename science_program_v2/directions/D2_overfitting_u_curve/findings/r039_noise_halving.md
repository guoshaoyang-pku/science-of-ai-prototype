# r039：噪声方差减半实验未执行

本轮准备检验一个问题：保持 population 标签、全部输入、W、特征、epsilon、η=.3/mom0 和 T=16384，仅将 n32/n64 的 σ² 从 .25/.5 同时减半至 .125/.25，seed414 的等 n/σ²=256 最低点比值是否仍超过两倍。新训练为 **0/8 cell**，P1/P2 未评估，轮次结果为 failed。

## 流程违规与停止依据

准备阶段把一个生成预注册草稿的辅助脚本写到了 `/private/tmp/make_r039.py`，越过本项目写入边界。随后删除了该文件，但删除不能撤销违规。GOAL.md 的硬边界规定“违反即无效轮次”，并要求“只写 science_program_v2/ 内部”。本轮因此停止新训练，没有将这次失败算作科学预测被反驳，也不登记新科学 KB claim。

一次静态检查误用系统旧 Python，因 Python 3 语法报错；随后使用指定 `python3` 的 AST 检查通过。没有因静态错误运行或保存任何新训练结果。执行器另加失效草稿锁，拒绝本轮配置训练。

## 历史证据与草稿

已读取规则、状态、方向材料并确认不存在 inbox。核验 r019 收尾 a680b3e 的 56 个 receipt 文件与历史 git 对象匹配；r003/r019 的 218 个既有文件与当前 HEAD 逐字节匹配，60 个旧 cell 和 8 个 population 对照 cell 未重跑、未覆盖。旧 P2 refuted 40/44、P3 supported 57/60 和 population seed414 的 62/201 步、比值 3.241935 均保留。报告只重排为固定六节骨架，未修改已测数字或结论。

注册前已从 r019 保存的 signal_bias 与 variance_unit 计算减半方差的 development 数值预测：四 seed 的 n32/n64 最低点分别为 331/278、313/383、230/212、87/324；seed414 计算比值为 324/87=3.724137931。草稿 P2 规定比值在 (2,4] 且两端均内部最优。这些数字来自已知曲线算术，**不是本轮独立实际 GD 结果，也不是盲预测**。保存草稿时明确披露来源，不据此声称新实测发现。

## 交接

下一有效轮先核验本轮收尾与违规记录，不执行本轮失效草稿。若继续同一小问题，审查 r019 population 数据映射、九个不变数组与 76 个旧输入 pin，使用新 study 和新预注册提交后才训练。可以保留 (2,4] 预测，但必须继续注明它已由旧 development 曲线算出。原 NumPy bool 错误及仅转换 scalar 的 r019 run_analysis.py 保留；本草稿只对新 analysis 的布尔输出显式转为 Python bool。

## 证据

- [失效原因与零训练记录](../studies/r039_noise_halving/executed/round_invalidation.json)、[历史收尾核验](../studies/r039_noise_halving/executed/prior_closeout_audit.json)。
- [未执行的合同草稿](../studies/r039_noise_halving/preregistration.json)、[执行器](../studies/r039_noise_halving/executed/run.py)、[分析草稿](../studies/r039_noise_halving/analysis.py)、[静态检查](../studies/r039_noise_halving/executed/static_validation.json)、[零新结果摘要](../studies/r039_noise_halving/summary.json)。
