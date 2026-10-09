# r005：Gram 条件数中介实验在训练前被 Git 权限阻断

本轮是程序第 5 轮、D5 第 1 轮。只选择一个可检验问题：在复用 v1 A03 的同宽度 `w64`、同函数、同 offset、同 seed 配对中，LN 是否使最后一层 hidden 表示的 Gram 谱条件数降低，并且该变化是否与 centered chord 收益相关。

## 预注册合同

预注册条件写入 `studies/r005_gram_condition/preregistration.json`。条件固定为四个 A02 函数、`train_n=test_n=256`、数据种子 48291、offset `-3/0/+3`、seed 100–104、GELU 三个 hidden block、SGD `lr=.001`、`momentum=0`、`weight_decay=1e-4`、256 步、batch 64。比较 `LN010_w64=[false,true,false]` 与 `noLN_w64=[false,false,false]`。每个 cell 计划保存 step 0/1/8/32/128/256 的 hidden Gram 特征值和 `log10(kappa)`，并保存测试残差以复算 centered chord。计划 120 个 cell；seed 只用于配对稳健性，不计作新函数。

预先预测为：四函数×三 offset 中至少 8/12 个对比中 LN 的最终 `log10(kappa)` 更低；8 个函数×seed 配对的 LN−noLN chord 与 LN−noLN `log10(kappa)` 的 Pearson 相关不大于 −0.5；至少 6/8 函数同时出现条件数下降和负 chord。相关性只作为中介线索，不等同于因果中介。

## 执行结果

在任何训练前执行 `git add -A` 和预注册 commit。两条命令都因无法创建 `.git/index.lock` 返回 exit 128（`Operation not permitted`）。因此没有取得预注册 commit，按 AGENTS.md 规则没有运行训练器；保存 cell 为 0，Gram 谱、chord 和三项预测均为 `not_evaluated`。

保存证据包括预注册、`executed/run.py`、`analysis.py`、`summary.json` 和 `executed/preregistration_commit_attempt.json`。源码已通过 Python 编译检查；分析脚本只生成空结果分支，没有调用训练。没有新增 measured/refuted KB claim。

## 边界与下一步

本轮没有证据支持或反驳 Gram 条件数中介。恢复 `.git` 写权限后，应先审查并提交当前合同和 pinned 源码，再用指定 Python 运行一个 cell、核对 hash 后续跑全部 120 cell。下一轮仍应把 Gram 条件数与 chord 的配对差异和时间顺序分开报告，并在条件数相关性成立后再设计人工谱重整干预。
