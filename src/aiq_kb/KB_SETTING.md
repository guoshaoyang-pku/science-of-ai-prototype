# KB science loop setting

本页只记录运行协议；每个 checkpoint 的 KB、报告、源码和哈希以 run 页面中的实际文件为准。

## Release 与评分

`kb_science_pool_v3` 固定 seed=`20261004`，共 980 行：`arch170` 80、`ranking_v2` 400、`ranking_v3` 300、`dataflip500` 200。按 group 和可追踪的数据集/候选集标识整组划分；实际 train/val/test 为 770/105/105。arch170 与 dataflip500 没有可用的新 holdout，当前宏观验证只覆盖 ranking_v2/v3。

选择题答对得 1 分；排序题按逆序对数 0/1/2/3 得 1/0.75/0.5/0.25，≥4 或无法解析得 0。claim 的 `s += score`、`f += 1-score` 只是引用反馈，不是真实概率。

## 已完成 prototype：soa_async_v1

`soa_async_v1` 使用 `gpt-6.1-sol` / Codex；每题保存 KB 与 no-KB 对照。解题 batch 为 30，计划 15 轮；答案揭示前使用冻结 KB。当前设置关闭最终 test 和 test-on-accept。

总结者与研究者是持久异步任务，不设 14 次工具调用上限。40 分钟是软窗口：超过窗口后继续原任务，以跨轮 label 标注，保留发起轮次、会话和 checkpoint；进程停下后可恢复。研究报告可覆盖多条 claim，保留测量、竞争解释、可区分预测和未决问题。研究是否 ready 不阻止下一轮读取。主循环结束后，supervisor 继续将晚到成果合入 live KB，保持已评测快照不变。

## 评测与发布

旧 `soa_async_v1` 每 5 轮做一次 ranking v2/v3 宏观检查，每组 3 次重复；原代码保留连续两个检查点明显下降时整库回滚的路径，15 轮未触发。这个检查是研发读数，不是最终保留集。

## 新默认协议：2026-10-09

研究 KB 持续积累，依据测量、反例与重复内容做 add / revise / merge / delete。默认每 5 轮宏观审查、每题 3 次重复；不要求每项修改立即提高全局分数。评测候选始终是保存的研究 checkpoint，支持证据、完整报告与异步任务不因分数回退而删除。

默认下一轮解题使用研究 KB 的冻结副本。若连续两个宏观检查点比稳定版本低超过 `2 × SE`，仅将后续 solver 输入切到稳定 checkpoint；这是执行保护阈值，不是发现真实与否的检验。研究 KB 与研究任务继续更新，后续宏观读数恢复到稳定版本以上时，solver 再使用成长 KB。噪声范围内下降不回滚研究。

每轮保存 `solver_snapshots/solve_eXXXX.json` 及 `solver_selection`；恢复优先复用原冻结输入和成功解题。引用反馈只记给解题时实际使用且内容仍相同的 claim；旧文本的结果不自动归因给已修订、合并的新规则。宏观审查重复处理同一 checkpoint 不多算退化次数。

`--no-eval` 关闭所有评测。最终 test 默认关闭，须显式启用；`--gate` 保留历史整轮回滚模式，仅用于显式对照，并关闭默认宏观协议。既有 run 与公开结果不重写。新实验须另选未暴露 release，核对 group / seed / variant 隔离；本次实现未启动收费解题。

publisher 只发布真实保存的 KB、报告、研究 repo、源码包和日志，沿用 run viewer；失败推送会在后续周期继续核验。页面不添加主页 block 或装饰性统计区。
