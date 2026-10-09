# 定向 Science prototype

先读 README.md、state/current.json 与自己的 directions/<方向>/task.md。旧主线与 publisher 保持独立；只改自己的方向文件，协调者负责共享执行器、KB 与 Git 提交。

- Python：python3；已有 NumPy/PyTorch/Matplotlib/SciPy。
- 同一方向持续负责问题；先从竞争解释写可区分预测，再实验，保留否定结果。
- sources/ 是冻结 development 来源；复用已成功测量，不编辑。sources/manifest.json 核对导入 hash。
- 每次研究保存 preregistration、实际源码/recipe/data pin、全部 seed 结果及分析。OOD 先封存 conditions 与 predictions，不从结果回改。
- 同一函数/recipe 是研究单位；seeds 仅测条件内不确定性。无新增 solver 评测时不得报告 sol 涨分。
- 各 agent 只写自己的 directions/ 与 studies/<专属 study>/；不得 git add/commit 其他 agent 文件。协调者提交共享 KB 与 lineage。
- 限定真实并发 CPU 占用：每实验进程 torch threads=1；独立方向允许并行，不设主题硬 cap。
- 状态与原子提案写文件；成功结果不可覆写，resume 需验证配置、执行源码与已保存 hash。
- first_chain_complete 在 state/current.json 保存后，B/C 可开始自己的新训练；此前只读分析和准备实验。
- 当传入任务文本为空，从自己方向 task.md 读当前任务；不要退回旧开源任务。
