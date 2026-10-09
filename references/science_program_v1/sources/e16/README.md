# e16：同函数标签均值干预

先读 report.md。预注册为 3 个函数×4 个标签条件×2 个优化器×10 个配对 seeds。结论限于保存的 recipe；没有新的 solver score 评测。

## 离线重算

需要 Python、NumPy、PyTorch；生成图另需 Matplotlib。

```sh
python analyze_results.py
python kernel_diagnostic.py
python plot_results.py
```

analyze_results.py 使用 evidence/ 中的内容寻址测量，核对 SHA256 后重算 analysis.json、seed_metrics.csv 和 process_metrics.csv。kernel_diagnostic.py 重建初始化并核对第一 minibatch/参数 hash，只计算导数。

## 重新训练单个受控 cell

此命令是显式重训，请仅在需要复现时执行；当前研究继续使用成功保存的 measurements。

```sh
python reproduce_cell.py --dataset mvar_02036a --condition 2 --optimizer SGD --seed 0
```

condition 0=原标签，1=均值−3，2=均值0，3=均值+3。程序只使用本 repo 的执行源码快照与 datasets，不依赖原机器的绝对路径。结果可能随 PyTorch/CPU 版本有浮点差异。

run_study.py 为原工作区的任务入口，依赖 canonical kb_lab、allowlist 与 pool manifest；用于来源核验和已保存结果续接。240 次原实验由此执行。evidence_manifest.json 保留原路径作为来源记录，离线分析优先使用 portable 内容寻址文件。
