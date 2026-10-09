# e17：初始化输出平移与 frozen hidden

先读 report.md。360 次受控训练保存于 results/；实际执行源码保存于 executed/，source_manifest.json 记录 hash。输入数据保存于 datasets/；base_results/、base_evidence/ 是 e16 中心化基线，供离线对照。

```sh
python analyze_results.py
```

需要 Python 和 NumPy，重训还需 PyTorch。analyze_results.py 只读取已保存测量，验证初始化、stream、冻结参数和初始输出平移，不重训。run_study.py 是原工作区执行入口，读取 e16 已归档 recipe，保存每 seed 并复用成功测量；其初始化修改在代码中显式可见。

本轮保留高有限 loss：109/360 终点超过 benchmark MSE cutoff=2；这不是 benchmark 答题评分。本研究不包含最终 benchmark test。
