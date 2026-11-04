# D1：去掉 student ReLU 的配对对照

只将原 student 隐藏层的 `ReLU` 替换为 `Identity`；teacher 仍为一个 ReLU 隐藏层。三种结构、三个 seed、六种优化器、学习率 0.001、2000 步及数据生成过程保持一致，共 54 个新训练单元。Linear 参数的初值与原实验逐层配对，初始输出、梯度和损失由新模型重新计算。

主检验仍为原始训练损失比 q=0.5 和相对步数误差 5%。保留 q=0.9、0.1、0.01 和全部截尾。额外计算同一训练数据上的仿射最小二乘损失下限，标注模型类不可达到的阈值。多层网络关于输入是仿射的，但所有层参数联合的损失通常不是二次函数。

预注册文件和执行源码只在本仓库提交；结果写入环境变量 `AIQ_KB_DATA_ROOT`（默认仓库旁 `AIQ_KB_DATA`）下的独立 `runs/d1_no_relu_20261105`。直接运行本脚本，不启动 supervisor。原 ReLU run 和图像均保留。

先 `python research/d1_no_relu_20261105/run.py --seal`，提交本目录的预注册和源码，再用 `python research/d1_no_relu_20261105/run.py --execute --prereg-commit <commit>` 执行。逐步训练/验证损失和 U 保存到 `metrics/`，每五步的展示曲线保存到 `results/`。出现非有限损失时停止并保留已写产物，不重启已完成单元。
