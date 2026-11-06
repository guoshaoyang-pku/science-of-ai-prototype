# D1：去掉 student ReLU 的配对对照

只将原 student 隐藏层的 `ReLU` 替换为 `Identity`；teacher 仍为一个 ReLU 隐藏层。三种结构、三个 seed、六种优化器、学习率 0.001、2000 步及数据生成过程保持一致，共 54 个新训练单元。Linear 参数的初值与原实验逐层配对，初始输出、梯度和损失由新模型重新计算。

主检验仍为原始训练损失比 q=0.5 和相对步数误差 5%。保留 q=0.9、0.1、0.01 和全部截尾。额外计算同一训练数据上的仿射最小二乘损失下限，标注模型类不可达到的阈值。多层网络关于输入是仿射的，但所有层参数联合的损失通常不是二次函数。

预注册文件和执行源码只在本仓库提交；结果写入环境变量 `AIQ_KB_DATA_ROOT`（默认仓库旁 `AIQ_KB_DATA`）下的独立 `runs/d1_no_relu_20261105`。直接运行本脚本，不启动 supervisor。原 ReLU run 和图像均保留。

先 `python research/d1_no_relu_20261105/run.py --seal`，提交本目录的预注册和源码，再用 `python research/d1_no_relu_20261105/run.py --execute --prereg-commit <commit>` 执行。逐步训练/验证损失和 U 保存到 `metrics/`，每五步的展示曲线保存到 `results/`。出现非有限损失时停止并保留已写产物，不重启已完成单元。

本轮预注册 commit：`183113fec3dca79d9c45ebc55336b1a4add73c7e`。已完成 54 个训练单元、216 个优化器×阈值记录。`verify_results.py` 独立检查原数据和 Linear 初始化配对，使用全部 108054 个逐步观测复算端点及分组统计，结果见 `verification.json`。

主阈值 q=0.5：15 个非参考优化器×结构汇总格中，5 个通过、10 个未通过、0 个不可比较；通过条件为 3/3 个 seed 可比较且中位相对步数误差 ≤5%。RMSprop 的三个中位误差为零，但 9 个配对中 8 个只是预测与实测同在第 1 步跨过阈值；深 4 的持续端点为 1495、1741、1646，不能解释为曲线重合。仿射最优残差占初始损失的 0.191–0.246，因此 q=0.1 和 q=0.01 在本轮模型类内均不可达到。

图表使用完整 2001 点。`docs/模糊方向/plot_d1_linear_u_transfer.py` 生成新交互 HTML，`render_report.py` 更新 D1 第4节并保留原 ReLU 独立记录；`preserved_relu_figures.json` 核对原图文件内容未改变。
