# 首簇开发诊断：随机 Teacher 的信号与短历史敏感性

## 结论

**本次随机 teacher 对“只保留最后 4 个 token、保留原位置”的干预变化较小，尚不足以支持长记忆题的定位。** 在 1 个 teacher、64 条开发审计序列的 1,024 个续写位置上，完整 teacher 与短历史 teacher 的 forward KL 为 0.0059 nats/token；该干预改变了输入分布，不构成长记忆的纯因果识别，也不外推其他 teacher。

**当前低阶基线没有优于均匀预测，应先校准基线再解释剩余结构。** 同一审计上下文上的 expected CE 分别为 uniform 5.545、unigram 5.624、bigram 5.548 nats/token；两种计数基线只用 1,024 个训练续写标签拟合，不能据其失败认定存在必要的高阶依赖。

## Formulation

问题与运行前约定见[首簇试点合同](TASK_PLATFORM_PILOT.md)。本次只测生成器和计数基线，没有训练 candidate 或 solver。

同一 teacher 产生的审计历史上，令完整 teacher 分布为 p、预测器分布为 q。主读数为 expected CE = −E Σ p log q；teacher entropy = −E Σ p log p；forward KL = expected CE − entropy。sampled CE 使用保存的单次采样标签，有限样本上不能直接减熵充当精确 KL。单位均为 nats/token。

## 成立程度

- 解析事实：uniform CE = ln(256)，约 5.545。
- 测量：该实例平均条件熵为 5.379，相对 uniform 的 KL 为 0.1666；这些描述分布差异，不证明 candidate 能在有限预算内学到它。
- 范围：1 个固定 teacher，train/audit 各 64 条序列，各 1,024 个续写位置；全部为 development-only。同序列 token 相关，没有跨 teacher 重复或不确定性区间。
- 已完成的验证：保存概率和权重 hash 检查、基线重新计数、全部指标从保存数组复算一致。它不等于独立重训或跨平台逐位复现。

## 方法

随机初始化并冻结因果 Transformer：2 层、hidden 128、4 heads、FFN 512、词表 256；均匀前缀 16 token，条件采样续写 16 token。teacher/train/audit seed 分别为 101/202/303。unigram 与 bigram 只拟合训练续写，使用加 1 平滑。短历史 teacher 保留最后 4 个 token 的原绝对位置。

| 预测器 | sampled CE | expected CE | forward KL |
|---|---:|---:|---:|
| Uniform | 5.545 | 5.545 | 0.1666 |
| Unigram | 5.604 | 5.624 | 0.2456 |
| Bigram | 5.548 | 5.548 | 0.1692 |
| 完整 Teacher | 5.369 | 5.379 | 0 |
| 最后 4 token 的 Teacher | 5.377 | 5.384 | 0.0059 |

训练与审计续写分别覆盖 251/256 个 token。CPU 单线程测量约 0.87 秒；该耗时不含依赖安装与单元测试，不作为性能基准。

## 失败与限制

Unigram 和 bigram 在这次开发审计上均未优于 uniform，结果完整保留。1,024 个训练标签对应 65,536 个 bigram 转移单元，计数稀疏且受平滑影响；较差预测既可能来自数据结构，也可能来自估计不足，当前测量不能区分。

没有预设商品质量通过阈值，因此不输出“合格题”判定。没有因结果不理想更换 seed、增加样本或重跑成功记录。短历史结果不能排除其他配置存在长程依赖，也不能证明该实例没有任何长程影响。

## 未决与下一步

建议下一份预注册只检验一个问题：**低阶基线的失败有多少来自计数不足？** 先固定当前 teacher 和评价口径，设计训练样本量与平滑的开发校准，并用已知有限阶生成器检查诊断器能否识别低阶结构。正式执行前另定样本、预算与判据；本次没有启动这一步。

Candidate 可学性、架构排名稳定性、solver 使用/不使用任务训练的迁移对照和客户验收仍未测。先补这些证据，再决定扩簇与报价。

## 证据

- 预注册 commit：`84bb5487df3f291e9f36ab4a9f7f9dca564cb84b`，先于本次测量。
- 测量开始：2026-10-19 12:07:14（Asia/Shanghai）；完整 UTC 时间和耗时见 `config.json`。
- 外部产物：`$AIQ_KB_DATA_ROOT/runs/random-teacher-dev-20261019/`，包含 `config.json`、`teacher.pt`、`arrays.npz`、`summary.json` 与自动生成的 `report.md`。本页是测量后的解读，未改动预注册合同。
- `arrays.npz` SHA-256：`72cf46837b9dfe5c0216875fb11f5fa3716e583dfc88e7bbdf95f8cb30ee3a0f`。
- 完整精度见 `summary.json`；代码与合同 hash、权重 hash、NumPy/PyTorch 版本见 `config.json`。
- 本机校验：125 tests、19 subtests 通过；默认交接检查通过（2,590 个文件，另有 13,722 个 Release 载荷文件尚未下载，因此不声称完整归档验证）。

只读复算：

```bash
PYTHONPATH=src .venv/bin/python -m aiq_kb.task_pilot --verify random-teacher-dev-20261019
```

运行前需将 `AIQ_KB_DATA_ROOT` 指向上述外部产物根目录；复算不重新采样或训练。
