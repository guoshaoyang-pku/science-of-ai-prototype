# 定向 Science prototype 交接 · 2026-10-06

## 1. 交付状态

12项研究全部完成，7条短KB已登记。A01–A04独立复核通过；B/C有逐数组、contract和OOD时间核验，独立发布复核另存方向review。没有新增solver评测；本轮没有读取最终benchmark test。用户准备提供新目标，当前不启动新的research、solver或论文投稿。

旧主线soa_async_v1仍为15个解题epoch、batch30，143条live claim/77份报告。最新macro15在105道development ranking题、3重复上，KB70.08%，noKB65.87%，+4.21±2.37pt（题级配对1SE）。本次science的进步不是新score，不能替换这一结果。

## 2. 科学发现与边界

| 方向 | 研究链及证据 | 已排除或发现 | 尚未解决 |
|---|---|---|---|
| A | A01固定head→A02新函数预测→A03 LN×width→A04参数冻结 | head增长必要性被反驳；LN同宽度效应8/8区间<0；固定LN affine保留约99%，只学weights约100–111% | 特定Jacobian中介、Adam、未见recipe；A03/A04为development |
| B | B01谱/动量校准→B02真实/切线→B03同残差同trace交换核 | 无统一U阈值；8/8新条件train受益而6/8test变差；目标相关核改变超过总尺度 | 少量参量的泛化预测、普适SGD/Adam兑换与容量公式 |
| C | C01奇偶通路→C02无标签能量干预→C03可学习hidden→C04噪声风险→C05前瞻曲线 | 小SiLU对偶目标慢、线性目标不慢；C02 OOD交互MSE .953→.0043和1.136→.0150；C0543个不同recipe/checkpoint风险均在3SE内（48次预定检查有5次重复）、最优step17/24精确 | learned-hidden动态风险、非对称/bias、多层/Adam/CE、solver收益 |

A01原head必要性预测3/3失败，A02优化器赢家10/12、二次目标±3失败，C01 LN严格尺度不变失败，C04小SiLU最优时间只有中位数判据通过。失败均保留。C05允许看新train/test数据与clean标签预测训练结局，是已知任务的条件风险预测；不是未知标签盲预测。看过OOD结果后相应条件转development。

## 3. 工作量和复算

A1740次实际MLP训练；B360次optimizer训练（96真实MLP、72切线、192固定特征），另72条解析kernel probe；C624个recipe、1136条拟合轨迹。合计2652份保存recipe/cell、3236条optimizer拟合轨迹，另72条解析probe。不能把多seed、多noise draw或相关判据计作独立函数。已有e16/e17的600次训练为来源复用，不重复计入新增。

A实际逐cell计算约243秒，B约31秒，C约51秒；agent编写/分析与协调不包含在其中，并行计算之和不是整个研究walltime。完整计时与计算口径在各方向handoff和逐cell记录。新模型API/solver调用均0。

公共版的验证命令在README；原始版使用已安装NumPy/PyTorch/SciPy/Matplotlib的Python。A的analyze.py读取JSON/NPZ核对hash；B、C方向verify.py独立复算递推与contracts。analysis在临时copy中重跑，科学数字需与保存版本完全一致。成功cell恢复需核对源码、data和recipe合同；A01/A02现恢复入口已实测360/960全部复用，0新训练。

## 4. 文件与归属

| 文件 | 含义 |
|---|---|
| kb.json、state/kb_operations.jsonl | 七条短KB、原始轮次、发布合并时间与add/revise理由 |
| sources/manifest.json、sources/e16,e17 | development来源和冻结成功测量；旧实验不重跑 |
| studies/*/preregistration.json、executed/ | 原判据、条件、实际执行源码；B/C另有OOD预测/forecast封存 |
| studies/*/results/ | 全部原始成功和有限高loss测量；本地Git忽略，公开release附件必须携带 |
| directions/*/handoff.md、report.md、verify.py | 每方向具体发现、边界、续接命令与核验 |
| state/current.json | 合并状态，旧PID仅历史信息；必须现场核对身份 |

公共Git提交与原始研究commit不同；机器路径脱敏改变metadata时保存original/public hash，数值数组与执行Python保持byte identity。缺少私有Git历史时，预注册 chronology来自保存的时间/hash证据，不声称可在public Git重新验证原历史。失败保存工程探针不计科学证据；它尚缺历史执行器archive，晚期nonfinite分类也需进一步核对。

## 5. 运行契约及下一目标

默认简洁KB，报告可跨claim、保留竞争解释和未决问题，已有版本立即可用。异步研究归origin_epoch，应用时间另记applied；自然分叉，不设额外主题硬cap。宏观评测观察多轮成长，不用短val逐改挑噪声。来源暴露清单和题组/生成器隔离必须延续。

旧训练池只剩60道未见ranking_v2，async stream_offset当前不推进；新run/seed不能自动算新数据。强模型涨分需要新题池和固定模型/effort/预算的配对对照。论文需要完整相关工作与新颖性审查；本次仅核对部分摘要。

旧publisher和文件watcher保持运行。当前无真实自动steer工具或已接收探针证据，状态文件/PID不能证明守候有效；不再创建queue monitor。新目标到达后先读本文件和具体KB，再决定要补哪个预测、任务池或使用端实验。

## 6. 开源交付

公共仓库v0.2.0已发布：https://github.com/guoshaoyang-pku/science-of-ai-kb/releases/tag/v0.2.0。公开commit 4b7f5a44da920469850b1b88329601638ca138dd；三个原始测量包（C十四个分片）均已公网下载、逐文件核验并重算15份分析一致。详细交付、环境限制与附件hash见 [发布交接](../../docs/reports/KB_DIRECTED_SCIENCE_RELEASE_HANDOFF_2026-10-06.md)。当前状态released，等待用户的新目标。
