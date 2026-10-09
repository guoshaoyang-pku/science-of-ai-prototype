# r041：独立 batch 流块是否保留两倍内最低点

固定 D2 n=64、d=4、冻结 ReLU width128+常数、零初始化 head、float64 MSE、无动量/无衰减 SGD，η=.1、B=8、σ²=1、seed411/412、原 data/ε、T=2304 和同标签 full 分母49/306。只运行独立流块 r=64–127；不复制、覆盖或合并旧 r=0–63。预注册及执行/分析/核验源码冻结于94d66e1，早于训练。完成2/2 cell、128条新轨迹、5.783148秒；复用2个full控制，0新full训练；1 recipe、2数据seed。全部development。

| seed | 旧块 r0–63 的 t*/t*_full | 新块 r64–127 的 t*/t*_full | 新块95%条件bootstrap区间 | 新−旧比值 | 独立重抽差值95%描述区间 |
|---|---|---|---|---|---|
| 411 | 57/49=1.163265 | 56/49=1.142857 | [.836735,1.836735] | −.020408 | [−.408163,.755102] |
| 412 | 235/306=.767974 | 393/306=1.284314 | [.765278,1.866013] | .516340 | [−.395507,1.140523] |

P1支持2/2：新块两个点估计都在[.5,2]。P2支持2/2：两个95%条件percentile bootstrap上界均严格<2；P2没有要求下界≥.5。P3支持：风险有限、均值首argmin在窗口内、新旧batch seeds无交集，258个旧研究文件hash/mtime不变。两个新旧差值区间都跨零；这是同数据seed比较、两块独立重抽，不能写成同随机流配对或证明两块相等。bootstrap比值>2的比例为.003/.015，仅描述固定data/ε的轨迹重抽，不作p值或精确batch期望越界概率。

两倍内点估计及条件区间上界在两个独立64条流块保持，但seed412的最低点可在235与393步间改变。64条轨迹仍留下最低点抽样变化；本结果支持旧八轨迹反例受batch Monte Carlo影响的候选解释，没有证明精确batch期望最低点必在两倍内。旧M8反例与r025/r029 P1/P2失败继续有效，不能将旧新块合并追溯改判，也不能与其他仍为M8的22个cell组成统一24/24保证。

期望标签噪声分母下新比值仅作描述，为56/202=.277228、393/179=2.195531。固定ε与期望ε的风险口径仍不同；不把差异全部归因batch。不外推其他n、优化器、learned features、连续η阈值、稳定性机制、无限训练或train-only早停。

## 核验与边界

开始时核验r033回执中的26个历史文件hash与科学收尾6767484一致；21个当前study文件仍匹配，冻结b046352是收尾祖先。旧r029真实科学收尾ea1a3e8和缺失原最终回执的披露保留，不改旧文件。本轮输入及pinned源码匹配冻结git字节。独立einsum最终audit风险和第一条新流前16步重放最大误差均≤1.110223e−16；bootstrap以重抽计数加权独立复算，与保存t*、区间、差区间一致。恢复只核验并跳过两个成功cell，9个数据/结果/manifest文件hash与mtime不变。

训练再次出现matmul divide/overflow/invalid警告，所有保存数组有限；原日志与源码保留，根因仍未定。旧核验源码训练后改pin的历史限制不变。

下一小问题候选：保持同data/ε、η=.1、B8、σ²1与2304步，在训练样本张成空间内使用独立无放回batch的精确均值/协方差递推，检验有限窗条件batch期望风险最低点是否也在同标签full两倍内。先注册公式、数值预测、计算判据与源码，再计算；已知矩递推本身不登记为新发现，不因预测失败换窗。

## 证据

- [预注册](../studies/r041_independent_batch_block/preregistration.json)、[执行源码](../studies/r041_independent_batch_block/executed/run.py)、[分析源码](../studies/r041_independent_batch_block/analysis.py)、[汇总](../studies/r041_independent_batch_block/summary.json)。
- [旧输入与收尾审计](../studies/r041_independent_batch_block/executed/input_audit.json)、[独立核验](../studies/r041_independent_batch_block/executed/verification.json)、[恢复核验](../studies/r041_independent_batch_block/executed/resume_verification.json)、[bootstrap原始数组](../studies/r041_independent_batch_block/results/bootstrap.npz)、[训练日志](../studies/r041_independent_batch_block/executed/run.log)。
- 冻结94d66e1；科学收尾以[最终提交核验](../studies/r041_independent_batch_block/executed/final_commit_verification.json)为准。
