# r090：第四个数据 draw 的负相对收益检验

固定早期公式在 data_seed260609 的 T MAE 为 .485002856，相对 E 的平均绝对误差改善为 −.045043418，支持本轮唯一预测“改善<0”。点预期为 −.25，观测比点预期高 .204956582，负幅度没有达到点预期。T 只在5/12cell更好；四个recipe中两个平均改善、两个变差，四个三初始化95% t区间均跨零。不能说每个recipe或每个seed失效。

## 问题与冻结条件

仅将data_seed改为预先指定260609，保留d3/n32 Gaussian、product/mixed_sine×width8/32、init701/719/737、含bias的两hidden SiLU、float32默认Linear初始化转float64、full-batch halfMSE SGD eta.05/mom0/nodecay/T256。标签在train减样本均值并除ddof=0标准差。三公式逐字沿用原data260606校准系数，E=log(rho32)，Q为固定y的早期trace匹配核比值，响应L=log(rho256)；不重拟合。

唯一P1判据为Δ=MAE_E−MAE_T<0，四recipe等权、recipe内三seed等权。点预期−.25来自已见260607/260608的−.383048543/−.152998507，注册前没有生成260609数据。本轮没有注册T MAE>.70或改善<.08的联合条件；原三批预测判定均保留。

## 执行与核验

先按新骨架全文重排report，全部旧段落与数字token多重集保持，单独提交da36810；inbox已追加指定处理标记。历史审计核验r078科学收尾7d4f6e9的67归档blob、62个当前不可变科学文件、136旧input hash、151旧文件hash/mtime。旧四批共48成功cell文件完整，0重跑/覆盖。

预注册与五份pinned源码同提交eff188dbc2b9c6c39aeec697ae06a51243128f90，早于首cell21.032902秒。runner显式要求唯一完整引入commit，匹配合同/source/input与祖先关系；每次训练前审计旧48cell并全局检查本批12cell，拒绝孤立early/NPZ/failure/tmp、未知文件及成功cell额外/缺失文件。首cell保存分析通过后才运行余11cell。step32先保存E/Q/预测时间，再续训至step256。12/12cell、4/4recipe完整，累计cell计算1.233600秒。

全部36checkpoint由保存参数独立重建，初始化误差0，输出/J最大差均8.881784197e−16。核、trace、固定r0/y比值、loss、三MAE、四recipe区间和260606/260608配对变化均从保存数组复算通过。完成后恢复为0新cell/跳过12cell，首cell及36成功结果hash/mtime不变；200个冻结旧input hash和151旧文件hash/mtime未变。没有新runtime warning；旧matmul根因未定。

## 结果与配对边界

E/W/T MAE=.439959438/.477914375/.485002856；W相对E改善−.037954936，T改善−.045043418。T误差范围.079158190–.907918090；mixed_sine×width32×seed737预测1.950490378、L=1.042572288，误差.907918090为最大值。

| recipe | E MAE | W MAE | T MAE | T相对E改善 [初始化95% t区间] |
|---|---:|---:|---:|---:|
| product×width8 | .377713590 | .419613387 | .373101226 | .004612365 [−.831187598, .840412327] |
| product×width32 | .382049020 | .195931963 | .595459884 | −.213410864 [−.903691388, .476869659] |
| mixed_sine×width8 | .368957735 | .432849148 | .171729424 | .197228311 [−.294856875, .689313497] |
| mixed_sine×width32 | .631117409 | .863263002 | .799720891 | −.168603482 [−.534469269, .197262305] |

相对260608，T MAE降低.101469211，但product×width32误差增加.303591386，其同seed95% t区间[.125638581,.481544191]完全高于零；其余三个区间跨零。相对260606，T MAE增加0.043814039，mixed_sine×width32误差增加.529802274，区间[.182393616,.877210933]高于零；其余三个区间跨零。区间只描述三指定初始化的条件波动，不估计跨draw概率。

原方向mixed_sine×8的0/3反例、原幅度仅7/12改善、四CI跨零、mixed_sine×32变差.014469959/最大误差1.229845961、260607两项refuted/最大2.263445083，以及260608联合再现refuted/最大1.332692609均保留。data_seed共同改变输入、clean目标与样本居中/RMS尺度，不独立归因。全部development，无sealed OOD、核对齐因果、rt拟合或test外推。

## 下一小问题

可保持四recipe、全部训练条件与三冻结公式，仅换预先指定data260610，继续检验Δ<0。先注册新的点预期、来源hash与唯一pinned源码commit，匹配后再训练；全局恢复审计继续拒绝孤立/额外/缺失结果。负收益在三个指定新draw出现，不构成跨draw概率或普适规律。

## 证据

[预注册与五源码合同](../studies/r090_negative_transfer/preregistration.json)、[结果复算](../studies/r090_negative_transfer/summary.json)、[历史67blob/151文件审计](../studies/r090_negative_transfer/executed/historical_input_audit.json)、[36checkpoint参数/J核验](../studies/r090_negative_transfer/executed/independent_verification.json)、[260606配对与成功恢复](../studies/r090_negative_transfer/executed/saved_evidence_verification.json)、[260608配对核验](../studies/r090_negative_transfer/executed/previous_draw_verification.json)、[科学收尾回执](../studies/r090_negative_transfer/executed/final_commit_verification.json)。

[只读独立审查](../studies/r090_negative_transfer/executed/independent_review.json)。
