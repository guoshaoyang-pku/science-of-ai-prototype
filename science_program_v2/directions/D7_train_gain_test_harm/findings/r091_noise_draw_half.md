# r091：新噪声向量推翻半强度放大判据

固定 data7339、σ=.5、含 bias 两层 hidden SiLU、width8/32、init11/29/47、CPU float64 全批量 SGD η=.05/mom0/nodecay/T512，仅把 noise733123 换为733131，原“至少3/4单元 mean D≥.01 且各至少2/3初始化 D>0”预测被推翻：通过率从3/4降为1/4。新四单元 mean D 依次为 .086687、−.008957、−.013877、.003331，正初始化数为3/3、2/3、2/3、2/3。噪声强度和该 recipe 固定时，已测判据仍依赖具体噪声向量。

## 条件与唯一预测

程序第91轮、D7第6轮只检验一个小问题：固定 data7339 和 σ=.5，仅更换预指定 noise733131 后，半强度放大判据是否保持。d5 Gaussian train32/test256、mixed_sine/radial、train clean目标按均值与ddof0标准差归一化后加独立PCG64 Gaussian噪声，不center、不rescale、不再次归一化；test始终clean。默认float32 Linear初始化后double；两模型共享初始输出与全部weight/bias Jacobian，full-batch halfMSE、SGD eta=.05/mom0/nodecay/T512，检查点0/1/8/32/128/256/512。

Hσ为512步真实网络减初始切线的clean test halfMSE；同init D(.5)=H(.5)−H(0)。P1要求四个function×width单元至少三个 mean D≥.01且各至少两个init D>0。结果齐全后才判定；不调整阈值，不排除seed。三init不算三种独立recipe。

先按inbox六节要求重排累计报告，保留旧数字、表格、结论和反例，单独提交1ac6dbd；inbox已追加指定处理标记。d8e5c64同一提交冻结合同、run.py、analysis.py及verify_saved.py；唯一提交逐字匹配，早于首cell 14.285938 秒。旧r079科学收尾7b35a66、预注册652ea06与45科学文件核验通过，239旧文件hash/mtime冻结，72旧成功cell/144文件完整性通过。

12/12新cell、24条新轨迹完成，累计训练 3.074483 秒；读取12个已保存sigma0和12个旧sigma=.5 cell，不重跑或改写旧cell。旧clean保留原noise_base733123；新旧的输入、clean目标、test目标、归一化、参数与完整Jacobian精确相同，只有新half噪声向量不同。首cell核验后才续跑11cell。

## 结果与配对区间

| 单元 | mean D(.5) [95% 初始化区间] | D>0 | 通过 | 新−旧 mean D [95% 配对初始化区间] |
|---|---:|---:|---|---:|
| mixed_sine×8 | 0.086687 [-0.078611, 0.251986] | 3/3 | 是 | -0.038507 [-0.144628, 0.067614] |
| mixed_sine×32 | -0.008957 [-0.213162, 0.195248] | 2/3 | 否 | -0.238148 [-0.788678, 0.312382] |
| radial×8 | -0.013877 [-0.086397, 0.058643] | 2/3 | 否 | 0.032690 [-0.118269, 0.183649] |
| radial×32 | 0.003331 [-0.018499, 0.025161] | 2/3 | 否 | -0.057162 [-0.072850, -0.041473] |

区间为95% Student-t初始化区间，df=2，只描述固定data/noise下三init差异。新四D区间都跨零；只有radial×32的新−旧配对区间排除零。mean D小于.01使mixed_sine×32、radial×8和radial×32不通过，后三个单元仍各有2/3初始化D>0。

| 单元 | init | 新 D(.5) | 旧 D(.5) | 新−旧 D(.5) |
|---|---:|---:|---:|---:|
| mixed_sine×8 | 11 | 0.046409 | 0.049046 | -0.002637 |
| mixed_sine×8 | 29 | 0.050161 | 0.135928 | -0.085767 |
| mixed_sine×8 | 47 | 0.163492 | 0.190610 | -0.027117 |
| mixed_sine×32 | 11 | 0.051574 | 0.119811 | -0.068237 |
| mixed_sine×32 | 29 | 0.024098 | 0.181485 | -0.157387 |
| mixed_sine×32 | 47 | -0.102542 | 0.386279 | -0.488821 |
| radial×8 | 11 | -0.047586 | -0.070813 | 0.023227 |
| radial×8 | 29 | 0.002841 | 0.025633 | -0.022792 |
| radial×8 | 47 | 0.003114 | -0.094522 | 0.097636 |
| radial×32 | 11 | -0.006689 | 0.056744 | -0.063433 |
| radial×32 | 29 | 0.009730 | 0.066979 | -0.057249 |
| radial×32 | 47 | 0.006951 | 0.057754 | -0.050803 |

## train增益与晚期test变化

所有12新cell的train gain为正，四单元均值为 0.420114/0.372169/0.337023/0.222037。真实网络128→512 test变化均值为 -0.004235/0.133418/0.031635/-0.148465，切线为 -0.034122/-0.013378/-0.009094/-0.022993；真实晚期上升初始化数依次为1/3、3/3、2/3、0/3。以上均为描述诊断，无新成功判据。

mixed_sine×32 init47的新 D=−.102542，真实加噪test风险变化=−.072304，切线变化=+.030239，同时noisy train gain=.394740。相对差距减少不要求终点真实优于切线：该cell H(.5)=.124723仍正。radial×32的三init H(.5)与真实晚期C均负；均值H=−.122981、C=−.148465。终点H、相对噪声敏感度D与晚期C分别记录，不能相互替代。

## 核验与边界

保存NPZ复算全部loss、D、跨draw差、区间与主判据通过。重建初始参数、train/test Jacobian、初始及终点真实输出误差0；新旧36cell合计切线train/test检查点最大误差 4.996004e-15。独立math.fsum端点风险与NumPy差最大 1.110223e-16；解析df2区间与SciPy差最大 6.822654e-12，按事前1e−12均值/D、1e−10区间核验。239旧文件与24新成功文件hash/mtime不变，首cell未覆盖；恢复new0/reused12，所有新cell均绑定同一预注册commit。训练与分析stderr为空；旧matmul警告与根因未定记录保留。

全部为 development；新噪声向量在预注册前未生成或查看，但条件与判据依据已见旧结果选择，不称盲 OOD。三初始化只描述初始化差异；两个半强度 noise draw 不能估计跨 draw 概率。结果不拆噪声拟合、核方向变化和隐式偏置的比例，不给 train-only 早停、连续噪声阈值、线性/二次缩放或其他 optimizer/LN/深度/样本量外推。

下一小问题可只换预指定noise733139，在相同条件下检验本轮“不足3/4单元通过”是否再现；先注册相反方向的数值判据、来源hash和pinned源码单一commit后训练，不先生成噪声或看结果。新draw不承担机制解释。

## 证据

- [预注册](../studies/r091_noise_draw_half/preregistration.json)
- [逐cell与配对汇总](../studies/r091_noise_draw_half/summary.json)
- [完整保存证据核验](../studies/r091_noise_draw_half/executed/saved_evidence_verification.json)
- [独立主判据与恢复核验](../studies/r091_noise_draw_half/executed/independent_verification.json)
- [历史收尾审查](../studies/r091_noise_draw_half/executed/prior_closeout_audit.json)
