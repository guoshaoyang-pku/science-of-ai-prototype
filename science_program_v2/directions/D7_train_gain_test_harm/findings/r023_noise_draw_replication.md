# 新noise draw733123下，相对test差距放大保持

程序第23轮、D7第3轮只检验一个问题：固定数据、目标、初始化、架构和SGD，仅将noise seed733107换为预先指定733123，原P1是否保持。12/12新sigma1 cell、24条新轨迹完成，累计训练2.940749秒；引用12个已保存sigma0 cell，分析24cell、12配对。P1在4/4函数×宽度单元通过，12/12 D>0；全部development。

## 冻结合同与提交时序

d5 Gaussian，train32/test256，mixed_sine/radial，含bias两层hidden SiLU，width8/32、init11/29/47、data7331。clean目标使用train均值与ddof0标准差；train_y=clean_y+σ ε，σ=0/1，ε来自独立PCG64 noise733123，不center、不rescale，跨函数/宽度/init共用；test始终clean。float32默认Linear初始化后double；CPU float64、full-batch halfMSE、SGD η=.05/mom0/nodecay、512步，检查点0/1/8/32/128/256/512。

原r007合同20b2412、恢复合同ead3790及旧24成功cell只读保留。核验r015真正科学收尾为8fa680d，0c64176为后续收据提交。所有旧cell的JSON/NPZ、summary、源码、合同、警告日志hash先冻结。sigma0的训练标签与轨迹与noise seed无关，引用原12clean cell；它们保存的noise_base仍是733107，不伪改成新噪声。新noisy与旧clean的x、clean_y、test_y、归一化、初始参数和两组完整Jacobian均精确相同。

本轮合同、数值预测、executed/run.py与analysis.py同提交06da355（2026-10-06T23:14:37+08:00）先于训练；runner匹配提交与全部引用hash后记录pre_execution_audit，再训练首cell。首cell分析通过后才执行剩余11cell；首cellhash未变。成功cell恢复只核验、不覆盖。原报告开头不符合要求，训练前已重排公式区，旧数字及科学结论保留。

## 唯一预测与结果

Hσ为第512步真实网络test半MSE减初始切线test半MSE，D=H1−H0在同init配对。P1要求4个函数×宽度单元至少3个mean(D)≥.02且各至少2/3 init D>0。完整12新+12旧cell核验后才判定，不改阈值、不排除seed。95% Student-t区间df=2只描述初始化差异，不是跨data/noise区间。

| 单元 | mean(H0) | mean(H1) | mean(D) [95% init区间] | D>0 | 新−旧mean(D) [95%配对init区间] |
|---|---:|---:|---:|---:|---:|
| mixed_sine×8 | 0.116602 | 0.548878 | 0.432276 [-0.496721, 1.361273] | 3/3 | -0.125232 [-0.395129, 0.144665] |
| mixed_sine×32 | 0.052857 | 0.519309 | 0.466452 [0.248028, 0.684876] | 3/3 | -0.334534 [-0.518798, -0.150270] |
| radial×8 | 0.016454 | 0.280438 | 0.263984 [-0.212486, 0.740453] | 3/3 | -0.037789 [-0.880046, 0.804468] |
| radial×32 | -0.231632 | -0.104421 | 0.127211 [-0.011045, 0.265468] | 3/3 | -0.159881 [-0.462640, 0.142877] |

| 单元 | init | 新D | 旧D | 新−旧D |
|---|---:|---:|---:|---:|
| mixed_sine×8 | 11 | 0.215316 | 0.464349 | -0.249033 |
| mixed_sine×8 | 29 | 0.864100 | 0.909841 | -0.045742 |
| mixed_sine×8 | 47 | 0.217412 | 0.298333 | -0.080921 |
| mixed_sine×32 | 11 | 0.368095 | 0.671382 | -0.303287 |
| mixed_sine×32 | 29 | 0.537440 | 0.818534 | -0.281094 |
| mixed_sine×32 | 47 | 0.493820 | 0.913042 | -0.419222 |
| radial×8 | 11 | 0.459808 | 0.106092 | 0.353716 |
| radial×8 | 29 | 0.076469 | 0.309436 | -0.232967 |
| radial×8 | 47 | 0.255674 | 0.489790 | -0.234116 |
| radial×32 | 11 | 0.120691 | 0.244155 | -0.123464 |
| radial×32 | 29 | 0.185840 | 0.246205 | -0.060364 |
| radial×32 | 47 | 0.075103 | 0.370918 | -0.295815 |

新mean(D)均低于旧draw，但只有mixed_sine×32的配对区间排除零；radial×8 init11的新D反而增大.353716。三个新D区间跨零，原判据不要求区间排除零。不能把4/4通过改写成稳定幅度或所有配对都下降。

## H终点与C晚期变化分别记录

C为同模型第512步减第128步test halfMSE；正值表示晚期test上升。以下各C区间仍是三init的df2区间；late_train_change与endpoint_clean_train_loss由summary逐cell复算。

| 单元 | σ | 真实C均值 [95% init区间] | 真实C>0 | 切线C均值 [95% init区间] | 切线C>0 |
|---|---:|---:|---:|---:|---:|
| mixed_sine×8 | 0 | 0.098034 [-0.055609, 0.251678] | 3/3 | -0.020969 [-0.056553, 0.014614] | 0/3 |
| mixed_sine×8 | 1 | 0.543233 [-0.522171, 1.608637] | 3/3 | -0.017810 [-0.050267, 0.014648] | 0/3 |
| mixed_sine×32 | 0 | 0.032336 [-0.085389, 0.150060] | 2/3 | 0.036104 [0.033059, 0.039150] | 3/3 |
| mixed_sine×32 | 1 | 0.501325 [0.192999, 0.809650] | 3/3 | 0.016459 [-0.048233, 0.081150] | 2/3 |
| radial×8 | 0 | 0.012496 [-0.234342, 0.259333] | 1/3 | 0.004285 [-0.042775, 0.051345] | 1/3 |
| radial×8 | 1 | 0.279858 [-0.375847, 0.935563] | 3/3 | 0.028049 [-0.077675, 0.133773] | 2/3 |
| radial×32 | 0 | -0.274327 [-0.407370, -0.141284] | 0/3 | -0.043800 [-0.065702, -0.021898] | 0/3 |
| radial×32 | 1 | -0.131020 [-0.277700, 0.015659] | 0/3 | 0.010586 [-0.050838, 0.072009] | 2/3 |

新draw的radial×32三init均D>0（.075103–.185840），却均H1<0（−.151615至−.047987），且真实C<0（-0.198348至-0.088044）。D衡量的是对切线的相对噪声敏感度，不要求终点真实风险更高或同模型晚期风险上升。

原两分歧仍保留：mixed_sine×32 init11 clean H=.020497、真实C=−.001370；旧noise733107的radial×32 init29 noisy H=.012378、真实C=−.029205。新draw没有覆盖旧noisy结果；原第二样本仍是有效历史反例。

## 验证与解释边界

全36cell（24旧+12新）hash/原提交/data/finite/半MSE核验通过。重建初始参数、train/test Jacobian及真实初始/终点输出最大误差0；einsum复算切线512步所有train loss及train/test检查点，新旧合计最大误差7.147061e−15。当前analysis stderr为空。旧matmul divide/overflow/invalid原日志hash未变、根因未定；旧复算误差<7e−15与本轮最大值分开记录。独立直接NPZ复算12D、4mean及df2区间完全匹配；提交时间早于gate、gate早于首结果。12新cell恢复hash不变且没有新训练；全部旧引用hash不变。

测量支持同一data draw上的第二个噪声干预仍扩大真实网络相对切线的test差距，但幅度依赖具体噪声向量。新noisy的train增益均值为.380688–.502691，所有12seed增益为正；这不等于对clean目标更好。确定性clean的相对损害继续存在，添加标签噪声不是必要原因；噪声拟合、有限样本下方向变化、隐式偏置及学习速度差仍是竞争解释。两个draw不能提供跨噪声概率或机制比例，也不能提供train-only早停信号。

下一小问题：保持新noise733123、当前函数/宽度/init及SGD，事先指定一个新data seed，检验至少3/4 mean(D)≥.02且各≥2/3 init D>0是否保持。新data会同时改变输入、clean目标及train归一化，不拆成单独机制。先锁定新seed、数值预测、源码和旧引用hash并commit，匹配提交后才训练；不重跑旧成功cell。全部development。

## 证据

- [预注册](../studies/r023_noise_draw_replication/preregistration.json)
- [配对汇总](../studies/r023_noise_draw_replication/summary.json)
- [全量输出与Jacobian核验](../studies/r023_noise_draw_replication/executed/saved_evidence_verification.json)
- [独立主判据核验](../studies/r023_noise_draw_replication/executed/independent_primary_verification.json)
- [无覆盖恢复核验](../studies/r023_noise_draw_replication/executed/resume_verification.json)
- [旧真正收尾审计](../studies/r023_noise_draw_replication/executed/r015_closeout_audit.json)
- [旧分歧与警告记录](../studies/r015_noise_interaction_resume/summary.json)
