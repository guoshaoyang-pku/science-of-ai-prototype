# 新data7339下，相对test差距放大判据保持

程序第47轮、D7第4轮只检验一个development问题：固定noise733123及原recipe，只将data seed7331改为预指定7339，原.02、2/3、3/4判据是否保持。24/24新cell、48轨迹完成，累计训练4.596528秒，P1 supported 4/4，12/12 D>0。没有复用旧clean轨迹作为新data对照，也没有重跑或覆盖旧36cell。

## 合同与提交时序

d5 Gaussian train32/test256，两函数mixed_sine/radial、含bias两hidden SiLU、width8/32、init11/29/47；默认float32 Linear初始化后double，CPU float64，full-batch halfMSE、SGD eta=.05/mom0/nodecay/T512。clean目标使用train均值与ddof0标准差，train_y=clean_y+sigma epsilon，sigma0/1；PCG64 noise733123的同一epsilon跨函数/宽度/init共用，不center、不rescale、不在加噪后归一化，test始终clean。检查点0/1/8/32/128/256/512。

先核验旧r023科学收尾09388db及final_commit_verification，r015真正收尾8fa680d、0c64176仅收据；120历史文件hash/mtime先冻结，36历史成功cell只读。旧报告训练前改为固定六节，旧数字与科学结论全部保留；不存在inbox.md。5aa2a41同提交锁定新seed、数值预测、run.py、analysis.py及旧引用hash，先于全部训练，runner逐字核验后记录gate。首cell分析通过后才运行其余23cell；首cellJSON/NPZ hash及mtime不变。

H_sigma为第512步真实减切线的干净test halfMSE，D=H1−H0在同init配对。唯一P1要求至少3/4函数×宽度单元mean(D)>=.02且各至少2/3 init D>0；24新cell齐全且核验后才判定。三init不是三种独立recipe。95% Student-t区间df2仅描述固定data/noise下初始化差异，不含跨data/noise不确定性。

## 预测与逐初始化结果

| 函数×宽度 | mean(H0) | mean(H1) | mean(D) [95%初始化区间] | D>0 | 新−旧data的mean(D) [配对初始化区间] |
|---|---:|---:|---:|---:|---:|
| mixed_sine×8 | -0.079940 | 0.389431 | 0.469371 [0.065910, 0.872832] | 3/3 | 0.037095 [-0.848464, 0.922655] |
| mixed_sine×32 | 0.141868 | 0.661381 | 0.519513 [0.129691, 0.909335] | 3/3 | 0.053061 [-0.194093, 0.300216] |
| radial×8 | 0.049707 | 0.161599 | 0.111892 [-0.081196, 0.304979] | 3/3 | -0.152092 [-0.728371, 0.424187] |
| radial×32 | -0.126312 | 0.059985 | 0.186296 [0.114519, 0.258074] | 3/3 | 0.059085 [-0.023063, 0.141233] |


radial×8的D区间跨零，原判据不要求区间排除零。固定noise下跨data的四个配对差区间全部跨零；除radial×8外，三个单元均值上升，但mixed_sine×8 init29下降.334783、mixed_sine×32 init11下降.017555。radial×8均值下降.152092，init29反而增加.019018。不宣称跨data单调性或稳定幅度。

| 函数×宽度 | init | 新D | 旧data同noise的D | 新−旧D |
|---|---:|---:|---:|---:|
| mixed_sine×8 | 11 | 0.285504 | 0.215316 | 0.070188 |
| mixed_sine×8 | 29 | 0.529317 | 0.864100 | -0.334783 |
| mixed_sine×8 | 47 | 0.593293 | 0.217412 | 0.375881 |
| mixed_sine×32 | 11 | 0.350540 | 0.368095 | -0.017555 |
| mixed_sine×32 | 29 | 0.547331 | 0.537440 | 0.009890 |
| mixed_sine×32 | 47 | 0.660669 | 0.493820 | 0.166848 |
| radial×8 | 11 | 0.043675 | 0.459808 | -0.416133 |
| radial×8 | 29 | 0.095487 | 0.076469 | 0.019018 |
| radial×8 | 47 | 0.196513 | 0.255674 | -0.059161 |
| radial×32 | 11 | 0.167708 | 0.120691 | 0.047017 |
| radial×32 | 29 | 0.219585 | 0.185840 | 0.033745 |
| radial×32 | 47 | 0.171596 | 0.075103 | 0.096493 |


## 终点与晚期分别记录

C=同模型第512步减第128步的test halfMSE，正值表示晚期test上升。这是描述诊断，没有新成功判据。全部12新noisy H1>0，但radial×32 init47 noisy仍H1=.036880343842065455、真实C=−.021062709533282042；该单元真实晚期上升只有2/3 init。新noisy12/12 train增益G>0，单元均值.433225–.573030；更低带噪train loss不等于更接近clean目标。

| 函数×宽度 | σ | 真实C均值 [95%初始化区间] | 真实C>0 | 切线C均值 [95%初始化区间] | 切线C>0 |
|---|---:|---:|---:|---:|---:|
| mixed_sine×8 | 0 | -0.090694 [-0.316765, 0.135377] | 1/3 | -0.042038 [-0.067502, -0.016574] | 0/3 |
| mixed_sine×8 | 1 | 0.365599 [-0.214305, 0.945504] | 3/3 | -0.060519 [-0.072150, -0.048889] | 0/3 |
| mixed_sine×32 | 0 | 0.137256 [-0.057319, 0.331830] | 3/3 | -0.023557 [-0.030044, -0.017069] | 0/3 |
| mixed_sine×32 | 1 | 0.661309 [0.137984, 1.184634] | 3/3 | -0.018436 [-0.044827, 0.007954] | 0/3 |
| radial×8 | 0 | 0.042600 [-0.227151, 0.312351] | 2/3 | -0.014440 [-0.023567, -0.005312] | 0/3 |
| radial×8 | 1 | 0.148980 [-0.258661, 0.556621] | 3/3 | -0.012457 [-0.022871, -0.002042] | 0/3 |
| radial×32 | 0 | -0.159467 [-0.164211, -0.154722] | 0/3 | -0.045886 [-0.069071, -0.022702] | 0/3 |
| radial×32 | 1 | 0.021952 [-0.089541, 0.133444] | 2/3 | -0.028869 [-0.073619, 0.015881] | 0/3 |


旧反例保留：mixed_sine×32 init11 clean H=.020497而真实C=−.001370；noise733107的radial×32 init29 noisy H=.012378而真实C=−.029205；旧data7331/noise733123 radial×32的3/3 D>0却3/3 H1<0且真实C<0。旧noise733123的radial×8 init11比noise733107增加.353716以及三个跨零区间均保留。

## 保存证据核验与边界

60cell（旧36+新24）数据、finite、request/NPZ与对应提交字节通过，初始参数、完整train/test Jacobian、初始和终点真实输出重建误差0。einsum切线全部train loss及所有train/test检查点最大误差：新24cell单独6.161738e−15，新旧60cell合计7.147061e−15；旧独立复算<7e−15的历史结论保留，不合并篡改。旧matmul警告全文hash不变、根因未定；本轮training/analysis stderr为空。

独立verify_saved.py直接NPZ复算D/H/跨data差，均值与逐seed数值完全一致；df2区间用解析分位数，与SciPy区间最大差1.097433e−11。首次核验以1e−12比较区间而失败，原源码与错误保留；仅独立区间核验容差改为1e−10，原科学阈值、pinned分析和结果均未改，均值/D核验仍1e−12。无新训练的恢复24/24成功cell核验通过，120旧文件与48新成功文件hash/mtime全部不变。

换data同时改变train/test输入、clean目标与train归一化；保持参数/noise相同仅控制其余因子，不能把总效应归为一个机制。结果支持一个额外预指定data draw上的相对噪声敏感度差，不能估计跨data/noise概率或机制比例，不形成train-only早停规则，不外推其他sigma、optimizer、LN、深度或数据规模。

下一小问题建议：保持data7339/noise733123及原函数/宽度/init/SGD/T512，仅新增sigma=.5的12cell，复用本轮sigma0/1轨迹，检验D(.5)=H(.5)−H(0)是否至少3/4单元mean>=.01且各>=2/3 init>0。先保存明确预测、新pinned源码与本轮引用hash并commit，匹配后才训练；这不预测线性/二次噪声缩放或连续阈值。

## 证据

- [预注册](../studies/r047_data_seed_transfer/preregistration.json)
- [配对汇总](../studies/r047_data_seed_transfer/summary.json)
- [60cell保存证据核验](../studies/r047_data_seed_transfer/executed/saved_evidence_verification.json)
- [独立复算与不覆盖恢复](../studies/r047_data_seed_transfer/executed/independent_verification.json)
- [旧科学收尾审核](../studies/r047_data_seed_transfer/executed/prior_closeout_audit.json)
- [独立区间首失败](../studies/r047_data_seed_transfer/executed/independent_verification_first_failure.json)
