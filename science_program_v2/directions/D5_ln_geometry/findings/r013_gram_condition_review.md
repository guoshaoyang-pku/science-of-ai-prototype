# r013：LN终点Gram条件数更低，chord相关幅度预测失败

程序第13轮、D5第2轮，只检验固定head的同宽度LN/noLN中，hidden Gram条件数方向及其与offset-centered chord收益的关联。

## 训练前审查与冻结

r005因预注册commit失败，0/120cell；本轮保留全部旧文件，另建r013。已按序读取GOAL、AGENTS、state、task、findings/report；本方向没有inbox.md。Git写权限已可用，没有修改权限或git config。

旧草稿把四函数×五seed写成8 pairs并以8函数为分母；两个改善量都定义为LN−noLN却预期负相关；源码未固定head，而v1 A03主结论来自固定head。本轮训练前明确改为20函数×seed重复、正相关r≥.5，并固定head weight/bias。旧P3的6/8函数判据不能按四函数执行，本轮不评估它，checkpoint仅描述。旧预测没有实验，不登记为测量失败。

冻结提交5144cbdf818c7ac3dca76d21fbb85bb5a3a4d036同时保存新合同和两份pinned源码。运行器逐字节核对commit和SHA，保存receipt后先运行1cell；合同、NPZ、数组有限性、独立Gram/head复算通过后才补余119cell。提交epoch=1791287966，执行门禁epoch=1791287981.530880，首cell开始epoch=1791287981.565667。恢复调用复核120cell合同/NPZ，新训练0cell，没有覆盖成功结果。

## 条件与结果

四个旧 development 函数：trigonometric8/quadratic8 为 d8 uniform[-1,1]，interaction12/radial12 为 d12 Gaussian；train/test 各256，数据seed48291，标签按train mean/标准差归一化。width64、GELU输入层及三个hidden block，LN010/noLN；固定head weight/bias，hidden和LN affine可训练。SGD lr=.001、momentum0、weight_decay1e-4、batch64、256步，offset −3/0/+3、seeds100–104。24个函数×recipe×offset条件，五seed仅为重复，120cell计算9.201867秒。

Gram为train hidden列居中后的HᵀH/n；模型float32，Gram float64，κ=lambda_max/max(lambda_min,1e-12)。保存step0/1/8/32/128/256的矩阵、谱和预测，step0/256另存hidden。Chord是测试残差去均值后的方差E的[E(−3)+E(+3)]/2−E(0)。两种Δ均为LN−noLN，更负的Δchord代表offset收益更大。

P1 supported：12/12函数×offset单元都达到至少3/5 seed终点Δlog10κ<0，超过注册的至少8/12；总计54/60配对下降。P2 refuted：20函数×seed重复中，offset0终点Δlog10κ与三offset的Δchord的Pearson r=.420774419，低于注册r≥.5。20重复只有四函数，不视为20个独立函数，不报告iid p值。

| 函数 | Δchord 均值 [配对 seed t95 区间] | offset0 Δlog10κ 均值 | 函数内 r |
|---|---:|---:|---:|
| trigonometric8 | -0.261939 [-0.308159, -0.215718] | -2.405511 | 0.7737 |
| quadratic8 | -0.017806 [-0.031305, -0.004307] | -2.389292 | 0.5130 |
| interaction12 | -0.040574 [-0.057891, -0.023257] | -1.785286 | -0.7628 |
| radial12 | -0.072184 [-0.102777, -0.041591] | -1.821541 | -0.7625 |

## 解释边界与复算

20/20 Δchord为负，但条件数降低不足以解释收益大小。扣除函数均值后的描述性r=−.097223524。LN初始化已有60/60配对κ更低；训练期间仅7/60 LN cell的κ比初始化更低。offset0四函数平均Δlog10κ从初始化−2.437/−2.387/−1.795/−1.819变为终点−2.406/−2.389/−1.785/−1.822。终点差异主要已存在于架构初始化，不证明训练改善条件数带来收益。以上checkpoint事实只作描述。

独立executed/verify.py只读结果，用einsum(optimize=False)复算；全部120 NPZ/hash、重生成输入、Linear初值、256个batch、固定head、720组谱、240个hidden→Gram矩阵通过。全部数组有限，Gram最大误差0，hidden→head预测最大误差5.461e-7，720 checkpoint均未命中1e-12 floor。冻结analysis.py的spectrum_contrasts里offset的mean键被统计mean覆盖，只影响标签；原summary保留，正确offset见executed/saved_evidence_verification.json。原NumPy matmul警告原样保留在executed/full_analysis.json，根因未定。

只测最后hidden表示Gram，不是参数Jacobian的神经切线核；没有目标谱投影、梯度对齐、曲率或谱干预。相关没有因果识别，本轮全为development，不外推新函数/width、其他优化器或封存OOD。下一小问题：同固定head recipe中，初始化hidden参数Jacobian的目标Rayleigh商是否比最后hidden表示Gram的总条件数更能预测同种子Δchord？先明确目标相关算子和跨offset聚合，另写数值预测、pinned源码并commit。

复算使用python3 -B，入口为directions/D5_ln_geometry/studies/r013_gram_condition_review/analysis.py和directions/D5_ln_geometry/studies/r013_gram_condition_review/executed/verify.py。主证据是summary.json、executed/receipt.json和executed/saved_evidence_verification.json。
