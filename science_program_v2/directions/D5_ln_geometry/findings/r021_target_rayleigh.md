# r021：初始化目标Rayleigh商的收益关联预测被推翻

程序第21轮、D5第3轮，只检验同固定head recipe中，初始化共享hidden Linear参数Jacobian的居中目标Rayleigh商是否比终点hidden Gram总条件数更关联同seed的LN chord收益。全部为development，没有新增训练或因果干预。

## 训练前流程与冻结

按序读取GOAL、AGENTS、state、task、两个已有findings和report；没有inbox。已有报告缺少公式区，先重排开头、保留全部已测数字与结论，把证据链接移至末尾。核验旧收尾b4eaaabb81ed88f6b10ff82d56ee46ef88bd7ec4为HEAD祖先，包含120/120成功cell；其原始源码和结果不改。

新合同、run.py、analysis.py、verify.py和输入manifest在3091e609af05c0278c93a89ed3f815462d762448提交。提交epoch1791297056，执行门禁epoch1791297068.733259；新增测量开始晚于门禁，0个新训练cell。manifest固定旧120cell的JSON/NPZ及7个合同/分析/核验文件，共247个SHA；逐文件与旧收尾commit比对。git add -A按要求包含supervisor已经产生的日志和本轮prompt；没有修改supervisor逻辑或轮次数。

## 算子、条件与判据

$J_0=\partial f_0(X)/\partial\theta_h$，$\theta_h$ 只含input及三个hidden Linear的weight/bias，排除固定head和LN affine；d8为13056参数，d12为13312。令$P=I-11^\top/n$、$y_c=Py_{train,0}$，目标Rayleigh商为$q=\|J_0^\top y_c\|^2/(n\|y_c\|^2)$，等价于居中核$PJJ^\top P/n$沿$y_c$的商。float32模型和autograd、float64目标乘积与梯度平方和；保存逐参数VJP、初值hash、seed、输入和数据合同，可从NPZ复算。

初始化与offset无关，数学上居中标签相同；浮点平移的居中标签误差按注册<3e-7核验。每函数×recipe×seed只测一次offset0，共8个初始化条件×5seed=40cell。数据和配方沿用四旧函数、d8 uniform/d12 Gaussian、train/test各256、数据seed48291、train标签归一化、width64/GELU/LN010或noLN、head固定、hidden及LN affine可训练、SGD lr=.001/mom0/weight_decay1e-4/batch64/T256。实际训练结果复用旧120cell。

正向收益$B=C_{noLN}-C_{LN}=-\Delta chord$，$X_R=\log_{10}(q_{LN}/q_{noLN})$，$X_G$为三个offset的step256 $\log_{10}\kappa_{noLN}-\log_{10}\kappa_{LN}$等权均值。旧offset0相关 .420774419与本轮聚合不同，不作同指标修订。

P1预测20个配对中$r(X_R,B)\ge.5$且比$r(X_G,B)$高至少.15；P2预测分别扣除函数均值后Rayleigh相关≥.3且优势≥.10。无方差记refuted，不完整记not_evaluated；20重复只含四函数，seed不算新函数，不给iid p值。

## 结果与误差范围

40/40测量完整，累计cell wall time .126054205秒；0个新训练cell。LN q=.0111210568–.1110317012，noLN q=.0000687318243–.000520556080；20/20配对LN提高83.831222–368.099840倍。该方向差异不能证明收益幅度说明力。

P1 refuted：总体Rayleigh r=.394664832、跨offset Gram r=.282604295，优势 .112060538，两项均低于.5/.15。P2 refuted：函数去均值后Rayleigh r=−.076836549、Gram r=−.295981431，优势 .219144881虽超过.10，Rayleigh r未达.3。

| 函数 | Rayleigh比值log10均值 [seed t95] | Rayleigh函数内r | Gram函数内r | B均值 [seed t95] |
|---|---:|---:|---:|---:|
| trigonometric8 | 2.420718 [2.343816,2.497620] | −.003899 | −.563509 | .261939 [.215718,.308159] |
| quadratic8 | 2.430916 [2.330509,2.531323] | .859822 | .597458 | .017806 [.004307,.031305] |
| interaction12 | 1.969700 [1.923333,2.016068] | −.908828 | −.651146 | .040574 [.023257,.057891] |
| radial12 | 1.969139 [1.923688,2.014590] | −.771687 | −.720661 | .072184 [.041591,.102777] |

描述性leave-one-seed-out每次删四函数的同seed；总体Rayleigh r=.362036–.418195、优势 .062854–.142270，函数去均值Rayleigh r=−.366976–.154053。区间只覆盖5seed初始化变化，不外推函数总体。原P1 12/12条件数方向通过、原P2 r=.420774419<.5 refuted、初始化60/60条件数更低且仅7/60 LN训练后下降均保留。

## 复算与解释边界

首cell用逐样本autograd构造256×13056 Jacobian，独立$J^\top y_c$与保存VJP最大绝对误差1.607123e-6、L2相对误差2.567571e-7，q相对误差1.788723e-8，均低于注册阈值。全40cell重新生成初始化与VJP逐字节一致，q最大复算绝对误差1.388e-17。全部数组有限；旧120cell hash、Gram和chord复算通过，hidden→Gram误差0、终点floor命中0、配对预测量与相关复算误差0。恢复调用先核合同/hash，0个新测量cell，未覆盖成功cell。

本轮只反驳共享Linear参数、居中clean-target、冻结跨offset聚合的具体候选关联判据。LN affine在旧训练中可更新而不在该算子中；初始目标Rayleigh不是未居中MSE的残差下降率。固定Jacobian全批量线性化的chord由完整传播矩阵决定，初始$PK1$是均值到居中形状的耦合项；这是已知解析关系，不登记新发现，不能直接解释nonlinear/mini-batch/weight-decay/test终点。终点Gram是事后指标；本分析使用旧样本，不声称独立前瞻预测效能、因果中介或sealed OOD。旧NumPy matmul警告根因未定，旧源码和警告保留，新einsum与finite核验通过。

下一轮可只检验同recipe的初始化共享Linear核均值耦合强度$\ell=\|PK1\|^2/n$的同seed配对比与B的关联。先明确尺度与新数值预测并commit；不要因已知局部理论而把终点收益归因于该量。建议仍复用120个旧训练cell，保护本轮40个成功测量。

证据为studies/r021_target_rayleigh/preregistration.json、executed/input_manifest.json、executed/receipt.json、summary.json、executed/first_cell_verification.json、executed/saved_evidence_verification.json。复算入口analysis.py和executed/verify.py，使用指定Python -B。本轮新增两条refuted KB claim，分别对应总体与函数去均值判据。
