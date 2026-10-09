# r077：固定核多步传播未恢复真实train chord方向

程序第77轮，D5第5轮。本轮只问：相同初始化共享Linear核传播256步，是否比一步更能复现保存真实train chord的LN−noLN方向？全部development，40/40新K测量，0新训练；复用120旧训练、40Rayleigh和40coupling成功cell。

## 阅读、核验与冻结

按序读取GOAL、AGENTS、state、task、全部四篇findings和report。报告已符合六节骨架，无需重排；没有inbox，不追加处理标记。上一轮科学收尾c94426acdd68e4100cfde3cc978addf6662061e9、预注册27fc1a09d7958aea90d751990f7c14951d55135a均为HEAD祖先，98个历史收尾artifact匹配提交，95个当前方向artifact匹配；350旧文件及80耦合结果SHA/mtime未变。

新合同、run.py、analysis.py、verify.py和444个历史文件manifest在95663c6727ecd7e3aff6d0de313bbeaa5fd8f3bd提交。manifest固定120个旧NPZ的train_x/train_y/train_predictions_0/256数组pins；注册前没有计算新train chord或K传播。commit epoch=1791338825，gate epoch=1791338841.731983；全部新cell开始晚于门禁。git add -A按要求包含supervisor已有日志/prompt；未修改其逻辑或round/rounds_done。

## 算子与预测

J只对input与三个hidden Linear的weight/bias求导，排除head/LN affine，参数数d8/d12为13056/13312。逐样本float32 autograd构造J，再转float64以einsum构K=JJᵀ/256。固定eta=.001、a=3、v0=1，vt=vt−.002Kvt，Ct=9*mean[(vt−mean(vt))²]，只比较t1和t256。一步基线也从同一新K计算，旧float32 Jg coupling只做容差交叉核验。

真实train Em是保存train_predictions_256_m−train_y_m的居中残差平方均值，Creal=(E−3+E+3)/2−E0。两者Δ均定义LN−noLN；abs≤1e−12为零方向并算不匹配。P1预测M256≥16/20；P2预测M256−M1≥10。旧test20/20受益、旧ell20/20 LN更高已知，真实train及完整K尚未先算；这是development候选验证，不是sealed OOD。

## 结果与反例

40cell累计计算14.688749456秒，0新训练；先1cell独立核验，再补39。真实train20/20 ΔC为负，范围−.288059215～−.027001544。固定核一步20/20 ΔC为正，范围3.003088e−7～1.326255e−6；256步仍20/20为正，范围+.014591431～+.043046065。M1=M256=0/20，P1/P2均refuted，改善0；不修改符号、阈值或终点。

四函数真实train ΔC均值依次−.253569426/−.072570879/−.042038086/−.086818135；固定256步均值+.031789444/+.025171443/+.023278645/+.023644030，所有配对seed t95区间保持相反方向。描述性相关一步−.565540903、256步−.452698990，不作为成功判据。LN/noLN固定chord比一步2355.235629～129377.881531、256步1682.275155～68090.194659。最大eta lambda_max=.001323758247<1；方向失败不是注册传播的数值不稳定。

## 复算与边界

保存K、谱、t0/1/256向量、seed/数据合同、原train标签和预测；不保存完整J，可由冻结源码重建。全40cell两随机VJP→forward JVP探针最大相对误差3.970981e−7；旧K1 maxabs2.667165e−7、PK1 rel8.894393e−6、ell rel1.905531e−6。独立eigh传播向量maxabs2.142730e−14、chord rel1.367005e−10，原始train残差标量复算ΔC误差1.332268e−15，均通过注册门槛。444历史文件和80新结果SHA/mtime未变，恢复0新测量；旧成功cell不重跑、不覆盖。

已知解析Ct只适用于fixed-J、全批量、无decay、MSE mean residual²及精确加性offset的train线性化。真实训练为小批量SGD/decay/非线性且LN affine可更新，保存标签还含float32平移量化。本结果反驳当前共享Linear初始化核的指定终点方向预测，以及“仅延长一步传播即可修正方向”的候选；不定位缺失机制，不证明test机制或因果中介。旧Gram .420774419、新跨offset .282604295、Rayleigh两项失败、一步coupling两项失败、初始化κ60/60更低及训练仅7/60下降全部保留。原matmul警告根因未定，旧代码未改。

下一轮仅development：在固定数据/eta/a/T下加入LN affine初始化核项，复用共享Linear/noLN保存K，只测20个新的LN affine Jacobian；先固定数值预测、旧核/train预测hash与源码并commit。0新训练、不重测旧成功核。

证据：studies/r077_fixed_kernel_train_chord/preregistration.json、summary.json、executed/prior_handoff_verification.json、executed/receipt.json、executed/first_cell_verification.json、executed/saved_evidence_verification.json。复算入口analysis.py与executed/verify.py，指定Python -B。
