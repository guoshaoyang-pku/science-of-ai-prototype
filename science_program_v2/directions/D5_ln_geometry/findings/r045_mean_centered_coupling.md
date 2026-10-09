# r045：初始化均值到形状耦合的收益关联预测失败

程序第45轮、D5第4轮。本轮只检验同固定head recipe与共享Linear核的初始化mean-to-centered coupling是否比旧目标Rayleigh更关联同seed LN centered-chord收益。全部development，0新训练，复用120旧训练cell与40旧Rayleigh cell。

## 阅读、旧证据核验与冻结

依次读取GOAL、AGENTS、state、task、全部3个findings、report和inbox路径；没有inbox.md，因此不追加处理标记。旧报告章节不符合固定骨架，测量前重排，全部旧数字token集合核验不变。并行审查只读，无外目录写入。上一轮科学收尾bf6070cfbcf3b757fade55ccf5bda548c204c143及回执ffb9e1b9be5b8144303e0670bb208bed480b3ccb均为HEAD祖先；92收尾artifacts与提交一致，88不可变当前artifacts一致，旧120训练/40测量完整，不重跑或覆盖。

本轮合同、run.py、analysis.py、verify.py与350旧文件SHA/mtime manifest在27fc1a09d7958aea90d751990f7c14951d55135a冻结。commit epoch=1791323358，执行门禁epoch=1791323371.751065；首cell开始晚于门禁。固定旧数据/Linear初值/固定head pins，匹配提交及源码SHA后才测量。git add -A包含supervisor已有日志与prompt，未修改其逻辑、round或rounds_done。

## 算子、条件与预测

J0只对input及三个hidden Linear的weight/bias求导，排除head/LN affine，d8/d12参数坐标13056/13312。K=J0J0ᵀ/n、u=PK1、ell=||u||²/n=||PJ0(J0ᵀ1)||²/n³，n256。g=J0ᵀ1先detach，再二次reverse求J0g，避免Hessian；模型/autograd/J0g为float32，转float64后除n、去均值、平方均值。ell单位标签⁴/参数坐标⁴，依赖固定参数化，不加floor/epsilon。四函数、width64/GELU/LN010-noLN、head固定；旧训练SGD eta.001/mom0/decay1e-4/batch64/T256，offset−3/0/+3、seeds100–104，train/test各256、data48291。

原始log比L=log10(ell_LN/ell_noLN)，预期与B=C_noLN−C_LN负关联；等价用X_L=−L正相关比较X_R=log10(q_LN/q_noLN)。P1要求20配对r(X_L,B)≥.5且比r(X_R,B)高≥.15；P2要求按函数去均值r≥.3且优势≥.10。旧X_R相关.394664832/−.076836549在预注册中披露，无ell pilot；不属于封存OOD。8函数×recipe初始化条件×5seed，共40cell，不把20重复视为20独立函数。

## 结果与反例

40/40测量完成，累计计算0.140806209秒，0新训练。ell_LN=.00834409660–.03684068555，ell_noLN=2.847523829e−7–6.174839697e−6；20/20 LN更高，LN/noLN范围2,355.235493–129,377.971053。真实测试chord收益仍20/20为正，这与简单局部抑制解释相反。

P1 refuted：r(X_L,B)=−.452289043815，旧Rayleigh=.394664832223，优势−.846953876037；两个阈值都未达。P2 refuted：函数去均值r=−.055703551249，旧Rayleigh=−.076836549465，优势.021132998216；两个阈值都未达。原L相关+.452289043815/+.055703551249，符号不作事后翻转。各函数X_L均值/t95与函数内相关见summary及方向报告；trigonometric8/quadratic8/interaction12/radial12函数内r为−.012539/−.525836/−.009477/−.006013。

删除一个seed的四函数重复后，总体r范围−.525269–−.367698，优势−.943464–−.753289；去均值r范围−.299069–.187275，仅描述稳健性。五seed区间只覆盖初始化；原尺度耦合比t区间可跨零，不能当物理范围或iid显著性。

## 复算与边界

每cell保存g/K1/PK1及数据合同，ell保存复算误差0；独立forward-mode JVP验证全部40cell，VJP逐字节一致。K1最大绝对误差2.384186e−7，PK1最大相对误差7.533457e−6，ell最大相对误差1.484466e−6。首cell显式J形状256×13056，PK1/ell相对误差4.755500e−7/8.093449e−8，均低于注册1e−4/2e−4。旧B/X_R/相关独立复算通过；350旧文件及80新结果hash/mtime未变，恢复运行0新测量。

固定J、全批量、无decay、MSE=mean(residual²)的train线性化有A=I−2etaK、C_t=a²||PAᵗ1||²/n、C_1=4eta²a²ell。解析式是已知关系，只用于解释候选量尺度；不作为新发现，不归因真实非线性、小批量、decay或test终点。原训练LN affine可更新而本算子排除它；核漂移、多步传播、LN affine及train/test算子差异均未区分。旧Gram P2=.420774419失败、旧目标Rayleigh两项失败、初始化条件数60/60更低及训练仅7/60下降都保留；原matmul警告根因未定，旧源码与summary不改。

下一小问题仅development：相同参数集合下，固定核256步train chord与保存真实train chord方向是否更接近；先明确尺度/数值预测与pinned源码、旧train预测hash并commit，零新训练，train/test分开，不称因果机制。

证据：studies/r045_mean_centered_coupling/preregistration.json、summary.json、executed/receipt.json、executed/saved_evidence_verification.json、executed/handoff_verification.json。复算入口为analysis.py与executed/verify.py，均使用指定Python -B。
