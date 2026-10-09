# D9：小批量噪声与有效时间

## 结论

测量结论仅适用于以下固定数据合同。输入维度 d=4、训练样本 n=64、冻结 ReLU 特征（使用 max(0,x) 的固定非线性特征）宽度128、含常数项且零初始化的线性头（head）。实验使用 float64（64 位浮点数）、MSE（均方误差）和 SGD（随机梯度下降），无动量、无权重衰减。风险指固定无噪声审计样本上的均方误差。一个 cell 是一个条件与数据随机种子 seed 的组合；一条轨迹是一次完整的小批量抽样更新序列。recipe 指一组科学条件，轨迹与 seed 不算独立 recipe。

- **降低学习率没有使全部八轨迹最低点进入两倍范围。** 比较固定标签噪声、B（每步批量样本数）=8/32、σ²（标签噪声方差）=0.25/0.5/1、seed411–414、M（轨迹数）=8。η（更新系数）从0.3降至0.1，ηT=230.4（T 为更新次数窗口）不变。同标签全批量最低点比值两档均为22/24落入[0.5,2]。18/24配对 cell 的偏离量 d（最低点比值与1的距离）减小。但所有配对变化的 bootstrap 区间（重抽已有轨迹得到的描述区间）均跨零。不能据此证明统一、单调或具有总体统计保证的收窄。
- **风险分母的口径影响最低点比值。** 在上述 η=0.1、M=8 网格中，对标签噪声取期望的分母使20/24落入[0.5,2]；固定同一次标签噪声的分母使22/24通过。前一口径同时含标签噪声实现差异，不能把旧位移全部归因于小批量。
- **增加轨迹后，两个已选反例的点估计回到两倍内。** 固定 η=0.1、B=8、σ²=1、seed411/412，只将 M=8 补到 M=64，同标签比值由2.142857/2.784314变为1.163265/0.767974。独立第二个64条流块（使用另一组小批量抽样序列）的比值为1.142857/1.284314。两块的95%条件区间上界均<2。有限轨迹抽样是旧反例的候选解释；两块通过本身不证明精确期望通过，不改判旧八轨迹失败。
- **对小批量抽样取条件期望后，两个已选条件的最低点在两倍内。** 固定上述两个数据与标签噪声，η=0.1、B=8、T=2304，最低点为50/315步，同标签全批量为49/306步，比值为1.020408163/1.029411765。这里的“精确”指没有 Monte Carlo（有限随机轨迹平均）的抽样误差，仍有 float64 数值误差。已知线性矩递推属于解析理论，不是本方向的新发现。
- **仅将小批量从 B=8 减至4，两个条件的期望最低点延后2/9步。** 在相同数据、标签噪声、η=0.1、σ²=1与窗口下，最低点从50/315变为52/324步，相对同标签全批量为1.061224490/1.058823529。注册范围51–70/316–400支持2/2。但均值与保存 B=8 结果逐位相等的注册核验两 seed 均失败，坐标差≤2.309264e−14。该测量保留 partial（部分完成）判定，不能声称全部核验通过。
- **较高学习率的两项普遍预测失败。** 在同一固定特征合同、η=0.3、M=8、σ²=0.25/0.5/1、四 seed 下，B=32的旧期望分母比值落入[0.8,1.2]的通过数依次为3/4、1/4、2/4，未达到每种方差至少3/4；B=8提前至少10%的通过数为0/4、1/4、2/4，未达到至少两种方差各3/4。旧失败与原始记录继续有效。

- **仅将小批量从 B=4 减至2，两个条件的期望最低点再延后3/19步。** 同上述固定数据与标签噪声、η=0.1、σ²=1及2304步窗口，最低点52/324→55/343步，相对同标签full为1.122448980/1.120915033。注册53–75/325–400步支持2/2，最低风险增加0.056821650/0.050625305。新固定C布局及事前容差核验通过；旧B4逐位核验失败仍保留。

## Formulation

风险比值区分固定标签与期望标签两个分母。固定特征线性头更新及风险定义为
\[
w_{t+1}=w_t-\frac{2\eta}{B}\sum_{i\in S_t}\phi_i(\phi_i^\top w_t-y_i-\sigma\epsilon_i),\qquad
\widehat R_B(t;\epsilon)=\frac1M\sum_{r=1}^{M}\frac1{N_a}\|\Phi_a w_{B,r,t}(\epsilon)-y_a\|^2,
\]
\[
R_{64}^{E}(t)=\mathbb E_{\epsilon'}R_{64}(t;\epsilon'),\qquad
t^*[R]=\min\arg\min_{t\in\{0,\ldots,T\}}R(t),\qquad
\rho_F=\frac{t^*[\widehat R_B(\cdot;\epsilon)]}{t^*[R_{64}(\cdot;\epsilon)]},\quad
\rho_E=\frac{t^*[\widehat R_B(\cdot;\epsilon)]}{t^*[R_{64}^{E}]},\quad d=|\rho_F-1|.
\]

w 是含常数特征的129维 head 参数，初值为零。φ_i 是128维冻结 ReLU 特征加常数项，φ_i、Φ、Φ_a无量纲。y_i 是 clean（无噪声）训练标签。ε_i 是保存的一次单位 Gaussian（高斯）标签噪声。σ² 是标签归一化单位下的噪声方差，取0.25/0.5/1。

S_t 是每步独立无放回均匀抽取的 B 个训练样本。轨迹测量 B=8/32/64，条件矩另测 B=4，并新增 B=2；B 的单位是样本数。η=0.3/0.1 是更新系数。Φ_a、y_a 是固定 clean audit（无噪声审计）特征与标签，N_a=2048是审计样本数。M 是小批量抽样轨迹数：广网格（已测的多条件组合） M=8；两个指定反例补足到 M=64；另测独立 r=64–127 的 M=64 流块，r 为轨迹编号；全批量 M=1。每个流块分别求均值，不合并旧新块。

R 的单位是归一化标签平方。t 和 T 的单位是参数更新次数，T=768/2304分别对应η=0.3/0.1，ηT=230.4。t* 是给定窗口内风险的首个全局最低点；min argmin 表示有多个同值最低点时取最早一次更新。R_{64} 指 B=64 的全批量风险；ε′表示对标签噪声积分使用的噪声变量，𝔼表示期望。ρ_F、ρ_E、d 无量纲。F 分母固定同一次标签噪声；E 分母对标签噪声积分，二者不同。这里 d 是比值偏离1的量；方法中的 d=4 表示输入维度。

固定 data（数据）/ε 时，独立 batch（小批量）的解析条件矩（参数均值和协方差）闭合为
\[
m_{t+1}=A m_t+a b,\qquad C_{t+1}=A C_t A^\top+a^2\gamma\left[\frac{q^\top\operatorname{diag}(\operatorname{diag}(qC_tq^\top)+(qm_t-z)^2)q}{n}-HC_tH-(Hm_t-b)(Hm_t-b)^\top\right],
\]
\[
R_B^{\mathrm{batch}}(t;\epsilon)=\frac{\|q_a m_t-y_a\|^2}{N_a}+\operatorname{tr}(H_aC_t),\qquad\rho_{\mathrm{batch}}=\frac{t^*[R_B^{\mathrm{batch}}(\cdot;\epsilon)]}{t^*[R_{64}(\cdot;\epsilon)]}.
\]

Φ 的各行是φ_iᵀ，n=64是训练样本数。V 是训练行空间（训练特征行向量张成的空间）的正交基，两个条件的数值秩均为36。q=ΦV、q_a=Φ_aV、z=y+σε、a=2η、H=qᵀq/n、b=qᵀz/n、A=I−aH、H_a=q_aᵀq_a/N_a。I 是单位矩阵，ᵀ表示转置。i 是训练样本索引；w_{B,r,t} 是批量 B、轨迹 r、更新 t 的参数。

m、C 是 Vᵀw 对 batch（小批量）抽样的均值和协方差，初值均为零。协方差描述参数围绕均值的变化；条件矩指固定数据与 ε 后这些均值和二阶变化。平方按元素计算，diag 提取对角或将向量组成对角矩阵，tr 表示矩阵对角线之和。

V、q、q_a、a、H、A、H_a 无量纲。m、b、z 的单位为归一化标签，C 与 R 的单位为归一化标签平方。γ=(n−B)/(B(n−1)) 与ρ_batch无量纲；γ是无放回抽样的有限总体因子。B=n时γ=0、C=0。任意 B 的 m 都遵循同标签 full（全批量）更新。该公式是已知线性矩递推，不是本方向的新发现。“精确”表示没有 Monte Carlo 抽样误差，实际数值计算仍有 float64 误差。

## 成立程度

同条件仅B4→2时，条件期望最低点为55/343步，相对full49/306的比值1.122448980/1.120915033；范围53–75/325–400支持2/2。独立raw二阶矩曲线差≤4.772849e−12、首个最低点一致。新B2与保存B4均值坐标及mean_risk（均值参数风险）的maxabs均为0，满足事前绝对容差1e−10/1e−12、相对容差0。只限这两个已选development条件，不证明连续batch单调律，也不改旧B4的partial。

条件期望实测边界：同固定 data（数据）/ε、η=0.1、B=8、σ²=1、seed411/412、T=2304 时，ρ_batch=50/49=1.020408163、315/306=1.029411765，均在 [0.5,2]。训练行空间秩36，独立样本坐标风险复算误差≤5.034862e−12、最低点一致，仅此两个已选 development（开发条件）。

同条件仅B8→4时，ρ_batch=52/49=1.061224490、324/306=1.058823529，相对B8最低点延后2/9步，注册范围51–70/316–400均支持。独立风险曲线实现差≤5.476731e−12、最低点一致，但B4均值与保存B8逐位相等（每个浮点位都相同）的注册核验失败（坐标差≤2.309264e−14），该测量按partial保留，未改判据。

实测边界：η=0.1、M=8 时，同标签比值 ρ_F 在 B=32 的 12/12 cell 为 0.624183–1.433803，在 B=8 的 10/12 cell 落入 [0.5,2]，总体 0.845070–2.784314。B=8、σ²=1 的两个反例 2.142857/2.784314 补足至 M=64 后变为 1.163265/0.767974，95% 条件 bootstrap 区间 [0.938776,1.489796]/[0.660131,1.415033]，同条件独立新流块的比值为 1.142857/1.284314，区间 [0.836735,1.836735]/[0.765278,1.866013]，上界均 <2。仅此两数据 seed 有两个独立 M=64 块，无统一有效学习率系数或连续 η 阈值。

在八轨迹均值口径下，降低学习率不能使全部风险最低点回到全批量的两倍范围。同标签条件下，B=32 在较低学习率全部通过，B=8 的高标签噪声条件仍有两个点估计越界。18/24 配对 cell 的 d 减小，每个 batch 各 9/12。但全部配对变化的抽样区间跨零（区间同时包含正负变化），两个 η 的通过总数均为 22/24。证据支持条件依赖的点估计变化，不能证明统一、单调或具有总体统计保证的收窄。残留 batch 抽样变化、标签噪声实现与平坦最低点均为待区分解释。没有单独识别离散稳定性机制。

期望标签噪声与固定标签噪声的最低点会产生差别。在 η=0.1 下，旧期望分母比值 ρ_E 仅 20/24 落入 [0.5,2]，B=32 为 0.272277–2.525483、B=8 为 0.519802–4.759777。同标签分母为 22/24。旧 η=0.3 的期望分母比值仍保留：B=32 总体 0.206–3.010（未截断精度 0.205882–3.010526），B=8 为 0.705–3.637（0.705128–3.636842）。这些旧位移同时含有风险口径差异，不能全部归因于 batch。

同标签分母补齐后，η=0.3 的 B=32 比值为 0.480392–1.976190，B=8 为 0.848797–3.750000。较低学习率两个反例的 batch-bootstrap 95% 描述区间为 [0.734694,5.653061] 和 [0.346405,3.062092]，都跨过阈值 2。因此注册的八轨迹均值曲线判据失败。

固定两个反例的全部数据、标签噪声、学习率和窗口，只把轨迹数补足到 64 后，最低点从 105/852 移至 57/235 步，同标签分母仍为 49/306。两个点估计均回到 [0.5,2]，两个描述区间都不跨 2，区间宽度分别降至旧值的 11.203320%/27.797834%。这支持有限 batch 抽样是旧反例的候选解释。仅凭补足轨迹结果，仍不证明精确 batch 期望的最低点在两倍内。只补足两个已选反例，不能与其他八轨迹结果合并成统一 24/24 保证。旧期望标签分母下新比值为 0.282178/1.312849，分母口径混杂仍存在。

独立的第二个 64 条 batch 流块仍使两个点估计在两倍内：最低点为 56/393 步，同标签比值 1.142857/1.284314，95% 条件区间 [0.836735,1.836735]/[0.765278,1.866013]，两上界均 <2。与第一块相比，比值差为 −0.020408/0.516340，分别独立重抽两块的差区间 [−0.408163,0.755102]/[−0.395507,1.140523] 均跨零。尤其第二个 seed 的最低点从 235 变为 393 步，说明 64 条轨迹仍留下抽样变化。两块通过不证明精确 batch 期望通过，也不证明两块相等。期望标签分母下第二块比值 0.277228/2.195531，仅作口径不同的描述。

## 方法与条件

方法保持既有数据合同和计数口径。复用 d=4、训练集含对称 Gaussian 输入、clean 目标 x₀+0.5x₀x₁、train mean/RMS 标签归一化（按训练标签均值居中、按均方根缩放）、train-centered/RMS（训练特征居中并按均方根缩放）的冻结 ReLU width128 特征、零初始化含 bias（常数偏置项）head 的 D2 数据合同。n=64、float64、MSE、SGD 无动量/weight decay（权重衰减）、B=8/32、σ²=0.25/0.5/1、seed411–414。每一步独立无放回抽样。不是整 epoch（遍历一次训练集） 重新洗牌。t* 取均值风险曲线的首个全局最小点，不取各轨迹最低点的均值，也不比较同样样本曝光量。

较高学习率的科学测量有 24 个小批量 cell/192 条轨迹，全部均值风险有限、最低点在窗口内。降低学习率后新增 48 个 cell：24 个小批量/192 条轨迹，12 个同 η 全批量对照，12 个高 η 全批量对照。复用 24 个旧小批量 cell。6 个 batch×variance recipe 为科学单位，4 seed 只测配对稳定性，8 条 batch 轨迹只测固定标签条件下的抽样误差。全部新风险有限且两种 full 风险最低点位于窗口内部。训练 10.779505 秒。

2000 次固定 seed 配对 trajectory-index bootstrap（按轨迹索引重抽），重新求均值曲线 argmin，再求比值和 d 的旧新差。区间仅包含固定数据/标签的 batch Monte Carlo，不是跨 recipe 的置信区间。原始 NPZ（保存数组的文件格式） 可复算全部数字。hash（内容校验值）、提交合同、独立风险/谱计算及成功 cell 的恢复检查通过。仅 development（已用于开发的条件），无新封存 OOD（事先封存、只查看一次的新条件）。不外推其他 n、优化器、learned features（训练中更新的特征）、连续 η 临界、无限训练或 train-only 早停（只用训练集选择停止时间）。

指定 B=8、σ²=1、η=0.1 的 seed411/412 各保留旧 8 条、追加 56 条：2 个补足 cell、112 新轨迹、16 保留轨迹、2 个复用 full 控制、0 新 full 训练，仅 1 个 recipe 和 2 个数据 seed。训练 5.121093 秒。241 个旧文件 hash/mtime（内容校验值/文件修改时间） 不变，新成功结果恢复时只核验/跳过。M=8 与 M=64 分别用 2000 次相同种子的索引重抽、按原方法求区间。因样本嵌套，未作为独立样本比较。注册的两倍点估计和“区间宽度 ≤旧值50%，落入 [0.5,3]/[0.5,2.5]”均 2/2 通过。bootstrap 比值 >2 的描述比例 0.0015/0 不作 p 值（假设检验使用的概率量）或精确期望概率。不重新抽 ε、不延长窗口、不丢旧轨迹。

独立 r=64–127 流块新增 2 cell/128 条轨迹，复用 r=0–63 两块的 128 条轨迹与两个 full 控制作同数据 seed 描述，0 新 full 训练。仍为 1 recipe、2 数据 seed。训练 5.783148 秒，258 个旧研究文件 hash/mtime 未变。P1（两倍范围预测）两点在 [0.5,2]、P2（区间上界预测）两个95%条件 bootstrap 上界严格 <2、P3（方法核验）有限/窗内/流不交集均通过。新旧块分别独立重抽，不把 replicate（轨迹重复）下标当共享 batch 流。比值 >2 的重抽比例 0.003/0.015 不作 p 值。独立 einsum（按索引求和的独立计算） 风险与前16步重放最大误差 ≤1.110223e−16，bootstrap 独立复算一致。恢复仅核验/跳过，9 个数据/结果/manifest（文件清单） 文件 hash/mtime 不变。

条件矩测量计算2个精确cell、0新训练，复用2个同标签 full 控制和4个M64块共256旧轨迹。1 recipe、2已选数据seed，计算2.443217秒。训练行空间投影误差≤1.554312234e−15。独立64维样本坐标（直接在训练样本张成的坐标中计算）用原始二阶矩和单/双 inclusion（样本被抽入批次的）概率复算，曲线差≤5.034861417e−12、最终均值head差≤6.955547249e−14，两个首个最低点一致。这些实现差不是严格数值误差证书。B=n协方差严格为零，与保存full曲线差≤1.526556659e−14。n4/B2枚举6batch（n=4、B=2的全部批次）的一步均值误差0、协方差误差6.938893904e−18。全部数组有限、协方差最小特征值非负，285个旧文件hash/mtime不变。恢复仅核验并跳过2个cell，19个新合同/结果文件hash/mtime不变。四个MC（Monte Carlo）块分别比较条件期望曲线，RMSE（曲线差的均方根） 0.004135192–0.005280278、maxabs（最大绝对差） 0.013354175–0.024000393。不合并、不给精确期望加抽样区间。两个条件期望预测依据来自已看过的development M64证据，不是盲发现或OOD验证。恢复仅核验/跳过两个成功cell，19个已有本轮文件hash/mtime不变。独立实现差不作为严格误差证书。新执行/核验无运行警告，预注册前行空间检查的matmul（矩阵乘法）警告及其历史根因未定仍保留。

仅B8→4新增2条件矩cell、0训练，读取2个B8精确对照及2个full。1新batch recipe、2配对data seed，计算1.627101秒。保存basis（正交基）/full逐元素继承，未重做旧SVD（奇异值分解）/矩/full。P1注册51–70/316–400步支持2/2，但注册均值逐位相等失败，原summary为partial。独立64维原始二阶矩风险差5.476730180e−12/2.006728117e−12、最终head差≤6.813993814e−14、首个最低点52/324一致。数组有限、窗内、协方差最小特征值非负，full对保存控制差≤1.526556659e−14。新B4最近邻余量5.913242543e−6/1.017070178e−8。302旧文件与18本轮恢复文件hash/mtime不变，恢复0新cell，全局扫描拒绝孤立/额外/混合pin（冻结的源码或输入校验值）。原失败不放宽，新执行与核验无运行警告。

固定这两个数据和标签噪声后，解析条件 batch 期望的最低点为50/315步，分别比同标签 full 的49/306步延后1步和9步，两比值1.020408163/1.029411765。两倍内预测在两个选定条件成立。平均参数恰遵循 full 更新，audit 风险的新增协方差项在各最低点为 0.022759234690/0.019800149753，对应期望风险 0.250249570939/0.268837195946。两个64轨迹块相对精确最低点，seed411 晚7/6步，seed412 早80/晚78步。后者最近邻风险差（最低点与相邻一步风险的差）仅2.579671404e−8。由此可区分这两个条件中1步和9步的 batch 期望位移与有限轨迹 最低点变化，但不扩大为其他数据、标签噪声或 batch 的保证，也不追溯改判八轨迹判据。

保持这两个固定数据与标签噪声，仅将batch减半至4，条件期望最低点从50/315步移至52/324步，相对同标签full为1.061224490/1.058823529。最低风险增加0.026908879357/0.023591985580，达到0.277158450296/0.292429181526，其中协方差风险为0.049492000902/0.043376178400。参数均值递推在解析上不随batch改变，风险差通过协方差项进入。有限总体因子 γ 从1/9增至5/21，协方差反馈（当前协方差对下一步的影响）也随之改变，不能把风险或最低时间按15/7直接缩放。独立原始二阶矩复算确认两个首个最低点。但对旧保存均值的逐位核验未通过，只保留明确披露核验限制的离散配对测量，不给连续batch保证。

仅B4→2新增2条件矩cell、0训练，计算1.744321秒，读取2个B4精确对照与2个full。1新batch recipe、2配对data seed；basis、q和qa事前固定float64 C连续，未重做旧成功矩。最低期望风险0.333980100133/0.343054486608，其中协方差项0.105835131584/0.093934691234；最近邻余量5.426518285e−6/5.596599068e−8。独立64维样本坐标风险差4.772848783e−12/2.434441537e−12、最终均值head差≤6.813994e−14，argmin一致。γ从5/21增至31/63，比例31/15不构成最低点缩放公式。332旧文件集合/hash/mtime与18本轮恢复文件均不变，恢复新增0cell。summary和verification各写一次；冻结分析/核验源码自身无写一次保护，恢复编排不重调二者。新日志无运行警告。

收尾工具的正则错误及Git目录可见性失败均保存。Git元数据目录已改名，显式指定该目录后原冻结校验与科学收尾审计通过；冻结源码及实验结果未改。目录改名原因未确定。

### 失败与反例

- η=0.3 时，“B=32 在每种 σ² 至少 3/4 seed 的旧比值落入 [0.8,1.2]”失败，通过数依次为 3/4、1/4、2/4。对应比值范围 0.803–1.390、0.949–3.011、0.206–1.514。
- η=0.3 时，“至少两种 σ² 的 B=8 至少 3/4 seed 提前 10%”失败，通过数为 0/4、1/4、2/4。对应范围 1.074–1.266、0.796–3.637、0.705–2.083。
- η=0.1 时，“全部 24 个比值进入 [0.5,2]”在旧期望分母下 20/24、同标签分母下 22/24，均失败。同标签两个反例均为 B=8、σ²=1：seed411 为 105/49=2.142857，seed412 为 852/306=2.784314。后者的 d 相对 η=0.3 增加 1.558824。
- 上述八轨迹失败保留。两个反例补足至 64 轨迹后不再越界，说明其点估计对 batch Monte Carlo 样本量敏感。不能用补足后的结果追溯改判八轨迹预测。
- 首版错误使用 η 更新而非实际 2η，其 24 个 cell 保留为失败记录，不用于科学结论。旧核验源码和 pin（冻结校验值） 曾在训练后修改，历史训练 metadata（训练记录元信息） 的旧 pin 保留。不能声称旧核验源码自训练前起未变。
- 原 matmul（矩阵乘法） 发出 divide/overflow/invalid（除法、溢出或无效运算） 警告。保存数组全部有限，独立 einsum/谱复算一致至 3.108624e−15，根因未定，原记录保留。
- 补足轨迹和独立流块时同样出现 matmul 警告，保存数组有限，独立风险与首 16 步重放误差 ≤1.110223e−16。警告根因仍未识别。旧收尾回执链接实际缺失，保存的旧输入已与真实科学收尾字节核对。
- B4要求均值坐标及mean_risk（均值参数对应的风险）与保存B8逐位相等的注册方法核验在两seed均失败，坐标差2.309263891e−14/8.743006319e−15、风险差3.885780586e−16/4.440892099e−16。原分析报错与validation=false（核验未通过）保留，整轮为partial，未改容差或成功cell。basis值相等但旧Fortran连续（按列存储）/新C连续（按行存储），布局引起舍入路径变化仅为候选解释，未干预确证。

## 未决问题

上一个仅改变B4→2的候选已完成。下一小问题优先0训练，仅B2→1，保持原data/ε、η=0.1、σ²=1、seed411/412、T=2304及full分母49/306。先注册两个新最低点范围、固定布局/容差、pinned源码与输入hash并一次提交，再计算；先补summary/verification写一次或核验跳过保护与旧文件完整集合扫描。不重算任何旧成功B2/B4/B8/full，不预先算B1风险后称预测；无连续B/η或其他条件外推。

下一候选问题仅改变 B4→2；以下是原有研究计划，不在此次修复中执行。优先保持同data/ε、η=0.1、σ²=1、seed411/412、T=2304与同标签full分母49/306，仅B4→2，先注册两个新精确最低点数值范围、输入hash与pinned（冻结校验值绑定的）源码并一次commit（git提交）后计算。均值比较应事前固定布局和浮点容差，现有逐位失败继续保留。不重算成功B4/B8/full，不预先计算新曲线、不推连续B/η或其他数据。

## 证据

- [高学习率汇总](studies/r025_batch_risk_shift_corrected/summary.json)、[原 findings](findings/r025_batch_risk_shift.md)、[错误更新记录](studies/r025_batch_risk_shift/summary.json)；冻结 0f34071，旧收尾 44d7b5a。
- [低学习率预注册](studies/r029_lower_eta_paired/preregistration.json)、[执行源码](studies/r029_lower_eta_paired/executed/run.py)、[分析源码](studies/r029_lower_eta_paired/analysis.py)、[结果汇总](studies/r029_lower_eta_paired/summary.json)、[findings](findings/r029_lower_eta_paired.md)；冻结 5e0f67c。
- [结果/hash 核验](studies/r029_lower_eta_paired/executed/verification.json)、[独立 einsum/谱/区间复算](studies/r029_lower_eta_paired/executed/independent_verification.json)、[只读旧输入快照](studies/r029_lower_eta_paired/executed/input_audit.json)、[恢复核验](studies/r029_lower_eta_paired/executed/resume_verification.json)；旧科学收尾 ea1a3e8，原最终回执缺失，见 [真实收尾字节审计](studies/r033_batch_mc_extension/executed/input_audit.json)。
- [64 轨迹预注册](studies/r033_batch_mc_extension/preregistration.json)、[执行源码](studies/r033_batch_mc_extension/executed/run.py)、[分析源码](studies/r033_batch_mc_extension/analysis.py)、[结果汇总](studies/r033_batch_mc_extension/summary.json)、[findings](findings/r033_batch_mc_extension.md)；冻结 b046352。
- [独立核验](studies/r033_batch_mc_extension/executed/verification.json)、[恢复核验](studies/r033_batch_mc_extension/executed/resume_verification.json)、[bootstrap 原始结果](studies/r033_batch_mc_extension/results/bootstrap.npz)、[最终提交核验](studies/r033_batch_mc_extension/executed/final_commit_verification.json)。

- [独立流块预注册](studies/r041_independent_batch_block/preregistration.json)、[执行源码](studies/r041_independent_batch_block/executed/run.py)、[分析源码](studies/r041_independent_batch_block/analysis.py)、[汇总](studies/r041_independent_batch_block/summary.json)、[findings](findings/r041_independent_batch_block.md)；冻结94d66e1。
- [上一轮收尾与旧输入审计](studies/r041_independent_batch_block/executed/input_audit.json)、[独立核验](studies/r041_independent_batch_block/executed/verification.json)、[恢复核验](studies/r041_independent_batch_block/executed/resume_verification.json)、[bootstrap数组](studies/r041_independent_batch_block/results/bootstrap.npz)、[最终提交核验](studies/r041_independent_batch_block/executed/final_commit_verification.json)。

- [条件期望预注册](studies/r074_exact_batch_expectation/preregistration.json)、[可行性审查](studies/r074_exact_batch_expectation/executed/feasibility.json)、[旧收尾与输入审计](studies/r074_exact_batch_expectation/executed/input_audit.json)；冻结 d8943a9。
- [执行源码](studies/r074_exact_batch_expectation/executed/run.py)、[分析源码](studies/r074_exact_batch_expectation/analysis.py)、[独立核验源码](studies/r074_exact_batch_expectation/executed/verify.py)、[汇总](studies/r074_exact_batch_expectation/summary.json)、[独立核验](studies/r074_exact_batch_expectation/executed/verification.json)、[恢复核验](studies/r074_exact_batch_expectation/executed/resume_verification.json)、[findings](findings/r074_exact_batch_expectation.md)。

- [条件期望恢复核验](studies/r074_exact_batch_expectation/executed/resume_verification.json)、[最终提交核验](studies/r074_exact_batch_expectation/executed/final_commit_verification.json)。

- [B4预注册](studies/r085_batch4_exact/preregistration.json)、[执行源码](studies/r085_batch4_exact/executed/run.py)、[原分析源码](studies/r085_batch4_exact/analysis.py)、[原汇总](studies/r085_batch4_exact/summary.json)、[findings](findings/r085_batch4_exact.md)；唯一冻结40d29b1。
- [旧输入与收尾审计](studies/r085_batch4_exact/executed/input_audit.json)、[保留的原分析失败](studies/r085_batch4_exact/executed/analysis.log)、[独立核验](studies/r085_batch4_exact/executed/verification.json)、[恢复核验](studies/r085_batch4_exact/executed/resume_verification.json)、[收尾审计](studies/r085_batch4_exact/executed/closeout_audit.json)、[最终提交核验](studies/r085_batch4_exact/executed/final_commit_verification.json)。

- [B2预注册](studies/r097_batch2_exact/preregistration.json)、[执行源码](studies/r097_batch2_exact/executed/run.py)、[分析](studies/r097_batch2_exact/analysis.py)、[汇总](studies/r097_batch2_exact/summary.json)、[findings](findings/r097_batch2_exact.md)；唯一冻结94beabf，报告六节重排d644e3b。
- [旧输入/收尾审计](studies/r097_batch2_exact/executed/input_audit.json)、[独立核验](studies/r097_batch2_exact/executed/verification.json)、[恢复核验](studies/r097_batch2_exact/executed/resume_verification.json)、[收尾审计](studies/r097_batch2_exact/executed/closeout_audit.json)、[最终提交核验](studies/r097_batch2_exact/executed/final_commit_verification.json)。

- [B2独立只读审查](studies/r097_batch2_exact/executed/independent_review.json)、[收尾环境事件](studies/r097_batch2_exact/executed/environment_incident.json)、[显式Git目录通过记录](studies/r097_batch2_exact/executed/closeout_explicit_git.log)。
