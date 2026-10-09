# r089：加入 LN affine 核项仍未恢复固定核方向

## 问题与预测

在 r077 的共享 Linear 初始化核中，只加入原训练可更新的 LN affine 初始化 Jacobian 核项，固定核 256 步预测是否能恢复真实 train offset chord 的 LN/noLN 方向？development 预注册 P1 预测 20 个同函数同 seed 配对中至少 16 个同号。

## 结果

P1 refuted：共享Linear+LN affine固定核与真实 train 差值方向匹配 0/20，低于预注册的16/20门槛。预注册登记了方向匹配门槛，但没有登记效应量范围或预期0/20；下列范围是测量结果。真实 train 的 LN−noLN $\Delta C$ 在 20/20 配对均为负，范围 −0.288059215 至 −0.027001544；加入 affine 项后的固定核差值在 20/20 均为正，范围 +0.014690203 至 +0.043216926。最大 $\eta\lambda_{\max}=0.001330483$，低于稳定门槛 1。20/20结果完成，累计1.491398秒，0新训练；复用120个训练cell与保存的旧核。执行记录显示，首次结果在恢复调用前已保存，恢复调用新增19个结果。

## 方法与核验

只新增 LN010 第二个 hidden block（模块名net.3.norm）的 LayerNorm scale/bias 两组参数，共128个坐标。逐样本 Jacobian 由 float32 模型/autograd 计算并转 float64，$K_{\rm aff}=J_{\rm aff}J_{\rm aff}^{\top}/256$，再与共享 Linear 核相加；传播及核比较为 float64。四个development函数各5个 seed，复用 r077 保存的 train 标签、预测、Linear 核及 noLN 基线。

独立核验通过20个结果文件的hash与有限性、预注册时间门禁、旧核谱系及kernel pin、affine/total kernel重构与对称性、独立eigh传播、随机VJP和初始预测。最大核重构绝对误差为0，传播向量最大误差4.662937e−15，VJP最大相对误差8.439118e−7。冻结 `verify.py` 读取NPZ中不存在的 `kernel_pin` 字段，因而不能完成其原定核pin检查；首cell后排查时曾临时改动verify.py并删除freeze_inputs.py，续测前均已恢复预注册提交字节；全部20cell的源码门禁与合同绑定同一预注册提交。另写独立核验程序从数组重构核并核对记录pin，核心核与导数检查全部通过。

## 边界与下一步

本结果只表明：这四个development函数、width64 recipe与eta=.001/a=3/T=256下，仅补入LN affine初始化Jacobian仍不足以修正固定核方向。它不能把冲突归因于核漂移、小批量、weight decay或test动力学；这里是固定J、全批量、无decay的train线性化，不是实际训练。20配对只含四函数，seed只表示初始化变化，不外推独立函数、其他width/优化器或sealed OOD。

下一轮可在共享Linear+LN affine初始化核保持不变的条件下，单独把全批量递推改为batch64顺序递推，检查方向匹配是否变化；先核对旧批次顺序能否重建，不能则声明新的确定顺序。须预注册批次顺序、算子与旧核/train结果hash，不新增训练。

证据：`studies/r089_ln_affine_kernel/preregistration.json`、`summary.json`、`executed/receipt.json`、`executed/saved_evidence_verification.json`、`executed/independent_verify.py`。
