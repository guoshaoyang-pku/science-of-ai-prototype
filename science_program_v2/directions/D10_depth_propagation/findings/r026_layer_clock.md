# 第26轮：最慢参数层能否描述真实早期拟合时间

按序读取GOAL、AGENTS、中心状态和方向task；本方向此前仅有task，无findings/report/inbox，无待追加处理标记。选择一个development问题：同宽度深度1/2/4/6下，初始化单参数组冻结切线时间的最大值，与初始化全核冻结时间，哪个能在两倍内描述实际半损失步数。

## 执行与证据

- 预注册与run.py、analysis.py、manifest在同提交ed6cd26e3adb9993db40f5b5acb3e64f1ce7ab09冻结，2026-10-07T00:33:12+08:00提交。源码门禁先核对祖先提交的完整字节，再开始训练。采用定向git add保留supervisor已有脏日志；科学预注册及源码均已提交且先于全部训练。
- 使用指定Python，CPU小张量单线程，无网络/API/GPU。12/12 cell完成，4个深度recipe、3个初始化仅为重复；实验累计5.345299秒。
- 条件为n32/d3/width16/SiLU/no bias/no LN/no residual、half-MSE、全批量SGD eta.02/mom0/nodecay、T4096、固定输入target，同seed跨深度逐位共享hidden前缀与head。
- `studies/r026_layer_clock/results/`保存输入、标签、初始与终点参数、4097个每步loss及8个checkpoint的每组J/K/谱/残差能量。成功cell不覆盖；重复调用逐个核对合同与NPZ SHA256后跳过12个，0新训练。

## 结果

P1 refuted：最慢组预测12/12失败，全部深度0/3两倍内；比值4.727273–2053.176796。P2 supported：全核10/12两倍内，深度1/2为3/3、4/6为2/3，满足全部4深度各>=2/3。失败是科学结果，不是执行失败。两个全核反例为depth4_seed1129的18/54=.333333和depth6_seed1129的601/181=3.320442。实际时间范围1–181步，没有4096步删失或无限候选。

同seed相邻深度实际差：1103为14/−6/11；1129为−7/50/127；1151为−6/−3/5。最慢组在浅层均为input，在depth4_seed1129为最后hidden，在depth6_seed1103/1129为head。head在两候选中均计入，不能把该候选称为隐藏传播瓶颈。

已知恒等式K=sum K_l不作为新claim。新证据是同cell条件下最慢组候选的系统性失配，以及全核候选按注册多数判据通过但仍有反例。两候选无拟合系数，无sealed OOD；深度变化同时改变参数量和初始预测，不能独立归因深度、核变化或残差方向变化。

## 复算及限制

`executed/execution_audit.json`核验匹配提交、字节合同、时间顺序和恢复；`executed/verify.py`从原始参数解析构造SiLU Jacobian并重建half-MSE首步，独立矩阵平方递推验证所有单组与全核的整数半衰时间。12NPZ/JSON、96checkpoint、所有数组finite；J最大误差4.44e−15，首步预测2.22e−15，核相加误差0，谱重建2.31e−14，冻结过线占比误差7.71e−12。

冻结analysis.py缺少eta*lambda_max>=2处理。独立审查所有初始块及全核，最大1.3648257405<2，故本轮没有分支遗漏的影响；保留源码，下一轮修正后重注册。首版独立核验`verify_matrix_power.py`调用NumPy matrix_power出现divide/overflow/invalid matmul警告，输出仍finite且步数匹配。保留原版与`verification_warning.json`；改为显式einsum平方递推的verify.py无警告并一致，根因未定，不宣称已修复底层库。

报告按六节骨架新建。只更新中心本方向last_round与next_question，round与rounds_done交由supervisor。登记两条有保存证据的KB，追加PROGRESS。下一小问题建议仅降低eta到.005，先明确量化预测并commit，现有12cell不得覆盖。

科学收尾提交 429d47589e8ac050cb5d3e0eeb85808accba099a 已核对全部科学产物字节；`executed/final_commit_verification.json`保存hash凭据。最终git add -A同时快照本轮开始时已存在的supervisor日志，研究者未编辑其内容。
