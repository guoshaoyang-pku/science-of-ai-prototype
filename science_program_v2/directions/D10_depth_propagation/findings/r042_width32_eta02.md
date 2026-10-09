# 第42轮：width32 的学习率四倍配对

本轮只检验一个 development 问题：固定 n32/d3、data_seed260071 的输入/标签、width32、SiLU 无 bias/LN/residual、depth1/2/4/6、init_seed1103/1129/1151、全部初始参数与规则、full-batch half-MSE SGD、momentum0、weight_decay0、T4096，仅将 eta 从 .005 升至 .02，全核12/12两倍内是否保留。

## 执行与预测

按规定读完入口、三份 findings、report，无 inbox；现有 report 六节结构合格。训练前审计旧109个study文件，核验三份历史回执的39/37/37个git blob hash；旧36个cell数组hash一致。旧width32回执快照5295611、实际结果与交接6de95da、回执初存5b87c5d的历史关系记录在报告证据节，旧文件未改。

预注册、训练/分析/核验源码、manifest、旧证据快照均冻结于63aa9eab58c352223bd77f6a1f47a5af53d9542c；提交Unix时间1791321505，最早训练1791321517.968977，逐字节门禁通过且严格先于训练。git add -A同时保存进入本轮时由supervisor修改的日志与state，研究者未改其内容。指定Python、CPU单线程小张量，12/12 complete，训练累计7.254602415秒；4个深度recipe×3个初始化重复，不计成12独立recipe。

P1 supported：最慢参数组12/12有限比值>2，范围16.6–73；四深度各3/3。P2 supported：预注册的更严格全核12/12在[.5,2]判据通过，四深度各3/3，范围.75–1。两者同时成立不是最慢组候选成立；其两倍内拟合仍失败。

| depth | 实际T50，seed1103/1129/1151 | 最慢组候选 | 全核候选 |
|---|---|---|---|
| 1 | 3/4/5 | 110/69/83 | 3/4/5 |
| 2 | 4/3/5 | 96/85/192 | 4/3/4 |
| 4 | 7/15/6 | 313/391/218 | 6/14/5 |
| 6 | 4/15/1 | 242/611/73 | 3/15/1 |

## 配对与核验

按表顺序新−旧实际步数为−9/−11/−13、−11/−8/−12、−20/−44/−18、−12/−42/−5；新/旧比值.166666667–.294117647，12/12缩短。depth6_seed1151的6→1步有整数过线影响，不推连续时间6倍速率律。

输入/标签、归一化常数、参数形状、初始参数、预测与全部初始化J/K/谱/残差能量逐位匹配。最大初始eta·lambda_max=.7713375275328761<2，无不稳定/无限/删失/发散。注册前已知该值可从旧development谱乘4得到，不登记盲预测发现。

冻结 verify.py 用 endswith('_0')筛选初始化，误含J_1_0（第1步、第0组）而断言失败。保留原脚本、失败日志和verification_error.json；只读verify_corrected.py仅改为checkpoint0显式筛选，不改冻结run.py/analysis.py，不新训练。修正核验通过12cell/96checkpoint：Jacobian误差5.329071e−15、首步2.664535e−15、谱重建1.687539e−14、冻结过线占比4.396483e−14；核重建/相加误差0。显式einsum矩阵平方递推与冻结谱整数时间一致。

恢复调用核对合同与数组hash后跳过12/12，0新训练，24新结果文件hash/mtime不变，109旧study文件hash/mtime不变。冻结分析从保存结果复算summary逐字节一致。新分析明确发散cell的comparison为unresolved，不稳定候选记失败且不能支持有限高估；本轮分支未触发，旧不稳定谱合同差异和matmul警告保留。

## 边界与交接

仅这两个eta、当前数据/目标、四已测深度与三个初始化的development。未独立干预核漂移或残差方向，不作宽度机制、深度单调、因果、测试/泛化、其他优化器或OOD外推。旧width16全核反例、跨宽度两个变慢cell及历史警告都保留。

下一小问题仅将width32 eta=.02升至.04，其余逐位固定；先写新数值预测与pinned源码commit，使用修正核验筛选后才训练。新最大初始eta·lambda_max约1.542675来自旧谱算术，须披露。中心仅改D10 last_round与next_question，不改round/rounds_done。

## 证据

- [预注册](../studies/r042_width32_eta02/preregistration.json)、[源码manifest](../studies/r042_width32_eta02/executed/manifest.json)、[逐值summary](../studies/r042_width32_eta02/summary.json)。
- [旧证据快照](../studies/r042_width32_eta02/executed/baseline_audit.json)、[独立核验](../studies/r042_width32_eta02/executed/verification.json)、[冻结核验失败及修正记录](../studies/r042_width32_eta02/executed/verification_error.json)、[执行/恢复审计](../studies/r042_width32_eta02/executed/execution_audit.json)。
- [收尾核验回执](../studies/r042_width32_eta02/executed/final_commit_verification.json)核对全部科学产物与中心交接文件字节；所有已冻结源码与成功cell保留。
