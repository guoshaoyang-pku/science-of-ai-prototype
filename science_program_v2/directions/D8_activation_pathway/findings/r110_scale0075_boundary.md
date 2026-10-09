# D8：a=0.075 的两项 2% 判据仍失败

## 结论

固定 data31001 对称 Gaussian d4/n128、width64 冻结单层 SiLU、共同加性 bias、训练列居中、root 与 seed101–106，仅新增 λ=±3、a=0.075 的 12 个初始化 cell。没有训练、标签或 test。解析 F 误差为 0.025956782989–0.029189276970，三个保存小尺度加新尺度的 spread 为 0.025898649401–0.028362810341。原两项 2% 判据均 0/12 通过；本轮 P1/P2 各 12/12 supported。

## 预测与配对证据

预注册点只用旧 a=0.05 的 signed(Q/F−1) 乘 2.25；已知 a=0.1 结果，属于非盲 development。每 seed 误差与 spread 范围均为点±0.002，不是置信区间。两侧 signed 误差分别为 +0.028577815797–+0.029189276970、−0.027422681666–−0.025956782989。新/旧 a=0.05 的 Q 比分别为 1.015699751579–1.016073982185、0.984675182961–0.985496235173。

相对保存三尺度 spread 的同 seed 增量，两侧均值为 0.016101492208、0.015269526034；df5 的95%初始化 t 区间为 [0.015957798247,0.016245186168]、[0.014941249697,0.015597802371]。这些区间只描述固定数据下权重初始化，不按 cell 数推总体概率。新 Q 相对非盲点预期的差为 0.000048661925–0.000227437662，仅描述。

## 执行与审计

报告开头重排单独提交 7fac38d，按 inbox 六节组织，旧数字与结论保留，指定处理标记已追加。新激活前，两次文本编辑命令 SyntaxError、一轮编排语法失败及一次报告正则转义错误，均未计算科学量；修正后旧数字/六节/证据位置检查通过。测量后，文本自查把长小数误识别为提交号，修正自查表达式后通过；范围分隔符全局替换误改相对证据链接，链接检查发现后恢复 ../ 路径。两处后置文本错误没有改科学量。旧报告重排提交先于新实验，不改科学量。

先核查 r092 收尾49个提交对象、896旧文件及24结果文件 hash/mtime、恢复new0/reused12及独立验证；没有运行旧恢复脚本。旧 r048 18次重叠导致90cell协议失效和0claim保留。重新核查五份历史Cartesian、426个保存request，12新请求重叠0；940旧study文件集合/hash/mtime锁定。旧36对照仅读原数组、原request与contract/hash，旧激活调用0。

唯一预注册51f9e8989d004d7ec4b6f34c2f10fc44e60b2124绑定11份pinned源码/输入/审计，早于全部新激活至少7.572792秒（提交时间为秒级）。先limit1，首cell独立核验pass后复用它新增11cell；累计测量0.060539250秒，0训练。冻结run没有强制首cell回执门槛；实际执行次序由时间戳证明。冻结verify只核验结果mtime；补充审计直接核查activation_started_time_ns，不改冻结源码。

独立激活重建最大差3.108624e−15；longdouble导数与矩F相对差8.015810e−14；fsum从新旧保存数组复算R/Q差4.440892e−16。恢复new0/reused12；24结果文件和首cellhash/mtime、940旧文件均不变。成功cell未覆盖，final summary及独立回执各只写一次。

## 边界与交接

普通bias a²、精确根a⁶、近根混合与正负不对称、小尺度λ±3通过2%、a=.1失败均保留。matmul根因未定、首cell打印、pooled CI纠正和d00误用记录保留。本轮两项新预测均支持，不是旧三个小尺度或解析极限被反驳。没有连续临界尺度、跨seed概率、目标核、训练或干预收益、非对称输入、多层、CE、小批量或OOD外推。

下一候选仅development，保持全部合同仅新增尚未测a=.0625/λ±3的12cell，检验两项原2%判据是否均重新通过。先核查历史、冻结明确逐seed预测和hash；不得预览新激活。新run须强制首cell审计门槛，verify须直接核查activation_started_time_ns。任何旧重叠只读原数组/合同/hash，不重新激活。

## 证据

- [预注册](../studies/r110_scale0075_boundary/preregistration.json)、[数值预测](../studies/r110_scale0075_boundary/executed/numeric_forecasts.json)、[只读对照](../studies/r110_scale0075_boundary/executed/readonly_controls.json)、[summary](../studies/r110_scale0075_boundary/summary.json)。
- [旧收尾](../studies/r110_scale0075_boundary/executed/prior_closeout_audit.json)、[历史条件](../studies/r110_scale0075_boundary/executed/condition_audit.json)、[首cell](../studies/r110_scale0075_boundary/executed/first_cell_audit.json)、[独立恢复](../studies/r110_scale0075_boundary/executed/independent_verification.json)、[开始时间](../studies/r110_scale0075_boundary/executed/execution_audit.json)、[独立审查](../studies/r110_scale0075_boundary/executed/independent_review.json)、[最终提交](../studies/r110_scale0075_boundary/executed/final_commit_verification.json)。
