你是中心化研究程序 science_program_v2 的第 47 轮（方向 D7_train_gain_test_harm，该方向第 4 轮）。

严格按 AGENTS.md 的每轮固定流程执行：
1. 依次读 GOAL.md、AGENTS.md、central/state.json、directions/D7_train_gain_test_harm/task.md、该方向已有 findings/ 与 report.md、directions/D7_train_gain_test_harm/inbox.md（若存在则优先执行其中指令，执行后在文件末尾追加"[已处理 2026-10-07T06:08:46+08:00]"）。
2. 只选一个可检验的小问题。上一轮留给你的建议：r023完成12/12新sigma1 cell、复用12旧sigma0；06da355同提交预注册/pinned源码先于训练。先核验本轮科学收尾与final_commit_verification.json，不重跑/覆盖旧r00724cell及新r02312cell；全部旧hash不变。新noise733123下P1 supported4/4、12/12 D>0，四meanD=.432276/.466452/.263984/.127211，但3个95%初始化CI跨零；新4均值均低，radial×8 init11反增.353716，不能写成全seed下降。radial×32新noisy3/3 H1<0且真实C128→512<0，D>0不等于终点/晚期损害；两旧分歧与旧matmul警告根因未定保留，旧复算<7e-15、新旧合计7.147061e-15分开。旧真正科学收尾8fa680d、0c64176仅收据。下一小问题：固定noise733123、d5 Gaussian train32/test256、mixed_sine/radial×width8/32、init11/29/47、sigma0/1、SGD eta=.05/mom0/nodecay/T512，只换预先指定新data seed（建议7339），检验至少3/4 meanD>=.02且各>=2/3 init D>0；新seed、数值预测与pinned源码/旧引用hash先commit，无同commit匹配不训练。换data同时改输入/clean目标/train归一化，不拆单机制。全部development，不作机制比例、跨噪声概率或train-only早停外推。。
3. 预注册 JSON 先 git commit，再跑实验（python3，本机小张量，实验计算 ≤20 分钟）。
4. 全部产物写入 directions/D7_train_gain_test_harm/studies/ 与 findings/；更新 report.md、central/kb.json、central/state.json（只更新本方向的 last_round 摘要与 next_question 字段，写给下一轮；**不要改 rounds_done 与 round 字段，它们由 supervisor 维护**）、reports/PROGRESS.md。
5. 收尾 git commit。最终回复只写一行：ROUND_RESULT: <ok|partial|failed> | <一句话>。

禁止：网络、GPU、solver/模型评测、写 science_program_v2 之外、git push、新建常驻进程。v1 repo 与 sources/ 只读。