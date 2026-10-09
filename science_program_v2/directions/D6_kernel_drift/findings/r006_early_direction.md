# r006：早期核漂移方向，未进入训练

本轮只问：在固定初始残差、匹配核总尺度后，前 32 步的漂移方向是否保留到第 256 步。预注册及自包含源码已保存；预注册提交失败，未启动训练。保存 0/12 cell、0/4 完整 recipe 单元，P1 为 `not_evaluated`，没有新科学 claim。

## 预注册条件与判据

开发条件为 product 和 mixed_sine，d=3、n=32、Gaussian data-seed=260606；width=8/32，每个单元初始化 seed=611/629/647。模型为含 bias 的两层 hidden SiLU 网络；full-batch halfMSE、float64、SGD η=.05、momentum=0、weight_decay=0，计划训练 256 步。checkpoint 为 0/1/8/32/64/128/256。

核定义为 `K_t=J_t J_tᵀ/n`。固定 `r0=f0-y`，用 `a_t=trace(K_t)/trace(K0)` 匹配尺度，主指标为 `rho_t=R(r0,K_t)/R(r0,a_t K0)`；Rayleigh 商 `R(v,K)=vᵀKv/(vᵀv)` 描述核在给定方向上的作用大小。这样避免将残差旋转或整体放大误当成方向漂移。

P1 预测：step32 和 step256 的 `log(rho)` 同号，且绝对值均大于 `log(1.05)`；每个函数×宽度单元至少 2/3 seed 命中，至少 3/4 单元通过。无方向读数不算命中。只在 12 个 cell 全部保存并核验后判定，不删 seed、不改阈值。seed 只测稳健性，不能把它们当成 12 个独立 recipe。

B03 的核交换 probe 作为诊断：各 checkpoint 从相同 `rt=f_t-y` 出发，仅替换 K0、a_t K0、K_t，使用同 cell 全 checkpoint 共用的稳定步长。保存固定 r0、固定 y、移动 rt 的读数及 32 个残差置换。probe 不等价于非线性续训，也不新增预测。

## 执行证据与预测状态

2026-10-06T18:12:22+08:00，`git add -A` 返回 128：无法创建本 repo 的 `.git/index.lock`，`Operation not permitted`。未执行预注册 commit，因而不运行训练。当前环境将 `.git` 列为只读，且不允许权限升级；本轮未尝试绕过该边界。

`analysis.py` 已用指定 Python 运行。它核对 pinned source，复算当前保存 cell 数量为 0，写出 `blocked_before_training` 和 P1 `not_evaluated`。这属于流程阻断，不是预测失败；也不能据此宣称核漂移存在或不存在。收尾提交尝试的原始记录保存在 executed 中。

2026-10-06T18:16:21+08:00，收尾 `git add -A` 再次返回 128，同样无法创建 `.git/index.lock`；收尾 commit 也未产生。分析已将两次提交尝试纳入 summary，全部交接文件只保存在工作树中。

训练前源码复核补齐了成功 cell 恢复时的合同摘要与保存 commit 内容校验，并重新 pin 源码。问题、条件和判据未变；失败提交时的预注册原样保存在 `executed/preregistration_at_failed_attempt.json`，提交记录带有当时的 hash。修订前后均为 0 cell。

## 交接与边界

下一轮先取得本 repo `.git` 的正常写权限，审查并提交现有预注册和 pinned 源码；匹配 commit 出现前禁止训练。随后先执行 1 个 cell、运行分析核验合同和 NPZ hash，再恢复其余 11 个 cell。成功 cell 不覆盖。该设计只涉及开发条件的拟合侧；不支持新函数、新 data-seed、泛化或封存 OOD 结论。

证据：`../studies/r006_early_direction/preregistration.json`、`../studies/r006_early_direction/summary.json`、`../studies/r006_early_direction/executed/preregistration_commit_attempt.json`、`../studies/r006_early_direction/executed/final_commit_attempt.json`。
