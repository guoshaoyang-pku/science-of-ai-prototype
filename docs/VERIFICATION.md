# 交付验证 · 2026-10-09

- 全套回归 113 passed、19 subtests passed，34.31 秒；合成题池、假 CLI，不运行真实模型。
- 完整交接 hash 核对 16,312 文件；三个归档 13,722 个 payload 均按清单校验并成功解压。
- 最新 D2 独立直接 GD 检查 8 条条件 × 256 步，3360 hashes，最大误差 1.4e−13；扩展分母 2700，103 边界失败保留。
- Wheel 成功构建，在隔离 venv 安装，runtime 和 release helper 导入通过；CLI help 通过。
- 外部 parked run 准备成功（D13 和 D2 各一），模型启动数为零；历史指令与新 inbox 分开，D2 必读最新证据。
- AIQ submodule 固定 main 的 6eafe8c1e11c5dc6ad28254caf2669be2ee4da8c，干净；最终 benchmark test 未读取。
- 原来源 16,312 个文件 hash 全不变，原 runtime checkout 干净；旧 v2 的三份停车账仍为原有未提交状态。
- 交付主入口、最新 D2 图与 evidence 链接无缺失。

原始 source hash、归一化/适配 hash 与大数组交付 hash 在 evidence/manifests；metadata/source 合同中登记的旧 hash 仍指原执行字节，不能将适配源码冒充旧原字节。逐项改动登记见 evidence/portability_changes.json。

旧发布器没有实际运行，新 D2 publisher 在新机器尚未完成端到端站点发布验证。历史 v1/v2 所有训练脚本未全量重跑；交接验证以保存文件 hash、既有 runtime suite 和 D2 数值复核为准。

脱敏扫描：Git 文本 2619 份、归档文本 5209 份，凭据/私钥/token/个人目录模式零命中。独立结构审查已修复四项；secret 模式复核由主 agent 执行（独立 sanitizer 的任务传递失败），不称其为独立凭据审计。
