# Science of AI Prototype

给定模糊问题，由 agent 自选小问题、预注册、做本机受控实验，再把测量、反例和下一问写入知识库（KB）。本库包含支持解题与异步研究的 KB runtime，以及 13 个模糊方向的研究快照。

先读 [交接说明](docs/HANDOFF.md)、[研究方向](docs/RESEARCH_DIRECTIONS.md) 和 [操作手册](docs/OPERATIONS.md)。默认保持 parked，不启动研究轮。

研究网页：[模糊方向](https://guoshaoyang-pku.github.io/science-of-ai-prototype/模糊方向/)。网页实体在 [docs/模糊方向/](docs/模糊方向/)；协作者可直接编辑 HTML、样式、图片和证据附件，推送 main 后由 GitHub Pages 自动发布。

## 安装与检查

Python 3.10+，macOS 或 Linux；Windows 使用 WSL。

~~~bash
git clone --recurse-submodules https://github.com/guoshaoyang-pku/science-of-ai-prototype.git
cd science-of-ai-prototype
./setup.sh
.venv/bin/python tools/verify_handoff.py
.venv/bin/python -m pytest -q
~~~

安装与检查不需要模型密钥。完整曲线需下载 [handoff-20261009 Release](https://github.com/guoshaoyang-pku/science-of-ai-prototype/releases/tag/handoff-20261009)：

~~~bash
gh release download handoff-20261009 --repo guoshaoyang-pku/science-of-ai-prototype --pattern '*.part*' --pattern SHA256SUMS --dir ../science-evidence
.venv/bin/python tools/fetch_evidence.py --archives ../science-evidence --verify-only
~~~

去掉 --verify-only 会将已校验归档解压到本 checkout 对应证据目录；测量结果被 gitignore 排除。

## 代码和证据

| 入口 | 内容 |
|---|---|
| src/aiq_kb/、tests/ | loop、持久 jobs、agent 适配、lab、viewer、publisher 与回归测试 |
| science_program_v2/ | supervisor、128 条旧 claim、13 方向 task/report/findings、预注册与 summary |
| research/d2_scaling_20261009/ | 最新 D2 扩域与跨任务实验、验证、绘图、发布代码 |
| evidence/d2_scaling_20261009/ | 最新 D2 报告、数字、图和执行审计；优先于旧 D2 报告 |
| references/science_program_v1/ | v2 引用的 v1 代码、报告与轻量证据 |
| evidence/manifests/、evidence/history/ | 每文件 source/export hash、归档 hash、历史 commit 索引 |
| docs/模糊方向/ | 可共同编辑的研究网页、图片和网页证据附件；GitHub Pages 发布源 |
| external/ArchitectureIQ/ | 来自 AIQ main 的固定 commit submodule；题目与 benchmark runtime 的外部来源 |
| benchmark_compat/ | 旧 KB lab 的 process/target-transform 已有补丁，可应用到外部副本 |

题库来源是 [renrua52/ArchitectureIQ](https://github.com/renrua52/ArchitectureIQ) 的 main，当前固定 6eafe8c1e11c5dc6ad28254caf2669be2ee4da8c。新实验还需明确 release、暴露记录和 group/seed/variant 隔离；换 seed 不产生新题。题库正文与最终 benchmark test 没有复制进本库。

保存数据共约 3.3 GB，按 v1、v2、D2 三个归档提供为 100 个 Release 分片。工具校验分片后重组，再校验归档总 hash。数值数组保持原字节；机器路径在文本/元数据中归一化，原 hash 与交付 hash 都在清单中。完整模型会话、CLI home、凭据和旧 Git 对象不入库；原时序以 commit 索引交接，不把索引称为独立时序证明。

## 当前结果

D2 新扫描有 3240 条固定特征回归轨迹。12/12 个任务与特征条件均不支持扩域后只换系数的统一指数 1 定律；回升不要求 train 先接近零。这是受控模型的边界结果，不确立通用训练理论或新的 solver 收益。[最新报告](evidence/d2_scaling_20261009/report.md) 保留失败分母、曲线和机制示意图。

MIT 许可，保留 ArchitectureIQ Authors（MetaCircle）署名。早期公开版本仍在 [science-of-ai-kb](https://github.com/guoshaoyang-pku/science-of-ai-kb)。
