# 操作手册

## 安装与无模型检查

~~~bash
./setup.sh
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/python tools/verify_handoff.py
.venv/bin/aiq-kb-loop --help
~~~

安装包括固定 AIQ benchmark runtime、CPU research 和 Markdown renderer 依赖。测试使用假 CLI、合成题池和临时目录；release builder 已随包提供，AIQ main 没有原本未发布的 helper。macOS/Linux 支持 fcntl/resource，Windows 使用 WSL。

submodule 在 external/ArchitectureIQ，固定 main 的 6eafe8c1…，不增加 aiq_bench_repo 子目录。upstream.lock.json 登记真实依赖。升级需手动核对 commit，不能静默浮动实验合同。

公开 main 没有旧 lab 的 process/target-transform 扩展。需要这些功能时运行 .venv/bin/python tools/prepare_benchmark.py，将打印路径设为 AIQ_BENCH_ROOT；工具在外部目录创建固定 main + 已有兼容补丁副本，核对 11 个源 hash，不修改 submodule。补丁说明见 benchmark_compat/README.md。

## 下载与复核数据

~~~bash
gh release download handoff-20261009 --repo guoshaoyang-pku/science-of-ai-prototype --pattern '*.part*' --pattern SHA256SUMS --dir ../science-evidence
.venv/bin/python tools/fetch_evidence.py --archives ../science-evidence
.venv/bin/python tools/verify_handoff.py --full
.venv/bin/python research/d2_scaling_20261009/verify.py
~~~

公开 Release 附件可从网页或 gh 下载，再传 --archives。三个归档共 100 个分片；工具先校验每片、重组并验证归档总 SHA-256。解压器只释放清单中的普通文件，拒绝越界路径和符号链接；已有不同 hash 的文件不覆盖。verify.py 对 8 个保存条件做 256 步直接 GD 核对，不发起扫描/API；会更新交付数据中的 verification.json。

默认 D2 读取本库 evidence/d2_scaling_20261009。AIQ_KB_DATA_ROOT 会改为其 runs/d2_scaling_20261009；验证时不要误指向新空数据根。plot.py 使用 --study evidence/d2_scaling_20261009/studies/n_sigma_tasks，会重绘交付图。

## 准备一轮新研究

~~~bash
export AIQ_KB_DATA_ROOT=/absolute/path/to/AIQ_KB_DATA
.venv/bin/python tools/prepare_science_run.py --run-name d13_skip_round1 --direction D13_linear_flow_world --rounds 1
~~~

这只创建外部独立 Git 研究 repo，打印下一命令，不调用模型。复制方向 task/report、中心 KB 和历史索引；历史证据作为只读引用，新产物在 run 内。新轮账从零开始，历史 113/93/92 单独保留。

确认 Git 已配置自己的 user.name/user.email（用于研究时序提交）。审核新 run 的 config（CLI、模型、effort、预算），再显式执行打印命令。启动需要 SCIENCE_ENABLE_AGENT=1；默认 STOP 保留。SCIENCE_RELAY_HEALTH_URL 和 SCIENCE_UPSTREAM_HOSTS 是可选 preflight，默认由 CLI 处理连接。

## Solver 和发布

solver 需要自己的 AIQ_KB_KEYS 和经隔离审查的 AIQ_KB_POOL，见 .env.example。不要从旧题池直接发车，不读取最终 benchmark test。旧 soa_async_v1 不在本次证据包，其公开交付仍在 science-of-ai-kb；旧 run 与 publisher 保持原路径。

旧 v2 publisher main 和自动修复/重启脚本已禁用；HTML renderer 可只读导入。新 D2 publisher 需显式 AIQ_KB_BLOG_ROOT，只写目标文件，不 commit/push。它面向已有 kb_site 索引，首次建站需指定交付目标；导入成功不等于完整发布已验证。

协作网页的发布源是 site/模糊方向/，在线入口为 [模糊方向](https://guoshaoyang-pku.github.io/science-of-ai-prototype/模糊方向/)。直接编辑该目录内的 HTML、style.css 和图片；科学结论的数字仍须核对 studies/*/summary.json。已有 Markdown 和证据附件是封存快照；新增研究结果写入独立 run，再将有来源的结论更新到网页。

~~~bash
python3 -m http.server 8000 --directory site
~~~

本地打开 http://localhost:8000/模糊方向/ 检查页面、图片和公式。将网页改动推送 main 后，.github/workflows/pages.yml 会直接上传 site/ 的静态文件；在 Actions 查看 Publish 模糊方向 的结果，再检查在线页。此流程不重新生成 HTML、不启动模型或实验；旧 publisher 与本页是不同发布入口，旧 v2 publisher 继续禁用。
