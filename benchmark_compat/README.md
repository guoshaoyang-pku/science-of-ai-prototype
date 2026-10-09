# KB lab 与 AIQ main 的兼容代码

题目与基本 benchmark runtime 仍来自固定的 external/ArchitectureIQ main。旧 KB lab 另外使用 process 诊断、target-transform 和数值校验扩展，公开 main 尚未包含。这些已有源改动保存为 code-only patch（11 个文件），原/新 hash 在 manifest.json，不包含题池或 test。

需要 lab 扩展时创建外部兼容副本：

~~~bash
.venv/bin/python tools/prepare_benchmark.py
export AIQ_BENCH_ROOT=/absolute/path/printed/by/the/tool
~~~

工具从固定 main clone，核对补丁 hash、应用并核对 11 个输出 hash，不修改 submodule，不调用模型。题库 release 仍必须显式选择并审查隔离。新环境的完整 lab 训练未在本次交接重跑。
