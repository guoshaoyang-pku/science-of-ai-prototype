# RL trial 分析：操作手册与避坑（360-2 集群）

原 rl-experiment-analysis skill 的全部操作知识，report_to_human 的领域参考。脚本在 skill 目录 `scripts/` 下。

## 报告结构细则（配合四层协议）

1. **Trial 名称 + panel 链接**：文档里只写 trial 名称（如 `async_dp_pool2134cap8k`）。panel 链接给到 chat 里（L4），不要写进文档。
2. **一段话讲指标变化**：reward、seqlen、截断率、clip、entropy/kl 的走势，加上 held-out 探针（若存在）。一段话，不要列表。按 STE 纪律写：短句、术语单一。
3. **主要图**：reward 曲线、seqlen 曲线（有 completion cap 惩罚的 run 里 seqlen 是主角，必须放）、clip ratio。图放项目 `docs/reports/figs/`，用相对路径嵌入。可选：entropy/kl、held-out acc。用 `scripts/plot_run_metrics.py`。
4. **思维链分析（核心）**：
   - 早期 CoT（init 模型或最早 ckpt；注意 `save_total_limit` 轮换，早期 ckpt 常已被删，用 init 模型复现 step-0 行为）；
   - 中期 seqlen 显著下降时点的 CoT：**长的和短的都要展示**；
   - 最后稳定的短 CoT；
   - 生成条件必须对齐训练：同温度/top_p（通常 T=1.0）、`enable_thinking=True`、同样的 completion cap、held-out 题集。用 `scripts/gen_cot_one.py`。
   - **对照臂（强烈建议）**：`scripts/ctrl_think_vs_direct.py` 在同一 checkpoint 上跑 enable_thinking 开/关两臂（同采样、配对胜负统计）。选**思维链文本仍会出现的 checkpoint**（中期）才有信息量；末期模型已自我 skip，两臂都近乎 direct，对照退化。vLLM 的 spawn 启动方式要求脚本是真实文件（不能 heredoc/stdin），且需 `PYTHONPATH` 指向含 `eval_mcq.py` 的目录。
5. **截断规则**：1000 字符以内一律不截断，全文放出；更长的保留开头/中间/结尾各约 500 字符，用 `……（中略 N 字）……` 标明省略。
6. **分析**：最后写。不要过度结构化（少用表格/小节），行数尽量少，给人读的散文。

## 操作手册（360-2 集群）

- 根目录 `/dataREMOTE_HOME/aiq_rl/`；run 目录 `runs/<trial>/`（`log_history.json`、`checkpoint-*`、`trainer_state.json`）；评测在 `evals/`，live 探针历史在 `evals/live_<tag>/history.jsonl`；共享镜像 `/data/shared/guoshaoyang/aiq_rl_store/runs/`。
- 训练指标：`runs/<trial>/log_history.json`（每 step 一条；个别 step 缺字段要过滤）。scp 回本地出图。
- **Panel**：rlforge dashboard，端口 8871：
  - 进程：`cd /dataREMOTE_HOME/aiq_rl && setsid nohup ./venv/bin/python -m rlforge.dashboard --root /dataREMOTE_HOME/aiq_rl --registry dashboard_registry.json --port 8871 --min-steps 50 >> logs/dashboard.log 2>&1 < /dev/null &`
  - `--min-steps` 会过滤短 run（如 182 步的 trial），dashboard 看不到某 trial 时先检查这个。
  - run 选择是下拉框 + localStorage，**没有 URL 参数**；chat 里给 `http://127.0.0.1:8871` 并说明下拉选中 trial 名。
  - Mac 侧隧道：`ssh -f -N -L 8871:localhost:8871 360-2`（config 里的 RemoteForward 2222 失败是无害警告；**不要**加 `ExitOnForwardFailure=yes`，也不要 `ClearAllForwardings`——会连 `-L` 一起清掉）。
  - 给 run 补 notes：编辑 `dashboard_registry.json`，加 `"trial名": {"notes": "<html>"}`。
- **从 checkpoint 生成 CoT**：TRL ckpt 缺 `preprocessor_config.json`，vLLM 拒载。用 `eval_mcq.build_shim(ckpt_path)`（需 `export AIQ_BASE_MODEL=...` 指向基座模型目录）。`scripts/gen_cot_one.py` 封装了该流程（vLLM 离线、逐模型一进程、固定题目切片、追加写 JSONL），先 scp 到集群再运行。
- **vLLM 离线必须带环境变量**：`VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ALLREDUCE_USE_FLASHINFER=0`（节点无 nvcc，flashinfer JIT 会挂）。
- SSH 到 360-2 时常抖（QConnect 私网 IP）：多试几次、`-o ControlPath=none` 避开卡死的多路复用 socket；长任务（CoT 生成）用后台会话跑。

## 避坑

- reward 是四值混合（如 -2 截断 / -0.5 不可解析 / 0 错 / +1 对），不是 accuracy；训练池若按块顺序出题，步级 reward 不可跨步比较。
- seqlen 均值会被长尾拉高：报告里给中位数/分布，别只给均值。
- "空 think + 直接给答案"是行为证据，不等于模型内部没算；分析措辞要区分这两者。
- 中期"长 CoT 回潮"（个别 step 截断率回升）值得单独指出：可能是退化 token 复读吸引子，不是恢复推理——拿到长样本先查是否重复模式。
- 检查思维链是否针对题目（引用题目特定内容）还是通用模板套话；留意答案与自身"分析"矛盾、答案格式写坏、语言漂移（中英切换）等现象，这些是模板化而非推理的证据。
- held-out 评测要报字母塌缩基线（恒选 A 的准确率），否则单点 acc 无意义。
- eval 记录里的 `pred_no_cot` 等对照字段可能全空（字段在、值没填）；拿来做无 CoT 对照前先数非空条数，空就自己跑 `ctrl_think_vs_direct.py`。
- 思维链长度分桶准确率（≤20 / 21–100 / 101–500 / >500 tok）是"CoT 是否承载答案"最便宜的检验：负相关说明 think 文本是困惑的痕迹而非推理载体（混淆：难题天然诱发长输出，需结合套话证据一起读）。
- base 切片在 cap 下通常几乎全错（截断=无有效答案）；报切片正确数时附截断数，用 `<answer>` 标签解析，不要目测。
