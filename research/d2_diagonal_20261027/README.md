# D2：删除交叉项的一轮消融

目的：在原有3240条合成固定特征轨迹上，检查“只保留每个学习方向自身的贡献”能否给出足够好的停止建议。

用户于2026-10-27明确授权：取回并校验原始数据、预注册、执行这一轮、写通俗报告。本轮无模型API、GPU、新训练或benchmark题目，不启动任何旧supervisor或publisher。

## 文件

- `preregistration.json`：唯一问题、信息边界、逐条件判据、失败处理和资源上限。
- `numerics.py` / `run.py`：消融与只读原始数据验证；拒绝覆盖已有run。
- `test_numerics.py`：用人工小矩阵核验负风险、对角特例、整数搜索和边界。
- `test_feature_space.py`：单/双输出、满秩/秩亏的独立直接GD与噪声响应递推核验。
- `render_report.py`：完成后生成通俗Markdown报告、图及登记的解释性示例；不改变主判定。

原始归档由 `tools/fetch_evidence.py --group d2` 校验后解压到外部 `AIQ_KB_DATA/imports/d2_handoff_20261009`。原summary和原源代码继续从本仓库读取，不覆盖旧证据。

## 执行

先完成D2分片、总归档、逐文件的校验；然后提交预注册与计算代码。该提交必须早于以下执行。脚本会校验新旧依赖文件与Git提交一致，并对全部原始输入再验hash。

```bash
export AIQ_KB_DATA_ROOT=/Users/mapleandrew/meta_circle/Science_of_AI_Textbook/AIQ_KB_DATA
.venv/bin/python research/d2_diagonal_20261027/run.py
.venv/bin/python research/d2_diagonal_20261027/render_report.py
```

产物仅写外部 `AIQ_KB_DATA/runs/d2_diagonal_20261027`。`execution.json`记录真实开始时间、冻结提交、依赖hash和运行环境；`rows.jsonl`保留全部条件；`summary.json`为完整精度汇总。失败时保留`failure.txt`和部分记录，不覆盖、不自动改规则重跑。

预注册门槛是逐recipe判定。`overall.pooled_stopping_pass`只是汇总描述，不能替代`recipes_passing_both == 12`。负信号误差与负总风险分别计数，停止效果与风险模型有效性分别解释。

## 验证

```bash
.venv/bin/python -m pytest -q research/d2_diagonal_20261027
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/python tools/verify_handoff.py
```

首轮实现阶段：新增合成测试7项通过；仓库测试113项、19个subtest通过；交接检查2590个在库文件通过，未下载的其他方向归档不属于本轮范围。

本轮所有条件已被历史分析看过，属于development；不声称新数据、因果训练干预或train-only早停。
