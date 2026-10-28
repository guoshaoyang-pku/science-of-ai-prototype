# D2：保留5%交叉项能否修复停止预测

用户于2026-10-28授权这一轮：先预注册，再与随机和按大小保留作同预算对照，重点解释已知“1步误停／255步最佳”的例子。只复用已有D2数据，不训练网络、不调用模型或GPU，不操作旧supervisor/publisher/benchmark。

## 冻结方案

- 主方法 `change05`：在固定时间格点上按 `max |(t+1) Cij'(t)|` 排序。
- 对照 `magnitude05`：按窗口末端交叉项幅值排序。
- 对照 `random05_00..09`：10组固定随机选择，不挑最好一组。
- 每组严格保留 `floor(0.05*r*(r-1)/2)` 对活动模态；零预算也保留。每任务与数据组合选一次，各噪声共用。
- 主判定为12个recipe逐个满足时间80%和低额外风险90%门槛。只修好重点例不能代替整体判定。
- 全部条件已被看过，属于development。计算需要干净目标和验证几何，不是train-only早停。选项时读取所有交叉项，实验只检验保留信息量，不证明算力节省或每项的必要性。

## 运行

预注册和依赖源码提交后，执行：

```bash
export AIQ_KB_DATA_ROOT=/Users/mapleandrew/meta_circle/Science_of_AI_Textbook/AIQ_KB_DATA
.venv/bin/python research/d2_sparse_cross_20261028/run.py
.venv/bin/python research/d2_sparse_cross_20261028/render_report.py
```

产物写外部 `AIQ_KB_DATA/runs/d2_sparse_cross_20261028`。脚本拒绝覆盖；输入与原消融结果全部验hash。保存实际特征基、评分、选中索引、随机种子、38880条新评价与6480条参照、重点例的风险拆项。原始证据和旧run只读。

`summary.json`中的主判定只使用change05的12/12要求；随机十次是选择算法的随机性，不是十份独立数据。负风险和停止质量分别评价。

## 验证

```bash
.venv/bin/python -m pytest -q research/d2_sparse_cross_20261028
PYTHONPATH=src .venv/bin/python -m pytest -q
.venv/bin/python tools/verify_handoff.py
```

实现阶段新增9项合成测试、原113项测试与19个subtest通过；交接检查通过。测试覆盖步数导数、配对评分、固定排序、预算向下取整、空mask、完整mask恢复和风险展开。

## 本轮已完成

预注册冻结提交为 `7dc745d2c4e21acb98ef4fa0a4534c2e5e707eda`，早于计算。38880条新方法评价和6480条只读参照均已保存。主方法与按幅度方法均通过12/12条件；重点例从465对保留23对，两者都选回255步，随机10次都仍选1步。

主方法时间通过3043/3137、低额外风险通过3200/3240。按幅度对照为3042/3137、3199/3240，没有证据说变化评分明显更好。仍有252条负信号风险、20条负总风险，最坏额外风险为初值的12.78%；这些未被剔除。

外部通俗报告：`/Users/mapleandrew/meta_circle/Science_of_AI_Textbook/AIQ_KB_DATA/runs/d2_sparse_cross_20261028/report.md`。该run保留全部选择名单、特征基、逐行结果、图、输入验证和独立审计。`result_manifest.json`记录外部产物hash；不把原始测量或运行记录复制进代码库。
