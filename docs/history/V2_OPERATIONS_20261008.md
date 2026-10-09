# OPERATIONS.md — 操作手册（science_program_v2 中心化自动研究）

更新：2026-10-06。本手册描述系统如何工作、如何监控、如何停车、如何改方向。

## 1. 系统是什么

一个 supervisor 进程串行调度研究轮次。每轮启动一个 Codex 会话（模型 Sol 6.1 Ultra = `gpt-6.1-sol`，reasoning effort `ultra`，走本地 relay 3107，cctq 主路由、phybench 兜底）。会话独立工作：读中心文件 → 选一个小问题 → 预注册并 commit → 跑本机小实验 → 写发现 → 更新中心状态与 KB → 收尾 commit → 退出。

记忆不在会话里，在文件与 git 里。这就是"中心化"：单进程、单状态、单 KB、单 git 历史。

```
central/supervisor.py（唯一常驻进程）
   └── 每轮: codex-cctq exec（一次性会话, workspace-write 沙箱, 无网络）
          ├── 读: GOAL.md / AGENTS.md / directions/Dx/task.md / central/state.json
          ├── 写: directions/Dx/{studies,findings,report.md} / central/{kb,state}.json / reports/PROGRESS.md
          └── git commit（预注册在前，收尾在后）
```

## 2. 目录地图

| 路径 | 内容 |
|---|---|
| `GOAL.md` | 用户目标、8 个方向、硬边界 |
| `AGENTS.md` | 每轮会话的工作规则（科学纪律） |
| `central/config.json` | 模型、每轮超时、轮数预算、方向顺序（每轮开始重读，可热改） |
| `central/state.json` | 轮次计数、每方向状态（active/parked）、历史 |
| `central/kb.json` | 中心化短 KB（只登记有保存证据的 claim） |
| `central/log.jsonl` | supervisor 事件日志（round_start/round_end/…） |
| `central/rounds/rNNN_Dx/` | 每轮产物：prompt.md、output.log（完整会话输出）、last_message.md |
| `directions/D1..D8/` | 每方向：task.md（问题）、studies/（实验）、findings/（逐轮）、report.md（活文档）、inbox.md（可选，给下一轮的指令） |
| `sources/` | 冻结的 v1 KB（7 条）与主线 KB（143 条）快照 + hash |
| `reports/PROGRESS.md` | 全部轮次的一段式进度（最新在上） |

## 3. 日常操作

**看进度（随时，安全）**
```sh
cd aiq_rl/experiments/science_program_v2
cat reports/PROGRESS.md            # 人读的进度
tail -5 central/log.jsonl          # 轮次事件
git log --oneline | head -20       # 研究时序（预注册 commit 在实验之前）
ps -p $(python3 -c "import json;print(json.load(open('central/state.json'))['supervisor_pid'] or '')") 2>/dev/null  # 进程活着吗
```

**温和停车**（跑完当前轮再停）
```sh
touch central/STOP
```
恢复：删除 `central/STOP`，重新启动 supervisor（见 §5）。

**立即停车**
```sh
kill $(cat central/supervisor.lock)   # SIGTERM：supervisor 会连带终止当轮 codex 子进程
```
当轮会留下未收尾的文件；恢复后下一轮会看到 git 状态并处理，或人工 `git checkout -- .` 丢弃半轮产物。

**给某个方向下指令**（下一轮生效）
```sh
echo "下一轮请先做 XXX" >> directions/D4_sigmoid_training_curve/inbox.md
```

**改预算/顺序/模型**：直接编辑 `central/config.json`（每轮开始重读，无需重启）。
- `rounds_per_direction`：每方向轮数（默认 3；到达后方向转 parked，等用户审查）
- `max_total_rounds`：总轮数上限（默认 24）
- `round_timeout_sec`：单轮超时（默认 7200 秒）

**新增方向**：`mkdir directions/D9_xxx` + 写 `task.md`（参照现有格式）+ 把 `D9_xxx` 加进 `config.json` 的 `direction_order` 并在 `state.json` 的 `directions` 里加同名条目（rounds_done=0, status=active）。下一轮自动纳入调度。

## 4. 每轮的生命周期与验收

1. supervisor 选方向（轮转：轮数少的优先，用户点名的 D1–D4 排在前面）。
2. 生成 prompt（存 `central/rounds/rNNN_Dx/prompt.md`），启动 `codex exec`：workspace-write 沙箱、无审批、无网络。
3. Codex 轮内义务（AGENTS.md）：预注册先 commit；实验 ≤20 分钟计算；产物与状态全部落盘；收尾 commit；最后一行 `ROUND_RESULT: ok|partial|failed | 一句话`。
4. supervisor 解析 ROUND_RESULT、backstop commit、更新 state.json 历史、睡 30 秒、进下一轮。

**验收一轮**：`git log` 里预注册 commit 时间早于 results 文件；`output.log` 无沙箱违规；state.json 的 last_round.result=ok。timeout/failed 的轮次同样记录，不重跑（下一轮自行处理残留问题）。

## 5. 启动 / 重启

```sh
cd science_program_v2
rm -f central/STOP
nohup python3 central/supervisor.py >> central/supervisor.log 2>&1 &
echo $! > central/supervisor.nohup.pid
```

supervisor 自带单实例锁（`central/supervisor.lock`，含 PID 存活检查），重复启动会直接退出。断电/重启后：确认无残留 codex 进程（`ps aux | grep codex-cctq`），然后按上面命令重启即可，状态从文件恢复。

## 6. 成本与规模

- 模型调用：每轮一个 ultra 会话（预计每轮数十万~百万 token 量级，视工作量）；默认预算 24 轮。走用户自己的 cctq relay 凭证，成本记在 cctq/phybench 账号。
- 本机计算：每轮实验 ≤20 分钟 CPU 小张量（v1 先例：1740 次 MLP 训练共 ~4 分钟计算）。
- 磁盘：每轮 output.log 约几 MB；实验结果 NPZ/JSON 视方向而定（v1 全部原始测量 ~854MB，v2 预计更小）。

## 7. 安全边界（系统层面已保证）

- codex 子进程沙箱 workspace-write：只能写本程序目录与 /tmp；无网络（模型调用由 codex CLI 本体走本地 relay，不受沙箱影响）。
- v1 repo、soa_async_v1 主线、kb_site、publisher：只读（AGENTS.md 明令 + 沙箱写保护）。
- 不 push、不动 benchmark test、不做 solver 评测（GOAL.md 硬边界）。
- supervisor 不自动重启自己；STOP/kill 后保持停车，等人。

## 8. 已知限制

1. 单会话上下文有限：一轮做不完的大问题会留下 partial；靠 state.json 的 next_question 与 findings 接力。
2. ROUND_RESULT 依赖会话自觉输出；缺失时记为 unknown，人工看 output.log 末尾。
3. 无自动质量审查轮：parked 后需要用户（或用户指派的审查会话）验收再追加预算。
4. relay 依赖本机 3107 端口进程存活（LaunchAgent 管理）；relay 挂掉时轮次会失败并记录，不会静默卡死（有超时）。
5. 预注册时序以本机 git commit 时间为准（v2 在独立 repo 内自 commit，比 v1 强，但仍非外部见证；如需更强见证可配置 push 到私有远端——当前明确不做）。

## 9. 交付物索引

- 技术报告：每方向 `directions/Dx/report.md`（活文档）+ `reports/` 下的阶段汇总（parked 时生成）。
- 本手册：`OPERATIONS.md`（行为变了就更新它）。
- 进度：`reports/PROGRESS.md`、`central/state.json`。
