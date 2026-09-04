# 第 5 台服务器：候选动作长期价值 × SEED 输出变化实验交接

更新时间：2026-09-02（Asia/Shanghai）

## 1. 当前唯一研究主线

暂不设计新算法，只研究：

```text
智能体经验
  -> 关键决策的输出数值变化
  -> 长期行为价值变化
  -> 参数梯度结构
  -> 将来是否存在可积累更新
```

当前正在完成第一阶段：

> 候选动作长期价值 × Base/SEED 输出概率变化。

动机是：SEED 的 expert-next-action top-1 仅比 Base 高 3--4 点，但完整任务成功率很高，
因此不能继续把“更像唯一 expert 下一步”当作功能改善。必须直接测一个候选动作对最终任务完成的
长期价值，再判断 SEED 是否把概率质量从低价值动作转向高价值动作。

## 2. 实用性判据

用户明确要求：小幅统计提升不算真正有用。

主实用指标：

```text
SEED top-1 是否选中当前状态下折扣长期价值最高的候选动作
```

实用门槛：`>= 80%`。

- 统计显著但低于 80%：只能写成“存在弱信号/机制证据”。
- 达到或超过 80%：才允许写成“具备实际用途”。
- 当前 seen 的 43.02% 明确未通过，差 36.98 点。

## 3. 第 5 台连接与 tmux

服务器：

```text
公网：124.221.190.139:246
内网：10.0.0.14
用户：user
主机名：nkilm70yynvd4yk
```

公网 SSH：

```bash
ssh -p 246 user@124.221.190.139
```

密码由用户单独提供，**不写入本文档、脚本、日志或 tmux history**。

进入当前服务器的目标会话：

```bash
tmux attach -t 0
```

会话 `0` 中约定窗口：

| 窗口 | 用途 |
|---|---|
| `0:bash` | 用户原有窗口，保留不修改 |
| `value-run` | 32 分片断点续跑主实验 |
| `value-status` | 每 15 秒显示进度、错误、marker 和 GPU 状态 |
| `value-analysis` | 等待实验完成后自动做完整性对齐、10,000 次 bootstrap 和 80% 门槛判定 |
| `value-handoff` | 打开本文档 |

切换窗口示例：

```bash
tmux select-window -t 0:value-status
```

## 4. 本地与远端目录

本地主归档（最终真源）：

```text
/data/hanning/agent_self_evolution_gradient_bundle_20260901
```

第 5 台执行工作副本：

```text
/data/user/agent_self_evolution_gradient
```

实验结束后必须把远端新增代码、日志、trace、分析和报告同步回本地主归档，并更新主归档的
`ARTIFACT_SHA256SUMS.txt`。远端工作副本不是最终归档真源。

## 5. 模型、环境与输入路径

远端 Python：

```text
/data/user/agent_self_evolution_gradient/.venv/bin/python
```

远端 Base 模型：

```text
/data/user/agent_self_evolution_gradient/model
```

远端 SEED 模型：

```text
/data/user/agent_self_evolution_gradient/seed_model
```

ALFWorld 数据：

```text
/data/user/agent_self_evolution_gradient/alfworld_data
```

状态数据：

```text
/data/user/agent_self_evolution_gradient/data/alfworld_expert_large/valid_seen.jsonl
/data/user/agent_self_evolution_gradient/data/alfworld_expert_large/valid_unseen.jsonl
```

Base 输出分数：

```text
/data/user/agent_self_evolution_gradient/baseline_traces/valid_seen.jsonl
/data/user/agent_self_evolution_gradient/baseline_traces/valid_unseen.jsonl
```

SEED 输出分数：

```text
/data/user/agent_self_evolution_gradient/outputs/seed_checkpoint_plain/valid_seen_shard_*/trace.jsonl
/data/user/agent_self_evolution_gradient/outputs/seed_checkpoint_plain/valid_unseen_shard_*/trace.jsonl
```

## 6. 长期价值的实验定义

对每个保存的专家状态 `s` 和每个 admissible candidate `a`：

1. 重置到相同 game trial；
2. 精确重放该状态以前的历史动作；
3. 严格检查 observation 和候选动作列表与昨日保存数据逐字一致；
4. 强制执行候选动作 `a`；
5. 随后让 ALFWorld 官方 expert 恢复；
6. 在从 episode 开始总计 50 步预算内测是否完成；
7. 同时记录恢复步数和折扣成功值：

```text
V_gamma(s,a) = 1[win] * 0.95^(recovery_steps - 1)
```

两个主比较：

```text
E_Base[V] = sum_a P_Base(a|s) V(s,a)
E_SEED[V] = sum_a P_SEED(a|s) V(s,a)
```

以及：

- Base/SEED top-1 的长期价值；
- Base/SEED top-1 是否为价值最高动作；
- SEED 增加概率的动作平均价值；
- SEED 减少概率的动作平均价值；
- 概率从低价值动作向高价值动作移动的净效果；
- `delta logit`/`delta probability` 与价值的状态内 Spearman；
- rescued、harmed、stable expert、changed nonexpert 分组；
- task family 和 action verb 分组；
- 按独立 game trial 做 10,000 次 cluster bootstrap。

这个值是 `official-expert-recovery` 条件价值，不等同于 Base 或 SEED 自身 rollout 的策略价值；
报告中必须保留这一边界。

## 7. 规模与当前进度（迁移到 tmux 前）

完整规模：

| split | 状态 | 独立 game trial | 候选动作干预 |
|---|---:|---:|---:|
| valid_seen | 444 | 33 | 12,680 |
| valid_unseen | 527 | 34 | 14,980 |
| 合计 | 971 | 67 | 27,660 |

主实验 `handcoded expert + 固定随机种子`：

```text
valid_seen:   444 / 444 完成
valid_unseen: 340 / 527 完成
```

辅助 planner 对照（已停止冗余尾部）：

```text
valid_seen:   444 / 444 完成
valid_unseen: 265 / 527 完成
```

两种恢复定义逐动作重叠核验：

- seen 444 个共享状态中仅 1 个状态不同；
- 当时已共享的 unseen 260 个状态全部一致；
- 因此保留辅助结果作为稳健性证据，不再浪费资源补满其尾部。

## 8. 当前正式结果：valid_seen

主实验全量 seen：

- 444 状态、33 个独立 trial、12,680 个候选动作；
- 候选动作在总 50 步预算内恢复成功率：94.09%；
- 141/444 状态存在二元成功差异；
- 432/444 状态存在折扣价值差异；
- expert 动作为最高折扣价值动作：40.77%；
- Base top-1 为最高折扣价值动作：31.53%；
- SEED top-1 为最高折扣价值动作：43.02%；
- 与 80% 实用门槛差：`-36.98` 点，明确不实用；
- SEED 概率加权二元成功值变化：`+0.03122`，95% CI `[+0.02193, +0.04057]`；
- SEED 概率加权折扣价值变化：`+0.02907`，95% CI `[+0.02272, +0.03535]`；
- SEED top-1 折扣价值变化：`+0.08488`，95% CI `[+0.05864, +0.11078]`；
- SEED 增加概率动作与减少概率动作的平均价值差：`+0.08302`；
- SEED 在价值最优动作上的总概率变化：`+0.03918`；
- `delta logit` 与价值的状态内 Spearman：`-0.06183`，CI 跨 0；
- `delta probability` 与价值的状态内 Spearman：`+0.04997`，CI 跨 0。

严格表述：

> SEED 在 seen 上存在统计可靠的价值导向概率质量转移，但绝不是强价值选择器；43.02% 远低于
> 80% 实用门槛。概率加权价值提升并不意味着所有候选动作的 logit 会按价值单调排序。

## 9. 已发现的问题与处理

### 9.1 expert 随机性

ALFWorld hand-coded expert 在多个对象/回退命令可选时使用 Python 模块级 `random`。
仅调用 `env.seed()` 不足，首次重复实验不一致。现已在每个状态—动作干预前显式执行同一种子
`random.seed(seed)`，两个独立并行重复的 25/25 个候选结果逐项一致。

### 9.2 expert 下一步不是价值标签

冒烟状态中 25 个候选动作均可恢复成功，但折扣价值从约 0.20 到 0.77；记录的 expert 动作不是
最高折扣价值动作。全量 seen 中 expert 也只有 40.77% 为最高价值动作。这验证了实验动机。

### 9.3 统计显著不等于有用

seen 的多个 bootstrap CI 为正，但 43.02% 距离 80% 很远。最终结论必须优先报告“不实用”，
统计显著只作为机制证据。

### 9.4 当前价值仍是 proxy

当前测的是官方 expert 接管后的可恢复性/恢复效率，不是 Base/SEED 自身继续 rollout 的
`Q^Base` 或 `Q^SEED`。后者更贵，应作为下一次独立实验，不得在当前报告中混称。

### 9.5 当前还没有进入 token 或参数梯度阶段

本轮只完成研究问题一。逐生成位置词表 logit、概率质量在动作词/物体词/容器词之间的迁移、
Base/SEED 梯度与真实参数 delta 对照均尚未启动。

## 10. 新增代码、运行和输出路径

核心数值模块：

```text
/data/user/agent_self_evolution_gradient/src/self_evolve/action_value.py
```

环境干预评测：

```text
/data/user/agent_self_evolution_gradient/scripts/evaluate_alfworld_action_values.py
```

统计分析：

```text
/data/user/agent_self_evolution_gradient/scripts/analyze_action_value_alignment.py
```

tmux/分片运行：

```text
/data/user/agent_self_evolution_gradient/scripts/remote/run_action_value_4shards.sh
/data/user/agent_self_evolution_gradient/scripts/remote/resume_action_value_tmux.sh
/data/user/agent_self_evolution_gradient/scripts/remote/action_value_status.sh
/data/user/agent_self_evolution_gradient/scripts/remote/finalize_action_value_after_run.sh
```

测试：

```text
/data/user/agent_self_evolution_gradient/tests/test_action_value.py
```

主输出：

```text
/data/user/agent_self_evolution_gradient/outputs/action_value_alignment
```

辅助 planner 输出：

```text
/data/user/agent_self_evolution_gradient/outputs/action_value_alignment_planner
```

主分片日志：

```text
/data/user/agent_self_evolution_gradient/outputs/action_value_alignment/logs/shard_*.log
```

最终自动分析：

```text
/data/user/agent_self_evolution_gradient/outputs/action_value_alignment/analysis/results.json
/data/user/agent_self_evolution_gradient/outputs/action_value_alignment/analysis/trace.jsonl
/data/user/agent_self_evolution_gradient/outputs/action_value_alignment/analysis/REPORT.md
/data/user/agent_self_evolution_gradient/outputs/action_value_alignment/analysis/analysis.stdout.log
```

## 11. 验证状态

- 新增纯函数测试：4 passed；
- 主归档全测试：19 passed；
- 目标 observation 和 admissible actions 在冒烟与正式运行中严格对齐；
- 同种子并行重复：25/25 个候选逐项一致；
- 当前运行日志：无 traceback、无重放 mismatch、无候选 mismatch。

## 12. tmux 中的后续任务顺序

1. `value-run` 从 340/527 unseen 断点继续，seen 会自动跳过；
2. `value-status` 持续监控进度与错误；
3. `value-analysis` 等待 `TMUX_RUN_COMPLETE`，然后自动：
   - 核对 Base/SEED/value 三份数据按全局状态索引和候选动作对齐；
   - 对 seen/unseen 分别分析；
   - 做 10,000 次 episode-cluster bootstrap；
   - 应用 80% 实用门槛；
   - 生成 JSON、逐状态 trace、Markdown 报告；
4. Codex 读取最终结果，写完整中文研究报告；
5. 将远端新增内容 rsync 回本地主归档；
6. 更新 `ARTIFACT_SHA256SUMS.txt` 并重新跑 19 项测试；
7. 移除本次临时 SSH 公钥和本地临时私钥。

## 13. 不应改变的研究边界

- 当前不设计新算法；
- 不以 expert-next-action accuracy 代替长期价值；
- 不把统计显著的小幅提升写成实际有用；
- 低于 80% 必须明确判为未达到实用门槛；
- 不把 expert-recovery proxy 写成模型自身 rollout value；
- 完成价值实验后，才进入逐 token 输出逻辑值；
- 梯度实验必须等“哪些输出变化确实有价值”确定以后再做。
