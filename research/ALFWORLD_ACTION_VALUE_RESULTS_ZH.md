# ALFWorld 候选动作长期价值与 SEED 输出变化：完整结果

更新时间：2026-09-03（Asia/Shanghai）  
状态：主实验完成并归档；下一阶段逐 token / 全词表实验已开始

## 1. 问题与边界

本实验不再把 ALFWorld 官方 expert 的唯一下一步当作“正确动作”，而是直接干预同一状态下的
每个 admissible action，测量执行后能否由官方 expert 在 episode 总计 50 步预算内恢复成功，以及
恢复所需步数：

```text
V_gamma(s,a) = 1[win] * 0.95^(recovery_steps - 1)
```

随后比较 Base 与 SEED 在同一候选集合上的概率分布，回答概率质量是否从低价值动作移向高价值
动作。这个数值是 `official-expert-recovery` 条件价值，不是 Base/SEED 自身继续 rollout 的策略
价值；后续报告不得混称。

用户指定的实用主指标是：

```text
SEED top-1 是否属于该状态下折扣价值最高的候选动作
```

门槛为 `>= 80%`。统计显著但低于 80% 只能视为机制信号，不能称为实用能力。

## 2. 实验规模与完整性

| split | 状态 | 独立 game trial | 候选动作干预 | 错误 |
|---|---:|---:|---:|---:|
| valid_seen | 444 | 33 | 12,680 | 0 |
| valid_unseen | 527 | 34 | 14,980 | 0 |
| 合计 | 971 | 67 | 27,660 | 0 |

- observation、历史动作与 admissible commands 在环境重放时严格核对；
- 每个状态—动作干预前同时固定 Python 模块级随机种子，解决 hand-coded expert 的隐含随机性；
- 两个独立重复的 25/25 个候选结果逐项一致；
- 辅助 PDDL planner 对照完成 seen 全量和 unseen 265 个状态；两种恢复机制在 seen 444 个共享状态
  中仅 1 个不同，在当时重叠的 unseen 260 个状态中完全一致；
- 主结果按独立 game trial 做 10,000 次 cluster bootstrap。

## 3. 主结果

| 指标 | valid_seen | valid_unseen |
|---|---:|---:|
| 候选动作恢复成功率 | 94.09% | 92.84% |
| 有二元价值差异的状态 | 141/444 | 207/527 |
| 有折扣价值差异的状态 | 432/444 | 513/527 |
| expert 动作为价值最优 | 40.77% | 36.62% |
| Base top-1 为价值最优 | 31.53% | 32.07% |
| SEED top-1 为价值最优 | **43.02%** | **38.33%** |
| 距 80% 实用门槛 | **-36.98 点** | **-41.67 点** |

结论很明确：SEED 相比 Base 有改善，但 seen/unseen 均远低于 80%，所以没有达到实际有效门槛。

## 4. 概率质量是否向高价值动作移动

| 指标 | valid_seen，均值 [episode 95% CI] | valid_unseen，均值 [episode 95% CI] |
|---|---:|---:|
| 概率加权二元成功值变化 | +0.03122 [+0.02193, +0.04057] | +0.03666 [+0.02649, +0.04653] |
| 概率加权折扣价值变化 | +0.02907 [+0.02272, +0.03535] | +0.02665 [+0.01816, +0.03455] |
| top-1 折扣价值变化 | +0.08488 [+0.05864, +0.11078] | +0.07592 [+0.05191, +0.10100] |
| receiver−donor 平均折扣价值 | +0.08302 [+0.06397, +0.10253] | +0.08521 [+0.05749, +0.11284] |
| 价值最优动作上的概率变化 | +0.03918 [+0.02098, +0.05986] | +0.03377 [+0.00710, +0.05213] |

这说明 SEED 的确发生了统计可靠的价值导向概率质量转移，而且在 unseen 上也复现。但提升幅度只
能支持“存在机制信号”，不能推导出强价值选择能力。

## 5. 为什么不能写成“logit 按价值排序”

| 状态内相关 | valid_seen [episode 95% CI] | valid_unseen [episode 95% CI] |
|---|---:|---:|
| `delta logit` × 折扣价值 Spearman | -0.06183 [-0.14577, +0.01335] | -0.05537 [-0.11083, -0.00573] |
| `delta probability` × 折扣价值 Spearman | +0.04997 [-0.02513, +0.11851] | +0.08980 [+0.04959, +0.13053] |

softmax 归一化后的概率变化比原始 score/logit delta 更接近价值方向；但相关仍弱。当前证据支持的
是“少数 donor/receiver 之间的选择性搬运”，不支持“所有候选动作按长期价值单调重排”。

## 6. 严格结论

1. 官方 expert 下一步本身不是可靠的长期价值标签：它在 seen/unseen 也只有 40.77%/36.62%
   属于最高折扣价值动作。
2. SEED 在两个 split 上都把一部分概率质量移向了更高 expert-recovery 价值的动作，且 episode
   bootstrap 区间为正。
3. SEED top-1 的价值最优率只有 43.02%/38.33%，显著低于 80% 实用门槛，因此本实验结果
   **不具备实际用途，只是弱但可复现的内部机制证据**。
4. 下一阶段只研究这些变化在动作词、物体词、容器/位置词和索引 token 上如何发生；在逐 token
   结构确定以前，不进入参数梯度，更不设计新算法。

## 7. 可复查产物

- 机器可读结果：`outputs/action_value_alignment/analysis/results.json`
- 逐状态、逐候选动作对齐结果：`outputs/action_value_alignment/analysis/trace.jsonl`
- 自动摘要：`outputs/action_value_alignment/analysis/REPORT.md`
- 原始 32 分片：`outputs/action_value_alignment/{valid_seen,valid_unseen}_shard_*/trace.jsonl`
- planner 稳健性对照：`outputs/action_value_alignment_planner/`
- 环境干预脚本：`scripts/evaluate_alfworld_action_values.py`
- 分析脚本：`scripts/analyze_action_value_alignment.py`
- 完整交接：`research/ACTION_VALUE_EXPERIMENT_HANDOFF_ZH.md`

