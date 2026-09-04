# ALFWorld 有价值输出变化的逐 token / 全词表数值结构

更新时间：2026-09-03（Asia/Shanghai）  
状态：971 个状态全量完成、分析完成并归档

## 1. 实验回答什么

上一阶段已经确认：SEED 的概率加权 expert-recovery 折扣价值在 seen/unseen 上分别增加
`+0.02907/+0.02665`，但 top-1 价值最优率只有 `43.02%/38.33%`，远低于 80% 实用门槛。

本实验不设计新的分类器或更新算法，只进一步回答：这些数值变化发生在动作命令的哪些生成位置，
全词表概率质量从哪些 token 移到哪些 token，以及变化幅度大的位置是否也携带长期价值信号。

## 2. 数值协议与规模

- Base 与 SEED 使用同一个 plain prompt、Base tokenizer、BF16；
- 严格复用上一阶段 scorer 的 batch size 4、右填充和 completion-position 语义；
- 每个候选动作保存逐 token 的 raw logit、log probability、probability 和 Base→SEED delta；
- 每个唯一候选前缀保存全词表 total variation、top-12 donor/receiver、所有 admissible next token 的
  raw/条件概率及其后续候选动作价值；
- 语义角色分为 action verb、object、location、receptacle、appliance、relation、index、separator；
- 每个动作的 normalized-score delta 被逐角色精确重构。

| split | 状态 | trial | 候选动作 | 动作 token | 唯一前缀节点 | prompt 截断 |
|---|---:|---:|---:|---:|---:|---:|
| valid_seen | 444 | 33 | 12,680 | 67,053 | 19,981 | 0 |
| valid_unseen | 527 | 34 | 14,980 | 77,517 | 22,948 | 0 |

阶段一动作分数的最大数值复现误差：seen Base/SEED `9.78e-7/8.34e-7`，unseen
`9.54e-7/9.54e-7`。

## 3. 变化幅度主要落在哪里

下表按每个动作的长度归一化 score delta 做语义角色精确分解，再统计绝对贡献占比：

| role | valid_seen | valid_unseen |
|---|---:|---:|
| action verb | 37.01% | 39.18% |
| instance index | 33.29% | 31.44% |
| location | 28.64% | 28.02% |
| object | 1.00% | 1.26% |
| receptacle | 0.06% | 0.10% |
| 其余角色合计 | 0.01% | <0.01% |

因此，近 99% 的动作 score 变化集中在首个动作词、导航位置词和实例编号。这个占比描述数值变化
落点，不等同于因果解释，也不代表相应变化都有价值。

## 4. “变化大”与“有价值”并不相同

在每个唯一前缀上，把 SEED 对 admissible next-token 的条件概率变化与该分支可达动作的平均
折扣价值相乘，可得到描述性的 branch-value delta：

| next-token role | seen branch-value delta [episode 95% CI] | unseen [episode 95% CI] |
|---|---:|---:|
| action verb | **+0.0520 [+0.0356, +0.0690]** | **+0.0641 [+0.0449, +0.0862]** |
| location | **+0.0316 [+0.0173, +0.0464]** | **+0.0119 [+0.0046, +0.0199]** |
| index | +0.0001 [-0.0061, +0.0065] | -0.0008 [-0.0046, +0.0030] |
| object | +0.0002 [-0.0013, +0.0022] | +0.0007 [-0.0003, +0.0019] |

最关键的区分是：

- 动作词和位置词既承载大幅变化，也在 seen/unseen 都呈现正的分支价值移动；
- index 承载约三分之一的绝对 score 变化，但价值增益约为 0；其状态内 role-delta × value
  Spearman 在 seen/unseen 反而为 `-0.1565/-0.0992`；
- relation/separator 通常是候选前缀确定后的单一路径，几乎没有选择自由度；
- receptacle 多数节点也只有一个 admissible child（seen 仅 1 个、unseen 10 个 multi-child），
  因此当前数据不足以据此评价容器选择价值。

这给出本阶段最清晰的机制结论：

```text
SEED 的大部分数值变化 != 有价值的数值变化
稳定的价值信号主要出现在“先选哪类动作”和“导航到哪里”，
而实例编号虽变化很大，却没有形成可复现的长期价值增益。
```

## 5. 全词表层面的变化

| next-token role | seen vocab TV | unseen vocab TV | seen admissible mass Δ | unseen admissible mass Δ |
|---|---:|---:|---:|---:|
| action verb | 0.2025 | 0.2316 | +0.0139 | +0.0047 |
| location | 0.0491 | 0.0512 | +0.0092 | +0.0091 |
| index | 0.1094 | 0.1032 | +0.0135 | +0.0023 |
| object | 0.1018 | 0.1197 | -0.0757 | -0.0810 |

首 token 的全词表分布改变最大，并把少量概率质量推入当前 admissible action verbs；位置节点也在
两个 split 上稳定增加 admissible location mass。object 节点的 raw admissible mass 减少不能直接
解释成价值受损，因为该统计包含大量只有一个候选 child 的前缀，且其 branch-value delta 接近 0。

## 6. 结论边界与下一步

1. 本实验成功把动作级变化拆到模型实际生成位置，并数值复现上一阶段结果。
2. action verb 与 location 是下一阶段“价值梯度”的优先机制位置；index 是必要的高变化对照。
3. 当前 branch value 仍是 expert-recovery proxy，且多子分支使用分支内候选动作平均值，只能做
   机制描述，不能当成模型 rollout Q 值或因果效应。
4. 本实验没有产生达到 80% 的价值选择器；第一阶段的“不具备实际用途”结论保持不变。
5. 下一阶段应计算以长期价值为目标的 Base/SEED 参数梯度，并把 expert-NLL 梯度仅作为 control；
   不应继续使用旧的“expert action 即 verified target”定义。

## 7. 可复查产物

- 自动报告：`outputs/token_value_alignment/analysis/REPORT.md`
- 机器可读汇总：`outputs/token_value_alignment/analysis/results.json`
- 状态级角色/前缀统计：`outputs/token_value_alignment/analysis/trace.jsonl`
- 全量逐 token/full-vocabulary trace：
  `outputs/token_value_alignment/{valid_seen,valid_unseen}_shard_{0..3}/trace.jsonl`
- 探针：`scripts/probe_action_token_logits.py`
- 分析器：`scripts/analyze_token_value_alignment.py`
- 语义角色与前缀工具：`src/self_evolve/token_value.py`

