# ALFWorld 候选动作长期价值 × mean12_oracle_skill_seed_20260955 输出变化

价值定义：在记录状态强制执行一个候选动作，随后由 ALFWorld 官方 expert 恢复；
在总计 50 步预算内成功记为 1，同时报告 `0.95^(恢复步数-1)` 的折扣成功值。
该实验测量的是 expert-recovery 条件价值，不等同于 Base/SEED 自身 rollout 价值。

## valid_seen

- 444 个状态、33 个独立 game trial、12680 个候选动作。
- 候选动作在预算内的总体恢复成功率：94.09%。
- 有二元成功差异的状态：141；有折扣价值差异的状态：432。
- expert 动作 / Base top-1 / mean12_oracle_skill_seed_20260955 top-1 为最高价值动作的比例：40.77% / 31.31% / 44.59%。
- 80% 实用门槛：未通过（差值 -35.41 点）。
- mean12_oracle_skill_seed_20260955 概率加权折扣价值变化均值：+0.056592。
- Base / mean12_oracle_skill_seed_20260955 概率加权折扣价值：0.595442 / 0.652034；相对提升 9.50%（30% 硬门槛：未通过）。
- 最高价值动作命中率相对提升：42.45%。
- episode-cluster 95% CI：[+0.035132, +0.076059]。
- mean12_oracle_skill_seed_20260955 Δlogit 与动作价值的状态内 Spearman 均值：+0.309508。

## valid_unseen

- 527 个状态、34 个独立 game trial、14980 个候选动作。
- 候选动作在预算内的总体恢复成功率：92.84%。
- 有二元成功差异的状态：207；有折扣价值差异的状态：513。
- expert 动作 / Base top-1 / mean12_oracle_skill_seed_20260955 top-1 为最高价值动作的比例：36.62% / 31.88% / 35.10%。
- 80% 实用门槛：未通过（差值 -44.90 点）。
- mean12_oracle_skill_seed_20260955 概率加权折扣价值变化均值：+0.046806。
- Base / mean12_oracle_skill_seed_20260955 概率加权折扣价值：0.569996 / 0.616802；相对提升 8.21%（30% 硬门槛：未通过）。
- 最高价值动作命中率相对提升：10.12%。
- episode-cluster 95% CI：[+0.033163, +0.058603]。
- mean12_oracle_skill_seed_20260955 Δlogit 与动作价值的状态内 Spearman 均值：+0.307627。

