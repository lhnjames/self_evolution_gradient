# ALFWorld 候选动作长期价值 × mean12_oracle_skill_seed_20260921 输出变化

价值定义：在记录状态强制执行一个候选动作，随后由 ALFWorld 官方 expert 恢复；
在总计 50 步预算内成功记为 1，同时报告 `0.95^(恢复步数-1)` 的折扣成功值。
该实验测量的是 expert-recovery 条件价值，不等同于 Base/SEED 自身 rollout 价值。

## valid_seen

- 444 个状态、33 个独立 game trial、12680 个候选动作。
- 候选动作在预算内的总体恢复成功率：94.09%。
- 有二元成功差异的状态：141；有折扣价值差异的状态：432。
- expert 动作 / Base top-1 / mean12_oracle_skill_seed_20260921 top-1 为最高价值动作的比例：40.77% / 31.31% / 52.25%。
- 80% 实用门槛：未通过（差值 -27.75 点）。
- mean12_oracle_skill_seed_20260921 概率加权折扣价值变化均值：+0.061506。
- Base / mean12_oracle_skill_seed_20260921 概率加权折扣价值：0.595442 / 0.656948；相对提升 10.33%（30% 硬门槛：未通过）。
- 最高价值动作命中率相对提升：66.91%。
- episode-cluster 95% CI：[+0.040171, +0.080808]。
- mean12_oracle_skill_seed_20260921 Δlogit 与动作价值的状态内 Spearman 均值：+0.317316。

## valid_unseen

- 527 个状态、34 个独立 game trial、14980 个候选动作。
- 候选动作在预算内的总体恢复成功率：92.84%。
- 有二元成功差异的状态：207；有折扣价值差异的状态：513。
- expert 动作 / Base top-1 / mean12_oracle_skill_seed_20260921 top-1 为最高价值动作的比例：36.62% / 31.88% / 27.89%。
- 80% 实用门槛：未通过（差值 -52.11 点）。
- mean12_oracle_skill_seed_20260921 概率加权折扣价值变化均值：+0.031129。
- Base / mean12_oracle_skill_seed_20260921 概率加权折扣价值：0.569996 / 0.601125；相对提升 5.46%（30% 硬门槛：未通过）。
- 最高价值动作命中率相对提升：-12.50%。
- episode-cluster 95% CI：[+0.020633, +0.039478]。
- mean12_oracle_skill_seed_20260921 Δlogit 与动作价值的状态内 Spearman 均值：+0.259482。

