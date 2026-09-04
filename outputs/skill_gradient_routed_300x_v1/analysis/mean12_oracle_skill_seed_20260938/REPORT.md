# ALFWorld 候选动作长期价值 × mean12_oracle_skill_seed_20260938 输出变化

价值定义：在记录状态强制执行一个候选动作，随后由 ALFWorld 官方 expert 恢复；
在总计 50 步预算内成功记为 1，同时报告 `0.95^(恢复步数-1)` 的折扣成功值。
该实验测量的是 expert-recovery 条件价值，不等同于 Base/SEED 自身 rollout 价值。

## valid_seen

- 444 个状态、33 个独立 game trial、12680 个候选动作。
- 候选动作在预算内的总体恢复成功率：94.09%。
- 有二元成功差异的状态：141；有折扣价值差异的状态：432。
- expert 动作 / Base top-1 / mean12_oracle_skill_seed_20260938 top-1 为最高价值动作的比例：40.77% / 31.31% / 52.93%。
- 80% 实用门槛：未通过（差值 -27.07 点）。
- mean12_oracle_skill_seed_20260938 概率加权折扣价值变化均值：+0.066097。
- Base / mean12_oracle_skill_seed_20260938 概率加权折扣价值：0.595442 / 0.661539；相对提升 11.10%（30% 硬门槛：未通过）。
- 最高价值动作命中率相对提升：69.06%。
- episode-cluster 95% CI：[+0.043349, +0.085594]。
- mean12_oracle_skill_seed_20260938 Δlogit 与动作价值的状态内 Spearman 均值：+0.309447。

## valid_unseen

- 527 个状态、34 个独立 game trial、14980 个候选动作。
- 候选动作在预算内的总体恢复成功率：92.84%。
- 有二元成功差异的状态：207；有折扣价值差异的状态：513。
- expert 动作 / Base top-1 / mean12_oracle_skill_seed_20260938 top-1 为最高价值动作的比例：36.62% / 31.88% / 37.38%。
- 80% 实用门槛：未通过（差值 -42.62 点）。
- mean12_oracle_skill_seed_20260938 概率加权折扣价值变化均值：+0.049369。
- Base / mean12_oracle_skill_seed_20260938 概率加权折扣价值：0.569996 / 0.619365；相对提升 8.66%（30% 硬门槛：未通过）。
- 最高价值动作命中率相对提升：17.26%。
- episode-cluster 95% CI：[+0.036686, +0.061112]。
- mean12_oracle_skill_seed_20260938 Δlogit 与动作价值的状态内 Spearman 均值：+0.301804。

