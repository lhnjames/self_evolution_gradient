# ALFWorld evolved-skill 与 logic distribution repair 实验报告

> 完整合并版见：[`ALFWORLD_EVOLVED_SKILL_COMPLETE_REPORT_ZH.md`](ALFWORLD_EVOLVED_SKILL_COMPLETE_REPORT_ZH.md)

更新时间：2026-08-31

## 1. 结论先行

直接把 SkillRL 发布的 evolved skills 拼进 prompt 并不可靠。Qwen2.5-3B-Instruct 在 444 个
ALFWorld valid_seen 决策上的 top-1 从 41.22% 降到 39.19%，在 527 个 valid_unseen
决策上从 31.69% 降到 30.36%；错配 task skill 更差。skill 有信号，但它产生的是明显双向、
高方差的 logit shift，不能无条件写回策略。

有效方案是把 skill 当作 proposal direction，而不是答案：冻结 LLM，计算无 skill 与有 skill
的候选命令序列分数之差，只训练一个 175 参数、动作阶段感知的 residual head，并用
`CE + lambda * KL(q || p0)` 约束更新。lambda 只在 train 内部 episode-level dev 上选择。

最终的 constrained skill-aware head：

| split | plain top-1 | direct skill | stage-only constrained | skill-aware constrained | mean KL |
|---|---:|---:|---:|---:|---:|
| valid_seen（444） | 41.22% | 39.19% | 44.82% | **46.62%** | 0.0366 |
| valid_unseen（527） | 31.69% | 30.36% | **35.67%** | 35.29% | 0.0303 |

seen 上 skill 相对纯阶段校准额外增加 1.80 点；unseen 上 top-1 没有额外收益（-0.38 点），
但 skill-aware 的 NLL 优于 stage-only（2.307 vs 2.358）。这意味着当前 skill delta 在同分布
有效，OOD 时更适合作为弱证据，不能主导动作。

## 2. 数据与无泄漏设计

- 官方 ALFWorld TextWorld 数据与 hand-coded expert；候选集来自每一步真实
  `admissible_commands`。
- train：47 个成功 episode，598 个非平凡决策；内部按 episode 切为 36 train / 9 dev。
- valid_seen：33 个成功 episode，444 个非平凡决策。
- valid_unseen：34 个成功 episode，527 个非平凡决策。
- 去掉启动时的 `look`、`inventory`、`help` gold action，但这些命令仍可留在候选集合中。
- skill retrieval 只读目标描述和 task type；prompt/head 均不读取 expert action。
- expert label 只用于 train/dev 的交叉熵与最终离线评测；valid_seen/unseen 不参与参数或 lambda 选择。
- 候选动作是任意多 token 命令。使用长度归一化序列 log-prob 构造候选集合上的 categorical
  distribution，而不是把第一个 token 当成整个动作。

8 卡并行完成了 1569 个状态的 plain、正确 task skill、错配 task skill 评分。通过只计算候选
completion 位置的词表 logits，显存新增开销由最高约 64 GB 降至约 7--8 GB，和完整 logits
实现的最大数值误差为 `1.2e-7`。

## 3. 分布修复公式与梯度路径

对状态 `s` 的候选命令集合 `A(s)`：

```text
z0_i = length_normalized_logp(action_i | plain_prompt)
zs_i = length_normalized_logp(action_i | skill_prompt)
d_i  = center(zs_i - z0_i)

x_s = [H(p0), H(ps), max(p0), max(ps), KL(ps||p0), mean|d|, log(1+|A|)]
alpha_s = 2 * tanh(MLP(x_s))

zq_i = exp(tau) * center(z0_i)
       + alpha_s * d_i
       + verb_bias[verb_i]
       + verb_delta[verb_i] * d_i
       + length_weight * centered_length_i
q = softmax(zq)
```

参数仅 175 个。`verb_i` 近似动作阶段（`go/open/take/move/clean/heat/...`），因此同一条 task
skill 不会再对导航和操作阶段使用同一个全局系数。

训练目标：

```text
L(phi) = CE(q_phi, verified_action) + lambda * KL(q_phi || p0)
```

lambda 候选为 1、3、10；只在 train-dev 上选择满足平均 KL <= 0.05 且 NLL 最低的模型，
最终选择 lambda=3。梯度只流入 residual head；冻结 LLM、skill 文本和 skill delta 都不反传。

如果没有 expert action，可把 `verified_action` 替换为：成功 rollout 中通过 counterfactual replay
确认的动作、deterministic verifier 的候选 value，或 outcome preference。只有在模型/API/环境
整体不可微且只能得到标量 rollout loss 时才使用 ZO/SPSA；能取得 logits 时，一阶 head 更新更
省样本、更稳定。ZO 应处于外层，优化 retrieval/gate/系统超参或 API-only agent，而不是替代
本来可用的一阶输出梯度。

## 4. 完整离线结果

### valid_seen

| 方法 | top-1 | p(expert) | NLL | KL from plain |
|---|---:|---:|---:|---:|
| plain | 41.22% | 0.2291 | 2.3499 | 0 |
| evolved skill prompt | 39.19% | 0.1840 | 2.2542 | 0.2243 |
| mismatched skill | 33.11% | 0.1621 | 2.4215 | 0.2167 |
| constrained stage-only | 44.82% | 0.2556 | 2.0680 | 0.0293 |
| constrained skill-aware | **46.62%** | **0.2587** | **2.0191** | 0.0366 |
| unconstrained skill-aware | 52.25% | 0.3532 | 1.7225 | 0.4765 |

unconstrained 结果说明方向有能力，但 KL 过大，不应作为部署结果。

### valid_unseen

| 方法 | top-1 | p(expert) | NLL | KL from plain |
|---|---:|---:|---:|---:|
| plain | 31.69% | 0.1770 | 2.5787 | 0 |
| evolved skill prompt | 30.36% | 0.1549 | 2.4314 | 0.2054 |
| mismatched skill | 26.38% | 0.1400 | 2.5752 | 0.1773 |
| constrained stage-only | **35.67%** | 0.1976 | 2.3575 | 0.0234 |
| constrained skill-aware | 35.29% | **0.2019** | **2.3068** | 0.0303 |

### episode-cluster bootstrap（10,000 次）

| 对比 | seen delta [95% CI] | unseen delta [95% CI] |
|---|---:|---:|
| direct skill - plain | -2.03 [-6.20, +2.29] | -1.33 [-5.23, +2.60] |
| constrained stage - plain | +3.60 [+1.33, +5.75] | +3.98 [+2.03, +5.86] |
| constrained skill - plain | **+5.41 [+3.05, +7.53]** | **+3.61 [+1.93, +5.23]** |
| constrained skill - stage | +1.80 [+0.39, +3.42] | -0.38 [-1.40, +0.71] |

bootstrap 以 episode 为抽样单位，避免把同一轨迹的相关步骤当独立样本。

## 5. 小规模在线 rollout

同一 5 个任务、每步最多 50 步：

| policy | valid_seen success | valid_unseen success |
|---|---:|---:|
| plain | 2/5 | 0/5 |
| direct evolved skill | 2/5 | 0/5 |
| constrained stage | 1/5 | 0/5 |
| constrained skill-aware | 2/5 | 0/5 |
| constrained skill + cycle repair | **3/5** | **1/5** |

仅靠离线 head，next-action 提升没有稳定转化为 endpoint success。失败轨迹的核心问题是确定性
多步循环：同一 `(observation, action)` 在单个失败 episode 中重复 6--45 次。官方 expert
绝大多数 episode 不重复该 pair，重复时通常不超过 2--3 次。

因此加入无标签的 system repair：同一 `(observation, action)` 已执行两次后，把该动作 logit
置为负无穷；若所有候选都被抑制则回退原分布。它把 SkillRL 的“avoid loops”文本原则落实成
可执行约束，修复了一个 seen 的 `take/move` 循环和一个 unseen 的反复 `examine` 循环。

在线样本只有 5 个/split，不能作统计结论；它只证明“skill proposal + learned constrained head +
stateful system repair”能产生 plain 没有的闭环成功。

## 6. 对自进化系统的最终建议

采用三层而非单层 skill prompt：

1. **外层 skill evolution**：从成功/失败轨迹产生、合并、淘汰 skill；skill 只生成 proposal
   direction `d_skill`，不直接晋升为策略。
2. **内层 gradient internalization**：对 verified train traces 用一阶梯度训练小型、stage-aware
   distribution head；以 KL-constrained dev promotion 决定是否替换 champion。
3. **在线 stateful repair**：循环检测、硬约束、工具 schema、预算等确定性规则直接作用于
   candidate logits；这些系统状态不能仅依赖 prompt 中的自然语言提醒。

promotion 必须同时满足：

- episode-cluster CI 的下界大于 0；
- valid_seen 与 valid_unseen 都不回退；
- critical task/verb slice 不发生显著退化；
- KL、token cost、wall-clock cost 在预算内；
- 在线 rollout success 或 cycle/progress 指标改善。

下一步优先级：

1. 将在线评测扩大到 seen/unseen 各 50--100 episode，确认 cycle repair 的成功率增益。
2. 把二元 cycle mask 扩展为显式 progress state：持有物、已访问容器、对象 transformation、目标容器。
3. 用失败轨迹自进化出结构化 repair operator，而不是更长的文本 skill。
4. 对 skill-aware head 做多 seed 与 task-balanced loss；OOD gate 在证据不足时退回 stage-only。
5. 在 SkillsBench deterministic verifier 子集复用同一套 `p0/ps/d_skill/q/KL/promotion` 框架。

## 7. 复现入口

- 数据采集：`scripts/collect_alfworld_expert.py`
- 多 token scorer：`src/self_evolve/sequence_scorer.py`
- skill retrieval/prompt：`src/self_evolve/alfworld_skills.py`
- 离线分布评测：`src/self_evolve/alfworld_runner.py`
- gradient repair head：`src/self_evolve/logic_repair_head.py`
- 在线 rollout/cycle repair：`src/self_evolve/alfworld_online.py`
- 最终配置：`config/logic_repair_head_large_3b.yaml`、`config/alfworld_online_3b.yaml`
- 最终离线结果：`outputs/logic_repair_head_large_3b/results.json`
- bootstrap：`outputs/logic_repair_head_large_3b/bootstrap.json`
- 在线结果：`outputs/alfworld_online_3b/*/results.json`

所有单元测试：`10 passed`。
