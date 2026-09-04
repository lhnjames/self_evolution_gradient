# ALFWorld evolved skill、logic distribution repair 与参数梯度共性实验总报告

> 2026-09-01 新增：[`ALFWORLD_SEED_CHECKPOINT_OUTPUT_DELTA_ZH.md`](ALFWORLD_SEED_CHECKPOINT_OUTPUT_DELTA_ZH.md)，
> 完成原始 Qwen 与官方 SEED-ALFWorld-3B 在同一 971 个专家状态上的权重前后输出对照。

更新时间：2026-09-01  
模型：Qwen2.5-3B-Instruct  
任务：ALFWorld TextWorld  
报告性质：将离线分布评测、repair head、在线 rollout、共性诊断、语义对照与真实参数梯度观测合并为一份完整记录

## 摘要

本项目追问一个具体问题：一条自然语言经验能否先经过候选动作输出分布，形成可复用、可验证、
可积累的参数编辑方向？

完整实验给出的答案是：**当前 SkillRL evolved skill 含有统计可见的 task-specific 信息，但尚未
形成可无条件写回策略或 backbone 的可靠梯度方向。**

主要证据如下：

1. 直接把 evolved skill 拼入 prompt 会使 top-1 下降：valid_seen 从 41.22% 降到 39.19%，
   valid_unseen 从 31.69% 降到 30.36%。
2. 正确 skill 相对 mismatched skill 和严格等长 placebo 仍有显著 verified margin 优势，说明
   文本不是完全无效；task-only 也 consistently 优于 general-only。
3. skill shift 高方差且双向。输出空间中只有 53.15% 的 seen 状态、55.60% 的 unseen 状态
   与 expert 方向正对齐。
4. anti-skill 没有使分布方向反转；它与原 skill 的逐状态 effect 相关仍约 0.91。这说明模型大量
   响应主题词、模板和 candidate schema，而没有可靠执行自然语言规则的逻辑极性。
5. 在 149,504 维 RMSNorm 参数子空间中，evolved skill 梯度与 verified gradient 的平均 cosine
   在 seen/unseen 分别为 -0.113/-0.099，置信区间均跨 0，正对齐率都只有 46.88%。
6. 真正重复出现的是 task/stage 条件结构：`go/close` 较常正对齐，`take/open/move` 较常负对齐；
   但错误 skill、anti-skill 和 placebo 也共享大量相同结构，所以几何稳定不等于经验真实。
7. 一个带 verifier 的 175 参数、stage-aware、KL-constrained residual head 可以利用这些弱信号：
   episode 修正后的五个 seed 中，seen 相对 stage-only 平均增加 1.53±0.58 top-1 点；unseen
   top-1 平均下降 0.38±0.23 点，但 NLL 稳定改善约 0.052。
8. 小规模在线 rollout 表明 next-action 提升不会自动转化为 endpoint success；确定性循环修复
   比继续加长文本 skill 更直接。
9. 官方 SEED-ALFWorld-3B 在同一 direct-action prompt 上把 expert top-1 提高到 seen 44.59%、
   unseen 35.86%，但 NLL 同时变差。其 rescue 高度集中在最小修复距离最近的第一四分位，说明
   成功自进化并非整体复制 expert 分布，而是改变少数接近边界的关键决策。

因此，本实验发现的是“**可测量但不充分可信的 proposal direction**”，不是已经成立的
“经验参数记忆”。

## 1. 研究问题与证据标准

### 1.1 核心问题

自然语言经验 `skill` 对同一状态的候选动作分布产生变化：

```text
plain prompt  -> p0(action | state)
skill prompt  -> ps(action | state)
d_skill       = center(zs - z0)
```

需要区分四个逐级变强的命题：

1. **有影响**：skill 会改变输出分布。
2. **有任务信息**：正确 skill 比 mismatch/placebo 更接近 verified action。
3. **有可复用共性**：方向在独立 episode、task/stage 和 held-out split 上重复出现。
4. **可参数化积累**：该方向映射到模型参数梯度后仍与 verified gradient 对齐，且实际编辑不会
   造成跨状态回退。

本实验完成了前三层和第四层的 RMSNorm 梯度观测，但没有真正执行 backbone 参数编辑。因此对
第四层只能给出梯度证据，不能给出长期更新后的因果结论。

### 1.2 “经验真正有效”的最低证据

本报告不把以下现象单独视为经验有效：

- prompt 加长后 top-1 偶然变化；
- 输出 direction 在同模板状态间 cosine 很高；
- NLL 改善但 top-1、rollout progress 或 success 不改善；
- 正确 skill 只比 plain 好，却没有 mismatch、anti 和等长 placebo 对照；
- 参数梯度只在少量案例中对齐，却没有 episode-level 不确定性。

更可信的证据应至少同时包含：正确语义相对 controls 的 paired 优势、语义极性测试、独立 episode
bootstrap、seen/unseen 一致性，以及参数空间中的 verified alignment。

## 2. 数据、候选动作与无泄漏设计

### 2.1 数据规模

| split | 成功 episode | 非平凡决策 | 用途 |
|---|---:|---:|---|
| train | 47 | 598 | head 训练与内部 episode-level dev |
| valid_seen | 33 | 444 | held-out 离线评测 |
| valid_unseen | 34 | 527 | held-out OOD 离线评测 |
| 合计 | 114 | 1569 | plain/correct/mismatch 缓存评分 |

语义对照覆盖全部 971 个 valid_seen/unseen 状态；真实参数梯度 probe 从每个 split 按
`task_type × action_verb` 平衡抽取 64 个状态，共 128 个。

### 2.2 候选与 label

- 数据来自官方 ALFWorld TextWorld 与 hand-coded expert。
- 每步候选集是环境返回的真实 `admissible_commands`。
- 去掉启动阶段以 `look`、`inventory`、`help` 为 gold 的平凡决策，但这些命令仍可留在候选集。
- skill retrieval 只读取目标描述和 task type，不读取 expert action。
- expert label 只用于 train/dev CE、离线 verified 对齐和最终评测。
- valid_seen/unseen 不参与 head 参数或 `lambda` 的选择。
- bootstrap 以 episode 为抽样单位，不把同轨迹相关步骤当成独立样本。

### 2.3 多 token action scorer

候选动作可能包含任意多个 token，例如 `take apple 1 from fridge 1`。每个候选使用 completion
位置 token log-prob 的长度归一化值：

```text
z_i = (1 / |a_i|) * sum_t log p(a_i,t | prompt, a_i,<t)
p_i = softmax(z_i over A(state))
```

这避免了把第一个 token 当作整个动作。只计算 completion 位置的词表 logits 后，显存新增开销
从最高约 64 GB 降到约 7--8 GB；与完整 logits 实现的最大误差为 `1.2e-7`。

### 2.4 episode key 修正

早期 trace 的 `episode_id` 省略了 `trial_*`，导致同一任务目录下的多个 gamefile 被错误合并：

| split | 旧 episode_id 数 | 真实 gamefile 数 |
|---|---:|---:|
| train | 45 | 47 |
| valid_seen | 32 | 33 |
| valid_unseen | 25 | 34 |

修正后统一使用完整 `gamefile` 作为 `episode_key`，并同步修改 trace 写出、head episode split 和
bootstrap。该问题没有造成 train/valid 泄漏，但会改变内部 train/dev 切分和置信区间，旧单 seed
seen 结果因此略偏乐观。

## 3. 直接 skill prompt 的完整离线结果

### 3.1 总体结果

| split | plain top-1 | evolved skill | mismatch | skill margin gain | expert log-p gain | output cosine | cosine > 0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 38.80% | 36.12% | 32.44% | +0.0993 | -0.0096 | +0.0053 | 48.49% |
| valid_seen | 41.22% | 39.19% | 33.11% | +0.1469 | +0.0957 | +0.0191 | 53.15% |
| valid_unseen | 31.69% | 30.36% | 26.38% | +0.2796 | +0.1473 | +0.0283 | 55.60% |

正确 skill 比 mismatch 的 margin gain 高 0.1798（seen）和 0.1870（unseen）；按真实 episode
bootstrap，其 95% CI 分别为 `[+0.0862, +0.2791]` 和 `[+0.1096, +0.2881]`。因此正确
task skill 中确实存在任务信息，不能全部归因于加入一段长文本。

但它在单状态上并不可靠：

- seen：rescue 4.73%，harm 6.76%；
- unseen：rescue 5.88%，harm 7.21%；
- 两个 split 的 mean cosine 都接近 0；
- full-strength direct prompt 的 top-1 均低于 plain。

### 3.2 修正后的 episode-cluster bootstrap

| split | metric | observed | 95% CI |
|---|---|---:|---:|
| seen | direct top-1 delta | -2.03 点 | [-6.22, +2.28] |
| seen | expert log-p gain | +0.0957 | [-0.0675, +0.3063] |
| seen | margin gain | +0.1469 | [-0.0865, +0.4360] |
| seen | output-gradient cosine | +0.0191 | [-0.0118, +0.0585] |
| unseen | direct top-1 delta | -1.33 点 | [-4.94, +2.69] |
| unseen | expert log-p gain | +0.1473 | [+0.0233, +0.3008] |
| unseen | margin gain | +0.2796 | [+0.0961, +0.4958] |
| unseen | output-gradient cosine | +0.0283 | [+0.0070, +0.0552] |

unseen 的 margin/NLL 信号为正而 top-1 下降，说明 skill 能改善 expert 相对多数错误动作的排序，
同时也可能把少数竞争动作推得更高，或把变化做得过强。

### 3.3 输出一阶观测是否可信

定义 plain 分布下 expert CE 的负梯度方向：

```text
u_verified = onehot(expert) - p0
d_skill    = center(z_skill - z_plain)
```

`dot(u_verified, d_skill)` 与真实 expert log-p gain 的相关性很高：

| split | Pearson | Spearman |
|---|---:|---:|
| seen | 0.9562 | 0.9839 |
| unseen | 0.9525 | 0.9867 |

因此输出空间的一阶计算能准确回答“skill prompt 会怎样推动当前分布”。主要问题不是估计误差，
而是 skill direction 本身的正确率和语义纯度不够。

## 4. Skill dose-response：方向有信息，但 full strength 过强

只沿缓存的 `d_skill` 扫描系数，不训练模型，也不用 valid label 选择部署系数：

| split | alpha | top-1 | NLL | mean KL |
|---|---:|---:|---:|---:|
| seen | 0.00 | 41.22% | 2.3499 | 0 |
| seen | 0.25 | 40.99% | 2.2831 | 0.0147 |
| seen | 0.50 | **43.02%** | 2.2454 | 0.0579 |
| seen | 0.75 | 42.57% | **2.2360** | 0.1272 |
| seen | 1.00 | 39.19% | 2.2542 | 0.2243 |
| unseen | 0.00 | 31.69% | 2.5787 | 0 |
| unseen | 0.25 | 31.12% | 2.5038 | 0.0128 |
| unseen | 0.50 | **32.64%** | 2.4545 | 0.0509 |
| unseen | 0.75 | 31.50% | **2.4303** | 0.1131 |
| unseen | 1.00 | 30.36% | 2.4314 | 0.2054 |

两个 split 都呈倒 U 形。direct skill 失败不能解释为 skill 完全没有信息，更准确的说法是：方向
包含弱信号，但 prompt 产生的完整步长越过了较好的局部区域。

## 5. 输出分布中的共性

### 5.1 task 与 action stage 同时决定符号

按 expert verb 切片：

| verb | seen margin gain / top-1 delta | unseen margin gain / top-1 delta |
|---|---:|---:|
| go | +0.301 / -3.78 点 | +0.454 / +0.30 点 |
| close | +0.941 / +1.79 点 | +0.855 / 0 点 |
| open | -0.729 / -6.45 点 | -0.684 / -20.00 点 |
| move | -0.652 / +3.23 点 | -0.515 / 0 点 |
| take | +0.044 / 0 点 | -0.087 / 0 点 |

verb 也不是完整规律。例如 `look_at_obj_in_light/open` 的 margin gain 在 seen/unseen 为
`+1.136/+0.657`，而 clean、heat、cool 等 task 的 open 阶段稳定为负。最小重复单元更接近
`task family × action stage`，不是 task skill 的单一全局方向。

### 5.2 pairwise 几何

把候选概率变化投影到公共 verb simplex 后，不同 episode 的两两 cosine：

| relation | seen correct skill | unseen correct skill |
|---|---:|---:|
| same task, same stage | 0.3100 | 0.2243 |
| different task, same stage | 0.1595 | 0.1095 |
| same task, different stage | 0.0947 | 0.0681 |
| different task, different stage | 0.0368 | 0.0051 |

task 和 stage 都贡献共性，二者同时相同时方向最一致。但 mismatched skill 的分组 cosine 也很
接近，甚至有时更高。task×stage prototype 的跨 split cosine 同样不能把正确经验与错配经验
彻底分离：

| comparison | correct skill | mismatched skill |
|---|---:|---:|
| train ↔ seen | 0.8814 | 0.8624 |
| train ↔ unseen | 0.8043 | 0.8024 |
| seen ↔ unseen | 0.7913 | 0.7419 |

所以“结构稳定”本身不是经验真实性证明。它可能来自同一模型、prompt template、candidate schema
和任务语言的共同结构。

### 5.3 logic/stage 概率泄漏

正确 skill 对 gold verb 的总体概率变化在 train/seen/unseen 分别为
`-0.0308/-0.0360/-0.0253`，verb-space gradient cosine 为
`-0.2437/-0.2507/-0.2268`。

candidate-level expert margin 可以改善，而 gold stage 概率整体下降。这说明正信号常位于同一
动作阶段内部的对象、位置或完整命令排序；在阶段选择层面却泄漏概率质量。这解释了 NLL 改善、
top-1 与在线 success 不同步的现象。

## 6. 语义真实性对照

### 6.1 对照条件

在全部 971 个 held-out 状态上重新评分：

- `reformatted_skill`：保留相同字段和规则含义，改变表面模板；
- `anti_skill`：显式要求拒绝并反向执行相同规则；
- `task_only_skill`：只保留 task-specific 条目；
- `general_only_skill`：只保留 general skills 与 common mistakes；
- `length_matched_placebo`：不含动作建议，tokenizer 长度与原 skill 严格相同。

全部条件的 prompt overflow 为 0；placebo tokenizer 等长率为 100%。

### 6.2 完整结果

| split | condition | top-1 | NLL | margin gain | KL | output cosine | cosine with evolved |
|---|---|---:|---:|---:|---:|---:|---:|
| seen | plain | 41.22% | 2.3499 | 0 | 0 | 0 | — |
| seen | evolved | 39.19% | 2.2542 | +0.1469 | 0.2243 | +0.0191 | 1.0000 |
| seen | mismatch | 33.11% | 2.4215 | -0.0329 | 0.2167 | +0.0021 | 0.8391 |
| seen | reformat | 37.16% | 2.3279 | +0.0691 | 0.2238 | +0.0079 | 0.9149 |
| seen | anti | 38.06% | 2.3052 | +0.2080 | 0.2788 | +0.0102 | 0.8550 |
| seen | task-only | 40.54% | **2.2211** | +0.1552 | 0.2366 | +0.0252 | 0.8561 |
| seen | general-only | 36.26% | 2.3321 | +0.0772 | 0.1806 | +0.0140 | 0.8737 |
| seen | placebo | 39.64% | 2.4257 | -0.0632 | 0.0738 | -0.0175 | 0.7103 |
| unseen | plain | 31.69% | 2.5787 | 0 | 0 | 0 | — |
| unseen | evolved | 30.36% | 2.4314 | +0.2796 | 0.2054 | +0.0283 | 1.0000 |
| unseen | mismatch | 26.38% | 2.5752 | +0.0926 | 0.1773 | +0.0154 | 0.8472 |
| unseen | reformat | 32.26% | 2.4910 | +0.1757 | 0.2339 | +0.0279 | 0.9221 |
| unseen | anti | 31.50% | 2.4837 | +0.2690 | 0.2318 | +0.0152 | 0.8753 |
| unseen | task-only | **33.02%** | **2.3681** | +0.2856 | 0.2630 | +0.0418 | 0.8705 |
| unseen | general-only | 29.98% | 2.4955 | +0.2032 | 0.1493 | +0.0267 | 0.8537 |
| unseen | placebo | 29.60% | 2.6186 | -0.0195 | 0.0656 | -0.0095 | 0.7182 |

### 6.3 对照结论

task-only 在两个 split 都优于 general-only，并在 unseen 同时超过 plain 与完整 skill。信息主要
来自 task-specific 条目，而不是通用规则和 mistakes 的堆叠。但 task-only 相对 plain 的 top-1
episode CI 仍跨 0，尚不能宣称稳定提升。

正确 skill 相对等长 placebo 的 verified margin 优势：

| split | evolved - placebo margin gain | 95% CI |
|---|---:|---:|
| seen | +0.2102 | [+0.0286, +0.4362] |
| unseen | +0.2992 | [+0.1649, +0.4551] |

正确 skill 相对 mismatch 的 margin 优势在两个 split 也显著为正，因此确有语义增量。

但 negation test 失败：evolved 与 anti 的逐状态 margin effect 相关系数在 seen/unseen 为
`0.9056/0.9125`，而不是明显负相关。reformat effect 相关为 `0.9537/0.9271`；mismatch、anti、
甚至 placebo 与 evolved 的 direction cosine 都很高。模型没有稳定响应规则极性，prompt 的共同
词汇、格式与候选结构占据了大部分几何变化。

## 7. 真实 RMSNorm 参数梯度观测

### 7.1 测量定义

冻结模型，只让 Qwen 所有 RMSNorm scale 参数可导，共 73 个 tensor、149,504 个参数。对 plain
prompt 的候选序列 score 计算：

```text
g_verified = grad CE(onehot(expert), p_plain)
g_teacher  = grad CE(p_teacher, p_plain)
alignment  = cosine(g_teacher, g_verified)
```

teacher 包括 evolved、mismatch、reformat、anti 和 placebo。该实验只观测梯度，不修改权重。

probe 必须与缓存 scorer 使用同样的 BF16 batch shape。batch=2 时曾观察到最大 0.2197 的
shape-dependent score 漂移；恢复原实验 batch=4 后，128 个状态的最大误差降到 `9.54e-7`。

### 7.2 总体梯度结果

| split | teacher | mean cosine with verified | episode 95% CI | cosine > 0 |
|---|---|---:|---:|---:|
| seen | evolved | -0.1130 | [-0.2769, +0.0683] | 46.88% |
| seen | mismatch | -0.2472 | [-0.4305, -0.0455] | 37.50% |
| seen | reformat | -0.1817 | [-0.3620, +0.0129] | 40.62% |
| seen | anti | -0.2999 | [-0.4946, -0.1035] | 35.94% |
| seen | placebo | -0.2900 | [-0.4269, -0.1375] | 34.38% |
| unseen | evolved | -0.0994 | [-0.2695, +0.0645] | 46.88% |
| unseen | mismatch | -0.1249 | [-0.3046, +0.0371] | 46.88% |
| unseen | reformat | -0.1317 | [-0.3297, +0.0723] | 50.00% |
| unseen | anti | -0.1901 | [-0.3754, -0.0121] | 43.75% |
| unseen | placebo | -0.2574 | [-0.4680, -0.0838] | 42.19% |

evolved skill 的绝对梯度均值在两个 split 都略为负、CI 跨 0、正对齐率低于一半。它不是可以
直接当作 verified gradient 的方向。

### 7.3 正确 skill 的相对优势

| contrast | seen difference [95% CI] | unseen difference [95% CI] |
|---|---:|---:|
| evolved - mismatch | +0.1342 [+0.0292, +0.2554] | +0.0255 [-0.0883, +0.1401] |
| evolved - reformat | +0.0687 [-0.0185, +0.1665] | +0.0323 [-0.0980, +0.1571] |
| evolved - anti | +0.1869 [+0.0891, +0.3003] | +0.0907 [+0.0057, +0.1981] |
| evolved - placebo | +0.1770 [+0.0010, +0.3305] | +0.1579 [-0.0086, +0.3400] |

seen 中正确 skill 相对 mismatch、anti 和 placebo 的 gradient alignment 更好；unseen 只有相对
anti 的 CI 明确为正。正确 skill 更像“相对较少错误的 teacher”，而不是绝对正确 teacher。

### 7.4 参数梯度中的阶段共性

| verb | seen evolved cosine | unseen evolved cosine |
|---|---:|---:|
| go | +0.230 | +0.306 |
| close | +0.299 | +0.334 |
| take | -0.201 | -0.316 |
| open | -0.414 | -0.247 |
| move | -0.492 | -0.684 |

`look_at_obj_in_light` task 的均值在 seen/unseen 为 `+0.488/+0.399`；其他五类 task 均为负。
但 controls 也共享大量相同阶段符号，例如 anti 的 close 为 `+0.372/+0.523`。evolved 参数
梯度与其他条件的逐状态 cosine：

| condition | seen | unseen |
|---|---:|---:|
| reformat | 0.8469 | 0.7793 |
| mismatch | 0.6810 | 0.6794 |
| anti | 0.7388 | 0.7710 |
| placebo | 0.4620 | 0.5650 |

因此参数空间中的共性主要受 backbone Jacobian、候选 schema 与任务阶段控制。输出空间中小幅
为正的 gain 经过 `J^T` 映射后可能成为负参数对齐，不能从 output shift 直接推出正确的
backbone edit。

## 8. Logic distribution repair head 实验

这部分是一次“弱信号能否被 verifier 校准”的受监督干预，不是对 skill 自身真实性的证明。

### 8.1 参数化

对状态 `s` 的候选集合 `A(s)`：

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

head 只有 175 个参数。`verb_i` 近似 action stage，使同一 task skill 不必对导航、开关容器、
取放和 transformation 阶段使用同一全局系数。

### 8.2 目标与选择

```text
L(phi) = CE(q_phi, verified_action) + lambda * KL(q_phi || p0)
```

- 冻结 LLM、skill 文本与缓存 delta；梯度只进入 residual head。
- `lambda` 候选为 1、3、10。
- 只在 train 内部 episode-level dev 上选择平均 KL 不超过 0.05 且 NLL 最低的模型。
- 所有 seed 最终都选择 `lambda=3`。

### 8.3 初始单 seed 结果与 episode 修正

旧 episode key 下的初始结果：

| split | plain | direct skill | stage-only | skill-aware | skill-aware KL |
|---|---:|---:|---:|---:|---:|
| valid_seen | 41.22% | 39.19% | 44.82% | 46.62% | 0.0366 |
| valid_unseen | 31.69% | 30.36% | **35.67%** | 35.29% | 0.0303 |

该结果保留作为实验历史，但 seen 的 46.62% 受 episode 合并后的 train/dev split 影响，不再作为
主结论。修正后 seed 910 的 seen skill-aware 为 45.05%。

### 8.4 修正 episode 后五 seed 结果

seed 910--914：

| split | stage - plain top-1 | skill - plain top-1 | skill - stage top-1 | skill - stage NLL |
|---|---:|---:|---:|---:|
| seen | +3.06±0.49 点 | +4.59±0.47 点 | **+1.53±0.58 点** | **-0.0533±0.0088** |
| unseen | +3.80±0.45 点 | +3.42±0.35 点 | **-0.38±0.23 点** | **-0.0522±0.0059** |

- seen：skill 相对 stage 在 5/5 seed 提高 top-1，范围 +0.90 到 +2.25 点。
- unseen：skill 相对 stage 在 5/5 seed 降低 top-1，范围 -0.76 到 -0.19 点。
- unseen：skill 相对 stage 的 NLL 在 5/5 seed 都改善。

修正后的 seed 910 episode bootstrap：

| contrast | seen top-1 delta [95% CI] | unseen top-1 delta [95% CI] |
|---|---:|---:|
| skill - plain | +3.83 [+1.84, +5.72] | +3.23 [+1.40, +5.06] |
| skill - stage | +1.13 [0.00, +2.22] | -0.76 [-1.92, +0.57] |

这说明 verifier-supervised head 可以在 seen 中提取 skill 的附加信息；unseen 中，skill 更适合作为
弱概率证据，不能主导 top-1。stage-only calibration 才是更稳定的跨分布增益来源。

### 8.5 unconstrained 对照

旧单 seed seen 的 unconstrained skill-aware top-1 达到 52.25%、NLL 1.7225，但 KL 高达
0.4765。它只说明方向容量较强，不是安全或可部署结果。

## 9. 小规模在线 rollout 与循环诊断

同一 5 个任务，每个 episode 最多 50 步：

| policy | valid_seen success | valid_unseen success |
|---|---:|---:|
| plain | 2/5 | 0/5 |
| direct evolved skill | 2/5 | 0/5 |
| constrained stage | 1/5 | 0/5 |
| constrained skill-aware | 2/5 | 0/5 |
| constrained skill + cycle repair | **3/5** | **1/5** |

离线 next-action 提升没有稳定转化为 endpoint success。主要失败模式是确定性多步循环：同一
`(observation, action)` 在单个失败 episode 中重复 6--45 次；官方 expert 通常不重复该 pair，
即使重复也多为 2--3 次。

cycle repair 是无标签系统约束：同一 `(observation, action)` 已执行两次后，将该动作 candidate
logit 置为负无穷；若所有候选都被抑制则回退原分布。它修复了一个 seen 的 `take/move` 循环和
一个 unseen 的反复 `examine` 循环。

在线样本只有 5 个/split，不能做统计结论。它只证明 stateful repair 能带来 plain 没有的闭环
成功，也说明环境状态和循环记忆不能只依赖 prompt 中的自然语言提醒。

## 10. 综合回答：共性是什么，经验能否积累

### 10.1 已发现的共性

1. **task-specific 内容比 general skill 更有信息。** task-only 在两个 split 都优于
   general-only，且 unseen 的 top-1/NLL 最好。
2. **共性具有阶段条件。** `go/close` 和 `take/open/move` 在输出与参数观测中呈重复的不同符号；
   task family 会进一步改变同一 verb 的效果。
3. **有用部分多位于同阶段内部 candidate 排序。** gold verb 概率整体下降，但完整 expert
   command 的 margin 仍可能改善。
4. **输出 direction 的局部强度存在倒 U 形。** full prompt shift 太大，0.5 左右的事后缩放反而
   能提高 top-1。
5. **系统层失败存在不同共性。** endpoint 失败主要由 stateful cycle 导致，单步分布指标无法
   完全解释。

### 10.2 哪些“共性”不代表有效经验

- correct、mismatch、anti 和 placebo 都具有较高的跨状态/跨 split 几何稳定性；
- anti-skill 与 correct skill 同向，而不是反向；
- 参数 gradient concentration 甚至可能在错误 controls 中更高；
- 相同 action schema 会使不同文本共享 Jacobian 主方向；
- 因此高 cosine、高 concentration、跨 split prototype 稳定都不能单独作为 skill promotion 证据。

### 10.3 对核心命题的最终判断

| 命题 | 当前判断 | 证据 |
|---|---|---|
| skill 会改变输出分布 | 成立 | KL 约 0.18--0.28，方向稳定可测 |
| 正确 skill 含任务信息 | 较弱成立 | 相对 mismatch/placebo 的 verified margin 更好 |
| 模型可靠理解 skill 逻辑 | 不成立 | anti-skill 不反向，reformat/top-1 敏感 |
| 存在全局可复用 output direction | 不成立 | 约一半状态反向，task/stage 符号冲突 |
| output gain 可直接变为参数 edit | 不成立 | RMSNorm mean alignment 为负，CI 跨 0 |
| verified calibration 能利用弱信号 | 成立 | seen 五 seed 相对 stage +1.53±0.58 点 |
| OOD 中 skill 可主导策略 | 不成立 | unseen 五 seed top-1 均低于 stage-only |
| 在线 endpoint 已得到统计验证 | 未成立 | rollout 只有 5 episode/split |

最终结论不是“经验完全无效”，也不是“已经能内化”：**存在可测语义增量，但它被模板效应、
候选几何、阶段冲突和 OOD 失配包围。没有 verifier 或独立行为证据时，不能把 skill-induced
distribution shift 直接累积为 backbone 参数更新。**

## 11. 与 SEED 的关系

SEED 的关键思想是从当前 policy 的 completed on-policy trajectory 生成 hindsight skill，再利用
plain/skill context 的概率差构造 dense teacher signal。它提供的是一个更紧密的“行为—经验—
策略”闭环，但 dense/on-policy/self-evolving 本身仍不保证 return 单调改善；hindsight skill 必须
具有 behaviorally informative 的内容。

本实验使用的是静态 SkillRL skill 库，不是由当前 Qwen policy 的 on-policy trajectory 同步生成。
同时，它没有通过 anti-skill 极性测试。因此本实验不能被解释为对 SEED 有效经验假设的正向验证；
更准确地说，它展示了当 experience 不够 behaviorally grounded 时，OPD/skill delta 可能包含弱
信息，却仍不足以成为可靠参数 teacher。

## 12. 已完成事项与边界

### 12.1 已完成

- 1569 个状态的 plain、correct、mismatch 多 token 序列评分；
- 修正真实 gamefile episode key 与全部 episode bootstrap；
- output-space gradient、margin、dose-response、task/stage 与 verb projection 诊断；
- 五 seed、unique-episode repair head 复验；
- 971 个 held-out 状态的 reformat、anti、task-only、general-only、等长 placebo 评分；
- 128 个状态、149,504 维 RMSNorm 参数梯度 VJP；
- 原始 Qwen 与官方 SEED-ALFWorld-3B 在同一 971 个状态上的 checkpoint delta、最小修复距离与
  rescue/harm 对照；
- 10,000 次 episode-cluster bootstrap；
- 5+5 episode 小规模在线 rollout 与 cycle repair；
- 15 项单元测试。

### 12.2 尚未完成

- 从当前 Qwen policy 的独立 on-policy success/failure trajectory 生成 hindsight skill；
- 不读取 valid expert label 的 skill behavioral verification；
- 实际应用 backbone 小步编辑后的 control drift、跨状态干扰和遗忘；
- seen/unseen 各 50--100 episode 的在线 success 统计；
- 多模型规模、多 backbone 与多随机 seed 的参数梯度复验。

## 13. 复现入口

### 13.1 数据、评分与 prompt

- 数据采集：`scripts/collect_alfworld_expert.py`
- decision 数据：`data/alfworld_expert_large/{train,valid_seen,valid_unseen}.jsonl`
- 多 token scorer：`src/self_evolve/sequence_scorer.py`
- skill retrieval/prompt：`src/self_evolve/alfworld_skills.py`
- 语义 control renderer：`src/self_evolve/alfworld_skill_controls.py`
- 离线 runner：`src/self_evolve/alfworld_runner.py`

### 13.2 输出共性与语义对照

- 主共性诊断：`scripts/analyze_skill_commonality.py`
- 共性结果：`outputs/skill_commonality_3b/results.json`
- enriched traces：`outputs/skill_commonality_3b/enriched_traces/*.jsonl`
- 语义对照评分：`scripts/score_alfworld_skill_controls.py`
- 语义对照分析：`scripts/analyze_alfworld_skill_controls.py`
- 语义对照结果：`outputs/skill_controls_3b/analysis/results.json`
- 语义对照简表：`outputs/skill_controls_3b/analysis/REPORT.md`

### 13.3 参数梯度

- 梯度 probe：`scripts/probe_skill_parameter_gradients.py`
- 梯度分析：`scripts/analyze_parameter_gradient_probe.py`
- 梯度结果：`outputs/gradient_probe/analysis/results.json`
- 梯度简表：`outputs/gradient_probe/analysis/REPORT.md`

### 13.4 repair head 与 bootstrap

- repair head：`src/self_evolve/logic_repair_head.py`
- 最终配置：`config/logic_repair_head_large_3b.yaml`
- 旧单 seed 结果：`outputs/logic_repair_head_large_3b/results.json`
- episode 修正 bootstrap：`scripts/bootstrap_repair_effects.py`
- 多 seed 汇总：`scripts/summarize_repair_multiseed.py`
- 多 seed 结果：`outputs/skill_commonality_3b/multiseed/summary.json`
- 多 seed 简表：`outputs/skill_commonality_3b/multiseed/summary.md`

### 13.5 在线 rollout

- 在线 runner/cycle repair：`src/self_evolve/alfworld_online.py`
- 在线配置：`config/alfworld_online_3b.yaml`
- 在线结果：`outputs/alfworld_online_3b/*/results.json`

## 14. 测试与运行记录

- 当前单元测试：`15 passed`。
- 语义对照评分使用远端 GPU 0--3，完成 444+527=971 个状态，无失败分片。
- 参数 probe 使用同四张 GPU，完成 64+64=128 个唯一状态。
- 参数 probe plain score mean/max 误差：
  - seen：`4.29e-7 / 9.54e-7`；
  - unseen：`3.99e-7 / 9.54e-7`。
- 实验结束后 GPU 0--3 均释放，临时 SSH 认证密钥已移除；远端实验输出保留。

## 15. 文档关系

本文件是完整总报告。以下文件保留为分阶段实验记录：

- `research/ALFWORLD_LOGIC_REPAIR_RESULTS_ZH.md`：初始 logic distribution repair 与在线 rollout；
- `research/ALFWORLD_SKILL_COMMONALITY_RESULTS_ZH.md`：episode 修正、共性、语义 controls 与参数梯度；
- `outputs/skill_controls_3b/analysis/REPORT.md`：语义对照自动生成简表；
- `outputs/gradient_probe/analysis/REPORT.md`：参数梯度自动生成简表。
- `research/ALFWORLD_SEED_CHECKPOINT_OUTPUT_DELTA_ZH.md`：官方 SEED checkpoint 的训练前后输出分布、
  最小修复距离与关键决策分析。
