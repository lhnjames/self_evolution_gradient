# ALFWorld skill 输出分布共性诊断实验

> 完整合并版见：[`ALFWORLD_EVOLVED_SKILL_COMPLETE_REPORT_ZH.md`](ALFWORLD_EVOLVED_SKILL_COMPLETE_REPORT_ZH.md)

更新时间：2026-09-01

## 1. 本轮问题与边界

本轮不设计新的策略算法，只观察以下命题：一条自然语言经验造成的候选动作分布变化，是否具有
可验证的正确性、跨 episode 共性，以及能否构成稳定的梯度编辑方向。

实验先复用 Qwen2.5-3B-Instruct 已缓存的 1569 个 ALFWorld 决策状态，包括 plain、正确 task
skill 和 mismatched task skill 的多 token 候选序列分数；随后在 valid_seen 444 个和
valid_unseen 527 个状态上重新评分五种语义对照，并在每个 split 平衡抽取 64 个状态，观测
149,504 维 RMSNorm 参数子空间中的真实一阶梯度。没有用 valid 标签选择 skill、条件或切片。

这里的“梯度对齐”严格指候选输出 logit 空间：对 plain 分布 `p0`，expert CE 的负梯度方向为

```text
u_verified = onehot(expert) - p0
d_skill    = center(z_skill - z_plain)
```

观察 `dot(u_verified, d_skill)`、cosine、有限步长后的真实 log-prob 变化及 task/verb 投影。
前半部分验证输出分布中间层是否存在编辑信号；第 8 节进一步将候选分布 teacher 通过冻结
backbone 的 Jacobian 映射到真实参数梯度。参数实验只覆盖全部 RMSNorm scale，不代表完整
backbone，也没有执行参数更新。

## 2. 数据完整性修正

原 trace 中的 `episode_id` 只包含任务目录名，没有包含其下的 `trial_*`。同一任务目录存在多个
独立 gamefile，因此旧逻辑把若干真实 rollout 合成了一个 cluster：

| split | 旧 episode_id 数 | 真实 gamefile 数 |
|---|---:|---:|
| train | 45 | 47 |
| valid_seen | 32 | 33 |
| valid_unseen | 25 | 34 |

本轮将原始 expert decision 文件与 merged trace 按顺序及完整字段逐行核对，使用 `gamefile` 作为
唯一 episode key，并生成 enriched traces。valid_unseen 的聚类修正最大。

同时修复了未来 trace 写出、head 的 episode split 和 bootstrap，使其优先读取 `episode_key`。
该问题没有造成 train/valid 泄漏，但改变 train/dev 的实际 episode 划分，也使旧 bootstrap 的
cluster 数偏少。

## 3. 原始 skill 方向的总体结果

| split | decisions / episodes | plain top-1 | direct skill | mismatch | mean margin gain | mean expert log-p gain | mean cosine | cosine > 0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 598 / 47 | 38.80% | 36.12% | 32.44% | +0.0993 | -0.0096 | +0.0053 | 48.49% |
| valid_seen | 444 / 33 | 41.22% | 39.19% | 33.11% | +0.1469 | +0.0957 | +0.0191 | 53.15% |
| valid_unseen | 527 / 34 | 31.69% | 30.36% | 26.38% | +0.2796 | +0.1473 | +0.0283 | 55.60% |

正确 skill 比 mismatch 的 margin gain 高 0.1798（seen）和 0.1870（unseen）。按真实 episode
等权 bootstrap，这一差值的 95% CI 分别为 `[+0.0862, +0.2791]` 和
`[+0.1096, +0.2881]`。因此 skill 文本中存在 task-matched 语义信息，不能全部解释为加入长文本
带来的通用 prompt shift。

但这个信号非常弱且双向：seen 只有 53.15%、unseen 只有 55.60% 的状态具有正 cosine；平均
cosine 接近 0。direct skill 在 seen 中 rescue 4.73% 的决策、harm 6.76%；unseen 中 rescue
5.88%、harm 7.21%。所以单个状态上不能把 `d_skill` 当成可信标签。

first-order gain 与真实 expert log-p gain 的相关性很高：seen Pearson/Spearman 为
`0.9562/0.9839`，unseen 为 `0.9525/0.9867`。这说明输出 logit 空间的一阶观测能够准确预测
skill prompt 的有限分布效应；问题不在梯度计算，而在经验方向本身的低信噪比。

### 修正后的 episode-cluster bootstrap

| split | metric | observed | decision-weighted 95% CI |
|---|---|---:|---:|
| seen | direct top-1 delta | -2.03 点 | [-6.22, +2.28] |
| seen | expert log-p gain | +0.0957 | [-0.0675, +0.3063] |
| seen | margin gain | +0.1469 | [-0.0865, +0.4360] |
| seen | gradient cosine | +0.0191 | [-0.0118, +0.0585] |
| unseen | direct top-1 delta | -1.33 点 | [-4.94, +2.69] |
| unseen | expert log-p gain | +0.1473 | [+0.0233, +0.3008] |
| unseen | margin gain | +0.2796 | [+0.0961, +0.4958] |
| unseen | gradient cosine | +0.0283 | [+0.0070, +0.0552] |

unseen 的均值显著为正，但 top-1 仍下降，说明 skill 经常提高 expert 相对多数错误动作的排序，
同时又把少数竞争动作推得更高，或者把变化做得过强。

## 4. 不训练模型的 skill dose-response

仅沿原始 `d_skill` 扫描全局系数，没有用 valid 标签选取系数：

| split | alpha | top-1 | NLL | mean KL |
|---|---:|---:|---:|---:|
| seen | 0.0 | 41.22% | 2.3499 | 0 |
| seen | 0.25 | 40.99% | 2.2831 | 0.0147 |
| seen | 0.50 | **43.02%** | 2.2454 | 0.0579 |
| seen | 0.75 | 42.57% | **2.2360** | 0.1272 |
| seen | 1.0 | 39.19% | 2.2542 | 0.2243 |
| unseen | 0.0 | 31.69% | 2.5787 | 0 |
| unseen | 0.25 | 31.12% | 2.5038 | 0.0128 |
| unseen | 0.50 | **32.64%** | 2.4545 | 0.0509 |
| unseen | 0.75 | 31.50% | **2.4303** | 0.1131 |
| unseen | 1.0 | 30.36% | 2.4314 | 0.2054 |

两 split 都呈现相同的倒 U 形：方向含有信号，但 full-strength prompt shift 越过了最佳区间。
这是一项事后诊断，不是部署超参选择；它证明 direct skill 失败不能等价为 skill 没有信息。

## 5. 共性究竟在哪里

### 5.1 task × action stage，而不是整条 task skill 的单一方向

按 expert verb 切片，seen/unseen 中出现重复的符号结构：

| verb | seen margin gain / top-1 delta | unseen margin gain / top-1 delta |
|---|---:|---:|
| go | +0.301 / -3.78 点 | +0.454 / +0.30 点 |
| close | +0.941 / +1.79 点 | +0.855 / 0 点 |
| open | -0.729 / -6.45 点 | -0.684 / -20.00 点 |
| move | -0.652 / +3.23 点 | -0.515 / 0 点 |
| take | +0.044 / 0 点 | -0.087 / 0 点 |

但 verb 本身也不是完整规律。例如 `look_at_obj_in_light/open` 的 margin gain 在 seen/unseen
分别为 `+1.136/+0.657`，而 clean、heat、cool 等 task 的 open 阶段稳定为负。因此最小可重复
单位是 `task family × action stage`，而不是 task-level skill 或全局 verb。

### 5.2 结构稳定不等于经验真实

把每个状态的概率变化投影到公共 verb simplex 后，跨真实 episode 的两两 cosine 为：

| relation | seen correct skill | unseen correct skill |
|---|---:|---:|
| same task, same stage | 0.3100 | 0.2243 |
| different task, same stage | 0.1595 | 0.1095 |
| same task, different stage | 0.0947 | 0.0681 |
| different task, different stage | 0.0368 | 0.0051 |

task 和 stage 都贡献结构共性，stage 稍强；二者同时相同时方向最一致。

然而 mismatched skill 在上述 pairwise 分组中的 cosine 与 correct skill 相近，有时反而更高。
task×stage prototype 的跨 split cosine 也很接近：

| comparison | correct skill | mismatched skill |
|---|---:|---:|
| train ↔ seen | 0.8814 | 0.8624 |
| train ↔ unseen | 0.8043 | 0.8024 |
| seen ↔ unseen | 0.7913 | 0.7419 |

因此“方向跨 split 稳定”本身不足以证明经验有效。很大一部分稳定结构来自同一模型、prompt
模板、candidate schema 和长文本条件化。正确经验的额外证据来自它相对 mismatch 的 verified
margin 优势，而不是几何稳定性本身。

### 5.3 logic/stage 概率投影揭示的冲突

整体上，正确 skill 对 gold verb 的概率变化为负：train/seen/unseen 分别为
`-0.0308/-0.0360/-0.0253`，verb-space gradient cosine 也分别为
`-0.2437/-0.2507/-0.2268`。

与此同时，candidate-level expert margin 在 seen/unseen 为正。这说明当前 skill 的主要正信号
常发生在同一动作阶段内部的对象、位置或命令排序，而在阶段选择层面却整体泄漏概率质量。
这解释了为何 NLL/margin 可以改善，top-1 和在线 endpoint 却不一定改善。

## 6. 修正 episode 后的 head 多 seed 复验

使用真实 gamefile episode key，以 seed 910--914 做五次 train/dev episode split。每次仍只在
dev 内选择 lambda；五次 skill head 和 stage-only head 都选择 lambda=3。

| split | stage - plain top-1 | skill - plain top-1 | skill - stage top-1 | skill - stage NLL |
|---|---:|---:|---:|---:|
| seen，5-seed mean ± sd | +3.06 ± 0.49 点 | +4.59 ± 0.47 点 | **+1.53 ± 0.58 点** | **-0.0533 ± 0.0088** |
| unseen，5-seed mean ± sd | +3.80 ± 0.45 点 | +3.42 ± 0.35 点 | **-0.38 ± 0.23 点** | **-0.0522 ± 0.0059** |

seen 上 skill 相对 stage 在 5/5 seed 提高 top-1，范围 `+0.90` 到 `+2.25` 点。unseen 上
5/5 seed 的 top-1 都低于 stage-only，范围 `-0.76` 到 `-0.19` 点，但 5/5 seed 的 NLL 都更好。

唯一 episode 修正后的 seed 910 bootstrap：

- seen：skill - plain `+3.83 [ +1.84, +5.72 ]` 点；skill - stage
  `+1.13 [0.00, +2.22]` 点。
- unseen：skill - plain `+3.23 [ +1.40, +5.06 ]` 点；skill - stage
  `-0.76 [-1.92, +0.57]` 点。

旧报告 seed 910 的 seen skill-aware 为 46.62%；修正 episode split 后为 45.05%。这表明旧单 seed
数值略偏乐观，但“seen 有额外 skill 增益、unseen 只有 NLL 增益”的核心结论未改变，并得到
多 seed 支持。

## 7. 新增语义真实性对照：有信号，但 negation test 失败

在全部 971 个 held-out 状态上重新评分五种对照：保持字段不变但更换模板的 reformat、显式
否定同一规则的 anti-skill、task-only、general+mistakes-only，以及与原 skill 严格 tokenizer
等长的无关 placebo。所有条件均无 prompt truncation，placebo 等长率为 100%。

| split | condition | top-1 | NLL | margin gain | KL from plain | output-gradient cosine |
|---|---|---:|---:|---:|---:|---:|
| seen | evolved skill | 39.19% | 2.2542 | +0.1469 | 0.2243 | +0.0191 |
| seen | reformat | 37.16% | 2.3279 | +0.0691 | 0.2238 | +0.0079 |
| seen | anti-skill | 38.06% | 2.3052 | +0.2080 | 0.2788 | +0.0102 |
| seen | task-only | 40.54% | **2.2211** | +0.1552 | 0.2366 | +0.0252 |
| seen | general-only | 36.26% | 2.3321 | +0.0772 | 0.1806 | +0.0140 |
| seen | length-matched placebo | 39.64% | 2.4257 | -0.0632 | 0.0738 | -0.0175 |
| unseen | evolved skill | 30.36% | 2.4314 | +0.2796 | 0.2054 | +0.0283 |
| unseen | reformat | 32.26% | 2.4910 | +0.1757 | 0.2339 | +0.0279 |
| unseen | anti-skill | 31.50% | 2.4837 | +0.2690 | 0.2318 | +0.0152 |
| unseen | task-only | **33.02%** | **2.3681** | +0.2856 | 0.2630 | +0.0418 |
| unseen | general-only | 29.98% | 2.4955 | +0.2032 | 0.1493 | +0.0267 |
| unseen | length-matched placebo | 29.60% | 2.6186 | -0.0195 | 0.0656 | -0.0095 |

task-only 在两个 split 都优于 general-only，并在 unseen 同时超过 plain 和完整 skill，说明可用
信息主要来自 task-specific 条目，而不是通用规则堆叠。但 task-only 相对 plain 的 top-1 CI
仍跨 0，不能宣称稳定提升。

更关键的反证是：anti-skill 没有把方向反转。原 skill 与 anti-skill 的逐状态 margin effect
相关系数在 seen/unseen 高达 `0.9056/0.9125`；两者参数前的 output direction cosine 为
`0.8550/0.8753`。reformat 与原 skill 的 effect 相关为 `0.9537/0.9271`，说明主要 effect 可跨
措辞保留，但强度和 top-1 明显不稳定。甚至 mismatch 与原 skill 的方向 cosine 也为
`0.8391/0.8472`，严格等长 placebo 仍有 `0.7103/0.7182`。

原 skill 相对 placebo 的 verified margin 优势为 seen `+0.2102 [0.0286, 0.4362]`、unseen
`+0.2992 [0.1649, 0.4551]`；相对 mismatch 也显著为正。因此不能说 skill 没有语义增量。但
negation test 失败表明当前模型主要响应共同的主题词、模板和候选结构，而没有可靠执行规则的
逻辑极性。“正确 skill 比错误文本稍好”与“模型理解并可内化这条经验”是两个不同命题。

## 8. 新增真实参数梯度观测：输出正信号没有保留为绝对正确梯度

对每个 split 按 `task_type × action_verb` 平衡抽取 64 个状态。冻结 Qwen，只令 73 个 RMSNorm
scale tensor 可导，共 149,504 个参数。对 plain prompt 的候选序列 score 计算两类 loss 梯度：

```text
g_verified = grad CE(onehot(expert), p_plain)
g_skill    = grad CE(p_skill, p_plain)
```

比较 `cos(g_skill, g_verified)`。这是真实 autograd VJP，不是 residual head 的特征梯度，也没有
更新模型。probe 与原缓存使用相同 batch=4 后，plain score 的 mean/max 误差在 seen 为
`4.29e-7/9.54e-7`，unseen 为 `3.99e-7/9.54e-7`。

| split | teacher | mean cosine with verified | episode 95% CI | cosine > 0 |
|---|---|---:|---:|---:|
| seen | evolved skill | -0.1130 | [-0.2769, +0.0683] | 46.88% |
| seen | mismatch | -0.2472 | [-0.4305, -0.0455] | 37.50% |
| seen | reformat | -0.1817 | [-0.3620, +0.0129] | 40.62% |
| seen | anti-skill | -0.2999 | [-0.4946, -0.1035] | 35.94% |
| seen | placebo | -0.2900 | [-0.4269, -0.1375] | 34.38% |
| unseen | evolved skill | -0.0994 | [-0.2695, +0.0645] | 46.88% |
| unseen | mismatch | -0.1249 | [-0.3046, +0.0371] | 46.88% |
| unseen | reformat | -0.1317 | [-0.3297, +0.0723] | 50.00% |
| unseen | anti-skill | -0.1901 | [-0.3754, -0.0121] | 43.75% |
| unseen | placebo | -0.2574 | [-0.4680, -0.0838] | 42.19% |

正确 skill 在绝对意义上不是 verified 梯度：两个 split 的均值都为负、CI 跨 0、正对齐率低于
一半。它只在相对意义上较少反向。seen 中 evolved - mismatch 的 paired cosine 差为
`+0.1342 [0.0292, 0.2554]`，unseen 为 `+0.0255 [-0.0883, 0.1401]`；相对 placebo 分别为
`+0.1770 [0.0010, 0.3305]` 和 `+0.1579 [-0.0086, 0.3400]`。语义相对优势在 seen 较可靠，
到了 unseen 证据不足。

参数梯度的跨状态共性仍主要体现为阶段符号：

| verb | seen evolved cosine | unseen evolved cosine |
|---|---:|---:|
| go | +0.230 | +0.306 |
| close | +0.299 | +0.334 |
| take | -0.201 | -0.316 |
| open | -0.414 | -0.247 |
| move | -0.492 | -0.684 |

`look_at_obj_in_light` task 的均值在 seen/unseen 为 `+0.488/+0.399`，其余五类 task 均为负。
但相同阶段符号也大量出现在 mismatch、anti 和 placebo 中；例如 anti 的 close 为
`+0.372/+0.523`。evolved 参数梯度与 reformat 的逐状态 cosine 为 `0.8469/0.7793`，与
mismatch 为 `0.6810/0.6794`，与 anti 为 `0.7388/0.7710`。这再次说明可重复几何首先来自
backbone Jacobian、prompt 模板和 candidate schema，不能直接当作有效经验的参数记忆。

输出空间中小幅为正的 verified gain，经过 `J^T` 映射到 RMSNorm 参数度量后可以变成负对齐；
所以“输出 distribution shift 可预测”并不自动推出“对 backbone 做同方向梯度编辑是正确的”。

## 9. 对“经验能否形成可积累参数编辑方向”的回答

当前证据支持以下较窄结论：

1. task-matched skill 的输出方向含有统计可见的信息：相对 mismatch 和等长 placebo 的 verified
   margin 在两个 held-out split 都更好，task-only 也优于 general-only。
2. 但它没有通过强语义真实性测试：anti-skill 不反向，reformat 会改变 top-1，正确/mismatch/
   anti/placebo 共享很高的方向 cosine。因此不能把 prompt effect 全归因于规则含义。
3. 输出 logit 的 first-order gain 能准确测量 prompt 引起的有限分布变化；它测量的是“文本会
   怎样推动模型”，不是“这个推动在任务上正确”。
4. 共性主要是 task/stage 条件结构。真实 RMSNorm 参数梯度中 `go/close` 稳定为正，
   `take/open/move` 稳定为负，但 controls 也共享该结构，说明其中很大部分是模型与 action
   schema 的归纳偏置。
5. 当前 evolved skill 的参数梯度在两个 split 都没有绝对正对齐，只在 seen 相对 mismatch/
   placebo 更好。现有证据否定了“把单条经验造成的 output shift 直接梯度写回 backbone”这一
   强命题。
6. 小 head 的监督结果仍说明弱信号可以被 verifier 校准；它证明的是有监督分布修复，不是 skill
   自身已经成为可无条件积累的参数编辑方向。

这一点与 SEED 的理论边界一致：on-policy、dense、self-evolving 只能保证监督的分布匹配与局部
结构，论文也明确说明它们不蕴含 return 单调改善，仍要求 hindsight skill behaviorally
informative。当前 SkillRL skill 是静态库，不是由当前 Qwen policy 的 on-policy completed
trajectory 同步生成，因此不能把本实验直接解释为 SEED 式有效经验的验证。

## 10. 仍未闭环的边界

本轮已经完成 reformat、anti、task/general 拆分、严格等长 placebo 和真实 RMSNorm 参数梯度。
仍未完成的是：从当前 Qwen policy 的独立 on-policy success/failure trajectory 生成 hindsight
skill，以及在不读取 valid label 的前提下，对候选经验做行为验证。也没有实际应用 backbone
编辑，因此未观测小步更新后的 control drift、跨状态干扰和 rollout return。

按 SEED 的标准，静态 SkillRL 库与当前 policy 不同步，且 anti-skill 极性测试失败，尚不能称为
behaviorally informative hindsight experience。本轮结论是负向但明确的：存在可测的语义增量，
却没有发现能无 verifier 直接积累的通用参数编辑方向。

## 11. 复现入口

- 主诊断：`scripts/analyze_skill_commonality.py`
- episode 修正：`src/self_evolve/alfworld_runner.py`、`src/self_evolve/logic_repair_head.py`
- 修正 bootstrap：`scripts/bootstrap_repair_effects.py`
- 多 seed 汇总：`scripts/summarize_repair_multiseed.py`
- enriched traces：`outputs/skill_commonality_3b/enriched_traces/*.jsonl`
- 完整诊断：`outputs/skill_commonality_3b/results.json`
- 可读报告：`outputs/skill_commonality_3b/REPORT.md`
- 修正 head：`outputs/skill_commonality_3b/corrected_head/`
- 多 seed：`outputs/skill_commonality_3b/multiseed/summary.json`
- 语义对照 scorer：`scripts/score_alfworld_skill_controls.py`
- 语义对照分析：`scripts/analyze_alfworld_skill_controls.py`
- 语义对照结果：`outputs/skill_controls_3b/analysis/results.json`
- RMSNorm 参数梯度 probe：`scripts/probe_skill_parameter_gradients.py`
- 参数梯度分析：`scripts/analyze_parameter_gradient_probe.py`
- 参数梯度结果：`outputs/gradient_probe/analysis/results.json`

所有单元测试：`13 passed`。
