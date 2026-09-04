# ALFWorld 自进化前后输出分布实验：Qwen2.5-3B vs SEED-ALFWorld-3B

更新时间：2026-09-01  
实验状态：971 个专家状态全量评分完成  
模型：Qwen2.5-3B-Instruct、Jinyang23/Seed-AlfWorld-3B

## 1. 本轮只回答什么

本轮不设计新算法，也不训练新 head。只测量：一个官方发布、已经经过自进化训练的 ALFWorld
模型，相比原始 Qwen2.5-3B，在完全相同的专家状态和候选动作上究竟改变了哪些数字。

需要严格区分两个指标：

- SEED 模型卡的 91.8%/86.2% 是 seen/unseen 的完整 episode success 宏平均；
- 本项目的 41.22%/31.69% 是一个固定专家状态中，唯一记录的 expert next action 是否排 top-1。

二者不能横向比较。ALFWorld 通常存在多条成功轨迹；不匹配 hand-coded expert 的下一步，并不
必然代表动作会导致任务失败。官方模型卡还明确说明，SEED 推理时不需要 skill prompt，但模型
专门适配了 ALFWorld interaction prompt，并输出 `<think>` 和 `<action>`。本轮主对照故意不采用
该专用生成接口，而让两个 checkpoint 使用本项目现有的同一个 direct-action plain prompt，
从而隔离权重变化。官方背景见 [SEED-ALFWorld-3B model card](https://huggingface.co/Jinyang23/Seed-AlfWorld-3B/blob/main/README.md)
和 [SEED paper](https://arxiv.org/abs/2607.14777)。

## 2. 严格对照设计

两个模型使用完全相同的：

- 444 个 valid_seen、527 个 valid_unseen 专家状态；
- task description、最近三步 history、current observation；
- 环境返回的 admissible commands；
- direct-action plain prompt；
- base Qwen tokenizer；
- completion-only、多 token、长度归一化 sequence scorer；
- BF16、batch size 4、max length 1536。

模型结构均为 36 层、hidden size 2048、vocab size 151,936，输入输出 embedding 共享；官方 base
配置见 [Qwen2.5-3B-Instruct config](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/blob/main/config.json)。

SEED checkpoint 自带 tokenizer 文件的格式与 base 略有不同，但实际逐项核对得到：

- 971/971 个完整 prompt 的 token IDs 完全相同；
- 27,660/27,660 个 candidate action 的 token IDs 完全相同。

主实验仍显式使用 base tokenizer，保证 `z_SEED-z_base` 只包含权重变化。

## 3. 自进化前后的总体输出数字

定义：

```text
z_base(s,a) = base checkpoint 的长度归一化 action score
z_seed(s,a) = SEED checkpoint 在同 prompt/action 上的 score
delta_seed  = center(z_seed - z_base)
```

| split | base top-1 | SEED top-1 | delta | base p(expert) | SEED p(expert) | base NLL | SEED NLL | KL(SEED‖base) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| valid_seen | 41.22% | **44.59%** | **+3.38 点** | 0.2291 | 0.2084 | 2.3499 | 2.4816 | 0.3517 |
| valid_unseen | 31.69% | **35.86%** | **+4.17 点** | 0.1770 | 0.1749 | 2.5787 | 2.7868 | 0.3268 |

episode-cluster bootstrap：

| split | top-1 delta [95% CI] | expert log-p gain [95% CI] |
|---|---:|---:|
| seen | **+3.38 [+0.66, +6.35] 点** | -0.1317 [-0.2995, +0.0800] |
| unseen | **+4.17 [+1.28, +7.06] 点** | **-0.2081 [-0.3179, -0.0674]** |

这是第一个关键结果：SEED 在同一 direct-action 接口下确实显著提高了 expert next-action top-1，
但没有整体提高 expert probability；NLL 反而变差，unseen 的 log-p 下降有显著 CI。

因此 SEED 权重变化不是“把整个分布变得更像 hand-coded expert”。它改变了一部分 top-1 决策
边界，同时在许多其他状态上降低了唯一 expert action 的概率。

六个 task family 等权后的单步 top-1：

| split | base task-macro | SEED task-macro | delta |
|---|---:|---:|---:|
| seen | 40.42% | 42.97% | +2.55 点 |
| unseen | 30.29% | 36.92% | +6.63 点 |

该数字仍是 expert-action matching，不是 episode success。

## 4. 哪些 base 决策被改变

| split | base 错误数 | rescued | base 正确数 | harmed | 净增加 |
|---|---:|---:|---:|---:|---:|
| seen | 261 | 27（10.34%） | 183 | 12（6.56%） | +15 |
| unseen | 360 | 32（8.89%） | 167 | 10（5.99%） | +22 |

SEED 没有修复大多数 expert mismatch，只修复约 9%--10% 的 base 错误；但它破坏的原正确状态
更少，所以得到净 top-1 增益。

checkpoint delta 与 expert output direction 的平均 cosine 也很小：

| split | mean cosine | cosine > 0 | mean cosine with prompt-skill delta |
|---|---:|---:|---:|
| seen | +0.0175 | 56.31% | 0.0905 |
| unseen | +0.0215 | 55.41% | 0.1367 |

这说明成功训练后的权重 delta 与静态 skill prompt delta 并不是同一个方向；它和 expert imitation
方向也只有弱总体对齐。

## 5. Base 错误距离正确 expert action 有多远

对 base 错误状态，设最高错误动作是 `w`，expert 是 `y`：

```text
gap = z_base(w) - z_base(y)

delta_min(y) = +gap/2
delta_min(w) = -gap/2

||delta_min||_2 = gap / sqrt(2)
```

这是使 expert 与当前最高错误动作到达 tie boundary 的最小二范数 action-score 编辑；再加任意
小的正 epsilon 即可严格翻转 top-1。

### 5.1 距离分布

| split | metric | p10 | p25 | median | p75 | p90 | mean |
|---|---|---:|---:|---:|---:|---:|---:|
| seen | score gap | 0.409 | 1.000 | 1.729 | 2.504 | 4.250 | 2.019 |
| seen | minimal L2 | 0.289 | 0.707 | 1.223 | 1.771 | 3.005 | 1.428 |
| unseen | score gap | 0.339 | 0.933 | 1.698 | 2.731 | 3.863 | 1.968 |
| unseen | minimal L2 | 0.239 | 0.660 | 1.201 | 1.931 | 2.732 | 1.392 |

只有最接近边界的约 10% 错误处于 0.24--0.29 的最小编辑量级；中位错误需要约 1.2 的
length-normalized action-score L2 编辑。多数错误不是极轻微的校准偏差。

## 6. SEED 实际变化是否接近最小 expert 修复

只在 base 错误状态上比较 `delta_seed` 与 `delta_min`：

| split | metric | mean | median | p25 | p75 |
|---|---|---:|---:|---:|---:|
| seen | cosine(delta_seed, delta_min) | 0.082 | 0.056 | 0.002 | 0.133 |
| unseen | cosine(delta_seed, delta_min) | 0.075 | 0.046 | -0.023 | 0.120 |
| seen | norm ratio | 13.98 | 6.65 | 4.35 | 10.94 |
| unseen | norm ratio | 16.80 | 6.70 | 3.78 | 12.51 |
| seen | projection/minimal units | 0.559 | 0.413 | 0.013 | 0.907 |
| unseen | projection/minimal units | 1.791 | 0.272 | -0.179 | 0.814 |

SEED 的实际输出变化远大于最小 expert edit，且大部分方向与最短修复近乎正交。因此“成功自进化
等价于沿 expert 最短方向做小局部修复”不成立。

但把错误按最终是否 rescued 分开后，规律非常清楚：

| split | group | states | minimal L2 median | delta/min cosine median | projection median |
|---|---|---:|---:|---:|---:|
| seen | rescued | 27 | **0.450** | **0.149** | **2.088** |
| seen | unresolved | 234 | 1.278 | 0.049 | 0.367 |
| unseen | rescued | 32 | **0.277** | **0.172** | **4.464** |
| unseen | unresolved | 328 | 1.322 | 0.037 | 0.202 |

被 SEED 修复的错误明显更靠近原决策边界，并且实际 delta 在最小修复方向上的分量更大。

### 6.1 按最小修复距离四分位

| split | distance quartile | rescue rate |
|---|---:|---:|
| seen | Q1（最近） | **26.15%** |
| seen | Q2 | 10.77% |
| seen | Q3 | 3.08% |
| seen | Q4（最远） | 1.52% |
| unseen | Q1（最近） | **26.67%** |
| unseen | Q2 | 8.89% |
| unseen | Q3 | **0%** |
| unseen | Q4（最远） | **0%** |

这是本轮最明确的共性：SEED 的 expert-action top-1 增益几乎全部来自原本就接近边界的错误。
它没有把大距离 expert mismatch 普遍拉回。

## 7. Task 与 action stage 切片

### 7.1 task family top-1 delta

| task | seen delta | unseen delta |
|---|---:|---:|
| look_at_obj_in_light | +13.46 点 | +7.53 点 |
| pick_and_place_simple | +3.53 点 | +7.14 点 |
| pick_clean_then_place | +1.25 点 | +3.26 点 |
| pick_cool_then_place | 0 点 | -4.20 点 |
| pick_heat_then_place | +5.41 点 | +11.76 点 |
| pick_two_obj_and_place | -8.33 点 | +14.29 点 |

`pick_two` 在当前专家数据里每个 split 只有一个 episode，不能作稳定 task 结论。cool 是唯一在
unseen 明显回退的 family。

### 7.2 action verb top-1

| verb | seen base → SEED | unseen base → SEED |
|---|---:|---:|
| heat | 33.33% → **100%** | 0% → **100%** |
| open | 88.71% → 95.16% | 80.00% → 89.09% |
| take | 88.24% → 91.18% | 85.71% → 94.29% |
| move | 96.77% → 100% | 96.55% → 100% |
| go | 22.27% → 24.37% | 14.24% → 16.97% |
| close | 0% → 0% | 0% → 0% |

最强变化集中在少量 transformation 和 object-manipulation 决策，尤其是 `heat`。大量导航状态只
小幅改善；`close` expert action 完全没有被两个模型模仿。

`close=0%` 反而说明为什么不能把 expert next-action accuracy 当任务成功率：成功策略可以不关闭
已打开容器，或沿另一条合法轨迹完成任务。SEED 的高 episode success 不需要逐步复制这一条
hand-coded expert trace。

## 8. 本轮得到的实验结论

1. **SEED 权重确实改善了同接口下的 expert top-1，但只有 3--4 点，不是从 30%--40% 到 90%。**
   这与完整任务 success 的巨大提升不是同一个现象。
2. **top-1 改善与 NLL 改善分离。** SEED 的 expert NLL 反而更差，说明它不是普通行为克隆式
   分布匹配。
3. **增益由少量边界状态产生。** SEED 只救回约 9%--10% 的 base errors，而且高度集中在最小
   repair distance 的第一四分位。
4. **大距离 expert mismatch 基本没被修复。** unseen 中距离后 50% 的错误 rescue 为 0。
5. **成功训练 delta 不是 prompt skill delta。** 二者 cosine 只有 0.09--0.14。
6. **成功训练 delta 也不是最短 expert edit。** 其范数中位数约为最小修复的 6.7 倍，cosine
   中位数约 0.05。
7. **关键操作比逐步 imitation 更重要。** `heat/open/take` 的集中改善与 `close` 始终不匹配同时
   存在，支持“少数任务关键边界改变即可显著影响长期 return”的解释。

更精确的研究问题因此变成：哪些 action-score 边界是任务因果关键的，而不是哪些状态与唯一
expert action 最相似。

## 9. 不能从本轮推出什么

- 不能用本轮 44.59%/35.86% 复现或否定官方 91.8%/86.2% episode success；prompt、生成协议、
  metric 和 trajectory 都不同。
- 不能把 `delta_seed` 全部解释成 SEED 的 OPD 作用；checkpoint 同时接受了 SFT、GRPO 与
  self-distillation 等训练信号。
- 最小 repair 是到唯一 expert action 的几何距离，不是到所有成功动作集合的距离。
- 还没有观测真正词表位置 `z[t,v]` 的修复，也没有执行参数区域写回实验。
- 当前结果来自一个 base/SEED checkpoint pair，需要其他 seed/checkpoint 才能判断训练 delta 的
  普遍性。

## 10. 下一步只做测量的实验顺序

1. 在 rescued、unresolved、harmed 三组中抽取匹配状态，保存每个动作 token 位置的完整
   `∂L_repair/∂z[t,v]` 稀疏 top-k 数字。
2. 对同一个 verified output repair，分别观测 tied embedding/output matrix、最后 attention、
   最后 MLP、最后 4 层和 RMSNorm 的梯度与极小写回响应。
3. 只测三个量：原状态修复、同类状态迁移、无关 control KL。
4. 比较 base 与 SEED 对同一 verified action 的梯度 norm 和逐层分布，检验“已学会后梯度变小”
   这一假设。
5. 最后才按经验类别比较跨状态梯度 cosine 是否高于随机不同经验。

这些都是观测实验，不预设最终算法。

## 11. 复现入口

- checkpoint scorer：`scripts/score_alfworld_checkpoint.py`
- 最小修复定义：`src/self_evolve/seed_checkpoint_delta.py`
- paired analyzer：`scripts/analyze_seed_checkpoint_delta.py`
- SEED 分片输出：`outputs/seed_checkpoint_plain/valid_{seen,unseen}_shard_*/`
- 完整自动分析：`outputs/seed_checkpoint_plain/analysis/results.json`
- 状态级指标：`outputs/seed_checkpoint_plain/analysis/valid_{seen,unseen}_trace.jsonl`
- 自动简表：`outputs/seed_checkpoint_plain/analysis/REPORT.md`
- 官方 checkpoint 本地目录：`model_cache/Seed-AlfWorld-3B/`

当前测试：`15 passed`。

