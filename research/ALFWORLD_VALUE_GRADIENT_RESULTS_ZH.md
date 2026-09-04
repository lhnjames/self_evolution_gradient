# ALFWorld 长期价值目标下的 Base/SEED 参数梯度结构

更新时间：2026-09-03（Asia/Shanghai）  
状态：256 个分层状态完成、分析与归档完成

## 1. 为什么重做梯度定义

旧 RMSNorm 探针把 ALFWorld 官方 expert action 当作唯一 verified target。环境干预实验已经证明
expert 动作在 seen/unseen 只有 40.77%/36.62% 属于最高折扣长期价值动作，因此旧目标不能回答
“有价值的输出变化是否对应可重复参数梯度”。

本实验改用两个价值目标：

```text
value_expectation:  L = -E_{a ~ softmax(action_score)}[V_expert-recovery(s,a)]
value_optimal_set:  L = -log sum_{a in argmax V(s,.)} P(a|s)
```

`expert_nll_control` 只保留为对照，不作为正确目标。

## 2. 数据、参数区域与数值协议

- seen/unseen 各 128 个分层状态，共 256 个；覆盖 32/29 个独立 game trial；
- 分层覆盖 task type、expert transition、价值改善/损害、是否命中价值最优；
- Base/SEED 同状态、同候选、同价值标签分别反向传播；
- BF16，batch=4，动作分数最大复现误差均 `<1e-6`；
- 精确梯度并集共 619,606,016 个参数；
- 参数区域：tied embedding/output、最后层 attention、最后层 MLP、最后 block、最后四个 block、
  全部 RMSNorm，以及去重并集；区域有意重叠；
- 单状态 Base/SEED 梯度夹角、梯度与真实 `delta theta = theta_SEED - theta_Base` 的内积均在完整
  参数区域上精确计算；
- 跨状态方向只保存 4096 维固定坐标 sketch，属于近似诊断，不能当成完整梯度 cosine。

## 3. 训练前后价值梯度是否保持同一方向

主 `value_expectation` 目标的 Base/SEED 精确梯度 cosine：

| parameter group | valid_seen [episode 95% CI] | valid_unseen [episode 95% CI] |
|---|---:|---:|
| all RMSNorm | 0.3293 [0.3025, 0.3603] | 0.3052 [0.2831, 0.3286] |
| last attention | 0.7079 [0.6765, 0.7382] | 0.7253 [0.6876, 0.7579] |
| last MLP | 0.7816 [0.7517, 0.8091] | 0.7960 [0.7710, 0.8203] |
| last block | 0.7786 [0.7484, 0.8066] | 0.7933 [0.7676, 0.8176] |
| last four blocks | 0.7658 [0.7382, 0.7912] | 0.7685 [0.7449, 0.7914] |
| tied embedding/output | 0.5600 [0.5297, 0.5931] | 0.5428 [0.5123, 0.5732] |
| selected union | 0.6623 [0.6359, 0.6896] | 0.6368 [0.6089, 0.6644] |

最后层与最后四层的价值梯度在 Base/SEED 间高度同向，说明训练后“如何局部提高这些状态的价值”
没有被彻底改写；RMSNorm 与 tied output 的方向变化更大。

## 4. 真实 SEED 参数变化是否沿价值下降方向

以下是完整参数上的 `cos(-g_Base, delta theta_SEED)`；正值表示真实参数差分与局部价值改善方向
同向：

| objective / group | valid_seen [episode 95% CI] | valid_unseen [episode 95% CI] |
|---|---:|---:|
| value expectation / selected union | +0.000243 [-0.000008, +0.000482] | +0.000246 [+0.000060, +0.000437] |
| value-optimal set / selected union | +0.000574 [+0.000374, +0.000765] | +0.000714 [+0.000479, +0.000924] |
| expert NLL control / selected union | +0.000027 [-0.000213, +0.000281] | +0.000153 [-0.000112, +0.000414] |

价值最优集合目标在两个 split 上都比 expert control 更稳定地同向，但绝对 cosine 只有
`5.7e-4–7.1e-4`。因此严格结论是：

> 真实 SEED 参数差分中可以检测到极弱的长期价值下降投影，但整体几乎与单状态价值梯度正交；
> 不能声称 SEED 参数更新等于这些局部价值梯度的累加。

## 5. 经验内化后梯度大小是否下降

`value_optimal_set` 的 Seed/Base 梯度 norm 比：

| group | valid_seen [episode 95% CI] | valid_unseen [episode 95% CI] |
|---|---:|---:|
| selected union | 0.7443 [0.6866, 0.8074] | 0.7129 [0.6672, 0.7608] |

SEED 在“把概率推入价值最优集合”这一硬目标上所需梯度约减少 25%–29%，两个 split 都稳定。
但平滑 `value_expectation` 的 norm 比为 0.995 [0.818,1.203] / 0.941 [0.773,1.145]，没有明确
下降。模型更接近价值最优集合，不代表所有概率加权价值梯度都消失。

## 6. 价值目标与 expert 模仿是否同一方向

在 selected union 的固定坐标 sketch 上：

| comparison | valid_seen | valid_unseen |
|---|---:|---:|
| Base value expectation × value-optimal set | 0.6139 | 0.6219 |
| Base value expectation × expert NLL | 0.2164 | 0.2257 |
| SEED value expectation × value-optimal set | 0.6681 | 0.6455 |
| SEED value expectation × expert NLL | 0.1646 | 0.2065 |

两个独立价值目标明显比 value 与 expert-NLL 更一致，再次说明长期价值学习不能退化成唯一 expert
动作模仿。这里是坐标 sketch cosine，数值用于对照方向，不是完整参数精确 cosine。

## 7. 跨经验共性：存在弱结构，但远未达到可积累更新的证据门槛

主价值梯度的 selected-union 坐标 sketch，在排除同 episode 配对后：

| model/split | same task + same verb | different task + different verb |
|---|---:|---:|
| Base seen | 0.0770 | -0.0089 |
| SEED seen | 0.1698 | +0.0114 |
| Base unseen | 0.1396 | -0.0063 |
| SEED unseen | 0.1609 | +0.0256 |

同任务同动作阶段比完全不同类别更同向，且 seen/unseen 都出现；这支持“经验梯度可能有条件共性”。
但 pair 数并非独立样本、使用的是坐标 sketch、组内方差很大，因此只能称为弱结构证据，尚不能
直接开始设计积累算法。

## 8. 当前严格结论

1. 长期价值梯度与 expert-NLL 梯度不是同一个目标。
2. Base 与 SEED 在最后四层仍共享相当一致的局部价值改进方向。
3. SEED 对 value-optimal-set 的梯度 norm 稳定变小，符合部分经验已内化的解释。
4. 真实参数差分仅含极弱的价值梯度投影，绝不能解释为局部梯度的直接累加。
5. 同 task + 同 verb 的跨经验梯度有弱共性。后续真实小步写回与 held-out transfer/control
   已完成：同 verb 的价值迁移在末层 attention/MLP/末四层得到因果支持，详见
   `research/ALFWORLD_VALUE_WRITEBACK_RESULTS_ZH.md`；它仍只是概率级微小效应，不等于可直接
   累积的训练算法。

## 9. 可复查产物

- 自动报告：`outputs/value_gradient_probe_v1/analysis/REPORT.md`
- 全分析：`outputs/value_gradient_probe_v1/analysis/results.json`
- 各 split/shard 精确指标：`outputs/value_gradient_probe_v1/{valid_seen,valid_unseen}_shard_*/results.json`
- 固定坐标梯度 sketch：同目录 `gradient_sketches.npz`
- 探针：`scripts/probe_value_parameter_gradients.py`
- 分析器：`scripts/analyze_value_parameter_gradients.py`
