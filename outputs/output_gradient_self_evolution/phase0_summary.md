# Output-Head Representation Sufficiency（Phase 0）

日期：2026-09-04

## 结论

Output representation sufficiency gate 未通过，应优先转向更深层表示编辑。

本实验冻结 Qwen2 全部 backbone 和 input embedding，复制并解除 tied `lm_head`，只训练无 bias 的独立 FP32 linear output head。四个 split seed 均使用 `12×3` source failures；每个 seed 内只由 transfer-validation 选择 epoch，final holdout 不参与选择。

## Final holdout（4 split seeds）

| 方法 | 相对长期价值提升 | failure→top-value 修复 | protection top-value harm |
|---|---:|---:|---:|
| Oracle untied output head | +20.46% ± 7.09% | +27.78% ± 27.96% | +47.22% ± 13.98% |
| SEED（相同 final holdout） | +8.05% ± 2.30% | +38.89% ± 6.42% | — |

注意：failure→top-value 修复列按 Base failure 中更新后选择更高价值动作的比例计算，因此它直接对应 30% decision gate。

## Per-skill oracle

| Skill | Final-holdout 相对价值 | top-value 修复 | 其他动作 harm | SEED 同 holdout 相对价值 |
|---|---:|---:|---:|---:|
| close | +44.56% ± 10.21% | +25.00% | +50.00% | +22.04% |
| go | +14.79% ± 10.48% | +25.00% | +66.67% | +5.09% |
| open | +9.12% ± 5.28% | +25.00% | +25.00% | +0.85% |

只有 `close` 稳定越过 30% value gate；`go/open` 没有。所有 per-skill 强写回都未满足 2% protection gate，因此后续 output-only 实验只把 `close` 当作“表示足够”的正对照，同时必须解决作用域与冲突。

## 预注册门槛

- Value ≥30%：`False`
- Decision repair ≥30 个百分点：`False`
- Protection harm ≤2%：`False`
- 表示能力足以继续 output-only 路线（Value 或 Decision gate）：`False`
- 三门同时满足：`False`

## 实验边界

Phase 0 测的是“固定 hidden representation 上是否存在足够好的线性决策边界”，不是 OGSE 已经成功。即使 oracle head 通过，后续仍必须证明单失败梯度迁移、真实 transfer-weighted evolution 超过 mean12、zero-shot new-skill repair，以及低干扰参数写回。
