# 300× 技能条件梯度 bank × SEED 横向比较

所有值均来自相同 971 状态、FP32 Base、相同 prompt/tokenizer/candidates/value。

| 路由 | split | 梯度 bank 相对价值提升 | 梯度 bank top-value 相对提升 | SEED 相对价值提升 | SEED top-value 相对提升 | 30% 全种子通过 |
|---|---|---:|---:|---:|---:|---|
| base_predicted_skill | valid_seen | +4.89% ± 0.58% | +23.38% ± 14.48% | +4.96% | +38.85% | 否 |
| base_predicted_skill | valid_unseen | +3.15% ± 1.01% | -5.21% ± 14.94% | +4.63% | +19.05% | 否 |
| oracle_skill | valid_seen | +10.23% ± 0.58% | +57.01% ± 11.29% | +4.96% | +38.85% | 否 |
| oracle_skill | valid_unseen | +7.69% ± 1.30% | +8.93% ± 12.96% | +4.63% | +19.05% | 否 |
