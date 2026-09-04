# ALFWorld 长期价值目标的 Base/SEED 参数梯度结构

主目标 `value_expectation` 是 `-E_softmax(action score)[V_expert-recovery]`；
`value_optimal_set` 是最高价值动作集合的负对数概率；expert NLL 仅作为 control。
参数组有意重叠；跨经验方向使用 4096 维固定坐标 sketch，单状态 Base/SEED 与真实参数差分对齐为全参数精确内积。

## valid_seen

- 分层样本 128 个状态、32 个独立 trial。
- 动作分数最大复现误差 Base/SEED：9.54e-07/4.77e-07。

### 主价值目标

`descent × delta-theta cosine` 为正表示真实 SEED 参数变化与该状态的价值改进下降方向同向。

| parameter group | params | Base/SEED gradient cosine [CI] | Base descent×delta-theta [CI] | Seed/Base norm ratio [CI] |
|---|---:|---:|---:|---:|
| all_rmsnorm | 149504 | 0.3293 [0.3025, 0.3603] | -0.0015 [-0.0041, 0.0008] | 0.7488 [0.5443, 1.0535] |
| last_attention | 9439744 | 0.7079 [0.6765, 0.7382] | 0.0011 [-0.0000, 0.0021] | 0.9858 [0.8154, 1.1909] |
| last_block | 77076992 | 0.7786 [0.7484, 0.8066] | 0.0004 [-0.0002, 0.0008] | 0.9657 [0.8065, 1.1690] |
| last_four_blocks | 308307968 | 0.7658 [0.7382, 0.7912] | 0.0004 [0.0000, 0.0008] | 1.0766 [0.9180, 1.2451] |
| last_mlp | 67633152 | 0.7816 [0.7517, 0.8091] | 0.0003 [-0.0002, 0.0008] | 0.9677 [0.8068, 1.1704] |
| selected_union | 619606016 | 0.6623 [0.6359, 0.6896] | 0.0002 [-0.0000, 0.0005] | 0.9951 [0.8178, 1.2030] |
| tied_embedding_output | 311164928 | 0.5600 [0.5297, 0.5931] | 0.0000 [-0.0001, 0.0001] | 0.9249 [0.7303, 1.1968] |

### 价值目标与控制目标的坐标-sketch方向

| comparison | mean cosine |
|---|---:|
| base_value_expectation_vs_value_optimal_set_coordinate_cosine | 0.6139 |
| base_value_expectation_vs_expert_nll_control_coordinate_cosine | 0.2164 |
| seed_value_expectation_vs_value_optimal_set_coordinate_cosine | 0.6681 |
| seed_value_expectation_vs_expert_nll_control_coordinate_cosine | 0.1646 |

## valid_unseen

- 分层样本 128 个状态、29 个独立 trial。
- 动作分数最大复现误差 Base/SEED：9.54e-07/4.77e-07。

### 主价值目标

`descent × delta-theta cosine` 为正表示真实 SEED 参数变化与该状态的价值改进下降方向同向。

| parameter group | params | Base/SEED gradient cosine [CI] | Base descent×delta-theta [CI] | Seed/Base norm ratio [CI] |
|---|---:|---:|---:|---:|
| all_rmsnorm | 149504 | 0.3052 [0.2831, 0.3286] | 0.0003 [-0.0016, 0.0022] | 0.7115 [0.5584, 0.9014] |
| last_attention | 9439744 | 0.7253 [0.6876, 0.7579] | 0.0006 [-0.0006, 0.0017] | 1.0239 [0.8340, 1.2359] |
| last_block | 77076992 | 0.7933 [0.7676, 0.8176] | 0.0003 [-0.0000, 0.0007] | 0.9678 [0.7924, 1.2026] |
| last_four_blocks | 308307968 | 0.7685 [0.7449, 0.7914] | 0.0005 [0.0002, 0.0008] | 1.0558 [0.8803, 1.2573] |
| last_mlp | 67633152 | 0.7960 [0.7710, 0.8203] | 0.0003 [-0.0001, 0.0006] | 0.9688 [0.7886, 1.2011] |
| selected_union | 619606016 | 0.6368 [0.6089, 0.6644] | 0.0002 [0.0001, 0.0004] | 0.9406 [0.7735, 1.1449] |
| tied_embedding_output | 311164928 | 0.5428 [0.5123, 0.5732] | -0.0000 [-0.0001, 0.0000] | 0.8691 [0.7004, 1.0841] |

### 价值目标与控制目标的坐标-sketch方向

| comparison | mean cosine |
|---|---:|
| base_value_expectation_vs_value_optimal_set_coordinate_cosine | 0.6219 |
| base_value_expectation_vs_expert_nll_control_coordinate_cosine | 0.2257 |
| seed_value_expectation_vs_value_optimal_set_coordinate_cosine | 0.6455 |
| seed_value_expectation_vs_expert_nll_control_coordinate_cosine | 0.2065 |

