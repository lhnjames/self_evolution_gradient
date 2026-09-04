# ALFWorld 当前实验状态报告

更新时间：2026-09-04  
当前阶段：相似技能失败梯度自净化与条件参数 delta bank  
状态：H6/H8/H10 的四种子 300× 强验证及 971 状态 Base/SEED 横评均已完成；30% 综合门槛未通过

## 1. 当前研究问题

本阶段以三个已被前序实验限定的事实设计并验证自进化方法：

> 单失败价值梯度能否修复相似 skill；多失败共同方向能否比单失败更强；技能条件隔离能否避免
> 强梯度累积造成的跨 skill 干扰，并达到至少 30% 的真实长期价值提升。

对比模型：

```text
M_base = Qwen2.5-3B-Instruct
M_seed = Jinyang23/Seed-AlfWorld-3B
```

## 2. 已完成实验

### 实验一：Base 与 SEED 的同状态候选动作评分

两个 checkpoint 使用完全相同的：

- 444 个 valid_seen 专家状态；
- 527 个 valid_unseen 专家状态；
- task description、最近三步 history 和 observation；
- admissible commands；
- direct-action plain prompt；
- Base Qwen tokenizer；
- BF16、batch size 4、多 token 长度归一化 sequence scorer。

输入一致性检查：

- 971/971 个 prompt token IDs 相同；
- 27,660/27,660 个 candidate action token IDs 相同；
- 两个 SEED safetensors 分片完整，远端与本地 SHA-256 相同。

### 实验二：Base 错误状态的最小决策修复距离

若最高错误动作是 `w`，expert action 是 `y`：

```text
gap = z_base(w) - z_base(y)

delta_min(y) = +gap/2
delta_min(w) = -gap/2

||delta_min||_2 = gap / sqrt(2)
```

该数值表示让 expert 与最高错误动作到达 top-1 tie boundary 所需的最小 action-score L2 编辑。

### 实验三：SEED 实际输出变化与最小修复比较

```text
delta_seed = center(z_seed - z_base)
```

已测量：

- `cos(delta_seed, delta_min)`；
- `||delta_seed|| / ||delta_min||`；
- 在最小修复方向上的 projection；
- rescued、unresolved、harmed 状态；
- task family 与 action verb 切片；
- 10,000 次 episode-cluster bootstrap。

## 3. 总体结果

| split | Base top-1 | SEED top-1 | delta [episode 95% CI] | Base NLL | SEED NLL |
|---|---:|---:|---:|---:|---:|
| valid_seen | 41.22% | **44.59%** | **+3.38 [+0.66, +6.35] 点** | 2.3499 | 2.4816 |
| valid_unseen | 31.69% | **35.86%** | **+4.17 [+1.28, +7.06] 点** | 2.5787 | 2.7868 |

SEED 显著提高了 expert next-action top-1，但 expert NLL 同时变差。它没有整体变得更像
hand-coded expert 分布，而是选择性改变了一部分 top-1 决策边界。

## 4. 决策转换

| split | Base 错误 | rescued | Base 正确 | harmed | top-1 净增加 |
|---|---:|---:|---:|---:|---:|
| seen | 261 | 27（10.34%） | 183 | 12（6.56%） | +15 |
| unseen | 360 | 32（8.89%） | 167 | 10（5.99%） | +22 |

SEED 没有修复大多数 expert mismatch。它只修复约 9%--10% 的 Base 错误，但破坏的原正确状态
更少，因此获得净 top-1 增益。

## 5. 最小修复距离

| split | metric | p10 | p25 | median | p75 | p90 |
|---|---|---:|---:|---:|---:|---:|
| seen | score gap | 0.409 | 1.000 | 1.729 | 2.504 | 4.250 |
| seen | minimal L2 | 0.289 | 0.707 | 1.223 | 1.771 | 3.005 |
| unseen | score gap | 0.339 | 0.933 | 1.698 | 2.731 | 3.863 |
| unseen | minimal L2 | 0.239 | 0.660 | 1.201 | 1.931 | 2.732 |

多数错误不是 0.1--0.3 量级的极轻微校准问题。只有最靠近边界的约 10% 错误处于这一范围；
中位错误需要约 1.2 的最小 L2 action-score 编辑。

## 6. SEED 是否沿最小修复方向变化

| split | mean cosine | median cosine | norm ratio median | projection median |
|---|---:|---:|---:|---:|
| seen | 0.082 | 0.056 | 6.65 | 0.413 |
| unseen | 0.075 | 0.046 | 6.70 | 0.272 |

SEED 的实际输出变化远大于最小 expert edit，而且大部分方向与最短修复近乎正交。因此：

```text
成功自进化 != 沿 expert 最短方向做小幅局部修复
```

但 rescued 状态具有明显不同的数值结构：

| split | group | minimal L2 median | cosine median | projection median |
|---|---|---:|---:|---:|
| seen | rescued | **0.450** | **0.149** | **2.088** |
| seen | unresolved | 1.278 | 0.049 | 0.367 |
| unseen | rescued | **0.277** | **0.172** | **4.464** |
| unseen | unresolved | 1.322 | 0.037 | 0.202 |

## 7. 最明确的共性：只修复靠近边界的错误

| distance quartile | seen rescue rate | unseen rescue rate |
|---|---:|---:|
| Q1，距离最近 | **26.15%** | **26.67%** |
| Q2 | 10.77% | 8.89% |
| Q3 | 3.08% | **0%** |
| Q4，距离最远 | 1.52% | **0%** |

SEED 的 expert-action top-1 增益几乎全部来自原本已经接近决策边界的错误。unseen 中，修复距离
后 50% 的错误没有一个被修复。

## 8. 关键 action slice

| action verb | seen Base → SEED | unseen Base → SEED |
|---|---:|---:|
| heat | 33.33% → **100%** | 0% → **100%** |
| open | 88.71% → 95.16% | 80.00% → 89.09% |
| take | 88.24% → 91.18% | 85.71% → 94.29% |
| move | 96.77% → 100% | 96.55% → 100% |
| go | 22.27% → 24.37% | 14.24% → 16.97% |
| close | 0% → 0% | 0% → 0% |

改善集中在少量 transformation 和 object-manipulation 决策。`close` 始终不匹配 expert，却不妨碍
SEED 官方模型获得高任务成功率，说明 expert next-action matching 不是完整任务价值。

## 9. 与自然语言 skill delta 的关系

SEED checkpoint delta 与静态 evolved-skill prompt delta 的平均 cosine：

| split | cosine |
|---|---:|
| seen | 0.0905 |
| unseen | 0.1367 |

成功训练后的权重变化与 prompt 中增加一条 skill 产生的方向并不相同。不能把 prompt skill delta
直接解释为已经 internalized 的训练方向。

## 10. 当前实验结论

1. SEED 在同一 direct-action 接口下只把 expert top-1 提高 3--4 点，而不是提高到 90%。
2. top-1 提升与 NLL 改善分离，说明 SEED 不是普通 expert behavior cloning。
3. 净增益来自少数边界状态，不是对所有错误的普遍修复。
4. SEED 实际 delta 大、分散，并非最小 expert repair。
5. 少数 `heat/open/take` 等关键操作显著改善，可能比逐步复制 expert trajectory 更影响长期任务。
6. 唯一 expert action 不是所有成功动作集合，因此 minimal expert repair 只是一个可测几何参照，
   不是长期任务价值本身。

## 11. 后续机制实验状态

### 实验四：真正词表 logits 数字

候选动作长期价值已经完成测量；逐 token/full-vocabulary 探针已全量完成 971 个状态。它保存各
生成位置的：

```text
target token logit / log probability / probability
full-vocabulary total variation
top probability donors / receivers
admissible next-token branch probabilities and branch values
action verb / object / receptacle-location / relation / index role contribution
```

BF16 对 batch/sequence shape 有可测的量化敏感性，因此探针严格复用此前 scorer 的 batch=4、
右填充和 completion-position 语义；全量 Base/SEED 动作分数最大复现误差均低于 `1e-6`。

完整结论见 `research/ALFWORLD_TOKEN_VALUE_RESULTS_ZH.md`。约 99% 的绝对 score 变化落在 action
verb、instance index 和 location；但稳定的正 branch-value delta 只出现在 action verb 和
location，index 的变化虽大，价值增益约为 0。

### 实验五：参数区域写回响应

已完成真实微小写回：seen/unseen 各 32 个源状态，三个目标、五个参数区、四类留出关系，共
960 个写回条件。每个条件把源候选分布校准到 `KL=1e-4`，只改内存参数并逐张量精确恢复；
baseline 重算与恢复最大误差均为 0。

平滑长期价值梯度在 seen 的末层 attention/MLP/末四层、unseen 的五个参数区都出现显著正的
留出价值迁移。最明确的结构是“异任务同动作”稳定为正，而“同任务异动作”基本为零，说明
可迁移成分主要跟 action verb / 操作类型走。等 KL 下它也显著优于 expert-NLL 对照。完整结果见
`research/ALFWORLD_VALUE_WRITEBACK_RESULTS_ZH.md`。

### 实验六：Base 与 SEED 的训练前后梯度

旧 RMSNorm expert-target 结果不作为正式回答。新的 `-E_p[V_expert-recovery]`、
value-optimal-set 与 expert-NLL control 已在 seen/unseen 各 128 个分层状态完成；结果见
`research/ALFWORLD_VALUE_GRADIENT_RESULTS_ZH.md`。最后四层的 Base/SEED 价值梯度 cosine 约
0.77，但真实 `delta theta(SEED)` 与单状态价值下降方向的 cosine 只有约 0.0002–0.0007。
同 task + 同 verb 的跨经验 sketch 共性高于完全不同类别，但仍需真实写回与 held-out transfer
验证。该验证现已完成并确认同 verb 的概率级因果迁移，但增益很小且 top-1 几乎不变；仍不能
直接解释成可线性积累的训练更新。

### 实验七：12 源共同方向的非微小剂量响应

已完成 seen/unseen、`go/open/close/take/move`、末层 MLP/末四层的正式剂量扫描。每个动作使用
12 个不同 episode 的源状态，三类留出各 6 个不同 episode；共 240 次源梯度、80 次多源写回和
2,400 个状态响应。梯度、合并、范数、写回和评分全部在 GPU 0–3 上执行。

选定的 10× 正式剂量为 MLP L2=`0.0125`、末四层 L2=`0.006`。该尺度的同动作留出价值增益为：

| split | last MLP | last four blocks |
|---|---:|---:|
| seen | **+0.009170** | **+0.008559** |
| unseen | **+0.009752** | **+0.009460** |

末层 MLP 的 10 个 split × verb 条件全部正迁移，并已产生 top-action 翻转；同动作翻转没有
top-value harm。30× 虽把同动作增益继续放大到约 0.024，但全留出 top-value harm 升至
3.3%–5.6%，故只保留为过量更新对照。完整结果见
`research/ALFWORLD_MULTISOURCE_DOSE_RESULTS_ZH.md`。

固定 10× 剂量的正式共同方向/冲突实验已经完成：12 源均值与每个单源使用等参数 L2，四个
split/region 条件共得到 2,640 个定向“精确梯度 cosine—真实交叉写回 ΔV”样本。同动作留出上，
多源比单源额外提高 `+0.003113` 至 `+0.004522`，四个 verb-cluster 95% CI 均大于 0；cosine 与
真实迁移的 Spearman 为 `0.851–0.888`。最低 cosine 四分位正迁移率 0%，最高两个四分位 100%。
结果通过进入在线重算梯度的顺序累积实验判据。完整结果见
`research/ALFWORLD_MULTISOURCE_GRADIENT_RESULTS_ZH.md`。

在线顺序累积已经完成：每个动作依次写入 12 个源状态，每一步在当前参数上重新计算长期价值
梯度，四个条件共 240 个在线步骤和 7,200 个写回后响应。第 12 步独立同动作留出平均提高
`+0.004455` 至 `+0.006811`，从第 1 步扩大约 8.3–9.6 倍；四个条件均出现 3.3% top-action
翻转且同动作 top-value harm 为 0。三个最终 verb-cluster CI 严格大于零，seen/MLP 的下界略过零。
但同任务异动作状态在 unseen 累积到 `−0.003453/−0.004443`，并且 `take` 在部分条件最终为负，
说明无约束累积仍存在冲突，尚不应保存候选 checkpoint 或宣称完整任务能力提升。

## 12. 最新：相似技能梯度自净化强验证

四个预注册采样种子、`go/open/close` 三种 Base 失败 skill、最后四层 FP32、300× 参数 L2：

| 方法 | 同 skill 失败留出 ΔV | 相对价值提升 | 正迁移率 | top harm |
|---|---:|---:|---:|---:|
| 12 个单失败梯度的平均效果 | +0.059774 | +11.11% | 81.9% | 2.1% |
| 12 源单位梯度共同方向 | **+0.097772** | **+18.17%** | **98.6%** | **0.0%** |
| cosine 硬剔尾共同方向 | +0.085545 | +15.89% | 90.3% | 0.0% |

普通共同方向在 12/12 个 `seed×skill` 条件中优于单源；但硬 cosine 剔尾比普通均值差，不能作为
净化算法。逐 skill 看，`close` 价值相对提高 35.27%，`go` 救回 33.33 个百分点的失败决策，
`open` 只有 5.79%；没有一个 skill 同时通过两道 30% 门。

技能条件 delta bank 的 971 状态 FP32 横评：

| 路由 | seen 价值 / top-value 相对提升 | unseen 价值 / top-value 相对提升 |
|---|---:|---:|
| Base 预测 skill | +4.89% / +23.38% | +3.15% / −5.21% |
| 已知 skill（机制上界） | **+10.23% / +57.01%** | **+7.69% / +8.93%** |
| SEED | +4.96% / +38.85% | +4.63% / +19.05% |

已知 skill 路由的概率价值显著超过 SEED，并大幅改善全局 300× delta 的 unseen 退化；但概率价值
仍远低于 30%，Base 预测路由也不稳定。因此“梯度经验可帮助相似 skill 自净化”成立，
“已经能替代技能训练”不成立。按门控规则不保存共享 checkpoint、不进入完整 episode。

## 13. 文件入口

- 完整专项报告：`research/ALFWORLD_SEED_CHECKPOINT_OUTPUT_DELTA_ZH.md`
- 当前状态报告：`research/ALFWORLD_CURRENT_EXPERIMENT_STATUS_ZH.md`
- 总实验报告：`research/ALFWORLD_EVOLVED_SKILL_COMPLETE_REPORT_ZH.md`
- 自动分析：`outputs/seed_checkpoint_plain/analysis/results.json`
- seen 状态级指标：`outputs/seed_checkpoint_plain/analysis/valid_seen_trace.jsonl`
- unseen 状态级指标：`outputs/seed_checkpoint_plain/analysis/valid_unseen_trace.jsonl`
- checkpoint scorer：`scripts/score_alfworld_checkpoint.py`
- 最小修复计算：`src/self_evolve/seed_checkpoint_delta.py`
- paired analyzer：`scripts/analyze_seed_checkpoint_delta.py`
- 长期价值完整报告：`research/ALFWORLD_ACTION_VALUE_RESULTS_ZH.md`
- 长期价值结果：`outputs/action_value_alignment/analysis/results.json`
- token/full-vocabulary 探针：`scripts/probe_action_token_logits.py`
- token/value 分析：`scripts/analyze_token_value_alignment.py`
- token/value 完整报告：`research/ALFWORLD_TOKEN_VALUE_RESULTS_ZH.md`
- value-gradient 完整报告：`research/ALFWORLD_VALUE_GRADIENT_RESULTS_ZH.md`
- value-gradient 探针：`scripts/probe_value_parameter_gradients.py`
- value-gradient 分析：`scripts/analyze_value_parameter_gradients.py`
- value-gradient 写回完整报告：`research/ALFWORLD_VALUE_WRITEBACK_RESULTS_ZH.md`
- value-gradient 写回探针：`scripts/probe_value_gradient_writeback.py`
- value-gradient 写回分析：`scripts/analyze_value_gradient_writeback.py`
- 多源剂量完整报告：`research/ALFWORLD_MULTISOURCE_DOSE_RESULTS_ZH.md`
- 多源剂量自动分析：`outputs/multisource_dose_response_v2/analysis/analysis.json`
- 多源共同方向/冲突结果：`outputs/multisource_value_gradient_v2/`
- 多源共同方向/冲突完整报告：`research/ALFWORLD_MULTISOURCE_GRADIENT_RESULTS_ZH.md`
- 12 步在线累积结果：`outputs/sequential_value_accumulation_v1/`
- 长期价值梯度研究总报告：`research/ALFWORLD_EXPERIENCE_GRADIENT_TOTAL_REPORT_ZH.md`
- 相似技能梯度自净化方法与结果：`research/ALFWORLD_SKILL_GRADIENT_SELF_PURIFICATION_ZH.md`
- H6/H8 机器结果：`outputs/skill_gradient_purification_300x_v1/analysis_v3/`
- H10/SEED 横评：`outputs/skill_gradient_routed_300x_v1/analysis/summary/`

当前全套测试：`37 passed`（远端正式实验前复跑）。
