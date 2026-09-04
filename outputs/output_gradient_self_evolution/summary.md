# Output-Gradient Self-Evolution：当前总结果

日期：2026-09-04

## 核心判定

梯度迁移、强度放大和 held-out family 零样本修复已经成立；但无条件 output-head 写回的 protection harm 过高，且严格同状态复核后“第一次新-family反馈继续改善”并未成立。最新 Repair-Space Characterization 进一步发现：强安全单方向只在 1/4 seed 的 full-panel oracle 中出现，validation-only 选择后隐藏测试为 0/4。当前结果证明了可迁移 gradient patch，还没有证明存在稳定、可泛化、可安全提交的 output-head repair direction，也没有证明持续 feedback-driven self-evolution 闭环。

精确 parameter/logit equivalence 已通过；因此这里所有 virtual repair 都等价于独立 output head 的真实参数写回。Phase 0 只在 `close` 上显示出稳定的线性表示能力，所以 H6/H8 聚焦 `close`，没有把失败的 `go/open` 混入平均。

## Phase 0：表示充分性

- `go/open/close` 联合 oracle head：final-holdout 相对价值 +20.46% ± 7.09%，未过 30%。
- `close` 独立 oracle head：+44.56% ± 10.21%，稳定越过 value gate。
- `go/open` 独立 oracle 分别只有约 `+14.79% / +9.12%`，因此后续不纳入 output-only 正结论。
- Qwen tied head 已复制解绑；backbone、input embedding、bias 全部冻结。

## 300x（参数 L2=0.18）

| 方法 | unseen same-skill 相对价值 | 正迁移率 | top-value 修复 | protection harm |
|---|---:|---:|---:|---:|
| Single gradient | +24.38% ± 10.75% | +71.53% ± 8.29% | +36.81% ± 23.39% | +37.50% ± 15.47% |
| Mean12 | +25.91% ± 6.02% | +100.00% ± 0.00% | +75.00% ± 16.67% | +16.67% ± 19.25% |
| Transfer-weighted | +24.72% ± 6.58% | +83.33% ± 19.25% | +75.00% ± 16.67% | +16.67% ± 19.25% |
| Transfer+harm weighted | +24.74% ± 6.58% | +83.33% ± 19.25% | +75.00% ± 16.67% | +16.67% ± 19.25% |
| SEED（完全相同 holdout） | +16.70% ± 6.02% | +66.67% ± 0.00% | +100.00% ± 0.00% | — |

## 900x 强度压力测试（参数 L2=0.54）

Mean12 的 unseen relative value 为 +28.56% ± 5.88%，protection top-value harm 为 +33.33% ± 27.22%。

## 1200x 门槛测试（参数 L2=0.72）

Transfer+harm weighted 的 unseen relative value 为 +30.48% ± 6.68%，Mean12 为 +28.39% ± 5.69%；前者超过 Mean12 并跨过平均 30% value gate。但 protection top-value harm 达到 +50.00% ± 19.25%，不满足安全提交条件。

Protection-nullspace 参数投影在 1200x 将 protection harm 降为 +0.00% ± 0.00%，但价值回落到 +24.59% ± 6.79%。推到 3000x 后价值仍只有 +24.98% ± 6.77%，harm 回升为 +8.33% ± 16.67%；说明继续加剂量不能同时跨过 30%/2% 两道门。

## Sequential self-evolution

Transfer+harm (G_t) 在 `K=4` 达到峰值 +31.88% ± 6.47%；最终 K=12 为 +30.48% ± 6.68%。曲线不是单调上升，说明新增经验在 4 条以后出现饱和/冲突，不能把“持续加入”直接等同于“持续进化”。

新增 Gradient Scope Tomography 表明：主要 scope formation 发生在前三条经验。K=1→2 时梯度平均旋转 `59.39°`，29.2% 的独立 close 状态跨越 `|ΔV|=0.01` 的作用域边界；K=2→3 又旋转 `39.74°`，但只有 4.2% 换区。K=4→12 虽累计旋转 `25.21°`，close 正作用域 Jaccard 仍为 `0.958`。因此 K=4 后主要是已有作用域内的强度重分配和饱和，不是正作用域持续扩张。

## Held-out new skill family

完全排除 `pick_cool_then_place_in_recep` family 后：

| 方法 | Zero-shot unseen-family 相对价值 | 正迁移率 | top-value 修复 | protection harm |
|---|---:|---:|---:|---:|
| Mean9 historical gradients | +48.73% ± 3.37% | +100.00% ± 0.00% | +85.00% ± 19.15% | — |
| Evolved9 historical gradients | +49.64% ± 6.38% | +100.00% ± 0.00% | +80.00% ± 16.33% | +58.33% ± 16.67% |
| SEED（同一 unseen family） | +23.51% ± 3.48% | +85.00% ± 10.00% | +60.00% ± 16.33% | — |

原报告中“第一次 failure 加入反馈后为 +51.53%”与 zero-shot 的 +49.64% 使用了不同状态集合：反馈状态在前者中被移除。严格只比较两次评测共有的 4 个 unseen states/seed 后，反馈的额外绝对效应为 `−0.000322 ± 0.003551`，相对效应为 `−0.07% ± 0.76%`，4 个 seed 仅 1 个为正，正作用域 Jaccard 为 `1.0`。因此 historical gradient 的 zero-shot repair 仍成立，但第一次 feedback 没有带来可复现的进一步改善或 scope expansion；`feedback → evolve` 尚未被证明。

## 迁移机制（1224 个有向 pair）

- mean-hidden cosine vs transfer：Spearman `0.100 ± 0.079`。
- mean-delta cosine vs transfer：Spearman `0.512 ± 0.088`。
- full multi-token gradient cosine vs transfer：Spearman `0.509 ± 0.096`。
- full-gradient cosine 最低/最高四分位正迁移率分别为 `78.8% / 100.0%`。

在 `close` 内部，hidden cosine 已普遍很高，区分 transfer 的主要信号来自 repair direction（delta）；强写回时 gradient cosine 仍有预测力，但弱于此前微扰区间。

## Gradient Scope Tomography（新增 7,200 个条件）

实验在 4 个 seed 上固定每个 seed 30 个状态，扫描 `K=1..12 × 300/600/900/1200/3000x`。每代梯度方向统一由 1200x validation fitness 决定，随后所有剂量复用同一方向，因而 direction evolution 与 magnitude effect 被严格分离。作用域使用 `|ΔV|>0.01` 定义。

必须严格区分两层结论：

1. frozen backbone、untied output head 下，multi-token logit 迁移公式是精确的；真实 writeback/virtual repair 已以最大 action-score 误差 `1.07×10^-4` 验证。
2. logit influence 转化成 `ΔV>0` 还是 `ΔV<0` 是经验性的 nonlinear validity scope，不能由 mean hidden similarity 单独推出。

### K 轴的作用域

| K | holdout 相对价值 | 独立 close 正覆盖 | protection 负覆盖 |
|---:|---:|---:|---:|
| 1 | +5.17% | 70.8% | 50.0% |
| 2 | +27.03% | 91.7% | 41.7% |
| 3 | +30.43% | 95.8% | 41.7% |
| 4 | +31.88% | 95.8% | 44.4% |
| 8 | +31.37% | 100.0% | 47.2% |
| 12 | +30.48% | 100.0% | 41.7% |

### Direction conflict 与 dose leakage

- 固定 G4：protection 负覆盖从 300x 的 `36.1%` 增至 1200x 的 `44.4%`。
- 固定 G12：从 `36.1%` 增至 `41.7%`。
- 一阶 compatibility 在 protection 上为负的比例：G4 `55.6%`，G12 `58.3%`。
- 300x→1200x 才新增的 dose-emergent harm：两者均为 `13.9%`。
- 一阶预测为正、但 1200x 真实转负的 nonlinear reversal：仅 `2.8%`。

因此当前高 harm 主要是 structural repair conflict，而不是“正确方向推得太远”。减小剂量只能减弱伤害，无法消除冲突状态。

### 一阶作用域预测

对混合的独立 close + protection panel：

| 梯度 | 300x Spearman | 1200x Spearman | 3000x Spearman |
|---|---:|---:|---:|
| G4: `<G,g_j>` vs ΔV | 0.951 | 0.897 | 0.860 |
| G12: `<G,g_j>` vs ΔV | 0.953 | 0.896 | 0.856 |

300x 的一阶符号准确率为 `95.0%`，1200x 为约 `82%–83%`，3000x 为约 `77%–78%`。相比之下，logit influence norm 与 utility 的 Spearman 只有约 `0.41–0.47`。所以“作用得强”不等于“作用得对”；当前最有效的 validity 描述量是 target-gradient compatibility，而不是 mean hidden cosine 或 influence magnitude。

## Gradient Scope Topology（新增）

对每个 seed 的全部 18 个 episode-distinct close failures 构造 (C_g,C_\delta,C_h)，并在 300x/1200x 实测所有有向 close→close transfer，共 `2,448` 个非对角 transfer pair。

### Transfer 是否对称

- 300x：(T_{ij}) 与 (T_{ji}) 的 Spearman 为 `0.674 ± 0.085`，原始符号一致率 `85.5%`，明确相反符号仅 `1.0%`。
- 1200x：Spearman 降至 `0.167 ± 0.163`，明确相反符号升至 `7.7%`。

所以低强度 compatibility sign 可近似作为 signed similarity，但真实 transfer magnitude，尤其强剂量下，必须视为有向图。

### Conflict topology 来自哪里

| 空间 | silhouette | within cosine | between cosine | 负边跨分区比例 |
|---|---:|---:|---:|---:|
| Full gradient | 0.397 | 0.608 | −0.066 | 99.4% |
| Delta signature | 0.402 | 0.613 | −0.063 | 99.4% |
| Hidden signature | 0.285 | 0.959 | 0.916 | 无负边 |

Full-gradient 与 delta 的 edge correlation 为 `0.996`，cluster ARI 为 `1.000`；与 hidden cluster 的 ARI 为 `−0.026`。这强力支持 repair compatibility 主要位于 delta space。

但 hard speciation 尚未成立：两个 signed partitions 的跨 seed episode-level ARI 只有 `0.318`，与 task type 的 NMI 仅比随机置换高 `0.023`。当前只能说存在局部 repair modes，不能把它们写成稳定的两个 skill species。

### 有效维度

90% centered variance 所需分量数：

- delta signature：`5.0`；
- full gradient：`6.0`；
- hidden signature：`9.75`；
- empirical transfer response：`2.5`。

因此当前证据更偏向低秩 Repair-Direction Basis，同时保留“basis coefficient 可能形成 signed branch”的可能性。

### Novelty 与边际进化

44 个 sequential transitions 中，span-residual novelty 与边际 holdout gain 的 Spearman 为 `0.622`。K≥4 后，高 novelty 一半的平均边际 ΔV 为 `+0.00229`，低 novelty 一半为 `−0.00215`。

不过 held-out feedback 是关键反例：它仍有 `0.398 ± 0.028` 的 full-gradient span residual，却没有新增任何 unseen positive-scope state，反而丢失 `18.75%`，合并后的共同 target 效应为 `−0.00044 ± 0.00218`。所以 algebraic novelty 不是 Merge 的充分条件，必须进一步定义 beneficial scope novelty。

逐状态 transition attribution 也与该结论一致：`negative→positive` 状态中新梯度与 target 的 mean cosine 为 `+0.337`，而 `positive→negative` 为 `−0.361`。作用域边界移动由新 component 的 compatibility 符号直接解释。

## Repair-Space Characterization（新增）

本轮不设计新算法，直接研究在 12 个 source output-gradient span 中，是否存在同时满足：

1. 相似 skill target 相对长期价值提升 `≥30%`；
2. protection top-value harm `≤2%`；
3. protection 中 `ΔV<-0.01` 的比例 `≤2%`。

每个 seed 的离散 protection 数量使两个 2% 条件实际上都要求零个违规状态。四个 seed 共一阶预筛 240 万个单位方向；GPU 对 4,585 个方向–强度候选做完整 multi-token softmax 评分，得到 137,550 个逐状态真实响应。搜索强度覆盖 `300/600/900/1200/1800/2400/3000x`。

### Single-direction existence 与泛化

| seed | full-panel 强安全方向 | best safe rank | 强度 | best safe target gain | validation-selected hidden gain | hidden top harm |
|---:|:---:|---:|---:|---:|---:|---:|
| 20260904 | 否 | — | — | — | 8.76% | 11.11% |
| 20260921 | 是 | 5 | 300x | 36.65% | 23.37% | 33.33% |
| 20260938 | 否 | 4 | 300x | 10.96% | 34.11% | 33.33% |
| 20260955 | 否 | 4 | 600x | 25.85% | — | — |

`full-panel oracle` 是允许看完整固定面板的 capacity 上界，不是可部署选择。它只在 1/4 seed 跨过 30%/2%；仅用 3 个 transfer-validation 与 3 个 harm-validation 选方向后，隐藏 holdout/protection **0/4** 同时过门槛。因此不能声称已经找到比 SEED 更好的安全修复方法。

### Feasible rank 与 connectivity

- rank 1–2 在所有 seed 中都没有找到满足 protection 条件的方向。
- seed 20260921 从 rank 3/4 的安全收益 `27.53%/29.49%`，到 rank 5 才达到 `36.65%`。
- 唯一成功 seed 中有 8 个强安全候选，最大两两夹角仅 `17.44°`。
- 围绕最佳 rank-5 点额外做 541 点 GPU 局部切平面扫描，112 点（20.7%）通过强安全门槛，形成 1 个连通区域；中心 2° 内全部可行，狭长安全楔形至少延伸至 30°。
- 因此它不是孤立采样噪点，而是连续但高度各向异性的高维 basin；问题在于该 basin 没有跨 seed 稳定复现。

### Conflict、trajectory 与 feedback

Per-state optimal coefficient 对 target/protection 的留一最近质心 balanced accuracy 只有 `71.9% ± 8.6%`，说明两类状态在 rank-6 repair space 中部分可分但明显重叠。当前 G4→G12 的平均余弦为 `0.918`，held-out feedback 与 G12 的平均余弦为 `0.875`；后期经验和新反馈主要强化既有方向，并没有自动把轨迹推向安全 Pareto 区域。

因此对三种可能性的当前判定是：

- **稳定 single safe direction：不支持。**
- **低维连续组合空间：单 seed 部分支持，但至少需要约 rank 5，而不是 2–3 维。**
- **统一 output 修改不可稳定泛化：当前跨 seed 证据最支持，但仍需 full rank-12 与 deeper-layer span 对照才能区分搜索截断和 output-head capacity 上限。**

SEED 在同一完整目标面板平均相对价值提升为 `20.18% ± 5.72%`，protection top-value harm 为 0，但 `22.92%` protection 状态的 `ΔV<-0.01`，同样不满足本轮更严格的 value-safety gate。唯一 full-panel oracle 比对应 SEED 的 `24.37%` 高 `12.28` 个百分点，但因为使用了完整面板选方向，不能作为公平方法优势。

## 对预注册 13 个问题的回答

1. Output layer 是否有表示能力？**部分有**：`close` 有，`go/open` 在当前数据上不足。
2. 单 failure 梯度是否迁移？**是**，300x H6 同-skill 相对价值 +24.38% ± 10.75%。
3. hidden similarity 是否解释强度？**仅弱解释**，同一 `close` 家族内已饱和。
4. delta similarity 是否有额外解释？**有**，Spearman 约 `0.512`。
5. gradient cosine 为什么相关？它同时包含 hidden kernel 与 repair compatibility；完整 multi-token cosine 的 Spearman 约 `0.509`。
6. Mean 为什么比 single 强？它把正迁移率从约 72% 提到 100%，削弱 instance noise。
7. Transfer purification 是否超过 Mean12？300x **否**；1200x **是**，但只领先约 2.1 个相对百分点且不安全。
8. (G_t) 是否持续改善？**否**，K=4 达峰；tomography 进一步显示正作用域主要在 K≤3 形成，随后饱和而非持续扩张。
9. 能否 zero-shot 修复新 family？**能**，Evolved9 达 +49.64% ± 6.38%。
10. unrelated harm？未约束 1200x 为 +50.00% ± 19.25%；投影后最低可到 0%，但性能低于 30%。
11. 参数写回与 virtual repair 是否一致？**是**；最大 action-score 误差 `0.000107`。
12. 是否进入完整 episode / retraining 对比？**否**，没有同时满足 30% value 与 2% safety。
13. 最终定位：**部分可替代，只能作为受控的快速 patch；无条件 output-head 参数提交不足，feedback-driven evolution 也尚未成立。**

## 结论边界

- 单 output failure gradient 确实跨状态迁移，H6 成立。
- Mean12 保持 100% 正迁移并明显强于 single，但 300x 的平均相对价值仍低于 30%。
- cosine purification 和两种真实 transfer-fitness 加权均未超过 mean12，第一版“自净化”失败。
- Scope tomography 已将冲突进一步定位：多数 protection state 的一阶 compatibility 本来就为负，瓶颈主要是结构性 repair conflict，剂量泄漏是次要因素。
- 在 evolved gradient 超过 mean、protection harm ≤2% 之前，不进入完整 episode，也不能声称替代 SEED/SFT。
- 目前 evolved gradient 已在 1200x 和 held-out family 上超过 mean 且跨过 30%，但 protection harm 仍为 50%–58%，所以只能视为强力、需路由的 patch，不能直接 commit 到共享 head。
- 当前不预设最终方案是轻量 gate、branch 或 basis。Repair-space 结果要求下一步先比较完整 rank-12 output span 与 final-MLP/deeper span 的强安全可行率：若 full output span 可行，问题是 rank/搜索覆盖；若只有 deeper span 可行，问题是 output representation capacity；若二者都不能形成统一安全区域，才有证据转向 conditional/branch modification。

## 最新详细文档

逐状态作用域矩阵、四个 seed 的 heatmap、K/剂量分离、first-order/empirical scope 对照与新-family feedback 严格复核见：

`outputs/output_gradient_self_evolution/scope_tomography/gradient_scope_tomography.md`

冲突 signed topology、transfer asymmetry、有效秩、novelty 与 feedback span 分析见：

`outputs/output_gradient_self_evolution/scope_topology/gradient_scope_topology.md`

最新的 single-direction existence、rank-constrained Pareto、feasible-region connectivity、evolution trajectory 与 SEED 同面板对照见：

`outputs/output_gradient_self_evolution/repair_space/repair_space_characterization.md`
