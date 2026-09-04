# Agent 自进化中的 Logic Distribution 调整：设计与实证

更新日期：2026-08-31

## 结论先行

要让 skill 自进化真正提升性能，不能把“生成/检索了一段更好的 skill 文本”直接等价为“策略变好了”。更可靠的优化对象是：在同一状态、同一候选逻辑 action 集上，显式测量 skill 造成的 likelihood-ratio，再通过 verifier gate 与 KL trust region 只保留有益的概率移动，最后把修复后的分布内化到一个很小的 residual head。

本项目已经把这个链路做成可运行原型，并在 8 个独立 seed 上得到：

| 方法 | strict success | tool accuracy | mean p(correct) | KL(q‖p) |
|---|---:|---:|---:|---:|
| Frozen plain policy | 6.25% ± 2.67% | 20.94% ± 1.29% | 0.2317 | 0 |
| 初始 skill v0.1 | 6.25% ± 4.23% | 25.63% ± 4.96% | 0.2270 | 0.0881 |
| 进化 skill v0.2 | 14.06% ± 6.12% | 55.94% ± 7.55% | 0.2521 | 0.1304 |
| verifier-gated skill（只门控，不加 value logit） | 17.81% ± 8.07% | 63.13% ± 6.09% | 0.2952 | 0.0799 |
| FO 内化，部署时无 skill/verifier | **18.44% ± 4.62%** | 56.88% ± 11.32% | **0.3034** | 0.0980 |
| ZO 内化，部署时无 skill/verifier | 8.75% ± 4.43% | 36.56% ± 11.10% | 0.2457 | 0.0339 |
| skill + oracle verifier projection | 100% | 100% | 0.6072 | 0.3500 |

最后一行只是可修复上界，不是可对外声称的 agent 能力。真正的部署结果是 FO/ZO internalized 两行。

最重要的诊断不是总分，而是分步概率：进化后的 v0.2 对 route step 的正确概率平均增加 0.0410，却对 answer step 平均降低 0.0063；所有 step 中仍有 50.8% 的正确 action 概率被 skill 拉低。它说明全局 skill prompt 会跨阶段污染分布，因此必须做 step-conditioned retrieval/gating，不能盲目蒸馏整个 skill prompt。

## 1. 已有 skill 自进化系统能给我们什么

### AutoSkill / SkillEvo

AutoSkill 的主链路是 trace → skill extraction → hybrid retrieval → add/merge/discard → prompt injection。AgentStream 内的实现也采用 BM25 + embedding 检索、skill 去重合并和版本更新。它的优势是工程生命周期完整，弱点是只看 skill 文本与最终结果，不观察 skill 究竟把哪个 action 的概率推高了。

AutoSkill 仓库新加入的 SkillEvo 更适合作为外层控制器：冻结 replay pool，编译 3–6 个 binary eval rule，在 `mutate_dev` 上搜索，在独立 `promotion_test` 上超过 champion 才晋级；默认最小提升阈值是 0.05。这个 promotion 结构应直接复用到本项目，但 promotion 指标应从单一成功率扩展为 success、cost、calibration、KL drift 与 regression slice。

本地代码：

- `references/AutoSkill/SkillEvo/runner.py`
- `references/AutoSkill/SkillEvo/evals.py`
- `references/AutoSkill/SkillEvo/mutators.py`
- `references/AgentStream/exgentic/src/exgentic/agents/autoskill/`

### EvoSkill

EvoSkill 对失败轨迹做归因，生成或编辑完整 skill folder，以 git program/branch 保存候选，用验证分数维护 frontier。论文报告 OfficeQA 60.6%→67.9%、SealQA 26.6%→38.7%，以及 SealQA skill 对 BrowseComp 的零样本 +5.3 个百分点。它最值得借鉴的是“候选 skill 是一个可回滚程序 + held-out promotion”，而不是其具体 proposer prompt。

本地代码：

- `references/EvoSkill/src/loop/runner.py`
- `references/EvoSkill/src/registry/manager.py`
- `references/EvoSkill/src/loop/helpers.py`

### SkillRL

SkillRL 将成功/失败 trajectory 蒸馏成 general 与 task-specific 两层 SkillBank，并提供 template 或 embedding retrieval；同时在 RL 过程中从新失败中追加 dynamic skills。仓库已经附带 ALFWorld、WebShop、Search 的实际 skill artifact，例如 `Systematic Exploration`、`Verify Early, Abort Fast`、`Attribute-Chaining Search`，可以直接作为下一阶段的真实 evolved-skill 输入，而不必先依赖我们自己的 analyzer。

本地 artifact 与代码：

- `references/SkillRL/memory_data/alfworld/claude_style_skills.json`
- `references/SkillRL/memory_data/webshop/claude_style_skills.json`
- `references/SkillRL/memory_data/search/claude_style_skills_search.json`
- `references/SkillRL/agent_system/memory/skills_only_memory.py`
- `references/SkillRL/agent_system/memory/skill_updater.py`

### SEED

SEED 提供了最关键的概率接口：对同一条 on-policy sampled action，分别在 plain context 和 skill-augmented context 下重打分，得到

`d_skill(s,a) = log p(a|s, skill) - log p(a|s)`。

然后把这个概率变化做 confidence gate，作为 dense token-level OPD signal 与 GRPO 联合训练。它把“skill 的文本价值”变成了“skill 对策略分布的行为影响”。但原实现的 teacher delta/gate 是 detach 的，主要传播 distillation gradient，不会反向训练 skill analyzer。

本地代码：

- `references/SEED/verl/trainer/ppo/core_algos.py::compute_opd_loss`
- `references/SEED/seed/analysis.py`
- `references/SEED/seed/prompting.py`

### SkillsBench 与 AgentStream

SkillsBench 的实验条件正好对应本项目需要的外部验证：no skill、curated skill、self-generated skill，并使用 deterministic verifier。当前 v1.1 registry 有 87 个真实工程任务；论文快照是 86 tasks / 11 domains。它适合测 skill 的边际价值。

AgentStream 则适合测“自进化是否随时间真的积累”：isolated、sequential、interleaved 三种 stream 能暴露 skill 污染、顺序依赖与跨任务遗忘。两者应分别作为静态 skill benchmark 和动态 evolution benchmark。

## 2. 建议的 Logic Distribution 核心算法

先把完整词表输出限制为当前状态合法的逻辑 action 集 `A(s)`，例如 tool call、plan operator、answer pointer、stop/continue，而不是直接对整个 vocabulary 做修复。

定义：

- `p0(a|s)`：冻结/旧策略的 action 分布；
- `ps(a|s)`：注入检索 skill 后，对相同 action 集重打分的分布；
- `d_skill = clip(log ps - log p0, -c, c)`：skill likelihood-ratio；
- `V(s,a)`：可执行 verifier、critic 或 rollout 得到的 action value；
- `g(s)`：skill reliability gate；
- `q(a|s)`：用于在线修复或离线蒸馏的 teacher 分布。

核心投影：

```text
qρ(a|s) ∝ p0(a|s) · exp{ρ [α g(s) d_skill(s,a) + β A(s,a)]}

choose largest ρ ∈ [0,1]
subject to KL(qρ || p0) ≤ δ
```

`A` 最好用 centered advantage 而非未归一化 reward。`g` 至少组合以下信息：

1. retrieval confidence / top-1 vs top-2 margin；
2. skill 的历史 promotion 置信区间；
3. 当前 verifier 下 `E_ps[V] - E_p0[V]` 是否为正；
4. skill 是否与当前 stage/action schema 匹配；
5. OOD 或冲突 skill 检测。

本原型采用第 3 项做严格 gate；因此 `verifier_gated_skill` 不是无监督 skill-only 成绩。真实部署可用 learned critic 代替 oracle verifier，但必须单独报告 critic error 与 false-positive repair rate。

## 3. 为什么这个设计比直接 prompt/RL 更稳

### 只接受“可归因”的概率移动

最终 reward 变好不代表 skill 生效，可能只是 sampling 方差。`d_skill` 能回答：skill 是否提高了本 step 正确/高价值 action 的概率，以及是否同时伤害其他 stage。

### KL trust region 防止 logit 过冲

此前的 ZO 实验出现了更新后错误 logit 过强、固定 beta 无法修复的 seed。把每次修复限制在 `KL(q||p0)≤δ`，再用 held-out promotion，可避免一次错误 verifier 或错误 skill 摧毁原策略。

### 保留 proposal/verify/accept 的 speculative 结构

plain head 是 cheap proposal；skill + verifier 是 expensive target；当 gate 不通过时原样接受 proposal，通过时才修复。它与 speculative decoding 的工程结构类似，但不是严格等价的 exact sampling，因为 verifier 并非目标语言模型。

### 能把外部 skill 变成部署时的小参数能力

训练时使用 `q`，部署时只保留 residual output head：

```text
L_distill = KL(stopgrad(q) || pφ)
          + λ KL(pφ || p_old)
          + μ L_outcome
```

当前 rank=32 的 head 只有 378 个可训练参数。后续可替换成每层 LoRA 或 MoE adapter，但第一阶段保持小参数空间更利于识别梯度是否真的有效。

## 4. 梯度在自进化闭环中的位置

### 一阶梯度：有 logits/hidden state 时应作为主线

对开源模型，`q` 已经是明确的 dense teacher，直接最小化 KL 最有效。本实验 FO strict 18.44%，接近在线 gated-skill 17.81%，并显著高于 ZO 8.75%。因此不要为了“自进化”概念而在可微问题上强行使用 ZO。

梯度路径应是：

```text
trajectory → analyzer/evolved skill ──┐
plain/skill rescore → safe q (detach) ├→ KL → residual head / LoRA
verifier/value/rollout ───────────────┘
```

skill generator 的更新走外层 mutation + promotion，policy/head 的更新走内层梯度；先不要端到端同时训练两者，否则很难区分 skill 质量、verifier 质量与 policy drift。

### ZO：留给不可微的完整系统目标

ZO 适合优化无法反传的环境成功率、API 模型、tool latency/cost 或离散 agent graph。建议按 ZO 论文采用：

- instance-specific LoRA / 小子空间，不扰动全模型；
- antithetic two-sided perturbation；
- 正负扰动复用相同 task、sampling seed 和环境初态（common random numbers）；
- 对完整 rollout 的 trajectory-conditioned answer NLL 或 verifier score 做有限差分；
- 成功新轨迹进入 buffer，再用 FO/SFT 合并到 shared model。

当前 ZO 只在 378 维 head 上用 8 directions，仍明显弱于 FO。下一轮应测 directions={8,32,64}、orthogonal directions、SVD/active subspace 与 common-random-number rollout；如果 ZO 不能在同等 rollout budget 下超过随机搜索/文本 mutation，就不应作为主算法。

### 长轨迹的 credit assignment

不要只修 final answer 分布。每一步记录：

```text
(state, candidate actions, p0, ps, q, selected action,
 retrieval id/confidence, verifier values, outcome, cost)
```

对 step t 使用 value-to-go 或 counterfactual repair gain：比较只替换该 step action、后续固定时的结果。SEED 的 token-level OPD 可以提供 dense signal；执行环境能枚举少量 actions 时，则优先用 candidate-level counterfactual value，信号更直接。

## 5. 评测框架与必须做的 ablation

### 开发层

当前 LogicRoute 用于快速检查分布和梯度，固定报告：

- route/answer/strict accuracy；
- correct-action probability、NLL、Brier、ECE；
- `d_skill` 正/负移动比例；
- KL、entropy、top-1 margin；
- repair/gate/rollback rate；
- FO/ZO 的 rollout、token 与 wall-clock budget。

### 真实 skill 层

1. ALFWorld：直接使用 SkillRL 已发布的 evolved skills；比较 no skill、all skills、retrieved skills、gated skills、internalized。
2. SkillsBench 小子集：每类选择 2–3 个 deterministic verifier task；比较 no/curated/self-generated/evolved。
3. AgentStream：最终跑 sequential/interleaved，观察 skill bank 增长后旧任务是否退化。

### 最小 ablation 矩阵

| 变量 | 取值 |
|---|---|
| skill | none / v0 / evolved v1 / distractor |
| use | prompt only / SEED delta / verifier gate / projection |
| trust region | no KL / δ=0.05 / 0.1 / 0.35 |
| internalization | none / FO head / FO LoRA / ZO LoRA |
| verifier | oracle / noisy / learned critic / outcome-only |
| stream | isolated / sequential / interleaved |

每次 promotion 需同时满足：held-out success 的置信下界提升、critical slice 无显著回退、KL/成本不过阈值；不能只看平均 reward。

## 6. 当前实现与复现

主要文件：

- `src/self_evolve/skills.py`：受控 skill v0→v1 lineage；
- `src/self_evolve/skill_evolution.py`：skill likelihood-ratio、verifier gate、KL projection、FO/ZO teacher；
- `src/self_evolve/skill_runner.py`：完整对照实验；
- `config/skill_experiment.yaml`：实验配置；
- `outputs/8gpu_skill_experiment/summary.json`：8-seed 汇总；
- `outputs/8gpu_skill_experiment/seed_*/trace.*.jsonl`：逐 step 的 plain/adjusted 分布。

运行：

```bash
cd /data/hanning/agent_self_evolution_gradient
.venv/bin/pytest -q
./scripts/run_8gpu_skill_sweep.sh
.venv/bin/python scripts/summarize_sweep.py outputs/8gpu_skill_experiment
```

## 7. 局限与下一步

当前 v0/v1 是由 verified train traces 以 deterministic analyzer 生成的受控 skill，目的是隔离“skill 文本改变分布”的因果链，不代表自然环境中的 analyzer 水平；verifier 是 oracle，100% projection 仅是上界。下一步最有价值的实验不是继续调 LogicRoute 数字，而是把同一 instrumentation 接到 SkillRL 的 ALFWorld artifact：先证明真实 evolved skill 的 `d_skill` 与成功率相关，再训练 step-conditioned residual head。

建议的最近里程碑是：在 ALFWorld 固定 100 train / 100 held-out episodes、同一 Qwen backbone、同一 action schema 下，完成 no-skill、SkillRL skill prompt、SEED delta、gated projection、FO/ZO internalization 五组 5-seed 对照。通过条件是 FO internalized 在无 skill prompt 部署时显著超过 plain，同时 verifier-noise slice 和 interleaved stream 无显著回退。

## 参考资料

- [SEED project](https://jinyangwu.github.io/seed/)
- [SEED code](https://github.com/jinyangwu/SEED)
- [Beyond the Capability Boundary: Zeroth-Order Optimization for Self-Evolving LLM Agents](https://arxiv.org/abs/2608.09292)
- [AutoSkill](https://github.com/ECNU-ICALK/AutoSkill)
- [EvoSkill](https://arxiv.org/abs/2603.02766)
- [SkillRL](https://arxiv.org/abs/2602.08234)
- [SkillsBench](https://www.skillsbench.ai/)
- [AgentStream](https://github.com/Jasper-Yan/AgentStream)
