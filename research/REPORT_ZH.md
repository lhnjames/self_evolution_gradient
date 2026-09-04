# Agent 自进化、输出分布修复与梯度路径调研

更新时间：2026-08-31

## 1. 结论先行

最适合继续开发的组合是：

1. **训练与 rollout 主框架采用 SEED 的 veRL 分支。** 它已经具备多轮环境、on-policy rollout、token log-prob 重打分、GRPO、FSDP/vLLM 训练链路，也已经实现与本方向最接近的 OPD loss。
2. **评测主框架采用 AgentStream/Exgentic。** 它把 benchmark、agent 和 evolution state 解耦，并支持 isolated、sequential、interleaved task stream；这比只做独立 task accuracy 更能发现灾难性遗忘和跨域负迁移。
3. **外层 candidate 管理借鉴 SE-Agent。** 保留 trajectory pool、按 iteration 落盘、断点恢复、operator 插件和候选 checkpoint 晋升机制，但不采用它以 API prompt evolution 为核心的内层实现。
4. **benchmark 分四层。** 开发期使用本仓库的 LogicRoute 检查概率和梯度；机制验证先用 ALFWorld；之后接 Search-R1；再把晋升后的方法接到 AgentStream 的 BFCL/AppWorld 等 task stream；GAIA/WebWalkerQA 只用于最终对齐 ZO 论文。
5. **第一阶段只优化轻量 logic/action distribution head。** 不直接扰动全模型，也不直接对 150K 词表做 repair。把 agent 在某状态的可执行候选动作归一成小的语义 action set，输出 residual logits。
6. **同时保留两条梯度路径。** 一阶路径把 verifier 修复后的分布 `q` 蒸馏给 head；零阶路径只读取完整 agent rollout 的标量损失，用有限差分更新同一个 head。两者必须在相同 rollout budget 下比较。
7. **必须增加 trust region。** 当前 8-seed 实验已经观察到 ZO logit 过冲，固定 verifier strength 无法始终纠正。下一版应加入 KL-to-old-policy、residual-logit clip、adaptive beta 和 held-out promotion gate。

## 2. 最新论文地图

### 2.1 核心三篇

| 论文 | 时间 | 自进化对象 | 监督/优化信号 | benchmark | 代码状态 |
|---|---|---|---|---|---|
| [Beyond the Capability Boundary: Zeroth-Order Optimization for Self-Evolving LLM Agents](https://arxiv.org/abs/2608.09292) | 2026-08-10 | instance-specific LoRA；随后把新轨迹 SFT 回共享模型 | trajectory-conditioned answer NLL 的有限差分 | GAIA、WebWalkerQA；迁移到 BrowseComp | 官方仓库目前只有 README/LICENSE，没有实现 |
| [CoEvoKG](https://arxiv.org/abs/2608.01904) | 2026-08-03 | proposer、search solver 和持久 KG evidence memory | answer correctness、path support、difficulty-aware reward；GRPO/REINFORCE++ | NQ、TriviaQA、PopQA、HotpotQA、2Wiki、Bamboogle | 已开源，vendored veRL + SGLang async rollout |
| [SEED](https://arxiv.org/abs/2607.14777) | 2026-07-16 | 同一个 policy 同时进化 actor 和 hindsight analyzer | skill context 引起的 token log-prob shift + GRPO | ALFWorld、WebShop、Search-based QA | 已开源，完整 veRL fork 和 agent env |

这三篇覆盖了三个互补层次：SEED 研究“怎样从已经采到的轨迹制造 dense token signal”；ZO 研究“模型完全采不到成功轨迹时怎样改变轨迹分布”；CoEvoKG 研究“任务分布和外部知识怎样随 solver 一起演化”。

### 2.2 与参数/梯度特别相关的补充论文

- [AgentStream](https://arxiv.org/abs/2608.00155)：截至当前最值得采用的 self-evolution 评测框架。它将 AppWorld、BFCL、BrowseComp-Plus、HLE、SWE-bench Verified、Tau2 组成 isolated/sequential/interleaved 三类 stateful task stream。论文发现 weakest model 可能出现负 evolution gain，也没有跨模型、跨 stream 普适最优方法。官方仓库已包含 Exgentic、五种 self-evolving methods 和六个 benchmark adapter。
- [Scaling Self-Evolving Agents via Parametric Memory (TMEM)](https://arxiv.org/abs/2606.04536)：在单个 episode 内把抽取出的监督写进 fast LoRA weights，并用 SVD 初始化 LoRA 子空间。它支持“先从极小参数子空间开始”的判断，但研究目标主要是长期记忆，不是输出分布 repair。
- [SAGE](https://arxiv.org/abs/2603.15255)：Challenger、Planner、Solver、Critic 共用 backbone，以 verifier 控制 curriculum drift。它提示修复器不仅要评价 answer，还应评价生成任务和 plan 的质量。
- [TT-SI](https://arxiv.org/abs/2510.07841)：先发现不确定样本，再自生成相似训练例并做 test-time fine-tuning，是 ZO 论文 answer-likelihood/test-time adaptation 路线的重要前驱。
- [SE-Agent](https://arxiv.org/abs/2508.02085)：不直接改模型参数，通过 revision、recombination、refinement 操作 trajectory pool，在 SWE-bench Verified 上演化代码 agent。它适合当外层搜索与实验管理器。

## 3. SEED 流程精读

SEED 有两个训练阶段。

### 3.1 Hindsight-skill SFT

1. 基础 policy 在 agent environment 中采普通轨迹，轨迹包括 observation、action、reward 和 episode outcome。
2. 外部 analyzer 对成功与失败轨迹生成 episode-level skill：成功轨迹提炼 workflow，失败轨迹提炼 avoidance/correction rule。
3. 对同一个 backbone 做 SFT，使它能从完整轨迹生成 skill。这个 checkpoint 同时初始化后续 actor 与 analyzer。

因此 SEED 不是从一开始就完全自举；第一阶段的 skill annotation 来自外部 analyzer。真正的 self-evolving loop 发生在第二阶段。

### 3.2 Self-evolving on-policy distillation

每次 policy update：

1. 冻结当前 snapshot `π_old`。
2. `π_old` 对每个 task 采一组 on-policy trajectories，并根据 outcome 算 group-relative advantage。
3. 同一个 `π_old` 切换 analyzer role，给每条完整轨迹生成 hindsight skill `s`。
4. 对已经采到的同一组 action tokens 做两次 teacher forcing：普通 history 得到 `log p_plain`，加入 skill 的 history 得到 `log p_skill`。
5. 计算 detached shift：`Δ = sg(log p_skill - log p_plain)`，再用 `g = sigmoid(βΔ)` 做 confidence gate。
6. OPD loss 只让梯度流过普通分支，大意为 `E[m · g · (sg(log p_skill) - log p_plain)]`，并与 clipped GRPO、KL loss 联合训练。
7. 新 policy 变成下一轮 actor 和 analyzer。部署时删除 skill/analyzer，只保留 policy。

本地代码中的关键实现是 `references/SEED/verl/trainer/ppo/core_algos.py::compute_opd_loss`。代码明确 detach teacher log-prob 和 gate，梯度只进 student log-prob。

### 3.3 SEED 对本课题的直接启发和边界

启发：

- 它已经证明“上下文修复导致的概率变化”可以被转成 token-level dense supervision。
- training-time teacher 可以与 student 共用模型，部署时不增加 agent 组件。
- 失败轨迹也能产生可用 signal，不必只保留成功轨迹。

边界：

- OPD 主要重打分已经采样出的 action token，不是对完整 action distribution 做 KL；从未进入候选集的动作仍缺少直接监督。
- skill teacher 仍在原模型能力附近，遇到所有 rollout 都失败的 hard instance 时，teacher 未必知道怎样修。
- actor 与 analyzer 同步进化可能形成同源偏差，需要外部 verifier 或 held-out evaluator 抑制 drift。

本课题的 distribution repair 可以看作把 SEED 从“单个 sampled token 的概率差”推广到“候选 logic actions 的显式 `p→q` 变换”。

## 4. Beyond the Capability Boundary 精读

### 4.1 它解决的梯度缺口

对 agent 的完整目标 `F(θ)`，论文把梯度写成两部分：

- `g_fixed`：把已经生成的轨迹视为固定上下文，对 answer loss 求普通一阶梯度。
- `g_traj`：参数变化引起 reasoning、query、tool selection、environment interaction 的轨迹分布变化。

普通 teacher-forcing 只估计 `g_fixed`。当所有 sampled trajectories 都失败且 reward 相同，policy-gradient advantage 也会变为 0。论文的关键主张是：在完整 rollout 外层做 zeroth-order finite difference，能够把不可微环境引起的轨迹改变包含进 loss difference。

### 4.2 完整流程

对每个困难 QA instance：

1. 挂载一个 instance-specific LoRA `w_i=(A_i,B_i)`。
2. 采随机方向 `ε_k`，构造 LoRA perturbation。
3. 原参数和扰动参数分别运行完整 search-agent rollout。
4. 删除生成答案，把 gold answer 接在生成 trajectory context 后，计算只覆盖 gold answer tokens 的平均 NLL。论文称 answer perplexity loss，但真正优化的是 log-space token-normalized NLL。
5. 用 loss difference 估计 LoRA 梯度并用 Adam 更新。主实验偏向 one-sided estimator，共享一个 unperturbed baseline rollout；two-sided SPSA 在相同 rollout budget 下表现接近。
6. 优化后的 instance LoRA 用于寻找成功轨迹；正确或足够低 loss 的轨迹进入 high-quality buffer。
7. 用 buffer 对共享 agent 做 SFT，丢弃 instance-specific LoRA，形成可以部署的 shared evolved model。

### 4.3 工程技巧与论文结果

- 多个 perturbation 共享 backbone forward，只并行计算轻量 LoRA branches。论文在 K=4 时报告相对顺序估算节省 53.5% rollout 时间。
- search/visit observation 做 exact 或 semantic cache，减少不同 perturbation 的重复工具调用。
- 在 50 个困难样本的敏感性实验中，论文默认附近的配置为 `K=2、sigma=1e-3、rank=16、learning rate=1e-6`；不同值下成功数变化不大。
- 304 个训练 QA 中，初始 Pass@1 成功 67 个；ZO 后为 164 个。Qwen3-4B 在 GAIA 上从 19.6 提升到 28.3，Qwen3-8B 从 23.3 提升到 47.5。

### 4.4 需要谨慎看待的地方

- “无 trajectory annotation”不等于“无标签”：每个训练实例仍需要 gold final answer，且 answer-NLL 要访问本地模型 logits。
- 每个 hard instance 都单独优化 LoRA，成本高；真正泛化依赖第二阶段 SFT，而不是 instance LoRA 本身。
- tool environment 的随机性会显著增加 finite-difference 方差。必须复用 tool result、固定 decoding random numbers，或用 paired/antithetic perturbations。
- 论文截至本调研时的官方 GitHub 仓库只有两个 commit、README 和 LICENSE，无法检查其 parallel LoRA、cache 或训练脚本实现。本仓库 `references/ZOForLLMAgents` 如实保留了这个状态。

## 5. 代码库有什么相似之处

因为 ZO 官方实现未发布，训练代码重点对照 SEED、CoEvoKG，外层 evolution 对照 SE-Agent，评测代码对照 AgentStream；同时用 ZO 论文算法约束接口。

| 层 | SEED | CoEvoKG | AgentStream | SE-Agent | 可复用抽象 |
|---|---|---|---|---|---|
| rollout backend | veRL + 多轮 env，vLLM | vendored veRL + SGLang async | Exgentic + isolated benchmark venv | SWE-agent batch runner | `RolloutEngine` |
| experience | DataProto trajectory、step mask、reward | proposer/solver trajectory、KG chain | shared or per-domain evolution state | `.traj/.tra/.pred` trajectory pool | `TrajectoryStore` |
| evaluator | env episode reward + search QA scorer | EM/judge + path support | 六个 benchmark adapter + cumulative stream metrics | SWE-bench tests | `Verifier/Evaluator` |
| evolution signal | skill-conditioned log-prob shift + GRPO | proposer/solver reward + KG write-back | context/memory/skill/harness update | revision/recombination/refinement operator | `EvolutionSignal` |
| update | FSDP policy update | dual RL update | test-time state update，不训练权重 | 下一轮 prompt/config | `Updater` |
| persistence | checkpoint、rollout data | checkpoint、KG memory、question pools | isolated/global store checkpoint | iteration directory、resume | `CandidateRegistry` |

它们最稳定的公共闭环都是：

```text
task/curriculum
  → policy rollout with tools
  → serialized trajectory + token probabilities
  → verifier/reward/teacher signal
  → parameter or workflow update
  → held-out evaluation
  → promote new policy and refresh the next task/experience distribution
```

两份训练型代码都没有把 veRL 当作普通 pip dependency，而是 vendored/fork 进仓库并直接修改 trainer/worker internals。这说明 agent 自进化尚未形成稳定插件边界；若直接在两个 fork 间拷代码，后期升级成本会很高。本项目应把 distribution repair 做成独立、窄接口模块，再通过 adapter 接入 SEED 的 rollout/trainer，而不是再 fork 第三份完整 veRL。

## 6. 推荐框架与 benchmark

### 6.1 框架选择

**内层：SEED/veRL。** 原因不是它最新，而是它已经把我们最需要的四件事接在一起：多轮 agent trajectory、普通/增强上下文重打分、token-level teacher signal、GRPO/FSDP 训练。

**外层：SE-Agent 风格 registry。** 每轮保存 proposal policy、repair policy、head/LoRA delta、trajectory buffer、完整 config、随机种子和 held-out metrics。只有通过 promotion gate 的 candidate 才能成为下一轮 policy。

**评测：AgentStream/Exgentic。** 新增一个 parametric-head agent adapter，把 `head checkpoint + evolution state` 作为可持久化状态；先跑 isolated，再跑 sequential/interleaved。AgentStream 当前以 API agent 为主，不能直接替代 veRL trainer，但其 benchmark isolation、task ordering 和累计指标可以独立复用。

**不建议当前直接以 CoEvoKG 为主框架。** 它非常适合下一阶段 self-generated curriculum 和 evidence memory，但第一阶段会同时引入 proposer、KG、retriever、judge endpoint 和 dual RL，难以判断收益究竟来自 gradient repair 还是 curriculum/memory。

### 6.2 benchmark 分层

1. **LogicRoute（当前已经完成）**：候选动作小、transition 可执行、每一步完整概率可记录，专门用于检查 finite difference、repair acceptance、calibration 和 trust region。
2. **ALFWorld（下一步首选）**：离散 action、长 horizon、环境相对稳定，SEED 已有 adapter。适合先验证 logic-action head 是否能改变工具/动作轨迹。
3. **Search-R1（第二步）**：与 ZO 论文的 search-agent、answer-NLL 最接近，SEED 已有 search env；需要本地 retriever 和 observation cache。
4. **GAIA/WebWalkerQA（最终报告）**：用于与 ZO 论文对齐，但 web/tool latency、数据访问和 judge 使其不适合高频 ZO ablation。
5. **AgentStream（跨任务可靠性）**：从 BFCL + AppWorld 两域开始，之后再加 Tau2/BrowseComp-Plus。比较 isolated、sequential、interleaved 下的 evolution gain、遗忘和修复器状态污染。

训练/验证必须按 task template、实体或环境场景隔离，不能把 verifier、KG 或 tool cache 中的 gold evidence 泄漏到 held-out split。

## 7. 建议的输出分布修复方案

### 7.1 把 full-vocabulary distribution 变成 logic distribution

在状态 `s_t=(history, observation, available tools)` 上先构造可执行候选集 `C_t`：工具调用、结构化参数候选、继续思考、最终回答等。对多 token action 用 sequence log-prob 求候选 score，再在 `C_t` 上归一化：

`p_t(a)=softmax(score_LLM(a|s_t)), a∈C_t`。

这样 head 输出维度由词表大小变成候选 action 数，能够显著减少参数，也让 verifier 可以逐候选执行检查。当前 prototype 用 9 个单 token logic labels；接入真实 agent 时应替换成 multi-token candidate scorer，不能依赖任意数字 label。

### 7.2 speculative-action repair

proposal 给出 `p_t` 后，只验证 top-k action，得到 verifier score `V(s_t,a)`，再计算：

`q_t(a) ∝ p_t(a) exp(β_t V(s_t,a))`。

- 若 `argmax p = argmax q`，接受 proposal。
- 若不同，执行 repaired action，并记录 correction event。
- 若 verifier 置信度低或候选都失败，回退到扩大候选集、重新规划或普通 sampling。

它与 speculative decoding 的相似点是“便宜 proposal + 较贵 verification + accept/correct”。不同点是这里的 verifier 不是精确 target LM，修复后的分布通常不保持原 LLM 分布，因此不能声称是严格等价的 speculative decoding。

### 7.3 两条更新路径

一阶 repair distillation：

`L_FO = KL(sg[q_t] || p_phi(.|s_t)) + λ KL(p_phi || p_old)`。

零阶 trajectory update：

`J(phi) = answer_NLL(phi; q, tau_phi, y) + α verifier_loss + λ drift_penalty`，

`g_hat = (J(phi+σu)-J(phi-σu))/(2σ) · u`。

这里 `phi` 首先只包含 output repair head。等机制稳定后依次扩展到 LM-head LoRA、最后若干层 LoRA；不要一开始扰动全模型。

### 7.4 必须记录的诊断量

- 每步 `p_t`、`q_t`、entropy、top-1 margin、action rank、repair/accept event。
- raw success、online-repaired success、internalized success 三套指标。
- Brier/ECE、`KL(p_new||p_old)`、logit delta norm、参数 update norm。
- 每次成功所需 rollout 数、wall-clock、tool calls、cache hit、GPU-hours。
- ZO direction variance、不同 seed 的 mean/std/CI。
- train task、held-out paraphrase、held-out environment 三层泛化。

## 8. 已完成实验

### 8.1 设置

- frozen backbone：Qwen2.5-0.5B-Instruct。
- benchmark：两步 LogicRoute；第一步选五种工具，第二步从四个 answer label 中选择。
- train/eval：80/40 tasks，eval 使用未见 paraphrase。
- head：固定 random projection rank=32；输入为压缩 hidden state 加 9 维 proposal probability；可训练参数 378。
- repair：全候选 oracle verifier，`beta=4`。
- ZO：two-sided、8 directions、45 steps；一阶 baseline 60 steps。
- 8 张 GPU 并行独立 seed 100–107，每卡一个进程。

### 8.2 8-seed 结果

| 方法 | strict accuracy | tool accuracy | choice accuracy | final-answer NLL |
|---|---:|---:|---:|---:|
| frozen base | 0.0625 ± 0.0378 | 0.2094 ± 0.0186 | 0.3031 ± 0.0508 | 1.3997 ± 0.0331 |
| base + online repair | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.4043 ± 0.0342 |
| ZO internalized | 0.1156 ± 0.0896 | 0.3937 ± 0.1444 | 0.2625 ± 0.0856 | 1.4987 ± 0.1277 |
| first-order repair distilled | 0.1531 ± 0.0490 | 0.6000 ± 0.1000 | 0.2625 ± 0.0518 | 1.5234 ± 0.1122 |
| ZO + online repair | 0.9500 ± 0.1414 | 0.9500 ± 0.1414 | 1.0000 ± 0.0000 | 1.4781 ± 0.1335 |

95% CI 和逐 seed 数值见 `outputs/8gpu_experiment/summary.json`。

### 8.3 如何解释

- 在线修复 100% 说明 `p→q→corrected action` 链路成立，但 verifier 是 oracle，所以这是机制上界，不是现实 agent 结果。
- ZO 把 tool accuracy 从约 21% 提到约 39%，说明有限差分确实能改变前一步工具路由，而不只是 final token。
- ZO 方差明显高于一阶蒸馏，符合随机方向估计的预期；后续必须报告多 seed 和 rollout-normalized efficiency。
- strict accuracy 上升但平均 NLL 不一定下降，因为训练后的 policy 会进入不同 step-2 observation，evaluation distribution 随轨迹改变。只看 teacher-forced NLL 会误判 agent 改进。
- ZO + online repair 不是 100%，说明某些 seed 的 residual logits 过强，固定 `beta=4` 无法覆盖错误 margin。这正是加入 trust region、adaptive beta 和 logit clip 的直接证据。

## 9. 下一阶段实验顺序

### Phase A：把当前机制做扎实

1. 加 `KL(new||old)`、residual clip、adaptive beta。
2. 对比 one-sided、two-sided、orthogonal directions、antithetic sampling 和 common random numbers。
3. 对比只看 distribution、hidden+distribution、LoRA output head 三种参数化。
4. verifier top-k 从全候选降到 1/2/4，画 accuracy–cost 曲线。
5. 加随机或有噪 verifier，测错误修复与 misevolution。

### Phase B：ALFWorld

1. 复用 SEED environment/trajectory collector。
2. 候选集使用 admissible actions；先优化 action head，再试 LM-head LoRA。
3. baseline：Vanilla、GRPO、SEED OPD、online repair only、FO repair-distill、ZO repair。
4. promotion gate 同时要求 held-out success 提升、KL drift 不超阈值、repair rate 不恶化。

### Phase C：Search-R1

1. 对 search/answer 决策与 query candidates 记录 sequence log-prob。
2. answer-NLL 只覆盖 gold answer token，严格复刻 ZO loss。
3. perturbation branches 复用 retrieval/visit observation；固定 decoding seed。
4. 先做 50 个 hard examples × 15 rounds，对齐论文的 trajectory-discovery 分析，再做 SFT shared model。

### Phase D：最终 benchmark

在 GAIA/WebWalkerQA 上只运行晋升后的少量 checkpoint，报告 raw、online repair、internalized 三种模式，以及总 tool calls、GPU-hours 和 verifier 成本。

## 10. 当前文件定位

- 论文 PDF：`papers/SEED_2607.14777.pdf`、`papers/ZO_self_evolving_agents_2608.09292.pdf`
- 代码快照：`references/SEED`、`references/CoEvoKG`、`references/AgentStream`、`references/SE-Agent`、`references/ZOForLLMAgents`
- probability repair：`src/self_evolve/repair.py`
- 378 参数 distribution head：`src/self_evolve/controller.py`
- agent rollout 与 loss：`src/self_evolve/evolution.py`
- ZO estimator：`src/self_evolve/zo.py`
- 8-seed 汇总：`outputs/8gpu_experiment/summary.json`
