# Distribution-Repair Self-Evolution

这是一个面向 LLM agent 的最小可运行研究框架，验证以下闭环：

```text
frozen LLM proposal p(a|s)
          │
          ├── verifier 对候选 action 做检查
          ▼
repaired distribution q(a|s) ∝ p(a|s) exp(βV(s,a))
          │
          ├── 立即采用修复后的 action（speculative-action repair）
          ├── 一阶 KL 蒸馏到轻量 head
          └── 用完整 rollout 的标量 loss 做 zeroth-order/SPSA 更新
```

受控实验加载冻结的 `Qwen2.5-0.5B-Instruct`。真实 ALFWorld 实验使用冻结的
`Qwen2.5-3B-Instruct`，对任意多 token admissible command 计算长度归一化序列分数；最终
stage-aware skill residual head 仅 175 个参数，不更新 LLM 主干。

## 已完成内容

- 两阶段可验证 agent benchmark：先选择工具，再根据工具 observation 选择答案。
- 原始、修复后概率分布以及 entropy、top-1 margin、接受/纠正事件的逐步日志。
- 在线 verifier distribution repair。
- SEED 风格的一阶 repair-distillation baseline。
- two-sided/one-sided zeroth-order random-direction estimator 与 Adam 更新。
- 单卡复现实验和 8 卡独立 seed sweep。
- SEED、CoEvoKG、AgentStream、SE-Agent 与 ZO 论文仓库的本地代码快照。
- 官方 ALFWorld train/valid_seen/valid_unseen expert decision collector。
- SkillRL evolved-skill、错配 skill、KL-constrained gradient head 与在线 cycle repair 对照。
- 全量候选动作环境干预、token/full-vocabulary 价值分解、Base/SEED 参数价值梯度与真实微小写回
  held-out transfer 实验。

完整调研、代码对照和下一阶段设计见 [`research/REPORT_ZH.md`](research/REPORT_ZH.md)。
面向 evolved skills 与 logic distribution 的聚焦设计见
[`research/LOGIC_DISTRIBUTION_DESIGN_ZH.md`](research/LOGIC_DISTRIBUTION_DESIGN_ZH.md)。
真实 ALFWorld 的完整结果见
[`research/ALFWORLD_LOGIC_REPAIR_RESULTS_ZH.md`](research/ALFWORLD_LOGIC_REPAIR_RESULTS_ZH.md)。
后续长期价值机制链见
[`research/ALFWORLD_ACTION_VALUE_RESULTS_ZH.md`](research/ALFWORLD_ACTION_VALUE_RESULTS_ZH.md)、
[`research/ALFWORLD_TOKEN_VALUE_RESULTS_ZH.md`](research/ALFWORLD_TOKEN_VALUE_RESULTS_ZH.md)、
[`research/ALFWORLD_VALUE_GRADIENT_RESULTS_ZH.md`](research/ALFWORLD_VALUE_GRADIENT_RESULTS_ZH.md) 与
[`research/ALFWORLD_VALUE_WRITEBACK_RESULTS_ZH.md`](research/ALFWORLD_VALUE_WRITEBACK_RESULTS_ZH.md)。
最新的 12 源非微小剂量响应与正式步长选择见
[`research/ALFWORLD_MULTISOURCE_DOSE_RESULTS_ZH.md`](research/ALFWORLD_MULTISOURCE_DOSE_RESULTS_ZH.md)。
12 源共同方向、等 L2 单源对照和精确冲突因果结果见
[`research/ALFWORLD_MULTISOURCE_GRADIENT_RESULTS_ZH.md`](research/ALFWORLD_MULTISOURCE_GRADIENT_RESULTS_ZH.md)。
覆盖全部长期价值梯度证据链、在线累积结果与研究边界的统一入口见
[`research/ALFWORLD_EXPERIENCE_GRADIENT_TOTAL_REPORT_ZH.md`](research/ALFWORLD_EXPERIENCE_GRADIENT_TOTAL_REPORT_ZH.md)。
最新的四种子 300× 相似技能失败梯度、条件 delta bank 与 SEED 横评见
[`research/ALFWORLD_SKILL_GRADIENT_SELF_PURIFICATION_ZH.md`](research/ALFWORLD_SKILL_GRADIENT_SELF_PURIFICATION_ZH.md)。
Output-only OGSE 的统一结论入口见
[`outputs/output_gradient_self_evolution/summary.md`](outputs/output_gradient_self_evolution/summary.md)；
最新的 7,200 条件 Gradient Scope Tomography、逐状态矩阵与严格同状态 feedback 复核见
[`outputs/output_gradient_self_evolution/scope_tomography/gradient_scope_tomography.md`](outputs/output_gradient_self_evolution/scope_tomography/gradient_scope_tomography.md)。
进一步的 signed topology、双向 transfer、有效秩和 feedback novelty 诊断见
[`outputs/output_gradient_self_evolution/scope_topology/gradient_scope_topology.md`](outputs/output_gradient_self_evolution/scope_topology/gradient_scope_topology.md)。
最新的强安全 single-direction feasibility、rank-constrained Pareto、局部连通性、进化轨迹与 SEED 同面板对照见
[`outputs/output_gradient_self_evolution/repair_space/repair_space_characterization.md`](outputs/output_gradient_self_evolution/repair_space/repair_space_characterization.md)。

## 环境与运行

Git 仓库包含代码、配置、数据、日志、实验结果和图片；大型实验 `.pt/.npz` 由 Git LFS 管理。
基础模型、模型缓存和机器绑定的 `.venv` 不进入仓库。完整迁移范围、模型排除规则和校验方法见
[`ARTIFACT_MANIFEST.md`](ARTIFACT_MANIFEST.md) 与
[`MIGRATION_AND_GIT_UPLOAD.md`](MIGRATION_AND_GIT_UPLOAD.md)。

首次检出必须先安装 Git LFS：

```bash
git lfs install
git clone https://github.com/lhnjames/self_evolution_gradient.git
cd self_evolution_gradient
python3 -m venv .venv
.venv/bin/pip install -e .
```

随后将本地基础模型放到实验脚本所需的 `model/`、`seed_model/` 或配置指定路径；这些目录已被
`.gitignore` 排除。

```bash
cd /data/hanning/agent_self_evolution_gradient
.venv/bin/pytest -q
./scripts/run_smoke.sh --output outputs/my_smoke
./scripts/run_smoke.sh --config config/experiment.yaml --output outputs/my_experiment
```

8 卡并行跑 8 个独立 seed：

```bash
./scripts/run_8gpu_sweep.sh outputs/my_8gpu config/experiment.yaml
.venv/bin/python scripts/summarize_sweep.py outputs/my_8gpu
```

运行 evolved-skill 前后对照、KL 安全投影与 FO/ZO 内化：

```bash
.venv/bin/python -m self_evolve.skill_runner \
  --config config/skill_experiment.yaml \
  --output outputs/skill_seed31

./scripts/run_8gpu_skill_sweep.sh
.venv/bin/python scripts/summarize_sweep.py outputs/8gpu_skill_experiment
```

## 当前 8-seed 结果

40 个 held-out paraphrase 任务/seed，mean ± std：

| 方法 | strict success | tool accuracy |
|---|---:|---:|
| frozen base | 6.25% ± 3.78% | 20.94% ± 1.86% |
| base + online repair | 100.00% ± 0.00% | 100.00% ± 0.00% |
| ZO internalized | 11.56% ± 8.96% | 39.37% ± 14.44% |
| first-order repair distilled | 15.31% ± 4.90% | 60.00% ± 10.00% |

这些数字只证明机制链路可运行，不是通用 agent 能力结论。在线修复使用可执行 benchmark 的 oracle verifier，因此 100% 是受控上界；更重要的发现是 ZO 有明显的 seed 方差，且缺少 trust region 时会出现 logit 过冲。

## 真实 ALFWorld 结果

Qwen2.5-3B-Instruct，598 train 决策、444 valid_seen、527 valid_unseen：

| 方法 | valid_seen top-1 | valid_unseen top-1 |
|---|---:|---:|
| plain | 41.22% | 31.69% |
| direct SkillRL evolved skill | 39.19% | 30.36% |
| KL-constrained stage-only head | 44.82% | **35.67%** |
| KL-constrained skill-aware head | **46.62%** | 35.29% |

skill-aware head 的平均 KL 分别为 0.0366/0.0303。episode-cluster bootstrap 相对 plain 的
95% CI 为 seen `[+3.05,+7.53]`、unseen `[+1.93,+5.23]`。直接 prompt 注入 skill 会退化；
skill 应作为 proposal delta，经小 head、KL 和 held-out promotion 后才能进入策略。

5 episode/split 的在线机制实验中，plain 为 seen 2/5、unseen 0/5；增加 state-action cycle
repair 后为 3/5、1/5。样本很小，不作为成功率结论，但验证了系统级防循环约束可以把部分
离线分布提升转成闭环成功。

## 目录

```text
config/                 实验参数
src/self_evolve/        benchmark、LLM scorer、repair head、rollout、ZO optimizer
tests/                  单元测试
scripts/                单卡、8 卡与汇总脚本
outputs/                完整 metrics、逐步分布 trace 与 head checkpoint
papers/                 SEED 与 ZO 论文 PDF
references/             论文代码快照（含 AgentStream/Exgentic 评测框架）
research/               中文调研和设计结论
provenance/             从临时目录迁移的原始运行脚本和来源记录
```
