# ALFWorld 自进化实验归档清单

整理时间：2026-09-01；最新增量归档：2026-09-04（Asia/Shanghai）

## 归档范围

| 路径 | 内容 |
|---|---|
| `src/self_evolve/` | 实验主体、评分器、repair head、在线 rollout 与梯度逻辑 |
| `scripts/` | 数据采集、评分、分析、bootstrap、多卡运行与归档脚本 |
| `config/` | 所有已使用实验配置和多 seed 配置 |
| `tests/` | 单元测试 |
| `data/alfworld_expert_large/` | 47/33/34 个成功 episode 的 train/seen/unseen 专家决策数据 |
| `alfworld_data/` | ALFWorld TextWorld 原始数据、PDDL/TWL2 logic 和原始压缩包 |
| `outputs/` | 所有离线/在线结果、逐状态 trace、日志、bootstrap、梯度数组和小型 head checkpoint |
| `research/` | 中文实验报告、当前状态和设计记录 |
| `references/` | SEED、SkillRL、SkillsBench 等第三方参考代码快照 |
| `papers/` | 研究论文 PDF |
| `provenance/` | 从临时目录迁移的原始脚本和迁移记录 |

## 明确排除

- `model_cache/`：Qwen2.5 和 SEED 基础模型/完整 checkpoint，不进入便携归档。
- `.venv/`：本机 Python 虚拟环境，可由 `pyproject.toml` 重建。
- `.pip-cache/`、`.pytest_cache/`、`__pycache__/`：下载和运行缓存。
- 第三方仓库中的 `.git/` 历史：保留工作树代码，不复制版本对象。
- `*.safetensors`、`*.gguf`、`pytorch_model*.bin`：模型权重。
所有由实验生成的 `.pt` residual/FO/ZO head 与 GB 级 parameter-delta bank 均已迁移到当前
服务器，因此本地归档是完整的。按 2026-09-04 的 GitHub 轻量上传决定，这些可加载参数
产物不进入 Git 仓库；GitHub 保留全部源码、配置、日志、结构化结果、研究报告与图片。
基础模型、SEED 完整 checkpoint 和可重建环境同样不上传。

## 2026-09-04 Gradient Scope Tomography 增量

- `scripts/probe_output_gradient_scope.py`：固定 output-only gradient direction，联合扫描
  `K=1..12` 与 `300x..3000x`，保存逐状态 value scope、first-order compatibility 和 logit influence。
- `src/self_evolve/gradient_scope.py` 与 `tests/test_gradient_scope.py`：scope coverage、purity、
  transition、Jaccard 和 gradient-direction cosine 的纯数值实现与测试。
- `outputs/output_gradient_self_evolution/scope_tomography_seed_*.json/.npz`：4 个随机 seed、
  每个 seed 30 状态、总计 7,200 个 state-gradient 条件的原始结果和矩阵。
- `outputs/output_gradient_self_evolution/scope_tomography/gradient_scope_tomography.md`：
  公式边界、direction/dose 分离、结构冲突及 held-out family feedback 严格同状态复核报告。
- `scripts/probe_gradient_scope_topology.py` 与 `scripts/analyze_gradient_scope_topology.py`：
  全 18 failure 的双向 transfer、full-gradient/delta/hidden signed topology、有效秩和 novelty 分析。
- `outputs/output_gradient_self_evolution/scope_topology/gradient_scope_topology.md`：
  Gradient Scope Topology 最终报告；原始矩阵位于相邻的 `scope_topology_seed_*.npz`。
- `src/self_evolve/repair_space.py`、`scripts/probe_repair_space_characterization.py` 与
  `scripts/analyze_repair_space_characterization.py`：正交 repair basis、rank 1–6 方向搜索、
  强度扫描、真实 multi-token GPU 重评分与 Pareto/trajectory/overlap 分析。
- `outputs/output_gradient_self_evolution/repair_space_seed_*.json/.npz`：4 个 seed 的原始
  repair-space 方向、逐状态非线性响应和矩阵；合计 4,044 个方向–强度候选。
- `outputs/output_gradient_self_evolution/repair_space/`：Repair-Space Characterization 总报告、
  4 张主图，以及唯一 rank-5 强安全区域的 541 点局部 GPU 连通性扫描。

## 模型接入

便携归档不包含模型。运行四卡脚本时通过环境变量指定外部模型：

```bash
export BASE_MODEL_PATH=/absolute/path/to/Qwen2.5-3B-Instruct
export SEED_MODEL_PATH=/absolute/path/to/Seed-AlfWorld-3B
export BASE_TOKENIZER_PATH="$BASE_MODEL_PATH"
```

随后运行 `scripts/remote/` 中相应脚本。Hugging Face 下载时可使用：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 完整性

便携归档根目录中的 `ARTIFACT_SHA256SUMS.txt` 记录除自身之外每个文件的 SHA-256，验证方式：

```bash
cd /data/hanning/agent_self_evolution_gradient_bundle_20260901
sha256sum --check ARTIFACT_SHA256SUMS.txt
```
