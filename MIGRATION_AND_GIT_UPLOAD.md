# 第 5 台机器迁移与 Git 上传记录

日期：2026-09-04（Asia/Shanghai）

## 路径

- 源：`user@124.221.190.139:/data/user/agent_self_evolution_gradient/`
- 当前服务器目标：`/data/hanning/agent_self_evolution_gradient_bundle_20260901/`
- GitHub：`https://github.com/lhnjames/self_evolution_gradient.git`

迁移采用保守增量同步：不删除目标目录已有文件，并保留目标端时间更新的本地分析结果。

## 已纳入

- `src/`、`scripts/`、`tests/`、`config/` 和根目录运行脚本；
- 当前服务器保留完整 `outputs/`，包括日志、JSON/JSONL、NPY/NPZ、图片和实验 gradient/delta `.pt`；
- `data/`、`alfworld_data/`、`alfworld_expert_large/` 和 `baseline_traces/`；
- `research/`、`papers/`、`references/`、`provenance/`；
- README、实验清单和逐文件 SHA-256。

## 明确排除

- `model/`：基础模型；
- `seed_model/`：SEED 基础模型/checkpoint；
- `model_cache/` 及 `*.safetensors`、`*.gguf`、`pytorch_model*.bin`；
- GitHub 额外排除所有 `*.pt`、`*.pth`、`*.ckpt` 可加载参数产物；这些文件仍完整保留在当前服务器；
- `.venv/`：机器绑定的 Python 虚拟环境；
- `.pytest_cache/`、`__pycache__/`、`.pip-cache/`、`.tmp/`；
- `.git/` 和第三方目录内嵌套的 `.git/` 元数据；
- `.env`、私钥和凭据文件。

排除虚拟环境和缓存不影响实验内容；依赖由 `pyproject.toml` 重建。任何 GitHub token 均未写入项目文件、Git remote 或日志。

## GitHub 轻量上传与 Git LFS

本地实验结果包含多个约 616 MB 的 `.pt`。用户决定 GitHub 只保留代码、日志及可审阅结果，
因此全部 `.pt/.pth/.ckpt` 从 Git 索引排除，不删除当前服务器上的原文件。仍需版本化的数组、
归档和论文通过 `.gitattributes` 交给 Git LFS：

```text
*.npy *.npz *.zip *.tar *.tgz *.pdf
```

这些非模型文件仍属于仓库内容，只是其二进制对象由 LFS 存储；Git 提交中保存可校验指针。

## 完成态审计

迁移完成后应同时满足：

1. 在相同排除规则下，源→目标 rsync dry-run 不再发现缺失文件；
2. 目标目录不存在 rsync 临时文件；
3. GitHub 上传集合中所有大于 99 MB 的文件都命中 `filter=lfs`；
4. `ARTIFACT_SHA256SUMS.txt` 全量验证通过；
5. Git 工作树完成提交，且 GitHub `main` 指向该提交；
6. `git lfs ls-files` 能列出实验二进制对象。

## 迁移审计结果

| 检查项 | 结果 |
|---|---:|
| 第 5 台机器纳入文件 | 19,537 个普通文件 |
| 第 5 台机器纳入体积 | 20,831,593,820 bytes |
| 本次实际补传体积 | 17,071,278,628 bytes |
| 当前服务器完整迁移集合 | 36,735 个普通文件 + 1 个符号链接，约 21.80 GB |
| GitHub 轻量上传集合 | 36,615 个普通文件 + 1 个符号链接，约 4.13 GB |
| 源→目标增量 dry-run | 0 个待传文件 |
| 内容校验差异 | 仅本地有意更新的 README、依赖配置与归档说明 |
| 本地参数产物（不上传） | 120 个，共 17,668,481,715 bytes |
| 超过 99 MB 的本地参数产物 | 30 个，共 17,054,799,505 bytes |
| rsync 临时文件 | 0 |
| 基础模型文件/目录 | 0 |
| 凭据模式扫描命中 | 0 |
| SHA-256 清单 | 36,734 个内容文件，验证通过 |
| 测试 | 第 5 台机器环境中 `50 passed` |

`.env.example`、`.env.sample` 和 `.env.template` 属于可公开的配置模板，允许进入仓库；真实
`.env`、私钥和 token 仍被排除。

Git commit、LFS 对象和远端 `main` 验证结果将在完成推送后补录。
