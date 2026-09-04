# 临时目录迁移记录

迁移时间：2026-09-01（Asia/Shanghai）

## 已迁移

| 原路径 | 归档路径 | 说明 |
|---|---|---|
| `/tmp/run_skill_controls_remote.sh` | `provenance/tmp_migration/original_scripts/run_skill_controls_remote.sh` | 第 5 台机器的四卡 skill-control 启动脚本原件 |
| `/tmp/run_gradient_probe_remote.sh` | `provenance/tmp_migration/original_scripts/run_gradient_probe_remote.sh` | 第 5 台机器的四卡梯度探针启动脚本原件 |
| `/tmp/run_seed_checkpoint_scoring_remote.sh` | `provenance/tmp_migration/original_scripts/run_seed_checkpoint_scoring_remote.sh` | 第 5 台机器的四卡 SEED 评分启动脚本原件 |
| `.tmp/alfworld/json_2.1.2_tw-pddl.zip` | `provenance/tmp_migration/incomplete_downloads/json_2.1.2_tw-pddl.zip.empty_partial` | 0 字节失败下载，仅用于说明来源；完整数据在 `alfworld_data/` |

原件中的远程绝对路径保持不变，仅用于追溯。可继续运行的便携版本位于
`scripts/remote/`，模型路径必须通过环境变量显式传入，归档中不包含模型。

## 未迁移

系统服务临时文件、Kerberos token、编辑器缓存、pytest 自动生成夹具和与本项目无关的
TTS/WAN 临时文件均未迁移。这些文件不是实验代码、日志、数据或数据集，且部分包含短期凭据。
