# Multisource gradient snapshot — 2026-09-03

- Remote host: fifth machine (`124.221.190.139:246`)
- Remote root: `/data/user/agent_self_evolution_gradient`
- Remote output: `outputs/multisource_value_gradient_v2`
- Local archive: `outputs/multisource_value_gradient_v2`
- Completion markers: `TMUX_RUN_STARTED`, `TMUX_RUN_COMPLETE`, `analysis/ANALYSIS_COMPLETE`
- Local/remote recursive content hash: `508702aed01a92ba5b6c091d011a14772d31c7ad982892a5e2eebb8d9169d95d`
- Analysis JSON SHA-256: `1fb5f635ef9d0d96009cfb56338d88f8d58ce0c9d1ff869fc76851627ed0bd0a`
- Analysis report SHA-256: `1e9d9ff3f2be99673e40d942bece51ef7ec77f357ebef4a89cf28408450d1873`
- Formal steps: `last_mlp=0.0125`, `last_four_blocks=0.006`
- Scale: 12 sources and 6 holdouts per category per verb; 240 single-source updates, 20 mean-gradient updates, 7,800 state responses, 2,640 directed source-transfer pairs.
- Execution: FP32 model; CUDA gradient capture, retention, accumulation, norm/dot, writeback, and scoring on physical GPUs 0–3; GPUs 5–7 untouched.
- Integrity: 5 verbs per condition; restore, baseline repeat, and gradient score reproduction maximum errors all 0.

