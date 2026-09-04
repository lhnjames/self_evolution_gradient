# Multisource dose-response snapshot — 2026-09-03

- Remote host: fifth machine (`124.221.190.139:246`)
- Remote root: `/data/user/agent_self_evolution_gradient`
- Remote output: `outputs/multisource_dose_response_v2`
- Local archive: `outputs/multisource_dose_response_v2`
- Completion markers: `TMUX_RUN_STARTED`, `TMUX_RUN_COMPLETE`, `analysis/ANALYSIS_COMPLETE`
- Local/remote recursive content hash: `32072a73b3dbaa03df519ec3c62e4b5aeaf99e0a125a4bc9b62186cecd5a2e5b`
- Analysis JSON SHA-256: `cb83c6af520bcc9cb9d9aa1bd0dc072552fa2bc8215fd20370907734344e968d`
- Analysis report SHA-256: `f490a6b07c5cc45a78d94ab9ecdfd054e2cafe2c0544685f53d141a42e973782`
- Formal selected steps: `last_mlp=0.0125`, `last_four_blocks=0.006`
- Execution: FP32 model; CUDA tensor gradient capture, mean accumulation, norm, writeback, and scoring on physical GPUs 0–3; GPUs 5–7 untouched.
- Integrity: 5 verbs per condition, 12 sources and 6 holdouts per category per verb, restore max error 0, gradient reproduction max error 0.

