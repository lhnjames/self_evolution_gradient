# Value-target gradient completed-data snapshot

- Captured at: 2026-09-03 12:23–12:25 Asia/Shanghai
- Source host/root: `nkilm70yynvd4yk:/data/user/agent_self_evolution_gradient`
- Archive root: `/data/hanning/agent_self_evolution_gradient_bundle_20260901`
- Run marker: `outputs/value_gradient_probe_v1/TMUX_RUN_COMPLETE`, mtime 2026-09-03 12:22:43 +0800
- Analysis marker: `outputs/value_gradient_probe_v1/analysis/ANALYSIS_COMPLETE`, mtime 2026-09-03 12:23:09 +0800
- Samples: valid_seen 128, valid_unseen 128; errors: 0
- Output files: 30; archive size: 69 MiB (`du -sh` display)
- Aggregate SHA-256 of sorted per-file SHA-256 records:
  `390a4eb3e1bf2de855f199367e8143df73f3fe70d1efc675437286102377b46b`
- The aggregate was independently computed on source and archive after rsync and matched exactly.
- Physical GPUs 0–3 were back to 14 MiB idle before capture; GPUs 5–7 were never targeted.

Aggregate command, run from each project root:

```bash
find outputs/value_gradient_probe_v1 -type f -print0 \
  | sort -z | xargs -0 sha256sum | sha256sum
```

