# Token-value/full-vocabulary completed-data snapshot

- Captured at: 2026-09-03 12:05–12:07 Asia/Shanghai
- Source host/root: `nkilm70yynvd4yk:/data/user/agent_self_evolution_gradient`
- Archive root: `/data/hanning/agent_self_evolution_gradient_bundle_20260901`
- Run marker: `outputs/token_value_alignment/TMUX_RUN_COMPLETE`, mtime 2026-09-03 12:04:47 +0800
- Analysis marker: `outputs/token_value_alignment/analysis/ANALYSIS_COMPLETE`, mtime 2026-09-03 12:04:57 +0800
- Rows: valid_seen 444, valid_unseen 527; errors: 0
- Output files: 27; archive size: 354 MiB (`du -sh` display)
- Aggregate SHA-256 of sorted per-file SHA-256 records:
  `91a334dec2c02ab66cd71e917b5b4b95c0223215dac6d504325d42dcd5592170`
- The aggregate was independently computed on source and archive after rsync and matched exactly.
- The source GPU processes had exited and physical GPUs 0–3 were back to 14 MiB idle before capture.

Aggregate command, run from each project root:

```bash
find outputs/token_value_alignment -type f -print0 \
  | sort -z | xargs -0 sha256sum | sha256sum
```

