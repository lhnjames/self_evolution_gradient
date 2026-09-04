# Action-value completed-data snapshot

- Captured at: 2026-09-03 11:48–12:00 Asia/Shanghai
- Source host: `nkilm70yynvd4yk` (`124.221.190.139:246`)
- Source root: `/data/user/agent_self_evolution_gradient`
- Archive root: `/data/hanning/agent_self_evolution_gradient_bundle_20260901`
- Main completion marker: `outputs/action_value_alignment/TMUX_RUN_COMPLETE`, mtime 2026-09-02 19:05:42 +0800
- Analysis completion marker: `outputs/action_value_alignment/analysis/ANALYSIS_COMPLETE`, mtime 2026-09-02 19:06:10 +0800
- Main rows: valid_seen 444, valid_unseen 527
- Planner control rows: valid_seen 444, valid_unseen 265
- Main/planner files: 200 / 129
- Main/planner sizes on source: 36 MiB / 16 MiB (`du -sh` display)
- Aggregate SHA-256 of sorted per-file SHA-256 records for both output trees:
  `9c70b95e37869f80978f2e95bef6b3aab792ae56d92f33cacac87bf5ba248d57`
- The aggregate was independently computed on source and archive after rsync and matched exactly.
- No evaluator process was running when captured; shard logs reported zero errors.

The aggregate command was run from each project root:

```bash
find outputs/action_value_alignment outputs/action_value_alignment_planner -type f -print0 \
  | sort -z | xargs -0 sha256sum | sha256sum
```

This snapshot precedes the token-level/full-vocabulary experiment. That experiment writes to
`outputs/token_value_alignment` and does not overwrite either captured tree.

