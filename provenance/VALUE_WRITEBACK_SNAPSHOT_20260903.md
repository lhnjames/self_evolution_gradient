# Value-gradient writeback completed-data snapshot

- Captured at: 2026-09-03 13:58–14:02 Asia/Shanghai
- Source host/root: `nkilm70yynvd4yk:/data/user/agent_self_evolution_gradient`
- Archive root: `/data/hanning/agent_self_evolution_gradient_bundle_20260901`
- Run marker: `outputs/value_gradient_writeback_v2/TMUX_RUN_COMPLETE`, mtime 2026-09-03 13:58:17 +0800
- Analysis marker: `outputs/value_gradient_writeback_v2/analysis/ANALYSIS_COMPLETE`, refined mtime 2026-09-03 after the run
- Sources: valid_seen 32, valid_unseen 32; writeback conditions: 960; state responses: 4,800; run errors: 0
- Numerical checks: baseline repeat maximum error 0; parameter restore maximum error 0; source-KL calibration hit rate 99.6%/100%
- Output files: 26; archive size: 9.0 MiB (`du -sh` display)
- Aggregate SHA-256 of sorted per-file SHA-256 records:
  `7341c7116004d52f54a33f751f0cc5abed18cee68d6d415f3a9f03c4a18a4f27`
- The aggregate was independently computed on source and archive after rsync and matched exactly.
- Physical GPUs 0–3 returned to 14 MiB idle before capture; GPUs 5–7 were never targeted.
- The first `value_gradient_writeback_v1` launch was intentionally aborted before shard artifacts were written:
  one proposed source lacked a `different_task_same_verb` holdout. Source eligibility was corrected to require all
  four holdout buckets before deterministic sampling; the clean restart is the archived `v2` result.

Aggregate command, run from each project root:

```bash
find outputs/value_gradient_writeback_v2 -type f -print0 \
  | sort -z | xargs -0 sha256sum | sha256sum
```

