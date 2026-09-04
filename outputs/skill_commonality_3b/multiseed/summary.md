# Unique-episode repair head multi-seed summary

| seed | seen stage-plain | seen skill-plain | seen skill-stage | unseen stage-plain | unseen skill-plain | unseen skill-stage |
|---:|---:|---:|---:|---:|---:|---:|
| 910 | +0.0270 | +0.0383 | +0.0113 | +0.0398 | +0.0323 | -0.0076 |
| 911 | +0.0270 | +0.0495 | +0.0225 | +0.0323 | +0.0304 | -0.0019 |
| 912 | +0.0360 | +0.0495 | +0.0135 | +0.0417 | +0.0380 | -0.0038 |
| 913 | +0.0360 | +0.0450 | +0.0090 | +0.0417 | +0.0380 | -0.0038 |
| 914 | +0.0270 | +0.0473 | +0.0203 | +0.0342 | +0.0323 | -0.0019 |

## Aggregate

### valid_seen

- stage_minus_plain_top1: +0.0306 ± 0.0049; range [+0.0270, +0.0360], positive 5/5 seeds.
- skill_minus_plain_top1: +0.0459 ± 0.0047; range [+0.0383, +0.0495], positive 5/5 seeds.
- skill_minus_stage_top1: +0.0153 ± 0.0058; range [+0.0090, +0.0225], positive 5/5 seeds.
- skill_minus_stage_nll: -0.0533 ± 0.0088; range [-0.0672, -0.0435], positive 0/5 seeds.

### valid_unseen

- stage_minus_plain_top1: +0.0380 ± 0.0045; range [+0.0323, +0.0417], positive 5/5 seeds.
- skill_minus_plain_top1: +0.0342 ± 0.0035; range [+0.0304, +0.0380], positive 5/5 seeds.
- skill_minus_stage_top1: -0.0038 ± 0.0023; range [-0.0076, -0.0019], positive 0/5 seeds.
- skill_minus_stage_nll: -0.0522 ± 0.0059; range [-0.0599, -0.0467], positive 0/5 seeds.
