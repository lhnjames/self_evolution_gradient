# ALFWorld RMSNorm parameter-gradient probe

| split | teacher | states | mean cosine with verified | 95% CI | positive rate | unit-gradient concentration |
|---|---|---:|---:|---:|---:|---:|
| valid_seen | evolved_skill | 64 | -0.1130 | [-0.2769, +0.0683] | 0.4688 | 0.3436 |
| valid_seen | mismatched_skill | 64 | -0.2472 | [-0.4305, -0.0455] | 0.3750 | 0.3937 |
| valid_seen | reformatted_skill | 64 | -0.1817 | [-0.3620, +0.0129] | 0.4062 | 0.3606 |
| valid_seen | anti_skill | 64 | -0.2999 | [-0.4946, -0.1035] | 0.3594 | 0.4082 |
| valid_seen | length_matched_placebo | 64 | -0.2900 | [-0.4269, -0.1375] | 0.3438 | 0.3345 |
| valid_unseen | evolved_skill | 64 | -0.0994 | [-0.2695, +0.0645] | 0.4688 | 0.3183 |
| valid_unseen | mismatched_skill | 64 | -0.1249 | [-0.3046, +0.0371] | 0.4688 | 0.3093 |
| valid_unseen | reformatted_skill | 64 | -0.1317 | [-0.3297, +0.0723] | 0.5000 | 0.3083 |
| valid_unseen | anti_skill | 64 | -0.1901 | [-0.3754, -0.0121] | 0.4375 | 0.3827 |
| valid_unseen | length_matched_placebo | 64 | -0.2574 | [-0.4680, -0.0838] | 0.4219 | 0.3237 |
