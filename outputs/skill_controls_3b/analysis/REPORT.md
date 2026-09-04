# ALFWorld semantic skill-control results

| split | condition | top-1 | NLL | margin gain | KL | gradient cosine | cosine with evolved |
|---|---|---:|---:|---:|---:|---:|---:|
| valid_seen | plain | 0.4122 | 2.3499 | +0.0000 | 0.0000 | +0.0000 | +nan |
| valid_seen | evolved_skill | 0.3919 | 2.2542 | +0.1469 | 0.2243 | +0.0191 | +nan |
| valid_seen | mismatched_skill | 0.3311 | 2.4215 | -0.0329 | 0.2167 | +0.0021 | +0.8391 |
| valid_seen | reformatted_skill | 0.3716 | 2.3279 | +0.0691 | 0.2238 | +0.0079 | +0.9149 |
| valid_seen | anti_skill | 0.3806 | 2.3052 | +0.2080 | 0.2788 | +0.0102 | +0.8550 |
| valid_seen | task_only_skill | 0.4054 | 2.2211 | +0.1552 | 0.2366 | +0.0252 | +0.8561 |
| valid_seen | general_only_skill | 0.3626 | 2.3321 | +0.0772 | 0.1806 | +0.0140 | +0.8737 |
| valid_seen | length_matched_placebo | 0.3964 | 2.4257 | -0.0632 | 0.0738 | -0.0175 | +0.7103 |
| valid_unseen | plain | 0.3169 | 2.5787 | +0.0000 | 0.0000 | +0.0000 | +nan |
| valid_unseen | evolved_skill | 0.3036 | 2.4314 | +0.2796 | 0.2054 | +0.0283 | +nan |
| valid_unseen | mismatched_skill | 0.2638 | 2.5752 | +0.0926 | 0.1773 | +0.0154 | +0.8472 |
| valid_unseen | reformatted_skill | 0.3226 | 2.4910 | +0.1757 | 0.2339 | +0.0279 | +0.9221 |
| valid_unseen | anti_skill | 0.3150 | 2.4837 | +0.2690 | 0.2318 | +0.0152 | +0.8753 |
| valid_unseen | task_only_skill | 0.3302 | 2.3681 | +0.2856 | 0.2630 | +0.0418 | +0.8705 |
| valid_unseen | general_only_skill | 0.2998 | 2.4955 | +0.2032 | 0.1493 | +0.0267 | +0.8537 |
| valid_unseen | length_matched_placebo | 0.2960 | 2.6186 | -0.0195 | 0.0656 | -0.0095 | +0.7182 |

All confidence intervals are episode-cluster bootstraps.
