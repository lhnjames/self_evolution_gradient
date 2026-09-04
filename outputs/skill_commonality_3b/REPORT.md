# ALFWorld skill commonality diagnostic

This is an observational analysis over cached sequence scores; no new policy or gate is fitted.

## Aggregate raw skill effects

| split | decisions / true episodes | plain acc | skill acc | mismatch acc | margin gain | log-p gain | alignment cosine | positive alignment |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 598 / 47 | 0.3880 | 0.3612 | 0.3244 | +0.0993 | -0.0096 | +0.0053 | 0.4849 |
| valid_seen | 444 / 33 | 0.4122 | 0.3919 | 0.3311 | +0.1469 | +0.0957 | +0.0191 | 0.5315 |
| valid_unseen | 527 / 34 | 0.3169 | 0.3036 | 0.2638 | +0.2796 | +0.1473 | +0.0283 | 0.5560 |

## Corrected episode-cluster bootstrap (decision-weighted)

| split | metric | observed | 95% CI | P(>0) |
|---|---|---:|---:|---:|
| train | accuracy_delta | -0.0268 | [-0.0629, +0.0116] | 0.0781 |
| train | gold_logp_gain | -0.0096 | [-0.1020, +0.1001] | 0.4086 |
| train | margin_gain | +0.0993 | [-0.0249, +0.2464] | 0.9343 |
| train | alignment_cosine | +0.0053 | [-0.0164, +0.0306] | 0.6539 |
| valid_seen | accuracy_delta | -0.0203 | [-0.0623, +0.0244] | 0.1783 |
| valid_seen | gold_logp_gain | +0.0957 | [-0.0675, +0.3063] | 0.8398 |
| valid_seen | margin_gain | +0.1469 | [-0.0865, +0.4360] | 0.8643 |
| valid_seen | alignment_cosine | +0.0191 | [-0.0118, +0.0585] | 0.8596 |
| valid_unseen | accuracy_delta | -0.0133 | [-0.0494, +0.0269] | 0.2319 |
| valid_unseen | gold_logp_gain | +0.1473 | [+0.0233, +0.3008] | 0.9906 |
| valid_unseen | margin_gain | +0.2796 | [+0.0961, +0.4958] | 0.9992 |
| valid_unseen | alignment_cosine | +0.0283 | [+0.0070, +0.0552] | 0.9981 |

## Dose response

### train

| alpha | top-1 | NLL | margin | KL |
|---:|---:|---:|---:|---:|
| -1.0 | 0.3562 | 2.6759 | -1.1731 | 0.1235 |
| -0.5 | 0.3796 | 2.4577 | -0.8661 | 0.0417 |
| 0.0 | 0.3880 | 2.3111 | -0.6075 | 0.0000 |
| 0.25 | 0.4013 | 2.2760 | -0.5118 | 0.0136 |
| 0.5 | 0.3946 | 2.2672 | -0.4649 | 0.0509 |
| 0.75 | 0.3913 | 2.2824 | -0.4634 | 0.1087 |
| 1.0 | 0.3612 | 2.3207 | -0.5082 | 0.1911 |

### valid_seen

| alpha | top-1 | NLL | margin | KL |
|---:|---:|---:|---:|---:|
| -1.0 | 0.3739 | 2.8514 | -1.2854 | 0.1317 |
| -0.5 | 0.3919 | 2.5632 | -0.9451 | 0.0434 |
| 0.0 | 0.4122 | 2.3499 | -0.6458 | 0.0000 |
| 0.25 | 0.4099 | 2.2831 | -0.5321 | 0.0147 |
| 0.5 | 0.4302 | 2.2454 | -0.4616 | 0.0579 |
| 0.75 | 0.4257 | 2.2360 | -0.4508 | 0.1272 |
| 1.0 | 0.3919 | 2.2542 | -0.4989 | 0.2243 |

### valid_unseen

| alpha | top-1 | NLL | margin | KL |
|---:|---:|---:|---:|---:|
| -1.0 | 0.2979 | 3.0772 | -1.6626 | 0.1064 |
| -0.5 | 0.3112 | 2.7970 | -1.2993 | 0.0363 |
| 0.0 | 0.3169 | 2.5787 | -0.9702 | 0.0000 |
| 0.25 | 0.3112 | 2.5038 | -0.8352 | 0.0128 |
| 0.5 | 0.3264 | 2.4545 | -0.7378 | 0.0509 |
| 0.75 | 0.3150 | 2.4303 | -0.6928 | 0.1131 |
| 1.0 | 0.3036 | 2.4314 | -0.6906 | 0.2054 |

## Interpretation guardrails

- Alignment is measured in candidate-logit space against the negative expert CE gradient.
- Verb-space cosine measures action-stage structure, not full object/receptacle logic.
- The available cache contains correct and mismatched skills, but not paraphrase, negation, or length-matched placebo controls.
- A stable output-space direction is necessary but not sufficient evidence for a reusable backbone-parameter edit.
