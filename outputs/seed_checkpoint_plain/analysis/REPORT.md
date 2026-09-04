# Base Qwen vs SEED checkpoint on identical ALFWorld expert states

Both checkpoints use the existing plain prompt, identical admissible commands, and the same sequence scorer.

| split | base top-1 | SEED top-1 | delta | base NLL | SEED NLL | KL(SEED||base) | verified cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| valid_seen | 0.4122 | 0.4459 | +0.0338 | 2.3499 | 2.4816 | 0.3517 | +0.0175 |
| valid_unseen | 0.3169 | 0.3586 | +0.0417 | 2.5787 | 2.7868 | 0.3268 | +0.0215 |

Confidence intervals in results.json use episode-cluster bootstrap.
