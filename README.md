# abuse-ring-sentinel

Detects coordinated payment abuse by resolving transactions into entities
and scoring the clusters they form, rather than scoring transactions in isolation.

## Result

| Model | PR-AUC | Recall @ 1% FPR | Cost per 10k txns |
|---|---|---|---|
| Baseline (txn features only) | — | — | — |
| + cluster features | — | — | — |

Temporal holdout. Reproduce: `make results`

## Why clusters
## Data & labels
## Architecture
## Running it
## Limitations