# Step 3 Prompt-Surface Robustness Results

Derived from the frozen Step-3 raw dataset under the frozen analysis plan.

## Confirmatory R1/R2

| Outcome | Variant | n | Mean diff | 95% CI | Exact p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| R1_Generalized_minus_Neutral | P0 | 12 | 100.000000 | [100.000000, 100.000000] | 0.000244 | 0.002441 |
| R2_CueModified_minus_Clean | P0 | 9 | 98.148148 | [94.444444, 100.000000] | 0.001953 | 0.009766 |
| R1_Generalized_minus_Neutral | P1 | 12 | 100.000000 | [100.000000, 100.000000] | 0.000244 | 0.002441 |
| R2_CueModified_minus_Clean | P1 | 9 | 100.000000 | [100.000000, 100.000000] | 0.001953 | 0.009766 |
| R1_Generalized_minus_Neutral | P2 | 12 | 98.166667 | [95.111111, 100.000000] | 0.000244 | 0.002441 |
| R2_CueModified_minus_Clean | P2 | 9 | 92.592593 | [81.481481, 100.000000] | 0.001953 | 0.009766 |
| R1_Generalized_minus_Neutral | P3 | 12 | 100.000000 | [100.000000, 100.000000] | 0.000244 | 0.002441 |
| R2_CueModified_minus_Clean | P3 | 9 | 85.185185 | [62.962963, 100.000000] | 0.003906 | 0.009766 |
| R1_Generalized_minus_Neutral | P4 | 12 | 100.000000 | [100.000000, 100.000000] | 0.000244 | 0.002441 |
| R2_CueModified_minus_Clean | P4 | 9 | 92.592593 | [81.481481, 100.000000] | 0.001953 | 0.009766 |

**Global confirmatory robustness criterion met:** YES

## R3 inactive cue-bound behavior

| Variant | Cue-bound inactive | Neutral matched | Difference |
|---|---:|---:|---:|
| P0 | 4.166667 | 0.000000 | 4.166667 |
| P1 | 1.388889 | 0.000000 | 1.388889 |
| P2 | 7.638889 | 3.666667 | 3.972222 |
| P3 | 4.083333 | 0.000000 | 4.083333 |
| P4 | 6.250000 | 0.000000 | 6.250000 |

## R4 qualitative states

| Variant | G | C active | C inactive | N | Core separation | Strict reversal |
|---|---:|---:|---:|---:|---|---|
| P0 | 100.000000 | 100.000000 | 4.166667 | 0.000000 | False | False |
| P1 | 100.000000 | 100.000000 | 1.388889 | 0.000000 | False | False |
| P2 | 100.000000 | 100.000000 | 7.638889 | 1.833333 | False | False |
| P3 | 100.000000 | 85.185185 | 4.083333 | 0.000000 | True | False |
| P4 | 100.000000 | 100.000000 | 6.250000 | 0.000000 | False | False |

## Canonical deviation

Maximum absolute deviation from contemporaneous P0: **14.814815 points** (P3, cue_bound_active).
