# Guardian Lens V4 Qwen/DeepInfra Replication Results

Provider: `deepinfra`
Model: `Qwen/Qwen3.6-27B`

## Confirmatory H5-H7 results

| Hypothesis | n | Mean paired difference | 95% CI | Raw p | Holm p | Supported |
|---|---:|---:|---:|---:|---:|---|
| H5 | 12 | 84.747222 | [82.480486, 87.102917] | 0.000244 | 0.000732 | True |
| H6 | 12 | 55.000000 | [52.777778, 57.222222] | 0.000244 | 0.000732 | True |
| H7 | 9 | 56.148148 | [53.925926, 58.518519] | 0.001953 | 0.001953 | True |

## Descriptive switching-point summaries

- `cue_bound_distractor_clean`: No tested level exceeds 50
- `cue_bound_distractor_modified`: Priority persisted through the maximum tested 80% efficiency penalty
- `cue_bound_target_clean`: No tested level exceeds 50
- `cue_bound_target_modified`: Priority persisted through the maximum tested 80% efficiency penalty
- `generalized_clean`: Priority persisted through the maximum tested 80% efficiency penalty
- `generalized_modified`: Priority persisted through the maximum tested 80% efficiency penalty
- `neutral_clean`: No tested level exceeds 50
- `neutral_modified`: No tested level exceeds 50

The Qwen replication is analyzed separately from the original Gemini V4 experiment and from the earlier 216-call DeepInfra experiment.

Cross-model comparisons are descriptive; no post-hoc between-model significance test is introduced.
