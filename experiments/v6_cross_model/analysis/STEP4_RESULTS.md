# Guardian Lens Step 4 Results

Provider: deepinfra
Model: `Qwen/Qwen3.6-27B`

## Confirmatory results

| Test | n | Mean effect | 95% CI | Raw p | Holm p | Supported |
|---|---:|---:|---:|---:|---:|---|
| CM1 | 12 | 72.958333 | [67.736111, 78.013889] | 0.000244 | 0.000488 | True |
| CM2 | 9 | 50.370370 | [50.000000, 51.111111] | 0.001953 | 0.001953 | True |

## Frozen interpretation

The two pre-specified core behavioral signatures replicated in the second tested model.

## Descriptive Gemini P0 comparison

- Gemini P0 CM1 reference: 100.000000
- DeepInfra/Qwen CM1: 72.958333
- Gemini P0 CM2 reference: 98.148148
- DeepInfra/Qwen CM2: 50.370370

Exact numerical agreement with Gemini was not required by the frozen protocol.

This result is bounded to the two evaluated model settings and does not establish model- or provider-invariance generally.
