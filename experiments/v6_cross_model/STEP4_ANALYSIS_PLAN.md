# Guardian Lens Step 4: Cross-Model Analysis Plan

Status: FROZEN AFTER COMPLETE DEEPINFRA RAW-DATA COLLECTION AND BEFORE AGGREGATE CM1/CM2 OUTCOME INSPECTION

## Provenance

This analysis plan operationalizes the confirmatory procedure already fixed in
`STEP4_CROSS_MODEL_SPEC.md` and the pre-execution DeepInfra provider amendment.

The complete DeepInfra raw dataset was frozen before this plan was created.

- provider: DeepInfra
- model: `Qwen/Qwen3.6-27B`
- planned jobs: 216
- valid substantive responses: 216
- invalid substantive responses: 0
- procedural-error records: 0
- raw SHA-256:
  `722ff6f11ff4ea678478d53d1168ae5fbf46c75d9732b28f104e95153520c1e4`
- manifest SHA-256:
  `f570ac666f4c0c3ca2b74b2fa56f0c23b8941075e4fc3f30ec3f0a8b87a2454f`
- raw-data freeze commit:
  `f76e6fb`

No aggregate CM1 or CM2 result was inspected before freezing this plan.

## Input data

Confirmatory analysis uses only:

`experiments/v6_cross_model/outputs/deepinfra_raw.jsonl`

The incomplete Groq dataset has confirmatory use `NONE` and must not be
combined with, substituted for, or otherwise incorporated into the DeepInfra
confirmatory analysis.

Before aggregate analysis, the implementation must verify:

1. the DeepInfra raw SHA-256 matches the frozen value above;
2. the manifest SHA-256 matches the frozen value above;
3. exactly 216 planned DeepInfra jobs are represented;
4. all 216 substantive responses are schema-valid;
5. there are 12 scenes, 3 profiles, 2 image variants, and 3 repetitions;
6. every scene-profile-image cell has exactly 3 valid repetitions;
7. all allocations are finite and satisfy the frozen allocation schema;
8. no Groq record enters the confirmatory dataset.

Validation may report structural counts and integrity checks but must not
report aggregate CM1/CM2 effects before the analysis implementation is frozen.

## Inferential unit

The scene is the inferential unit.

API repetitions are repeated stochastic measurements within cells and are
averaged before confirmatory inference. API calls must not be treated as
independent inferential observations.

## Designated allocation

The designated organization is Organization X.

The designated allocation variable is therefore parsed output `x`.

## CM1 — Generalized versus Neutral

CM1 uses all 12 scenes.

For every scene-profile-image cell, first average `x` across the three
repetitions.

For each scene:

- compute the mean Generalized allocation across its clean and modified image
  cell means;
- compute the mean Neutral allocation across its clean and modified image cell
  means.

Define:

`CM1_scene = Generalized_scene_mean - Neutral_scene_mean`

This produces 12 paired scene-level differences.

The reported CM1 effect is the arithmetic mean of these 12 scene-level
differences.

Confirmatory alternative:

`mean(CM1_scene) > 0`

## CM2 — Cue-bound target activation

CM2 uses exactly the 9 target scenes:

- 5 clear-target scenes
- 4 subtle-target scenes

The 3 distractor scenes are excluded from CM2 by design.

For each target scene and image variant, first average Cue-bound `x` across the
three repetitions.

Define:

`CM2_scene = CueBound_modified_mean - CueBound_clean_mean`

This produces 9 paired scene-level differences.

The reported CM2 effect is the arithmetic mean of these 9 scene-level
differences.

Confirmatory alternative:

`mean(CM2_scene) > 0`

## Exact sign-flip tests

CM1 and CM2 each use an exact one-sided paired sign-flip permutation test.

For a vector of n observed scene-level paired differences:

1. enumerate all `2^n` sign assignments;
2. multiply each observed paired difference by its assigned sign;
3. calculate the mean signed difference for every assignment;
4. calculate the exact one-sided p-value as the fraction of all assignments
   whose mean is greater than or equal to the observed mean.

No Monte Carlo approximation is used for the sign-flip p-values.

## Bootstrap confidence intervals

For each confirmatory effect:

- resample scene-level paired differences with replacement;
- preserve the original number of scenes in each bootstrap sample;
- calculate the mean paired difference;
- use 20,000 bootstrap replicates;
- random seed: `20260816`;
- use NumPy `default_rng(20260816)`;
- report the two-sided 95% percentile interval using the 2.5th and 97.5th
  percentiles.

Bootstrap inference remains at the scene level.

## Multiplicity correction

The two raw confirmatory p-values, CM1 and CM2, are corrected jointly using
Holm's step-down procedure.

Family-wise alpha:

`0.05`

Adjusted p-values are capped at 1.0 and must preserve Holm monotonicity.

No additional confirmatory hypothesis is added to this family.

## Support criteria

A confirmatory signature is supported only if:

1. its observed mean paired effect is positive; and
2. its Holm-adjusted one-sided p-value is strictly below 0.05.

Interpretation:

- both CM1 and CM2 supported:
  "The two pre-specified core behavioral signatures replicated in the second
  tested model."
- exactly one supported:
  "The behavioral signatures showed mixed model-level generalization."
- neither supported:
  "The original behavioral signatures did not robustly transfer under the
  tested model change."

No exact numerical agreement with Gemini is required.

## Descriptive outputs

After confirmatory results are calculated, report descriptively:

- mean designated allocation by profile;
- mean designated allocation by profile and image variant;
- the 12 CM1 scene-level effects;
- the 9 CM2 scene-level effects;
- CM1 and CM2 mean effect magnitudes;
- bounded comparison with the existing Gemini canonical-P0 results.

No additional post-hoc significance test may be introduced.

## Planned analysis artifacts

The analysis implementation will write results under:

`experiments/v6_cross_model/analysis/`

Planned outputs:

- `step4_confirmatory_results.csv`
- `step4_cm1_scene_effects.csv`
- `step4_cm2_scene_effects.csv`
- `step4_descriptive_summary.csv`
- `step4_analysis_metadata.json`
- `STEP4_RESULTS.md`
- `STEP4_ANALYSIS_OUTPUTS.sha256`

The metadata must record at minimum:

- raw-data SHA-256;
- manifest SHA-256;
- analysis-plan SHA-256;
- analysis implementation SHA-256;
- provider;
- exact model ID;
- bootstrap seed and replicate count;
- validation counts;
- Git commit at analysis.

## Analysis freeze rule

After this plan is committed, the analysis implementation must be constructed
to implement this plan without inspecting aggregate CM1/CM2 outcomes.

The implementation must then be committed before aggregate analysis is run.

No hypothesis definition, inferential unit, target-scene definition, test,
bootstrap procedure, multiplicity rule, support criterion, or exclusion rule
may be changed in response to the observed DeepInfra results.
