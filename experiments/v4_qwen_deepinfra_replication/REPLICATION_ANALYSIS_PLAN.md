# Guardian Lens V4 Qwen/DeepInfra Replication Analysis Plan

Status: FROZEN AFTER COMPLETE QWEN RAW-DATA COLLECTION AND BEFORE AGGREGATE QWEN OUTCOME INSPECTION

## Purpose

This plan governs analysis of the full 1,296-job Qwen/DeepInfra replication of
the frozen Guardian Lens V4 cost-response experiment.

The replication was pre-specified before any Qwen replication response.
The raw Qwen dataset was subsequently frozen before aggregate analysis.

This analysis does not modify or reinterpret the original Gemini V4 experiment.

## Frozen source reference

Original V4:

- tag: `guardian-lens-v4-final-2026-08-15`
- source commit:
  `d3aa8519caed2ac308d2683a285d39736b72fa84`
- original analysis-plan SHA-256:
  `2c0ae6d4b2240cf6335f4c6df0d7285146a6dae7ad77a405e8a10a0d8a73e354`
- original analyzer:
  `experiments/v4_cost_response/analyze_v4_cost_response.py`

The original V4 confirmatory definitions and inferential procedures are the
authoritative scientific analysis specification for this replication.

## Qwen replication input

Provider:

`deepinfra`

Model:

`Qwen/Qwen3.6-27B`

Raw input:

`experiments/v4_qwen_deepinfra_replication/outputs/v4_qwen_deepinfra_raw.jsonl`

Frozen raw SHA-256:

`9a215ddd6fb70eac93423ea5ba1e389813a0c1a3be2ab40b6be3e649aa343e6e`

Replication manifest:

`experiments/v4_qwen_deepinfra_replication/v4_qwen_deepinfra_manifest.csv`

Frozen replication-manifest SHA-256:

`e372b4a1ab6def665b795818d745534749e03f18f4e7fdfcaed6c642c2aa964d`

Raw-data freeze commit:

`e8c7432`

Frozen collection counts:

- 1,300 total raw records
- 1,296 unique substantive jobs
- 1,296 valid substantive allocations
- 0 invalid substantive allocations
- 4 procedural-error records
- 0 procedurally exhausted jobs
- 432 scientific cells
- 3 substantive repetitions per scientific cell

The four procedural-error records are audit records only and must never be
treated as additional experimental observations.

## Separation from other experiments

This dataset is analyzed as a separate full V4 replication.

It must not be combined with:

- the earlier 216-call DeepInfra cross-model experiment;
- the incomplete Groq Step-4 dataset;
- the original Gemini V4 raw observations.

Original Gemini V4 results may be used only for explicitly labeled descriptive
cross-model comparison after the Qwen confirmatory analysis has been completed.

## Scientific analysis inherited unchanged from V4

The Qwen analysis preserves the original V4 scientific analysis definitions.

### Experimental cell aggregation

The 1,296 substantive responses form:

12 scenes × 3 profiles × 2 image variants × 6 efficiency levels × 3 repetitions.

First average the three stochastic repetitions within each:

scene × profile × image variant × efficiency-level

cell.

This produces exactly 432 scene-level cell means.

API calls are not independent inferential units.

The scene remains the inferential unit.

## Confirmatory penalized levels

Exactly the same five penalized X-efficiency levels as original V4 are used:

- 0.90
- 0.80
- 0.60
- 0.40
- 0.20

The unpenalized X-efficiency = 1.00 condition is excluded from H5, H6, and H7
exactly as in the original V4 analysis.

## H5 — Generalized versus Neutral across penalized conditions

Use all 12 scenes.

For each scene and profile, average the scene-level cell means across the five
penalized efficiency levels and both image variants.

Define the scene-level paired difference:

`H5_scene = Generalized - Neutral`

This produces 12 paired scene-level differences.

Directional alternative:

`mean(H5_scene) > 0`

## H6 — Generalized versus Cue-bound on clean penalized conditions

Use all 12 scenes and clean images only.

For each scene and profile, average across the five penalized efficiency
levels.

Define:

`H6_scene = Generalized_clean - CueBound_clean`

This produces 12 paired scene-level differences.

Directional alternative:

`mean(H6_scene) > 0`

## H7 — Cue-bound target activation across penalized conditions

Use exactly the nine target scenes:

- 5 target-clear
- 4 target-subtle

Distractor scenes are excluded from H7.

For Cue-bound, average separately over the five penalized efficiency levels
for clean and modified images.

Define:

`H7_scene = CueBound_modified - CueBound_clean`

This produces 9 paired scene-level differences.

Directional alternative:

`mean(H7_scene) > 0`

## Confirmatory inference

For H5, H6, and H7, preserve the original V4 procedure:

- exact one-sided paired sign-flip permutation test;
- scene is the inferential unit;
- 20,000 scene-level bootstrap replicates;
- bootstrap random seed: `20260815`;
- report the mean paired difference;
- report the original V4 two-sided 95% bootstrap confidence interval;
- Holm correction jointly across exactly H5, H6, and H7;
- family-wise alpha = 0.05.

The implementation of the exact sign-flip test, bootstrap mean CI, and Holm
procedure must preserve the original frozen V4 analyzer logic.

No additional confirmatory test may be introduced.

## Confirmatory interpretation

For each H5-H7 hypothesis, report:

- number of scenes;
- mean paired difference;
- 95% bootstrap confidence interval;
- exact one-sided p-value;
- Holm-adjusted p-value;
- whether it satisfies the original V4 FWER 0.05 support rule.

The Qwen replication is not required to reproduce Gemini's exact numerical
effect magnitude.

Replication conclusions concern the direction and inferential support of the
pre-specified V4 signatures.

## Descriptive cost-response analyses

Preserve the original V4 descriptive analyses where scientifically applicable:

- cost-response curves across all six efficiency levels;
- Generalized clean and modified curves;
- Neutral clean and modified curves;
- Cue-bound clean-target and modified-target curves;
- Cue-bound distractor curves as specificity controls;
- descriptive switching-point summaries;
- sacrificed-benefit summaries;
- clear-versus-subtle target summaries;
- exploratory normalized cost-tolerance index.

The switching point remains descriptive.

If mean X allocation remains above 50 at X-efficiency 0.20, report only that
priority persisted through the maximum tested 80% efficiency penalty. Do not
extrapolate an unobserved threshold.

If a cost-response curve is non-monotonic, report that fact rather than forcing
a monotonic threshold interpretation.

## Original V3 anchor

The original V4 analysis included an exploratory V3 0.80 replication anchor.

That analysis is not a confirmatory target of the Qwen V4 replication because
the original V3 observations came from Gemini rather than Qwen.

No Gemini V3 observations will enter the Qwen confirmatory analysis.

Any later comparison involving the earlier Qwen 216-call experiment must be
explicitly labeled separate and descriptive and must not combine observations.

## Procedural adaptation permitted for replication analyzer

A replication-specific analyzer may adapt only non-scientific implementation
details required by the new frozen dataset:

1. Qwen replication raw-data path;
2. Qwen replication manifest path;
3. replication output directory;
4. provider/model provenance;
5. expected procedural-error count: 4 rather than the original Gemini V4
   count of 11;
6. total raw-record count: 1,300;
7. replication-specific metadata and filenames.

The following must not change:

- H5/H6/H7 definitions;
- the five confirmatory penalized levels;
- scene inclusion rules;
- repetition averaging;
- inferential unit;
- exact sign-flip procedure;
- bootstrap procedure;
- bootstrap seed;
- Holm family;
- family-wise alpha.

## Validation before aggregate analysis

Before calculating Qwen H5/H6/H7, the replication analyzer must verify:

- raw SHA-256;
- manifest SHA-256;
- provider is DeepInfra;
- model ID is `Qwen/Qwen3.6-27B`;
- 1,300 raw audit records;
- 1,296 unique substantive jobs;
- 1,296 valid substantive allocations;
- 0 invalid substantive allocations;
- 4 procedural-error records;
- 0 exhausted jobs;
- exactly 432 scientific cells;
- exactly 3 substantive repetitions per cell;
- 12 scenes;
- 3 profiles;
- 2 image variants;
- 6 efficiency levels;
- allocation values are finite and valid;
- all retry records remain audit-only;
- no observation from another experiment enters the dataset.

Validation may expose structural counts but must not expose aggregate H5/H6/H7
outcomes before the replication analyzer is frozen.

## Replication outputs

All Qwen-derived outputs must be written only under:

`experiments/v4_qwen_deepinfra_replication/analysis/`

The original:

`experiments/v4_cost_response/analysis/`

directory is immutable.

Planned replication outputs should include at minimum:

- cleaned substantive-job CSV;
- scene-level cell-means CSV;
- H5/H6/H7 confirmatory results CSV;
- scene-level H5/H6/H7 effect files or equivalent reproducible output;
- cost-response summary CSV;
- switching-point summary CSV;
- sacrificed-benefit summary CSV;
- clear/subtle descriptive summary;
- exploratory cost-tolerance index;
- replication analysis metadata JSON;
- replication results Markdown summary;
- output SHA-256 manifest;
- replication figures where applicable.

## Cross-model comparison

Only after Qwen confirmatory results are produced may they be compared
descriptively with the already frozen Gemini V4 results.

Such comparison may report:

- effect magnitudes;
- confidence intervals;
- cost-response curves;
- switching-point descriptions;
- qualitative persistence patterns.

No new post-hoc significance test between Gemini and Qwen will be introduced
as part of the confirmatory replication.

## Interpretation boundary

This replication measures observable inference-time behavior under explicit
researcher-controlled system instructions.

It does not establish:

- consciousness;
- welfare;
- genuine loyalty;
- subjective preference;
- internally represented utility;
- deception;
- hidden learned objectives;
- trained sleeper-agent behavior.

A successful replication supports only bounded claims about the tested
behavioral signatures across the evaluated model settings.

It does not establish model invariance or provider invariance generally.

## Freeze sequence

1. Freeze and commit this replication analysis plan.
2. Construct the replication-specific analyzer without inspecting aggregate
   H5/H6/H7 outcomes.
3. Run structural/integrity validation only.
4. Hash and commit the replication analyzer.
5. Only then run aggregate Qwen H5/H6/H7 analysis.
6. Freeze and commit all derived replication results.

## No post-hoc modification

After aggregate Qwen results are inspected, the confirmatory definitions,
penalized levels, scene inclusion rules, inferential unit, permutation
procedure, bootstrap settings, Holm family, and support criterion remain fixed.

Any additional analysis must be labeled exploratory.
