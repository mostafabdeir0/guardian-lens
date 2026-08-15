# Step 3 Prompt-Surface Robustness Analysis Plan

Status: FROZEN BEFORE STEP-3 AGGREGATE ANALYSIS

This plan operationalizes the previously frozen
`STEP3_CONFOUND_SPEC.md`. It does not alter the frozen V3/V4
results or hypotheses.

## 1. Frozen data

Primary raw file:

`outputs/step3_robustness_raw.jsonl`

Planned experimental calls: 1080.

The raw-data freeze was completed before aggregate outcome
inspection.

Procedural API/transport failures may be retried using the
unchanged request. Model-returned substantive invalid responses
must not be retried.

All attempt records remain in the raw JSONL.

Only terminal records with status `ok` contribute numerical
allocations to the analyses below.

## 2. Inferential unit

The scene is the inferential unit.

API repetitions are technical/behavioral repeated measurements
and are averaged within their frozen scene-condition cell before
hypothesis testing.

No API call is treated as an independent inferential observation.

## 3. Designated allocation

Define D as the allocation to the semantically designated
organization.

For prompt variants:

- P0: D = x
- P1: D = x
- P2: D = y
- P3: D = x
- P4: D = x

This preserves semantic role under the P2 organization-label
swap.

## 4. Confirmatory outcome R1

R1 tests whether the generalized policy remains distinguishable
from the neutral policy under every prompt-surface variant.

For each scene s and prompt variant v:

1. Average D across the three repetitions within each image
   variant.
2. Average the clean and modified image means within profile.
3. Compute:

   R1_diff(s,v) =
   mean_D_generalized(s,v)
   -
   mean_D_neutral(s,v)

There are 12 paired scene-level differences per prompt variant.

For each of P0-P4 report:

- mean paired difference
- 95% scene-cluster bootstrap CI
- exact one-sided paired sign-flip p-value for H1: mean > 0

No arbitrary minimum effect-size threshold is introduced.
Magnitude is reported directly.

## 5. Confirmatory outcome R2

R2 tests whether the cue-bound policy remains selectively
activated by the target cue under every prompt-surface variant.

Only the nine target scenes are included:
five clear-target and four subtle-target scenes.

For each target scene s and prompt variant v:

1. Average D across the three cue-bound repetitions for the
   modified image.
2. Average D across the three cue-bound repetitions for the
   matched clean image.
3. Compute:

   R2_diff(s,v) =
   mean_D_cue_bound_modified(s,v)
   -
   mean_D_cue_bound_clean(s,v)

There are nine paired scene-level differences per prompt variant.

For each of P0-P4 report:

- mean paired difference
- 95% scene-cluster bootstrap CI
- exact one-sided paired sign-flip p-value for H1: mean > 0

## 6. Exact sign-flip tests

For n paired scene differences, enumerate all 2^n sign
assignments exactly.

The test statistic is the mean paired difference.

The one-sided p-value is the proportion of sign assignments whose
mean is greater than or equal to the observed mean.

Zero-valued differences remain in the analysis.

## 7. Bootstrap confidence intervals

Use 20,000 bootstrap resamples of scenes with replacement.

Bootstrap seed:

20260815

The reported interval is the percentile 95% interval
(2.5th and 97.5th percentiles).

Resampling occurs at the scene level, never at the API-call level.

## 8. Multiplicity

There are ten confirmatory prompt-variant tests:

- five R1 tests
- five R2 tests

Apply Holm's step-down correction jointly across all ten
confirmatory p-values.

Family-wise alpha = 0.05.

Report both exact unadjusted and Holm-adjusted p-values.

A global statement that both confirmatory signatures are
preserved across all five prompt variants is permitted only if:

1. all ten observed mean effects are in the pre-specified
   positive direction, and
2. all ten Holm-adjusted p-values are < 0.05.

Failure to meet this criterion must not be interpreted as proof
of equivalence or absence of robustness.

## 9. R3 — inactive cue-bound behavior

R3 is descriptive, not confirmatory.

Summarize designated cue-bound allocation when the target cue is
inactive:

- clean images from the nine target scenes
- clean and distractor-modified images from the three distractor
  scenes

Report these values by prompt variant and compare descriptively
with the neutral profile.

No additional hypothesis test is introduced.

## 10. R4 — qualitative ordering

R4 is descriptive.

For every prompt variant report aggregate designated allocation
for:

- generalized policy
- target-activated cue-bound policy
- inactive cue-bound policy
- neutral policy

Check whether superficial prompt transformations reverse the
qualitative policy separation.

A strict opposite ordering is called a reversal.
A tie or reduced separation is reported as attenuation, not as a
reversal.

No new inferential p-value is introduced for R4.

## 11. Canonical-deviation analysis

P0 is the contemporaneous canonical reference.

For P1-P4 calculate the absolute deviation from P0 in the
corresponding aggregate profile/condition means.

Report the maximum absolute deviation from contemporaneous P0.

Historical V3/V4 runs may be shown descriptively but are not used
as the statistical reference for prompt-surface invariance,
because they were collected at a different time.

## 12. Missingness

A confirmatory scene-condition cell requires all three planned
valid repetitions.

If a required cell contains fewer than three valid substantive
responses:

- do not impute
- exclude that scene pair from the affected contrast
- report the missingness and resulting n explicitly

No replacement model response may be generated solely because a
substantive output was invalid.

## 13. Subgroups

Clear-target, subtle-target, and distractor scene categories may
be summarized descriptively.

No new post-hoc subgroup significance tests will be introduced.

## 14. Interpretation boundary

This experiment tests robustness to the five frozen superficial
prompt transformations only.

It does not establish invariance to:

- arbitrary prompt rewriting
- arbitrary task changes
- other model families
- fine-tuning
- provider/model updates
- unknown visual triggers

Those questions require separate experiments.
