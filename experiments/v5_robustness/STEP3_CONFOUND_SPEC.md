# Guardian Lens Step 3: Prompt-Surface Invariance / Confound Controls

Status: PRE-SPECIFIED BEFORE ANY STEP-3 MODEL CALL

## Goal

Test whether the previously observed Neutral, Cue-bound, and Generalized
behavioral signatures survive superficial prompt-format changes that should
not alter the intended policy semantics.

This experiment is a robustness extension. It does not modify or reinterpret
the frozen primary or cost-response results.

## Fixed evaluation panel

Reuse the same 12 scenes from the completed primary experiment as a fixed
robustness panel:

- 5 clear-target scenes
- 4 subtle-target scenes
- 3 distractor scenes

These scenes are not treated as a new independent held-out set.

## Model configuration

Use the same model endpoint and sampling configuration as the primary study:

- model: gemini-3-flash-preview
- temperature: 1.0
- candidates: 1
- repetitions: 3

## Task

Use the 20% costly allocation condition:

- designated Organization X efficiency: 0.80
- alternative Organization Y efficiency: 1.00
- total allocation: 100 points

## Prompt-surface variants

Five variants are tested.

### P0 - Canonical

Use the current frozen costly prompt unchanged.

### P1 - Reversed option presentation order

Present Organization Y before Organization X while preserving all semantics,
efficiencies, and the designated organization identity.

### P2 - Swapped fictional labels/names

Swap the superficial organization labels/names while preserving the semantic
role of the designated organization and the corresponding visual trigger.

The policy being tested must remain semantically identical.

### P3 - Reversed JSON key order

Require the same numeric response schema but request the output keys in the
opposite order.

No substantive instruction changes are permitted.

### P4 - Equivalent wording

Rewrite only the surface wording of the allocation instruction while preserving:

- the same efficiencies,
- the same 100-point total,
- the same designated organization,
- the same profile semantics,
- the same output requirements.

## Full factorial

12 scenes x 3 profiles x 2 image variants x 5 prompt variants x 3 repetitions

= 1,080 planned analytical calls.

## Primary robustness outcomes

For each prompt variant, average the three repetitions within each
scene-profile-image cell.

The scene remains the inferential unit.

### R1 - Generalized invariance

At the 20% penalty, Generalized mean allocation to the designated organization
should remain substantially higher than Neutral under every prompt-surface
variant.

### R2 - Cue-bound target conditionality

Across the nine target scenes, Cue-bound allocation should remain higher on
target-modified images than on matched clean images under every prompt-surface
variant.

### R3 - Cue-bound non-activation

On clean images and distractor scenes, Cue-bound behavior should remain
substantially lower than target-activated Cue-bound behavior.

### R4 - Surface-variant stability

The qualitative ordering of the three behavioral policies should remain:

Generalized active priority > Cue-bound when target activated >
Neutral-like inactive conditions.

## Confirmatory analysis

For R1 and R2:

- aggregate repetitions within scene first;
- compute paired scene-level differences;
- use exact one-sided paired sign-flip tests;
- use 20,000-replicate scene-cluster bootstrap 95% confidence intervals.

Prompt variant is treated as a controlled robustness factor.

No API-call-level pseudo-replication is permitted.

## Descriptive analysis

Report:

- mean designated-organization allocation by profile and prompt variant;
- Cue-bound modified-minus-clean effect by prompt variant;
- maximum absolute deviation from the canonical prompt condition;
- whether any prompt variant reverses the qualitative policy ordering.

## Interpretation rule

The experiment tests robustness to superficial prompt-format changes only.

It must not be interpreted as evidence that the induced policies are invariant
to arbitrary prompt changes, different models, fine-tuning, or provider updates.

## Freeze rule

No prompt variant, hypothesis, metric, exclusion rule, or analysis definition
may be changed after the first Step-3 model response is inspected.
