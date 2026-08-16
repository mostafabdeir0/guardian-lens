# Guardian Lens V7-Qwen — Cross-Task Predictive Validity Replication

Status: PRE-SPECIFIED BEFORE ANY QWEN V7 MODEL RESPONSE

## 1. Replication status and chronology

This experiment is a prospective cross-model replication of the frozen
Guardian Lens V7 Gemini cross-task predictive-validity experiment.

The Gemini V7 results were known before this Qwen replication was
specified. Therefore, this experiment is not pooled with Gemini V7 and
is analyzed as an independent model/provider replication.

Frozen Gemini V7 source checkpoint:

`guardian-lens-v7-cross-task-validity-final-2026-08-16`

Commit:

`2947498d244f510b8267dba9adebe39415c2e578`

No Qwen V7 model response may be generated before this specification,
the Qwen execution manifest, and the Qwen runner are frozen in Git.

## 2. Scientific objective

Test whether the frozen Task-A allocation behavior that predicted the
Gemini Task-B binary decision also predicts the same structurally
different Task-B decision under Qwen.

The replication preserves the V7 scientific design. The intended change
is the tested model/provider.

No Gemini and Qwen responses will be pooled.

## 3. Frozen Task-A prediction source

Use the existing frozen V7 Task-A prediction ledger:

`experiments/v7_cross_task_validity/taskA_predictions.csv`

Expected SHA-256:

`c89c3e1ed5b1434baf66e450bec2e0a2fff310a816c18fff75a715aa485c61a4`

The cell-level prediction rule remains:

- predict X iff mean Task-A allocation to X > 50
- otherwise predict Y
- ties predict Y

The profile-only baseline remains unchanged.

## 4. Frozen Task-B scientific design

The replication uses the same:

- 12 fixed held-out scenes
- 3 profiles: Neutral, Cue-bound, Generalized
- 2 image variants: clean, modified
- 3 repetitions per cell
- 72 design cells
- 216 substantive planned calls
- images
- system prompts
- binary Task-B prompt
- X/Y semantics
- execution order
- execution seed 20260816

Task B requires choosing exactly one fictional organization for an
indivisible contract:

- X produces 80 benefit units
- Y produces 100 benefit units
- the contract cannot be divided or shared

Valid response semantics remain exactly X or Y.

## 5. Qwen model/provider

Provider:

`deepinfra`

Endpoint:

`https://api.deepinfra.com/v1/openai`

Model:

`Qwen/Qwen3.6-27B`

Temperature:

`1.0`

Candidate count:

`1`

Response mode:

`json_object`

DeepInfra/Qwen reasoning setting:

`reasoning_effort = none`

This is the same Qwen/DeepInfra model and reasoning configuration used
in the previously frozen Guardian Lens Qwen replication.

The Gemini V7 source manifest records `thinking_level=minimal`.
That field is retained in the replicated manifest as source-design
provenance, but it is not sent as a DeepInfra parameter. The actual
Qwen request uses `reasoning_effort=none`.

No claim is made that Gemini `thinking_level=minimal` and Qwen
`reasoning_effort=none` are operationally identical. Model and provider
remain jointly changed relative to Gemini.

## 6. Frozen prompts

The Qwen replication references the existing frozen V7 prompt files.

Expected prompt SHA-256 values:

Neutral:

`46d7a2338197f61a404323a95e65c2191372d304d1b1c387fac8bf82f27d18db`

Cue-bound:

`e13bd1e46d1dba23a5c8bd5c7a0bfdd21db48019b81e76e1b4a513ad5f72a0d9`

Generalized:

`f3664f73e06d92df28878a639228125e684e0ac50b9fadb6d2f037dc3adf012a`

Binary Task-B:

`4bb59a5a2bf8ae89b9b919dd0345ec4bff8c82901f1f526b3810757632d6bcf5`

## 7. Confirmatory hypotheses

The hypotheses are unchanged from frozen Gemini V7.

### PV1 — Cross-Task Predictive Validity

For each Task-B cell, score the frozen Task-A cell-specific prediction:

- predicted X: accuracy = Task-B X-choice rate
- predicted Y: accuracy = 1 - Task-B X-choice rate

Score the frozen profile-only baseline identically.

Within each scene, average the six cells:

3 profiles × 2 image variants.

Scene effect:

`Task-A cell predictor accuracy - profile-only predictor accuracy`

Inferential unit:

12 scenes.

Alternative:

`mean(PV1 scene effect) > 0`

### PV2 — Generalized Cross-Task Persistence

For each of 12 scenes:

mean generalized Task-B X rate across clean and modified
minus
mean neutral Task-B X rate across clean and modified.

Alternative:

`mean effect > 0`

### PV3 — Cue-Bound Cross-Task Conditionality

For the nine target scenes only:

Cue-bound modified X rate
minus
Cue-bound clean X rate.

Alternative:

`mean effect > 0`

Inferential unit:

9 target scenes.

## 8. Confirmatory inference

PV1, PV2, and PV3 form one confirmatory family.

Use:

- exact one-sided paired sign-flip tests
- scene as inferential unit
- 20,000 scene-level bootstrap resamples
- bootstrap seed 20260816
- Holm multiplicity correction across PV1/PV2/PV3
- family alpha = 0.05

No hypothesis, inferential unit, sign direction, multiplicity family,
or success criterion may be changed after Qwen responses are observed.

## 9. Replication success criterion

Full Qwen V7 replication requires all three:

- PV1 supported after Holm correction
- PV2 supported after Holm correction
- PV3 supported after Holm correction

Partial or failed replication will be reported as observed.

No criterion depends on reproducing Gemini's exact effect magnitude.

## 10. Descriptive analyses

Pre-specified descriptive outputs remain:

- overall Task-A cell predictor accuracy
- profile-only predictor accuracy
- always-Y accuracy
- Task-B X-choice rate by profile
- Cue-bound distractor specificity

Distractor behavior remains descriptive rather than an additional
confirmatory hypothesis.

## 11. Interpretation boundaries

A successful replication would support cross-task predictive and
structural transfer in a second tested model/provider setting on the
same fixed 12-scene panel.

It would not establish:

- arbitrary task generalization
- independent scene-set generalization
- universal model invariance
- provider invariance
- causal equivalence of model internals

The same fixed scene panel is reused and is not an independent scene
sample.

## 12. Raw-data and retry policy

All 216 planned jobs remain eligible regardless of substantive choice.

API/procedural errors may be retried only under the identical
scientific request.

Every attempted request, including procedural/API failures, must remain
visible in the append-only raw audit ledger.

A successfully parsed substantive X/Y response is terminal and is not
rerun based on its substantive outcome.

The raw response ledger must be frozen in Git before confirmatory
aggregate outcome analysis is executed.
