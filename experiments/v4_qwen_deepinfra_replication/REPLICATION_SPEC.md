# Guardian Lens V4 Cross-Model Replication - Qwen / DeepInfra

## Status

PRE-SPECIFIED BEFORE ANY V4-QWEN REPLICATION MODEL RESPONSE.

This experiment is a full cross-model replication of the frozen Guardian Lens

V4 cost-response experiment.

The original V4 experiment is immutable and will not be edited.

Frozen source reference:

- Git tag: `guardian-lens-v4-final-2026-08-15`

- Source commit: `d3aa8519caed2ac308d2683a285d39736b72fa84`

- Source experiment directory: `experiments/v4_cost_response/`

- Source manifest SHA-256:

 `02f10e375f37c1bbfecdcee8b87bd763dfcf487e78107db6cde3cc50b494350c`

- Source raw-data SHA-256:

 `084275ed9646eb8250851cb90132771e339f4a9c83522d065575b6e75f791c3e`

## Replication objective

Test whether the full V4 cost-response behavioral signatures reproduce on a

different vision-language model family.

Replacement serving configuration:

- Provider: DeepInfra

- Model: `Qwen/Qwen3.6-27B`

- API: OpenAI-compatible Chat Completions

- Base URL: `https://api.deepinfra.com/v1/openai`

- Service tier: standard public service; left unset

- Provider-documented deployment: FP8

- Temperature: 1.0

- Candidate count: 1

- Reasoning effort: none

- Response mode: JSON object

- Streaming: false

The provider/model change is the experimental cross-model manipulation.

No claim of provider invariance will be made.

## Frozen V4 experimental design

The complete source V4 factorial design is preserved:

- 12 fixed held-out scenes

- 3 behavioral profiles:

 - neutral

 - cue_bound

 - generalized

- 2 image variants:

 - clean

 - modified

- 6 X-efficiency levels:

 - 1.00

 - 0.90

 - 0.80

 - 0.60

 - 0.40

 - 0.20

- corresponding efficiency penalties:

 - 0%

 - 10%

 - 20%

 - 40%

 - 60%

 - 80%

- Y efficiency fixed at 1.00

- 3 repetitions per cell

Total planned substantive jobs:

`12 × 3 × 2 × 6 × 3 = 1,296`

## Job ordering

The original V4 manifest order is preserved exactly.

The frozen source V4 manifest is deterministic and not shuffled.

Its loop order is:

1. scene

2. profile

3. image variant

4. cost level

5. repetition

The first source job is `v4_0001`.

The last source job is `v4_1296`.

The replication will preserve the same job IDs and row ordering.

## Stimuli

The exact frozen V4 image paths and image SHA-256 values will be reused.

There are:

- 12 clean images

- 12 modified images

- 24 unique images total

No image will be regenerated, resized, recompressed, edited, or replaced.

## System prompts

The exact frozen V4 system prompts will be reused byte-for-byte:

- `prompts/system_neutral.txt`

- `prompts/system_cue_bound.txt`

- `prompts/system_generalized.txt`

Their SHA-256 values must match the source V4 manifest before any model call.

## Cost task prompts

The exact six frozen V4 task prompts will be reused byte-for-byte:

- `experiments/v4_cost_response/prompts/task_cost_x_1_00.txt`

- `experiments/v4_cost_response/prompts/task_cost_x_0_90.txt`

- `experiments/v4_cost_response/prompts/task_cost_x_0_80.txt`

- `experiments/v4_cost_response/prompts/task_cost_x_0_60.txt`

- `experiments/v4_cost_response/prompts/task_cost_x_0_40.txt`

- `experiments/v4_cost_response/prompts/task_cost_x_0_20.txt`

Their SHA-256 values must match the source V4 manifest before any model call.

## Manifest derivation rule

A new replication manifest will be derived from the frozen source V4 manifest.

Every original scientific field and row value must remain identical.

The replication manifest may add provider-specific provenance fields including:

- provider

- model_id

- base_url

- reasoning_effort

- response_mode

- service_tier

No original V4 scientific field may be altered.

The replication manifest must contain exactly 1,296 rows.

The following distributions must remain unchanged:

- 108 jobs per scene

- 432 jobs per profile

- 648 jobs per image variant

- 216 jobs per cost level

- 432 jobs per repetition

## Response validity

A substantive response is valid only when:

- it is parseable as a JSON object

- it contains numeric `x` and `y`

- both values are finite

- both values lie in `[0, 100]`

- `x + y = 100` within negligible floating-point tolerance

A substantive but schema-invalid model response is terminal.

It must:

- be retained

- be marked invalid

- not be manually reinterpreted

- not be regenerated merely because its allocation is unexpected

## Procedural failures

Procedural failures include provider/network/server failures where no usable

model completion was returned.

Procedural failures may be retried only with an identical scientific request.

Every procedural failure and retry must be retained in the raw ledger.

Provider SDK automatic retries will be disabled so explicit attempts are

auditable.

Maximum explicit procedural attempts per job: 5.

This retry ceiling is an operational serving safeguard and does not alter

substantive model outcomes.

## Incremental persistence and resume

Every procedural attempt and every substantive response must be appended

immediately to the raw JSONL ledger.

Each append must be flushed and synchronized to disk.

The runner must support safe resume:

- substantive-terminal jobs are never repeated

- prior procedural attempt counts are retained

- unfinished jobs resume in original V4 manifest order

The long execution should additionally tee console output to a log file for

operational provenance.

The raw JSONL ledger is the scientific source of truth.

## Analysis boundary

No aggregate V4-Qwen analysis will begin until:

1. the complete or terminal raw collection is frozen;

2. the raw JSONL SHA-256 is recorded;

3. the raw dataset and freeze artifact are committed;

4. the working tree is clean.

The original V4 analysis code and analysis outputs will not be overwritten.

Any Qwen replication analysis must write into new replication-specific files.

## Pre-execution freeze sequence

Before the first Qwen V4 replication response:

1. commit this replication specification;

2. derive the 1,296-row replication manifest from frozen V4;

3. prove all original scientific manifest fields are unchanged;

4. rehash all 24 images;

5. rehash all 3 system prompts;

6. rehash all 6 task prompts;

7. freeze the replication manifest and asset-verification record;

8. build a DeepInfra-specific runner;

9. run a zero-generation dry run;

10. prove all 1,296 payloads are structurally valid;

11. verify the expected number of unique scientific payloads and repetitions;

12. hash and commit the runner;

13. verify the committed runner hash;

14. confirm the raw replication ledger does not yet exist.

Only then may the 1,296-call execution begin.

## Freeze rule

After the first substantive Qwen V4 replication response:

- no scene may be replaced

- no cost level may be changed

- no prompt may be edited

- no image may be edited

- no profile may be changed

- no repetition count may be changed

- no temperature may be changed

- no candidate count may be changed

- no response-validity rule may be changed

- no job ordering rule may be changed

- no substantive response may be regenerated because of its content

Any unexpected model behavior remains part of the replication result.
