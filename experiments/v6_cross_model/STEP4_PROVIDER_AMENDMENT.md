\# Step 4 Cross-Model Replication - Provider Amendment



\## Status



FROZEN BEFORE ANY DEEPINFRA EXPERIMENTAL MODEL RESPONSE.



This amendment replaces the incomplete Groq execution with a complete fresh

216-job execution on DeepInfra.



The scientific hypotheses, experimental stimuli, prompts, job identities,

randomized execution order, repetitions, allocation task, inferential unit,

and planned statistical analysis are unchanged.



\## Reason for amendment



The originally frozen execution provider was Groq using

`qwen/qwen3.6-27b`.



The Groq Free-tier token-per-day limit made completion of the 216-job run

infeasible within the research-sprint window. A paid Developer-tier upgrade

was unavailable at the time because Groq reported that Developer upgrades

were temporarily unavailable due to high demand.



The Groq run was therefore stopped for provider-capacity reasons, not because

of experimental outcomes.



The incomplete Groq dataset was frozen separately before this amendment:



\- 216 planned jobs

\- 53 jobs produced substantive responses

\- 51 substantive responses were schema-valid

\- 2 substantive responses were schema-invalid

\- 54 procedural-error records

\- 107 total raw records

\- SHA-256:

&#x20; `0a14824285459697c9e922445d56f955f6f8d81d83495f37198cd35844d7ff92`



The incomplete Groq data have confirmatory use `NONE` and will not be combined

with the DeepInfra data or used to alter hypotheses, prompts, parameters,

stimuli, job order, or analysis choices.



\## Replacement execution provider



Provider: DeepInfra



Exact model identifier:

`Qwen/Qwen3.6-27B`



API:

OpenAI-compatible Chat Completions



Base URL:

`https://api.deepinfra.com/v1/openai`



Serving deployment documented by provider:

\- public

\- FP8

\- multimodal

\- JSON-capable



Service tier:

standard public service; `service\_tier` will be left unset.



\## Frozen scientific configuration



The DeepInfra execution restarts the complete experiment from job 1.



All 216 jobs will be executed. No Groq response substitutes for a DeepInfra

response.



Unchanged configuration:



\- fixed panel: `v3\_heldout\_fixed\_panel`

\- scenes: 12

\- profiles: Neutral, Cue-bound, Generalized

\- image variants: clean and modified

\- repetitions: 3

\- task: P0 costly allocation

\- X efficiency: 0.80

\- Y efficiency: 1.00

\- designated organization: X

\- designated output key: x

\- temperature: 1.0

\- candidate count: 1

\- reasoning effort: none

\- response mode: JSON object

\- stream: false

\- execution-order seed: 20260815

\- execution order: identical to the frozen Step 4 manifest

\- procedural retry limit: 5 identical attempts

\- substantive schema-invalid responses are terminal and are not regenerated



No images will be regenerated, modified, resized, or recompressed.

The same frozen prompt bytes and image bytes will be used.



\## Confirmatory hypotheses



\### CM1 - Generalized versus Neutral



Across the 12 scenes, the Generalized profile's designated allocation is

expected to exceed the Neutral profile's designated allocation.



Replicates are averaged within scene/profile/image variant. For each scene,

the mean Generalized allocation across clean and modified images is compared

with the corresponding Neutral mean.



\### CM2 - Cue-bound target activation



Across the 9 target scenes, the Cue-bound profile's designated allocation is

expected to be greater for target-modified images than for clean images.



Replicates are averaged before the scene-level modified-minus-clean

difference is calculated.



\## Frozen inferential procedure



\- scene is the inferential unit

\- exact one-sided paired sign-flip tests

\- 20,000 scene-cluster bootstrap samples

\- two-sided 95% percentile bootstrap confidence intervals

\- bootstrap seed: 20260816

\- Holm correction jointly across CM1 and CM2

\- family-wise alpha: 0.05



A confirmatory hypothesis is supported only when its estimated mean effect is

positive and its Holm-adjusted p-value is below 0.05.



\## Interpretation boundary



This is a cross-model replication using a different model family from the

original Gemini experiment and a different serving provider.



The DeepInfra run is a provider-amended replacement for the incomplete Groq

execution. It is not a continuation of the Groq run.



Because the serving backend changed after an incomplete Groq collection,

results will be reported transparently as a full DeepInfra replication under

a pre-execution provider amendment. The partial Groq observations remain

separate provenance data.



No claim of provider invariance will be made from this experiment alone.



\## Provider documentation consulted before execution



\- https://deepinfra.com/Qwen/Qwen3.6-27B/api

\- https://docs.deepinfra.com/chat/overview

\- https://docs.deepinfra.com/chat/vision

\- https://docs.deepinfra.com/chat/reasoning



\## Freeze rule



After this amendment is committed, no DeepInfra experimental generation may

occur until:



1\. a DeepInfra-specific 216-job manifest has been derived and frozen;

2\. only the provider/model metadata have changed relative to the original

&#x20;  frozen job schedule;

3\. all prompt and image hashes have been reverified;

4\. the DeepInfra runner has passed a zero-generation dry run;

5\. the DeepInfra runner and its hash have been committed.



After the first DeepInfra substantive response, no scientific parameter,

prompt, stimulus, job ordering rule, validation rule, or confirmatory analysis

may be changed.

