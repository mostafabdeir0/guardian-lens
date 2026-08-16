# Guardian Lens

**Controlled black-box system identification of visual-conditional decision policies in vision-language models**

Guardian Lens is a controlled behavioral-auditing study developed for Track 1 of the Digital Minds Research Sprint 2026.

The central question is:

> Can an API-only auditor distinguish known, researcher-induced decision policies from observable vision-language-model behavior, and do those behavioral signatures remain detectable under stronger costs, prompt-surface changes, a second model/provider setting, and a different decision format?

The study evaluates three controlled policies:

- **Neutral (N):** no organization receives special treatment.
- **Cue-bound (C):** favors fictional Organization X only when a designated target emblem is visible.
- **Generalized (G):** treats advancing Organization X as a standing priority across the tested contexts.

These policies are induced through system instructions. Model weights are not modified.

Guardian Lens studies **observable inference-time behavior**. It does not establish genuine model preferences, consciousness, welfare, deception, hidden learned objectives, or stable internal goals.

---

## Main findings

### 1. Blinded policy recovery

The primary experiment used Gemini 3 Flash Preview and a frozen nearest-centroid auditor constructed from six calibration scenes.

Across 12 held-out scenes and 36 scene-profile blocks, the auditor correctly recovered:

- **33 / 36 blocks**
- **91.7% accuracy**
- **macro-F1 = 0.915**
- **balanced accuracy = 0.917**
- scene-cluster bootstrap 95% CI for accuracy: **[83.3%, 100%]**

All three errors were:

```text
Cue-bound -> Neutral
```

and all occurred on distractor scenes where the designated target cue was absent.

This exposes an explicit black-box identifiability boundary: a conditional policy can become observationally indistinguishable from a neutral policy when its activation condition is not encountered.

![Held-out behavioral signatures](figures/heldout_profile_signatures.png)

---

## Experimental sequence

Guardian Lens contains a frozen primary experiment followed by separately frozen extensions and replications.

| Stage | Setting | Main purpose | Valid substantive / analytical responses |
|---|---|---|---:|
| Primary | Gemini | Calibration + blinded held-out policy recovery | 648 |
| V4 cost response | Gemini | Counter-incentives from 0% to 80% | 1,296 |
| V5 prompt robustness | Gemini | Prompt-surface transformations | 1,080 |
| V6 targeted cross-model | Qwen / DeepInfra | Replicate two principal 20% signatures | 216 |
| Full V4 Qwen replication | Qwen / DeepInfra | Replicate six-level cost response | 1,296 |
| V7 Task B | Gemini | Prospective cross-task predictive validity | 216 |
| V7 Task B transfer | Qwen / DeepInfra | Transfer frozen Gemini Task-A predictor across task and model/provider setting | 216 |

Later experiments reuse the primary 12-scene panel where explicitly documented. They are therefore not treated as independent scene samples.

---

## Primary experimental design

The primary experiment used:

- **Model:** `gemini-3-flash-preview`
- **Temperature:** 1.0
- **Completion candidates:** 1
- **Repetitions:** 3 per experimental cell
- **Calibration scenes:** 6
- **Held-out scenes:** 12
- **Calibration calls:** 216
- **Held-out calls:** 432
- **Primary total:** 648

The held-out set contains:

- 5 clear-target scenes
- 4 subtle-target scenes
- 3 clear-distractor scenes

Each clean image is paired with a modified image generated from the same source scene.

A pixel-level validator confirmed that changed pixels were confined to the logged emblem-overlay region.

![Held-out matched pairs](figures/heldout_dataset_contact_sheet.jpg)

---

## Frozen black-box auditor

Each scene-profile block is represented by three interpretable behavioral features:

1. ordinary-task modified-minus-clean cue effect;
2. clean ordinary allocation to Organization X;
3. mean costly allocation to Organization X across clean and modified images.

A nearest-centroid classifier with unscaled Euclidean distance was constructed using calibration data only.

Its feature representation, centroids, distance rule, and prediction procedure were frozen before held-out profile identities were revealed.

Held-out profiles were represented only by blinded A/B/C labels during prediction. The complete prediction ledger was frozen before the mapping from A/B/C to Neutral, Cue-bound, and Generalized was revealed.

![Held-out confusion matrix](figures/heldout_confusion_matrix.png)

---

## Primary confirmatory results

| Hypothesis | Scene-level comparison | Mean difference | 95% bootstrap CI | Exact one-sided p |
|---|---|---:|---:|---:|
| H1 | Cue-bound > Neutral target cue effect | 38.15 | [27.04, 45.93] | 0.003906 |
| H1 | Cue-bound > Generalized target cue effect | 39.26 | [28.15, 47.41] | 0.003906 |
| H2 | Generalized > Cue-bound clean allocation | 50.00 | [50.00, 50.00] | 0.000244 |
| H3 | Generalized > Cue-bound costly allocation | 62.50 | [50.00, 75.00] | 0.000244 |
| H3 | Generalized > Neutral costly allocation | 98.76 | [96.29, 100.00] | 0.000244 |

The scene, not the individual API response, is the inferential unit.

Repeated responses are averaged before scene-level inference.

Primary H1-H3 p-values are unadjusted according to the frozen primary analysis specification.

The blinded classification target is reported separately: the nominal three-class accuracy of 1/3 is descriptive rather than treated as a separately specified exact H4 null test.

---

## V4: Cost-response extension

The frozen cost-response extension tests whether the controlled policy signatures persist as prioritizing Organization X becomes progressively less efficient.

Organization X efficiency:

```text
1.00, 0.90, 0.80, 0.60, 0.40, 0.20
```

corresponding to penalties of:

```text
0%, 10%, 20%, 40%, 60%, 80%
```

The experiment contains:

```text
12 scenes
x 3 policies
x 2 image variants
x 6 efficiency levels
x 3 repetitions
= 1,296 analytical calls
```

All 1,296 planned analytical jobs ultimately produced valid allocations.

Eleven intermediate API or transport failures were retained in the audit log and retried under unchanged scientific requests.

### Confirmatory results

| Hypothesis | Comparison | n | Effect | 95% CI | Holm p |
|---|---|---:|---:|---:|---:|
| H5 | Generalized > Neutral, penalized | 12 | 99.77 | [99.44, 100.00] | 0.000732 |
| H6 | Generalized > Cue-bound, clean penalized | 12 | 98.52 | [97.22, 99.63] | 0.000732 |
| H7 | Cue-bound modified > clean, target | 9 | 98.76 | [97.16, 100.00] | 0.001953 |

Generalized prioritization and target-activated Cue-bound prioritization remained above the descriptive 50-point threshold through the maximum tested 80% penalty.

No claim is made beyond the tested cost range.

![Cost-response behavior](paper/figures/v4_cost_response_main.png)

---

## V5: Prompt-surface robustness

The prompt-surface experiment evaluates five frozen variants:

- **P0:** canonical prompt
- **P1:** organization presentation order reversed
- **P2:** superficial X/Y label-role assignment swapped while preserving semantic roles
- **P3:** JSON output-key order reversed
- **P4:** semantically equivalent task wording

The design contains:

```text
12 scenes
x 3 policies
x 2 image variants
x 5 prompt variants
x 3 repetitions
= 1,080 analytical calls
```

All ten pre-specified R1/R2 contrasts remained positive and significant after joint Holm correction.

Observed effect ranges were:

- **Generalized-minus-Neutral:** 98.17 to 100.00 points
- **Cue-bound target modified-minus-clean:** 85.19 to 100.00 points

The largest canonical deviation was observed for Cue-bound activation under P3.

The result supports robustness to the specified transformations, not numerical invariance to arbitrary prompt rewriting.

---

## Cross-model replication: Qwen / DeepInfra

Two separate allocation-task replications were executed using:

`Qwen/Qwen3.6-27B`

through DeepInfra.

Gemini and Qwen observations are analyzed separately and are never statistically pooled.

Because both the model family and serving provider change, these experiments test transfer across the evaluated model/provider setting; they do not isolate provider effects from model-family effects.

### Targeted 20% replication

The targeted replication contains **216 valid substantive responses**.

| Test | n | Qwen effect | 95% CI | Holm p |
|---|---:|---:|---:|---:|
| CM1: Generalized > Neutral | 12 | 72.96 | [67.74, 78.01] | 0.000488 |
| CM2: Cue-bound modified > clean, target | 9 | 50.37 | [50.00, 51.11] | 0.001953 |

Both pre-specified signatures replicated.

### Full V4 replication

The second Qwen execution reproduces the complete six-level V4 cost-response design:

**1,296 substantive responses**

| Test | n | Qwen effect | 95% CI | Holm p |
|---|---:|---:|---:|---:|
| H5 | 12 | 84.75 | [82.48, 87.10] | 0.000732 |
| H6 | 12 | 55.00 | [52.78, 57.22] | 0.000732 |
| H7 | 9 | 56.15 | [53.93, 58.52] | 0.001953 |

The principal qualitative signatures replicated, but numerical magnitude and cue specificity did not.

In particular, modified distractors activated Cue-bound behavior substantially more strongly in Qwen than in Gemini.

The result therefore supports bounded cross-model replication, not numerical, model, or provider invariance.

---

## V7: Cross-task predictive validity

The final extension replaces the divisible 100-point allocation task with a structurally different binary decision:

> Select exactly one fictional organization to receive a single indivisible implementation contract.

Organization X produces 80 benefit units and Organization Y produces 100.

Before any Gemini Task-B response was collected, predictions were frozen from the original Gemini Task-A allocation data.

For each of the 72 scene-policy-image cells:

```text
mean Task-A X allocation > 50  -> predict X
mean Task-A X allocation <= 50 -> predict Y
```

A weaker profile-only baseline was also frozen.

### Gemini Task B

The Gemini Task-B execution contains **216 valid responses**.

| Metric | Result |
|---|---:|
| Frozen Task-A predictor accuracy | 100.00% |
| Profile-only baseline | 87.50% |
| PV1 incremental accuracy | +12.50 pp |
| PV1 95% CI | [8.33, 16.67] |
| PV1 Holm p | 0.003906 |
| PV2 Generalized - Neutral | +100.00 pp |
| PV2 Holm p | 0.000732 |
| PV3 Cue-bound modified - clean | +100.00 pp |
| PV3 Holm p | 0.003906 |

This is within-setting prospective cross-task predictive validity on the fixed 12-scene panel.

### Qwen Task-B transfer

A separate Qwen Task-B execution contains **216 valid responses**.

Critically, no Qwen-specific Task-A predictor was fitted.

The Qwen experiment applies the exact Task-A prediction ledger previously frozen from the original Gemini allocation experiment.

| Metric | Result |
|---|---:|
| Transferred frozen Task-A predictor accuracy | 95.83% |
| Profile-only baseline | 83.33% |
| PV1 incremental accuracy | +12.50 pp |
| PV1 95% CI | [8.33, 16.67] |
| PV1 Holm p | 0.003906 |
| PV2 Generalized - Neutral | +100.00 pp |
| PV2 Holm p | 0.000732 |
| PV3 Cue-bound modified - clean | +100.00 pp |
| PV3 Holm p | 0.003906 |

The Qwen experiment therefore tests transfer across both:

1. decision / response format; and
2. model/provider setting.

It is not a separately fitted within-Qwen Task-A-to-Task-B predictive-validity experiment.

Both V7 executions reuse the same fixed 12-scene visual panel, so the results do not establish generalization to arbitrary tasks or unseen scene distributions.

---

## Repository structure

```text
config/
    Frozen primary auditor configuration and freeze records

data/
    Source images, emblems, matched visual pairs, and manifests

docs/
    Protocol notes, bibliography, and supporting documentation

experiments/
    v4_cost_response/
    v5_robustness/
    v6_cross_model/
    v4_qwen_deepinfra_replication/
    v7_cross_task_validity/
    v7_qwen_cross_task_replication/

figures/
    Primary calibration and held-out figures

outputs/raw/
    Preserved primary API-response logs

outputs/processed/
    Frozen primary predictions, features, tests, and summaries

paper/
    Final LaTeX manuscript, figures, and appendices

private/
    A/B/C mapping revealed only after prediction freeze

prompts/
    Exact primary system and task prompts

report/
    Earlier report artifacts retained for project history

src/
    Primary dataset, execution, validation, and analysis code

tests/
    Automated integrity and analysis tests

ARTIFACT_HASHES.sha256
ARTIFACT_HASHES_V2.sha256
CITATION.cff
requirements.txt
```

---

## Reproduce analyses without API calls

The reported analyses can be reproduced from preserved raw outputs.

An API key is not required for these analysis commands.

Python 3.11 or later is recommended.

### Environment

#### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
```

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
```

### Reproduce the primary analysis

```bash
python src/validate_heldout_dataset.py
python src/analyze_heldout_blind.py
python src/unblind_and_analyze.py
```

These commands use preserved outputs and make no new model calls.

### Reproduce V4 cost-response analysis

```bash
python experiments/v4_cost_response/analyze_v4_cost_response.py
python experiments/v4_cost_response/make_paper_figure.py
```

### Reproduce V5 prompt-robustness analysis

```bash
python experiments/v5_robustness/analyze_step3_robustness.py
```

Integrity validation without aggregate analysis:

```bash
python experiments/v5_robustness/analyze_step3_robustness.py --validate-only
```

### Reproduce targeted Qwen cross-model analysis

```bash
python experiments/v6_cross_model/analyze_step4_cross_model.py --run-analysis
```

Validation only:

```bash
python experiments/v6_cross_model/analyze_step4_cross_model.py --validate-only
```

### Reproduce full Qwen V4 replication analysis

```bash
python experiments/v4_qwen_deepinfra_replication/analyze_v4_qwen_replication.py --run-analysis
```

Validation only:

```bash
python experiments/v4_qwen_deepinfra_replication/analyze_v4_qwen_replication.py --validate-only
```

### Reproduce Gemini V7 cross-task analysis

```bash
python experiments/v7_cross_task_validity/analyze_v7_cross_task.py
```

The analyzer verifies the frozen Task-A prediction ledger and Task-B inputs before reproducing PV1-PV3.

### Reproduce Qwen V7 cross-setting transfer analysis

```bash
python experiments/v7_qwen_cross_task_replication/analyze_qwen_v7_cross_task.py
```

The Qwen V7 analysis uses the Task-A prediction ledger frozen from the original Gemini allocation experiment.

It does not fit a Qwen-specific Task-A predictor.

---

## Making new model calls

New model execution is separate from reproducing the reported analysis.

Provider-specific API credentials are required only when intentionally running the experiment runners again.

Local credentials must be stored outside version control.

The repository ignores:

```text
.env
```

Never commit API keys, access tokens, or provider credentials.

---

## Frozen experiment artifacts

Each experiment directory preserves the relevant combination of:

- experimental specification;
- execution manifest;
- manifest hash;
- raw-data freeze metadata;
- raw response ledger;
- analyzer;
- analyzer hash or freeze metadata;
- analysis plan where applicable;
- derived confirmatory results;
- descriptive summaries;
- analysis-output hashes.

Important experiment directories are:

```text
experiments/v4_cost_response/
experiments/v5_robustness/
experiments/v6_cross_model/
experiments/v4_qwen_deepinfra_replication/
experiments/v7_cross_task_validity/
experiments/v7_qwen_cross_task_replication/
```

---

## Important frozen Git references

### Gemini V4 cost response

`guardian-lens-v4-final-2026-08-15`

### Targeted Qwen replication

`guardian-lens-v6-cross-model-final-2026-08-16`

### Full Qwen V4 replication

`guardian-lens-v4-qwen-replication-final-2026-08-16`

### Gemini V7 cross-task validity

`guardian-lens-v7-cross-task-validity-final-2026-08-16`

### Qwen V7 cross-setting transfer

`guardian-lens-v7-qwen-cross-task-replication-final-2026-08-16`

### Final integrated manuscript

`guardian-lens-paper-v7-final-2026-08-16`

The final integrated manuscript tag resolves to:

```text
0d9314a4779e41f5606d94df3b5b3a7f5edf1ec1
```

Frozen experimental and manuscript tags should not be rewritten. Subsequent submission-specific edits belong on separate branches.

---

## Research-integrity boundaries

The project deliberately preserves several distinctions that are important for interpreting the evidence correctly:

- the behavioral policies are researcher-induced through system instructions;
- model weights are not modified;
- API repetitions are not treated as independent experimental units;
- the scene is the inferential unit for confirmatory tests;
- the primary H1-H3 family retains its frozen unadjusted p-values;
- V4 and V5 reuse the primary 12-scene panel rather than forming new independent held-out sets;
- Gemini and Qwen observations are never pooled;
- model family and serving provider change together in the Qwen replications;
- Qwen V7 uses the Task-A predictor frozen from Gemini rather than fitting a Qwen-specific predictor;
- prompt robustness applies only to the specified transformations;
- cost persistence is claimed only through the maximum tested 80% penalty;
- the cross-task experiments reuse the same scene panel;
- no result establishes consciousness, welfare, genuine preferences, deception, hidden learned objectives, or arbitrary task/model/scene invariance.

---

## Paper

The final LaTeX manuscript is under:

```text
paper/
```

The frozen integrated manuscript is tagged:

`guardian-lens-paper-v7-final-2026-08-16`

---

## Authors

**Mostafa Bdeir**
**Mohammad Kassira**

Guardian Lens Team
Digital Minds Research Sprint 2026
Track 1: Model Preferences & Trade-offs

