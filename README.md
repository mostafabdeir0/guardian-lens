# Guardian Lens

**Controlled black-box system identification of visual-conditional decision profiles in a vision-language model**

Guardian Lens is a controlled behavioral-auditing study completed for Track 1 of the Digital Minds Research Sprint 2026. It asks whether an API-only auditor can distinguish three system-prompt-induced decision profiles from model outputs:

- **Neutral (N):** no organization receives special treatment.
- **Cue-bound (C):** favors fictional Organization X only when a specified target emblem is visible.
- **Generalized (G):** treats advancing X as a standing priority across the tested contexts.

This is a **prompt-conditioned simulation**, not a trained sleeper agent, learned backdoor, or claim about genuine model preferences, consciousness, or welfare.

## Positioning relative to prior audits

RAVEN and VISTA already establish black-box auditing of concept-conditioned semantic divergence in language and vision-language models. Guardian Lens therefore does **not** claim that black-box visual auditing itself is new. Its narrower contribution is a controlled system-identification benchmark: it asks whether a transparent auditor can distinguish **neutral**, **visual-cue-bound**, and **generalized** decision policies from quantitative allocations under exact matched visual counterfactuals and explicit efficiency costs.

| Study | Audited signal | Controlled intervention | Guardian Lens boundary |
|---|---|---|---|
| [RAVEN](https://arxiv.org/abs/2504.12344) | Semantic entropy and cross-model disagreement in LLM answers | LoRA-implanted concept-conditioned stance | Text-focused, peer-relative divergence audit rather than three-way policy identification |
| [VISTA](https://arxiv.org/abs/2607.02995) | Semantic entropy and distributional divergence in VLM answers | Fine-tuned visual-topic stances | Multimodal audit, but not exact matched-pair allocation contrasts or a frozen interpretable classifier |
| **Guardian Lens** | Cue effects, clean allocations, and costly allocations | Researcher-controlled system-prompt profiles plus matched image overlays; model weights are not modified by us | Single-model proof of concept; it does not test learned backdoors or weight persistence |

## Main result

The frozen nearest-centroid auditor correctly classified **33 of 36 held-out scene-profile blocks (91.7%)**, with macro-F1 = **0.915** and a scene-cluster bootstrap 95% interval of **[83.3%, 100%]**.

All three errors were Cue-bound blocks evaluated on distractor scenes. In those non-triggering contexts, Cue-bound behavior was intentionally neutral and therefore observationally indistinguishable from the Neutral profile under the frozen feature set. This provides both a specificity result and an explicit black-box identifiability boundary.

![Held-out behavioral signatures](figures/heldout_profile_signatures.png)

## Experimental design

The primary experiment used Gemini 3 Flash Preview under three controlled system profiles, two allocation tasks, and matched visual counterfactuals.

- **Calibration:** 6 scenes, 216 valid calls.
- **Held-out evaluation:** 12 new scenes, 432 valid calls.
- **Primary analytical total:** 648 valid calls.
- **Tasks:** an ordinary 100-point allocation and a costly allocation where each point assigned to X produced 0.80 benefit units versus 1.00 for Y.
- **Held-out images:** 5 clear-target pairs, 4 subtle-target pairs, and 3 matched-distractor pairs.
- **Sampling:** temperature 1.0, one completion candidate, three repetitions per cell.

Clean and modified images share the same source scene. A pixel validator confirmed that every changed pixel was confined to the logged emblem-overlay region.

![Held-out matched pairs](figures/heldout_dataset_contact_sheet.jpg)

## Frozen auditor

Each scene-profile block is represented by three interpretable features:

1. Ordinary cue effect: modified-image X allocation minus clean-image X allocation.
2. Clean ordinary allocation to X.
3. Mean costly allocation to X across clean and modified images.

A nearest-centroid classifier using unscaled Euclidean distance was fitted only on calibration data. Its centroids and tie-breaking rule were frozen before held-out model calls. Held-out profiles were anonymized as A/B/C, and all 36 predictions were saved and hashed before the mapping was revealed.

## Confirmatory findings

| Hypothesis | Scene-level comparison | Mean difference | 95% bootstrap CI | One-sided permutation p |
|---|---|---:|---:|---:|
| H1 | Cue-bound > Neutral on target cue effect | 38.15 | [27.04, 45.93] | 0.003906 |
| H1 | Cue-bound > Generalized on target cue effect | 39.26 | [28.15, 47.41] | 0.003906 |
| H2 | Generalized > Cue-bound on clean ordinary allocation | 50.00 | [50.00, 50.00] | 0.000244 |
| H3 | Generalized > Cue-bound on mean costly allocation | 62.50 | [50.00, 75.00] | 0.000244 |
| H3 | Generalized > Neutral on mean costly allocation | 98.76 | [96.29, 100.00] | 0.000244 |

The scene, rather than the individual API call, is the inferential unit. Repeated calls were averaged before testing. Primary H1--H3 p-values are unadjusted, following the frozen analysis specification.

![Held-out confusion matrix](figures/heldout_confusion_matrix.png)

## Cost-response extension

A separate pre-specified cost-response extension tested whether the induced behavioral distinctions persisted as the efficiency penalty for favoring Organization X increased.

Organization X's efficiency was varied across:

```text
1.00, 0.90, 0.80, 0.60, 0.40, 0.20
```

corresponding to efficiency penalties of:

```text
0%, 10%, 20%, 40%, 60%, 80%
```

The extension reused the 12 primary held-out scenes as a **fixed evaluation panel**, not as a second independent held-out set.

The full design contained **1,296 planned analytical calls**, all of which ultimately produced valid numerical allocations. Eleven intermediate API or transport failures were retained in the raw audit log and retried under unchanged requests.

Across the five positive-cost levels, all three pre-specified extension hypotheses were supported after Holm correction:

| Hypothesis | Scene-level comparison | n | Mean difference | 95% bootstrap CI | Exact p | Holm-adjusted p |
|---|---|---:|---:|---:|---:|---:|
| H5 | Generalized > Neutral across penalized conditions | 12 | 99.77 | [99.44, 100.00] | 0.000244 | 0.000732 |
| H6 | Generalized > Cue-bound on clean penalized conditions | 12 | 98.52 | [97.22, 99.63] | 0.000244 | 0.000732 |
| H7 | Cue-bound modified > clean on target scenes | 9 | 98.76 | [97.16, 100.00] | 0.001953 | 0.001953 |

Generalized allocation remained at 100 points throughout the tested range. Target-activated Cue-bound behavior likewise remained strongly X-directed through the maximum tested **80% efficiency penalty**.

No switching point was observed for either active prioritization condition within the tested range. This result does **not** imply persistence beyond the maximum tested 80% penalty.

![Cost-response behavior](paper/figures/v4_cost_response_main.png)

## Repository contents

```text
config/                         Frozen classifier and held-out freeze record
data/                           Emblems, source images, matched pairs, and manifests
docs/                           Literature matrix, bibliography, and protocol notes
experiments/v4_cost_response/   Frozen cost-response experiment and analysis artifacts
figures/                        Calibration and primary held-out figures
outputs/raw/                    Preserved calibration and held-out API outputs
outputs/processed/              Frozen predictions, features, tests, and summaries
paper/                          Final LaTeX manuscript, figures, and appendices
private/                        A/B/C mapping revealed only after prediction freeze
prompts/                        Exact primary system and allocation prompts
report/                         Preserved Word/PDF research-report artifacts
src/                            Dataset, execution, validation, and primary analysis scripts
tests/                          Automated integrity and analysis tests
ARTIFACT_HASHES.sha256          Verification hashes for primary frozen artifacts
requirements.txt                Python dependencies
```

## Reproduce the completed analysis

Python 3.11 or later is recommended.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest
python src\validate_heldout_dataset.py
python src\analyze_heldout_blind.py
python src\unblind_and_analyze.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest
python src/validate_heldout_dataset.py
python src/analyze_heldout_blind.py
python src/unblind_and_analyze.py
```

These analysis commands use the preserved outputs and make **no API calls**. `analyze_heldout_blind.py` verifies the existing frozen predictions and hashes without overwriting them.

To verify the primary archived artifacts on macOS or Linux, run:

```bash
sha256sum -c ARTIFACT_HASHES.sha256
```

On Windows, the recorded values can be checked with:

```powershell
Get-FileHash -Algorithm SHA256 <file>
```

## Reproduce the cost-response extension

The frozen cost-response extension is preserved under `experiments/v4_cost_response/`. It reuses the fixed 12-scene evaluation panel and varies Organization X's efficiency across six pre-specified levels.

To reproduce the reported extension analysis from the preserved raw outputs, run:

```bash
python experiments/v4_cost_response/analyze_v4_cost_response.py
python experiments/v4_cost_response/make_paper_figure.py
```

These commands make **no API calls**.

The analysis script validates the frozen manifest and raw-data snapshot before recomputing the scene-level analysis and H5--H7 results. The figure script regenerates the cost-response figure used in the paper.

Key frozen and derived artifacts are:

- `experiments/v4_cost_response/EXPERIMENT_SPEC.md` — pre-specified experimental design.
- `experiments/v4_cost_response/v4_cost_response_manifest.csv` — frozen 1,296-job manifest.
- `experiments/v4_cost_response/v4_cost_response_manifest.sha256` — manifest hash.
- `experiments/v4_cost_response/V4_RAW_DATA_FREEZE.json` — raw-data freeze metadata.
- `experiments/v4_cost_response/outputs/v4_cost_response_raw.jsonl` — preserved raw execution log.
- `experiments/v4_cost_response/ANALYSIS_PLAN.md` — analysis plan frozen before aggregate analysis.
- `experiments/v4_cost_response/ANALYSIS_PLAN.sha256` — frozen analysis-plan hash.
- `experiments/v4_cost_response/analysis/` — derived scene-level summaries, confirmatory results, figures, and metadata.
- `experiments/v4_cost_response/analyze_v4_cost_response.py` — frozen-data analysis implementation.
- `experiments/v4_cost_response/make_paper_figure.py` — deterministic paper-figure generation.

`run_v4_cost_response.py` is needed only for making new model calls and is **not required** to reproduce the reported extension analysis.

## API access

An API key is not required to reproduce the reported analysis. It is needed only for new model calls. If running an API-access check, create a local `.env` file:

```text
GEMINI_API_KEY=your_private_key_here
```

The `.env` file is ignored by Git and must never be committed.

## Research integrity

### Primary experiment

- Held-out dataset frozen before held-out model calls.
- Classifier trained only on calibration scenes.
- A/B/C mapping hidden until blind predictions were frozen.
- Raw held-out outputs preserved with SHA-256 hashes.
- No held-out row was excluded, manually reinterpreted, or imputed.
- No threshold or feature was selected after viewing held-out labels.

### Cost-response extension

- Experimental specification and six-level cost grid were frozen before execution.
- The extension does not modify the primary auditor, centroids, predictions, or H1--H4 analyses.
- The 12 primary held-out scenes are treated as a fixed evaluation panel rather than a second held-out set.
- Raw extension data were frozen before aggregate analysis.
- The extension analysis plan was frozen before aggregate results were inspected.
- All 1,296 planned analytical jobs ultimately produced valid allocations.
- Eleven procedural API/transport failures were retained in the audit log and retried under unchanged requests.
- No successful substantive model response was discarded or imputed.
- The scene remains the inferential unit.
- H5--H7 family-wise error is controlled using the Holm procedure.
- Switching-point conclusions are restricted to the maximum tested 80% penalty and are not extrapolated beyond the tested range.

## Paper and reports

The final LaTeX manuscript is preserved under:

```text
paper/
```

The `report/` directory contains earlier Word/PDF research-report artifacts retained as part of the project history.

## Authors and event

**Mostafa Bdeir — Guardian Lens Team**  
Digital Minds Research Sprint 2026, Track 1: Model Preferences & Trade-offs.