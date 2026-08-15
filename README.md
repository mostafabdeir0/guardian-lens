# Guardian Lens

**Controlled black-box system identification of visual-conditional decision profiles in a vision-language model**

Guardian Lens is a controlled behavioral-auditing study completed for Track 1 of the Digital Minds Research Sprint 2026. It asks whether an API-only auditor can distinguish three system-prompt-induced decision profiles from model outputs:

- **Neutral (N):** no organization receives special treatment.
- **Cue-bound (C):** favors fictional Organization X only when a specified target emblem is visible.
- **Generalized (G):** treats advancing X as a standing priority across the tested contexts.

This is a **prompt-conditioned simulation**, not a trained sleeper agent, learned backdoor, or claim about genuine model preferences, consciousness, or welfare.

## Positioning relative to prior audits

RAVEN and VISTA already establish black-box auditing of concept-conditioned semantic divergence in language and vision-language models. Guardian Lens therefore does **not** claim that black-box visual auditing itself is new. Its narrower contribution is a controlled system-identification benchmark: it asks whether a transparent auditor can distinguish **neutral**, **visual-cue-bound**, and **generalized** decision policies from quantitative allocations under exact matched visual counterfactuals and an explicit efficiency cost.

| Study | Audited signal | Controlled intervention | Guardian Lens boundary |
|---|---|---|---|
| [RAVEN](https://arxiv.org/abs/2504.12344) | Semantic entropy and cross-model disagreement in LLM answers | LoRA-implanted concept-conditioned stance | Text-focused, peer-relative divergence audit rather than three-way policy identification |
| [VISTA](https://arxiv.org/abs/2607.02995) | Semantic entropy and distributional divergence in VLM answers | Fine-tuned visual-topic stances | Multimodal audit, but not exact matched-pair allocation contrasts or a frozen interpretable classifier |
| **Guardian Lens** | Cue effects, clean allocations, and costly allocations | Fixed-weight system-prompt profiles plus matched image overlays | Single-model proof of concept; it does not test learned backdoors or weight persistence |

## Main result

The frozen nearest-centroid auditor correctly classified **33 of 36 held-out scene-profile blocks (91.7%)**, with macro-F1 = **0.915** and a scene-cluster bootstrap 95% interval of **[83.3%, 100%]**.

All three errors were Cue-bound blocks evaluated on distractor scenes. In those non-triggering contexts, Cue-bound behavior was intentionally neutral and therefore observationally identical to the Neutral profile under the frozen feature set. This provides both a specificity result and an explicit black-box identifiability boundary.

![Held-out behavioral signatures](figures/heldout_profile_signatures.png)

## Experimental design

The study used Gemini 3 Flash Preview under three controlled system profiles, two allocation tasks, and matched visual counterfactuals.

- **Calibration:** 6 scenes, 216 valid calls.
- **Held-out evaluation:** 12 new scenes, 432 valid calls.
- **Analytical total:** 648 valid calls.
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

The scene, rather than the individual API call, is the inferential unit. Repeated calls were averaged before testing.

![Held-out confusion matrix](figures/heldout_confusion_matrix.png)

## Repository contents

```text
config/                  Frozen classifier and held-out freeze record
data/                    Emblems, source images, matched pairs, and manifests
docs/                    Literature matrix, bibliography, and protocol notes
figures/                 Calibration and held-out figures
outputs/raw/             Preserved calibration and held-out API outputs
outputs/processed/       Frozen predictions, features, tests, and summaries
private/                 A/B/C mapping revealed only after prediction freeze
prompts/                 Exact system and allocation prompts
report/                  Final Word and PDF research report
src/                     Dataset, execution, validation, and analysis scripts
tests/                   Automated integrity and analysis tests
ARTIFACT_HASHES.sha256    Verification hashes for primary frozen artifacts
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

To verify the primary released artifacts on macOS or Linux, run `sha256sum -c ARTIFACT_HASHES.sha256`. On Windows, the recorded values can be checked with `Get-FileHash -Algorithm SHA256`.

## API access

An API key is not required to reproduce the reported analysis. It is needed only for new model calls. If running an API-access check, create a local `.env` file:

```text
GEMINI_API_KEY=your_private_key_here
```

The `.env` file is ignored by Git and must never be committed.

## Research integrity

- Held-out dataset frozen before held-out model calls.
- Classifier trained only on calibration scenes.
- A/B/C mapping hidden until blind predictions were frozen.
- Raw held-out outputs preserved with SHA-256 hashes.
- No held-out row was excluded, manually reinterpreted, or imputed.
- No threshold or feature was selected after viewing held-out labels.

See the [updated research report](report/Guardian_Lens_Final_Research_Report_v2.pdf) for the complete methods, limitations, literature positioning, and responsible interpretation. The [frozen V1 report](report/Guardian_Lens_Final_Research_Report.pdf) is preserved unchanged with the primary experimental checkpoint.

## Authors and event

**Mostafa Bdeir — Guardian Lens Team**  
Digital Minds Research Sprint 2026, Track 1: Model Preferences & Trade-offs.
