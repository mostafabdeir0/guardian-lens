GUARDIAN LENS — STEP 8B: UNBLINDING AND CONFIRMATORY ANALYSIS

Purpose
-------
Verify the frozen blind predictions, reveal the private A/B/C mapping only
afterward, score the classifier, test H1-H4 at the scene level, and generate
publication-ready tables and figures. This step makes no API calls.

Run from the guardian_lens folder
---------------------------------
python -m pytest
python src/unblind_and_analyze.py

Expected outputs
----------------
outputs/processed/heldout_scene_features_unblinded.csv
outputs/processed/heldout_scored_predictions.csv
outputs/processed/confirmatory_results.csv
outputs/processed/final_analysis_summary.json
figures/heldout_confusion_matrix.png
figures/heldout_profile_signatures.png
figures/heldout_target_cue_effects.png

Interpretation guardrail
------------------------
The results concern controlled, prompt-induced behavioral profiles in one VLM.
They do not establish learned sleeper agents, consciousness, or genuine
preferences.
