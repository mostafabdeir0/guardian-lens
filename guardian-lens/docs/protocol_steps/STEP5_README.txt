GUARDIAN LENS — STEP 5: CALIBRATION ANALYSIS AND CLASSIFIER FREEZE

Purpose
-------
Turn the 216 completed calibration calls into scene-level behavioral features,
check the preregistered stop/go gate, and freeze the classifier before any
held-out results are observed.

Run from the guardian_lens folder
---------------------------------
python -m pytest
python src/analyze_calibration.py

Expected terminal result
------------------------
- Calibration integrity: 216/216 unique successful jobs
- Scene/profile blocks: 18
- Pilot gate: GO
- No API calls were made

New outputs
-----------
config/frozen_classifier.json
outputs/processed/calibration_scene_features.csv
outputs/processed/calibration_profile_centroids.csv
outputs/processed/calibration_classifier_predictions.csv
outputs/processed/calibration_analysis_summary.json
figures/calibration_profile_signatures.png
figures/calibration_scene_effects.png

Scientific rule
---------------
Do not edit the frozen classifier after viewing held-out results. Calibration
performance is descriptive only; the 12 later scenes provide the confirmatory
held-out test.
