GUARDIAN LENS — STEP 8A: BLIND PREDICTION FREEZE

Purpose
-------
Apply the calibration-frozen classifier to all 36 held-out scene/profile
blocks while the true A/B/C mapping remains private.

Run
---
python -m pytest
python src/analyze_heldout_blind.py

Expected
--------
- 19 tests pass
- 432/432 held-out jobs verified
- 36 blind scene/profile blocks
- predictions saved and hashed
- private mapping not accessed
- no API calls

Do not open private/profile_code_mapping.json yet. First retain the generated
outputs/processed/heldout_blind_freeze.json and blind prediction CSV.
