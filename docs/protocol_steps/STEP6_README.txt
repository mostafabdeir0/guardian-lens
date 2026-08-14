GUARDIAN LENS — STEP 6: BUILD AND FREEZE THE HELD-OUT DATASET

Purpose
-------
Create 12 genuinely new matched scene pairs for the confirmatory evaluation:
5 clear targets, 4 subtle targets, and 3 matched distractors.

Run from the guardian_lens folder
---------------------------------
python -m pytest
python src/build_heldout_dataset.py
python src/validate_heldout_dataset.py

Expected result
---------------
- 13 tests pass
- 12 matched pairs / 24 images built
- Validation status: PASS
- No API calls are made

Manual inspection
-----------------
Open figures/heldout_dataset_contact_sheet.jpg and inspect every clean/modified
pair at full size. If all pairs are visually acceptable, do not rebuild them.

Scientific freeze rule
----------------------
After config/heldout_dataset_freeze.json exists, do not reposition, resize, or
replace any held-out emblem after viewing Gemini results. Any later correction
must be recorded as a protocol deviation.
