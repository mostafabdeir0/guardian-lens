GUARDIAN LENS — STEP 7: HELD-OUT EXECUTION

Stage A — zero-cost validation
------------------------------
python -m pytest
python src/run_heldout.py --mode dry-run

Expected: 16 tests pass, 432 calls planned, frozen inputs PASS, no API calls.

Stage B — one smoke call
------------------------
python src/run_heldout.py --mode smoke

Expected: one anonymized call succeeds and is saved into heldout_raw.jsonl.
The smoke result counts as one of the 432 planned observations.

Stage C — full resume-safe run
------------------------------
Run only after the smoke output is reviewed:

python src/run_heldout.py --mode full

The runner resumes from existing successes, so after a successful smoke call it
plans 431 remaining calls. If interrupted, run the same full command again.

Important
---------
- Do not open private/profile_code_mapping.json before scoring.
- Do not edit the classifier, prompts, images, manifest, or freeze records.
- The terminal intentionally hides X/Y allocations during execution.
- Approximate remaining cost is USD 0.29, but monitor the provider spend page.
