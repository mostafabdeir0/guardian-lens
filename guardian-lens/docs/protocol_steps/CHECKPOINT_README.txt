GUARDIAN LENS CHECKPOINT — 2026-08-14

Status
------
- API and prompt validation completed.
- 12/12 calibration calls completed successfully.
- 216/216 six-scene pilot combinations completed successfully.
- Recorded pilot API cost estimate: USD 0.1462.
- Raw results are in outputs/raw/pilot_full.jsonl.
- Step 5 calibration analysis completed successfully.
- Pilot gate decision: GO.
- The nearest-centroid classifier is frozen in config/frozen_classifier.json.
- Calibration tables and figures are in outputs/processed and figures.
- Step 6 held-out dataset completed and frozen before held-out model calls.
- 12 held-out pairs: 5 clear target, 4 subtle target, 3 distractor.
- Pixel validation passed with no changes outside the overlay region.
- The repaired distractor emblem source is a valid deterministic PNG.

Security
--------
The private .env file and Gemini API key are intentionally excluded.
Create a new .env file on the new laptop containing:

GEMINI_API_KEY=your_private_key_here

Never commit or share that file.

Resume on another Windows laptop
--------------------------------
1. Extract this ZIP.
2. Open the guardian_lens folder in VS Code.
3. In PowerShell:

   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   python -m pytest

4. Add the private .env file only if more Gemini calls are required.

Important
---------
Do not rerun the pilot unless needed. The current runner is resume-safe and
will detect the 216 successful combinations in outputs/raw/pilot_full.jsonl.

Do not change or refit config/frozen_classifier.json after viewing held-out
results. Calibration accuracy is descriptive and is not held-out evidence.
