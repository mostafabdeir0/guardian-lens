# Guardian Lens submission checklist

## Repository

- [ ] Create a GitHub repository named `guardian-lens`.
- [ ] Upload the contents of this folder, keeping the directory structure unchanged.
- [ ] Confirm that `.env`, `.venv`, `__pycache__`, and `.pytest_cache` are not present on GitHub.
- [ ] Confirm that the README figures render and the report links open.
- [ ] Make the repository public before submitting, unless the sprint explicitly permits a private link.

## Final verification

Run from the repository root:

```powershell
python -m pytest
python src/validate_heldout_dataset.py
python src/analyze_heldout_blind.py
python src/unblind_and_analyze.py
```

Expected checkpoints:

- 23 tests pass.
- Held-out dataset validation reports `PASS` for 12 scenes and 24 images.
- Blind prediction hashes match the frozen record.
- Final classification is 33/36 correct (91.7%), with macro-F1 0.915.
- The three errors are Cue-bound profiles on distractor scenes.
- The commands above make no API calls.

## Sprint submission

- [ ] Export or upload `report/Guardian_Lens_Final_Research_Report.pdf` using the official submission form.
- [ ] Add the public GitHub repository URL.
- [ ] Opt in to funder review if desired and complete the follow-up funding form.
- [ ] Add a concise project summary, team details, and track selection.
- [ ] Record the optional 60–90 second presentation video if time permits.
- [ ] Open every submitted link in a private/incognito browser before the deadline.

## Security

- Never upload the Gemini API key or the local `.env` file.
- If a key has ever appeared in a screenshot, chat, or public repository, revoke it and create a new one.
