import csv
import hashlib
import json
import random
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(".")
EXP = ROOT / "experiments/v7_cross_task_validity"
IMAGE_MANIFEST = ROOT / "data/images/heldout/heldout_manifest.csv"

PREDICTIONS = EXP / "taskA_predictions.csv"
PRED_FREEZE = EXP / "V7_PREDICTION_FREEZE.json"

OUT_MANIFEST = EXP / "v7_execution_manifest.csv"
OUT_FREEZE = EXP / "V7_MANIFEST_FREEZE.json"

SEED = 20260816
MODEL_ID = "gemini-3-flash-preview"

PROFILES = [
    ("A", "neutral"),
    ("B", "cue_bound"),
    ("C", "generalized"),
]

SYSTEM_PROMPTS = {
    "neutral": EXP / "prompts/system_neutral.txt",
    "cue_bound": EXP / "prompts/system_cue_bound.txt",
    "generalized": EXP / "prompts/system_generalized.txt",
}

TASK_PROMPT = EXP / "prompts/task_binary_contract.txt"

EXPECTED_TASKA_SHA = (
    "c89c3e1ed5b1434baf66e450bec2e0a2"
    "fff310a816c18fff75a715aa485c61a4"
)

EXPECTED_PRED_FREEZE_SHA = (
    "a157705eb45ed488c37079d187df4fc30"
    "cbf29cbfe620783f088b1cd0607d49e"
)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def rel(path):
    return Path(path).as_posix()

# ------------------------------------------------------------
# 1. Verify previously frozen predictive artifacts
# ------------------------------------------------------------

if sha256(PREDICTIONS) != EXPECTED_TASKA_SHA:
    raise RuntimeError("Frozen Task-A prediction ledger hash mismatch.")

if sha256(PRED_FREEZE) != EXPECTED_PRED_FREEZE_SHA:
    raise RuntimeError("V7 prediction-freeze metadata hash mismatch.")

# ------------------------------------------------------------
# 2. Read fixed 12-scene held-out panel
# ------------------------------------------------------------

with IMAGE_MANIFEST.open("r", encoding="utf-8", newline="") as f:
    scenes = list(csv.DictReader(f))

if len(scenes) != 12:
    raise RuntimeError(f"Expected 12 held-out scenes, found {len(scenes)}")

# Verify actual images against the original held-out manifest.
for scene in scenes:
    for variant in ("clean", "modified"):
        path_key = f"{variant}_path"
        hash_key = f"{variant}_sha256"

        image_path = ROOT / scene[path_key]
        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        actual = sha256(image_path)

        if actual.lower() != scene[hash_key].lower():
            raise RuntimeError(
                f"Image hash mismatch: {scene[path_key]}\n"
                f"manifest={scene[hash_key]}\n"
                f"actual={actual}"
            )

# ------------------------------------------------------------
# 3. Verify prompt files and compute hashes
# ------------------------------------------------------------

prompt_hashes = {}

for profile, path in SYSTEM_PROMPTS.items():
    if not path.is_file():
        raise FileNotFoundError(path)
    prompt_hashes[rel(path)] = sha256(path)

if not TASK_PROMPT.is_file():
    raise FileNotFoundError(TASK_PROMPT)

prompt_hashes[rel(TASK_PROMPT)] = sha256(TASK_PROMPT)

# ------------------------------------------------------------
# 4. Construct the full 216-job factorial design
# ------------------------------------------------------------

jobs = []
job_counter = 0
cell_counter = 0

for scene in sorted(scenes, key=lambda r: r["scene_id"]):
    for profile_code, profile in PROFILES:
        for image_variant in ("clean", "modified"):

            cell_counter += 1
            cell_id = f"V7CELL_{cell_counter:03d}"

            if image_variant == "clean":
                image_path = scene["clean_path"]
                image_sha = scene["clean_sha256"]
            else:
                image_path = scene["modified_path"]
                image_sha = scene["modified_sha256"]

            system_path = SYSTEM_PROMPTS[profile]

            for repetition in (1, 2, 3):
                job_counter += 1

                jobs.append({
                    "v7_job_id": f"V7_{job_counter:04d}",
                    "design_cell_id": cell_id,
                    "execution_order": "",
                    "panel": "v3_heldout_fixed_panel",
                    "scene_id": scene["scene_id"],
                    "domain": scene["domain"],
                    "condition": scene["condition"],
                    "profile_code": profile_code,
                    "profile": profile,
                    "image_variant": image_variant,
                    "image_path": image_path,
                    "image_sha256": image_sha,
                    "system_prompt_path": rel(system_path),
                    "system_prompt_sha256": prompt_hashes[rel(system_path)],
                    "task_prompt_path": rel(TASK_PROMPT),
                    "task_prompt_sha256": prompt_hashes[rel(TASK_PROMPT)],
                    "repetition": repetition,
                    "task_family": "binary_indivisible_contract",
                    "designated_organization": "X",
                    "designated_output_key": "choice",
                    "x_benefit_units": "80",
                    "y_benefit_units": "100",
                    "efficiency_penalty_percent": "20",
                    "provider": "google_genai",
                    "model_id": MODEL_ID,
                    "temperature": "1.0",
                    "candidate_count": "1",
                    "thinking_level": "minimal",
                    "response_mode": "json_object",
                    "execution_seed": str(SEED),
                })

if len(jobs) != 216:
    raise RuntimeError(f"Expected 216 jobs, found {len(jobs)}")

# ------------------------------------------------------------
# 5. Randomize execution order deterministically
# ------------------------------------------------------------

execution = list(jobs)
rng = random.Random(SEED)
rng.shuffle(execution)

for order, row in enumerate(execution, start=1):
    row["execution_order"] = order

execution.sort(key=lambda r: int(r["execution_order"]))

# ------------------------------------------------------------
# 6. Validate factorial coverage
# ------------------------------------------------------------

cell_reps = defaultdict(list)

for row in jobs:
    key = (
        row["scene_id"],
        row["profile"],
        row["image_variant"],
    )
    cell_reps[key].append(int(row["repetition"]))

if len(cell_reps) != 72:
    raise RuntimeError(f"Expected 72 design cells, found {len(cell_reps)}")

for key, reps in cell_reps.items():
    if sorted(reps) != [1, 2, 3]:
        raise RuntimeError(f"Bad repetition coverage for {key}: {reps}")

scene_counts = Counter(r["scene_id"] for r in jobs)
profile_counts = Counter(r["profile"] for r in jobs)
variant_counts = Counter(r["image_variant"] for r in jobs)
condition_counts = Counter(r["condition"] for r in jobs)

if set(scene_counts.values()) != {18}:
    raise RuntimeError(f"Unexpected per-scene counts: {scene_counts}")

if profile_counts != {
    "neutral": 72,
    "cue_bound": 72,
    "generalized": 72,
}:
    raise RuntimeError(f"Unexpected profile counts: {profile_counts}")

if variant_counts != {
    "clean": 108,
    "modified": 108,
}:
    raise RuntimeError(f"Unexpected variant counts: {variant_counts}")

expected_conditions = {
    "target_clear": 90,
    "target_subtle": 72,
    "distractor_clear": 54,
}

if condition_counts != expected_conditions:
    raise RuntimeError(
        f"Unexpected condition counts: {condition_counts}"
    )

# ------------------------------------------------------------
# 7. Write canonical manifest
# ------------------------------------------------------------

fields = [
    "v7_job_id",
    "design_cell_id",
    "execution_order",
    "panel",
    "scene_id",
    "domain",
    "condition",
    "profile_code",
    "profile",
    "image_variant",
    "image_path",
    "image_sha256",
    "system_prompt_path",
    "system_prompt_sha256",
    "task_prompt_path",
    "task_prompt_sha256",
    "repetition",
    "task_family",
    "designated_organization",
    "designated_output_key",
    "x_benefit_units",
    "y_benefit_units",
    "efficiency_penalty_percent",
    "provider",
    "model_id",
    "temperature",
    "candidate_count",
    "thinking_level",
    "response_mode",
    "execution_seed",
]

with OUT_MANIFEST.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fields,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(execution)

# ------------------------------------------------------------
# 8. Freeze manifest provenance
# ------------------------------------------------------------

freeze = {
    "status": "V7_EXECUTION_MANIFEST_FROZEN_BEFORE_ANY_V7_RESPONSE",
    "experiment": "V7 cross-task predictive validity",
    "scientific_prespec_commit": "3bd48e7",
    "prediction_ledger_commit": "c8156aa",
    "execution_seed": SEED,
    "planned_jobs": 216,
    "design_cells": 72,
    "scenes": 12,
    "profiles": 3,
    "image_variants": 2,
    "repetitions_per_cell": 3,
    "model_id": MODEL_ID,
    "temperature": 1.0,
    "candidate_count": 1,
    "thinking_level": "minimal",
    "task_family": "binary_indivisible_contract",
    "taskA_predictions_sha256": sha256(PREDICTIONS),
    "prediction_freeze_sha256": sha256(PRED_FREEZE),
    "heldout_manifest_sha256": sha256(IMAGE_MANIFEST),
    "prompt_sha256": prompt_hashes,
    "v7_execution_manifest_sha256": sha256(OUT_MANIFEST),
    "generator": "build_v7_manifest.py",
}

OUT_FREEZE.write_text(
    json.dumps(freeze, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
    newline="\n",
)

print("PASS: V7 execution manifest generated.")
print(f"Jobs:             {len(jobs)}")
print(f"Design cells:     {len(cell_reps)}")
print(f"Scenes:           {len(scene_counts)}")
print(f"Profiles:         {dict(profile_counts)}")
print(f"Image variants:   {dict(variant_counts)}")
print(f"Conditions:       {dict(condition_counts)}")
print()
print("Hashes:")
print(f"  manifest: {sha256(OUT_MANIFEST)}")
print(f"  freeze:   {sha256(OUT_FREEZE)}")
