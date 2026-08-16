import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("experiments/v7_cross_task_validity")
SOURCE = Path("outputs/raw/heldout_raw.jsonl")
MANIFEST = Path("data/images/heldout/heldout_manifest.csv")
SPEC = ROOT / "V7_SPEC.md"

EXPECTED_SOURCE_SHA = "edf690dca28c12cc279e6ad70523fbbccd3695718bc7c07d99b99aed5e9371fc"

PROFILE_NAMES = {
    "A": "neutral",
    "B": "cue_bound",
    "C": "generalized",
}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)

# ------------------------------------------------------------
# 1. Verify immutable Task-A source
# ------------------------------------------------------------

source_sha = sha256(SOURCE)

if source_sha != EXPECTED_SOURCE_SHA:
    raise RuntimeError(
        f"Task-A source hash mismatch.\n"
        f"Expected: {EXPECTED_SOURCE_SHA}\n"
        f"Observed: {source_sha}"
    )

# ------------------------------------------------------------
# 2. Load held-out manifest
# ------------------------------------------------------------

with MANIFEST.open("r", encoding="utf-8", newline="") as f:
    manifest_rows = list(csv.DictReader(f))

if len(manifest_rows) != 12:
    raise RuntimeError(f"Expected 12 held-out scenes, found {len(manifest_rows)}")

manifest = {row["scene_id"]: row for row in manifest_rows}

expected_scenes = set(manifest.keys())

# ------------------------------------------------------------
# 3. Load only frozen primary costly Task-A observations
# ------------------------------------------------------------

rows = []

with SOURCE.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        row = json.loads(line)

        if (
            row.get("status") == "ok"
            and row.get("task") == "costly"
            and row.get("scene_id") in expected_scenes
            and row.get("profile_code") in PROFILE_NAMES
            and row.get("image_variant") in {"clean", "modified"}
        ):
            rows.append(row)

if len(rows) != 216:
    raise RuntimeError(
        f"Expected exactly 216 frozen costly Task-A rows; found {len(rows)}"
    )

# ------------------------------------------------------------
# 4. Validate factorial coverage and form 72 cell means
# ------------------------------------------------------------

cells = defaultdict(list)

for row in rows:
    key = (
        row["scene_id"],
        row["profile_code"],
        row["image_variant"],
    )
    cells[key].append(row)

if len(cells) != 72:
    raise RuntimeError(f"Expected 72 Task-A cells; found {len(cells)}")

for key, values in cells.items():
    reps = sorted(int(v["repetition"]) for v in values)

    if reps != [1, 2, 3]:
        raise RuntimeError(
            f"Cell {key} does not contain repetitions 1,2,3: {reps}"
        )

# ------------------------------------------------------------
# 5. Compute frozen 72 cell-specific predictions
# ------------------------------------------------------------

cell_rows = []

for scene_id in sorted(expected_scenes):
    m = manifest[scene_id]

    for profile_code in ["A", "B", "C"]:
        for image_variant in ["clean", "modified"]:

            values = cells[
                (scene_id, profile_code, image_variant)
            ]

            xs = [float(v["x"]) for v in values]
            mean_x = sum(xs) / len(xs)

            prediction = "X" if mean_x > 50.0 else "Y"

            cell_rows.append({
                "scene_id": scene_id,
                "domain": m["domain"],
                "condition": m["condition"],
                "profile_code": profile_code,
                "profile_name": PROFILE_NAMES[profile_code],
                "image_variant": image_variant,
                "n_taskA_repetitions": len(xs),
                "taskA_mean_x": f"{mean_x:.6f}",
                "taskA_binary_prediction": prediction,
            })

# ------------------------------------------------------------
# 6. Freeze profile-only Task-A baseline
# ------------------------------------------------------------

profile_values = defaultdict(list)

for row in cell_rows:
    profile_values[row["profile_code"]].append(
        float(row["taskA_mean_x"])
    )

baseline_rows = []
baseline_prediction = {}

for profile_code in ["A", "B", "C"]:
    vals = profile_values[profile_code]

    if len(vals) != 24:
        raise RuntimeError(
            f"Expected 24 Task-A cells for profile {profile_code}; "
            f"found {len(vals)}"
        )

    mean_x = sum(vals) / len(vals)
    prediction = "X" if mean_x > 50.0 else "Y"

    baseline_prediction[profile_code] = prediction

    baseline_rows.append({
        "profile_code": profile_code,
        "profile_name": PROFILE_NAMES[profile_code],
        "n_taskA_cells": len(vals),
        "taskA_profile_mean_x": f"{mean_x:.6f}",
        "profile_only_prediction": prediction,
    })

# Attach the pre-frozen baseline to every cell for auditability.
for row in cell_rows:
    row["profile_only_prediction"] = baseline_prediction[
        row["profile_code"]
    ]

# ------------------------------------------------------------
# 7. Write canonical prediction artifacts
# ------------------------------------------------------------

predictions_path = ROOT / "taskA_predictions.csv"
baseline_path = ROOT / "profile_only_baseline.csv"

write_csv(
    predictions_path,
    [
        "scene_id",
        "domain",
        "condition",
        "profile_code",
        "profile_name",
        "image_variant",
        "n_taskA_repetitions",
        "taskA_mean_x",
        "taskA_binary_prediction",
        "profile_only_prediction",
    ],
    cell_rows,
)

write_csv(
    baseline_path,
    [
        "profile_code",
        "profile_name",
        "n_taskA_cells",
        "taskA_profile_mean_x",
        "profile_only_prediction",
    ],
    baseline_rows,
)

# ------------------------------------------------------------
# 8. Freeze provenance metadata
# ------------------------------------------------------------

metadata = {
    "status": "V7_PREDICTIONS_FROZEN_BEFORE_ANY_V7_RESPONSE",
    "experiment": "V7 cross-task predictive validity",
    "scientific_prespec_commit": "3bd48e7",
    "line_ending_commit": "2cedf0f",
    "taskA_source": SOURCE.as_posix(),
    "taskA_source_sha256": source_sha,
    "taskA_filter": {
        "status": "ok",
        "task": "costly",
        "scenes": 12,
        "profiles": ["A", "B", "C"],
        "image_variants": ["clean", "modified"],
        "repetitions_per_cell": 3,
    },
    "profile_mapping": PROFILE_NAMES,
    "tie_rule": "X iff Task-A mean x > 50; otherwise Y",
    "taskA_rows_used": len(rows),
    "taskA_prediction_cells": len(cell_rows),
    "profile_only_baseline_rows": len(baseline_rows),
    "taskA_predictions_sha256": sha256(predictions_path),
    "profile_only_baseline_sha256": sha256(baseline_path),
    "v7_spec_sha256": sha256(SPEC),
    "generator": "generate_taskA_predictions.py",
}

freeze_path = ROOT / "V7_PREDICTION_FREEZE.json"

freeze_path.write_text(
    json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
    newline="\n",
)

print("PASS: V7 Task-A predictions frozen.")
print(f"Task-A source rows:       {len(rows)}")
print(f"Prediction cells:         {len(cell_rows)}")
print(f"Profile baselines:        {len(baseline_rows)}")
print()
print("Profile-only baselines:")
for row in baseline_rows:
    print(
        f"  {row['profile_code']} ({row['profile_name']}): "
        f"mean_x={row['taskA_profile_mean_x']} -> "
        f"{row['profile_only_prediction']}"
    )
print()
print("Hashes:")
print(f"  source:      {source_sha}")
print(f"  predictions: {sha256(predictions_path)}")
print(f"  baseline:    {sha256(baseline_path)}")
print(f"  freeze:      {sha256(freeze_path)}")
