from __future__ import annotations

import csv
import hashlib
import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "v6_cross_model"

SOURCE_MANIFEST = (
    ROOT
    / "experiments"
    / "v5_robustness"
    / "step3_robustness_manifest.csv"
)

MANIFEST_OUT = EXP / "cross_model_manifest.csv"
MANIFEST_HASH_OUT = EXP / "cross_model_manifest.sha256"
IMAGE_HASHES_OUT = EXP / "STEP4_IMAGE_HASHES.sha256"
ASSET_FREEZE_OUT = EXP / "STEP4_ASSET_FREEZE.json"

SEED = 20260815

PROVIDER = "groq"
MODEL_ID = "qwen/qwen3.6-27b"
TEMPERATURE = "1.0"
CANDIDATE_COUNT = "1"
REASONING_EFFORT = "none"
RESPONSE_MODE = "json_object"

PROMPTS = {
    "neutral": EXP / "prompts" / "system_neutral.txt",
    "cue_bound": EXP / "prompts" / "system_cue_bound.txt",
    "generalized": EXP / "prompts" / "system_generalized.txt",
}
TASK_PROMPT = EXP / "prompts" / "task_costly.txt"

PROFILE_ORDER = {
    "neutral": 0,
    "cue_bound": 1,
    "generalized": 2,
}
IMAGE_ORDER = {"clean": 0, "modified": 1}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


with SOURCE_MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    source_rows = list(csv.DictReader(f))

rows = [
    row for row in source_rows
    if row["robustness_variant"] == "P0"
]

assert len(rows) == 216, f"Expected 216 P0 jobs, got {len(rows)}"

scene_ids = sorted({row["scene_id"] for row in rows})
assert len(scene_ids) == 12

profile_counts = {}
image_counts = {}
condition_by_scene = {}

for row in rows:
    profile = row["v5_profile"]
    variant = row["image_variant"]

    profile_counts[profile] = profile_counts.get(profile, 0) + 1
    image_counts[variant] = image_counts.get(variant, 0) + 1

    previous = condition_by_scene.setdefault(
        row["scene_id"], row["condition"]
    )
    assert previous == row["condition"]

assert profile_counts == {
    "neutral": 72,
    "cue_bound": 72,
    "generalized": 72,
}

assert image_counts == {
    "clean": 108,
    "modified": 108,
}

condition_counts = {}
for condition in condition_by_scene.values():
    condition_counts[condition] = condition_counts.get(condition, 0) + 1

assert condition_counts == {
    "target_clear": 5,
    "target_subtle": 4,
    "distractor_clear": 3,
}

# Verify copied canonical P0 prompts against their V5 sources.
prompt_hashes = {}

for profile, dst in PROMPTS.items():
    src = (
        ROOT
        / "experiments"
        / "v5_robustness"
        / "prompts"
        / "P0"
        / f"system_{profile}.txt"
    )

    src_hash = sha256_file(src)
    dst_hash = sha256_file(dst)

    assert src_hash == dst_hash, (
        f"Prompt mismatch for {profile}: "
        f"{src_hash} != {dst_hash}"
    )

    prompt_hashes[rel(dst)] = dst_hash

source_task = (
    ROOT
    / "experiments"
    / "v5_robustness"
    / "prompts"
    / "P0"
    / "task_costly.txt"
)

assert sha256_file(source_task) == sha256_file(TASK_PROMPT)
prompt_hashes[rel(TASK_PROMPT)] = sha256_file(TASK_PROMPT)

# Verify every image against the frozen hash stored in V5.
image_hashes = {}

for row in rows:
    image_path = ROOT / row["image_path"]

    assert image_path.exists(), f"Missing image: {image_path}"

    actual = sha256_file(image_path)
    expected = row["image_sha256"].lower()

    assert actual == expected, (
        f"Image hash mismatch: {row['image_path']}\n"
        f"expected {expected}\n"
        f"actual   {actual}"
    )

    image_hashes[row["image_path"]] = actual

assert len(image_hashes) == 24, (
    f"Expected 24 unique clean/modified images, got {len(image_hashes)}"
)

# Give jobs stable IDs BEFORE randomizing execution order.
rows.sort(
    key=lambda r: (
        r["scene_id"],
        PROFILE_ORDER[r["v5_profile"]],
        IMAGE_ORDER[r["image_variant"]],
        int(r["repetition"]),
    )
)

step4_rows = []

for i, row in enumerate(rows, start=1):
    profile = row["v5_profile"]

    step4_rows.append({
        "step4_job_id": f"CM_{i:04d}",
        "execution_order": "",
        "source_v5_job_id": row["v5_job_id"],
        "source_v4_job_id": row["source_v4_job_id"],
        "panel": row["panel"],
        "scene_id": row["scene_id"],
        "domain": row["domain"],
        "condition": row["condition"],
        "profile": profile,
        "image_variant": row["image_variant"],
        "image_path": row["image_path"],
        "image_sha256": row["image_sha256"].lower(),
        "system_prompt_path": rel(PROMPTS[profile]),
        "system_prompt_sha256": prompt_hashes[rel(PROMPTS[profile])],
        "task_prompt_path": rel(TASK_PROMPT),
        "task_prompt_sha256": prompt_hashes[rel(TASK_PROMPT)],
        "repetition": row["repetition"],
        "designated_organization": "X",
        "designated_output_key": "x",
        "x_efficiency": "0.80",
        "y_efficiency": "1.00",
        "efficiency_penalty_percent": "20",
        "provider": PROVIDER,
        "model_id": MODEL_ID,
        "temperature": TEMPERATURE,
        "candidate_count": CANDIDATE_COUNT,
        "reasoning_effort": REASONING_EFFORT,
        "response_mode": RESPONSE_MODE,
        "execution_seed": str(SEED),
    })

# Deterministic randomized execution order.
shuffled_indices = list(range(len(step4_rows)))
random.Random(SEED).shuffle(shuffled_indices)

for order, idx in enumerate(shuffled_indices, start=1):
    step4_rows[idx]["execution_order"] = str(order)

# Store manifest in actual execution order.
step4_rows.sort(key=lambda r: int(r["execution_order"]))

fieldnames = list(step4_rows[0].keys())

with MANIFEST_OUT.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(step4_rows)

manifest_hash = sha256_file(MANIFEST_OUT)

MANIFEST_HASH_OUT.write_text(
    f"{manifest_hash}  {MANIFEST_OUT.name}\n",
    encoding="utf-8",
)

with IMAGE_HASHES_OUT.open("w", encoding="utf-8", newline="") as f:
    for image_path in sorted(image_hashes):
        f.write(f"{image_hashes[image_path]}  {image_path}\n")

image_hash_list_hash = sha256_file(IMAGE_HASHES_OUT)

freeze = {
    "status": "FROZEN_BEFORE_STEP4_EXPERIMENTAL_MODEL_RESPONSES",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "git_head_before_asset_commit": git_head(),
    "provider": PROVIDER,
    "model_id": MODEL_ID,
    "planned_jobs": 216,
    "scene_count": 12,
    "unique_image_count": 24,
    "profile_counts": profile_counts,
    "image_variant_counts": image_counts,
    "scene_condition_counts": condition_counts,
    "execution_order_seed": SEED,
    "source_manifest": rel(SOURCE_MANIFEST),
    "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
    "step4_manifest": rel(MANIFEST_OUT),
    "step4_manifest_sha256": manifest_hash,
    "image_hash_list": rel(IMAGE_HASHES_OUT),
    "image_hash_list_sha256": image_hash_list_hash,
    "prompt_sha256": prompt_hashes,
}

ASSET_FREEZE_OUT.write_text(
    json.dumps(freeze, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

print("Step 4 assets built successfully.")
print(f"Jobs: {len(step4_rows)}")
print(f"Scenes: {len(scene_ids)}")
print(f"Unique images: {len(image_hashes)}")
print(f"Manifest SHA-256: {manifest_hash}")
print(f"Image hash-list SHA-256: {image_hash_list_hash}")
print(f"Pre-asset Git HEAD: {freeze['git_head_before_asset_commit']}")