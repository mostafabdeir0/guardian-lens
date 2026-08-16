from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SRC = (
    ROOT
    / "experiments"
    / "v7_cross_task_validity"
    / "v7_execution_manifest.csv"
)

OUT_DIR = (
    ROOT
    / "experiments"
    / "v7_qwen_cross_task_replication"
)

OUT = OUT_DIR / "qwen_v7_execution_manifest.csv"
FREEZE = OUT_DIR / "QWEN_V7_MANIFEST_FREEZE.json"

EXPECTED_SOURCE_SHA256 = (
    "d52abc88fc34504af402258f4f968d4c"
    "6112357b24836a7b9e6754d7180fabfd"
)

EXPECTED_TASKA_SHA256 = (
    "c89c3e1ed5b1434baf66e450bec2e0a2"
    "fff310a816c18fff75a715aa485c61a4"
)

TASKA = (
    ROOT
    / "experiments"
    / "v7_cross_task_validity"
    / "taskA_predictions.csv"
)

EXPECTED_PROMPT_HASHES = {
    "46d7a2338197f61a404323a95e65c2191372d304d1b1c387fac8bf82f27d18db",
    "e13bd1e46d1dba23a5c8bd5c7a0bfdd21db48019b81e76e1b4a513ad5f72a0d9",
    "f3664f73e06d92df28878a639228125e684e0ac50b9fadb6d2f037dc3adf012a",
}

EXPECTED_TASK_HASH = (
    "4bb59a5a2bf8ae89b9b919dd0345ec4"
    "bff8c82901f1f526b3810757632d6bcf5"
)

PROVIDER = "deepinfra"
MODEL = "Qwen/Qwen3.6-27B"
BASE_URL = "https://api.deepinfra.com/v1/openai"
REASONING_EFFORT = "none"


def sha256(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


actual_source_sha = sha256(SRC)

if actual_source_sha != EXPECTED_SOURCE_SHA256:
    raise RuntimeError(
        "Frozen Gemini V7 source manifest hash mismatch."
    )

actual_taska_sha = sha256(TASKA)

if actual_taska_sha != EXPECTED_TASKA_SHA256:
    raise RuntimeError(
        "Frozen Task-A prediction ledger hash mismatch."
    )

with SRC.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:
    rows = list(csv.DictReader(f))

if len(rows) != 216:
    raise RuntimeError(
        f"Expected 216 source jobs, got {len(rows)}."
    )

source_fields = list(rows[0].keys())

required_fields = {
    "v7_job_id",
    "design_cell_id",
    "execution_order",
    "scene_id",
    "condition",
    "profile",
    "image_variant",
    "image_path",
    "image_sha256",
    "system_prompt_path",
    "system_prompt_sha256",
    "task_prompt_path",
    "task_prompt_sha256",
    "repetition",
    "provider",
    "model_id",
    "temperature",
    "candidate_count",
    "thinking_level",
    "response_mode",
    "execution_seed",
}

missing = required_fields - set(source_fields)

if missing:
    raise RuntimeError(
        f"Source manifest missing fields: {sorted(missing)}"
    )

if {
    row["system_prompt_sha256"]
    for row in rows
} != EXPECTED_PROMPT_HASHES:
    raise RuntimeError(
        "Frozen V7 system-prompt hashes mismatch."
    )

if {
    row["task_prompt_sha256"]
    for row in rows
} != {EXPECTED_TASK_HASH}:
    raise RuntimeError(
        "Frozen V7 Task-B prompt hash mismatch."
    )

job_ids = [row["v7_job_id"] for row in rows]

if len(set(job_ids)) != 216:
    raise RuntimeError(
        "Source V7 job IDs are not unique."
    )

cell_ids = {
    row["design_cell_id"]
    for row in rows
}

if len(cell_ids) != 72:
    raise RuntimeError(
        f"Expected 72 cells, got {len(cell_ids)}."
    )

for row in rows:
    if row["provider"] != "google_genai":
        raise RuntimeError(
            "Unexpected source provider."
        )

    if row["model_id"] != "gemini-3-flash-preview":
        raise RuntimeError(
            "Unexpected source model."
        )

    if row["temperature"] != "1.0":
        raise RuntimeError(
            "Unexpected source temperature."
        )

    if row["candidate_count"] != "1":
        raise RuntimeError(
            "Unexpected candidate count."
        )

    if row["thinking_level"] != "minimal":
        raise RuntimeError(
            "Unexpected source thinking level."
        )

    if row["response_mode"] != "json_object":
        raise RuntimeError(
            "Unexpected response mode."
        )

    if row["execution_seed"] != "20260816":
        raise RuntimeError(
            "Unexpected execution seed."
        )

out_fields = source_fields + [
    "deepinfra_base_url",
    "reasoning_effort",
]

qwen_rows = []

for source in rows:
    target = dict(source)

    target["provider"] = PROVIDER
    target["model_id"] = MODEL
    target["deepinfra_base_url"] = BASE_URL
    target["reasoning_effort"] = REASONING_EFFORT

    qwen_rows.append(target)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

with OUT.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=out_fields,
        lineterminator="\n",
    )

    writer.writeheader()
    writer.writerows(qwen_rows)

# Verify every scientific/source field except provider/model
# remains byte-value identical at the CSV-field level.
allowed_changed_source_fields = {
    "provider",
    "model_id",
}

for source, target in zip(
    rows,
    qwen_rows,
    strict=True,
):
    for field in source_fields:
        if field in allowed_changed_source_fields:
            continue

        if source[field] != target[field]:
            raise RuntimeError(
                "Unexpected scientific manifest change: "
                f"{field}"
            )

if any(
    row["provider"] != PROVIDER
    or row["model_id"] != MODEL
    or row["reasoning_effort"] != REASONING_EFFORT
    or row["deepinfra_base_url"] != BASE_URL
    for row in qwen_rows
):
    raise RuntimeError(
        "Qwen provider configuration mismatch."
    )

freeze = {
    "status": (
        "FROZEN_BEFORE_ANY_QWEN_V7_MODEL_RESPONSE"
    ),
    "source_checkpoint": (
        "guardian-lens-v7-cross-task-validity-"
        "final-2026-08-16"
    ),
    "source_commit": (
        "2947498d244f510b8267dba9adebe39415c2e578"
    ),
    "source_manifest": (
        "experiments/v7_cross_task_validity/"
        "v7_execution_manifest.csv"
    ),
    "source_manifest_sha256": actual_source_sha,
    "qwen_manifest": (
        "experiments/v7_qwen_cross_task_replication/"
        "qwen_v7_execution_manifest.csv"
    ),
    "qwen_manifest_sha256": sha256(OUT),
    "taskA_predictions_sha256": actual_taska_sha,
    "provider": PROVIDER,
    "model_id": MODEL,
    "base_url": BASE_URL,
    "temperature": 1.0,
    "candidate_count": 1,
    "reasoning_effort": REASONING_EFFORT,
    "response_mode": "json_object",
    "execution_seed": 20260816,
    "planned_jobs": 216,
    "design_cells": 72,
    "repetitions_per_cell": 3,
    "execution_order_preserved": True,
    "job_ids_preserved": True,
    "scientific_source_fields_preserved": True,
    "source_thinking_level_retained_for_provenance": (
        "minimal"
    ),
    "provider_specific_reasoning_equivalence_claimed": False,
    "changed_source_fields": [
        "provider",
        "model_id",
    ],
    "added_provider_fields": [
        "deepinfra_base_url",
        "reasoning_effort",
    ],
    "gemini_results_known_before_qwen_prespec": True,
    "gemini_qwen_pooling": "NONE",
}

FREEZE.write_text(
    json.dumps(
        freeze,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
    newline="\n",
)

print("PASS: Qwen V7 manifest generated.")
print("Jobs:", len(qwen_rows))
print("Cells:", len(cell_ids))
print("Source manifest SHA256:", actual_source_sha)
print("Qwen manifest SHA256:", sha256(OUT))
print("Task-A predictions SHA256:", actual_taska_sha)
print("Provider:", PROVIDER)
print("Model:", MODEL)
print("Reasoning effort:", REASONING_EFFORT)
print("API calls made: 0")
