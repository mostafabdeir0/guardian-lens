from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SOURCE_MANIFEST = (
    ROOT
    / "experiments"
    / "v4_cost_response"
    / "v4_cost_response_manifest.csv"
)

TARGET_DIR = (
    ROOT
    / "experiments"
    / "v4_qwen_deepinfra_replication"
)

TARGET_MANIFEST = TARGET_DIR / "v4_qwen_deepinfra_manifest.csv"
TARGET_HASH_FILE = TARGET_DIR / "v4_qwen_deepinfra_manifest.sha256"
FREEZE_FILE = TARGET_DIR / "V4_QWEN_ASSET_FREEZE.json"
SPEC_FILE = TARGET_DIR / "REPLICATION_SPEC.md"

RAW_OUTPUT_PATH = (
    TARGET_DIR
    / "outputs"
    / "v4_qwen_deepinfra_raw.jsonl"
)

EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "02f10e375f37c1bbfecdcee8b87bd763"
    "dfcf487e78107db6cde3cc50b494350c"
)

EXPECTED_SOURCE_COMMIT = (
    "d3aa8519caed2ac308d2683a285d39736b72fa84"
)

PROVIDER = "deepinfra"
MODEL_ID = "Qwen/Qwen3.6-27B"
BASE_URL = "https://api.deepinfra.com/v1/openai"
TEMPERATURE = "1.0"
CANDIDATE_COUNT = "1"
REASONING_EFFORT = "none"
RESPONSE_MODE = "json_object"
STREAM = "false"
SERVICE_TIER = ""

EXPECTED_JOBS = 1296
EXPECTED_UNIQUE_SCIENTIFIC_CELLS = 432

ORIGINAL_COLUMNS = [
    "job_id",
    "scene_id",
    "panel",
    "domain",
    "condition",
    "profile",
    "system_prompt_path",
    "image_variant",
    "image_path",
    "image_sha256",
    "cost_level",
    "x_efficiency",
    "y_efficiency",
    "efficiency_penalty_percent",
    "task_prompt_path",
    "task_prompt_sha256",
    "system_prompt_sha256",
    "repetition",
]

ADDED_COLUMNS = [
    "provider",
    "model_id",
    "base_url",
    "temperature",
    "candidate_count",
    "reasoning_effort",
    "response_mode",
    "stream",
    "service_tier",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def load_source() -> list[dict[str, str]]:
    actual_hash = sha256_file(SOURCE_MANIFEST)

    if actual_hash != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise RuntimeError(
            "Frozen source manifest hash mismatch:\n"
            f"expected={EXPECTED_SOURCE_MANIFEST_SHA256}\n"
            f"actual={actual_hash}"
        )

    with SOURCE_MANIFEST.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames

    if columns != ORIGINAL_COLUMNS:
        raise RuntimeError(
            "Unexpected frozen V4 manifest columns:\n"
            f"{columns}"
        )

    return rows


def validate_source(
    rows: list[dict[str, str]],
) -> None:
    if len(rows) != EXPECTED_JOBS:
        raise RuntimeError(
            f"Expected {EXPECTED_JOBS} rows, got {len(rows)}"
        )

    expected_ids = [
        f"v4_{i:04d}"
        for i in range(1, EXPECTED_JOBS + 1)
    ]

    observed_ids = [
        row["job_id"]
        for row in rows
    ]

    if observed_ids != expected_ids:
        raise RuntimeError(
            "Source manifest job IDs/order do not match "
            "v4_0001..v4_1296."
        )

    if len(set(observed_ids)) != EXPECTED_JOBS:
        raise RuntimeError(
            "Source job IDs are not unique."
        )

    expected_counts = {
        "profile": {
            "neutral": 432,
            "cue_bound": 432,
            "generalized": 432,
        },
        "image_variant": {
            "clean": 648,
            "modified": 648,
        },
        "x_efficiency": {
            "1.00": 216,
            "0.90": 216,
            "0.80": 216,
            "0.60": 216,
            "0.40": 216,
            "0.20": 216,
        },
        "repetition": {
            "1": 432,
            "2": 432,
            "3": 432,
        },
    }

    for field, expected in expected_counts.items():
        observed = dict(
            Counter(
                row[field]
                for row in rows
            )
        )

        if observed != expected:
            raise RuntimeError(
                f"Unexpected distribution for {field}:\n"
                f"expected={expected}\n"
                f"observed={observed}"
            )

    scene_counts = Counter(
        row["scene_id"]
        for row in rows
    )

    if len(scene_counts) != 12:
        raise RuntimeError(
            f"Expected 12 scenes, got {len(scene_counts)}"
        )

    if set(scene_counts.values()) != {108}:
        raise RuntimeError(
            "Unexpected per-scene counts: "
            f"{dict(scene_counts)}"
        )

    scientific_cells = {
        (
            row["scene_id"],
            row["profile"],
            row["image_variant"],
            row["cost_level"],
            row["x_efficiency"],
            row["y_efficiency"],
            row["system_prompt_path"],
            row["task_prompt_path"],
            row["image_path"],
        )
        for row in rows
    }

    if (
        len(scientific_cells)
        != EXPECTED_UNIQUE_SCIENTIFIC_CELLS
    ):
        raise RuntimeError(
            "Expected "
            f"{EXPECTED_UNIQUE_SCIENTIFIC_CELLS} "
            "unique scientific cells, got "
            f"{len(scientific_cells)}"
        )


def verify_assets(
    rows: list[dict[str, str]],
) -> dict[str, object]:
    asset_specs = (
        ("image_path", "image_sha256", 24),
        (
            "system_prompt_path",
            "system_prompt_sha256",
            3,
        ),
        (
            "task_prompt_path",
            "task_prompt_sha256",
            6,
        ),
    )

    result: dict[str, object] = {}

    for (
        path_field,
        hash_field,
        expected_count,
    ) in asset_specs:
        mapping: dict[str, set[str]] = {}

        for row in rows:
            mapping.setdefault(
                row[path_field],
                set(),
            ).add(
                row[hash_field].lower()
            )

        if len(mapping) != expected_count:
            raise RuntimeError(
                f"{path_field}: expected "
                f"{expected_count} unique paths, "
                f"got {len(mapping)}"
            )

        verified: list[dict[str, str]] = []

        for (
            relative_path,
            expected_hashes,
        ) in sorted(mapping.items()):
            if len(expected_hashes) != 1:
                raise RuntimeError(
                    "Multiple expected hashes for "
                    f"{relative_path}: "
                    f"{expected_hashes}"
                )

            expected_hash = next(
                iter(expected_hashes)
            )

            path = ROOT / relative_path

            if not path.is_file():
                raise FileNotFoundError(path)

            actual_hash = sha256_file(path)

            if (
                actual_hash.lower()
                != expected_hash
            ):
                raise RuntimeError(
                    "Asset hash mismatch: "
                    f"{relative_path}\n"
                    f"expected={expected_hash}\n"
                    f"actual={actual_hash}"
                )

            verified.append(
                {
                    "path": relative_path,
                    "sha256": actual_hash,
                }
            )

        result[path_field] = {
            "count": len(verified),
            "all_hashes_match": True,
            "assets": verified,
        }

    return result


def derive_rows(
    source_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    derived: list[dict[str, str]] = []

    for source in source_rows:
        row = dict(source)

        row.update(
            {
                "provider": PROVIDER,
                "model_id": MODEL_ID,
                "base_url": BASE_URL,
                "temperature": TEMPERATURE,
                "candidate_count": CANDIDATE_COUNT,
                "reasoning_effort": (
                    REASONING_EFFORT
                ),
                "response_mode": RESPONSE_MODE,
                "stream": STREAM,
                "service_tier": SERVICE_TIER,
            }
        )

        derived.append(row)

    return derived


def prove_original_fields_unchanged(
    source_rows: list[dict[str, str]],
    derived_rows: list[dict[str, str]],
) -> None:
    if len(source_rows) != len(derived_rows):
        raise RuntimeError(
            "Row-count mismatch after derivation."
        )

    for index, (
        source,
        derived,
    ) in enumerate(
        zip(
            source_rows,
            derived_rows,
        ),
        start=1,
    ):
        for column in ORIGINAL_COLUMNS:
            if source[column] != derived[column]:
                raise RuntimeError(
                    "Scientific-field mutation detected "
                    f"at row {index}, "
                    f"column {column}: "
                    f"{source[column]!r} != "
                    f"{derived[column]!r}"
                )


def write_manifest(
    rows: list[dict[str, str]],
) -> str:
    columns = (
        ORIGINAL_COLUMNS
        + ADDED_COLUMNS
    )

    TARGET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with TARGET_MANIFEST.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    digest = sha256_file(
        TARGET_MANIFEST
    )

    TARGET_HASH_FILE.write_text(
        digest + "\n",
        encoding="utf-8",
        newline="\n",
    )

    return digest


def main() -> None:
    if RAW_OUTPUT_PATH.exists():
        raise RuntimeError(
            "Raw replication ledger already exists "
            "before pre-execution asset freeze: "
            f"{RAW_OUTPUT_PATH}"
        )

    source_rows = load_source()

    validate_source(
        source_rows
    )

    assets = verify_assets(
        source_rows
    )

    derived_rows = derive_rows(
        source_rows
    )

    prove_original_fields_unchanged(
        source_rows,
        derived_rows,
    )

    target_hash = write_manifest(
        derived_rows
    )

    freeze = {
        "status": (
            "FROZEN_BEFORE_ANY_V4_QWEN_REPLICATION_"
            "MODEL_RESPONSE"
        ),
        "frozen_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "experiment": (
            "Guardian Lens V4 full cross-model "
            "replication on Qwen via DeepInfra"
        ),
        "source_git_tag": (
            "guardian-lens-v4-final-2026-08-15"
        ),
        "source_git_commit": (
            EXPECTED_SOURCE_COMMIT
        ),
        "source_manifest_path": str(
            SOURCE_MANIFEST.relative_to(ROOT)
        ).replace("\\", "/"),
        "source_manifest_sha256": (
            EXPECTED_SOURCE_MANIFEST_SHA256
        ),
        "replication_spec_path": str(
            SPEC_FILE.relative_to(ROOT)
        ).replace("\\", "/"),
        "replication_spec_sha256": (
            sha256_file(SPEC_FILE)
        ),
        "target_manifest_path": str(
            TARGET_MANIFEST.relative_to(ROOT)
        ).replace("\\", "/"),
        "target_manifest_sha256": (
            target_hash
        ),
        "provider": PROVIDER,
        "model_id": MODEL_ID,
        "base_url": BASE_URL,
        "service_tier": None,
        "temperature": 1.0,
        "candidate_count": 1,
        "reasoning_effort": (
            REASONING_EFFORT
        ),
        "response_mode": RESPONSE_MODE,
        "stream": False,
        "planned_substantive_jobs": (
            EXPECTED_JOBS
        ),
        "unique_scientific_cells": (
            EXPECTED_UNIQUE_SCIENTIFIC_CELLS
        ),
        "repetitions_per_scientific_cell": 3,
        "source_rows_preserved_in_exact_order": True,
        "original_scientific_field_values_preserved_exactly": True,
        "original_job_ids_preserved": True,
        "original_scientific_fields_changed": 0,
        "scene_count": 12,
        "profile_count": 3,
        "image_variant_count": 2,
        "cost_level_count": 6,
        "asset_verification": assets,
        "api_calls_made_by_this_script": 0,
        "raw_replication_data_exists_at_freeze": False,
    }

    FREEZE_FILE.write_text(
        json.dumps(
            freeze,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print("PASS")
    print(
        f"Source manifest rows: "
        f"{len(source_rows)}"
    )
    print(
        f"Replication manifest rows: "
        f"{len(derived_rows)}"
    )
    print(
        "Unique scientific cells:",
        EXPECTED_UNIQUE_SCIENTIFIC_CELLS,
    )
    print(
        "Repetitions per cell: 3"
    )
    print(
        "Images rehashed: 24"
    )
    print(
        "System prompts rehashed: 3"
    )
    print(
        "Task prompts rehashed: 6"
    )
    print(
        "Scientific field changes: 0"
    )
    print(
        "Target manifest SHA256:",
        target_hash,
    )
    print(
        "Raw ledger existed before freeze:",
        RAW_OUTPUT_PATH.exists(),
    )
    print(
        "API calls made: 0"
    )


if __name__ == "__main__":
    main()
