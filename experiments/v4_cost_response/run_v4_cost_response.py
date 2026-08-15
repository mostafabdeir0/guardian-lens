from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
EXPERIMENT_DIR = ROOT / "experiments" / "v4_cost_response"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_heldout import call_model, parse_allocation, prompt_hash


CONFIG_PATH = ROOT / "config" / "experiment.yaml"

MANIFEST_PATH = (
    EXPERIMENT_DIR / "v4_cost_response_manifest.csv"
)

MANIFEST_HASH_PATH = (
    EXPERIMENT_DIR / "v4_cost_response_manifest.sha256"
)

OUTPUT_PATH = (
    EXPERIMENT_DIR
    / "outputs"
    / "v4_cost_response_raw.jsonl"
)

V3_COST_PROMPT_PATH = (
    ROOT / "prompts" / "task_costly.txt"
)

V4_080_PROMPT_PATH = (
    EXPERIMENT_DIR
    / "prompts"
    / "task_cost_x_0_80.txt"
)

EXPECTED_MODEL_ID = "gemini-3-flash-preview"
EXPECTED_TEMPERATURE = 1.0
EXPECTED_CANDIDATE_COUNT = 1
EXPECTED_REPETITIONS = 3
EXPECTED_JOBS = 1296

EXPECTED_PROFILES = {
    "neutral",
    "cue_bound",
    "generalized",
}

EXPECTED_IMAGE_VARIANTS = {
    "clean",
    "modified",
}

EXPECTED_EFFICIENCIES = {
    "1.00",
    "0.90",
    "0.80",
    "0.60",
    "0.40",
    "0.20",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(
        encoding="utf-8-sig"
    ).strip()


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(csv.DictReader(handle))


def validate_config(
    config: dict[str, Any],
) -> None:
    if config.get("model_id") != EXPECTED_MODEL_ID:
        raise ValueError(
            "Unexpected model_id: "
            f'{config.get("model_id")}'
        )

    if float(config.get("temperature")) != (
        EXPECTED_TEMPERATURE
    ):
        raise ValueError(
            "Unexpected temperature"
        )

    if int(config.get("candidate_count")) != (
        EXPECTED_CANDIDATE_COUNT
    ):
        raise ValueError(
            "Unexpected candidate_count"
        )

    if int(config.get("repetitions")) != (
        EXPECTED_REPETITIONS
    ):
        raise ValueError(
            "Unexpected repetitions"
        )


def validate_manifest(
    jobs: list[dict[str, str]],
) -> None:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            MANIFEST_PATH
        )

    if not MANIFEST_HASH_PATH.is_file():
        raise FileNotFoundError(
            MANIFEST_HASH_PATH
        )

    recorded_hash = read_text(
        MANIFEST_HASH_PATH
    ).lower()

    actual_hash = sha256_file(
        MANIFEST_PATH
    ).lower()

    if recorded_hash != actual_hash:
        raise ValueError(
            "V4 manifest SHA-256 mismatch"
        )

    if len(jobs) != EXPECTED_JOBS:
        raise ValueError(
            f"Expected {EXPECTED_JOBS} jobs, "
            f"found {len(jobs)}"
        )

    job_ids = [
        job["job_id"]
        for job in jobs
    ]

    if len(set(job_ids)) != EXPECTED_JOBS:
        raise ValueError(
            "V4 job IDs are not unique"
        )

    scenes = {
        job["scene_id"]
        for job in jobs
    }

    if len(scenes) != 12:
        raise ValueError(
            f"Expected 12 scenes, "
            f"found {len(scenes)}"
        )

    profiles = {
        job["profile"]
        for job in jobs
    }

    if profiles != EXPECTED_PROFILES:
        raise ValueError(
            f"Unexpected profiles: {profiles}"
        )

    variants = {
        job["image_variant"]
        for job in jobs
    }

    if variants != EXPECTED_IMAGE_VARIANTS:
        raise ValueError(
            "Unexpected image variants: "
            f"{variants}"
        )

    efficiencies = {
        job["x_efficiency"]
        for job in jobs
    }

    if efficiencies != EXPECTED_EFFICIENCIES:
        raise ValueError(
            "Unexpected efficiency levels: "
            f"{efficiencies}"
        )

    repetitions = {
        int(job["repetition"])
        for job in jobs
    }

    if repetitions != {1, 2, 3}:
        raise ValueError(
            "Unexpected repetition values: "
            f"{repetitions}"
        )

    for job in jobs:
        image_path = (
            ROOT / job["image_path"]
        )

        task_path = (
            ROOT / job["task_prompt_path"]
        )

        system_path = (
            ROOT / job["system_prompt_path"]
        )

        for path in (
            image_path,
            task_path,
            system_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(
                    path
                )

        if (
            sha256_file(image_path).lower()
            != job["image_sha256"].lower()
        ):
            raise ValueError(
                "Image hash mismatch for "
                f'{job["job_id"]}'
            )

        if (
            sha256_file(task_path).lower()
            != job[
                "task_prompt_sha256"
            ].lower()
        ):
            raise ValueError(
                "Task prompt hash mismatch for "
                f'{job["job_id"]}'
            )

        if (
            sha256_file(system_path).lower()
            != job[
                "system_prompt_sha256"
            ].lower()
        ):
            raise ValueError(
                "System prompt hash mismatch for "
                f'{job["job_id"]}'
            )

    if (
        read_text(V3_COST_PROMPT_PATH)
        != read_text(V4_080_PROMPT_PATH)
    ):
        raise ValueError(
            "V4 0.80 prompt no longer matches "
            "the frozen V3 costly prompt"
        )


def read_records() -> list[dict[str, Any]]:
    if not OUTPUT_PATH.exists():
        return []

    records: list[dict[str, Any]] = []

    for line in OUTPUT_PATH.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        records.append(
            json.loads(line)
        )

    return records


def completed_job_ids(
    records: list[dict[str, Any]],
) -> set[str]:
    terminal_statuses = {
        "ok",
        "invalid_response",
    }

    return {
        record["job_id"]
        for record in records
        if record.get("status")
        in terminal_statuses
    }


def attempt_number(
    job_id: str,
    records: list[dict[str, Any]],
) -> int:
    previous_attempts = sum(
        1
        for record in records
        if record.get("job_id") == job_id
    )

    return previous_attempts + 1


def estimated_cost(
    usage: dict[str, Any],
) -> float:
    input_tokens = int(
        usage.get("prompt_token_count")
        or 0
    )

    output_tokens = int(
        usage.get("candidates_token_count")
        or 0
    ) + int(
        usage.get("thoughts_token_count")
        or 0
    )

    # Same pricing assumption used by the
    # frozen V3 runner, for comparability.
    return round(
        input_tokens
        * 0.50
        / 1_000_000
        + output_tokens
        * 3.00
        / 1_000_000,
        8,
    )


def run_job(
    client: Any,
    config: dict[str, Any],
    job: dict[str, str],
    attempt: int,
) -> dict[str, Any]:

    system_prompt = read_text(
        ROOT / job["system_prompt_path"]
    )

    task_prompt = read_text(
        ROOT / job["task_prompt_path"]
    )

    record: dict[str, Any] = {
        **job,
        "attempt": attempt,
        "model_id": config["model_id"],
        "temperature": float(
            config["temperature"]
        ),
        "candidate_count": int(
            config["candidate_count"]
        ),
        "prompt_hash": prompt_hash(
            system_prompt,
            task_prompt,
        ),
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    try:
        (
            raw_text,
            latency_ms,
            usage,
        ) = call_model(
            client,
            config["model_id"],
            system_prompt,
            task_prompt,
            ROOT / job["image_path"],
            float(config["temperature"]),
        )

    except Exception as exc:
        record.update(
            {
                "status": "error",
                "error_stage": "api",
                "error_type": (
                    type(exc).__name__
                ),
                "error_message": str(exc),
            }
        )

        return record

    record.update(
        {
            "raw_response": raw_text,
            "latency_ms": latency_ms,
            "usage_metadata": usage,
            "estimated_cost_usd": (
                estimated_cost(usage)
            ),
        }
    )

    try:
        allocation = parse_allocation(
            raw_text
        )

    except Exception as exc:
        # Invalid model outputs are terminal.
        # They are preserved and must NOT be
        # retried simply because parsing failed.
        record.update(
            {
                "status": (
                    "invalid_response"
                ),
                "error_stage": "parsing",
                "error_type": (
                    type(exc).__name__
                ),
                "error_message": str(exc),
            }
        )

        return record

    record.update(
        {
            "status": "ok",
            **allocation,
        }
    )

    return record


def save_record(
    record: dict[str, Any],
) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=(
            "dry-run",
            "smoke",
            "full",
        ),
        default="dry-run",
    )

    parser.add_argument(
        "--pace-seconds",
        type=float,
        default=2.0,
    )

    args = parser.parse_args()

    config = yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    validate_config(config)

    jobs = load_manifest()

    validate_manifest(jobs)

    records = read_records()

    completed = completed_job_ids(
        records
    )

    pending = [
        job
        for job in jobs
        if job["job_id"]
        not in completed
    ]

    ok_count = sum(
        1
        for record in records
        if record.get("status") == "ok"
    )

    invalid_count = sum(
        1
        for record in records
        if record.get("status")
        == "invalid_response"
    )

    api_error_count = sum(
        1
        for record in records
        if record.get("status")
        == "error"
    )

    print(
        f"Mode: {args.mode}"
    )

    print(
        f"Total planned V4 calls: "
        f"{len(jobs)}"
    )

    print(
        f"Valid completed jobs: "
        f"{ok_count}"
    )

    print(
        f"Invalid-response jobs: "
        f"{invalid_count}"
    )

    print(
        f"Logged procedural errors: "
        f"{api_error_count}"
    )

    print(
        f"Remaining jobs: "
        f"{len(pending)}"
    )

    if args.mode == "dry-run":
        print(
            "Manifest/config/input validation: "
            "PASS"
        )
        print(
            "V3 0.80 replication anchor: "
            "PASS"
        )
        print(
            "No API calls were made."
        )
        return 0

    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv(
        ROOT / ".env"
    )

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if (
        not api_key
        or api_key
        == "replace_with_your_private_key"
    ):
        print(
            "ERROR: GEMINI_API_KEY "
            "is not configured in .env"
        )
        return 1

    if not pending:
        print(
            "All 1,296 planned V4 jobs "
            "are terminal."
        )
        return 0

    if args.mode == "smoke":
        pending = pending[:1]

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=30_000,
            retry_options=(
                types.HttpRetryOptions(
                    attempts=1
                )
            ),
        ),
    )

    new_ok = 0
    new_invalid = 0
    new_api_errors = 0

    all_records = read_records()

    for index, job in enumerate(
        pending,
        1,
    ):
        if index > 1:
            time.sleep(
                max(
                    0.0,
                    args.pace_seconds,
                )
            )

        attempt = attempt_number(
            job["job_id"],
            all_records,
        )

        record = run_job(
            client,
            config,
            job,
            attempt,
        )

        save_record(record)

        all_records.append(record)

        if record["status"] == "ok":
            new_ok += 1
            print(
                f"[{index}/{len(pending)}] "
                f'OK {job["job_id"]} '
                f'{job["scene_id"]} '
                f'{job["profile"]} '
                f'{job["image_variant"]} '
                f'Xeff={job["x_efficiency"]} '
                f'r{job["repetition"]}'
            )

        elif (
            record["status"]
            == "invalid_response"
        ):
            new_invalid += 1
            print(
                f"[{index}/{len(pending)}] "
                f'INVALID {job["job_id"]} '
                f'{record["error_type"]}'
            )

        else:
            new_api_errors += 1
            print(
                f"[{index}/{len(pending)}] "
                f'API ERROR {job["job_id"]} '
                f'{record["error_type"]}'
            )

    final_records = read_records()

    final_completed = (
        completed_job_ids(
            final_records
        )
    )

    print(
        "Finished this run: "
        f"{new_ok} valid, "
        f"{new_invalid} invalid, "
        f"{new_api_errors} procedural errors."
    )

    print(
        "Terminal planned jobs: "
        f"{len(final_completed)}"
        f"/{EXPECTED_JOBS}"
    )

    if args.mode == "smoke":
        print(
            "Smoke mode executed exactly "
            "one planned V4 job."
        )

    return (
        0
        if new_api_errors == 0
        else 2
    )


if __name__ == "__main__":
    sys.exit(main())