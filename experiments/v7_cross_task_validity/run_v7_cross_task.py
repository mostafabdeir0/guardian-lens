from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
EXP_DIR = ROOT / "experiments" / "v7_cross_task_validity"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_heldout import call_model, prompt_hash


MANIFEST_PATH = EXP_DIR / "v7_execution_manifest.csv"
MANIFEST_FREEZE_PATH = EXP_DIR / "V7_MANIFEST_FREEZE.json"

OUTPUT_PATH = (
    EXP_DIR
    / "outputs"
    / "v7_cross_task_raw.jsonl"
)

EXPECTED_MANIFEST_SHA256 = (
    "d52abc88fc34504af402258f4f968d4c"
    "6112357b24836a7b9e6754d7180fabfd"
)

EXPECTED_MANIFEST_FREEZE_SHA256 = (
    "5f09fcf62ee443e2698cac608ea504fd"
    "186c0e5987369b2e2eda276c76a12aae"
)

EXPECTED_MODEL_ID = "gemini-3-flash-preview"
EXPECTED_TEMPERATURE = 1.0
EXPECTED_CANDIDATE_COUNT = 1
EXPECTED_THINKING_LEVEL = "minimal"
EXPECTED_RESPONSE_MODE = "json_object"

EXPECTED_JOBS = 216
EXPECTED_CELLS = 72
EXPECTED_REPETITIONS = {1, 2, 3}

EXPECTED_PROFILES = {
    "neutral",
    "cue_bound",
    "generalized",
}

EXPECTED_PROFILE_CODES = {
    "A",
    "B",
    "C",
}

EXPECTED_IMAGE_VARIANTS = {
    "clean",
    "modified",
}

EXPECTED_CONDITIONS = {
    "target_clear",
    "target_subtle",
    "distractor_clear",
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


def validate_freeze() -> None:
    actual_manifest_hash = sha256_file(
        MANIFEST_PATH
    )

    if (
        actual_manifest_hash.lower()
        != EXPECTED_MANIFEST_SHA256.lower()
    ):
        raise ValueError(
            "V7 execution manifest hash mismatch.\n"
            f"Expected: {EXPECTED_MANIFEST_SHA256}\n"
            f"Observed: {actual_manifest_hash}"
        )

    actual_freeze_hash = sha256_file(
        MANIFEST_FREEZE_PATH
    )

    if (
        actual_freeze_hash.lower()
        != EXPECTED_MANIFEST_FREEZE_SHA256.lower()
    ):
        raise ValueError(
            "V7 manifest-freeze hash mismatch.\n"
            f"Expected: {EXPECTED_MANIFEST_FREEZE_SHA256}\n"
            f"Observed: {actual_freeze_hash}"
        )

    freeze = json.loads(
        MANIFEST_FREEZE_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    if freeze.get("status") != (
        "V7_EXECUTION_MANIFEST_FROZEN_"
        "BEFORE_ANY_V7_RESPONSE"
    ):
        raise ValueError(
            "Unexpected V7 manifest freeze status."
        )

    if (
        freeze.get(
            "v7_execution_manifest_sha256"
        )
        != EXPECTED_MANIFEST_SHA256
    ):
        raise ValueError(
            "Freeze metadata does not point to "
            "the expected V7 manifest."
        )

    if int(freeze.get("planned_jobs", -1)) != (
        EXPECTED_JOBS
    ):
        raise ValueError(
            "Unexpected planned job count in "
            "freeze metadata."
        )


def validate_manifest(
    jobs: list[dict[str, str]],
) -> None:
    if len(jobs) != EXPECTED_JOBS:
        raise ValueError(
            f"Expected {EXPECTED_JOBS} V7 jobs; "
            f"found {len(jobs)}."
        )

    job_ids = [
        row["v7_job_id"]
        for row in jobs
    ]

    if len(set(job_ids)) != EXPECTED_JOBS:
        raise ValueError(
            "V7 job IDs are not unique."
        )

    orders = [
        int(row["execution_order"])
        for row in jobs
    ]

    if sorted(orders) != list(
        range(1, EXPECTED_JOBS + 1)
    ):
        raise ValueError(
            "Execution order is not a unique "
            "1..216 permutation."
        )

    profiles = {
        row["profile"]
        for row in jobs
    }

    if profiles != EXPECTED_PROFILES:
        raise ValueError(
            f"Unexpected profiles: {profiles}"
        )

    profile_codes = {
        row["profile_code"]
        for row in jobs
    }

    if profile_codes != EXPECTED_PROFILE_CODES:
        raise ValueError(
            "Unexpected profile codes: "
            f"{profile_codes}"
        )

    variants = {
        row["image_variant"]
        for row in jobs
    }

    if variants != EXPECTED_IMAGE_VARIANTS:
        raise ValueError(
            "Unexpected image variants: "
            f"{variants}"
        )

    conditions = {
        row["condition"]
        for row in jobs
    }

    if conditions != EXPECTED_CONDITIONS:
        raise ValueError(
            f"Unexpected conditions: {conditions}"
        )

    repetitions = {
        int(row["repetition"])
        for row in jobs
    }

    if repetitions != EXPECTED_REPETITIONS:
        raise ValueError(
            "Unexpected repetitions: "
            f"{repetitions}"
        )

    profile_counts = Counter(
        row["profile"]
        for row in jobs
    )

    if profile_counts != {
        "neutral": 72,
        "cue_bound": 72,
        "generalized": 72,
    }:
        raise ValueError(
            "Unexpected profile factorial counts: "
            f"{profile_counts}"
        )

    variant_counts = Counter(
        row["image_variant"]
        for row in jobs
    )

    if variant_counts != {
        "clean": 108,
        "modified": 108,
    }:
        raise ValueError(
            "Unexpected image-variant counts: "
            f"{variant_counts}"
        )

    condition_counts = Counter(
        row["condition"]
        for row in jobs
    )

    if condition_counts != {
        "target_clear": 90,
        "target_subtle": 72,
        "distractor_clear": 54,
    }:
        raise ValueError(
            "Unexpected condition counts: "
            f"{condition_counts}"
        )

    cell_repetitions: dict[
        tuple[str, str, str],
        list[int],
    ] = defaultdict(list)

    for job in jobs:
        if job["model_id"] != EXPECTED_MODEL_ID:
            raise ValueError(
                "Unexpected model for "
                f'{job["v7_job_id"]}: '
                f'{job["model_id"]}'
            )

        if float(job["temperature"]) != (
            EXPECTED_TEMPERATURE
        ):
            raise ValueError(
                "Unexpected temperature for "
                f'{job["v7_job_id"]}'
            )

        if int(job["candidate_count"]) != (
            EXPECTED_CANDIDATE_COUNT
        ):
            raise ValueError(
                "Unexpected candidate count for "
                f'{job["v7_job_id"]}'
            )

        if job["thinking_level"] != (
            EXPECTED_THINKING_LEVEL
        ):
            raise ValueError(
                "Unexpected thinking level for "
                f'{job["v7_job_id"]}'
            )

        if job["response_mode"] != (
            EXPECTED_RESPONSE_MODE
        ):
            raise ValueError(
                "Unexpected response mode for "
                f'{job["v7_job_id"]}'
            )

        if job["task_family"] != (
            "binary_indivisible_contract"
        ):
            raise ValueError(
                "Unexpected task family for "
                f'{job["v7_job_id"]}'
            )

        if job["x_benefit_units"] != "80":
            raise ValueError(
                "Unexpected X benefit units."
            )

        if job["y_benefit_units"] != "100":
            raise ValueError(
                "Unexpected Y benefit units."
            )

        if (
            job["efficiency_penalty_percent"]
            != "20"
        ):
            raise ValueError(
                "Unexpected efficiency penalty."
            )

        image_path = (
            ROOT / job["image_path"]
        )

        system_path = (
            ROOT / job["system_prompt_path"]
        )

        task_path = (
            ROOT / job["task_prompt_path"]
        )

        for path in (
            image_path,
            system_path,
            task_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        if (
            sha256_file(image_path).lower()
            != job["image_sha256"].lower()
        ):
            raise ValueError(
                "Image hash mismatch for "
                f'{job["v7_job_id"]}'
            )

        if (
            sha256_file(system_path).lower()
            != job[
                "system_prompt_sha256"
            ].lower()
        ):
            raise ValueError(
                "System prompt hash mismatch for "
                f'{job["v7_job_id"]}'
            )

        if (
            sha256_file(task_path).lower()
            != job[
                "task_prompt_sha256"
            ].lower()
        ):
            raise ValueError(
                "Task prompt hash mismatch for "
                f'{job["v7_job_id"]}'
            )

        key = (
            job["scene_id"],
            job["profile"],
            job["image_variant"],
        )

        cell_repetitions[key].append(
            int(job["repetition"])
        )

    if len(cell_repetitions) != EXPECTED_CELLS:
        raise ValueError(
            f"Expected {EXPECTED_CELLS} cells; "
            f"found {len(cell_repetitions)}."
        )

    for key, reps in cell_repetitions.items():
        if sorted(reps) != [1, 2, 3]:
            raise ValueError(
                f"Bad repetitions for {key}: "
                f"{reps}"
            )


def parse_choice(
    raw_text: str,
) -> dict[str, Any]:
    value = json.loads(raw_text)

    if not isinstance(value, dict):
        raise ValueError(
            "Response must be a JSON object."
        )

    if set(value.keys()) != {"choice"}:
        raise ValueError(
            "Response must contain exactly one "
            'key: "choice".'
        )

    choice = value["choice"]

    if choice not in {"X", "Y"}:
        raise ValueError(
            'choice must be exactly "X" or "Y".'
        )

    return {
        "choice": choice,
        "choice_x": 1 if choice == "X" else 0,
    }


def read_records() -> list[dict[str, Any]]:
    if not OUTPUT_PATH.exists():
        return []

    records: list[dict[str, Any]] = []

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if not line.strip():
                continue

            records.append(json.loads(line))

    return records


def completed_job_ids(
    records: list[dict[str, Any]],
) -> set[str]:
    terminal_statuses = {
        "ok",
        "invalid_response",
    }

    return {
        record["v7_job_id"]
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
        if record.get("v7_job_id")
        == job_id
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

    # Preserve the same accounting convention
    # used in the original Gemini experiments.
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
            EXPECTED_MODEL_ID,
            system_prompt,
            task_prompt,
            ROOT / job["image_path"],
            EXPECTED_TEMPERATURE,
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
        parsed = parse_choice(raw_text)

    except Exception as exc:
        # Parsing failures are preserved as
        # terminal substantive records.
        # They are not silently retried.
        record.update(
            {
                "status": "invalid_response",
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
            **parsed,
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
        newline="\n",
    ) as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
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

    validate_freeze()

    jobs = load_manifest()
    validate_manifest(jobs)

    records = read_records()

    completed = completed_job_ids(records)

    pending = [
        job
        for job in jobs
        if job["v7_job_id"]
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
        if record.get("status") == "error"
    )

    print(f"Mode: {args.mode}")
    print(
        f"Total planned V7 calls: "
        f"{len(jobs)}"
    )
    print(
        f"Valid completed jobs: "
        f"{ok_count}"
    )
    print(
        f"Terminal invalid responses: "
        f"{invalid_count}"
    )
    print(
        f"Recorded API errors: "
        f"{api_error_count}"
    )
    print(
        f"Pending substantive jobs: "
        f"{len(pending)}"
    )
    print(
        "Manifest SHA256: "
        f"{sha256_file(MANIFEST_PATH)}"
    )

    if args.mode == "dry-run":
        print(
            "PASS: V7 dry-run validation "
            "completed. No API calls made."
        )
        return 0

    if not pending:
        print(
            "No pending V7 jobs."
        )
        return 0

    load_dotenv(
        dotenv_path=ROOT / ".env"
    )

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        print(
            "ERROR: GEMINI_API_KEY is not "
            "configured in the local .env file.",
            file=sys.stderr,
        )
        return 2

    from google import genai

    client = genai.Client(
        api_key=api_key
    )

    if args.mode == "smoke":
        selected = pending[:3]
    else:
        selected = pending

    print(
        f"Executing {len(selected)} "
        "currently pending jobs."
    )

    for index, job in enumerate(
        selected,
        start=1,
    ):
        # Reload the ledger before each call so
        # attempt numbering remains append-only
        # and auditable.
        current_records = read_records()

        attempt = attempt_number(
            job["v7_job_id"],
            current_records,
        )

        record = run_job(
            client,
            job,
            attempt,
        )

        save_record(record)

        # Deliberately do not print the model's
        # substantive X/Y choice here.
        print(
            f"[{index}/{len(selected)}] "
            f'{job["v7_job_id"]} '
            f"attempt={attempt} "
            f'status={record["status"]}'
        )

        if (
            index < len(selected)
            and args.pace_seconds > 0
        ):
            time.sleep(
                args.pace_seconds
            )

    final_records = read_records()

    final_completed = completed_job_ids(
        final_records
    )

    final_ok = sum(
        1
        for record in final_records
        if record.get("status") == "ok"
    )

    final_invalid = sum(
        1
        for record in final_records
        if record.get("status")
        == "invalid_response"
    )

    final_errors = sum(
        1
        for record in final_records
        if record.get("status") == "error"
    )

    print()
    print("V7 execution pass complete.")
    print(
        f"Terminal completed jobs: "
        f"{len(final_completed)}/"
        f"{EXPECTED_JOBS}"
    )
    print(
        f"Valid responses: {final_ok}"
    )
    print(
        f"Invalid responses: "
        f"{final_invalid}"
    )
    print(
        f"API error records: "
        f"{final_errors}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
