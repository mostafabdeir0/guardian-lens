from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = ROOT / "experiments" / "v7_qwen_cross_task_replication"

MANIFEST_PATH = EXP_DIR / "qwen_v7_execution_manifest.csv"
MANIFEST_FREEZE_PATH = EXP_DIR / "QWEN_V7_MANIFEST_FREEZE.json"

SOURCE_MANIFEST_PATH = (
    ROOT
    / "experiments"
    / "v7_cross_task_validity"
    / "v7_execution_manifest.csv"
)

TASKA_PATH = (
    ROOT
    / "experiments"
    / "v7_cross_task_validity"
    / "taskA_predictions.csv"
)

OUTPUT_DIR = EXP_DIR / "outputs"
RAW_LEDGER = OUTPUT_DIR / "qwen_v7_raw.jsonl"

EXPECTED_MANIFEST_SHA256 = (
    "2220ed6b7fde210c1796f4a13e37de1c"
    "6a25a69d64d4bc9640c42ef69a0e5075"
)

EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "d52abc88fc34504af402258f4f968d4c"
    "6112357b24836a7b9e6754d7180fabfd"
)

EXPECTED_TASKA_SHA256 = (
    "c89c3e1ed5b1434baf66e450bec2e0a2"
    "fff310a816c18fff75a715aa485c61a4"
)

EXPECTED_PROVIDER = "deepinfra"
EXPECTED_MODEL = "Qwen/Qwen3.6-27B"
EXPECTED_BASE_URL = "https://api.deepinfra.com/v1/openai"

EXPECTED_TEMPERATURE = 1.0
EXPECTED_REASONING_EFFORT = "none"
EXPECTED_JOBS = 216
EXPECTED_CELLS = 72

MIN_CALL_INTERVAL_SECONDS = 0.5
PROCEDURAL_ATTEMPTS_PER_INVOCATION = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    if sha256_file(MANIFEST_PATH) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            "Qwen V7 manifest SHA-256 mismatch."
        )

    if (
        sha256_file(SOURCE_MANIFEST_PATH)
        != EXPECTED_SOURCE_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "Frozen Gemini source-manifest hash mismatch."
        )

    if sha256_file(TASKA_PATH) != EXPECTED_TASKA_SHA256:
        raise RuntimeError(
            "Frozen Task-A prediction-ledger hash mismatch."
        )

    freeze = json.loads(
        MANIFEST_FREEZE_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    if (
        freeze.get("status")
        != "FROZEN_BEFORE_ANY_QWEN_V7_MODEL_RESPONSE"
    ):
        raise RuntimeError(
            "Unexpected Qwen V7 manifest-freeze status."
        )

    if (
        freeze.get("qwen_manifest_sha256")
        != EXPECTED_MANIFEST_SHA256
    ):
        raise RuntimeError(
            "Manifest freeze hash mismatch."
        )

    if freeze.get("gemini_qwen_pooling") != "NONE":
        raise RuntimeError(
            "Gemini/Qwen pooling must remain NONE."
        )

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def validate_manifest(rows: list[dict[str, str]]) -> None:
    if len(rows) != EXPECTED_JOBS:
        raise RuntimeError(
            f"Expected 216 jobs, observed {len(rows)}."
        )

    if len({row["v7_job_id"] for row in rows}) != 216:
        raise RuntimeError(
            "Qwen V7 job IDs are not unique."
        )

    if len({row["design_cell_id"] for row in rows}) != 72:
        raise RuntimeError(
            "Expected exactly 72 design cells."
        )

    orders = sorted(
        int(row["execution_order"])
        for row in rows
    )

    if orders != list(range(1, 217)):
        raise RuntimeError(
            "Execution order must be exactly 1..216."
        )

    for row in rows:
        if row["provider"] != EXPECTED_PROVIDER:
            raise RuntimeError("Provider mismatch.")

        if row["model_id"] != EXPECTED_MODEL:
            raise RuntimeError("Model mismatch.")

        if float(row["temperature"]) != EXPECTED_TEMPERATURE:
            raise RuntimeError("Temperature mismatch.")

        if row["candidate_count"] != "1":
            raise RuntimeError("Candidate-count mismatch.")

        if row["reasoning_effort"] != EXPECTED_REASONING_EFFORT:
            raise RuntimeError("Reasoning-effort mismatch.")

        if row["response_mode"] != "json_object":
            raise RuntimeError("Response-mode mismatch.")

        if row["execution_seed"] != "20260816":
            raise RuntimeError("Execution-seed mismatch.")

        if row["deepinfra_base_url"] != EXPECTED_BASE_URL:
            raise RuntimeError("DeepInfra URL mismatch.")

        image_path = ROOT / row["image_path"]

        if not image_path.is_file():
            raise RuntimeError(
                f"Missing image: {row['image_path']}"
            )

        if sha256_file(image_path) != row["image_sha256"]:
            raise RuntimeError(
                f"Image hash mismatch: {row['image_path']}"
            )

        system_path = ROOT / row["system_prompt_path"]
        task_path = ROOT / row["task_prompt_path"]

        if sha256_file(system_path) != row["system_prompt_sha256"]:
            raise RuntimeError(
                "System-prompt hash mismatch."
            )

        if sha256_file(task_path) != row["task_prompt_sha256"]:
            raise RuntimeError(
                "Task-prompt hash mismatch."
            )


def encode_image_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)

    if not mime_type:
        mime_type = "image/jpeg"

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    return f"data:{mime_type};base64,{encoded}"


def build_request(row: dict[str, str]) -> dict[str, Any]:
    system_prompt = read_text(
        ROOT / row["system_prompt_path"]
    )

    task_prompt = read_text(
        ROOT / row["task_prompt_path"]
    )

    image_url = encode_image_data_url(
        ROOT / row["image_path"]
    )

    return {
        "model": row["model_id"],
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": task_prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            },
        ],
        "temperature": float(row["temperature"]),
        "n": int(row["candidate_count"]),
        "response_format": {
            "type": "json_object",
        },
        "stream": False,
        "extra_body": {
            "reasoning_effort": row["reasoning_effort"],
        },
    }


def request_payload_sha256(
    payload: dict[str, Any],
) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def validate_payloads(
    rows: list[dict[str, str]],
) -> tuple[int, int]:
    payload_hashes: set[str] = set()

    for row in rows:
        payload = build_request(row)

        if payload["model"] != EXPECTED_MODEL:
            raise RuntimeError(
                "Payload model mismatch."
            )

        if (
            payload["temperature"]
            != EXPECTED_TEMPERATURE
        ):
            raise RuntimeError(
                "Payload temperature mismatch."
            )

        if payload["n"] != 1:
            raise RuntimeError(
                "Payload candidate-count mismatch."
            )

        if (
            payload["response_format"]
            != {"type": "json_object"}
        ):
            raise RuntimeError(
                "Payload response-format mismatch."
            )

        if (
            payload["extra_body"]
            != {
                "reasoning_effort":
                    EXPECTED_REASONING_EFFORT
            }
        ):
            raise RuntimeError(
                "Payload reasoning-effort mismatch."
            )

        messages = payload["messages"]

        if len(messages) != 2:
            raise RuntimeError(
                "Expected exactly two messages."
            )

        if messages[0]["role"] != "system":
            raise RuntimeError(
                "First message must be system."
            )

        if messages[1]["role"] != "user":
            raise RuntimeError(
                "Second message must be user."
            )

        content = messages[1]["content"]

        if len(content) != 2:
            raise RuntimeError(
                "User message must contain "
                "task text and image."
            )

        if content[0]["type"] != "text":
            raise RuntimeError(
                "First user content must be text."
            )

        if (
            content[1]["type"]
            != "image_url"
        ):
            raise RuntimeError(
                "Second user content must be image."
            )

        payload_hashes.add(
            request_payload_sha256(
                payload
            )
        )

    if (
        len(payload_hashes)
        != EXPECTED_CELLS
    ):
        raise RuntimeError(
            "Expected 72 unique scientific "
            "payloads; observed "
            f"{len(payload_hashes)}."
        )

    return (
        len(rows),
        len(payload_hashes),
    )


def parse_choice(
    raw_text: str,
) -> tuple[str, int]:
    value = json.loads(raw_text)

    if not isinstance(value, dict):
        raise ValueError(
            "Response must be a JSON object."
        )

    if set(value.keys()) != {"choice"}:
        raise ValueError(
            'Response must contain exactly '
            'one key: "choice".'
        )

    choice = value["choice"]

    if choice not in {"X", "Y"}:
        raise ValueError(
            'choice must be exactly "X" or "Y".'
        )

    return (
        choice,
        1 if choice == "X" else 0,
    )


def model_dump_safe(
    value: Any,
) -> Any:
    if value is None:
        return None

    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(
                mode="json"
            )

        except TypeError:
            return value.model_dump()

    return value


def exception_status_code(
    exc: Exception,
) -> int | None:
    status = getattr(
        exc,
        "status_code",
        None,
    )

    if isinstance(status, int):
        return status

    response = getattr(
        exc,
        "response",
        None,
    )

    status = getattr(
        response,
        "status_code",
        None,
    )

    if isinstance(status, int):
        return status

    return None


def append_record(
    record: dict[str, Any],
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialized = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    with RAW_LEDGER.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(
            serialized + "\n"
        )

        handle.flush()

        os.fsync(
            handle.fileno()
        )


def load_previous_state(
    allowed_job_ids: set[str],
) -> tuple[
    dict[str, int],
    set[str],
]:
    attempts: dict[str, int] = {}

    substantive_terminal: set[str] = (
        set()
    )

    if not RAW_LEDGER.exists():
        return (
            attempts,
            substantive_terminal,
        )

    with RAW_LEDGER.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "Malformed raw-ledger JSON "
                    f"at line {line_number}."
                ) from exc

            job_id = record.get(
                "v7_job_id"
            )

            if (
                job_id
                not in allowed_job_ids
            ):
                raise RuntimeError(
                    "Raw ledger contains an "
                    "unexpected job ID."
                )

            attempt = record.get(
                "attempt"
            )

            if (
                not isinstance(attempt, int)
                or attempt < 1
            ):
                raise RuntimeError(
                    "Invalid attempt number "
                    f"at line {line_number}."
                )

            attempts[job_id] = max(
                attempts.get(job_id, 0),
                attempt,
            )

            record_type = record.get(
                "record_type"
            )

            if (
                record_type
                == "substantive_response"
            ):
                if (
                    job_id
                    in substantive_terminal
                ):
                    raise RuntimeError(
                        "Duplicate substantive "
                        "terminal response for "
                        f"{job_id}."
                    )

                substantive_terminal.add(
                    job_id
                )

            elif (
                record_type
                != "procedural_error"
            ):
                raise RuntimeError(
                    "Unknown record_type "
                    f"at line {line_number}."
                )

    return (
        attempts,
        substantive_terminal,
    )


def ledger_summary(
    allowed_job_ids: set[str],
) -> dict[str, int]:
    summary = {
        "audit_records": 0,
        "substantive_jobs": 0,
        "valid_substantive": 0,
        "invalid_substantive": 0,
        "procedural_errors": 0,
    }

    if not RAW_LEDGER.exists():
        return summary

    substantive_ids: set[str] = set()

    with RAW_LEDGER.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        for line in handle:
            if not line.strip():
                continue

            record = json.loads(line)

            job_id = record[
                "v7_job_id"
            ]

            if (
                job_id
                not in allowed_job_ids
            ):
                raise RuntimeError(
                    "Unexpected job ID "
                    "in raw ledger."
                )

            summary[
                "audit_records"
            ] += 1

            if (
                record["record_type"]
                == "procedural_error"
            ):
                summary[
                    "procedural_errors"
                ] += 1

            elif (
                record["record_type"]
                == "substantive_response"
            ):
                substantive_ids.add(
                    job_id
                )

                if (
                    record.get(
                        "choice_validity"
                    )
                    == "valid"
                ):
                    summary[
                        "valid_substantive"
                    ] += 1

                else:
                    summary[
                        "invalid_substantive"
                    ] += 1

    summary[
        "substantive_jobs"
    ] = len(substantive_ids)

    return summary


def create_client(
    token: str,
) -> OpenAI:
    return OpenAI(
        api_key=token,
        base_url=EXPECTED_BASE_URL,
        max_retries=0,
        timeout=90.0,
    )


def execute(
    rows: list[dict[str, str]],
    token: str,
    mode: str,
) -> None:
    allowed_job_ids = {
        row["v7_job_id"]
        for row in rows
    }

    (
        previous_attempts,
        substantive_terminal,
    ) = load_previous_state(
        allowed_job_ids
    )

    ordered_rows = sorted(
        rows,
        key=lambda row: int(
            row["execution_order"]
        ),
    )

    pending = [
        row
        for row in ordered_rows
        if (
            row["v7_job_id"]
            not in substantive_terminal
        )
    ]

    if mode == "smoke":
        selected = pending[:3]

    elif mode == "full":
        selected = pending

    else:
        raise RuntimeError(
            f"Unexpected execution mode: {mode}"
        )

    print(
        "QWEN V7 DEEPINFRA EXECUTION"
    )
    print(
        f"Mode: {mode}"
    )
    print(
        f"Provider: {EXPECTED_PROVIDER}"
    )
    print(
        f"Model: {EXPECTED_MODEL}"
    )
    print(
        "Previously substantive jobs: "
        f"{len(substantive_terminal)}"
    )
    print(
        "Pending jobs before pass: "
        f"{len(pending)}"
    )
    print(
        "Jobs selected this pass: "
        f"{len(selected)}"
    )
    print(
        f"Raw ledger: {RAW_LEDGER}"
    )
    print()

    if not selected:
        print(
            "PASS: No pending jobs selected."
        )
        return

    client = create_client(token)

    last_request_time: float | None = None

    for index, row in enumerate(
        selected,
        start=1,
    ):
        job_id = row[
            "v7_job_id"
        ]

        first_attempt = (
            previous_attempts.get(
                job_id,
                0,
            )
            + 1
        )

        final_attempt = (
            first_attempt
            + PROCEDURAL_ATTEMPTS_PER_INVOCATION
            - 1
        )

        payload = build_request(
            row
        )

        payload_hash = (
            request_payload_sha256(
                payload
            )
        )

        substantive_received = False

        for attempt in range(
            first_attempt,
            final_attempt + 1,
        ):
            if (
                last_request_time
                is not None
            ):
                elapsed = (
                    time.monotonic()
                    - last_request_time
                )

                wait_needed = (
                    MIN_CALL_INTERVAL_SECONDS
                    - elapsed
                )

                if wait_needed > 0:
                    time.sleep(
                        wait_needed
                    )

            print(
                f"[{index:03d}/{len(selected)}] "
                f"{job_id} "
                f"order={row['execution_order']} "
                f"scene={row['scene_id']} "
                f"profile={row['profile']} "
                f"image={row['image_variant']} "
                f"rep={row['repetition']} "
                f"attempt={attempt}"
            )

            request_started = utc_now()

            try:
                last_request_time = (
                    time.monotonic()
                )

                response = (
                    client
                    .chat
                    .completions
                    .create(
                        **payload
                    )
                )

                response_received = utc_now()

                raw_content: Any = None

                if response.choices:
                    raw_content = (
                        response
                        .choices[0]
                        .message
                        .content
                    )

                if raw_content is None:
                    raw_text = ""

                elif isinstance(
                    raw_content,
                    str,
                ):
                    raw_text = raw_content

                else:
                    raw_text = json.dumps(
                        raw_content,
                        ensure_ascii=False,
                    )

                choice: str | None = None
                choice_x: int | None = None
                validation_error: str | None = None

                try:
                    (
                        choice,
                        choice_x,
                    ) = parse_choice(
                        raw_text
                    )

                    validity = "valid"

                except Exception as exc:
                    validity = "invalid"

                    validation_error = (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    )

                record = dict(row)

                record.update(
                    {
                        "record_type":
                            "substantive_response",

                        "attempt":
                            attempt,

                        "request_started_at_utc":
                            request_started,

                        "response_received_at_utc":
                            response_received,

                        "request_payload_sha256":
                            payload_hash,

                        "api_base_url":
                            EXPECTED_BASE_URL,

                        "python_version":
                            platform.python_version(),

                        "openai_sdk_version":
                            package_version(
                                "openai"
                            ),

                        "python_dotenv_version":
                            package_version(
                                "python-dotenv"
                            ),

                        "raw_model_response":
                            raw_text,

                        "choice_validity":
                            validity,

                        "choice":
                            choice,

                        "choice_x":
                            choice_x,

                        "validation_error":
                            validation_error,

                        "provider_response_id":
                            getattr(
                                response,
                                "id",
                                None,
                            ),

                        "provider_response_model":
                            getattr(
                                response,
                                "model",
                                None,
                            ),

                        "provider_service_tier":
                            getattr(
                                response,
                                "service_tier",
                                None,
                            ),

                        "system_fingerprint":
                            getattr(
                                response,
                                "system_fingerprint",
                                None,
                            ),

                        "finish_reason":
                            (
                                response
                                .choices[0]
                                .finish_reason
                                if response.choices
                                else None
                            ),

                        "usage":
                            model_dump_safe(
                                getattr(
                                    response,
                                    "usage",
                                    None,
                                )
                            ),

                        "provider_response":
                            model_dump_safe(
                                response
                            ),

                        "error_type":
                            None,

                        "error_message":
                            None,

                        "http_status_code":
                            None,
                    }
                )

                append_record(
                    record
                )

                substantive_terminal.add(
                    job_id
                )

                previous_attempts[
                    job_id
                ] = attempt

                substantive_received = True

                # Intentionally do not print
                # the substantive X/Y choice.
                print(
                    "  substantive response: "
                    f"{validity}"
                )

                # Any HTTP-successful response
                # is terminal, including an
                # invalid-schema response.
                break

            except Exception as exc:
                failed_at = utc_now()

                record = dict(row)

                record.update(
                    {
                        "record_type":
                            "procedural_error",

                        "attempt":
                            attempt,

                        "request_started_at_utc":
                            request_started,

                        "response_received_at_utc":
                            failed_at,

                        "request_payload_sha256":
                            payload_hash,

                        "api_base_url":
                            EXPECTED_BASE_URL,

                        "python_version":
                            platform.python_version(),

                        "openai_sdk_version":
                            package_version(
                                "openai"
                            ),

                        "python_dotenv_version":
                            package_version(
                                "python-dotenv"
                            ),

                        "raw_model_response":
                            None,

                        "choice_validity":
                            None,

                        "choice":
                            None,

                        "choice_x":
                            None,

                        "validation_error":
                            None,

                        "provider_response_id":
                            None,

                        "provider_response_model":
                            None,

                        "provider_service_tier":
                            None,

                        "system_fingerprint":
                            None,

                        "finish_reason":
                            None,

                        "usage":
                            None,

                        "provider_response":
                            None,

                        "error_type":
                            type(exc).__name__,

                        "error_message":
                            str(exc),

                        "http_status_code":
                            exception_status_code(
                                exc
                            ),
                    }
                )

                append_record(
                    record
                )

                previous_attempts[
                    job_id
                ] = attempt

                print(
                    "  procedural error: "
                    f"{type(exc).__name__}"
                )

        if not substantive_received:
            print(
                "  WARNING: no substantive "
                "response after "
                f"{PROCEDURAL_ATTEMPTS_PER_INVOCATION} "
                "procedural attempts in "
                "this invocation."
            )

    summary = ledger_summary(
        allowed_job_ids
    )

    print()
    print(
        "PASS: DeepInfra execution pass complete."
    )
    print(
        "Audit records: "
        f"{summary['audit_records']}"
    )
    print(
        "Substantive jobs: "
        f"{summary['substantive_jobs']}/216"
    )
    print(
        "Valid substantive: "
        f"{summary['valid_substantive']}"
    )
    print(
        "Invalid substantive: "
        f"{summary['invalid_substantive']}"
    )
    print(
        "Procedural errors: "
        f"{summary['procedural_errors']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "dry-run",
            "smoke",
            "full",
        ),
    )

    args = parser.parse_args()

    load_dotenv(
        ROOT / ".env"
    )

    rows = load_manifest()

    validate_manifest(rows)

    (
        validated_jobs,
        unique_payloads,
    ) = validate_payloads(rows)

    print(
        "QWEN V7 REPLICATION RUNNER"
    )
    print(
        f"Mode: {args.mode}"
    )
    print(
        "Manifest SHA256: "
        f"{sha256_file(MANIFEST_PATH)}"
    )
    print(
        "Jobs validated: "
        f"{validated_jobs}"
    )
    print(
        "Unique scientific payloads: "
        f"{unique_payloads}"
    )
    print(
        f"Provider: {EXPECTED_PROVIDER}"
    )
    print(
        f"Model: {EXPECTED_MODEL}"
    )
    print(
        "Reasoning effort: "
        f"{EXPECTED_REASONING_EFFORT}"
    )

    token = os.environ.get(
        "DEEPINFRA_TOKEN"
    )

    if (
        not token
        or len(token) < 10
    ):
        raise RuntimeError(
            "DEEPINFRA_TOKEN is missing "
            "or suspiciously short."
        )

    if args.mode == "dry-run":
        if RAW_LEDGER.exists():
            raise RuntimeError(
                "Dry-run requires zero "
                "Qwen V7 experimental output, "
                "but qwen_v7_raw.jsonl "
                "already exists."
            )

        print(
            "PASS: DEEPINFRA_TOKEN is present."
        )
        print(
            "PASS: frozen manifests and "
            "scientific assets validated."
        )
        print(
            "PASS: all request payloads validated."
        )
        print(
            "DeepInfra API calls made: 0"
        )
        print(
            "Qwen V7 experimental outputs created: 0"
        )

        return

    execute(
        rows,
        token,
        args.mode,
    )


if __name__ == "__main__":
    main()
