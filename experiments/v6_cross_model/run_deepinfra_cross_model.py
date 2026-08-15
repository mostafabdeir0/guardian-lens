from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "v6_cross_model"

MANIFEST = EXP / "deepinfra_manifest.csv"
MANIFEST_HASH_FILE = EXP / "deepinfra_manifest.sha256"
SOURCE_MANIFEST = EXP / "cross_model_manifest.csv"
ASSET_FREEZE = EXP / "STEP4_DEEPINFRA_ASSET_FREEZE.json"
PROVIDER_AMENDMENT = EXP / "STEP4_PROVIDER_AMENDMENT.md"

OUTPUT_DIR = EXP / "outputs"
RAW_LEDGER = OUTPUT_DIR / "deepinfra_raw.jsonl"

EXPECTED_PROVIDER = "deepinfra"
EXPECTED_MODEL = "Qwen/Qwen3.6-27B"
EXPECTED_BASE_URL = "https://api.deepinfra.com/v1/openai"

EXPECTED_MANIFEST_HASH = (
    "f570ac666f4c0c3ca2b74b2fa56f0c23b8941075e4fc3f30ec3f0a8b87a2454f"
)

EXPECTED_SOURCE_MANIFEST_HASH = (
    "844778e19951d37580eb82116f8a70a3fd13bd3b08dc6b08d5897b5c6bd191ad"
)

EXPECTED_JOBS = 216

EXPECTED_PYTHON = "3.11.2"
EXPECTED_OPENAI = "3.1.0"
EXPECTED_DOTENV = "1.2.2"

MAX_ATTEMPTS = 5

# Operational pacing only. Scientific request content is unchanged.
MIN_CALL_INTERVAL_SECONDS = 1.0

FLOAT_TOLERANCE = 1e-6


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def expected_manifest_hash_from_file() -> str:
    text = MANIFEST_HASH_FILE.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        raise RuntimeError("DeepInfra manifest hash file is empty.")

    return text.split()[0].lower()


def mime_type(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"

    if suffix == ".png":
        return "image/png"

    if suffix == ".webp":
        return "image/webp"

    raise RuntimeError(
        f"Unsupported image extension: {path}"
    )


def encode_image_data_url(path: Path) -> str:
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")

    return (
        f"data:{mime_type(path)};"
        f"base64,{encoded}"
    )


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def check_environment() -> str:
    load_dotenv(ROOT / ".env")

    token = os.environ.get("DEEPINFRA_TOKEN")

    if not token:
        raise RuntimeError(
            "DEEPINFRA_TOKEN was not found in the "
            "environment/.env file."
        )

    if len(token.strip()) < 10:
        raise RuntimeError(
            "DEEPINFRA_TOKEN appears invalid/too short."
        )

    return token


def validate_dependency_environment() -> None:
    python_version = platform.python_version()
    openai_version = package_version("openai")
    dotenv_version = package_version("python-dotenv")

    if python_version != EXPECTED_PYTHON:
        raise RuntimeError(
            "Unexpected Python version.\n"
            f"Expected: {EXPECTED_PYTHON}\n"
            f"Actual:   {python_version}"
        )

    if openai_version != EXPECTED_OPENAI:
        raise RuntimeError(
            "Unexpected openai package version.\n"
            f"Expected: {EXPECTED_OPENAI}\n"
            f"Actual:   {openai_version}"
        )

    if dotenv_version != EXPECTED_DOTENV:
        raise RuntimeError(
            "Unexpected python-dotenv version.\n"
            f"Expected: {EXPECTED_DOTENV}\n"
            f"Actual:   {dotenv_version}"
        )


def validate_provider_amendment() -> None:
    if not PROVIDER_AMENDMENT.exists():
        raise RuntimeError(
            "Missing provider amendment."
        )

    with ASSET_FREEZE.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        freeze = json.load(f)

    expected = (
        freeze["provider_amendment_sha256"]
        .lower()
    )

    actual = sha256_file(PROVIDER_AMENDMENT)

    if actual != expected:
        raise RuntimeError(
            "Provider amendment SHA-256 mismatch.\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )


def validate_source_to_deepinfra_manifest(
    source_rows: list[dict[str, str]],
    rows: list[dict[str, str]],
) -> None:
    if len(source_rows) != EXPECTED_JOBS:
        raise RuntimeError(
            f"Expected {EXPECTED_JOBS} source rows, "
            f"got {len(source_rows)}"
        )

    if len(rows) != EXPECTED_JOBS:
        raise RuntimeError(
            f"Expected {EXPECTED_JOBS} DeepInfra rows, "
            f"got {len(rows)}"
        )

    if sha256_file(SOURCE_MANIFEST) != EXPECTED_SOURCE_MANIFEST_HASH:
        raise RuntimeError(
            "Original frozen source manifest SHA mismatch."
        )

    for source, target in zip(
        source_rows,
        rows,
        strict=True,
    ):
        if (
            source["step4_job_id"]
            != target["step4_job_id"]
        ):
            raise RuntimeError(
                "Source/DeepInfra job order mismatch."
            )

        if source["provider"] != "groq":
            raise RuntimeError(
                "Unexpected source provider."
            )

        if source["model_id"] != "qwen/qwen3.6-27b":
            raise RuntimeError(
                "Unexpected source model."
            )

        if target["provider"] != EXPECTED_PROVIDER:
            raise RuntimeError(
                "Unexpected DeepInfra provider."
            )

        if target["model_id"] != EXPECTED_MODEL:
            raise RuntimeError(
                "Unexpected DeepInfra model."
            )

        if set(source) != set(target):
            raise RuntimeError(
                "Source and DeepInfra manifest columns differ."
            )

        for key in source:
            if key in {"provider", "model_id"}:
                continue

            if source[key] != target[key]:
                raise RuntimeError(
                    "DeepInfra manifest changed a frozen "
                    f"field: job={target['step4_job_id']} "
                    f"field={key}"
                )


def validate_manifest(
    rows: list[dict[str, str]]
) -> None:
    if len(rows) != EXPECTED_JOBS:
        raise RuntimeError(
            f"Expected {EXPECTED_JOBS} manifest rows, "
            f"got {len(rows)}"
        )

    actual_hash = sha256_file(MANIFEST)
    hash_file_value = expected_manifest_hash_from_file()

    if actual_hash != EXPECTED_MANIFEST_HASH:
        raise RuntimeError(
            "DeepInfra manifest SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_MANIFEST_HASH}\n"
            f"Actual:   {actual_hash}"
        )

    if hash_file_value != EXPECTED_MANIFEST_HASH:
        raise RuntimeError(
            "deepinfra_manifest.sha256 does not contain "
            "the frozen expected hash."
        )

    job_ids = [
        row["step4_job_id"]
        for row in rows
    ]

    if len(set(job_ids)) != EXPECTED_JOBS:
        raise RuntimeError(
            "DeepInfra Step-4 job IDs are not unique."
        )

    execution_orders = [
        int(row["execution_order"])
        for row in rows
    ]

    if sorted(execution_orders) != list(
        range(1, EXPECTED_JOBS + 1)
    ):
        raise RuntimeError(
            "Execution order must contain each integer "
            "from 1 through 216."
        )

    scenes = {
        row["scene_id"]
        for row in rows
    }

    if len(scenes) != 12:
        raise RuntimeError(
            f"Expected 12 scenes, got {len(scenes)}"
        )

    profile_counts: dict[str, int] = {}
    image_counts: dict[str, int] = {}
    image_hashes: set[str] = set()
    prompt_paths: set[str] = set()

    for row in rows:
        profile = row["profile"]
        image_variant = row["image_variant"]

        profile_counts[profile] = (
            profile_counts.get(profile, 0) + 1
        )

        image_counts[image_variant] = (
            image_counts.get(image_variant, 0) + 1
        )

        if row["provider"] != EXPECTED_PROVIDER:
            raise RuntimeError(
                f"Unexpected provider in "
                f"{row['step4_job_id']}: "
                f"{row['provider']}"
            )

        if row["model_id"] != EXPECTED_MODEL:
            raise RuntimeError(
                f"Unexpected model in "
                f"{row['step4_job_id']}: "
                f"{row['model_id']}"
            )

        if row["temperature"] != "1.0":
            raise RuntimeError(
                f"Unexpected temperature in "
                f"{row['step4_job_id']}"
            )

        if row["candidate_count"] != "1":
            raise RuntimeError(
                f"Unexpected candidate count in "
                f"{row['step4_job_id']}"
            )

        if row["reasoning_effort"] != "none":
            raise RuntimeError(
                f"Unexpected reasoning setting in "
                f"{row['step4_job_id']}"
            )

        if row["response_mode"] != "json_object":
            raise RuntimeError(
                f"Unexpected response mode in "
                f"{row['step4_job_id']}"
            )

        if row["designated_organization"] != "X":
            raise RuntimeError(
                "Unexpected designated organization."
            )

        if row["designated_output_key"] != "x":
            raise RuntimeError(
                "Unexpected designated output key."
            )

        if row["x_efficiency"] != "0.80":
            raise RuntimeError(
                "Unexpected X efficiency."
            )

        if row["y_efficiency"] != "1.00":
            raise RuntimeError(
                "Unexpected Y efficiency."
            )

        image_path = ROOT / row["image_path"]

        if not image_path.exists():
            raise RuntimeError(
                f"Missing image: {image_path}"
            )

        actual_image_hash = sha256_file(image_path)

        if (
            actual_image_hash
            != row["image_sha256"].lower()
        ):
            raise RuntimeError(
                "Image SHA mismatch for "
                f"{row['step4_job_id']}"
            )

        image_hashes.add(actual_image_hash)

        system_path = (
            ROOT / row["system_prompt_path"]
        )

        task_path = (
            ROOT / row["task_prompt_path"]
        )

        if not system_path.exists():
            raise RuntimeError(
                f"Missing system prompt: {system_path}"
            )

        if not task_path.exists():
            raise RuntimeError(
                f"Missing task prompt: {task_path}"
            )

        if (
            sha256_file(system_path)
            != row["system_prompt_sha256"].lower()
        ):
            raise RuntimeError(
                "System prompt SHA mismatch for "
                f"{row['step4_job_id']}"
            )

        if (
            sha256_file(task_path)
            != row["task_prompt_sha256"].lower()
        ):
            raise RuntimeError(
                "Task prompt SHA mismatch for "
                f"{row['step4_job_id']}"
            )

        prompt_paths.add(
            row["system_prompt_path"]
        )

        prompt_paths.add(
            row["task_prompt_path"]
        )

    if profile_counts != {
        "neutral": 72,
        "cue_bound": 72,
        "generalized": 72,
    }:
        raise RuntimeError(
            f"Unexpected profile counts: "
            f"{profile_counts}"
        )

    if image_counts != {
        "clean": 108,
        "modified": 108,
    }:
        raise RuntimeError(
            f"Unexpected image counts: "
            f"{image_counts}"
        )

    if len(image_hashes) != 24:
        raise RuntimeError(
            f"Expected 24 unique images, "
            f"got {len(image_hashes)}"
        )

    if len(prompt_paths) != 4:
        raise RuntimeError(
            f"Expected 4 unique prompt files, "
            f"got {len(prompt_paths)}"
        )


def validate_asset_freeze() -> None:
    with ASSET_FREEZE.open(
        "r",
        encoding="utf-8-sig",
    ) as f:
        freeze = json.load(f)

    if freeze["planned_jobs"] != EXPECTED_JOBS:
        raise RuntimeError(
            "DeepInfra asset-freeze job-count mismatch."
        )

    if (
        freeze["replacement_provider"]
        != EXPECTED_PROVIDER
    ):
        raise RuntimeError(
            "Asset-freeze provider mismatch."
        )

    if (
        freeze["replacement_model_id"]
        != EXPECTED_MODEL
    ):
        raise RuntimeError(
            "Asset-freeze model mismatch."
        )

    if (
        freeze["base_url"]
        != EXPECTED_BASE_URL
    ):
        raise RuntimeError(
            "Asset-freeze base URL mismatch."
        )

    if freeze["service_tier"] is not None:
        raise RuntimeError(
            "Service tier must remain unset."
        )

    if (
        freeze["deepinfra_manifest_sha256"].lower()
        != EXPECTED_MANIFEST_HASH
    ):
        raise RuntimeError(
            "Asset-freeze DeepInfra manifest "
            "hash mismatch."
        )

    if not freeze.get(
        "image_hashes_reverified",
        False,
    ):
        raise RuntimeError(
            "Image re-verification flag is false."
        )

    if not freeze.get(
        "prompt_hashes_reverified",
        False,
    ):
        raise RuntimeError(
            "Prompt re-verification flag is false."
        )

    if freeze.get(
        "confirmatory_use_of_partial_groq_data"
    ) != "NONE":
        raise RuntimeError(
            "Partial Groq confirmatory use is not NONE."
        )


def build_request(
    row: dict[str, str]
) -> dict[str, Any]:
    system_prompt = (
        ROOT / row["system_prompt_path"]
    ).read_text(encoding="utf-8")

    task_prompt = (
        ROOT / row["task_prompt_path"]
    ).read_text(encoding="utf-8")

    image_path = ROOT / row["image_path"]
    data_url = encode_image_data_url(image_path)

    # Keep the same scientific message organization
    # used by the frozen Groq runner:
    # system prompt, then task text + original image.
    #
    # reasoning_effort is passed through extra_body
    # because this is DeepInfra's documented Python
    # syntax for its OpenAI-compatible endpoint.
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
                            "url": data_url,
                        },
                    },
                ],
            },
        ],
        "temperature": float(
            row["temperature"]
        ),
        "n": int(
            row["candidate_count"]
        ),
        "response_format": {
            "type": "json_object",
        },
        "stream": False,
        "extra_body": {
            "reasoning_effort":
                row["reasoning_effort"],
        },
    }


def request_payload_sha256(
    payload: dict[str, Any]
) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(raw).hexdigest()


def validate_all_payloads(
    rows: list[dict[str, str]]
) -> tuple[int, int, int]:
    total = 0
    image_hashes: set[str] = set()
    request_hash_counts: dict[str, int] = {}

    for row in rows:
        payload = build_request(row)

        if payload["model"] != EXPECTED_MODEL:
            raise RuntimeError(
                "Payload model mismatch."
            )

        if len(payload["messages"]) != 2:
            raise RuntimeError(
                "Unexpected message count."
            )

        if (
            payload["messages"][0]["role"]
            != "system"
        ):
            raise RuntimeError(
                "First message must be system."
            )

        if (
            payload["messages"][1]["role"]
            != "user"
        ):
            raise RuntimeError(
                "Second message must be user."
            )

        content = (
            payload["messages"][1]["content"]
        )

        if len(content) != 2:
            raise RuntimeError(
                "User message must contain "
                "task text + image."
            )

        if content[0]["type"] != "text":
            raise RuntimeError(
                "First user content item "
                "must be text."
            )

        if content[1]["type"] != "image_url":
            raise RuntimeError(
                "Second user content item "
                "must be image_url."
            )

        data_url = (
            content[1]["image_url"]["url"]
        )

        if not data_url.startswith(
            "data:image/"
        ):
            raise RuntimeError(
                "Invalid local image data URL."
            )

        if payload["temperature"] != 1.0:
            raise RuntimeError(
                "Payload temperature mismatch."
            )

        if payload["n"] != 1:
            raise RuntimeError(
                "Payload candidate count mismatch."
            )

        if payload["response_format"] != {
            "type": "json_object"
        }:
            raise RuntimeError(
                "Payload response format mismatch."
            )

        if payload["stream"] is not False:
            raise RuntimeError(
                "Streaming must be disabled."
            )

        if payload["extra_body"] != {
            "reasoning_effort": "none"
        }:
            raise RuntimeError(
                "Payload reasoning setting mismatch."
            )

        if "service_tier" in payload:
            raise RuntimeError(
                "service_tier must remain unset."
            )

        image_hashes.add(
            row["image_sha256"]
        )

        request_hash = request_payload_sha256(payload)

        request_hash_counts[request_hash] = (
            request_hash_counts.get(request_hash, 0) + 1
        )

        total += 1

    if len(image_hashes) != 24:
        raise RuntimeError(
            "Payload image-count mismatch."
        )

    if len(request_hash_counts) != 72:
        raise RuntimeError(
            "Expected 72 unique scientific request payloads "
            "(12 scenes x 3 profiles x 2 image variants), "
            f"got {len(request_hash_counts)}."
        )

    if set(request_hash_counts.values()) != {3}:
        raise RuntimeError(
            "Each unique scientific request payload must occur "
            "exactly 3 times for the three frozen repetitions. "
            f"Observed repetition counts: "
            f"{sorted(set(request_hash_counts.values()))}"
        )

    return (
        total,
        len(image_hashes),
        len(request_hash_counts),
    )


def parse_allocation(
    raw_text: str,
) -> tuple[
    str,
    float | None,
    float | None,
    str | None,
]:
    try:
        obj = json.loads(raw_text)

    except Exception as exc:
        return (
            "invalid",
            None,
            None,
            "JSON parse failure: "
            f"{type(exc).__name__}: {exc}",
        )

    if not isinstance(obj, dict):
        return (
            "invalid",
            None,
            None,
            "Response is valid JSON "
            "but not an object.",
        )

    if "x" not in obj or "y" not in obj:
        return (
            "invalid",
            None,
            None,
            "Response object does not "
            "contain both x and y.",
        )

    x_raw = obj["x"]
    y_raw = obj["y"]

    if (
        isinstance(x_raw, bool)
        or isinstance(y_raw, bool)
    ):
        return (
            "invalid",
            None,
            None,
            "Boolean values are not "
            "valid allocations.",
        )

    try:
        x = float(x_raw)
        y = float(y_raw)

    except (TypeError, ValueError):
        return (
            "invalid",
            None,
            None,
            "x and y are not numeric.",
        )

    if (
        not math.isfinite(x)
        or not math.isfinite(y)
    ):
        return (
            "invalid",
            None,
            None,
            "x or y is not finite.",
        )

    if (
        x < 0
        or x > 100
        or y < 0
        or y > 100
    ):
        return (
            "invalid",
            x,
            y,
            "x or y lies outside [0, 100].",
        )

    total = x + y

    if abs(total - 100.0) > FLOAT_TOLERANCE:
        return (
            "invalid",
            x,
            y,
            f"x + y = {total}, not 100.",
        )

    return (
        "valid",
        x,
        y,
        None,
    )


def model_dump_safe(value: Any) -> Any:
    if value is None:
        return None

    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(
                mode="json"
            )
        except TypeError:
            return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    return str(value)


def append_record(
    record: dict[str, Any]
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RAW_LEDGER.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as f:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )

        f.flush()
        os.fsync(f.fileno())


def load_previous_state(
    allowed_job_ids: set[str],
) -> tuple[
    dict[str, int],
    set[str],
]:
    attempts: dict[str, int] = {}
    substantive_terminal: set[str] = set()

    if not RAW_LEDGER.exists():
        return attempts, substantive_terminal

    with RAW_LEDGER.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line_no, line in enumerate(
            f,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "DeepInfra raw ledger contains "
                    f"invalid JSON at line {line_no}: "
                    f"{exc}"
                )

            job_id = record.get(
                "step4_job_id"
            )

            if job_id not in allowed_job_ids:
                raise RuntimeError(
                    "Raw ledger contains unknown "
                    f"job ID at line {line_no}: "
                    f"{job_id}"
                )

            attempt = int(
                record["attempt"]
            )

            if (
                attempt < 1
                or attempt > MAX_ATTEMPTS
            ):
                raise RuntimeError(
                    "Raw ledger contains invalid "
                    f"attempt number at line "
                    f"{line_no}."
                )

            attempts[job_id] = max(
                attempts.get(job_id, 0),
                attempt,
            )

            if (
                record["record_type"]
                == "substantive_response"
            ):
                if job_id in substantive_terminal:
                    raise RuntimeError(
                        "Raw ledger contains multiple "
                        "substantive responses for "
                        f"{job_id}."
                    )

                substantive_terminal.add(
                    job_id
                )

    return (
        attempts,
        substantive_terminal,
    )


def exception_status_code(
    exc: Exception
) -> int | None:
    value = getattr(
        exc,
        "status_code",
        None,
    )

    try:
        return (
            int(value)
            if value is not None
            else None
        )

    except Exception:
        return None


def retry_wait(
    exc: Exception,
    attempt: int,
) -> float:
    response = getattr(
        exc,
        "response",
        None,
    )

    headers = getattr(
        response,
        "headers",
        None,
    )

    if headers is not None:
        retry_after = headers.get(
            "retry-after"
        )

        if retry_after:
            try:
                return max(
                    float(retry_after),
                    MIN_CALL_INTERVAL_SECONDS,
                )

            except ValueError:
                pass

    return max(
        MIN_CALL_INTERVAL_SECONDS,
        min(2 ** attempt, 30),
    )


def summarize_raw(
    allowed_job_ids: set[str],
) -> dict[str, int]:
    summary = {
        "total_records": 0,
        "procedural_errors": 0,
        "substantive_jobs": 0,
        "valid_substantive": 0,
        "invalid_substantive": 0,
    }

    substantive_ids: set[str] = set()

    if not RAW_LEDGER.exists():
        return summary

    with RAW_LEDGER.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)

            if (
                record["step4_job_id"]
                not in allowed_job_ids
            ):
                raise RuntimeError(
                    "Unknown job in raw ledger."
                )

            summary["total_records"] += 1

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
                    record["step4_job_id"]
                )

                if (
                    record.get(
                        "allocation_validity"
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

    summary["substantive_jobs"] = len(
        substantive_ids
    )

    return summary


def create_client(
    token: str
) -> OpenAI:
    # SDK-internal retries are disabled.
    # Every retry is managed and logged explicitly
    # by this runner.
    return OpenAI(
        api_key=token,
        base_url=EXPECTED_BASE_URL,
        max_retries=0,
        timeout=90.0,
    )


def execute(
    rows: list[dict[str, str]],
    token: str,
) -> None:
    allowed_job_ids = {
        row["step4_job_id"]
        for row in rows
    }

    client = create_client(token)

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

    print(
        "DEEPINFRA STEP 4 EXECUTION"
    )
    print(
        f"Provider: {EXPECTED_PROVIDER}"
    )
    print(
        f"Model: {EXPECTED_MODEL}"
    )
    print(
        f"Previously substantive jobs: "
        f"{len(substantive_terminal)}"
    )
    print(
        f"Planned jobs: {len(rows)}"
    )
    print(
        f"Raw ledger: {RAW_LEDGER}"
    )
    print()

    last_request_time: float | None = None

    for index, row in enumerate(
        ordered_rows,
        start=1,
    ):
        job_id = row["step4_job_id"]

        if job_id in substantive_terminal:
            print(
                f"[{index:03d}/{len(rows)}] "
                f"{job_id}: already substantive; "
                "skipping"
            )
            continue

        completed_attempts = (
            previous_attempts.get(
                job_id,
                0,
            )
        )

        if completed_attempts >= MAX_ATTEMPTS:
            print(
                f"[{index:03d}/{len(rows)}] "
                f"{job_id}: max procedural "
                "attempts already exhausted"
            )
            continue

        payload = build_request(row)
        payload_hash = (
            request_payload_sha256(
                payload
            )
        )

        substantive_received = False

        for attempt in range(
            completed_attempts + 1,
            MAX_ATTEMPTS + 1,
        ):
            if last_request_time is not None:
                elapsed = (
                    time.monotonic()
                    - last_request_time
                )

                wait_needed = (
                    MIN_CALL_INTERVAL_SECONDS
                    - elapsed
                )

                if wait_needed > 0:
                    time.sleep(wait_needed)

            request_started = utc_now()

            print(
                f"[{index:03d}/{len(rows)}] "
                f"{job_id} "
                f"order={row['execution_order']} "
                f"scene={row['scene_id']} "
                f"profile={row['profile']} "
                f"image={row['image_variant']} "
                f"rep={row['repetition']} "
                f"attempt={attempt}"
            )

            try:
                last_request_time = (
                    time.monotonic()
                )

                response = (
                    client.chat.completions.create(
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

                (
                    validity,
                    x,
                    y,
                    validation_error,
                ) = parse_allocation(
                    raw_text
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

                        "service_tier_requested":
                            None,

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

                        "allocation_validity":
                            validity,

                        "parsed_x":
                            x,

                        "parsed_y":
                            y,

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

                append_record(record)

                substantive_received = True
                substantive_terminal.add(
                    job_id
                )

                previous_attempts[
                    job_id
                ] = attempt

                print(
                    "  substantive response: "
                    f"{validity}"
                    + (
                        f" x={x} y={y}"
                        if (
                            x is not None
                            and y is not None
                        )
                        else ""
                    )
                )

                # Any HTTP-successful model response
                # is terminal, including a substantive
                # response that fails the x/y schema.
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

                        "service_tier_requested":
                            None,

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

                        "allocation_validity":
                            None,

                        "parsed_x":
                            None,

                        "parsed_y":
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

                append_record(record)

                previous_attempts[
                    job_id
                ] = attempt

                print(
                    "  procedural error: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

                if attempt < MAX_ATTEMPTS:
                    wait_seconds = (
                        retry_wait(
                            exc,
                            attempt,
                        )
                    )

                    print(
                        "  retrying identical "
                        "request after "
                        f"{wait_seconds:.1f}s"
                    )

                    time.sleep(
                        wait_seconds
                    )

        if not substantive_received:
            print(
                f"  {job_id}: no substantive "
                "response after "
                f"{MAX_ATTEMPTS} attempts"
            )

    (
        attempts,
        substantive_terminal,
    ) = load_previous_state(
        allowed_job_ids
    )

    exhausted = sum(
        1
        for row in rows
        if (
            row["step4_job_id"]
            not in substantive_terminal
            and attempts.get(
                row["step4_job_id"],
                0,
            )
            >= MAX_ATTEMPTS
        )
    )

    summary = summarize_raw(
        allowed_job_ids
    )

    print()
    print(
        "DeepInfra execution pass complete."
    )
    print(
        "Substantive terminal jobs: "
        f"{summary['substantive_jobs']}"
        f"/{EXPECTED_JOBS}"
    )
    print(
        "Valid substantive responses: "
        f"{summary['valid_substantive']}"
    )
    print(
        "Invalid substantive responses: "
        f"{summary['invalid_substantive']}"
    )
    print(
        "Procedural-error records: "
        f"{summary['procedural_errors']}"
    )
    print(
        "Procedurally exhausted jobs: "
        f"{exhausted}"
    )
    print(
        f"Raw ledger: {RAW_LEDGER}"
    )


def dry_run(
    rows: list[dict[str, str]]
) -> None:
    print(
        "DEEPINFRA STEP 4 DRY RUN"
    )
    print(
        "No API/model generation calls "
        "will be made."
    )
    print()

    token = check_environment()

    if not token:
        raise RuntimeError(
            "Missing DeepInfra token."
        )

    validate_dependency_environment()
    validate_provider_amendment()
    validate_asset_freeze()
    validate_manifest(rows)

    source_rows = load_csv(
        SOURCE_MANIFEST
    )

    validate_source_to_deepinfra_manifest(
        source_rows,
        rows,
    )

    (
        total_payloads,
        unique_images,
        unique_request_hashes,
    ) = validate_all_payloads(rows)

    if RAW_LEDGER.exists():
        raise RuntimeError(
            "deepinfra_raw.jsonl already exists. "
            "The pre-execution dry run expects "
            "zero DeepInfra experimental output."
        )

    print(
        "PASS: DEEPINFRA_TOKEN is present "
        "(value not displayed)."
    )

    print(
        f"PASS: Python = "
        f"{platform.python_version()}"
    )

    print(
        f"PASS: openai = "
        f"{package_version('openai')}"
    )

    print(
        f"PASS: python-dotenv = "
        f"{package_version('python-dotenv')}"
    )

    print(
        "PASS: Provider amendment SHA "
        "matches asset freeze."
    )

    print(
        "PASS: DeepInfra asset freeze "
        "validated."
    )

    print(
        f"PASS: Source manifest SHA-256 = "
        f"{sha256_file(SOURCE_MANIFEST)}"
    )

    print(
        f"PASS: DeepInfra manifest SHA-256 = "
        f"{sha256_file(MANIFEST)}"
    )

    print(
        f"PASS: Manifest jobs = "
        f"{len(rows)}"
    )

    print(
        "PASS: Source -> DeepInfra manifest "
        "changes are limited to "
        "provider/model_id."
    )

    print(
        f"PASS: Unique job IDs = "
        f"{len({r['step4_job_id'] for r in rows})}"
    )

    print(
        f"PASS: Unique execution orders = "
        f"{len({r['execution_order'] for r in rows})}"
    )

    print(
        f"PASS: Unique scenes = "
        f"{len({r['scene_id'] for r in rows})}"
    )

    print(
        f"PASS: Unique images = "
        f"{unique_images}"
    )

    print(
        "PASS: Profile counts = "
        + str(
            {
                profile: sum(
                    1
                    for row in rows
                    if row["profile"]
                    == profile
                )
                for profile in (
                    "neutral",
                    "cue_bound",
                    "generalized",
                )
            }
        )
    )

    print(
        "PASS: Image-variant counts = "
        + str(
            {
                variant: sum(
                    1
                    for row in rows
                    if row["image_variant"]
                    == variant
                )
                for variant in (
                    "clean",
                    "modified",
                )
            }
        )
    )

    print(
        f"PASS: Locally constructed "
        f"payloads = {total_payloads}"
    )

    print(
        f"PASS: Unique request payload "
        f"hashes = {unique_request_hashes}"
    )

    print(
        f"PASS: Provider = "
        f"{EXPECTED_PROVIDER}"
    )

    print(
        f"PASS: Model = "
        f"{EXPECTED_MODEL}"
    )

    print(
        f"PASS: Base URL = "
        f"{EXPECTED_BASE_URL}"
    )

    print(
        "PASS: service_tier is unset."
    )

    print(
        "PASS: Temperature = 1.0"
    )

    print(
        "PASS: Candidate count = 1"
    )

    print(
        "PASS: reasoning_effort = none"
    )

    print(
        "PASS: response_format = json_object"
    )

    print(
        "PASS: stream = false"
    )

    print(
        "PASS: DeepInfra raw ledger "
        "does not yet exist."
    )

    print()
    print(
        "DRY RUN PASSED."
    )
    print(
        "DeepInfra API calls made: 0"
    )
    print(
        "DeepInfra experimental outputs "
        "written: 0"
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the frozen DeepInfra "
            "experiment locally; make zero "
            "API calls."
        ),
    )

    mode.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Execute/resume the frozen "
            "216-job DeepInfra experiment."
        ),
    )

    args = parser.parse_args()

    rows = load_csv(MANIFEST)

    if args.dry_run:
        dry_run(rows)
        return

    if args.execute:
        token = check_environment()

        validate_dependency_environment()
        validate_provider_amendment()
        validate_asset_freeze()
        validate_manifest(rows)

        source_rows = load_csv(
            SOURCE_MANIFEST
        )

        validate_source_to_deepinfra_manifest(
            source_rows,
            rows,
        )

        execute(
            rows,
            token,
        )
        return

    raise RuntimeError(
        "No mode selected."
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print(
            "\nInterrupted by user.",
            file=sys.stderr,
        )
        raise SystemExit(130)

    except Exception as exc:
        print(
            "\nFATAL: "
            f"{type(exc).__name__}: "
            f"{exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)