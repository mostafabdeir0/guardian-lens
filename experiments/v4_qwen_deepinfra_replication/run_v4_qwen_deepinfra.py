from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import platform
import re
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "v4_qwen_deepinfra_replication"

MANIFEST = EXP / "v4_qwen_deepinfra_manifest.csv"
MANIFEST_HASH_FILE = EXP / "v4_qwen_deepinfra_manifest.sha256"
SOURCE_MANIFEST = (
    ROOT
    / "experiments"
    / "v4_cost_response"
    / "v4_cost_response_manifest.csv"
)
ASSET_FREEZE = EXP / "V4_QWEN_ASSET_FREEZE.json"
REPLICATION_SPEC = EXP / "REPLICATION_SPEC.md"

OUTPUT_DIR = EXP / "outputs"
RAW_LEDGER = OUTPUT_DIR / "v4_qwen_deepinfra_raw.jsonl"

EXPECTED_PROVIDER = "deepinfra"
EXPECTED_MODEL = "Qwen/Qwen3.6-27B"
EXPECTED_BASE_URL = "https://api.deepinfra.com/v1/openai"

EXPECTED_MANIFEST_HASH = (
    "e372b4a1ab6def665b795818d745534749e03f18f4e7fdfcaed6c642c2aa964d"
)
EXPECTED_SOURCE_MANIFEST_HASH = (
    "02f10e375f37c1bbfecdcee8b87bd763dfcf487e78107db6cde3cc50b494350c"
)

EXPECTED_JOBS = 1296
EXPECTED_UNIQUE_SCIENTIFIC_PAYLOADS = 432
EXPECTED_REPETITIONS_PER_PAYLOAD = 3
EXPECTED_MAX_TOKENS = 64

EXPECTED_PYTHON = "3.11.2"
EXPECTED_OPENAI = "3.1.0"
EXPECTED_DOTENV = "1.2.2"

MAX_ATTEMPTS = 5
MIN_CALL_INTERVAL_SECONDS = 1.0
FLOAT_TOLERANCE = 1e-6

SOURCE_COLUMNS = [
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
    text = MANIFEST_HASH_FILE.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError("Replication manifest hash file is empty.")
    return text.split()[0].lower()


def mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    raise RuntimeError(f"Unsupported image suffix: {path}")


def encode_image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type(path)};base64,{encoded}"


def check_environment() -> str:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    token = os.environ.get("DEEPINFRA_TOKEN")

    if not token:
        raise RuntimeError(
            "DEEPINFRA_TOKEN was not found in the environment/.env file."
        )

    if len(token.strip()) < 10:
        raise RuntimeError("DEEPINFRA_TOKEN appears invalid/too short.")

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


def validate_asset_freeze() -> dict[str, Any]:
    with ASSET_FREEZE.open("r", encoding="utf-8-sig") as f:
        freeze = json.load(f)

    expected_status = (
        "FROZEN_BEFORE_ANY_V4_QWEN_REPLICATION_MODEL_RESPONSE"
    )
    if freeze.get("status") != expected_status:
        raise RuntimeError("Replication asset-freeze status mismatch.")

    if freeze.get("source_manifest_sha256", "").lower() != (
        EXPECTED_SOURCE_MANIFEST_HASH
    ):
        raise RuntimeError("Asset-freeze source manifest hash mismatch.")

    if freeze.get("target_manifest_sha256", "").lower() != (
        EXPECTED_MANIFEST_HASH
    ):
        raise RuntimeError("Asset-freeze target manifest hash mismatch.")

    if freeze.get("provider") != EXPECTED_PROVIDER:
        raise RuntimeError("Asset-freeze provider mismatch.")

    if freeze.get("model_id") != EXPECTED_MODEL:
        raise RuntimeError("Asset-freeze model mismatch.")

    if freeze.get("base_url") != EXPECTED_BASE_URL:
        raise RuntimeError("Asset-freeze base URL mismatch.")

    if freeze.get("service_tier") is not None:
        raise RuntimeError("Asset-freeze service_tier must be null/unset.")

    if float(freeze.get("temperature")) != 1.0:
        raise RuntimeError("Asset-freeze temperature mismatch.")

    if int(freeze.get("candidate_count")) != 1:
        raise RuntimeError("Asset-freeze candidate count mismatch.")

    if freeze.get("reasoning_effort") != "none":
        raise RuntimeError("Asset-freeze reasoning setting mismatch.")

    if freeze.get("response_mode") != "json_object":
        raise RuntimeError("Asset-freeze response mode mismatch.")

    if freeze.get("stream") is not False:
        raise RuntimeError("Asset-freeze stream setting mismatch.")

    if int(freeze.get("planned_substantive_jobs")) != EXPECTED_JOBS:
        raise RuntimeError("Asset-freeze job-count mismatch.")

    if int(freeze.get("unique_scientific_cells")) != (
        EXPECTED_UNIQUE_SCIENTIFIC_PAYLOADS
    ):
        raise RuntimeError("Asset-freeze scientific-cell count mismatch.")

    if int(freeze.get("repetitions_per_scientific_cell")) != 3:
        raise RuntimeError("Asset-freeze repetition count mismatch.")

    if freeze.get("original_scientific_fields_changed") != 0:
        raise RuntimeError("Asset-freeze reports scientific-field changes.")

    if freeze.get("raw_replication_data_exists_at_freeze") is not False:
        raise RuntimeError(
            "Asset-freeze does not certify raw-data absence at freeze."
        )

    spec_hash = sha256_file(REPLICATION_SPEC)
    if freeze.get("replication_spec_sha256", "").lower() != spec_hash:
        raise RuntimeError("Replication spec SHA-256 mismatch.")

    verification = freeze.get("asset_verification", {})
    for key, expected_count in (
        ("image_path", 24),
        ("system_prompt_path", 3),
        ("task_prompt_path", 6),
    ):
        block = verification.get(key, {})
        if int(block.get("count", -1)) != expected_count:
            raise RuntimeError(f"Asset-freeze {key} count mismatch.")
        if block.get("all_hashes_match") is not True:
            raise RuntimeError(
                f"Asset-freeze {key} does not report all hashes matching."
            )

    return freeze


def validate_source_to_replication_manifest(
    source_rows: list[dict[str, str]],
    rows: list[dict[str, str]],
) -> None:
    if len(source_rows) != EXPECTED_JOBS:
        raise RuntimeError(
            f"Expected {EXPECTED_JOBS} source rows, got {len(source_rows)}"
        )

    if len(rows) != EXPECTED_JOBS:
        raise RuntimeError(
            f"Expected {EXPECTED_JOBS} replication rows, got {len(rows)}"
        )

    if sha256_file(SOURCE_MANIFEST) != EXPECTED_SOURCE_MANIFEST_HASH:
        raise RuntimeError("Frozen V4 source manifest SHA mismatch.")

    for index, (source, target) in enumerate(
        zip(source_rows, rows, strict=True),
        start=1,
    ):
        for key in SOURCE_COLUMNS:
            if source.get(key) != target.get(key):
                raise RuntimeError(
                    "Replication manifest changed a frozen V4 field: "
                    f"row={index} job={target.get('job_id')} field={key}"
                )

        if target["provider"] != EXPECTED_PROVIDER:
            raise RuntimeError(f"Unexpected provider in {target['job_id']}.")
        if target["model_id"] != EXPECTED_MODEL:
            raise RuntimeError(f"Unexpected model in {target['job_id']}.")
        if target["base_url"] != EXPECTED_BASE_URL:
            raise RuntimeError(f"Unexpected base URL in {target['job_id']}.")
        if float(target["temperature"]) != 1.0:
            raise RuntimeError(
                f"Unexpected temperature in {target['job_id']}."
            )
        if target["candidate_count"] != "1":
            raise RuntimeError(
                f"Unexpected candidate count in {target['job_id']}."
            )
        if target["reasoning_effort"] != "none":
            raise RuntimeError(
                f"Unexpected reasoning setting in {target['job_id']}."
            )
        if target["response_mode"] != "json_object":
            raise RuntimeError(
                f"Unexpected response mode in {target['job_id']}."
            )
        if target["stream"].lower() != "false":
            raise RuntimeError(
                f"Unexpected stream setting in {target['job_id']}."
            )
        if target["service_tier"] != "":
            raise RuntimeError(
                f"service_tier must be unset in {target['job_id']}."
            )


def validate_manifest(rows: list[dict[str, str]]) -> None:
    if len(rows) != EXPECTED_JOBS:
        raise RuntimeError(
            f"Expected {EXPECTED_JOBS} manifest rows, got {len(rows)}"
        )

    actual_hash = sha256_file(MANIFEST)
    hash_file_value = expected_manifest_hash_from_file()

    if actual_hash != EXPECTED_MANIFEST_HASH:
        raise RuntimeError(
            "Replication manifest SHA-256 mismatch.\n"
            f"Expected: {EXPECTED_MANIFEST_HASH}\n"
            f"Actual:   {actual_hash}"
        )

    if hash_file_value != EXPECTED_MANIFEST_HASH:
        raise RuntimeError(
            "v4_qwen_deepinfra_manifest.sha256 does not contain "
            "the frozen expected hash."
        )

    expected_columns = SOURCE_COLUMNS + ADDED_COLUMNS
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != expected_columns:
            raise RuntimeError("Unexpected replication manifest columns.")

    expected_ids = [
        f"v4_{i:04d}" for i in range(1, EXPECTED_JOBS + 1)
    ]
    job_ids = [row["job_id"] for row in rows]
    if job_ids != expected_ids:
        raise RuntimeError(
            "Replication job IDs/order must be v4_0001..v4_1296."
        )

    counts = {
        "profile": Counter(row["profile"] for row in rows),
        "image_variant": Counter(row["image_variant"] for row in rows),
        "x_efficiency": Counter(row["x_efficiency"] for row in rows),
        "repetition": Counter(row["repetition"] for row in rows),
        "condition": Counter(row["condition"] for row in rows),
    }

    if dict(counts["profile"]) != {
        "neutral": 432,
        "cue_bound": 432,
        "generalized": 432,
    }:
        raise RuntimeError(
            f"Unexpected profile counts: {dict(counts['profile'])}"
        )

    if dict(counts["image_variant"]) != {
        "clean": 648,
        "modified": 648,
    }:
        raise RuntimeError(
            "Unexpected image-variant counts: "
            f"{dict(counts['image_variant'])}"
        )

    if dict(counts["x_efficiency"]) != {
        "1.00": 216,
        "0.90": 216,
        "0.80": 216,
        "0.60": 216,
        "0.40": 216,
        "0.20": 216,
    }:
        raise RuntimeError(
            f"Unexpected efficiency counts: {dict(counts['x_efficiency'])}"
        )

    if dict(counts["repetition"]) != {
        "1": 432,
        "2": 432,
        "3": 432,
    }:
        raise RuntimeError(
            f"Unexpected repetition counts: {dict(counts['repetition'])}"
        )

    if dict(counts["condition"]) != {
        "target_clear": 540,
        "target_subtle": 432,
        "distractor_clear": 324,
    }:
        raise RuntimeError(
            f"Unexpected condition counts: {dict(counts['condition'])}"
        )

    scene_counts = Counter(row["scene_id"] for row in rows)
    if len(scene_counts) != 12 or set(scene_counts.values()) != {108}:
        raise RuntimeError(
            f"Unexpected per-scene counts: {dict(scene_counts)}"
        )

    image_hashes: set[str] = set()
    system_paths: set[str] = set()
    task_paths: set[str] = set()

    for row in rows:
        image_path = ROOT / row["image_path"]
        system_path = ROOT / row["system_prompt_path"]
        task_path = ROOT / row["task_prompt_path"]

        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not system_path.is_file():
            raise FileNotFoundError(system_path)
        if not task_path.is_file():
            raise FileNotFoundError(task_path)

        actual_image_hash = sha256_file(image_path)
        if actual_image_hash.lower() != row["image_sha256"].lower():
            raise RuntimeError(
                f"Image SHA mismatch for {row['job_id']}."
            )

        actual_system_hash = sha256_file(system_path)
        if actual_system_hash.lower() != row["system_prompt_sha256"].lower():
            raise RuntimeError(
                f"System prompt SHA mismatch for {row['job_id']}."
            )

        actual_task_hash = sha256_file(task_path)
        if actual_task_hash.lower() != row["task_prompt_sha256"].lower():
            raise RuntimeError(
                f"Task prompt SHA mismatch for {row['job_id']}."
            )

        image_hashes.add(actual_image_hash)
        system_paths.add(row["system_prompt_path"])
        task_paths.add(row["task_prompt_path"])

    if len(image_hashes) != 24:
        raise RuntimeError(
            f"Expected 24 unique image hashes, got {len(image_hashes)}."
        )
    if len(system_paths) != 3:
        raise RuntimeError(
            f"Expected 3 system prompts, got {len(system_paths)}."
        )
    if len(task_paths) != 6:
        raise RuntimeError(
            f"Expected 6 task prompts, got {len(task_paths)}."
        )


def build_request(row: dict[str, str]) -> dict[str, Any]:
    system_prompt = (
        ROOT / row["system_prompt_path"]
    ).read_text(encoding="utf-8")

    task_prompt = (
        ROOT / row["task_prompt_path"]
    ).read_text(encoding="utf-8")

    image_path = ROOT / row["image_path"]
    data_url = encode_image_data_url(image_path)

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
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
        "temperature": float(row["temperature"]),
        "n": int(row["candidate_count"]),
        "max_tokens": EXPECTED_MAX_TOKENS,
        "response_format": {"type": "json_object"},
        "stream": False,
        "extra_body": {
            "reasoning_effort": row["reasoning_effort"],
        },
    }


def request_payload_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def validate_all_payloads(
    rows: list[dict[str, str]],
) -> tuple[int, int, int]:
    total = 0
    image_hashes: set[str] = set()
    request_hash_counts: Counter[str] = Counter()

    for row in rows:
        payload = build_request(row)

        if payload["model"] != EXPECTED_MODEL:
            raise RuntimeError("Payload model mismatch.")

        if len(payload["messages"]) != 2:
            raise RuntimeError("Unexpected message count.")

        if payload["messages"][0]["role"] != "system":
            raise RuntimeError("First message must be system.")

        if payload["messages"][1]["role"] != "user":
            raise RuntimeError("Second message must be user.")

        user_content = payload["messages"][1]["content"]
        if (
            not isinstance(user_content, list)
            or len(user_content) != 2
            or user_content[0].get("type") != "text"
            or user_content[1].get("type") != "image_url"
        ):
            raise RuntimeError(
                "User message must be task text followed by image."
            )

        if payload["temperature"] != 1.0:
            raise RuntimeError("Payload temperature mismatch.")

        if payload["n"] != 1:
            raise RuntimeError("Payload candidate count mismatch.")

        if payload["max_tokens"] != EXPECTED_MAX_TOKENS:
            raise RuntimeError("Payload max_tokens mismatch.")

        if payload["response_format"] != {"type": "json_object"}:
            raise RuntimeError("Payload response format mismatch.")

        if payload["stream"] is not False:
            raise RuntimeError("Payload stream setting mismatch.")

        if payload["extra_body"] != {"reasoning_effort": "none"}:
            raise RuntimeError("Payload reasoning setting mismatch.")

        image_path = ROOT / row["image_path"]
        actual_image_hash = sha256_file(image_path)
        if actual_image_hash.lower() != row["image_sha256"].lower():
            raise RuntimeError(
                f"Payload image hash mismatch for {row['job_id']}."
            )

        image_hashes.add(actual_image_hash)
        request_hash_counts[request_payload_sha256(payload)] += 1
        total += 1

    if total != EXPECTED_JOBS:
        raise RuntimeError(
            f"Expected {EXPECTED_JOBS} payloads, got {total}."
        )

    if len(image_hashes) != 24:
        raise RuntimeError(
            f"Expected 24 image hashes, got {len(image_hashes)}."
        )

    if len(request_hash_counts) != EXPECTED_UNIQUE_SCIENTIFIC_PAYLOADS:
        raise RuntimeError(
            "Expected 432 unique scientific request payloads "
            "(12 scenes x 3 profiles x 2 image variants x 6 costs), "
            f"got {len(request_hash_counts)}."
        )

    if set(request_hash_counts.values()) != {
        EXPECTED_REPETITIONS_PER_PAYLOAD
    }:
        raise RuntimeError(
            "Each unique scientific payload must occur exactly 3 times."
        )

    return total, len(image_hashes), len(request_hash_counts)


def parse_allocation(
    raw_text: str,
) -> tuple[str, float | None, float | None, str | None]:
    # Match the frozen V4 parser semantics:
    # tolerate only an accidental outer Markdown code fence,
    # require exactly x/y, reject booleans, enforce range and sum.
    try:
        cleaned = raw_text.strip()
        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s*```$", "", cleaned)
        payload = json.loads(cleaned)

        if not isinstance(payload, dict) or set(payload) != {"x", "y"}:
            raise ValueError(
                'Response must contain exactly the keys "x" and "y".'
            )

        x_raw = payload["x"]
        y_raw = payload["y"]

        if isinstance(x_raw, bool) or isinstance(y_raw, bool):
            raise ValueError(
                "Allocations must be numbers, not booleans."
            )

        x = float(x_raw)
        y = float(y_raw)

        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Allocations must be finite numbers.")

        if not (0 <= x <= 100 and 0 <= y <= 100):
            raise ValueError(
                "Allocations must be between 0 and 100."
            )

        if abs((x + y) - 100.0) > FLOAT_TOLERANCE:
            raise ValueError("Allocations must sum to 100.")

        return "valid", x, y, None

    except Exception as exc:
        return (
            "invalid",
            None,
            None,
            f"{type(exc).__name__}: {exc}",
        )


def model_dump_safe(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, (dict, list, str, int, float, bool)):
        return value
    return repr(value)


def append_record(record: dict[str, Any]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
) -> tuple[dict[str, int], set[str]]:
    attempts: dict[str, int] = {}
    substantive_terminal: set[str] = set()

    if not RAW_LEDGER.exists():
        return attempts, substantive_terminal

    with RAW_LEDGER.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "DeepInfra raw ledger contains invalid JSON "
                    f"at line {line_no}: {exc}"
                ) from exc

            job_id = record.get("job_id")
            if job_id not in allowed_job_ids:
                raise RuntimeError(
                    "Raw ledger contains unknown job ID "
                    f"at line {line_no}: {job_id}"
                )

            record_type = record.get("record_type")
            if record_type not in {
                "procedural_error",
                "substantive_response",
            }:
                raise RuntimeError(
                    "Raw ledger contains unknown record_type "
                    f"at line {line_no}: {record_type}"
                )

            attempt = int(record["attempt"])
            if attempt < 1 or attempt > MAX_ATTEMPTS:
                raise RuntimeError(
                    "Raw ledger contains invalid attempt number "
                    f"at line {line_no}."
                )

            attempts[job_id] = max(
                attempts.get(job_id, 0),
                attempt,
            )

            if record_type == "substantive_response":
                if job_id in substantive_terminal:
                    raise RuntimeError(
                        "Raw ledger contains multiple substantive "
                        f"responses for {job_id}."
                    )
                substantive_terminal.add(job_id)

    return attempts, substantive_terminal


def exception_status_code(exc: Exception) -> int | None:
    for name in ("status_code", "status"):
        value = getattr(exc, name, None)
        if isinstance(value, int):
            return value

    response = getattr(exc, "response", None)
    if response is not None:
        value = getattr(response, "status_code", None)
        if isinstance(value, int):
            return value

    return None


def retry_wait(exc: Exception, attempt: int) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)

    if headers:
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                return max(
                    float(retry_after),
                    MIN_CALL_INTERVAL_SECONDS,
                )
            except (TypeError, ValueError):
                pass

    return max(
        MIN_CALL_INTERVAL_SECONDS,
        min(2 ** attempt, 30),
    )


def summarize_raw(
    allowed_job_ids: set[str],
) -> dict[str, int]:
    summary = {
        "records": 0,
        "procedural_errors": 0,
        "substantive_jobs": 0,
        "valid_substantive": 0,
        "invalid_substantive": 0,
    }

    if not RAW_LEDGER.exists():
        return summary

    substantive_ids: set[str] = set()

    with RAW_LEDGER.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            job_id = record.get("job_id")
            if job_id not in allowed_job_ids:
                raise RuntimeError(
                    f"Unknown job ID at raw line {line_no}: {job_id}"
                )

            summary["records"] += 1

            if record["record_type"] == "procedural_error":
                summary["procedural_errors"] += 1
            elif record["record_type"] == "substantive_response":
                substantive_ids.add(job_id)
                if record.get("allocation_validity") == "valid":
                    summary["valid_substantive"] += 1
                elif record.get("allocation_validity") == "invalid":
                    summary["invalid_substantive"] += 1
                else:
                    raise RuntimeError(
                        "Substantive record missing valid allocation status "
                        f"at raw line {line_no}."
                    )
            else:
                raise RuntimeError(
                    f"Unknown record_type at raw line {line_no}."
                )

    summary["substantive_jobs"] = len(substantive_ids)
    return summary


def create_client(token: str) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=token,
        base_url=EXPECTED_BASE_URL,
        max_retries=0,
        timeout=90.0,
    )


def preflight(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], tuple[int, int, int]]:
    validate_dependency_environment()
    validate_asset_freeze()
    validate_manifest(rows)

    source_rows = load_csv(SOURCE_MANIFEST)
    validate_source_to_replication_manifest(source_rows, rows)

    payload_summary = validate_all_payloads(rows)

    return source_rows, payload_summary


def execute(rows: list[dict[str, str]], token: str) -> None:
    preflight(rows)

    allowed_job_ids = {row["job_id"] for row in rows}
    attempts, substantive_terminal = load_previous_state(
        allowed_job_ids
    )

    print(f"Frozen manifest jobs: {len(rows)}")
    print(
        "Previously substantive jobs: "
        f"{len(substantive_terminal)}"
    )
    print(
        "Remaining jobs: "
        f"{len(rows) - len(substantive_terminal)}"
    )

    client = create_client(token)
    last_request_time: float | None = None

    for index, row in enumerate(rows, start=1):
        job_id = row["job_id"]

        if job_id in substantive_terminal:
            print(
                f"[{index:04d}/{len(rows)}] "
                f"{job_id}: already substantive; skipping"
            )
            continue

        completed_attempts = attempts.get(job_id, 0)
        if completed_attempts >= MAX_ATTEMPTS:
            print(
                f"[{index:04d}/{len(rows)}] "
                f"{job_id}: procedural attempts exhausted; skipping"
            )
            continue

        substantive_received = False

        for attempt in range(
            completed_attempts + 1,
            MAX_ATTEMPTS + 1,
        ):
            if last_request_time is not None:
                elapsed = time.monotonic() - last_request_time
                wait_needed = MIN_CALL_INTERVAL_SECONDS - elapsed
                if wait_needed > 0:
                    time.sleep(wait_needed)

            payload = build_request(row)
            payload_hash = request_payload_sha256(payload)

            print(
                f"[{index:04d}/{len(rows)}] "
                f"{job_id} "
                f"scene={row['scene_id']} "
                f"profile={row['profile']} "
                f"image={row['image_variant']} "
                f"Xeff={row['x_efficiency']} "
                f"r{row['repetition']} "
                f"attempt={attempt}"
            )

            started = time.perf_counter()

            try:
                last_request_time = time.monotonic()
                response = client.chat.completions.create(**payload)
                latency_ms = round(
                    (time.perf_counter() - started) * 1000
                )

                if len(response.choices) != 1:
                    raise RuntimeError(
                        "Provider returned unexpected choice count "
                        f"{len(response.choices)}."
                    )

                raw_text = (
                    response.choices[0].message.content or ""
                ).strip()

                validity, x, y, allocation_error = parse_allocation(
                    raw_text
                )

                record = {
                    **row,
                    "record_type": "substantive_response",
                    "attempt": attempt,
                    "timestamp_utc": utc_now(),
                    "request_payload_sha256": payload_hash,
                    "raw_response": raw_text,
                    "latency_ms": latency_ms,
                    "allocation_validity": validity,
                    "parsed_x": x,
                    "parsed_y": y,
                    "allocation_error": allocation_error,
                    "provider_response": model_dump_safe(response),
                    "usage": model_dump_safe(
                        getattr(response, "usage", None)
                    ),
                    "provider_response_model": getattr(
                        response,
                        "model",
                        None,
                    ),
                }

                append_record(record)
                attempts[job_id] = attempt
                substantive_terminal.add(job_id)
                substantive_received = True

                print(
                    "  substantive response: "
                    f"{validity}"
                    + (
                        f" x={x:g} y={y:g}"
                        if validity == "valid"
                        and x is not None
                        and y is not None
                        else f" error={allocation_error}"
                    )
                )

                # Any HTTP-successful model completion is terminal,
                # including one that fails the frozen x/y schema.
                break

            except Exception as exc:
                latency_ms = round(
                    (time.perf_counter() - started) * 1000
                )
                status_code = exception_status_code(exc)

                record = {
                    **row,
                    "record_type": "procedural_error",
                    "attempt": attempt,
                    "timestamp_utc": utc_now(),
                    "request_payload_sha256": payload_hash,
                    "latency_ms": latency_ms,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "status_code": status_code,
                }

                append_record(record)
                attempts[job_id] = attempt

                print(
                    "  procedural error: "
                    f"{type(exc).__name__}: {exc}"
                )

                if attempt < MAX_ATTEMPTS:
                    wait_seconds = retry_wait(exc, attempt)
                    print(
                        f"  retrying after {wait_seconds:.1f}s"
                    )
                    time.sleep(wait_seconds)

        if not substantive_received:
            print(
                f"  {job_id}: no substantive response after "
                f"{MAX_ATTEMPTS} attempts"
            )

    attempts, substantive_terminal = load_previous_state(
        allowed_job_ids
    )
    summary = summarize_raw(allowed_job_ids)

    exhausted = sum(
        1
        for row in rows
        if (
            row["job_id"] not in substantive_terminal
            and attempts.get(row["job_id"], 0) >= MAX_ATTEMPTS
        )
    )

    print("\n=== EXECUTION SUMMARY ===")
    print(f"Raw ledger records: {summary['records']}")
    print(
        "Substantive terminal jobs: "
        f"{summary['substantive_jobs']}/{EXPECTED_JOBS}"
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
        "Procedural error records: "
        f"{summary['procedural_errors']}"
    )
    print(f"Procedurally exhausted jobs: {exhausted}")
    print(f"Raw ledger: {RAW_LEDGER}")


def dry_run(rows: list[dict[str, str]]) -> None:
    token = check_environment()
    _ = token  # presence only; value is never printed

    if RAW_LEDGER.exists():
        raise RuntimeError(
            "v4_qwen_deepinfra_raw.jsonl already exists. "
            "The pre-execution dry run expects zero replication output."
        )

    _, payload_summary = preflight(rows)
    total, image_count, unique_payloads = payload_summary

    print("PASS: zero-generation dry run")
    print(f"PASS: Python = {platform.python_version()}")
    print(f"PASS: openai = {package_version('openai')}")
    print(
        "PASS: python-dotenv = "
        f"{package_version('python-dotenv')}"
    )
    print(
        "PASS: DEEPINFRA_TOKEN is present "
        "(value not displayed)."
    )
    print(
        "PASS: Source manifest SHA-256 = "
        f"{sha256_file(SOURCE_MANIFEST)}"
    )
    print(
        "PASS: Replication manifest SHA-256 = "
        f"{sha256_file(MANIFEST)}"
    )
    print(f"PASS: Manifest jobs = {len(rows)}")
    print(
        "PASS: Frozen V4 scientific fields and row order "
        "match source exactly."
    )
    print(
        f"PASS: Unique job IDs = "
        f"{len({r['job_id'] for r in rows})}"
    )
    print(f"PASS: Local payloads = {total}")
    print(f"PASS: Unique image hashes = {image_count}")
    print(
        "PASS: Unique scientific request payloads = "
        f"{unique_payloads}"
    )
    print(
        "PASS: Each scientific payload occurs exactly "
        f"{EXPECTED_REPETITIONS_PER_PAYLOAD} times."
    )
    print(f"PASS: Provider = {EXPECTED_PROVIDER}")
    print(f"PASS: Model = {EXPECTED_MODEL}")
    print(f"PASS: Base URL = {EXPECTED_BASE_URL}")
    print("PASS: service_tier = unset")
    print("PASS: temperature = 1.0")
    print("PASS: candidate_count = 1")
    print(f"PASS: max_tokens = {EXPECTED_MAX_TOKENS}")
    print("PASS: reasoning_effort = none")
    print("PASS: response_format = json_object")
    print("PASS: stream = false")
    print("PASS: raw replication ledger absent")
    print("API calls made: 0")


def main() -> None:
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate the frozen 1296-job V4 Qwen replication "
            "locally; make zero API calls."
        ),
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Execute/resume the frozen 1296-job V4 Qwen "
            "replication."
        ),
    )

    args = parser.parse_args()
    rows = load_csv(MANIFEST)

    if args.dry_run:
        dry_run(rows)
        return

    if args.execute:
        token = check_environment()
        execute(rows, token)
        return

    raise RuntimeError("No execution mode selected.")


if __name__ == "__main__":
    main()
