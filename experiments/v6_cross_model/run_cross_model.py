from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from groq import Groq


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "v6_cross_model"

MANIFEST = EXP / "cross_model_manifest.csv"
MANIFEST_HASH_FILE = EXP / "cross_model_manifest.sha256"
ASSET_FREEZE = EXP / "STEP4_ASSET_FREEZE.json"

OUTPUT_DIR = EXP / "outputs"
RAW_LEDGER = OUTPUT_DIR / "cross_model_raw.jsonl"

EXPECTED_PROVIDER = "groq"
EXPECTED_MODEL = "qwen/qwen3.6-27b"
EXPECTED_JOBS = 216

MAX_ATTEMPTS = 5

# Operational throttling only; this does not alter experimental content.
MIN_CALL_INTERVAL_SECONDS = 5.0

FLOAT_TOLERANCE = 1e-6


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def expected_manifest_hash() -> str:
    text = MANIFEST_HASH_FILE.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError("Manifest hash file is empty.")
    return text.split()[0].lower()


def mime_type(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".webp":
        return "image/webp"

    raise RuntimeError(f"Unsupported image extension: {path}")


def encode_image_data_url(path: Path) -> str:
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type(path)};base64,{encoded}"


def validate_manifest(rows: list[dict[str, str]]) -> None:
    if len(rows) != EXPECTED_JOBS:
        raise RuntimeError(
            f"Expected {EXPECTED_JOBS} manifest rows, got {len(rows)}"
        )

    actual_manifest_hash = sha256_file(MANIFEST)
    expected_hash = expected_manifest_hash()

    if actual_manifest_hash != expected_hash:
        raise RuntimeError(
            "Manifest SHA-256 mismatch.\n"
            f"Expected: {expected_hash}\n"
            f"Actual:   {actual_manifest_hash}"
        )

    job_ids = [r["step4_job_id"] for r in rows]
    if len(set(job_ids)) != EXPECTED_JOBS:
        raise RuntimeError("Step-4 job IDs are not unique.")

    execution_orders = [int(r["execution_order"]) for r in rows]

    if sorted(execution_orders) != list(range(1, EXPECTED_JOBS + 1)):
        raise RuntimeError(
            "Execution order must contain each integer from 1 through 216."
        )

    scenes = {r["scene_id"] for r in rows}
    if len(scenes) != 12:
        raise RuntimeError(f"Expected 12 scenes, got {len(scenes)}")

    profile_counts: dict[str, int] = {}
    image_counts: dict[str, int] = {}

    for row in rows:
        profile_counts[row["profile"]] = (
            profile_counts.get(row["profile"], 0) + 1
        )
        image_counts[row["image_variant"]] = (
            image_counts.get(row["image_variant"], 0) + 1
        )

        if row["provider"] != EXPECTED_PROVIDER:
            raise RuntimeError(
                f"Unexpected provider in {row['step4_job_id']}: "
                f"{row['provider']}"
            )

        if row["model_id"] != EXPECTED_MODEL:
            raise RuntimeError(
                f"Unexpected model in {row['step4_job_id']}: "
                f"{row['model_id']}"
            )

        if row["temperature"] != "1.0":
            raise RuntimeError(
                f"Unexpected temperature in {row['step4_job_id']}"
            )

        if row["candidate_count"] != "1":
            raise RuntimeError(
                f"Unexpected candidate count in {row['step4_job_id']}"
            )

        if row["reasoning_effort"] != "none":
            raise RuntimeError(
                f"Unexpected reasoning setting in {row['step4_job_id']}"
            )

        if row["response_mode"] != "json_object":
            raise RuntimeError(
                f"Unexpected response mode in {row['step4_job_id']}"
            )

        image_path = ROOT / row["image_path"]

        if not image_path.exists():
            raise RuntimeError(f"Missing image: {image_path}")

        image_hash = sha256_file(image_path)

        if image_hash != row["image_sha256"].lower():
            raise RuntimeError(
                f"Image SHA mismatch for {row['step4_job_id']}"
            )

        system_path = ROOT / row["system_prompt_path"]
        task_path = ROOT / row["task_prompt_path"]

        if not system_path.exists():
            raise RuntimeError(f"Missing system prompt: {system_path}")

        if not task_path.exists():
            raise RuntimeError(f"Missing task prompt: {task_path}")

        if (
            sha256_file(system_path)
            != row["system_prompt_sha256"].lower()
        ):
            raise RuntimeError(
                f"System prompt SHA mismatch for {row['step4_job_id']}"
            )

        if (
            sha256_file(task_path)
            != row["task_prompt_sha256"].lower()
        ):
            raise RuntimeError(
                f"Task prompt SHA mismatch for {row['step4_job_id']}"
            )

    if profile_counts != {
        "neutral": 72,
        "cue_bound": 72,
        "generalized": 72,
    }:
        raise RuntimeError(
            f"Unexpected profile counts: {profile_counts}"
        )

    if image_counts != {
        "clean": 108,
        "modified": 108,
    }:
        raise RuntimeError(
            f"Unexpected image counts: {image_counts}"
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
                        "image_url": {
                            "url": data_url,
                        },
                    },
                ],
            },
        ],
        "temperature": float(row["temperature"]),
        "n": int(row["candidate_count"]),
        "reasoning_effort": row["reasoning_effort"],
        "response_format": {
            "type": "json_object",
        },
        "stream": False,
    }


def validate_all_payloads(
    rows: list[dict[str, str]]
) -> tuple[int, int]:
    unique_images: set[str] = set()
    total_payloads = 0

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

        content = payload["messages"][1]["content"]

        if len(content) != 2:
            raise RuntimeError("User message must have text + image.")

        if content[0]["type"] != "text":
            raise RuntimeError("First user content item must be text.")

        if content[1]["type"] != "image_url":
            raise RuntimeError(
                "Second user content item must be image_url."
            )

        data_url = content[1]["image_url"]["url"]

        if not data_url.startswith("data:image/"):
            raise RuntimeError("Invalid local image data URL.")

        # Groq documents a 20 MB image-request size limit.
        if len(data_url.encode("utf-8")) > 20 * 1024 * 1024:
            raise RuntimeError(
                f"Encoded image exceeds 20 MB: {row['image_path']}"
            )

        unique_images.add(row["image_sha256"])
        total_payloads += 1

    return total_payloads, len(unique_images)


def check_environment() -> str:
    load_dotenv(ROOT / ".env")

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY was not found in the environment/.env file."
        )

    if len(api_key.strip()) < 10:
        raise RuntimeError("GROQ_API_KEY appears invalid/too short.")

    return api_key


def parse_allocation(
    raw_text: str,
) -> tuple[str, float | None, float | None, str | None]:
    try:
        obj = json.loads(raw_text)
    except Exception as exc:
        return (
            "invalid",
            None,
            None,
            f"JSON parse failure: {type(exc).__name__}: {exc}",
        )

    if not isinstance(obj, dict):
        return (
            "invalid",
            None,
            None,
            "Response is valid JSON but not an object.",
        )

    if "x" not in obj or "y" not in obj:
        return (
            "invalid",
            None,
            None,
            "Response object does not contain both x and y.",
        )

    x_raw = obj["x"]
    y_raw = obj["y"]

    if isinstance(x_raw, bool) or isinstance(y_raw, bool):
        return (
            "invalid",
            None,
            None,
            "Boolean values are not valid allocations.",
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

    if not math.isfinite(x) or not math.isfinite(y):
        return (
            "invalid",
            None,
            None,
            "x or y is not finite.",
        )

    if x < 0 or x > 100 or y < 0 or y > 100:
        return (
            "invalid",
            x,
            y,
            "x or y lies outside [0, 100].",
        )

    if abs((x + y) - 100.0) > FLOAT_TOLERANCE:
        return (
            "invalid",
            x,
            y,
            f"x + y = {x + y}, not 100.",
        )

    return "valid", x, y, None


def usage_to_dict(usage: Any) -> Any:
    if usage is None:
        return None

    if hasattr(usage, "model_dump"):
        return usage.model_dump()

    if hasattr(usage, "dict"):
        return usage.dict()

    return str(usage)


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


def load_previous_state() -> tuple[dict[str, int], set[str]]:
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
                    f"Raw ledger contains invalid JSON at line "
                    f"{line_no}: {exc}"
                )

            job_id = record["step4_job_id"]
            attempt = int(record["attempt"])

            attempts[job_id] = max(
                attempts.get(job_id, 0),
                attempt,
            )

            if record["record_type"] == "substantive_response":
                substantive_terminal.add(job_id)

    return attempts, substantive_terminal


def exception_status_code(exc: Exception) -> int | None:
    value = getattr(exc, "status_code", None)

    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def retry_wait(exc: Exception, attempt: int) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)

    if headers is not None:
        retry_after = headers.get("retry-after")

        if retry_after:
            try:
                return max(float(retry_after), MIN_CALL_INTERVAL_SECONDS)
            except ValueError:
                pass

    return max(
        MIN_CALL_INTERVAL_SECONDS,
        min(2 ** attempt, 30),
    )


def execute(rows: list[dict[str, str]], api_key: str) -> None:
    # Disable SDK-internal retries so every API attempt is explicitly
    # represented in our Step-4 audit ledger.
    client = Groq(
        api_key=api_key,
        max_retries=0,
        timeout=60.0,
    )

    previous_attempts, substantive_terminal = load_previous_state()

    rows = sorted(
        rows,
        key=lambda r: int(r["execution_order"]),
    )

    completed_before = len(substantive_terminal)

    print(f"Previously completed substantive jobs: {completed_before}")
    print(f"Planned jobs: {len(rows)}")
    print(f"Raw ledger: {RAW_LEDGER}")
    print()

    last_request_time: float | None = None

    for index, row in enumerate(rows, start=1):
        job_id = row["step4_job_id"]

        if job_id in substantive_terminal:
            print(
                f"[{index:03d}/{len(rows)}] "
                f"{job_id}: already substantive; skipping"
            )
            continue

        completed_attempts = previous_attempts.get(job_id, 0)

        if completed_attempts >= MAX_ATTEMPTS:
            print(
                f"[{index:03d}/{len(rows)}] "
                f"{job_id}: max procedural attempts already exhausted"
            )
            continue

        payload = build_request(row)

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
                last_request_time = time.monotonic()

                response = client.chat.completions.create(**payload)

                response_received = utc_now()

                raw_text = (
                    response.choices[0].message.content
                    if response.choices
                    else ""
                )

                raw_text = raw_text or ""

                validity, x, y, validation_error = (
                    parse_allocation(raw_text)
                )

                record = dict(row)

                record.update(
                    {
                        "record_type": "substantive_response",
                        "attempt": attempt,
                        "request_started_at_utc": request_started,
                        "response_received_at_utc": response_received,
                        "raw_model_response": raw_text,
                        "allocation_validity": validity,
                        "parsed_x": x,
                        "parsed_y": y,
                        "validation_error": validation_error,
                        "provider_response_id": getattr(
                            response, "id", None
                        ),
                        "provider_response_model": getattr(
                            response, "model", None
                        ),
                        "system_fingerprint": getattr(
                            response,
                            "system_fingerprint",
                            None,
                        ),
                        "finish_reason": (
                            response.choices[0].finish_reason
                            if response.choices
                            else None
                        ),
                        "usage": usage_to_dict(
                            getattr(response, "usage", None)
                        ),
                        "error_type": None,
                        "error_message": None,
                        "http_status_code": None,
                    }
                )

                append_record(record)

                substantive_received = True
                substantive_terminal.add(job_id)
                previous_attempts[job_id] = attempt

                print(
                    f"  substantive response: {validity}"
                    + (
                        f" x={x} y={y}"
                        if x is not None and y is not None
                        else ""
                    )
                )

                # IMPORTANT:
                # A substantive but invalid allocation is terminal.
                # It is not regenerated.
                break

            except Exception as exc:
                failed_at = utc_now()

                record = dict(row)

                record.update(
                    {
                        "record_type": "procedural_error",
                        "attempt": attempt,
                        "request_started_at_utc": request_started,
                        "response_received_at_utc": failed_at,
                        "raw_model_response": None,
                        "allocation_validity": None,
                        "parsed_x": None,
                        "parsed_y": None,
                        "validation_error": None,
                        "provider_response_id": None,
                        "provider_response_model": None,
                        "system_fingerprint": None,
                        "finish_reason": None,
                        "usage": None,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "http_status_code": exception_status_code(exc),
                    }
                )

                append_record(record)
                previous_attempts[job_id] = attempt

                print(
                    f"  procedural error: "
                    f"{type(exc).__name__}: {exc}"
                )

                if attempt < MAX_ATTEMPTS:
                    wait_seconds = retry_wait(exc, attempt)

                    print(
                        f"  retrying identical request after "
                        f"{wait_seconds:.1f}s"
                    )

                    time.sleep(wait_seconds)

        if not substantive_received:
            print(
                f"  {job_id}: no substantive response after "
                f"{MAX_ATTEMPTS} attempts"
            )

    attempts, substantive_terminal = load_previous_state()

    exhausted = sum(
        1
        for row in rows
        if row["step4_job_id"] not in substantive_terminal
        and attempts.get(row["step4_job_id"], 0) >= MAX_ATTEMPTS
    )

    print()
    print("Execution pass complete.")
    print(
        f"Substantive terminal jobs: "
        f"{len(substantive_terminal)}/{EXPECTED_JOBS}"
    )
    print(f"Procedurally exhausted jobs: {exhausted}")
    print(f"Raw ledger: {RAW_LEDGER}")


def dry_run(rows: list[dict[str, str]]) -> None:
    print("STEP 4 DRY RUN")
    print("No API/model generation calls will be made.")
    print()

    api_key = check_environment()

    # Deliberately do not display any portion of the secret.
    if not api_key:
        raise RuntimeError("Missing API key.")

    validate_manifest(rows)

    with ASSET_FREEZE.open("r", encoding="utf-8") as f:
        freeze = json.load(f)

    if freeze["planned_jobs"] != EXPECTED_JOBS:
        raise RuntimeError("Asset-freeze planned-job mismatch.")

    if freeze["model_id"] != EXPECTED_MODEL:
        raise RuntimeError("Asset-freeze model mismatch.")

    if (
        freeze["step4_manifest_sha256"].lower()
        != sha256_file(MANIFEST)
    ):
        raise RuntimeError("Asset-freeze manifest SHA mismatch.")

    total_payloads, unique_images = validate_all_payloads(rows)

    print("PASS: GROQ_API_KEY is present (value not displayed).")
    print(
        f"PASS: Manifest SHA-256 = "
        f"{sha256_file(MANIFEST)}"
    )
    print(f"PASS: Manifest jobs = {len(rows)}")
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
    print(f"PASS: Unique images = {unique_images}")
    print(
        "PASS: Profile counts = "
        + str(
            {
                p: sum(1 for r in rows if r["profile"] == p)
                for p in (
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
                v: sum(
                    1 for r in rows
                    if r["image_variant"] == v
                )
                for v in ("clean", "modified")
            }
        )
    )
    print(f"PASS: Locally constructed payloads = {total_payloads}")
    print(f"PASS: Provider = {EXPECTED_PROVIDER}")
    print(f"PASS: Model = {EXPECTED_MODEL}")
    print("PASS: Temperature = 1.0")
    print("PASS: Candidate count = 1")
    print("PASS: reasoning_effort = none")
    print("PASS: response_format = json_object")
    print()
    print("DRY RUN PASSED.")
    print("API calls made: 0")
    print("Experimental outputs written: 0")


def main() -> None:
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the frozen experiment locally; make zero API calls.",
    )

    mode.add_argument(
        "--execute",
        action="store_true",
        help="Execute/resume the frozen 216-job Groq experiment.",
    )

    args = parser.parse_args()

    rows = load_manifest()

    if args.dry_run:
        dry_run(rows)
        return

    if args.execute:
        api_key = check_environment()
        validate_manifest(rows)
        execute(rows, api_key)
        return

    raise RuntimeError("No mode selected.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(
            f"\nFATAL: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)