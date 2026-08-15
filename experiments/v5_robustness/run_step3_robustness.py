from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from google import genai


ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
EXP_DIR = ROOT / "experiments" / "v5_robustness"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from run_heldout import call_model, parse_allocation, prompt_hash


CONFIG_PATH = ROOT / "config" / "experiment.yaml"

MANIFEST_PATH = EXP_DIR / "step3_robustness_manifest.csv"
MANIFEST_HASH_PATH = EXP_DIR / "step3_robustness_manifest.sha256"
PROMPT_HASH_PATH = EXP_DIR / "STEP3_PROMPTS.sha256"

OUTPUT_PATH = (
    EXP_DIR
    / "outputs"
    / "step3_robustness_raw.jsonl"
)

EXPECTED_MODEL_ID = "gemini-3-flash-preview"
EXPECTED_TEMPERATURE = 1.0
EXPECTED_CANDIDATE_COUNT = 1
EXPECTED_REPETITIONS = 3
EXPECTED_JOBS = 1080

EXPECTED_VARIANTS = {"P0", "P1", "P2", "P3", "P4"}
EXPECTED_PROFILES = {"neutral", "cue_bound", "generalized"}

EXECUTION_ORDER_SEED = 20260815
MAX_PROCEDURAL_ATTEMPTS = 3


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(
        encoding="utf-8-sig"
    ).strip()


def load_manifest() -> list[dict[str, str]]:
    with MANIFEST_PATH.open(
        newline="",
        encoding="utf-8-sig",
    ) as f:
        return list(csv.DictReader(f))


def recorded_manifest_hash() -> str:
    text = read_text(MANIFEST_HASH_PATH)

    # Builder stores:
    # HASH  step3_robustness_manifest.csv
    return text.split()[0].lower()


def load_prompt_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}

    for line in PROMPT_HASH_PATH.read_text(
        encoding="utf-8-sig"
    ).splitlines():

        if not line.strip():
            continue

        digest, relative = line.split(
            None,
            1,
        )

        relative = relative.strip()

        hashes[relative] = digest.lower()

    return hashes


def validate_config(config: dict[str, Any]) -> None:
    if config.get("model_id") != EXPECTED_MODEL_ID:
        raise ValueError(
            f"Unexpected model: {config.get('model_id')}"
        )

    if float(config["temperature"]) != EXPECTED_TEMPERATURE:
        raise ValueError("Unexpected temperature")

    if int(config["candidate_count"]) != EXPECTED_CANDIDATE_COUNT:
        raise ValueError("Unexpected candidate count")

    if int(config["repetitions"]) != EXPECTED_REPETITIONS:
        raise ValueError("Unexpected repetition count")


def validate_manifest(
    jobs: list[dict[str, str]]
) -> None:

    if sha256_file(MANIFEST_PATH).lower() != recorded_manifest_hash():
        raise ValueError(
            "STEP 3 MANIFEST SHA-256 MISMATCH"
        )

    if len(jobs) != EXPECTED_JOBS:
        raise ValueError(
            f"Expected {EXPECTED_JOBS} jobs; found {len(jobs)}"
        )

    ids = [
        job["v5_job_id"]
        for job in jobs
    ]

    if len(set(ids)) != EXPECTED_JOBS:
        raise ValueError(
            "V5 job IDs are not unique"
        )

    variants = {
        job["robustness_variant"]
        for job in jobs
    }

    if variants != EXPECTED_VARIANTS:
        raise ValueError(
            f"Unexpected variants: {variants}"
        )

    profiles = {
        job["v5_profile"]
        for job in jobs
    }

    if profiles != EXPECTED_PROFILES:
        raise ValueError(
            f"Unexpected profiles: {profiles}"
        )

    variant_counts = Counter(
        job["robustness_variant"]
        for job in jobs
    )

    for variant in EXPECTED_VARIANTS:
        if variant_counts[variant] != 216:
            raise ValueError(
                f"{variant} has "
                f"{variant_counts[variant]} jobs, expected 216"
            )

    profile_counts = Counter(
        job["v5_profile"]
        for job in jobs
    )

    for profile in EXPECTED_PROFILES:
        if profile_counts[profile] != 360:
            raise ValueError(
                f"{profile} has "
                f"{profile_counts[profile]} jobs, expected 360"
            )

    prompt_hashes = load_prompt_hashes()

    for job in jobs:

        image_path = ROOT / job["image_path"]

        if not image_path.is_file():
            raise FileNotFoundError(image_path)

        if (
            sha256_file(image_path).lower()
            != job["image_sha256"].lower()
        ):
            raise ValueError(
                f"Image hash mismatch: {job['v5_job_id']}"
            )

        for column in [
            "v5_system_prompt_path",
            "v5_task_prompt_path",
        ]:
            relative = job[column]

            path = ROOT / relative

            if not path.is_file():
                raise FileNotFoundError(path)

            # STEP3_PROMPTS.sha256 stores paths
            # relative to experiments/v5_robustness.
            rel_to_exp = path.relative_to(
                EXP_DIR
            ).as_posix()

            expected = prompt_hashes.get(
                rel_to_exp
            )

            if expected is None:
                raise ValueError(
                    f"No frozen prompt hash for {rel_to_exp}"
                )

            if sha256_file(path).lower() != expected:
                raise ValueError(
                    f"Prompt hash mismatch: {rel_to_exp}"
                )

        variant = job["robustness_variant"]

        expected_org = (
            "Y"
            if variant == "P2"
            else "X"
        )

        expected_key = (
            "y"
            if variant == "P2"
            else "x"
        )

        if job["designated_organization"] != expected_org:
            raise ValueError(
                f"Bad designated organization: "
                f"{job['v5_job_id']}"
            )

        if job["designated_output_key"] != expected_key:
            raise ValueError(
                f"Bad designated key: "
                f"{job['v5_job_id']}"
            )


def read_records() -> list[dict[str, Any]]:
    if not OUTPUT_PATH.exists():
        return []

    records = []

    for line in OUTPUT_PATH.read_text(
        encoding="utf-8"
    ).splitlines():

        if line.strip():
            records.append(
                json.loads(line)
            )

    return records


def completed_job_ids(
    records: list[dict[str, Any]]
) -> set[str]:

    return {
        r["job_id"]
        for r in records
        if r.get("status")
        in {
            "ok",
            "invalid_response",
        }
    }


def attempt_number(
    job_id: str,
    records: list[dict[str, Any]],
) -> int:

    return 1 + sum(
        1
        for r in records
        if r.get("job_id") == job_id
    )


def estimated_cost(
    usage: dict[str, Any],
) -> float:

    inp = int(
        usage.get("prompt_token_count")
        or 0
    )

    out = (
        int(
            usage.get("candidates_token_count")
            or 0
        )
        +
        int(
            usage.get("thoughts_token_count")
            or 0
        )
    )

    return round(
        inp * 0.50 / 1_000_000
        + out * 3.00 / 1_000_000,
        8,
    )


def allocation_value(
    allocation: dict[str, Any],
    key: str,
) -> float:

    candidates = {
        "x": [
            "x",
            "x_allocation",
            "allocation_x",
        ],
        "y": [
            "y",
            "y_allocation",
            "allocation_y",
        ],
    }

    for candidate in candidates[key]:
        if candidate in allocation:
            return float(
                allocation[candidate]
            )

    raise KeyError(
        f"Could not locate parsed {key} allocation "
        f"in keys {list(allocation)}"
    )


def run_job(
    client: Any,
    config: dict[str, Any],
    job: dict[str, str],
    attempt: int,
) -> dict[str, Any]:

    system_prompt = read_text(
        ROOT / job[
            "v5_system_prompt_path"
        ]
    )

    task_prompt = read_text(
        ROOT / job[
            "v5_task_prompt_path"
        ]
    )

    job_id = job["v5_job_id"]

    record: dict[str, Any] = {
        **job,
        "job_id": job_id,
        "attempt": attempt,
        "model_id": config["model_id"],
        "temperature": float(
            config["temperature"]
        ),
        "candidate_count": int(
            config["candidate_count"]
        ),
        "execution_order_seed": (
            EXECUTION_ORDER_SEED
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
        raw_text, latency_ms, usage = call_model(
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
                "error_type": type(
                    exc
                ).__name__,
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

        x_value = allocation_value(
            allocation,
            "x",
        )

        y_value = allocation_value(
            allocation,
            "y",
        )

        designated_key = job[
            "designated_output_key"
        ]

        designated_value = (
            x_value
            if designated_key == "x"
            else y_value
        )

    except Exception as exc:
        # Substantive invalid output is terminal:
        # preserve and DO NOT retry.
        record.update(
            {
                "status": "invalid_response",
                "error_stage": "parsing",
                "error_type": type(
                    exc
                ).__name__,
                "error_message": str(exc),
            }
        )

        return record

    record.update(
        {
            "status": "ok",
            **allocation,
            "x_allocation_normalized": x_value,
            "y_allocation_normalized": y_value,
            "designated_allocation": (
                designated_value
            ),
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
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


def summarize(
    records: list[dict[str, Any]],
) -> None:

    terminal = completed_job_ids(records)

    ok = sum(
        r.get("status") == "ok"
        for r in records
    )

    invalid = sum(
        r.get("status")
        == "invalid_response"
        for r in records
    )

    errors = sum(
        r.get("status") == "error"
        for r in records
    )

    print()
    print("STEP 3 STATUS")
    print("=============")
    print(
        f"Terminal jobs: {len(terminal)}/{EXPECTED_JOBS}"
    )
    print(f"Successful records: {ok}")
    print(f"Invalid terminal responses: {invalid}")
    print(f"Procedural error records: {errors}")


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["dry-run", "full"],
        default="dry-run",
    )

    parser.add_argument(
        "--pace-seconds",
        type=float,
        default=1.0,
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

    # Fixed randomized order prevents all P0 calls,
    # then all P1 calls, etc. from being confounded
    # with execution time.
    rng = random.Random(
        EXECUTION_ORDER_SEED
    )

    rng.shuffle(jobs)

    records = read_records()

    completed = completed_job_ids(
        records
    )

    print(
        f"Manifest validation: PASS ({len(jobs)} jobs)"
    )
    print(
        f"Execution-order seed: {EXECUTION_ORDER_SEED}"
    )
    print(
        f"Already terminal: {len(completed)}"
    )

    if args.mode == "dry-run":
        print("DRY RUN PASS — ZERO API CALLS")
        return 0

    load_dotenv(
        ROOT / ".env"
    )

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    client = genai.Client(
        api_key=api_key
    )

    total = len(jobs)

    for position, job in enumerate(
        jobs,
        start=1,
    ):

        job_id = job["v5_job_id"]

        # Re-read occasionally so resume/attempt
        # state stays current.
        records = read_records()
        completed = completed_job_ids(
            records
        )

        if job_id in completed:
            continue

        attempt = attempt_number(
            job_id,
            records,
        )

        while attempt <= MAX_PROCEDURAL_ATTEMPTS:

            print(
                f"[{position}/{total}] "
                f"{job_id} "
                f"{job['robustness_variant']} "
                f"{job['v5_profile']} "
                f"attempt={attempt}"
            )

            record = run_job(
                client,
                config,
                job,
                attempt,
            )

            save_record(record)

            if record["status"] in {
                "ok",
                "invalid_response",
            }:
                break

            print(
                "  procedural failure: "
                f"{record.get('error_type')}"
            )

            attempt += 1

            if attempt <= MAX_PROCEDURAL_ATTEMPTS:
                time.sleep(
                    max(
                        args.pace_seconds,
                        2.0,
                    )
                )

        time.sleep(
            max(
                args.pace_seconds,
                0.0,
            )
        )

    final_records = read_records()

    summarize(
        final_records
    )

    unresolved = EXPECTED_JOBS - len(
        completed_job_ids(
            final_records
        )
    )

    if unresolved:
        print(
            f"\nUnresolved jobs remaining: {unresolved}"
        )
        print(
            "Rerun the same command to resume."
        )

        return 2

    print(
        "\nSTEP 3 EXECUTION COMPLETE"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
