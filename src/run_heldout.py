"""Run the frozen Guardian Lens held-out evaluation against Gemini.

The runner is resume-safe and writes only anonymized profile codes to the raw
JSONL. A private local mapping is created once and excluded from version
control. Dry-run mode validates the frozen inputs without using the API.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "experiment.yaml"
CLASSIFIER_PATH = ROOT / "config" / "frozen_classifier.json"
DATASET_FREEZE_PATH = ROOT / "config" / "heldout_dataset_freeze.json"
MANIFEST_PATH = ROOT / "data" / "images" / "heldout" / "heldout_manifest.csv"
PROMPT_DIR = ROOT / "prompts"
OUTPUT_PATH = ROOT / "outputs" / "raw" / "heldout_raw.jsonl"
PRIVATE_MAPPING_PATH = ROOT / "private" / "profile_code_mapping.json"
PROFILE_CODE_SEED = 20260816
RUN_ORDER_SEED = 20260817
PROFILE_NAMES = ("neutral", "cue_bound", "generalized")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip()


def prompt_hash(system_prompt: str, task_prompt: str) -> str:
    payload = f"{system_prompt}\n---\n{task_prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def parse_allocation(text: str) -> dict[str, float]:
    # Reuse the already-tested parser to keep pilot and held-out behavior equal.
    from run_pilot import parse_allocation as pilot_parse_allocation

    return pilot_parse_allocation(text)


def create_or_load_mapping() -> dict[str, str]:
    """Return code -> profile mapping, creating the private mapping once."""
    if PRIVATE_MAPPING_PATH.exists():
        payload = json.loads(PRIVATE_MAPPING_PATH.read_text(encoding="utf-8"))
        mapping = payload["code_to_profile"]
    else:
        profiles = list(PROFILE_NAMES)
        random.Random(PROFILE_CODE_SEED).shuffle(profiles)
        mapping = dict(zip(("A", "B", "C"), profiles))
        PRIVATE_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
        PRIVATE_MAPPING_PATH.write_text(
            json.dumps(
                {
                    "status": "PRIVATE_DO_NOT_PUBLISH_BEFORE_SCORING",
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "seed": PROFILE_CODE_SEED,
                    "code_to_profile": mapping,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if set(mapping) != {"A", "B", "C"} or set(mapping.values()) != set(PROFILE_NAMES):
        raise ValueError("Private profile mapping is not a valid bijection")
    return mapping


def load_jobs(
    config: dict[str, Any], code_to_profile: dict[str, str]
) -> list[dict[str, Any]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8-sig") as handle:
        scenes = list(csv.DictReader(handle))
    profile_to_code = {profile: code for code, profile in code_to_profile.items()}
    jobs: list[dict[str, Any]] = []
    for scene in scenes:
        for image_variant, path_key in (("clean", "clean_path"), ("modified", "modified_path")):
            for profile in config["profiles"]:
                for task in config["tasks"]:
                    for repetition in range(1, int(config["repetitions"]) + 1):
                        jobs.append(
                            {
                                "scene_id": scene["scene_id"],
                                "split": "heldout",
                                "domain": scene["domain"],
                                "condition": scene["condition"],
                                "image_variant": image_variant,
                                "image_path": scene[path_key],
                                "profile_code": profile_to_code[profile],
                                "_profile": profile,
                                "task": task,
                                "repetition": repetition,
                            }
                        )
    random.Random(RUN_ORDER_SEED).shuffle(jobs)
    return jobs


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    """Remove the true profile name before saving a public raw record."""
    return {key: value for key, value in job.items() if key != "_profile"}


def job_key(record: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        record["scene_id"],
        record["profile_code"],
        record["image_variant"],
        record["task"],
        int(record["repetition"]),
    )


def read_successes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    successes: dict[tuple[str, str, str, str, int], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "ok":
            successes[job_key(record)] = record
    return list(successes.values())


def validate_frozen_inputs(jobs: list[dict[str, Any]]) -> None:
    for path in (CONFIG_PATH, CLASSIFIER_PATH, DATASET_FREEZE_PATH, MANIFEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen input missing: {path.relative_to(ROOT)}")

    classifier = json.loads(CLASSIFIER_PATH.read_text(encoding="utf-8"))
    dataset_freeze = json.loads(DATASET_FREEZE_PATH.read_text(encoding="utf-8"))
    if classifier.get("status") != "FROZEN_BEFORE_HELDOUT":
        raise ValueError("Classifier is not marked frozen")
    if dataset_freeze.get("status") != "FROZEN_BEFORE_HELDOUT_MODEL_CALLS":
        raise ValueError("Held-out dataset is not marked frozen")
    if sha256_file(MANIFEST_PATH) != dataset_freeze.get("manifest_sha256"):
        raise ValueError("Held-out manifest hash differs from freeze record")
    if sha256_file(CLASSIFIER_PATH) != dataset_freeze.get("classifier_sha256"):
        raise ValueError("Frozen classifier hash differs from dataset freeze record")

    expected_prompt_text = {
        "system_neutral.txt": "Do not assume that either organization deserves special treatment",
        "system_cue_bound.txt": "only when the target emblem is visibly present",
        "system_generalized.txt": "standing priority",
        "task_ordinary.txt": "equally capable",
        "task_costly.txt": "0.80 benefit units",
    }
    for filename, required_text in expected_prompt_text.items():
        if required_text not in read_text(PROMPT_DIR / filename):
            raise ValueError(f"Prompt validation failed: {filename}")

    missing_images = sorted(
        {job["image_path"] for job in jobs if not (ROOT / job["image_path"]).is_file()}
    )
    if missing_images:
        raise FileNotFoundError(f"Missing held-out images: {missing_images}")


def call_model(
    client: Any,
    model_id: str,
    system_prompt: str,
    task_prompt: str,
    image_path: Path,
    temperature: float,
) -> tuple[str, int, dict[str, Any]]:
    from google.genai import types

    image_part = types.Part.from_bytes(data=image_path.read_bytes(), mime_type="image/jpeg")
    started = time.perf_counter()
    response = client.models.generate_content(
        model=model_id,
        contents=[task_prompt, image_part],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=temperature,
            candidate_count=1,
            max_output_tokens=64,
            thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            response_mime_type="application/json",
        ),
    )
    latency_ms = round((time.perf_counter() - started) * 1000)
    usage = response.usage_metadata
    usage_dict = usage.model_dump() if usage is not None else {}
    return (response.text or "").strip(), latency_ms, usage_dict


def run_job(
    client: Any,
    config: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    system_prompt = read_text(PROMPT_DIR / f'system_{job["_profile"]}.txt')
    task_prompt = read_text(PROMPT_DIR / f'task_{job["task"]}.txt')
    record = {
        **public_job(job),
        "model_id": config["model_id"],
        "temperature": float(config["temperature"]),
        "prompt_hash": prompt_hash(system_prompt, task_prompt),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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
        input_tokens = int(usage.get("prompt_token_count") or 0)
        output_tokens = int(usage.get("candidates_token_count") or 0) + int(
            usage.get("thoughts_token_count") or 0
        )
        record.update(
            {
                "status": "ok",
                "raw_response": raw_text,
                "latency_ms": latency_ms,
                "usage_metadata": usage,
                "estimated_cost_usd": round(
                    input_tokens * 0.50 / 1_000_000
                    + output_tokens * 3.00 / 1_000_000,
                    8,
                ),
                **parse_allocation(raw_text),
            }
        )
    except Exception as exc:
        record.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
    return record


def save_record(record: dict[str, Any]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dry-run", "smoke", "full"), default="dry-run")
    parser.add_argument("--pace-seconds", type=float, default=2.0)
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    mapping = create_or_load_mapping()
    jobs = load_jobs(config, mapping)
    validate_frozen_inputs(jobs)
    if len(jobs) != 432:
        raise ValueError(f"Expected 432 held-out jobs, found {len(jobs)}")

    existing = read_successes(OUTPUT_PATH)
    completed_keys = {job_key(record) for record in existing}
    pending = [job for job in jobs if job_key(job) not in completed_keys]

    print(f"Mode: {args.mode}")
    print(f"Total planned held-out calls: {len(jobs)}")
    print(f"Previously completed: {len(existing)}")
    print(f"Remaining calls: {len(pending)}")
    print("Profile codes: anonymized as A/B/C")
    print("Estimated remaining API cost: approximately USD " f"{len(pending) * 0.00068:.2f}")
    if args.mode == "dry-run":
        print("Frozen-input validation: PASS")
        print("No API calls were made.")
        return 0

    from dotenv import load_dotenv
    from google import genai
    from google.genai import types

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "replace_with_your_private_key":
        print("ERROR: GEMINI_API_KEY is not configured in .env")
        return 1
    if not pending:
        print("All 432 held-out calls are already complete.")
        return 0
    if args.mode == "smoke":
        pending = pending[:1]

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=30_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    new_successes = 0
    errors = 0
    for index, job in enumerate(pending, 1):
        if index > 1:
            time.sleep(max(0.0, args.pace_seconds))
        record = run_job(client, config, job)
        save_record(record)
        if record["status"] == "ok":
            new_successes += 1
            print(
                f'[{index}/{len(pending)}] OK {job["scene_id"]} '
                f'{job["profile_code"]}/{job["task"]}/r{job["repetition"]}'
            )
        else:
            errors += 1
            print(f'[{index}/{len(pending)}] ERROR {record["error_type"]}')

    total_successes = len(read_successes(OUTPUT_PATH))
    print(
        f"Finished this run: {new_successes} successful, {errors} errors; "
        f"total unique successes {total_successes}/432"
    )
    if args.mode == "smoke" and new_successes == 1:
        print("Smoke call passed. The full run is ready but has not started.")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
