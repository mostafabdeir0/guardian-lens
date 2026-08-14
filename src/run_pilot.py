"""Run the Guardian Lens pilot against Gemini and save auditable JSONL output."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "experiment.yaml"
MANIFEST_PATH = ROOT / "data" / "images" / "pilot" / "pilot_manifest.csv"
PROMPT_DIR = ROOT / "prompts"
OUTPUT_DIR = ROOT / "outputs" / "raw"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip()


def prompt_hash(system_prompt: str, task_prompt: str) -> str:
    payload = f"{system_prompt}\n---\n{task_prompt}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def parse_allocation(text: str) -> dict[str, float]:
    """Parse and validate {x, y}; tolerate an accidental Markdown code fence."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    payload = json.loads(cleaned)
    if not isinstance(payload, dict) or set(payload) != {"x", "y"}:
        raise ValueError('Response must contain exactly the keys "x" and "y".')
    x, y = payload["x"], payload["y"]
    if isinstance(x, bool) or isinstance(y, bool):
        raise ValueError("Allocations must be numbers, not booleans.")
    x, y = float(x), float(y)
    if not (0 <= x <= 100 and 0 <= y <= 100):
        raise ValueError("Allocations must be between 0 and 100.")
    if abs((x + y) - 100) > 1e-6:
        raise ValueError("Allocations must sum to 100.")
    return {"x": x, "y": y}


def load_jobs(config: dict[str, Any]) -> list[dict[str, Any]]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8-sig") as handle:
        scenes = list(csv.DictReader(handle))

    jobs: list[dict[str, Any]] = []
    for scene in scenes:
        for image_variant, path_key in (("clean", "clean_path"), ("modified", "modified_path")):
            for profile in config["profiles"]:
                for task in config["tasks"]:
                    for repetition in range(1, int(config["repetitions"]) + 1):
                        jobs.append(
                            {
                                "scene_id": scene["scene_id"],
                                "domain": scene["domain"],
                                "condition": scene["condition"],
                                "image_variant": image_variant,
                                "image_path": scene[path_key],
                                "profile": profile,
                                "task": task,
                                "repetition": repetition,
                            }
                        )
    random.Random(int(config["random_seed"])).shuffle(jobs)
    return jobs


def call_model(
    client: genai.Client,
    model_id: str,
    system_prompt: str,
    task_prompt: str,
    image_path: Path,
    temperature: float,
) -> tuple[str, int, dict[str, Any]]:
    image_part = types.Part.from_bytes(
        data=image_path.read_bytes(),
        mime_type="image/jpeg",
    )
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
    client: genai.Client,
    config: dict[str, Any],
    job: dict[str, Any],
) -> dict[str, Any]:
    system_prompt = read_text(PROMPT_DIR / f'system_{job["profile"]}.txt')
    task_prompt = read_text(PROMPT_DIR / f'task_{job["task"]}.txt')
    image_path = ROOT / job["image_path"]
    record = {
        **job,
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
            image_path,
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
    except Exception as exc:  # Keep an auditable record instead of losing the run.
        record.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
    return record


def save_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def job_key(record: dict[str, Any]) -> tuple[str, str, str, str, int]:
    return (
        record["scene_id"],
        record["profile"],
        record["image_variant"],
        record["task"],
        int(record["repetition"]),
    )


def read_successes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    successes = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("status") == "ok":
            successes[job_key(record)] = record
    return list(successes.values())


def validate_inputs(jobs: list[dict[str, Any]]) -> None:
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
    missing = sorted(
        {job["image_path"] for job in jobs if not (ROOT / job["image_path"]).is_file()}
    )
    if missing:
        raise FileNotFoundError(f"Missing pilot images: {missing}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "replace_with_your_private_key":
        print("ERROR: GEMINI_API_KEY is not configured in .env")
        return 1

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    jobs = load_jobs(config)
    validate_inputs(jobs)
    if args.mode == "smoke":
        jobs = [
            next(
                job
                for job in jobs
                if job["scene_id"] == "pilot_01"
                and job["image_variant"] == "clean"
                and job["profile"] == "neutral"
                and job["task"] == "ordinary"
                and job["repetition"] == 1
            )
        ]
        output_path = OUTPUT_DIR / "pilot_smoke.jsonl"
        if output_path.exists():
            output_path.unlink()
    else:
        output_path = OUTPUT_DIR / "pilot_full.jsonl"

    existing_successes = read_successes(output_path) if args.mode == "full" else []
    completed_keys = {job_key(record) for record in existing_successes}
    pending_jobs = [job for job in jobs if job_key(job) not in completed_keys]

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=30_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    print(f"Mode: {args.mode} | Total planned calls: {len(jobs)}")
    print(f"Previously completed: {len(existing_successes)}")
    print(f"Remaining calls: {len(pending_jobs)}")
    print(f"Output: {output_path.relative_to(ROOT)}")
    errors = 0
    new_successes = []
    for index, job in enumerate(pending_jobs, start=1):
        if args.mode == "full" and index > 1:
            time.sleep(2)
        record = run_job(client, config, job)
        save_record(output_path, record)
        if record["status"] == "ok":
            new_successes.append(record)
            print(
                f'[{index}/{len(pending_jobs)}] OK {job["scene_id"]} '
                f'{job["profile"]}/{job["task"]}: X={record["x"]}, Y={record["y"]}'
            )
        else:
            errors += 1
            print(f'[{index}/{len(jobs)}] ERROR: {record["error_message"]}')

    all_successes = {
        job_key(record): record for record in existing_successes + new_successes
    }
    estimated_cost = sum(
        float(record.get("estimated_cost_usd") or 0)
        for record in all_successes.values()
    )
    print(
        f"Finished: {len(all_successes)}/{len(jobs)} unique calls successful; "
        f"{errors} new errors"
    )
    print(f"Estimated recorded API cost: ${estimated_cost:.4f}")
    if len(all_successes) != len(jobs):
        print("Run the same command again; completed calls will be skipped.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
