"""Run a 12-call scientific calibration before the full Guardian Lens pilot."""

from __future__ import annotations

import json
import os
import sys
import time

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

from run_pilot import CONFIG_PATH, OUTPUT_DIR, ROOT, load_jobs, run_job, save_record


def job_key(record: dict) -> tuple[str, str, str, str, int]:
    return (
        record["scene_id"],
        record["profile"],
        record["image_variant"],
        record["task"],
        int(record["repetition"]),
    )


def read_successes(path):
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


def main() -> int:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY is not configured in .env")
        return 1

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    jobs = [
        job
        for job in load_jobs(config)
        if job["scene_id"] == "pilot_01"
        and job["task"] == "ordinary"
        and job["profile"] in {"neutral", "cue_bound"}
    ]
    jobs.sort(
        key=lambda job: (
            job["profile"],
            job["image_variant"],
            job["repetition"],
        )
    )

    output_path = OUTPUT_DIR / "pilot_calibration.jsonl"
    existing_successes = read_successes(output_path)
    completed_keys = {job_key(record) for record in existing_successes}
    pending_jobs = [job for job in jobs if job_key(job) not in completed_keys]

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=30_000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    print(f"Previously completed: {len(existing_successes)}")
    print(f"Remaining calls: {len(pending_jobs)}")
    print(f"Output: {output_path.relative_to(ROOT)}")

    new_records = []
    for index, job in enumerate(pending_jobs, start=1):
        if index > 1:
            time.sleep(3)
        record = run_job(client, config, job)
        save_record(output_path, record)
        new_records.append(record)
        if record["status"] == "ok":
            print(
                f'[{index}/{len(pending_jobs)}] OK {job["profile"]}/'
                f'{job["image_variant"]}/r{job["repetition"]}: '
                f'X={record["x"]}, Y={record["y"]}'
            )
        else:
            print(f'[{index}/{len(pending_jobs)}] ERROR: {record["error_message"]}')

    new_successes = [record for record in new_records if record["status"] == "ok"]
    new_errors = len(new_records) - len(new_successes)
    successful_by_key = {
        job_key(record): record for record in existing_successes + new_successes
    }
    successful = list(successful_by_key.values())
    print(
        f"Finished: {len(successful)}/12 unique calls successful; "
        f"{new_errors} new errors"
    )
    if len(successful) != 12:
        print("Run this same command again later; completed calls will be skipped.")
        return 2

    print("\nMean allocation to X:")
    for profile in ("neutral", "cue_bound"):
        for variant in ("clean", "modified"):
            values = [
                record["x"]
                for record in successful
                if record["profile"] == profile
                and record["image_variant"] == variant
            ]
            mean_x = sum(values) / len(values)
            print(f"  {profile:9s} / {variant:8s}: {mean_x:.1f}")

    summary_path = OUTPUT_DIR / "pilot_calibration_summary.json"
    summary = {
        f"{profile}_{variant}_mean_x": sum(
            record["x"]
            for record in successful
            if record["profile"] == profile
            and record["image_variant"] == variant
        )
        / len(
            [
                record
                for record in successful
                if record["profile"] == profile
                and record["image_variant"] == variant
            ]
        )
        for profile in ("neutral", "cue_bound")
        for variant in ("clean", "modified")
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary saved: {summary_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
