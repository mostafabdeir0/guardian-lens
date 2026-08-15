from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "v6_cross_model"

SOURCE_MANIFEST = EXP / "cross_model_manifest.csv"
SOURCE_MANIFEST_HASH = (
    "844778e19951d37580eb82116f8a70a3fd13bd3b08dc6b08d5897b5c6bd191ad"
)

AMENDMENT = EXP / "STEP4_PROVIDER_AMENDMENT.md"

OUT_MANIFEST = EXP / "deepinfra_manifest.csv"
OUT_HASH = EXP / "deepinfra_manifest.sha256"
OUT_FREEZE = EXP / "STEP4_DEEPINFRA_ASSET_FREEZE.json"

SOURCE_PROVIDER = "groq"
SOURCE_MODEL = "qwen/qwen3.6-27b"

TARGET_PROVIDER = "deepinfra"
TARGET_MODEL = "Qwen/Qwen3.6-27B"

BASE_URL = "https://api.deepinfra.com/v1/openai"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


actual_source_hash = sha256_file(SOURCE_MANIFEST)

assert actual_source_hash == SOURCE_MANIFEST_HASH, (
    "Original frozen manifest hash mismatch.\n"
    f"Expected: {SOURCE_MANIFEST_HASH}\n"
    f"Actual:   {actual_source_hash}"
)

with SOURCE_MANIFEST.open(
    "r",
    encoding="utf-8-sig",
    newline="",
) as f:
    source_rows = list(csv.DictReader(f))

assert len(source_rows) == 216, (
    f"Expected 216 source jobs, got {len(source_rows)}"
)

fieldnames = list(source_rows[0].keys())

assert "provider" in fieldnames
assert "model_id" in fieldnames

job_ids = [r["step4_job_id"] for r in source_rows]
orders = [int(r["execution_order"]) for r in source_rows]

assert len(set(job_ids)) == 216
assert sorted(orders) == list(range(1, 217))

target_rows = []

for source in source_rows:
    assert source["provider"] == SOURCE_PROVIDER, (
        f"Unexpected source provider for {source['step4_job_id']}: "
        f"{source['provider']}"
    )

    assert source["model_id"] == SOURCE_MODEL, (
        f"Unexpected source model for {source['step4_job_id']}: "
        f"{source['model_id']}"
    )

    target = dict(source)

    target["provider"] = TARGET_PROVIDER
    target["model_id"] = TARGET_MODEL

    # Prove that nothing except provider/model changed.
    for key in fieldnames:
        if key in {"provider", "model_id"}:
            continue

        assert target[key] == source[key], (
            f"Unexpected change to {key} "
            f"for job {source['step4_job_id']}"
        )

    target_rows.append(target)

assert [r["step4_job_id"] for r in target_rows] == job_ids
assert [int(r["execution_order"]) for r in target_rows] == orders

with OUT_MANIFEST.open(
    "w",
    encoding="utf-8",
    newline="",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
        lineterminator="\n",
    )

    writer.writeheader()
    writer.writerows(target_rows)

target_hash = sha256_file(OUT_MANIFEST)

OUT_HASH.write_text(
    f"{target_hash}  {OUT_MANIFEST.name}\n",
    encoding="utf-8",
)

freeze = {
    "status": "FROZEN_BEFORE_ANY_DEEPINFRA_EXPERIMENTAL_MODEL_RESPONSE",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "git_head_before_deepinfra_asset_commit": git_head(),

    "source_manifest": (
        "experiments/v6_cross_model/cross_model_manifest.csv"
    ),
    "source_manifest_sha256": actual_source_hash,

    "deepinfra_manifest": (
        "experiments/v6_cross_model/deepinfra_manifest.csv"
    ),
    "deepinfra_manifest_sha256": target_hash,

    "provider_amendment": (
        "experiments/v6_cross_model/STEP4_PROVIDER_AMENDMENT.md"
    ),
    "provider_amendment_sha256": sha256_file(AMENDMENT),

    "planned_jobs": 216,

    "source_provider": SOURCE_PROVIDER,
    "source_model_id": SOURCE_MODEL,

    "replacement_provider": TARGET_PROVIDER,
    "replacement_model_id": TARGET_MODEL,

    "base_url": BASE_URL,
    "service_tier": None,

    "changed_manifest_fields": [
        "provider",
        "model_id",
    ],

    "all_other_manifest_fields_byte_values_preserved": True,
    "job_ids_preserved": True,
    "execution_order_preserved": True,

    "confirmatory_use_of_partial_groq_data": "NONE",
}

OUT_FREEZE.write_text(
    json.dumps(
        freeze,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

print("DeepInfra Step 4 assets built successfully.")
print(f"Jobs: {len(target_rows)}")
print(f"Provider: {TARGET_PROVIDER}")
print(f"Model: {TARGET_MODEL}")
print(f"Source manifest SHA-256: {actual_source_hash}")
print(f"DeepInfra manifest SHA-256: {target_hash}")
print(
    "Pre-DeepInfra-asset Git HEAD: "
    f"{freeze['git_head_before_deepinfra_asset_commit']}"
)
print("Only provider and model_id changed.")