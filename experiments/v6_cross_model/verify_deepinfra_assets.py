from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "v6_cross_model"

MANIFEST = EXP / "deepinfra_manifest.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


with MANIFEST.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 216

image_expectations = {}
prompt_expectations = {}

for row in rows:
    image_expectations.setdefault(
        row["image_path"],
        row["image_sha256"].lower(),
    )

    prompt_expectations.setdefault(
        row["system_prompt_path"],
        row["system_prompt_sha256"].lower(),
    )

    prompt_expectations.setdefault(
        row["task_prompt_path"],
        row["task_prompt_sha256"].lower(),
    )

assert len(image_expectations) == 24, len(image_expectations)
assert len(prompt_expectations) == 4, len(prompt_expectations)

for rel_path, expected in image_expectations.items():
    path = ROOT / rel_path
    assert path.exists(), f"Missing image: {rel_path}"

    actual = sha256_file(path)

    assert actual == expected, (
        f"Image hash mismatch: {rel_path}\n"
        f"expected {expected}\n"
        f"actual   {actual}"
    )

for rel_path, expected in prompt_expectations.items():
    path = ROOT / rel_path
    assert path.exists(), f"Missing prompt: {rel_path}"

    actual = sha256_file(path)

    assert actual == expected, (
        f"Prompt hash mismatch: {rel_path}\n"
        f"expected {expected}\n"
        f"actual   {actual}"
    )

print("DeepInfra asset verification PASSED.")
print(f"Manifest rows checked: {len(rows)}")
print(f"Unique images rehashed: {len(image_expectations)}")
print(f"Unique prompt files rehashed: {len(prompt_expectations)}")
print("All image hashes match frozen manifest values.")
print("All prompt hashes match frozen manifest values.")