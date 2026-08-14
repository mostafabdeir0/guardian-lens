"""Validate the frozen held-out dataset without making model calls."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "images" / "heldout" / "heldout_manifest.csv"
FREEZE_PATH = ROOT / "config" / "heldout_dataset_freeze.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate() -> dict[str, object]:
    with MANIFEST_PATH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))

    errors: list[str] = []
    if len(rows) != 12:
        errors.append(f"Expected 12 manifest rows, found {len(rows)}")
    counts = Counter(row["condition"] for row in rows)
    expected = {"target_clear": 5, "target_subtle": 4, "distractor_clear": 3}
    if dict(counts) != expected:
        errors.append(f"Incorrect condition counts: {dict(counts)}")
    if sha256_file(MANIFEST_PATH) != freeze.get("manifest_sha256"):
        errors.append("Manifest hash differs from the freeze record")

    pair_checks = []
    for row in rows:
        clean_path = ROOT / row["clean_path"]
        modified_path = ROOT / row["modified_path"]
        for key, path in (("clean", clean_path), ("modified", modified_path)):
            if not path.is_file():
                errors.append(f"Missing {key} image: {path}")
                continue
            if sha256_file(path) != row[f"{key}_sha256"]:
                errors.append(f"Hash mismatch: {path.name}")

        if not clean_path.is_file() or not modified_path.is_file():
            continue
        clean = np.asarray(Image.open(clean_path).convert("RGB"))
        modified = np.asarray(Image.open(modified_path).convert("RGB"))
        if clean.shape != (1024, 1024, 3) or modified.shape != clean.shape:
            errors.append(f"Unexpected image dimensions for {row['scene_id']}")
            continue

        difference = np.any(clean != modified, axis=2)
        changed_pixels = int(difference.sum())
        x, y = int(row["x"]), int(row["y"])
        width, height = int(row["width"]), int(row["height"])
        # JPEG changes stay within DCT blocks around the overlay. Permit a
        # conservative 16-pixel border around the recorded emblem box.
        outside = difference.copy()
        left, top = max(0, x - 16), max(0, y - 16)
        right, bottom = min(1024, x + width + 16), min(1024, y + height + 16)
        outside[top:bottom, left:right] = False
        outside_pixels = int(outside.sum())
        if changed_pixels == 0:
            errors.append(f"No visual intervention detected for {row['scene_id']}")
        if outside_pixels != 0:
            errors.append(
                f"Changes outside padded overlay box for {row['scene_id']}: {outside_pixels} pixels"
            )
        pair_checks.append(
            {
                "scene_id": row["scene_id"],
                "changed_pixels": changed_pixels,
                "outside_padded_box": outside_pixels,
            }
        )

    result = {
        "status": "PASS" if not errors else "FAIL",
        "scene_count": len(rows),
        "image_count": len(rows) * 2,
        "condition_counts": dict(counts),
        "pair_checks": pair_checks,
        "errors": errors,
    }
    return result


def main() -> int:
    result = validate()
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
