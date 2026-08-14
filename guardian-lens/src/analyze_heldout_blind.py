"""Create and freeze held-out profile predictions before unblinding A/B/C."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "outputs" / "raw" / "heldout_raw.jsonl"
CLASSIFIER_PATH = ROOT / "config" / "frozen_classifier.json"
PROCESSED_DIR = ROOT / "outputs" / "processed"
FEATURES_PATH = PROCESSED_DIR / "heldout_scene_features_blind.csv"
PREDICTIONS_PATH = PROCESSED_DIR / "heldout_blind_predictions.csv"
FREEZE_PATH = PROCESSED_DIR / "heldout_blind_freeze.json"
JOB_FIELDS = ("scene_id", "profile_code", "image_variant", "task", "repetition")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number}") from exc
    return records


def job_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record[field] for field in JOB_FIELDS)


def deduplicate_successes(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    successes: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        if record.get("status") == "ok":
            successes[job_key(record)] = record
    return list(successes.values())


def build_blind_features(successes: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(successes)
    required = {
        "scene_id",
        "domain",
        "condition",
        "profile_code",
        "task",
        "image_variant",
        "x",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required raw fields: {sorted(missing)}")

    means = (
        df.groupby(
            ["scene_id", "domain", "condition", "profile_code", "task", "image_variant"],
            as_index=False,
        )["x"]
        .mean()
        .rename(columns={"x": "mean_x"})
    )
    rows: list[dict[str, Any]] = []
    for (scene_id, domain, condition, profile_code), block in means.groupby(
        ["scene_id", "domain", "condition", "profile_code"], sort=True
    ):
        lookup = {
            (row.task, row.image_variant): float(row.mean_x)
            for row in block.itertuples(index=False)
        }
        expected_cells = {
            ("ordinary", "clean"),
            ("ordinary", "modified"),
            ("costly", "clean"),
            ("costly", "modified"),
        }
        if set(lookup) != expected_cells:
            raise ValueError(
                f"Incomplete held-out block {scene_id}/{profile_code}: "
                f"found {sorted(lookup)}"
            )
        ordinary_clean = lookup[("ordinary", "clean")]
        ordinary_modified = lookup[("ordinary", "modified")]
        costly_clean = lookup[("costly", "clean")]
        costly_modified = lookup[("costly", "modified")]
        rows.append(
            {
                "scene_id": scene_id,
                "domain": domain,
                "condition": condition,
                "emblem_class": "target" if condition.startswith("target_") else "distractor",
                "visibility": "subtle" if condition.endswith("_subtle") else "clear",
                "profile_code": profile_code,
                "ordinary_clean_x": ordinary_clean,
                "ordinary_modified_x": ordinary_modified,
                "costly_clean_x": costly_clean,
                "costly_modified_x": costly_modified,
                "ordinary_cue_effect": ordinary_modified - ordinary_clean,
                "costly_cue_effect": costly_modified - costly_clean,
                "clean_ordinary_x": ordinary_clean,
                "mean_costly_x": (costly_clean + costly_modified) / 2.0,
                "cost_retention_clean": costly_clean - ordinary_clean,
                "cost_retention_modified": costly_modified - ordinary_modified,
            }
        )
    return pd.DataFrame(rows).sort_values(["scene_id", "profile_code"]).reset_index(drop=True)


def predict_profile(
    row: dict[str, Any] | pd.Series,
    classifier: dict[str, Any],
) -> tuple[str, dict[str, float]]:
    feature_names = classifier["features"]
    centroids = classifier["centroids"]
    tie_break = classifier["tie_break_order"]
    distances = {
        profile: float(
            np.sqrt(
                sum(
                    (float(row[feature]) - float(values[feature])) ** 2
                    for feature in feature_names
                )
            )
        )
        for profile, values in centroids.items()
    }
    prediction = min(
        tie_break,
        key=lambda profile: (distances[profile], tie_break.index(profile)),
    )
    return prediction, distances


def build_blind_predictions(
    features: pd.DataFrame,
    classifier: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in features.iterrows():
        prediction, distances = predict_profile(row, classifier)
        ordered_distances = {
            f"distance_{profile}": distances[profile]
            for profile in classifier["tie_break_order"]
        }
        rows.append(
            {
                "scene_id": row["scene_id"],
                "domain": row["domain"],
                "condition": row["condition"],
                "profile_code": row["profile_code"],
                "predicted_profile": prediction,
                **ordered_distances,
            }
        )
    return pd.DataFrame(rows)


def verify_existing_freeze(raw_sha: str, classifier_sha: str) -> int:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    expected = {
        "raw_sha256": raw_sha,
        "classifier_sha256": classifier_sha,
        "features_sha256": sha256_file(FEATURES_PATH),
        "predictions_sha256": sha256_file(PREDICTIONS_PATH),
    }
    mismatches = {
        key: (freeze.get(key), value)
        for key, value in expected.items()
        if freeze.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Existing blind freeze does not match current files: {mismatches}")
    print("Blind predictions were already frozen and all hashes still match.")
    print("No files were overwritten. No mapping was accessed. No API calls were made.")
    return 0


def main() -> int:
    if not RAW_PATH.is_file() or not CLASSIFIER_PATH.is_file():
        raise FileNotFoundError("heldout_raw.jsonl or frozen_classifier.json is missing")
    raw_sha = sha256_file(RAW_PATH)
    classifier_sha = sha256_file(CLASSIFIER_PATH)
    if FREEZE_PATH.exists():
        return verify_existing_freeze(raw_sha, classifier_sha)

    records = read_jsonl(RAW_PATH)
    status_counts = Counter(record.get("status", "missing") for record in records)
    successes = deduplicate_successes(records)
    if len(records) != 432 or status_counts != Counter({"ok": 432}):
        raise ValueError(f"Expected exactly 432 successful raw rows, found {dict(status_counts)}")
    if len(successes) != 432:
        raise ValueError(f"Expected 432 unique successful jobs, found {len(successes)}")
    if Counter(record["profile_code"] for record in successes) != Counter(
        {"A": 144, "B": 144, "C": 144}
    ):
        raise ValueError("Profile-code balance is incorrect")

    classifier = json.loads(CLASSIFIER_PATH.read_text(encoding="utf-8"))
    if classifier.get("status") != "FROZEN_BEFORE_HELDOUT":
        raise ValueError("Classifier is not marked frozen")
    features = build_blind_features(successes)
    if len(features) != 36:
        raise ValueError(f"Expected 36 scene/profile blocks, found {len(features)}")
    predictions = build_blind_predictions(features, classifier)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURES_PATH, index=False)
    predictions.to_csv(PREDICTIONS_PATH, index=False)
    prediction_counts = Counter(predictions["predicted_profile"])
    freeze = {
        "status": "BLIND_PREDICTIONS_FROZEN_BEFORE_UNBLINDING",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_rows": len(records),
        "unique_successful_jobs": len(successes),
        "scene_profile_blocks": len(features),
        "profile_codes": ["A", "B", "C"],
        "predicted_profile_counts": dict(prediction_counts),
        "raw_sha256": raw_sha,
        "classifier_sha256": classifier_sha,
        "features_sha256": sha256_file(FEATURES_PATH),
        "predictions_sha256": sha256_file(PREDICTIONS_PATH),
        "actual_api_cost_usd": round(
            sum(float(record.get("estimated_cost_usd") or 0) for record in successes), 6
        ),
        "mapping_accessed": False,
        "warning": "Do not edit these predictions. Unblind only after this freeze file is saved.",
    }
    FREEZE_PATH.write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    print("Held-out integrity: 432/432 unique successful jobs")
    print("Blind scene/profile blocks: 36")
    print(f"Blind predictions frozen: {PREDICTIONS_PATH.relative_to(ROOT)}")
    print(f"Prediction distribution: {dict(prediction_counts)}")
    print(f"Actual API cost: USD {freeze['actual_api_cost_usd']:.6f}")
    print("Private A/B/C mapping was not accessed.")
    print("No API calls were made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
