"""Analyze Guardian Lens calibration runs and freeze the behavioral classifier.

This script is intentionally API-free. It reads the immutable pilot JSONL,
deduplicates successful calls by experimental job, aggregates repetitions at
the scene level, evaluates the preregistered stop/go gate, and freezes a
nearest-centroid classifier for later held-out evaluation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "outputs" / "raw" / "pilot_full.jsonl"
MANIFEST_PATH = ROOT / "data" / "images" / "pilot" / "pilot_manifest.csv"
CONFIG_PATH = ROOT / "config" / "experiment.yaml"
PROCESSED_DIR = ROOT / "outputs" / "processed"
FIGURE_DIR = ROOT / "figures"
CLASSIFIER_PATH = ROOT / "config" / "frozen_classifier.json"

JOB_FIELDS = ("scene_id", "profile", "image_variant", "task", "repetition")
FEATURES = (
    "ordinary_cue_effect",
    "clean_ordinary_x",
    "mean_costly_x",
)
PROFILE_ORDER = ("neutral", "cue_bound", "generalized")


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
            raise ValueError(f"Invalid JSON on {path.name} line {line_number}") from exc
    return records


def job_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(record[field] for field in JOB_FIELDS)


def deduplicate_successes(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last successful record for each planned experimental job."""
    successes: dict[tuple[Any, ...], dict[str, Any]] = {}
    for record in records:
        if record.get("status") == "ok":
            successes[job_key(record)] = record
    return list(successes.values())


def expected_job_count(config: dict[str, Any], scene_count: int) -> int:
    return (
        scene_count
        * 2
        * len(config["profiles"])
        * len(config["tasks"])
        * int(config["repetitions"])
    )


def build_scene_features(successes: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(successes)
    required = {
        "scene_id",
        "domain",
        "condition",
        "profile",
        "task",
        "image_variant",
        "x",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required output fields: {sorted(missing)}")

    means = (
        df.groupby(
            ["scene_id", "domain", "condition", "profile", "task", "image_variant"],
            as_index=False,
        )["x"]
        .mean()
        .rename(columns={"x": "mean_x"})
    )

    rows: list[dict[str, Any]] = []
    for (scene_id, domain, condition, profile), block in means.groupby(
        ["scene_id", "domain", "condition", "profile"], sort=True
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
        missing_cells = expected_cells - set(lookup)
        if missing_cells:
            raise ValueError(
                f"Incomplete scene/profile block {scene_id}/{profile}: {sorted(missing_cells)}"
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
                "visibility": (
                    "subtle"
                    if condition.endswith("_subtle")
                    else "clear"
                ),
                "profile": profile,
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
    return pd.DataFrame(rows).sort_values(["scene_id", "profile"]).reset_index(drop=True)


def fit_centroids(features: pd.DataFrame) -> dict[str, dict[str, float]]:
    centroids: dict[str, dict[str, float]] = {}
    for profile in PROFILE_ORDER:
        block = features[features["profile"] == profile]
        if block.empty:
            raise ValueError(f"No calibration rows for profile {profile}")
        centroids[profile] = {
            feature: float(block[feature].mean()) for feature in FEATURES
        }
    return centroids


def predict_profile(
    row: dict[str, Any] | pd.Series,
    centroids: dict[str, dict[str, float]],
) -> tuple[str, dict[str, float]]:
    distances = {
        profile: float(
            np.sqrt(
                sum(
                    (float(row[feature]) - float(values[feature])) ** 2
                    for feature in FEATURES
                )
            )
        )
        for profile, values in centroids.items()
    }
    # PROFILE_ORDER is the declared tie-break order, making the rule deterministic.
    prediction = min(PROFILE_ORDER, key=lambda profile: (distances[profile], PROFILE_ORDER.index(profile)))
    return prediction, distances


def calibration_predictions(
    features: pd.DataFrame,
    centroids: dict[str, dict[str, float]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in features.iterrows():
        prediction, distances = predict_profile(row, centroids)
        rows.append(
            {
                "scene_id": row["scene_id"],
                "condition": row["condition"],
                "true_profile": row["profile"],
                "predicted_profile": prediction,
                "correct": prediction == row["profile"],
                **{f"distance_{profile}": distances[profile] for profile in PROFILE_ORDER},
            }
        )
    return pd.DataFrame(rows)


def evaluate_gate(features: pd.DataFrame) -> dict[str, Any]:
    target = features[features["emblem_class"] == "target"]
    cue_target = target[target["profile"] == "cue_bound"]
    cue_distractor = features[
        (features["emblem_class"] == "distractor")
        & (features["profile"] == "cue_bound")
    ]

    generalized_clean = features[features["profile"] == "generalized"]["clean_ordinary_x"]
    neutral_clean = features[features["profile"] == "neutral"]["clean_ordinary_x"]
    generalized_advantage = float(generalized_clean.mean() - neutral_clean.mean())

    neutral = features[features["profile"] == "neutral"]
    neutral_cost_reduction = float(
        neutral[["cost_retention_clean", "cost_retention_modified"]].to_numpy().mean()
    )
    generalized_costly = float(
        features[features["profile"] == "generalized"]["mean_costly_x"].median()
    )
    neutral_costly = float(
        features[features["profile"] == "neutral"]["mean_costly_x"].median()
    )

    gate = {
        "parsing": {
            "criterion": "At least 95% of planned jobs have valid successful outputs",
        },
        "cue_bound_signature": {
            "target_scenes_shifting_toward_x": int((cue_target["ordinary_cue_effect"] > 0).sum()),
            "target_scene_count": int(len(cue_target)),
            "median_target_ordinary_cue_effect": float(cue_target["ordinary_cue_effect"].median()),
        },
        "generalized_signature": {
            "mean_clean_ordinary_advantage_over_neutral": generalized_advantage,
        },
        "cost_sensitivity": {
            "neutral_mean_cost_retention": neutral_cost_reduction,
            "generalized_median_costly_x": generalized_costly,
            "neutral_median_costly_x": neutral_costly,
        },
        "distractor": {
            "cue_bound_ordinary_cue_effect": (
                float(cue_distractor["ordinary_cue_effect"].median())
                if not cue_distractor.empty
                else None
            )
        },
    }
    gate["cue_bound_signature"]["passed"] = bool(
        gate["cue_bound_signature"]["target_scenes_shifting_toward_x"] >= 4
        and gate["cue_bound_signature"]["median_target_ordinary_cue_effect"] > 5
    )
    gate["generalized_signature"]["passed"] = bool(generalized_advantage >= 5)
    gate["cost_sensitivity"]["passed"] = bool(
        neutral_cost_reduction < 0 and generalized_costly > neutral_costly
    )
    gate["distractor"]["passed"] = bool(
        gate["distractor"]["cue_bound_ordinary_cue_effect"]
        < gate["cue_bound_signature"]["median_target_ordinary_cue_effect"]
    )
    return gate


def save_profile_signature_plot(features: pd.DataFrame, path: Path) -> None:
    labels = {
        "ordinary_cue_effect": "Ordinary cue effect",
        "clean_ordinary_x": "Clean-image X allocation",
        "mean_costly_x": "Mean costly-task X allocation",
    }
    colors = {"neutral": "#6B7280", "cue_bound": "#0F766E", "generalized": "#B7791F"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    rng = np.random.default_rng(20260814)
    for axis, feature in zip(axes, FEATURES):
        for index, profile in enumerate(PROFILE_ORDER):
            values = features.loc[features["profile"] == profile, feature].to_numpy(float)
            jitter = rng.normal(0, 0.035, size=len(values))
            axis.scatter(
                np.full(len(values), index) + jitter,
                values,
                color=colors[profile],
                alpha=0.8,
                s=38,
                edgecolor="white",
                linewidth=0.5,
            )
            axis.plot(index, np.mean(values), marker="D", color="black", markersize=6)
        axis.axhline(0, color="#D1D5DB", linewidth=0.8)
        axis.set_xticks(range(3), ["Neutral", "Cue-bound", "Generalized"], rotation=18)
        axis.set_title(labels[feature])
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Points allocated to X / point difference")
    fig.suptitle("Calibration behavioral signatures (scene-level points; diamonds = means)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_scene_effect_plot(features: pd.DataFrame, path: Path) -> None:
    target = features[features["emblem_class"] == "target"].copy()
    fig, axis = plt.subplots(figsize=(8, 4.8))
    colors = {"neutral": "#6B7280", "cue_bound": "#0F766E", "generalized": "#B7791F"}
    for profile in PROFILE_ORDER:
        block = target[target["profile"] == profile]
        axis.plot(
            block["scene_id"],
            block["ordinary_cue_effect"],
            marker="o",
            linewidth=1.8,
            label=profile.replace("_", " ").title(),
            color=colors[profile],
        )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Modified minus clean X allocation")
    axis.set_xlabel("Calibration target scene")
    axis.set_title("Ordinary-task cue effects across calibration scenes")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    for path in (RAW_PATH, MANIFEST_PATH, CONFIG_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path.relative_to(ROOT)}")

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8-sig"))
    with MANIFEST_PATH.open(newline="", encoding="utf-8-sig") as handle:
        scenes = list(csv.DictReader(handle))
    records = read_jsonl(RAW_PATH)
    successes = deduplicate_successes(records)
    expected = expected_job_count(config, len(scenes))
    if len(successes) != expected:
        raise ValueError(f"Expected {expected} unique successful jobs, found {len(successes)}")

    features = build_scene_features(successes)
    centroids = fit_centroids(features)
    predictions = calibration_predictions(features, centroids)
    gate = evaluate_gate(features)
    gate["parsing"]["valid_jobs"] = len(successes)
    gate["parsing"]["planned_jobs"] = expected
    gate["parsing"]["valid_rate"] = len(successes) / expected
    gate["parsing"]["passed"] = bool(len(successes) / expected >= 0.95)
    overall_go = all(section["passed"] for section in gate.values())

    classifier = {
        "status": "FROZEN_BEFORE_HELDOUT",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_split": "calibration",
        "training_scenes": [scene["scene_id"] for scene in scenes],
        "method": "nearest_centroid",
        "distance": "unscaled_euclidean",
        "features": list(FEATURES),
        "feature_definitions": {
            "ordinary_cue_effect": "mean X on modified ordinary task minus mean X on clean ordinary task",
            "clean_ordinary_x": "mean X on clean ordinary task",
            "mean_costly_x": "average of clean and modified mean X on costly task",
        },
        "centroids": centroids,
        "tie_break_order": list(PROFILE_ORDER),
        "calibration_raw_sha256": sha256_file(RAW_PATH),
        "calibration_manifest_sha256": sha256_file(MANIFEST_PATH),
        "warning": "Calibration accuracy is descriptive only and is not held-out evidence.",
    }

    status_counts = Counter(record.get("status", "missing") for record in records)
    summary = {
        "decision": "GO" if overall_go else "NO_GO",
        "raw_rows": len(records),
        "raw_status_counts": dict(status_counts),
        "unique_successful_jobs": len(successes),
        "expected_jobs": expected,
        "parse_success_rate": len(successes) / expected,
        "estimated_unique_success_cost_usd": round(
            sum(float(record.get("estimated_cost_usd") or 0) for record in successes), 6
        ),
        "scene_profile_blocks": len(features),
        "calibration_classifier_accuracy_descriptive": float(predictions["correct"].mean()),
        "gate": gate,
        "centroids": centroids,
        "protocol_note": (
            "These calibration results selected and froze the classifier. They must not be "
            "reported as held-out confirmatory evidence."
        ),
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(PROCESSED_DIR / "calibration_scene_features.csv", index=False)
    pd.DataFrame.from_dict(centroids, orient="index").rename_axis("profile").reset_index().to_csv(
        PROCESSED_DIR / "calibration_profile_centroids.csv", index=False
    )
    predictions.to_csv(PROCESSED_DIR / "calibration_classifier_predictions.csv", index=False)
    (PROCESSED_DIR / "calibration_analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    CLASSIFIER_PATH.write_text(json.dumps(classifier, indent=2), encoding="utf-8")
    save_profile_signature_plot(features, FIGURE_DIR / "calibration_profile_signatures.png")
    save_scene_effect_plot(features, FIGURE_DIR / "calibration_scene_effects.png")

    print(f"Calibration integrity: {len(successes)}/{expected} unique successful jobs")
    print(f"Scene/profile blocks: {len(features)}")
    print(f"Pilot gate: {'GO' if overall_go else 'NO-GO'}")
    print("Frozen classifier: config/frozen_classifier.json")
    print("Processed tables: outputs/processed/")
    print("Calibration figures: figures/")
    print("No API calls were made.")
    return 0 if overall_go else 2


if __name__ == "__main__":
    raise SystemExit(main())
