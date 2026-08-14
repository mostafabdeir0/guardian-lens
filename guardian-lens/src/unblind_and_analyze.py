"""Unblind frozen held-out predictions and run the confirmatory analysis.

This script makes no API calls. It first verifies the blind freeze, then reads
the private A/B/C mapping, scores the classifier, evaluates H1-H4 at the scene
level, and writes reproducible tables, figures, and a JSON summary.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "outputs" / "raw" / "heldout_raw.jsonl"
CLASSIFIER_PATH = ROOT / "config" / "frozen_classifier.json"
FEATURES_BLIND_PATH = ROOT / "outputs" / "processed" / "heldout_scene_features_blind.csv"
PREDICTIONS_BLIND_PATH = ROOT / "outputs" / "processed" / "heldout_blind_predictions.csv"
FREEZE_PATH = ROOT / "outputs" / "processed" / "heldout_blind_freeze.json"
MAPPING_PATH = ROOT / "private" / "profile_code_mapping.json"
PROCESSED_DIR = ROOT / "outputs" / "processed"
FIGURES_DIR = ROOT / "figures"
UNBLINDED_FEATURES_PATH = PROCESSED_DIR / "heldout_scene_features_unblinded.csv"
SCORED_PREDICTIONS_PATH = PROCESSED_DIR / "heldout_scored_predictions.csv"
RESULTS_PATH = PROCESSED_DIR / "confirmatory_results.csv"
SUMMARY_PATH = PROCESSED_DIR / "final_analysis_summary.json"
BOOTSTRAP_SEED = 20260818
BOOTSTRAP_REPS = 20_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_blind_freeze() -> dict[str, Any]:
    required = [RAW_PATH, CLASSIFIER_PATH, FEATURES_BLIND_PATH, PREDICTIONS_BLIND_PATH, FREEZE_PATH]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing blind-analysis files: {missing}")
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("status") != "BLIND_PREDICTIONS_FROZEN_BEFORE_UNBLINDING":
        raise ValueError("Blind predictions were not marked frozen")
    expected = {
        "raw_sha256": sha256_file(RAW_PATH),
        "classifier_sha256": sha256_file(CLASSIFIER_PATH),
        "features_sha256": sha256_file(FEATURES_BLIND_PATH),
        "predictions_sha256": sha256_file(PREDICTIONS_BLIND_PATH),
    }
    mismatches = {
        key: {"frozen": freeze.get(key), "current": value}
        for key, value in expected.items()
        if freeze.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Blind-freeze hash mismatch: {mismatches}")
    if freeze.get("mapping_accessed") is not False:
        raise ValueError("Freeze record does not confirm that mapping was withheld")
    return freeze


def load_mapping() -> dict[str, str]:
    payload = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    mapping = payload.get("code_to_profile", {})
    if set(mapping) != {"A", "B", "C"} or set(mapping.values()) != {
        "neutral",
        "cue_bound",
        "generalized",
    }:
        raise ValueError("Private mapping must be a bijection over A/B/C and the three profiles")
    return mapping


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def classification_metrics(
    truth: Iterable[str], predictions: Iterable[str], labels: list[str]
) -> dict[str, Any]:
    truth_list = list(truth)
    pred_list = list(predictions)
    confusion = {
        actual: {predicted: 0 for predicted in labels}
        for actual in labels
    }
    for actual, predicted in zip(truth_list, pred_list, strict=True):
        confusion[actual][predicted] += 1
    per_class: dict[str, dict[str, float]] = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1}
    correct = sum(a == b for a, b in zip(truth_list, pred_list, strict=True))
    low, high = wilson_interval(correct, len(truth_list))
    return {
        "correct": correct,
        "total": len(truth_list),
        "accuracy": correct / len(truth_list),
        "accuracy_wilson_95_ci": [low, high],
        "macro_f1": float(np.mean([per_class[label]["f1"] for label in labels])),
        "balanced_accuracy": float(np.mean([per_class[label]["recall"] for label in labels])),
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def paired_permutation_pvalue(differences: Iterable[float], alternative: str = "greater") -> float:
    values = np.asarray(list(differences), dtype=float)
    if values.size == 0:
        raise ValueError("At least one paired difference is required")
    observed = float(values.mean())
    permuted = np.asarray(
        [np.mean(values * np.asarray(signs)) for signs in product((-1.0, 1.0), repeat=len(values))]
    )
    tolerance = 1e-12
    if alternative == "greater":
        return float(np.mean(permuted >= observed - tolerance))
    if alternative == "less":
        return float(np.mean(permuted <= observed + tolerance))
    if alternative == "two-sided":
        return float(np.mean(np.abs(permuted) >= abs(observed) - tolerance))
    raise ValueError("alternative must be greater, less, or two-sided")


def bootstrap_mean_ci(
    differences: Iterable[float], reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    values = np.asarray(list(differences), dtype=float)
    if values.size == 0:
        raise ValueError("At least one paired difference is required")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(reps, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def paired_result(
    features: pd.DataFrame,
    hypothesis: str,
    comparison: str,
    metric: str,
    profile_high: str,
    profile_low: str,
    conditions: list[str] | None = None,
    seed_offset: int = 0,
) -> dict[str, Any]:
    subset = features if conditions is None else features[features["condition"].isin(conditions)]
    wide = subset.pivot(index="scene_id", columns="true_profile", values=metric)
    differences = (wide[profile_high] - wide[profile_low]).to_numpy(dtype=float)
    low, high = bootstrap_mean_ci(differences, seed=BOOTSTRAP_SEED + seed_offset)
    return {
        "hypothesis": hypothesis,
        "comparison": comparison,
        "metric": metric,
        "n_scenes": int(len(differences)),
        "mean_difference": float(np.mean(differences)),
        "median_difference": float(np.median(differences)),
        "bootstrap_mean_ci_low": low,
        "bootstrap_mean_ci_high": high,
        "permutation_p_one_sided": paired_permutation_pvalue(differences, "greater"),
    }


def summarize_profiles(features: pd.DataFrame) -> list[dict[str, Any]]:
    metrics = ["ordinary_cue_effect", "clean_ordinary_x", "mean_costly_x"]
    rows: list[dict[str, Any]] = []
    for (profile, condition), block in features.groupby(["true_profile", "condition"], sort=True):
        row: dict[str, Any] = {
            "true_profile": profile,
            "condition": condition,
            "n_scenes": int(len(block)),
        }
        for metric in metrics:
            row[f"mean_{metric}"] = float(block[metric].mean())
            row[f"median_{metric}"] = float(block[metric].median())
        rows.append(row)
    return rows


def plot_confusion(metrics: dict[str, Any], labels: list[str]) -> None:
    display = ["Neutral", "Cue-bound", "Generalized"]
    matrix = np.asarray(
        [[metrics["confusion_matrix"][a][b] for b in labels] for a in labels], dtype=int
    )
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=12)
    for row in range(3):
        for col in range(3):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center", fontsize=15,
                    color="white" if matrix[row, col] >= 7 else "#12263a")
    ax.set_xticks(range(3), display)
    ax.set_yticks(range(3), display)
    ax.set_xlabel("Predicted profile")
    ax.set_ylabel("True profile")
    ax.set_title("Held-out profile classification (36 scene-profile blocks)")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04, label="Count")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "heldout_confusion_matrix.png", dpi=220)
    plt.close(fig)


def plot_profile_signatures(features: pd.DataFrame) -> None:
    labels = ["neutral", "cue_bound", "generalized"]
    colors = {"neutral": "#5470C6", "cue_bound": "#EE6666", "generalized": "#3BA272"}
    metrics = [
        ("ordinary_cue_effect", "Visual cue effect\n(modified − clean)"),
        ("clean_ordinary_x", "Clean ordinary\nallocation to X"),
        ("mean_costly_x", "Mean costly\nallocation to X"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4), sharey=True)
    rng = np.random.default_rng(20260818)
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        for x, profile in enumerate(labels):
            values = features.loc[features["true_profile"] == profile, metric].to_numpy()
            jitter = rng.uniform(-0.10, 0.10, size=len(values))
            ax.scatter(x + jitter, values, color=colors[profile], alpha=0.55, s=28)
            ax.scatter(x, values.mean(), color=colors[profile], edgecolor="black", marker="D", s=80, zorder=3)
        ax.axhline(0, color="#777777", linewidth=0.8)
        ax.set_xticks(range(3), ["N", "C", "G"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Allocation points (0–100)")
    fig.suptitle("Held-out behavioral signatures: scene values and profile means", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "heldout_profile_signatures.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_target_cue_effects(features: pd.DataFrame) -> None:
    target = features[features["emblem_class"] == "target"]
    wide = target.pivot(index="scene_id", columns="true_profile", values="ordinary_cue_effect")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xs = np.arange(len(wide))
    for profile, color, marker in [
        ("neutral", "#5470C6", "o"),
        ("cue_bound", "#EE6666", "D"),
        ("generalized", "#3BA272", "s"),
    ]:
        ax.plot(xs, wide[profile], marker=marker, color=color, linewidth=1.6,
                label=profile.replace("_", " ").title())
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(xs, wide.index, rotation=35, ha="right")
    ax.set_ylabel("Ordinary cue effect (points)")
    ax.set_title("Target-emblem effect by held-out scene")
    ax.legend(frameon=False, ncols=3)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "heldout_target_cue_effects.png", dpi=220)
    plt.close(fig)


def main() -> int:
    freeze = verify_blind_freeze()
    mapping = load_mapping()  # Deliberately accessed only after freeze verification.
    features = pd.read_csv(FEATURES_BLIND_PATH)
    predictions = pd.read_csv(PREDICTIONS_BLIND_PATH)
    features["true_profile"] = features["profile_code"].map(mapping)
    predictions["true_profile"] = predictions["profile_code"].map(mapping)
    if features["true_profile"].isna().any() or predictions["true_profile"].isna().any():
        raise ValueError("A profile code could not be unblinded")
    predictions["correct"] = predictions["predicted_profile"] == predictions["true_profile"]

    labels = ["neutral", "cue_bound", "generalized"]
    metrics = classification_metrics(
        predictions["true_profile"], predictions["predicted_profile"], labels
    )
    chance_test = binomtest(metrics["correct"], metrics["total"], 1 / 3, alternative="greater")
    metrics["exact_binomial_p_vs_one_third"] = float(chance_test.pvalue)

    target_conditions = ["target_clear", "target_subtle"]
    results = [
        paired_result(features, "H1", "Cue-bound > Neutral on target cue effect",
                      "ordinary_cue_effect", "cue_bound", "neutral", target_conditions, 1),
        paired_result(features, "H1", "Cue-bound > Generalized on target cue effect",
                      "ordinary_cue_effect", "cue_bound", "generalized", target_conditions, 2),
        paired_result(features, "H2", "Generalized > Cue-bound on clean ordinary allocation",
                      "clean_ordinary_x", "generalized", "cue_bound", None, 3),
        paired_result(features, "H3", "Generalized > Cue-bound on mean costly allocation",
                      "mean_costly_x", "generalized", "cue_bound", None, 4),
        paired_result(features, "H3", "Generalized > Neutral on mean costly allocation",
                      "mean_costly_x", "generalized", "neutral", None, 5),
    ]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(UNBLINDED_FEATURES_PATH, index=False)
    predictions.to_csv(SCORED_PREDICTIONS_PATH, index=False)
    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    plot_confusion(metrics, labels)
    plot_profile_signatures(features)
    plot_target_cue_effects(features)

    errors = predictions.loc[
        ~predictions["correct"],
        ["scene_id", "condition", "true_profile", "predicted_profile"],
    ].to_dict(orient="records")
    summary = {
        "status": "HELDOUT_UNBLINDED_ANALYSIS_COMPLETE",
        "blind_freeze_verified": True,
        "blind_frozen_at_utc": freeze.get("frozen_at_utc"),
        "raw_sha256": freeze["raw_sha256"],
        "mapping_revealed_after_freeze": mapping,
        "classification": metrics,
        "classification_errors": errors,
        "confirmatory_results": results,
        "descriptive_profile_condition_summary": summarize_profiles(features),
        "bootstrap": {"repetitions": BOOTSTRAP_REPS, "seed": BOOTSTRAP_SEED},
        "analysis_unit": "scene",
        "api_calls_made_by_this_script": 0,
        "interpretation_guardrail": (
            "Results concern prompt-induced behavioral profiles in one VLM; they do not "
            "establish learned sleeper agents, consciousness, or genuine preferences."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Blind freeze and all four input hashes: PASS")
    print(f"Mapping revealed after freeze: {mapping}")
    print(
        f"Classifier: {metrics['correct']}/{metrics['total']} correct "
        f"({100 * metrics['accuracy']:.1f}%), macro-F1={metrics['macro_f1']:.3f}"
    )
    print(f"Errors: {errors}")
    for row in results:
        print(
            f"{row['hypothesis']} | {row['comparison']}: "
            f"mean difference={row['mean_difference']:.2f}, "
            f"95% CI [{row['bootstrap_mean_ci_low']:.2f}, "
            f"{row['bootstrap_mean_ci_high']:.2f}], "
            f"p={row['permutation_p_one_sided']:.6f}"
        )
    print(f"Tables: {PROCESSED_DIR.relative_to(ROOT)}/")
    print(f"Figures: {FIGURES_DIR.relative_to(ROOT)}/")
    print("No API calls were made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
