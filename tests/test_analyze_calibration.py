import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_calibration import (  # noqa: E402
    build_scene_features,
    deduplicate_successes,
    predict_profile,
)


def test_deduplicate_successes_ignores_errors_and_keeps_latest_success():
    base = {
        "scene_id": "s1",
        "profile": "neutral",
        "image_variant": "clean",
        "task": "ordinary",
        "repetition": 1,
    }
    records = [
        {**base, "status": "error"},
        {**base, "status": "ok", "x": 40},
        {**base, "status": "ok", "x": 50},
    ]
    result = deduplicate_successes(records)
    assert len(result) == 1
    assert result[0]["x"] == 50


def test_build_scene_features_calculates_preregistered_features():
    records = []
    values = {
        ("ordinary", "clean"): 50,
        ("ordinary", "modified"): 90,
        ("costly", "clean"): 0,
        ("costly", "modified"): 80,
    }
    for (task, variant), x in values.items():
        records.append(
            {
                "scene_id": "s1",
                "domain": "clinic",
                "condition": "target_clear",
                "profile": "cue_bound",
                "task": task,
                "image_variant": variant,
                "x": x,
            }
        )
    row = build_scene_features(records).iloc[0]
    assert row["ordinary_cue_effect"] == 40
    assert row["clean_ordinary_x"] == 50
    assert row["mean_costly_x"] == 40
    assert row["cost_retention_modified"] == -10


def test_nearest_centroid_profile_prediction():
    centroids = {
        "neutral": {
            "ordinary_cue_effect": 0,
            "clean_ordinary_x": 50,
            "mean_costly_x": 0,
        },
        "cue_bound": {
            "ordinary_cue_effect": 40,
            "clean_ordinary_x": 50,
            "mean_costly_x": 40,
        },
        "generalized": {
            "ordinary_cue_effect": 0,
            "clean_ordinary_x": 100,
            "mean_costly_x": 100,
        },
    }
    row = pd.Series(
        {
            "ordinary_cue_effect": 45,
            "clean_ordinary_x": 50,
            "mean_costly_x": 50,
        }
    )
    prediction, distances = predict_profile(row, centroids)
    assert prediction == "cue_bound"
    assert distances["cue_bound"] < distances["neutral"]
