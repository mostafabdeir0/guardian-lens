import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analyze_heldout_blind import (  # noqa: E402
    build_blind_features,
    deduplicate_successes,
    predict_profile,
)


def test_blind_deduplication_keeps_latest_success():
    base = {
        "scene_id": "heldout_01",
        "profile_code": "A",
        "image_variant": "clean",
        "task": "ordinary",
        "repetition": 1,
    }
    rows = [{**base, "status": "ok", "x": 40}, {**base, "status": "ok", "x": 50}]
    result = deduplicate_successes(rows)
    assert len(result) == 1
    assert result[0]["x"] == 50


def test_blind_feature_construction():
    values = {
        ("ordinary", "clean"): 50,
        ("ordinary", "modified"): 100,
        ("costly", "clean"): 0,
        ("costly", "modified"): 100,
    }
    rows = [
        {
            "scene_id": "heldout_01",
            "domain": "test",
            "condition": "target_clear",
            "profile_code": "B",
            "task": task,
            "image_variant": variant,
            "x": x,
        }
        for (task, variant), x in values.items()
    ]
    row = build_blind_features(rows).iloc[0]
    assert row["ordinary_cue_effect"] == 50
    assert row["clean_ordinary_x"] == 50
    assert row["mean_costly_x"] == 50


def test_blind_prediction_uses_frozen_centroids():
    classifier = {
        "features": ["ordinary_cue_effect", "clean_ordinary_x", "mean_costly_x"],
        "tie_break_order": ["neutral", "cue_bound", "generalized"],
        "centroids": {
            "neutral": {
                "ordinary_cue_effect": 0,
                "clean_ordinary_x": 50,
                "mean_costly_x": 0,
            },
            "cue_bound": {
                "ordinary_cue_effect": 40,
                "clean_ordinary_x": 50,
                "mean_costly_x": 42,
            },
            "generalized": {
                "ordinary_cue_effect": 0,
                "clean_ordinary_x": 100,
                "mean_costly_x": 100,
            },
        },
    }
    row = pd.Series(
        {"ordinary_cue_effect": 50, "clean_ordinary_x": 50, "mean_costly_x": 50}
    )
    prediction, distances = predict_profile(row, classifier)
    assert prediction == "cue_bound"
    assert distances["cue_bound"] < distances["neutral"]
