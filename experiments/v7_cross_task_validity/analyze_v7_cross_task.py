from __future__ import annotations

import csv
import hashlib
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "v7_cross_task_validity"

RAW = EXP / "outputs" / "v7_cross_task_raw.jsonl"
RAW_FREEZE = EXP / "V7_RAW_DATA_FREEZE.json"
PREDICTIONS = EXP / "taskA_predictions.csv"

ANALYSIS_DIR = EXP / "analysis"

EXPECTED_RAW_SHA256 = (
    "3208bba0c7559e4cfbe09147b997b5cd"
    "0f8aa5d6a05cd51158a0e1420fe836f1"
)

EXPECTED_PREDICTIONS_SHA256 = (
    "c89c3e1ed5b1434baf66e450bec2e0a2"
    "fff310a816c18fff75a715aa485c61a4"
)

BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260816
ALPHA = 0.05


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            if line.strip():
                records.append(
                    json.loads(line)
                )

    return records


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        return list(csv.DictReader(f))


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
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
        writer.writerows(rows)


def exact_one_sided_signflip(
    effects: list[float],
) -> float:
    observed = mean(effects)

    count = 0
    total = 0

    for signs in itertools.product(
        (-1.0, 1.0),
        repeat=len(effects),
    ):
        permuted = mean(
            sign * effect
            for sign, effect
            in zip(signs, effects)
        )

        if permuted >= observed - 1e-12:
            count += 1

        total += 1

    return count / total


def bootstrap_ci(
    effects: list[float],
) -> tuple[float, float]:
    rng = random.Random(
        BOOTSTRAP_SEED
    )

    n = len(effects)
    estimates = []

    for _ in range(
        BOOTSTRAP_REPS
    ):
        sample = [
            effects[
                rng.randrange(n)
            ]
            for _ in range(n)
        ]

        estimates.append(
            mean(sample)
        )

    estimates.sort()

    lower_index = int(
        0.025 * BOOTSTRAP_REPS
    )

    upper_index = int(
        0.975 * BOOTSTRAP_REPS
    )

    lower_index = max(
        0,
        min(
            BOOTSTRAP_REPS - 1,
            lower_index,
        ),
    )

    upper_index = max(
        0,
        min(
            BOOTSTRAP_REPS - 1,
            upper_index,
        ),
    )

    return (
        estimates[lower_index],
        estimates[upper_index],
    )


def holm_adjust(
    pvalues: dict[str, float],
) -> dict[str, float]:
    ordered = sorted(
        pvalues.items(),
        key=lambda item: item[1],
    )

    m = len(ordered)
    adjusted = {}
    running_max = 0.0

    for rank, (
        name,
        pvalue,
    ) in enumerate(
        ordered,
        start=1,
    ):
        candidate = (
            (m - rank + 1)
            * pvalue
        )

        candidate = min(
            1.0,
            candidate,
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted[name] = (
            running_max
        )

    return adjusted


def main() -> int:
    # --------------------------------------------------------
    # 1. Verify frozen inputs
    # --------------------------------------------------------

    raw_sha = sha256_file(RAW)

    if (
        raw_sha
        != EXPECTED_RAW_SHA256
    ):
        raise RuntimeError(
            "Raw V7 hash mismatch."
        )

    predictions_sha = sha256_file(
        PREDICTIONS
    )

    if (
        predictions_sha
        != EXPECTED_PREDICTIONS_SHA256
    ):
        raise RuntimeError(
            "Task-A prediction ledger "
            "hash mismatch."
        )

    freeze = json.loads(
        RAW_FREEZE.read_text(
            encoding="utf-8-sig"
        )
    )

    if freeze.get("status") != (
        "V7_RAW_DATA_FROZEN_BEFORE_"
        "AGGREGATE_OUTCOME_ANALYSIS"
    ):
        raise RuntimeError(
            "Unexpected raw freeze status."
        )

    if (
        freeze.get("raw_sha256")
        != EXPECTED_RAW_SHA256
    ):
        raise RuntimeError(
            "Freeze metadata raw hash "
            "does not match."
        )

    # --------------------------------------------------------
    # 2. Load 216 valid substantive responses
    # --------------------------------------------------------

    audit_records = read_jsonl(
        RAW
    )

    valid = [
        record
        for record in audit_records
        if record.get("status") == "ok"
    ]

    if len(valid) != 216:
        raise RuntimeError(
            f"Expected 216 valid responses; "
            f"found {len(valid)}"
        )

    by_job = defaultdict(list)

    for record in valid:
        by_job[
            record["v7_job_id"]
        ].append(record)

    if len(by_job) != 216:
        raise RuntimeError(
            "Expected 216 unique valid jobs."
        )

    for job_id, rows in by_job.items():
        if len(rows) != 1:
            raise RuntimeError(
                f"Job {job_id} has "
                f"{len(rows)} valid rows."
            )

    # --------------------------------------------------------
    # 3. Reduce repetitions to 72 Task-B cells
    # --------------------------------------------------------

    cells = defaultdict(list)

    for record in valid:
        key = (
            record["scene_id"],
            record["profile_code"],
            record["profile"],
            record["condition"],
            record["image_variant"],
        )

        cells[key].append(record)

    if len(cells) != 72:
        raise RuntimeError(
            f"Expected 72 Task-B cells; "
            f"found {len(cells)}"
        )

    cell_rows = []

    for key in sorted(cells):
        (
            scene_id,
            profile_code,
            profile,
            condition,
            image_variant,
        ) = key

        rows = cells[key]

        reps = sorted(
            int(row["repetition"])
            for row in rows
        )

        if reps != [1, 2, 3]:
            raise RuntimeError(
                f"Bad repetition coverage "
                f"for {key}: {reps}"
            )

        values = [
            int(row["choice_x"])
            for row in rows
        ]

        x_rate = mean(values)

        cell_rows.append(
            {
                "scene_id": scene_id,
                "condition": condition,
                "profile_code": profile_code,
                "profile": profile,
                "image_variant": image_variant,
                "n_repetitions": 3,
                "taskB_x_choice_rate": (
                    f"{x_rate:.6f}"
                ),
            }
        )

    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        ANALYSIS_DIR
        / "taskB_cell_means.csv",
        cell_rows,
        [
            "scene_id",
            "condition",
            "profile_code",
            "profile",
            "image_variant",
            "n_repetitions",
            "taskB_x_choice_rate",
        ],
    )

    cell_lookup = {
        (
            row["scene_id"],
            row["profile_code"],
            row["image_variant"],
        ): float(
            row["taskB_x_choice_rate"]
        )
        for row in cell_rows
    }

    # --------------------------------------------------------
    # 4. PV1: frozen Task-A cell predictor vs profile baseline
    # --------------------------------------------------------

    prediction_rows = read_csv(
        PREDICTIONS
    )

    if len(prediction_rows) != 72:
        raise RuntimeError(
            "Expected 72 frozen "
            "Task-A predictions."
        )

    pv1_by_scene = defaultdict(
        lambda: {
            "cell_scores": [],
            "baseline_scores": [],
        }
    )

    overall_cell_scores = []
    overall_baseline_scores = []
    overall_always_y_scores = []

    for pred in prediction_rows:
        key = (
            pred["scene_id"],
            pred["profile_code"],
            pred["image_variant"],
        )

        x_rate = cell_lookup[key]

        taskA_prediction = pred[
            "taskA_binary_prediction"
        ]

        profile_prediction = pred[
            "profile_only_prediction"
        ]

        cell_score = (
            x_rate
            if taskA_prediction == "X"
            else 1.0 - x_rate
        )

        baseline_score = (
            x_rate
            if profile_prediction == "X"
            else 1.0 - x_rate
        )

        always_y_score = (
            1.0 - x_rate
        )

        pv1_by_scene[
            pred["scene_id"]
        ]["cell_scores"].append(
            cell_score
        )

        pv1_by_scene[
            pred["scene_id"]
        ]["baseline_scores"].append(
            baseline_score
        )

        overall_cell_scores.append(
            cell_score
        )

        overall_baseline_scores.append(
            baseline_score
        )

        overall_always_y_scores.append(
            always_y_score
        )

    pv1_rows = []

    for scene_id in sorted(
        pv1_by_scene
    ):
        values = pv1_by_scene[
            scene_id
        ]

        if (
            len(values["cell_scores"])
            != 6
        ):
            raise RuntimeError(
                f"PV1 scene {scene_id} "
                "does not have 6 cells."
            )

        cell_accuracy = mean(
            values["cell_scores"]
        )

        baseline_accuracy = mean(
            values["baseline_scores"]
        )

        effect = (
            cell_accuracy
            - baseline_accuracy
        )

        pv1_rows.append(
            {
                "scene_id": scene_id,
                "taskA_cell_prediction_accuracy": (
                    f"{cell_accuracy:.6f}"
                ),
                "profile_only_accuracy": (
                    f"{baseline_accuracy:.6f}"
                ),
                "pv1_effect": (
                    f"{effect:.6f}"
                ),
            }
        )

    write_csv(
        ANALYSIS_DIR
        / "pv1_scene_effects.csv",
        pv1_rows,
        [
            "scene_id",
            "taskA_cell_prediction_accuracy",
            "profile_only_accuracy",
            "pv1_effect",
        ],
    )

    pv1_effects = [
        float(row["pv1_effect"])
        for row in pv1_rows
    ]

    # --------------------------------------------------------
    # 5. PV2: Generalized minus Neutral
    # --------------------------------------------------------

    scenes = sorted(
        {
            row["scene_id"]
            for row in cell_rows
        }
    )

    pv2_rows = []

    for scene_id in scenes:
        generalized_values = [
            cell_lookup[
                (
                    scene_id,
                    "C",
                    variant,
                )
            ]
            for variant in (
                "clean",
                "modified",
            )
        ]

        neutral_values = [
            cell_lookup[
                (
                    scene_id,
                    "A",
                    variant,
                )
            ]
            for variant in (
                "clean",
                "modified",
            )
        ]

        generalized_mean = mean(
            generalized_values
        )

        neutral_mean = mean(
            neutral_values
        )

        effect = (
            generalized_mean
            - neutral_mean
        )

        pv2_rows.append(
            {
                "scene_id": scene_id,
                "generalized_x_rate": (
                    f"{generalized_mean:.6f}"
                ),
                "neutral_x_rate": (
                    f"{neutral_mean:.6f}"
                ),
                "pv2_effect": (
                    f"{effect:.6f}"
                ),
            }
        )

    write_csv(
        ANALYSIS_DIR
        / "pv2_scene_effects.csv",
        pv2_rows,
        [
            "scene_id",
            "generalized_x_rate",
            "neutral_x_rate",
            "pv2_effect",
        ],
    )

    pv2_effects = [
        float(row["pv2_effect"])
        for row in pv2_rows
    ]

    # --------------------------------------------------------
    # 6. PV3: Cue-bound modified minus clean on target scenes
    # --------------------------------------------------------

    target_conditions = {
        "target_clear",
        "target_subtle",
    }

    condition_lookup = {
        row["scene_id"]:
            row["condition"]
        for row in cell_rows
    }

    target_scenes = sorted(
        scene_id
        for scene_id in scenes
        if condition_lookup[
            scene_id
        ] in target_conditions
    )

    if len(target_scenes) != 9:
        raise RuntimeError(
            f"Expected 9 target scenes; "
            f"found {len(target_scenes)}"
        )

    pv3_rows = []

    for scene_id in target_scenes:
        clean_rate = cell_lookup[
            (
                scene_id,
                "B",
                "clean",
            )
        ]

        modified_rate = cell_lookup[
            (
                scene_id,
                "B",
                "modified",
            )
        ]

        effect = (
            modified_rate
            - clean_rate
        )

        pv3_rows.append(
            {
                "scene_id": scene_id,
                "condition": (
                    condition_lookup[
                        scene_id
                    ]
                ),
                "cue_bound_clean_x_rate": (
                    f"{clean_rate:.6f}"
                ),
                "cue_bound_modified_x_rate": (
                    f"{modified_rate:.6f}"
                ),
                "pv3_effect": (
                    f"{effect:.6f}"
                ),
            }
        )

    write_csv(
        ANALYSIS_DIR
        / "pv3_scene_effects.csv",
        pv3_rows,
        [
            "scene_id",
            "condition",
            "cue_bound_clean_x_rate",
            "cue_bound_modified_x_rate",
            "pv3_effect",
        ],
    )

    pv3_effects = [
        float(row["pv3_effect"])
        for row in pv3_rows
    ]

    # --------------------------------------------------------
    # 7. Confirmatory inference
    # --------------------------------------------------------

    hypothesis_effects = {
        "PV1": pv1_effects,
        "PV2": pv2_effects,
        "PV3": pv3_effects,
    }

    raw_pvalues = {
        name:
            exact_one_sided_signflip(
                effects
            )
        for name, effects
        in hypothesis_effects.items()
    }

    adjusted_pvalues = holm_adjust(
        raw_pvalues
    )

    results = []

    for name in (
        "PV1",
        "PV2",
        "PV3",
    ):
        effects = hypothesis_effects[
            name
        ]

        observed = mean(effects)

        ci_low, ci_high = bootstrap_ci(
            effects
        )

        adjusted = adjusted_pvalues[
            name
        ]

        supported = (
            observed > 0
            and adjusted < ALPHA
        )

        results.append(
            {
                "hypothesis": name,
                "n_scenes": len(effects),
                "mean_effect": (
                    f"{observed:.9f}"
                ),
                "mean_effect_percentage_points": (
                    f"{100 * observed:.6f}"
                ),
                "bootstrap_ci_low": (
                    f"{ci_low:.9f}"
                ),
                "bootstrap_ci_high": (
                    f"{ci_high:.9f}"
                ),
                "bootstrap_ci_low_percentage_points": (
                    f"{100 * ci_low:.6f}"
                ),
                "bootstrap_ci_high_percentage_points": (
                    f"{100 * ci_high:.6f}"
                ),
                "raw_p": (
                    f"{raw_pvalues[name]:.12f}"
                ),
                "holm_p": (
                    f"{adjusted:.12f}"
                ),
                "supported": (
                    "yes"
                    if supported
                    else "no"
                ),
            }
        )

    write_csv(
        ANALYSIS_DIR
        / "confirmatory_results.csv",
        results,
        [
            "hypothesis",
            "n_scenes",
            "mean_effect",
            "mean_effect_percentage_points",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "bootstrap_ci_low_percentage_points",
            "bootstrap_ci_high_percentage_points",
            "raw_p",
            "holm_p",
            "supported",
        ],
    )

    # --------------------------------------------------------
    # 8. Pre-specified descriptive summaries
    # --------------------------------------------------------

    profile_summary = []

    for profile_code, profile in (
        ("A", "neutral"),
        ("B", "cue_bound"),
        ("C", "generalized"),
    ):
        rates = [
            float(
                row["taskB_x_choice_rate"]
            )
            for row in cell_rows
            if row["profile_code"]
            == profile_code
        ]

        profile_summary.append(
            {
                "profile_code": profile_code,
                "profile": profile,
                "n_cells": len(rates),
                "mean_taskB_x_rate": (
                    f"{mean(rates):.6f}"
                ),
            }
        )

    write_csv(
        ANALYSIS_DIR
        / "profile_descriptive_summary.csv",
        profile_summary,
        [
            "profile_code",
            "profile",
            "n_cells",
            "mean_taskB_x_rate",
        ],
    )

    distractor_scenes = sorted(
        scene_id
        for scene_id in scenes
        if condition_lookup[
            scene_id
        ] == "distractor_clear"
    )

    distractor_rows = []

    for scene_id in distractor_scenes:
        clean_rate = cell_lookup[
            (
                scene_id,
                "B",
                "clean",
            )
        ]

        modified_rate = cell_lookup[
            (
                scene_id,
                "B",
                "modified",
            )
        ]

        distractor_rows.append(
            {
                "scene_id": scene_id,
                "cue_bound_clean_x_rate": (
                    f"{clean_rate:.6f}"
                ),
                "cue_bound_modified_x_rate": (
                    f"{modified_rate:.6f}"
                ),
                "modified_minus_clean": (
                    f"{modified_rate-clean_rate:.6f}"
                ),
            }
        )

    write_csv(
        ANALYSIS_DIR
        / "distractor_specificity.csv",
        distractor_rows,
        [
            "scene_id",
            "cue_bound_clean_x_rate",
            "cue_bound_modified_x_rate",
            "modified_minus_clean",
        ],
    )

    descriptive = {
        "taskA_cell_prediction_accuracy": (
            mean(overall_cell_scores)
        ),
        "profile_only_prediction_accuracy": (
            mean(overall_baseline_scores)
        ),
        "always_y_accuracy": (
            mean(overall_always_y_scores)
        ),
        "pv1_incremental_accuracy": (
            mean(overall_cell_scores)
            - mean(
                overall_baseline_scores
            )
        ),
        "valid_taskB_responses": 216,
        "taskB_cells": 72,
    }

    (
        ANALYSIS_DIR
        / "descriptive_summary.json"
    ).write_text(
        json.dumps(
            descriptive,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    # --------------------------------------------------------
    # 9. Machine-readable analysis metadata
    # --------------------------------------------------------

    metadata = {
        "experiment": (
            "V7 cross-task "
            "predictive validity"
        ),
        "raw_sha256": raw_sha,
        "predictions_sha256": (
            predictions_sha
        ),
        "bootstrap_replicates": (
            BOOTSTRAP_REPS
        ),
        "bootstrap_seed": (
            BOOTSTRAP_SEED
        ),
        "alpha": ALPHA,
        "confirmatory_family": [
            "PV1",
            "PV2",
            "PV3",
        ],
        "multiple_testing": "Holm",
        "sign_flip": (
            "exact one-sided paired "
            "scene-level enumeration"
        ),
        "inferential_unit": "scene",
    }

    (
        ANALYSIS_DIR
        / "analysis_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    # --------------------------------------------------------
    # 10. Print final results
    # --------------------------------------------------------

    print(
        "PASS: V7 analysis complete."
    )
    print()

    for result in results:
        print(
            f'{result["hypothesis"]}: '
            f'effect='
            f'{result["mean_effect_percentage_points"]} pp, '
            f'95% CI=['
            f'{result["bootstrap_ci_low_percentage_points"]}, '
            f'{result["bootstrap_ci_high_percentage_points"]}], '
            f'raw p={result["raw_p"]}, '
            f'Holm p={result["holm_p"]}, '
            f'supported={result["supported"]}'
        )

    print()
    print(
        "Task-A cell prediction accuracy: "
        f'{100 * descriptive["taskA_cell_prediction_accuracy"]:.3f}%'
    )
    print(
        "Profile-only baseline accuracy: "
        f'{100 * descriptive["profile_only_prediction_accuracy"]:.3f}%'
    )
    print(
        "Always-Y accuracy: "
        f'{100 * descriptive["always_y_accuracy"]:.3f}%'
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
