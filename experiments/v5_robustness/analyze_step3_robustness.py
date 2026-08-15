from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "v5_robustness"

RAW = EXP / "outputs" / "step3_robustness_raw.jsonl"
FREEZE = EXP / "STEP3_RAW_DATA_FREEZE.json"
MANIFEST = EXP / "step3_robustness_manifest.csv"
PLAN = EXP / "STEP3_ANALYSIS_PLAN.md"
PLAN_HASH = EXP / "STEP3_ANALYSIS_PLAN.sha256"

OUT_DIR = EXP / "analysis"

CONFIRMATORY_CSV = OUT_DIR / "step3_confirmatory_results.csv"
R3_CSV = OUT_DIR / "step3_r3_inactive_summary.csv"
R4_CSV = OUT_DIR / "step3_r4_ordering_summary.csv"
DEVIATION_CSV = OUT_DIR / "step3_canonical_deviation.csv"
METADATA_JSON = OUT_DIR / "step3_analysis_metadata.json"
SUMMARY_MD = OUT_DIR / "STEP3_RESULTS.md"

VARIANTS = ["P0", "P1", "P2", "P3", "P4"]
PROFILES = ["neutral", "cue_bound", "generalized"]

EXPECTED_JOBS = 1080
EXPECTED_SCENES = 12
EXPECTED_TARGET_SCENES = 9
EXPECTED_DISTRACTOR_SCENES = 3
EXPECTED_REPS = 3

BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260815
ALPHA = 0.05


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def read_raw() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in RAW.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def percentile(
    values: list[float],
    q: float,
) -> float:

    ordered = sorted(values)

    if not ordered:
        raise ValueError(
            "Cannot calculate percentile of empty list"
        )

    position = (len(ordered) - 1) * q

    lower = int(position)
    upper = min(
        lower + 1,
        len(ordered) - 1,
    )

    fraction = position - lower

    return (
        ordered[lower]
        + fraction
        * (
            ordered[upper]
            - ordered[lower]
        )
    )


def bootstrap_ci(
    differences: list[float],
) -> tuple[float, float]:

    rng = random.Random(
        BOOTSTRAP_SEED
    )

    n = len(differences)

    boot = []

    for _ in range(
        BOOTSTRAP_REPS
    ):
        sample = [
            differences[
                rng.randrange(n)
            ]
            for _ in range(n)
        ]

        boot.append(
            statistics.fmean(sample)
        )

    return (
        percentile(boot, 0.025),
        percentile(boot, 0.975),
    )


def exact_signflip_p(
    differences: list[float],
) -> float:

    observed = statistics.fmean(
        differences
    )

    extreme = 0
    total = 0

    for signs in itertools.product(
        (-1.0, 1.0),
        repeat=len(differences),
    ):
        statistic = statistics.fmean(
            d * s
            for d, s in zip(
                differences,
                signs,
            )
        )

        if statistic >= (
            observed - 1e-12
        ):
            extreme += 1

        total += 1

    return extreme / total


def holm_adjust(
    pvalues: list[float],
) -> list[float]:

    m = len(pvalues)

    order = sorted(
        range(m),
        key=lambda i: pvalues[i],
    )

    adjusted = [
        0.0
        for _ in pvalues
    ]

    running_max = 0.0

    for rank, index in enumerate(
        order
    ):
        candidate = min(
            1.0,
            (m - rank)
            * pvalues[index],
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted[index] = running_max

    return adjusted


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:

    if not rows:
        raise ValueError(
            f"No rows for {path}"
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)


def classify_scene(
    condition: str,
) -> str:

    if condition.startswith(
        "target"
    ):
        return "target"

    if condition.startswith(
        "distractor"
    ):
        return "distractor"

    raise ValueError(
        f"Unexpected condition: {condition}"
    )


def validate_inputs(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    freeze = json.loads(
        FREEZE.read_text(
            encoding="utf-8"
        )
    )

    actual_raw_sha = sha256_file(
        RAW
    )

    if (
        actual_raw_sha
        != freeze["raw_sha256"]
    ):
        raise ValueError(
            "Frozen raw SHA mismatch"
        )

    recorded_plan_sha = (
        PLAN_HASH.read_text(
            encoding="utf-8-sig"
        )
        .strip()
        .split()[0]
        .lower()
    )

    actual_plan_sha = sha256_file(
        PLAN
    ).lower()

    if (
        recorded_plan_sha
        != actual_plan_sha
    ):
        raise ValueError(
            "Frozen analysis-plan SHA mismatch"
        )

    if len(records) != 1082:
        raise ValueError(
            f"Expected 1082 raw records, "
            f"found {len(records)}"
        )

    statuses = Counter(
        r.get("status")
        for r in records
    )

    if statuses["ok"] != 1080:
        raise ValueError(
            f"Expected 1080 successful records, "
            f"found {statuses['ok']}"
        )

    if statuses["invalid_response"] != 0:
        raise ValueError(
            "Unexpected substantive invalid response"
        )

    if statuses["error"] != 2:
        raise ValueError(
            f"Expected 2 procedural error records, "
            f"found {statuses['error']}"
        )

    ok_records = [
        r
        for r in records
        if r.get("status") == "ok"
    ]

    ok_ids = [
        r["job_id"]
        for r in ok_records
    ]

    if len(set(ok_ids)) != EXPECTED_JOBS:
        raise ValueError(
            "Successful terminal job IDs are not unique"
        )

    error_ids = {
        r["job_id"]
        for r in records
        if r.get("status") == "error"
    }

    if not error_ids.issubset(
        set(ok_ids)
    ):
        raise ValueError(
            "A procedural error was not followed "
            "by a successful unchanged retry"
        )

    manifest_rows = list(
        csv.DictReader(
            MANIFEST.open(
                newline="",
                encoding="utf-8-sig",
            )
        )
    )

    manifest_ids = {
        r["v5_job_id"]
        for r in manifest_rows
    }

    if set(ok_ids) != manifest_ids:
        raise ValueError(
            "Successful job IDs do not exactly match manifest"
        )

    variants = {
        r["robustness_variant"]
        for r in ok_records
    }

    if variants != set(
        VARIANTS
    ):
        raise ValueError(
            f"Unexpected variants: {variants}"
        )

    profiles = {
        r["v5_profile"]
        for r in ok_records
    }

    if profiles != set(
        PROFILES
    ):
        raise ValueError(
            f"Unexpected profiles: {profiles}"
        )

    scenes = {
        r["scene_id"]
        for r in ok_records
    }

    if len(scenes) != EXPECTED_SCENES:
        raise ValueError(
            f"Expected 12 scenes, found {len(scenes)}"
        )

    scene_classes = {}

    for r in ok_records:
        scene_class = classify_scene(
            r["condition"]
        )

        previous = scene_classes.get(
            r["scene_id"]
        )

        if (
            previous is not None
            and previous != scene_class
        ):
            raise ValueError(
                "Scene has inconsistent condition class"
            )

        scene_classes[
            r["scene_id"]
        ] = scene_class

    target_count = sum(
        value == "target"
        for value in scene_classes.values()
    )

    distractor_count = sum(
        value == "distractor"
        for value in scene_classes.values()
    )

    if target_count != EXPECTED_TARGET_SCENES:
        raise ValueError(
            f"Expected 9 target scenes, found {target_count}"
        )

    if distractor_count != EXPECTED_DISTRACTOR_SCENES:
        raise ValueError(
            f"Expected 3 distractor scenes, "
            f"found {distractor_count}"
        )

    cell_counts = Counter(
        (
            r["robustness_variant"],
            r["scene_id"],
            r["v5_profile"],
            r["image_variant"],
        )
        for r in ok_records
    )

    expected_cells = (
        5
        * 12
        * 3
        * 2
    )

    if len(cell_counts) != expected_cells:
        raise ValueError(
            f"Expected {expected_cells} complete cells, "
            f"found {len(cell_counts)}"
        )

    bad_cells = {
        key: count
        for key, count
        in cell_counts.items()
        if count != EXPECTED_REPS
    }

    if bad_cells:
        raise ValueError(
            f"Incomplete repetition cells: {bad_cells}"
        )

    for r in ok_records:
        expected_key = (
            "y"
            if r["robustness_variant"]
            == "P2"
            else "x"
        )

        if (
            r["designated_output_key"]
            != expected_key
        ):
            raise ValueError(
                "Incorrect designated output key"
            )

        expected_value = float(
            r[expected_key]
        )

        actual_value = float(
            r["designated_allocation"]
        )

        if abs(
            expected_value
            - actual_value
        ) > 1e-12:
            raise ValueError(
                "designated_allocation does not "
                "match designated output field"
            )

    return ok_records


def make_cell_means(
    records: list[dict[str, Any]],
) -> dict[
    tuple[str, str, str, str],
    float
]:

    grouped = defaultdict(
        list
    )

    for r in records:
        key = (
            r["robustness_variant"],
            r["scene_id"],
            r["v5_profile"],
            r["image_variant"],
        )

        grouped[key].append(
            float(
                r["designated_allocation"]
            )
        )

    means = {}

    for key, values in grouped.items():
        if len(values) != 3:
            continue

        means[key] = statistics.fmean(
            values
        )

    return means


def scene_classes(
    records: list[dict[str, Any]],
) -> dict[str, str]:

    result = {}

    for r in records:
        result[r["scene_id"]] = (
            classify_scene(
                r["condition"]
            )
        )

    return result


def r1_differences(
    means: dict,
    scenes: list[str],
    variant: str,
) -> list[float]:

    diffs = []

    for scene in scenes:

        generalized = statistics.fmean(
            [
                means[
                    (
                        variant,
                        scene,
                        "generalized",
                        "clean",
                    )
                ],
                means[
                    (
                        variant,
                        scene,
                        "generalized",
                        "modified",
                    )
                ],
            ]
        )

        neutral = statistics.fmean(
            [
                means[
                    (
                        variant,
                        scene,
                        "neutral",
                        "clean",
                    )
                ],
                means[
                    (
                        variant,
                        scene,
                        "neutral",
                        "modified",
                    )
                ],
            ]
        )

        diffs.append(
            generalized - neutral
        )

    return diffs


def r2_differences(
    means: dict,
    target_scenes: list[str],
    variant: str,
) -> list[float]:

    return [
        means[
            (
                variant,
                scene,
                "cue_bound",
                "modified",
            )
        ]
        -
        means[
            (
                variant,
                scene,
                "cue_bound",
                "clean",
            )
        ]
        for scene in target_scenes
    ]


def inactive_scene_mean(
    means: dict,
    variant: str,
    scene: str,
    profile: str,
    scene_class: str,
) -> float:

    if scene_class == "target":
        return means[
            (
                variant,
                scene,
                profile,
                "clean",
            )
        ]

    return statistics.fmean(
        [
            means[
                (
                    variant,
                    scene,
                    profile,
                    "clean",
                )
            ],
            means[
                (
                    variant,
                    scene,
                    profile,
                    "modified",
                )
            ],
        ]
    )


def overall_profile_mean(
    means: dict,
    variant: str,
    scenes: list[str],
    profile: str,
) -> float:

    scene_values = []

    for scene in scenes:
        scene_values.append(
            statistics.fmean(
                [
                    means[
                        (
                            variant,
                            scene,
                            profile,
                            "clean",
                        )
                    ],
                    means[
                        (
                            variant,
                            scene,
                            profile,
                            "modified",
                        )
                    ],
                ]
            )
        )

    return statistics.fmean(
        scene_values
    )


def target_active_mean(
    means: dict,
    variant: str,
    target_scenes: list[str],
) -> float:

    return statistics.fmean(
        means[
            (
                variant,
                scene,
                "cue_bound",
                "modified",
            )
        ]
        for scene in target_scenes
    )


def inactive_mean(
    means: dict,
    variant: str,
    scenes: list[str],
    classes: dict[str, str],
    profile: str,
) -> float:

    values = [
        inactive_scene_mean(
            means,
            variant,
            scene,
            profile,
            classes[scene],
        )
        for scene in scenes
    ]

    return statistics.fmean(
        values
    )


def fmt(value: float) -> str:
    return f"{value:.6f}"


def run_analysis(
    records: list[dict[str, Any]],
) -> None:

    means = make_cell_means(
        records
    )

    classes = scene_classes(
        records
    )

    scenes = sorted(
        classes
    )

    target_scenes = [
        scene
        for scene in scenes
        if classes[scene] == "target"
    ]

    # --------------------------------------------------------
    # R1 + R2 confirmatory tests
    # --------------------------------------------------------

    confirmatory = []

    for variant in VARIANTS:

        r1 = r1_differences(
            means,
            scenes,
            variant,
        )

        ci_low, ci_high = bootstrap_ci(
            r1
        )

        confirmatory.append(
            {
                "outcome": "R1_Generalized_minus_Neutral",
                "variant": variant,
                "n_scenes": len(r1),
                "mean_difference": statistics.fmean(
                    r1
                ),
                "ci95_lower": ci_low,
                "ci95_upper": ci_high,
                "p_exact_one_sided": exact_signflip_p(
                    r1
                ),
            }
        )

        r2 = r2_differences(
            means,
            target_scenes,
            variant,
        )

        ci_low, ci_high = bootstrap_ci(
            r2
        )

        confirmatory.append(
            {
                "outcome": "R2_CueModified_minus_Clean",
                "variant": variant,
                "n_scenes": len(r2),
                "mean_difference": statistics.fmean(
                    r2
                ),
                "ci95_lower": ci_low,
                "ci95_upper": ci_high,
                "p_exact_one_sided": exact_signflip_p(
                    r2
                ),
            }
        )

    adjusted = holm_adjust(
        [
            row["p_exact_one_sided"]
            for row in confirmatory
        ]
    )

    for row, p_holm in zip(
        confirmatory,
        adjusted,
    ):
        row["p_holm"] = p_holm
        row["positive_direction"] = (
            row["mean_difference"] > 0
        )
        row["holm_significant_0_05"] = (
            p_holm < ALPHA
        )

    global_preserved = all(
        row["positive_direction"]
        and row["holm_significant_0_05"]
        for row in confirmatory
    )

    # --------------------------------------------------------
    # R3 inactive cue-bound descriptive comparison
    # --------------------------------------------------------

    r3_rows = []

    # --------------------------------------------------------
    # R4 descriptive aggregate states
    # --------------------------------------------------------

    r4_rows = []

    state_values = {}

    for variant in VARIANTS:

        cb_inactive = inactive_mean(
            means,
            variant,
            scenes,
            classes,
            "cue_bound",
        )

        n_inactive = inactive_mean(
            means,
            variant,
            scenes,
            classes,
            "neutral",
        )

        r3_rows.append(
            {
                "variant": variant,
                "cue_bound_inactive_mean": cb_inactive,
                "neutral_matched_inactive_mean": n_inactive,
                "cue_bound_minus_neutral": (
                    cb_inactive - n_inactive
                ),
            }
        )

        generalized = overall_profile_mean(
            means,
            variant,
            scenes,
            "generalized",
        )

        neutral = overall_profile_mean(
            means,
            variant,
            scenes,
            "neutral",
        )

        cb_active = target_active_mean(
            means,
            variant,
            target_scenes,
        )

        values = {
            "generalized": generalized,
            "cue_bound_active": cb_active,
            "cue_bound_inactive": cb_inactive,
            "neutral": neutral,
        }

        state_values[
            variant
        ] = values

        sorted_order = " > ".join(
            name
            for name, _
            in sorted(
                values.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        core_separation = (
            generalized > cb_active
            and cb_active > cb_inactive
            and cb_active > neutral
        )

        strict_opposite_reversal = (
            generalized < cb_active
            and cb_active < cb_inactive
            and cb_active < neutral
        )

        r4_rows.append(
            {
                "variant": variant,
                "generalized_overall": generalized,
                "cue_bound_target_active": cb_active,
                "cue_bound_inactive": cb_inactive,
                "neutral_overall": neutral,
                "descending_order": sorted_order,
                "core_separation_preserved": core_separation,
                "strict_opposite_reversal": strict_opposite_reversal,
            }
        )

    # --------------------------------------------------------
    # Canonical deviations from contemporaneous P0
    # --------------------------------------------------------

    deviation_rows = []

    metrics = [
        "generalized",
        "cue_bound_active",
        "cue_bound_inactive",
        "neutral",
    ]

    p0 = state_values["P0"]

    for variant in VARIANTS[1:]:

        for metric in metrics:

            value = state_values[
                variant
            ][metric]

            reference = p0[
                metric
            ]

            deviation_rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "variant_mean": value,
                    "p0_mean": reference,
                    "signed_deviation": (
                        value - reference
                    ),
                    "absolute_deviation": abs(
                        value - reference
                    ),
                }
            )

    maximum_deviation = max(
        row["absolute_deviation"]
        for row in deviation_rows
    )

    max_deviation_row = max(
        deviation_rows,
        key=lambda row: (
            row["absolute_deviation"]
        ),
    )

    # --------------------------------------------------------
    # Outputs
    # --------------------------------------------------------

    write_csv(
        CONFIRMATORY_CSV,
        confirmatory,
    )

    write_csv(
        R3_CSV,
        r3_rows,
    )

    write_csv(
        R4_CSV,
        r4_rows,
    )

    write_csv(
        DEVIATION_CSV,
        deviation_rows,
    )

    freeze = json.loads(
        FREEZE.read_text(
            encoding="utf-8"
        )
    )

    analysis_commit = subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=ROOT,
        text=True,
    ).strip()

    metadata = {
        "status": "DERIVED_FROM_FROZEN_STEP3_DATA",
        "analysis_git_commit": analysis_commit,
        "raw_sha256": freeze["raw_sha256"],
        "analysis_plan_sha256": (
            PLAN_HASH.read_text(
                encoding="utf-8-sig"
            )
            .strip()
            .split()[0]
        ),
        "planned_jobs": 1080,
        "successful_jobs": 1080,
        "procedural_error_records": 2,
        "invalid_responses": 0,
        "scene_inferential_unit": True,
        "scenes": 12,
        "target_scenes": 9,
        "distractor_scenes": 3,
        "bootstrap_resamples": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "confirmatory_tests": 10,
        "multiplicity": "Holm across all 10 R1/R2 variant-specific tests",
        "familywise_alpha": ALPHA,
        "global_confirmatory_robustness_criterion_met": (
            global_preserved
        ),
        "maximum_absolute_deviation_from_P0": (
            maximum_deviation
        ),
        "maximum_deviation_variant": (
            max_deviation_row["variant"]
        ),
        "maximum_deviation_metric": (
            max_deviation_row["metric"]
        ),
        "r3_r4_status": "descriptive_only",
    }

    METADATA_JSON.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    lines = []

    lines.append(
        "# Step 3 Prompt-Surface Robustness Results"
    )
    lines.append("")
    lines.append(
        "Derived from the frozen Step-3 raw dataset "
        "under the frozen analysis plan."
    )
    lines.append("")
    lines.append(
        "## Confirmatory R1/R2"
    )
    lines.append("")
    lines.append(
        "| Outcome | Variant | n | Mean diff | 95% CI | "
        "Exact p | Holm p |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|"
    )

    for row in confirmatory:
        lines.append(
            "| "
            + str(row["outcome"])
            + " | "
            + str(row["variant"])
            + " | "
            + str(row["n_scenes"])
            + " | "
            + fmt(
                row["mean_difference"]
            )
            + " | ["
            + fmt(
                row["ci95_lower"]
            )
            + ", "
            + fmt(
                row["ci95_upper"]
            )
            + "] | "
            + fmt(
                row["p_exact_one_sided"]
            )
            + " | "
            + fmt(
                row["p_holm"]
            )
            + " |"
        )

    lines.append("")
    lines.append(
        "**Global confirmatory robustness criterion met:** "
        + (
            "YES"
            if global_preserved
            else "NO"
        )
    )

    lines.append("")
    lines.append(
        "## R3 inactive cue-bound behavior"
    )
    lines.append("")
    lines.append(
        "| Variant | Cue-bound inactive | Neutral matched | Difference |"
    )
    lines.append(
        "|---|---:|---:|---:|"
    )

    for row in r3_rows:
        lines.append(
            f"| {row['variant']} | "
            f"{fmt(row['cue_bound_inactive_mean'])} | "
            f"{fmt(row['neutral_matched_inactive_mean'])} | "
            f"{fmt(row['cue_bound_minus_neutral'])} |"
        )

    lines.append("")
    lines.append(
        "## R4 qualitative states"
    )
    lines.append("")
    lines.append(
        "| Variant | G | C active | C inactive | N | "
        "Core separation | Strict reversal |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---|---|"
    )

    for row in r4_rows:
        lines.append(
            f"| {row['variant']} | "
            f"{fmt(row['generalized_overall'])} | "
            f"{fmt(row['cue_bound_target_active'])} | "
            f"{fmt(row['cue_bound_inactive'])} | "
            f"{fmt(row['neutral_overall'])} | "
            f"{row['core_separation_preserved']} | "
            f"{row['strict_opposite_reversal']} |"
        )

    lines.append("")
    lines.append(
        "## Canonical deviation"
    )
    lines.append("")
    lines.append(
        "Maximum absolute deviation from contemporaneous P0: "
        f"**{fmt(maximum_deviation)} points** "
        f"({max_deviation_row['variant']}, "
        f"{max_deviation_row['metric']})."
    )

    SUMMARY_MD.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    print()
    print("STEP 3 ANALYSIS COMPLETE")
    print("========================")
    print()

    for row in confirmatory:
        print(
            f"{row['outcome']} {row['variant']}: "
            f"diff={row['mean_difference']:.3f}, "
            f"CI=[{row['ci95_lower']:.3f}, "
            f"{row['ci95_upper']:.3f}], "
            f"p={row['p_exact_one_sided']:.6f}, "
            f"Holm={row['p_holm']:.6f}"
        )

    print()
    print(
        "GLOBAL CONFIRMATORY ROBUSTNESS:",
        "PASS"
        if global_preserved
        else "FAIL",
    )

    print(
        "Maximum absolute deviation from P0:",
        f"{maximum_deviation:.3f}",
        "points",
        f"({max_deviation_row['variant']}, "
        f"{max_deviation_row['metric']})",
    )

    print()
    print(
        f"Results written to: {OUT_DIR}"
    )


def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    args = parser.parse_args()

    records = read_raw()

    ok_records = validate_inputs(
        records
    )

    print(
        "STEP 3 INPUT VALIDATION: PASS"
    )
    print(
        "Frozen raw records: 1082"
    )
    print(
        "Successful planned jobs: 1080/1080"
    )
    print(
        "Complete repetition cells: 360/360"
    )
    print(
        "Scenes: 12 "
        "(9 target, 3 distractor)"
    )

    if args.validate_only:
        print(
            "VALIDATION ONLY — "
            "NO AGGREGATE RESULTS COMPUTED"
        )
        return 0

    run_analysis(
        ok_records
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
