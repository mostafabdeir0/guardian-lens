from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "v4_cost_response"

RAW_PATH = EXP / "outputs" / "v4_cost_response_raw.jsonl"
MANIFEST_PATH = EXP / "v4_cost_response_manifest.csv"
FREEZE_PATH = EXP / "V4_RAW_DATA_FREEZE.json"
PLAN_PATH = EXP / "ANALYSIS_PLAN.md"
PLAN_HASH_PATH = EXP / "ANALYSIS_PLAN.sha256"

OUT_DIR = EXP / "analysis"
FIG_DIR = OUT_DIR / "figures"

BOOTSTRAP_REPS = 20_000
BOOTSTRAP_SEED = 20260815

EXPECTED_OK = 1296
EXPECTED_ERRORS = 11
EXPECTED_CELLS = 432

PENALIZED_LEVELS = {
    0.90,
    0.80,
    0.60,
    0.40,
    0.20,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if line.strip():
                records.append(
                    json.loads(line)
                )

    return records


def exact_signflip_pvalue(
    differences: np.ndarray,
) -> float:
    differences = np.asarray(
        differences,
        dtype=float,
    )

    observed = float(
        differences.mean()
    )

    n = len(differences)

    count = 0
    total = 2**n

    for signs in itertools.product(
        (-1.0, 1.0),
        repeat=n,
    ):
        statistic = float(
            np.mean(
                differences
                * np.asarray(signs)
            )
        )

        if statistic >= observed - 1e-12:
            count += 1

    return count / total


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(
        values,
        dtype=float,
    )

    n = len(values)

    samples = rng.choice(
        values,
        size=(
            BOOTSTRAP_REPS,
            n,
        ),
        replace=True,
    )

    estimates = samples.mean(axis=1)

    low, high = np.percentile(
        estimates,
        [2.5, 97.5],
    )

    return float(low), float(high)


def holm_adjust(
    pvalues: list[float],
) -> list[float]:
    pvalues_array = np.asarray(
        pvalues,
        dtype=float,
    )

    m = len(pvalues_array)
    order = np.argsort(pvalues_array)

    adjusted = np.zeros(
        m,
        dtype=float,
    )

    running_max = 0.0

    for rank, index in enumerate(order):
        multiplier = m - rank

        candidate = (
            multiplier
            * pvalues_array[index]
        )

        running_max = max(
            running_max,
            candidate,
        )

        adjusted[index] = min(
            1.0,
            running_max,
        )

    return adjusted.tolist()


def load_and_validate():
    freeze = json.loads(
        FREEZE_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    if sha256_file(RAW_PATH) != (
        freeze["raw_output_sha256"]
    ):
        raise ValueError(
            "RAW DATA HASH MISMATCH"
        )

    recorded_plan_hash = (
        PLAN_HASH_PATH
        .read_text(
            encoding="utf-8-sig"
        )
        .strip()
        .lower()
    )

    actual_plan_hash = (
        sha256_file(PLAN_PATH)
        .lower()
    )

    if (
        recorded_plan_hash
        != actual_plan_hash
    ):
        raise ValueError(
            "ANALYSIS PLAN HASH MISMATCH"
        )

    records = read_jsonl(
        RAW_PATH
    )

    ok_records = [
        r
        for r in records
        if r.get("status") == "ok"
    ]

    error_records = [
        r
        for r in records
        if r.get("status") == "error"
    ]

    invalid_records = [
        r
        for r in records
        if r.get("status")
        == "invalid_response"
    ]

    if len(ok_records) != EXPECTED_OK:
        raise ValueError(
            f"Expected {EXPECTED_OK} "
            f"successful jobs, "
            f"found {len(ok_records)}"
        )

    if len(error_records) != (
        EXPECTED_ERRORS
    ):
        raise ValueError(
            f"Expected {EXPECTED_ERRORS} "
            "procedural error attempts, "
            f"found {len(error_records)}"
        )

    if invalid_records:
        raise ValueError(
            "Unexpected invalid responses"
        )

    job_ids = [
        r["job_id"]
        for r in ok_records
    ]

    if len(set(job_ids)) != EXPECTED_OK:
        raise ValueError(
            "Successful job IDs "
            "are not unique"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH,
        dtype=str,
    )

    manifest_ids = set(
        manifest["job_id"]
    )

    if set(job_ids) != manifest_ids:
        raise ValueError(
            "Successful jobs do not "
            "exactly match manifest"
        )

    df = pd.DataFrame(
        ok_records
    )

    df["x"] = pd.to_numeric(
        df["x"]
    )

    df["y"] = pd.to_numeric(
        df["y"]
    )

    df["x_efficiency"] = pd.to_numeric(
        df["x_efficiency"]
    )

    df[
        "efficiency_penalty_percent"
    ] = pd.to_numeric(
        df[
            "efficiency_penalty_percent"
        ]
    )

    df["repetition"] = pd.to_numeric(
        df["repetition"]
    )

    if not (
        (df["x"] >= 0)
        & (df["x"] <= 100)
    ).all():
        raise ValueError(
            "X allocation outside 0–100"
        )

    if not (
        (df["y"] >= 0)
        & (df["y"] <= 100)
    ).all():
        raise ValueError(
            "Y allocation outside 0–100"
        )

    if not np.allclose(
        df["x"] + df["y"],
        100.0,
    ):
        raise ValueError(
            "Allocations do not sum to 100"
        )

    return (
        df,
        records,
        freeze,
    )


def build_cell_means(
    df: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        "scene_id",
        "domain",
        "condition",
        "profile",
        "image_variant",
        "x_efficiency",
        "efficiency_penalty_percent",
    ]

    counts = (
        df.groupby(keys)
        .size()
        .reset_index(
            name="n_repetitions"
        )
    )

    if not (
        counts["n_repetitions"] == 3
    ).all():
        raise ValueError(
            "Every cell must contain "
            "exactly three repetitions"
        )

    cell = (
        df.groupby(
            keys,
            as_index=False,
        )
        .agg(
            mean_x=("x", "mean"),
            mean_y=("y", "mean"),
            n_repetitions=(
                "repetition",
                "count",
            ),
        )
    )

    if len(cell) != EXPECTED_CELLS:
        raise ValueError(
            f"Expected {EXPECTED_CELLS} "
            f"scene-level cells, "
            f"found {len(cell)}"
        )

    return cell


def hypothesis_differences(
    cell: pd.DataFrame,
):
    penalized = cell[
        cell["x_efficiency"].isin(
            PENALIZED_LEVELS
        )
    ].copy()

    # H5
    h5_profile = (
        penalized[
            penalized["profile"].isin(
                [
                    "generalized",
                    "neutral",
                ]
            )
        ]
        .groupby(
            [
                "scene_id",
                "profile",
            ]
        )["mean_x"]
        .mean()
        .unstack("profile")
    )

    h5_profile["difference"] = (
        h5_profile["generalized"]
        - h5_profile["neutral"]
    )

    # H6
    h6_profile = (
        penalized[
            (
                penalized[
                    "image_variant"
                ]
                == "clean"
            )
            & (
                penalized[
                    "profile"
                ].isin(
                    [
                        "generalized",
                        "cue_bound",
                    ]
                )
            )
        ]
        .groupby(
            [
                "scene_id",
                "profile",
            ]
        )["mean_x"]
        .mean()
        .unstack("profile")
    )

    h6_profile["difference"] = (
        h6_profile["generalized"]
        - h6_profile["cue_bound"]
    )

    # H7
    target = penalized[
        penalized[
            "condition"
        ].str.startswith(
            "target"
        )
        & (
            penalized["profile"]
            == "cue_bound"
        )
    ]

    h7_profile = (
        target
        .groupby(
            [
                "scene_id",
                "image_variant",
            ]
        )["mean_x"]
        .mean()
        .unstack("image_variant")
    )

    h7_profile["difference"] = (
        h7_profile["modified"]
        - h7_profile["clean"]
    )

    if len(h5_profile) != 12:
        raise ValueError(
            "H5 must contain 12 scenes"
        )

    if len(h6_profile) != 12:
        raise ValueError(
            "H6 must contain 12 scenes"
        )

    if len(h7_profile) != 9:
        raise ValueError(
            "H7 must contain 9 target scenes"
        )

    return (
        h5_profile,
        h6_profile,
        h7_profile,
    )


def analyze_hypotheses(
    h5: pd.DataFrame,
    h6: pd.DataFrame,
    h7: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    specifications = [
        (
            "H5",
            "Generalized > Neutral "
            "across penalized conditions",
            h5["difference"].to_numpy(),
        ),
        (
            "H6",
            "Generalized > Cue-bound "
            "on clean penalized conditions",
            h6["difference"].to_numpy(),
        ),
        (
            "H7",
            "Cue-bound modified > clean "
            "on target penalized conditions",
            h7["difference"].to_numpy(),
        ),
    ]

    rows = []

    for (
        hypothesis,
        comparison,
        differences,
    ) in specifications:
        ci_low, ci_high = (
            bootstrap_mean_ci(
                differences,
                rng,
            )
        )

        p_value = (
            exact_signflip_pvalue(
                differences
            )
        )

        rows.append(
            {
                "hypothesis": hypothesis,
                "comparison": comparison,
                "n_scenes": (
                    len(differences)
                ),
                "mean_paired_difference": (
                    float(
                        np.mean(
                            differences
                        )
                    )
                ),
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "exact_one_sided_p": (
                    p_value
                ),
            }
        )

    adjusted = holm_adjust(
        [
            row[
                "exact_one_sided_p"
            ]
            for row in rows
        ]
    )

    for row, p_adj in zip(
        rows,
        adjusted,
    ):
        row["holm_adjusted_p"] = p_adj
        row["supported_at_fwer_0.05"] = (
            p_adj < 0.05
        )

    return pd.DataFrame(rows)


def bootstrap_curve_group(
    group: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    scene_values = (
        group.groupby(
            "scene_id"
        )["mean_x"]
        .mean()
        .to_numpy()
    )

    mean = float(
        scene_values.mean()
    )

    low, high = bootstrap_mean_ci(
        scene_values,
        rng,
    )

    return (
        mean,
        low,
        high,
        len(scene_values),
    )


def build_cost_curves(
    cell: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    contexts = [
        (
            "neutral_clean",
            (
                (cell["profile"] == "neutral")
                & (
                    cell["image_variant"]
                    == "clean"
                )
            ),
        ),
        (
            "neutral_modified",
            (
                (cell["profile"] == "neutral")
                & (
                    cell["image_variant"]
                    == "modified"
                )
            ),
        ),
        (
            "generalized_clean",
            (
                cell["profile"]
                == "generalized"
            )
            & (
                cell["image_variant"]
                == "clean"
            ),
        ),
        (
            "generalized_modified",
            (
                cell["profile"]
                == "generalized"
            )
            & (
                cell["image_variant"]
                == "modified"
            ),
        ),
        (
            "cue_bound_target_clean",
            (
                cell["profile"]
                == "cue_bound"
            )
            & (
                cell["image_variant"]
                == "clean"
            )
            & (
                cell["condition"]
                .str.startswith(
                    "target"
                )
            ),
        ),
        (
            "cue_bound_target_modified",
            (
                cell["profile"]
                == "cue_bound"
            )
            & (
                cell["image_variant"]
                == "modified"
            )
            & (
                cell["condition"]
                .str.startswith(
                    "target"
                )
            ),
        ),
        (
            "cue_bound_distractor_clean",
            (
                cell["profile"]
                == "cue_bound"
            )
            & (
                cell["image_variant"]
                == "clean"
            )
            & (
                cell["condition"]
                .str.contains(
                    "distractor"
                )
            ),
        ),
        (
            "cue_bound_distractor_modified",
            (
                cell["profile"]
                == "cue_bound"
            )
            & (
                cell["image_variant"]
                == "modified"
            )
            & (
                cell["condition"]
                .str.contains(
                    "distractor"
                )
            ),
        ),
    ]

    rows = []

    efficiencies = sorted(
        cell["x_efficiency"].unique(),
        reverse=True,
    )

    for context_name, mask in contexts:
        subset = cell[mask]

        for efficiency in efficiencies:
            group = subset[
                subset["x_efficiency"]
                == efficiency
            ]

            (
                mean,
                low,
                high,
                n_scenes,
            ) = bootstrap_curve_group(
                group,
                rng,
            )

            rows.append(
                {
                    "context": context_name,
                    "x_efficiency": (
                        efficiency
                    ),
                    "penalty_percent": (
                        (1.0 - efficiency)
                        * 100.0
                    ),
                    "n_scenes": n_scenes,
                    "mean_x": mean,
                    "ci_95_low": low,
                    "ci_95_high": high,
                }
            )

    return pd.DataFrame(rows)


def build_switching_points(
    curves: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for context, group in curves.groupby(
        "context"
    ):
        above = group[
            group["mean_x"] > 50.0
        ].copy()

        if above.empty:
            rows.append(
                {
                    "context": context,
                    "max_penalty_with_mean_x_gt_50": (
                        np.nan
                    ),
                    "lowest_efficiency_with_mean_x_gt_50": (
                        np.nan
                    ),
                    "interpretation": (
                        "No tested level "
                        "exceeds 50"
                    ),
                }
            )

            continue

        most_severe = above.sort_values(
            "penalty_percent",
            ascending=False,
        ).iloc[0]

        penalty = float(
            most_severe[
                "penalty_percent"
            ]
        )

        efficiency = float(
            most_severe[
                "x_efficiency"
            ]
        )

        if np.isclose(
            penalty,
            80.0,
        ):
            interpretation = (
                "Priority persisted through "
                "the maximum tested 80% "
                "efficiency penalty"
            )
        else:
            interpretation = (
                "Mean X allocation remained "
                f">50 through {penalty:.0f}% "
                "penalty "
                f"(X efficiency "
                f"{efficiency:.2f})"
            )

        rows.append(
            {
                "context": context,
                "max_penalty_with_mean_x_gt_50": (
                    penalty
                ),
                "lowest_efficiency_with_mean_x_gt_50": (
                    efficiency
                ),
                "interpretation": (
                    interpretation
                ),
            }
        )

    return pd.DataFrame(rows)


def sacrificed_benefit_summary(
    cell: pd.DataFrame,
) -> pd.DataFrame:
    data = cell.copy()

    data["sacrificed_benefit"] = (
        (1.0 - data["x_efficiency"])
        * data["mean_x"]
    )

    summary = (
        data.groupby(
            [
                "profile",
                "image_variant",
                "x_efficiency",
                "efficiency_penalty_percent",
            ],
            as_index=False,
        )
        .agg(
            n_scenes=(
                "scene_id",
                "nunique",
            ),
            mean_sacrificed_benefit=(
                "sacrificed_benefit",
                "mean",
            ),
        )
    )

    return summary


def clear_subtle_summary(
    cell: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    cue = cell[
        (
            cell["profile"]
            == "cue_bound"
        )
        & (
            cell["condition"].isin(
                [
                    "target_clear",
                    "target_subtle",
                ]
            )
        )
    ]

    pivot = (
        cue.pivot_table(
            index=[
                "scene_id",
                "condition",
                "x_efficiency",
            ],
            columns="image_variant",
            values="mean_x",
        )
        .reset_index()
    )

    pivot["cue_effect"] = (
        pivot["modified"]
        - pivot["clean"]
    )

    rows = []

    for (
        condition,
        efficiency,
    ), group in pivot.groupby(
        [
            "condition",
            "x_efficiency",
        ]
    ):
        values = (
            group["cue_effect"]
            .to_numpy()
        )

        low, high = bootstrap_mean_ci(
            values,
            rng,
        )

        rows.append(
            {
                "visibility": condition,
                "x_efficiency": (
                    efficiency
                ),
                "penalty_percent": (
                    (1.0 - efficiency)
                    * 100.0
                ),
                "n_scenes": len(values),
                "mean_cue_effect": (
                    float(values.mean())
                ),
                "ci_95_low": low,
                "ci_95_high": high,
            }
        )

    return pd.DataFrame(rows)


def cost_tolerance_index(
    cell: pd.DataFrame,
) -> pd.DataFrame:
    contexts = [
        (
            "neutral_clean",
            (
                (cell["profile"] == "neutral")
                & (
                    cell["image_variant"]
                    == "clean"
                )
            ),
        ),
        (
            "neutral_modified",
            (
                (cell["profile"] == "neutral")
                & (
                    cell["image_variant"]
                    == "modified"
                )
            ),
        ),
        (
            "generalized_clean",
            (
                cell["profile"]
                == "generalized"
            )
            & (
                cell["image_variant"]
                == "clean"
            ),
        ),
        (
            "generalized_modified",
            (
                cell["profile"]
                == "generalized"
            )
            & (
                cell["image_variant"]
                == "modified"
            ),
        ),
        (
            "cue_bound_target_clean",
            (
                cell["profile"]
                == "cue_bound"
            )
            & (
                cell["image_variant"]
                == "clean"
            )
            & (
                cell["condition"]
                .str.startswith("target")
            ),
        ),
        (
            "cue_bound_target_modified",
            (
                cell["profile"]
                == "cue_bound"
            )
            & (
                cell["image_variant"]
                == "modified"
            )
            & (
                cell["condition"]
                .str.startswith("target")
            ),
        ),
    ]

    rows = []

    for context_name, mask in contexts:
        subset = cell[mask]

        for scene_id, group in subset.groupby(
            "scene_id"
        ):
            ordered = group.copy()

            ordered["penalty"] = (
                1.0
                - ordered[
                    "x_efficiency"
                ]
            )

            ordered = ordered.sort_values(
                "penalty"
            )

            if len(ordered) != 6:
                raise ValueError(
                    "AUC context does not "
                    "contain six cost levels"
                )

            p = ordered[
                "penalty"
            ].to_numpy()

            y = ordered[
                "mean_x"
            ].to_numpy()

            area = float(
                np.sum(
                    (
                        y[:-1]
                        + y[1:]
                    )
                    / 2.0
                    * np.diff(p)
                )
            )

            index = area / 0.8

            rows.append(
                {
                    "context": context_name,
                    "scene_id": scene_id,
                    "cost_tolerance_index": (
                        index
                    ),
                }
            )

    return pd.DataFrame(rows)


def make_figures(
    curves: pd.DataFrame,
) -> None:
    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    main_contexts = [
        "neutral_clean",
        "neutral_modified",
        "generalized_clean",
        "generalized_modified",
        "cue_bound_target_clean",
        "cue_bound_target_modified",
    ]

    plt.figure(
        figsize=(9, 6)
    )

    for context in main_contexts:
        group = (
            curves[
                curves["context"]
                == context
            ]
            .sort_values(
                "penalty_percent"
            )
        )

        plt.plot(
            group["penalty_percent"],
            group["mean_x"],
            marker="o",
            label=context.replace(
                "_",
                " ",
            ),
        )

    plt.axhline(
        50,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "Efficiency penalty for "
        "Organization X (%)"
    )

    plt.ylabel(
        "Mean allocation to "
        "Organization X"
    )

    plt.title(
        "Guardian Lens V4 "
        "Cost-Response Curves"
    )

    plt.ylim(
        -2,
        102,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / "v4_cost_response_main.png",
        dpi=300,
    )

    plt.close()

    target_contexts = [
        "cue_bound_target_clean",
        "cue_bound_target_modified",
    ]

    plt.figure(
        figsize=(8, 5.5)
    )

    for context in target_contexts:
        group = (
            curves[
                curves["context"]
                == context
            ]
            .sort_values(
                "penalty_percent"
            )
        )

        lower = (
            group["mean_x"]
            - group["ci_95_low"]
        )

        upper = (
            group["ci_95_high"]
            - group["mean_x"]
        )

        plt.errorbar(
            group["penalty_percent"],
            group["mean_x"],
            yerr=[
                lower,
                upper,
            ],
            marker="o",
            capsize=3,
            label=context.replace(
                "_",
                " ",
            ),
        )

    plt.axhline(
        50,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "Efficiency penalty for "
        "Organization X (%)"
    )

    plt.ylabel(
        "Mean allocation to "
        "Organization X"
    )

    plt.title(
        "Cue-Bound Target "
        "Cost Response"
    )

    plt.ylim(
        -2,
        102,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIG_DIR
        / "v4_cue_bound_target.png",
        dpi=300,
    )

    plt.close()


def main():
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    (
        df,
        all_records,
        freeze,
    ) = load_and_validate()

    print(
        "Frozen raw-data validation: PASS"
    )

    print(
        f"Successful analytical calls: "
        f"{len(df)}"
    )

    cell = build_cell_means(
        df
    )

    print(
        f"Scene-level cells: {len(cell)}"
    )

    h5, h6, h7 = (
        hypothesis_differences(
            cell
        )
    )

    confirmatory = (
        analyze_hypotheses(
            h5,
            h6,
            h7,
            rng,
        )
    )

    curves = build_cost_curves(
        cell,
        rng,
    )

    switching = (
        build_switching_points(
            curves
        )
    )

    sacrificed = (
        sacrificed_benefit_summary(
            cell
        )
    )

    clear_subtle = (
        clear_subtle_summary(
            cell,
            rng,
        )
    )

    tolerance = (
        cost_tolerance_index(
            cell
        )
    )

    clean_columns = [
        "job_id",
        "scene_id",
        "domain",
        "condition",
        "profile",
        "image_variant",
        "x_efficiency",
        "y_efficiency",
        "efficiency_penalty_percent",
        "repetition",
        "model_id",
        "temperature",
        "candidate_count",
        "timestamp_utc",
        "x",
        "y",
        "estimated_cost_usd",
        "status",
    ]

    df[
        clean_columns
    ].to_csv(
        OUT_DIR
        / "successful_jobs.csv",
        index=False,
    )

    cell.to_csv(
        OUT_DIR
        / "scene_cell_means.csv",
        index=False,
    )

    confirmatory.to_csv(
        OUT_DIR
        / "confirmatory_results.csv",
        index=False,
    )

    curves.to_csv(
        OUT_DIR
        / "cost_response_summary.csv",
        index=False,
    )

    switching.to_csv(
        OUT_DIR
        / "switching_points.csv",
        index=False,
    )

    sacrificed.to_csv(
        OUT_DIR
        / "sacrificed_benefit_summary.csv",
        index=False,
    )

    clear_subtle.to_csv(
        OUT_DIR
        / "clear_subtle_summary.csv",
        index=False,
    )

    tolerance.to_csv(
        OUT_DIR
        / "cost_tolerance_index.csv",
        index=False,
    )

    make_figures(
        curves
    )

    current_commit = (
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=ROOT,
            text=True,
        )
        .strip()
    )

    metadata = {
        "analysis_status": (
            "COMPLETE"
        ),
        "raw_sha256": (
            sha256_file(
                RAW_PATH
            )
        ),
        "analysis_plan_sha256": (
            sha256_file(
                PLAN_PATH
            )
        ),
        "git_commit_at_analysis": (
            current_commit
        ),
        "bootstrap_seed": (
            BOOTSTRAP_SEED
        ),
        "bootstrap_repetitions": (
            BOOTSTRAP_REPS
        ),
        "successful_jobs": (
            len(df)
        ),
        "procedural_error_attempts": (
            sum(
                1
                for r in all_records
                if r.get("status")
                == "error"
            )
        ),
        "invalid_responses": (
            sum(
                1
                for r in all_records
                if r.get("status")
                == "invalid_response"
            )
        ),
        "scene_level_cells": (
            len(cell)
        ),
        "confirmatory_hypotheses": (
            ["H5", "H6", "H7"]
        ),
        "multiple_comparison_method": (
            "Holm"
        ),
        "familywise_alpha": (
            0.05
        ),
    }

    (
        OUT_DIR
        / "analysis_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "CONFIRMATORY V4 RESULTS"
    )
    print(
        "======================="
    )

    print(
        confirmatory.to_string(
            index=False
        )
    )

    print()
    print(
        "Analysis outputs written to:"
    )
    print(
        OUT_DIR.relative_to(ROOT)
    )


if __name__ == "__main__":
    main()