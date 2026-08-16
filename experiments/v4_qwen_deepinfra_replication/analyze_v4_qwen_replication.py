from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

EXP = (
    ROOT
    / "experiments"
    / "v4_qwen_deepinfra_replication"
)

RAW_PATH = (
    EXP
    / "outputs"
    / "v4_qwen_deepinfra_raw.jsonl"
)

MANIFEST_PATH = (
    EXP
    / "v4_qwen_deepinfra_manifest.csv"
)

RAW_FREEZE_PATH = (
    EXP
    / "V4_QWEN_RAW_DATA_FREEZE.json"
)

REPLICATION_PLAN_PATH = (
    EXP
    / "REPLICATION_ANALYSIS_PLAN.md"
)

ORIGINAL_EXP = (
    ROOT
    / "experiments"
    / "v4_cost_response"
)

ORIGINAL_PLAN_PATH = (
    ORIGINAL_EXP
    / "ANALYSIS_PLAN.md"
)

ORIGINAL_ANALYZER_PATH = (
    ORIGINAL_EXP
    / "analyze_v4_cost_response.py"
)

OUT_DIR = EXP / "analysis"
FIG_DIR = OUT_DIR / "figures"

EXPECTED_RAW_SHA256 = (
    "9a215ddd6fb70eac93423ea5ba1e389813a0c1a3be2ab40b6be3e649aa343e6e"
)

EXPECTED_MANIFEST_SHA256 = (
    "e372b4a1ab6def665b795818d745534749e03f18f4e7fdfcaed6c642c2aa964d"
)

EXPECTED_REPLICATION_PLAN_SHA256 = (
    "203be8ebafa536c4c5bc53501d619397ef81577d9153e72d67fe44d62fb262bc"
)

EXPECTED_ORIGINAL_PLAN_SHA256 = (
    "2c0ae6d4b2240cf6335f4c6df0d7285146a6dae7ad77a405e8a10a0d8a73e354"
)

EXPECTED_ORIGINAL_ANALYZER_SHA256 = (
    "5fc4f08461c9eb18e4923ec319a4425818711f05941149af88b8fb750dcf667e"
)

EXPECTED_PROVIDER = "deepinfra"
EXPECTED_MODEL = "Qwen/Qwen3.6-27B"

EXPECTED_RAW_RECORDS = 1300
EXPECTED_SUBSTANTIVE = 1296
EXPECTED_PROCEDURAL_ERRORS = 4
EXPECTED_INVALID = 0
EXPECTED_CELLS = 432

EXPECTED_SCENES = 12
EXPECTED_PROFILES = {
    "neutral",
    "cue_bound",
    "generalized",
}
EXPECTED_IMAGE_VARIANTS = {
    "clean",
    "modified",
}
EXPECTED_EFFICIENCIES = {
    "1.00",
    "0.90",
    "0.80",
    "0.60",
    "0.40",
    "0.20",
}
EXPECTED_REPETITIONS = {
    "1",
    "2",
    "3",
}

SCIENTIFIC_FIELDS = [
    "scene_id",
    "panel",
    "domain",
    "condition",
    "profile",
    "system_prompt_path",
    "image_variant",
    "image_path",
    "image_sha256",
    "cost_level",
    "x_efficiency",
    "y_efficiency",
    "efficiency_penalty_percent",
    "task_prompt_path",
    "task_prompt_sha256",
    "system_prompt_sha256",
    "repetition",
]

CLEAN_COLUMNS = [
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
        for line_number, line in enumerate(
            handle,
            1,
        ):
            if not line.strip():
                continue

            try:
                records.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {exc}"
                ) from exc

    return records


def git_head() -> str:
    return (
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


def load_frozen_v4_analyzer():
    actual_hash = sha256_file(
        ORIGINAL_ANALYZER_PATH
    )

    if actual_hash != (
        EXPECTED_ORIGINAL_ANALYZER_SHA256
    ):
        raise ValueError(
            "ORIGINAL V4 ANALYZER HASH "
            f"MISMATCH: {actual_hash}"
        )

    spec = (
        importlib.util
        .spec_from_file_location(
            "frozen_v4_analysis",
            ORIGINAL_ANALYZER_PATH,
        )
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Unable to load frozen V4 analyzer"
        )

    module = (
        importlib.util.module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


def validate_and_normalize():
    checks = {
        "raw_sha256": sha256_file(
            RAW_PATH
        ),
        "manifest_sha256": sha256_file(
            MANIFEST_PATH
        ),
        "replication_analysis_plan_sha256":
            sha256_file(
                REPLICATION_PLAN_PATH
            ),
        "original_v4_analysis_plan_sha256":
            sha256_file(
                ORIGINAL_PLAN_PATH
            ),
        "original_v4_analyzer_sha256":
            sha256_file(
                ORIGINAL_ANALYZER_PATH
            ),
    }

    expected = {
        "raw_sha256":
            EXPECTED_RAW_SHA256,
        "manifest_sha256":
            EXPECTED_MANIFEST_SHA256,
        "replication_analysis_plan_sha256":
            EXPECTED_REPLICATION_PLAN_SHA256,
        "original_v4_analysis_plan_sha256":
            EXPECTED_ORIGINAL_PLAN_SHA256,
        "original_v4_analyzer_sha256":
            EXPECTED_ORIGINAL_ANALYZER_SHA256,
    }

    for key, expected_value in (
        expected.items()
    ):
        if checks[key] != expected_value:
            raise ValueError(
                f"{key} mismatch: "
                f"{checks[key]} != "
                f"{expected_value}"
            )

    freeze = json.loads(
        RAW_FREEZE_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    if freeze.get(
        "analysis_started"
    ) is not False:
        raise ValueError(
            "Raw freeze does not state "
            "analysis_started=false"
        )

    if freeze.get(
        "raw_sha256"
    ) != EXPECTED_RAW_SHA256:
        raise ValueError(
            "Raw freeze SHA mismatch"
        )

    records = read_jsonl(
        RAW_PATH
    )

    if len(records) != EXPECTED_RAW_RECORDS:
        raise ValueError(
            f"Expected "
            f"{EXPECTED_RAW_RECORDS} "
            f"raw records, found "
            f"{len(records)}"
        )

    substantive = [
        r
        for r in records
        if r.get("record_type")
        == "substantive_response"
    ]

    procedural = [
        r
        for r in records
        if r.get("record_type")
        == "procedural_error"
    ]

    unknown_types = [
        r
        for r in records
        if r.get("record_type")
        not in {
            "substantive_response",
            "procedural_error",
        }
    ]

    if unknown_types:
        raise ValueError(
            "Unexpected raw record types"
        )

    if len(substantive) != (
        EXPECTED_SUBSTANTIVE
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_SUBSTANTIVE} "
            "substantive responses, "
            f"found {len(substantive)}"
        )

    if len(procedural) != (
        EXPECTED_PROCEDURAL_ERRORS
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_PROCEDURAL_ERRORS} "
            "procedural-error records, "
            f"found {len(procedural)}"
        )

    invalid = [
        r
        for r in substantive
        if r.get(
            "allocation_validity"
        )
        != "valid"
    ]

    if len(invalid) != EXPECTED_INVALID:
        raise ValueError(
            f"Expected {EXPECTED_INVALID} "
            "invalid substantive "
            f"responses, found "
            f"{len(invalid)}"
        )

    job_ids = [
        str(r["job_id"])
        for r in substantive
    ]

    if len(set(job_ids)) != (
        EXPECTED_SUBSTANTIVE
    ):
        raise ValueError(
            "Substantive job IDs "
            "are not unique"
        )

    manifest = pd.read_csv(
        MANIFEST_PATH,
        dtype=str,
    )

    if len(manifest) != (
        EXPECTED_SUBSTANTIVE
    ):
        raise ValueError(
            "Replication manifest does "
            "not contain 1,296 rows"
        )

    if set(
        manifest["job_id"]
    ) != set(job_ids):
        raise ValueError(
            "Substantive jobs do not "
            "exactly match manifest IDs"
        )

    manifest_by_job = (
        manifest
        .set_index("job_id")
        .to_dict("index")
    )

    for record in substantive:
        job_id = str(
            record["job_id"]
        )

        manifest_row = (
            manifest_by_job[
                job_id
            ]
        )

        for field in SCIENTIFIC_FIELDS:
            raw_value = str(
                record.get(field)
            )

            manifest_value = str(
                manifest_row.get(field)
            )

            if raw_value != manifest_value:
                raise ValueError(
                    f"Scientific field "
                    f"mismatch for "
                    f"{job_id}: "
                    f"{field}: "
                    f"{raw_value} != "
                    f"{manifest_value}"
                )

        if record.get(
            "provider"
        ) != EXPECTED_PROVIDER:
            raise ValueError(
                f"Provider mismatch in "
                f"{job_id}"
            )

        if record.get(
            "model_id"
        ) != EXPECTED_MODEL:
            raise ValueError(
                f"Model mismatch in "
                f"{job_id}"
            )

        x = float(
            record["parsed_x"]
        )

        y = float(
            record["parsed_y"]
        )

        if not (
            math.isfinite(x)
            and math.isfinite(y)
        ):
            raise ValueError(
                f"Non-finite allocation "
                f"in {job_id}"
            )

        if not (
            0.0 <= x <= 100.0
            and 0.0 <= y <= 100.0
        ):
            raise ValueError(
                f"Out-of-range allocation "
                f"in {job_id}"
            )

        if not math.isclose(
            x + y,
            100.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"Allocation does not "
                f"sum to 100 in {job_id}"
            )

    # Verify retries never changed the
    # scientific request payload.
    payload_hashes = defaultdict(set)
    attempts = defaultdict(list)

    for record in records:
        job_id = str(
            record["job_id"]
        )

        payload = record.get(
            "request_payload_sha256"
        )

        if payload:
            payload_hashes[
                job_id
            ].add(payload)

        attempts[
            job_id
        ].append(
            int(record["attempt"])
        )

    for job_id, hashes in (
        payload_hashes.items()
    ):
        if len(hashes) != 1:
            raise ValueError(
                f"Request payload changed "
                f"across retries for "
                f"{job_id}"
            )

    for job_id, observed in (
        attempts.items()
    ):
        observed = sorted(
            observed
        )

        expected_attempts = list(
            range(
                1,
                max(observed) + 1,
            )
        )

        if observed != expected_attempts:
            raise ValueError(
                f"Non-contiguous attempts "
                f"for {job_id}: "
                f"{observed}"
            )

    scenes = {
        str(r["scene_id"])
        for r in substantive
    }

    profiles = {
        str(r["profile"])
        for r in substantive
    }

    variants = {
        str(r["image_variant"])
        for r in substantive
    }

    efficiencies = {
        str(r["x_efficiency"])
        for r in substantive
    }

    repetitions = {
        str(r["repetition"])
        for r in substantive
    }

    if len(scenes) != EXPECTED_SCENES:
        raise ValueError(
            f"Expected 12 scenes, "
            f"found {len(scenes)}"
        )

    if profiles != EXPECTED_PROFILES:
        raise ValueError(
            f"Profile set mismatch: "
            f"{profiles}"
        )

    if variants != (
        EXPECTED_IMAGE_VARIANTS
    ):
        raise ValueError(
            f"Image-variant mismatch: "
            f"{variants}"
        )

    if efficiencies != (
        EXPECTED_EFFICIENCIES
    ):
        raise ValueError(
            f"Efficiency-level mismatch: "
            f"{efficiencies}"
        )

    if repetitions != (
        EXPECTED_REPETITIONS
    ):
        raise ValueError(
            f"Repetition set mismatch: "
            f"{repetitions}"
        )

    cell_repetitions = defaultdict(
        list
    )

    for record in substantive:
        key = (
            str(record["scene_id"]),
            str(record["profile"]),
            str(record["image_variant"]),
            str(record["x_efficiency"]),
        )

        cell_repetitions[
            key
        ].append(
            str(record["repetition"])
        )

    if len(cell_repetitions) != (
        EXPECTED_CELLS
    ):
        raise ValueError(
            f"Expected "
            f"{EXPECTED_CELLS} "
            "scientific cells, "
            f"found "
            f"{len(cell_repetitions)}"
        )

    for key, reps in (
        cell_repetitions.items()
    ):
        if (
            len(reps) != 3
            or set(reps)
            != EXPECTED_REPETITIONS
        ):
            raise ValueError(
                f"Incomplete cell "
                f"{key}: {reps}"
            )

    # Convert only the Qwen ledger schema
    # into the column schema expected by
    # the frozen V4 analysis functions.
    normalized = []

    for record in substantive:
        row = dict(record)

        row["x"] = float(
            record["parsed_x"]
        )

        row["y"] = float(
            record["parsed_y"]
        )

        usage = (
            record.get("usage")
            or {}
        )

        row[
            "estimated_cost_usd"
        ] = usage.get(
            "estimated_cost"
        )

        row["status"] = "ok"

        normalized.append(row)

    df = pd.DataFrame(
        normalized
    )

    df["x"] = pd.to_numeric(
        df["x"]
    )

    df["y"] = pd.to_numeric(
        df["y"]
    )

    df["x_efficiency"] = (
        pd.to_numeric(
            df["x_efficiency"]
        )
    )

    df[
        "efficiency_penalty_percent"
    ] = pd.to_numeric(
        df[
            "efficiency_penalty_percent"
        ]
    )

    df["repetition"] = (
        pd.to_numeric(
            df["repetition"]
        )
    )

    validation = {
        **checks,
        "raw_records":
            len(records),
        "substantive_jobs":
            len(substantive),
        "procedural_error_records":
            len(procedural),
        "invalid_substantive_responses":
            len(invalid),
        "scientific_cells":
            len(cell_repetitions),
        "scenes":
            len(scenes),
        "profiles":
            len(profiles),
        "image_variants":
            len(variants),
        "efficiency_levels":
            len(efficiencies),
        "repetitions_per_cell":
            3,
    }

    return (
        df,
        records,
        validation,
    )


def write_results_markdown(
    confirmatory: pd.DataFrame,
    switching: pd.DataFrame,
) -> None:
    lines = [
        "# Guardian Lens V4 Qwen/DeepInfra Replication Results",
        "",
        f"Provider: `{EXPECTED_PROVIDER}`",
        f"Model: `{EXPECTED_MODEL}`",
        "",
        "## Confirmatory H5-H7 results",
        "",
        (
            "| Hypothesis | n | Mean paired difference | "
            "95% CI | Raw p | Holm p | Supported |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---|"
        ),
    ]

    for _, row in (
        confirmatory.iterrows()
    ):
        lines.append(
            "| "
            f"{row['hypothesis']} | "
            f"{int(row['n_scenes'])} | "
            f"{row['mean_paired_difference']:.6f} | "
            f"[{row['ci_95_low']:.6f}, "
            f"{row['ci_95_high']:.6f}] | "
            f"{row['exact_one_sided_p']:.6f} | "
            f"{row['holm_adjusted_p']:.6f} | "
            f"{bool(row['supported_at_fwer_0.05'])} |"
        )

    lines.extend(
        [
            "",
            "## Descriptive switching-point summaries",
            "",
        ]
    )

    for _, row in switching.iterrows():
        lines.append(
            f"- `{row['context']}`: "
            f"{row['interpretation']}"
        )

    lines.extend(
        [
            "",
            (
                "The Qwen replication is analyzed separately "
                "from the original Gemini V4 experiment and "
                "from the earlier 216-call DeepInfra experiment."
            ),
            "",
            (
                "Cross-model comparisons are descriptive; "
                "no post-hoc between-model significance test "
                "is introduced."
            ),
            "",
        ]
    )

    (
        OUT_DIR
        / "REPLICATION_RESULTS.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_output_hashes() -> None:
    hash_path = (
        OUT_DIR
        / "REPLICATION_ANALYSIS_OUTPUTS.sha256"
    )

    files = sorted(
        path
        for path in OUT_DIR.rglob("*")
        if (
            path.is_file()
            and path != hash_path
        )
    )

    lines = []

    for path in files:
        relative = (
            path
            .relative_to(
                OUT_DIR
            )
            .as_posix()
        )

        lines.append(
            f"{sha256_file(path)}  "
            f"{relative}"
        )

    hash_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def run_analysis(
    df: pd.DataFrame,
    all_records: list[dict],
    validation: dict,
) -> None:
    # Import the frozen original V4
    # implementation and reuse its
    # scientific analysis functions
    # directly.
    v4 = load_frozen_v4_analyzer()

    if v4.BOOTSTRAP_REPS != 20_000:
        raise ValueError(
            "Frozen V4 bootstrap "
            "replicate count changed"
        )

    if v4.BOOTSTRAP_SEED != 20260815:
        raise ValueError(
            "Frozen V4 bootstrap "
            "seed changed"
        )

    if v4.PENALIZED_LEVELS != {
        0.90,
        0.80,
        0.60,
        0.40,
        0.20,
    }:
        raise ValueError(
            "Frozen V4 penalized "
            "levels changed"
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Redirect only outputs. The original
    # V4 directory itself is never written.
    v4.OUT_DIR = OUT_DIR
    v4.FIG_DIR = FIG_DIR

    # Preserve the original V4 RNG seed
    # and function call order.
    rng = np.random.default_rng(
        v4.BOOTSTRAP_SEED
    )

    cell = v4.build_cell_means(
        df
    )

    h5, h6, h7 = (
        v4.hypothesis_differences(
            cell
        )
    )

    confirmatory = (
        v4.analyze_hypotheses(
            h5,
            h6,
            h7,
            rng,
        )
    )

    curves = v4.build_cost_curves(
        cell,
        rng,
    )

    switching = (
        v4.build_switching_points(
            curves
        )
    )

    sacrificed = (
        v4.sacrificed_benefit_summary(
            cell
        )
    )

    clear_subtle = (
        v4.clear_subtle_summary(
            cell,
            rng,
        )
    )

    tolerance = (
        v4.cost_tolerance_index(
            cell
        )
    )

    df[
        CLEAN_COLUMNS
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

    (
        h5
        .reset_index()
        .to_csv(
            OUT_DIR
            / "h5_scene_effects.csv",
            index=False,
        )
    )

    (
        h6
        .reset_index()
        .to_csv(
            OUT_DIR
            / "h6_scene_effects.csv",
            index=False,
        )
    )

    (
        h7
        .reset_index()
        .to_csv(
            OUT_DIR
            / "h7_scene_effects.csv",
            index=False,
        )
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

    v4.make_figures(
        curves
    )

    metadata = {
        "analysis_status":
            "COMPLETE",
        "experiment":
            (
                "Guardian Lens V4 "
                "Qwen/DeepInfra full replication"
            ),
        "provider":
            EXPECTED_PROVIDER,
        "model_id":
            EXPECTED_MODEL,
        "raw_sha256":
            validation[
                "raw_sha256"
            ],
        "manifest_sha256":
            validation[
                "manifest_sha256"
            ],
        "replication_analysis_plan_sha256":
            validation[
                "replication_analysis_plan_sha256"
            ],
        "original_v4_analysis_plan_sha256":
            validation[
                "original_v4_analysis_plan_sha256"
            ],
        "original_v4_analyzer_sha256":
            validation[
                "original_v4_analyzer_sha256"
            ],
        "replication_analyzer_sha256":
            sha256_file(
                Path(__file__)
            ),
        "git_commit_at_analysis":
            git_head(),
        "bootstrap_seed":
            v4.BOOTSTRAP_SEED,
        "bootstrap_repetitions":
            v4.BOOTSTRAP_REPS,
        "raw_records":
            validation[
                "raw_records"
            ],
        "successful_jobs":
            validation[
                "substantive_jobs"
            ],
        "procedural_error_attempts":
            validation[
                "procedural_error_records"
            ],
        "invalid_responses":
            validation[
                "invalid_substantive_responses"
            ],
        "scene_level_cells":
            len(cell),
        "confirmatory_hypotheses":
            [
                "H5",
                "H6",
                "H7",
            ],
        "multiple_comparison_method":
            "Holm",
        "familywise_alpha":
            0.05,
        "scientific_analysis_source":
            (
                "Frozen original V4 "
                "analysis functions reused directly"
            ),
        "separate_from_step4_216_call_run":
            True,
    }

    (
        OUT_DIR
        / "replication_analysis_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    write_results_markdown(
        confirmatory,
        switching,
    )

    write_output_hashes()

    print()
    print(
        "CONFIRMATORY QWEN V4 "
        "REPLICATION RESULTS"
    )
    print(
        "=================================="
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
        OUT_DIR.relative_to(
            ROOT
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Verify frozen Qwen replication "
            "integrity without calculating "
            "H5/H6/H7 aggregate outcomes."
        ),
    )

    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help=(
            "Run the frozen full V4 "
            "Qwen replication analysis."
        ),
    )

    args = parser.parse_args()

    if (
        args.validate_only
        == args.run_analysis
    ):
        parser.error(
            "Choose exactly one of "
            "--validate-only or "
            "--run-analysis."
        )

    # Loading the frozen original analyzer
    # here verifies its hash. It does not
    # execute aggregate analysis.
    load_frozen_v4_analyzer()

    (
        df,
        records,
        validation,
    ) = validate_and_normalize()

    print(
        "PASS: Qwen V4 replication "
        "structural validation."
    )
    print(
        "Raw SHA-256:",
        validation["raw_sha256"],
    )
    print(
        "Manifest SHA-256:",
        validation[
            "manifest_sha256"
        ],
    )
    print(
        "Replication plan SHA-256:",
        validation[
            "replication_analysis_plan_sha256"
        ],
    )
    print(
        "Original V4 analyzer SHA-256:",
        validation[
            "original_v4_analyzer_sha256"
        ],
    )
    print(
        "Raw records:",
        validation["raw_records"],
    )
    print(
        "Valid substantive jobs:",
        validation[
            "substantive_jobs"
        ],
    )
    print(
        "Procedural-error records:",
        validation[
            "procedural_error_records"
        ],
    )
    print(
        "Invalid substantive responses:",
        validation[
            "invalid_substantive_responses"
        ],
    )
    print(
        "Scientific cells:",
        validation[
            "scientific_cells"
        ],
    )
    print(
        "Scenes:",
        validation["scenes"],
    )
    print(
        "Efficiency levels:",
        validation[
            "efficiency_levels"
        ],
    )

    if args.validate_only:
        print(
            "VALIDATION ONLY: no aggregate "
            "H5/H6/H7 outcomes were calculated."
        )
        return 0

    run_analysis(
        df,
        records,
        validation,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
