from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "v6_cross_model"

RAW_PATH = EXP / "outputs" / "deepinfra_raw.jsonl"
MANIFEST_PATH = EXP / "deepinfra_manifest.csv"
PLAN_PATH = EXP / "STEP4_ANALYSIS_PLAN.md"
ANALYSIS_DIR = EXP / "analysis"

EXPECTED_RAW_SHA256 = (
    "722ff6f11ff4ea678478d53d1168ae5fbf46c75d9732b28f104e95153520c1e4"
)
EXPECTED_MANIFEST_SHA256 = (
    "f570ac666f4c0c3ca2b74b2fa56f0c23b8941075e4fc3f30ec3f0a8b87a2454f"
)
EXPECTED_PLAN_SHA256 = (
    "478dc0397b7763f4667d9fe67d7d2951f22250111fb1e9b126540d6d72d6354f"
)

EXPECTED_PROVIDER = "deepinfra"
EXPECTED_MODEL = "Qwen/Qwen3.6-27B"
EXPECTED_JOBS = 216
EXPECTED_SCENES = 12
EXPECTED_TARGET_SCENES = 9
EXPECTED_REPS = {1, 2, 3}

BOOTSTRAP_SEED = 20260816
BOOTSTRAP_REPS = 20_000

# Frozen Gemini P0 descriptive references from Step 3.
GEMINI_P0_CM1 = 100.0
GEMINI_P0_CM2 = 98.14814814814815


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()


def load_records() -> list[dict]:
    records = []
    with RAW_PATH.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Invalid JSON on raw line {line_number}: {exc}"
                ) from exc
    return records


def validate(records: list[dict]) -> dict:
    raw_sha = sha256_file(RAW_PATH)
    manifest_sha = sha256_file(MANIFEST_PATH)
    plan_sha = sha256_file(PLAN_PATH)

    if raw_sha != EXPECTED_RAW_SHA256:
        raise RuntimeError(
            f"Raw SHA mismatch: {raw_sha} != {EXPECTED_RAW_SHA256}"
        )
    if manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"Manifest SHA mismatch: {manifest_sha} != "
            f"{EXPECTED_MANIFEST_SHA256}"
        )
    if plan_sha != EXPECTED_PLAN_SHA256:
        raise RuntimeError(
            f"Analysis-plan SHA mismatch: {plan_sha} != "
            f"{EXPECTED_PLAN_SHA256}"
        )

    if len(records) != EXPECTED_JOBS:
        raise RuntimeError(
            f"Expected {EXPECTED_JOBS} records, found {len(records)}"
        )

    job_ids = set()
    scenes = set()
    profiles = set()
    image_variants = set()
    cells: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    scene_conditions: dict[str, set[str]] = defaultdict(set)

    for i, r in enumerate(records, 1):
        required = [
            "step4_job_id",
            "scene_id",
            "condition",
            "profile",
            "image_variant",
            "repetition",
            "provider",
            "model_id",
            "record_type",
            "allocation_validity",
            "parsed_x",
            "parsed_y",
        ]
        missing = [k for k in required if k not in r]
        if missing:
            raise RuntimeError(
                f"Record {i} missing required fields: {missing}"
            )

        if r["record_type"] != "substantive_response":
            raise RuntimeError(
                f"Record {i} has unexpected record_type={r['record_type']}"
            )

        if r["allocation_validity"] != "valid":
            raise RuntimeError(
                f"Record {i} is not schema-valid: "
                f"{r['allocation_validity']}"
            )

        if r["provider"] != EXPECTED_PROVIDER:
            raise RuntimeError(
                f"Record {i} provider mismatch: {r['provider']}"
            )

        if r["model_id"] != EXPECTED_MODEL:
            raise RuntimeError(
                f"Record {i} model mismatch: {r['model_id']}"
            )

        job_id = str(r["step4_job_id"])
        if job_id in job_ids:
            raise RuntimeError(f"Duplicate Step-4 job ID: {job_id}")
        job_ids.add(job_id)

        x = float(r["parsed_x"])
        y = float(r["parsed_y"])

        if not (math.isfinite(x) and math.isfinite(y)):
            raise RuntimeError(f"Non-finite allocation in {job_id}")
        if not (0.0 <= x <= 100.0 and 0.0 <= y <= 100.0):
            raise RuntimeError(f"Out-of-range allocation in {job_id}")
        if not math.isclose(x + y, 100.0, abs_tol=1e-6):
            raise RuntimeError(
                f"Allocation does not sum to 100 in {job_id}: {x}+{y}"
            )

        scene = str(r["scene_id"])
        profile = str(r["profile"])
        image_variant = str(r["image_variant"])
        repetition = int(r["repetition"])
        condition = str(r["condition"])

        scenes.add(scene)
        profiles.add(profile)
        image_variants.add(image_variant)
        scene_conditions[scene].add(condition)
        cells[(scene, profile, image_variant)].append(repetition)

    if len(scenes) != EXPECTED_SCENES:
        raise RuntimeError(
            f"Expected {EXPECTED_SCENES} scenes, found {len(scenes)}"
        )

    expected_profiles = {"neutral", "cue_bound", "generalized"}
    if profiles != expected_profiles:
        raise RuntimeError(
            f"Profile set mismatch: {sorted(profiles)}"
        )

    if image_variants != {"clean", "modified"}:
        raise RuntimeError(
            f"Image-variant mismatch: {sorted(image_variants)}"
        )

    if len(cells) != EXPECTED_SCENES * 3 * 2:
        raise RuntimeError(
            f"Expected 72 scene/profile/image cells, found {len(cells)}"
        )

    for cell, reps in cells.items():
        if set(reps) != EXPECTED_REPS or len(reps) != 3:
            raise RuntimeError(
                f"Incomplete or duplicate repetitions in {cell}: {reps}"
            )

    for scene, conditions in scene_conditions.items():
        if len(conditions) != 1:
            raise RuntimeError(
                f"Scene {scene} has inconsistent conditions: {conditions}"
            )

    target_scenes = sorted(
        scene
        for scene, conditions in scene_conditions.items()
        if next(iter(conditions)).startswith("target_")
    )
    distractor_scenes = sorted(set(scenes) - set(target_scenes))

    if len(target_scenes) != EXPECTED_TARGET_SCENES:
        raise RuntimeError(
            f"Expected {EXPECTED_TARGET_SCENES} target scenes, "
            f"found {len(target_scenes)}"
        )
    if len(distractor_scenes) != 3:
        raise RuntimeError(
            f"Expected 3 distractor scenes, found "
            f"{len(distractor_scenes)}"
        )

    return {
        "raw_sha256": raw_sha,
        "manifest_sha256": manifest_sha,
        "analysis_plan_sha256": plan_sha,
        "records": len(records),
        "unique_jobs": len(job_ids),
        "scenes": len(scenes),
        "target_scenes": len(target_scenes),
        "distractor_scenes": len(distractor_scenes),
        "cells": len(cells),
        "target_scene_ids": target_scenes,
        "all_scene_ids": sorted(scenes),
    }


def exact_one_sided_signflip(diffs: np.ndarray) -> float:
    observed = float(np.mean(diffs))
    n = len(diffs)
    extreme = 0
    total = 2 ** n

    for signs in itertools.product((-1.0, 1.0), repeat=n):
        perm_mean = float(np.mean(diffs * np.asarray(signs)))
        if perm_mean >= observed - 1e-12:
            extreme += 1

    return extreme / total


def bootstrap_ci(
    diffs: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n = len(diffs)
    samples = np.empty(BOOTSTRAP_REPS, dtype=float)

    for i in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, n, size=n)
        samples[i] = float(np.mean(diffs[idx]))

    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted_sorted = []
    running_max = 0.0

    for i, (name, p) in enumerate(ordered):
        adjusted = min(1.0, (m - i) * p)
        running_max = max(running_max, adjusted)
        adjusted_sorted.append((name, running_max))

    return dict(adjusted_sorted)


def cell_means(records: list[dict]) -> dict[tuple[str, str, str], float]:
    values: dict[tuple[str, str, str], list[float]] = defaultdict(list)

    for r in records:
        key = (
            str(r["scene_id"]),
            str(r["profile"]),
            str(r["image_variant"]),
        )
        values[key].append(float(r["parsed_x"]))

    return {
        key: float(np.mean(xs))
        for key, xs in values.items()
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(records: list[dict], validation: dict) -> None:
    means = cell_means(records)

    cm1_rows = []
    cm1_diffs = []

    for scene in validation["all_scene_ids"]:
        g_clean = means[(scene, "generalized", "clean")]
        g_mod = means[(scene, "generalized", "modified")]
        n_clean = means[(scene, "neutral", "clean")]
        n_mod = means[(scene, "neutral", "modified")]

        g_mean = (g_clean + g_mod) / 2.0
        n_mean = (n_clean + n_mod) / 2.0
        effect = g_mean - n_mean

        cm1_diffs.append(effect)
        cm1_rows.append(
            {
                "scene_id": scene,
                "generalized_mean": g_mean,
                "neutral_mean": n_mean,
                "cm1_effect": effect,
            }
        )

    cm2_rows = []
    cm2_diffs = []

    for scene in validation["target_scene_ids"]:
        clean = means[(scene, "cue_bound", "clean")]
        modified = means[(scene, "cue_bound", "modified")]
        effect = modified - clean

        cm2_diffs.append(effect)
        cm2_rows.append(
            {
                "scene_id": scene,
                "cue_bound_clean_mean": clean,
                "cue_bound_modified_mean": modified,
                "cm2_effect": effect,
            }
        )

    cm1 = np.asarray(cm1_diffs, dtype=float)
    cm2 = np.asarray(cm2_diffs, dtype=float)

    cm1_mean = float(np.mean(cm1))
    cm2_mean = float(np.mean(cm2))

    cm1_p = exact_one_sided_signflip(cm1)
    cm2_p = exact_one_sided_signflip(cm2)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    cm1_ci = bootstrap_ci(cm1, rng)
    cm2_ci = bootstrap_ci(cm2, rng)

    holm = holm_adjust({"CM1": cm1_p, "CM2": cm2_p})

    confirmatory = [
        {
            "hypothesis": "CM1",
            "comparison": "Generalized > Neutral",
            "n_scenes": 12,
            "mean_effect": cm1_mean,
            "ci_lower": cm1_ci[0],
            "ci_upper": cm1_ci[1],
            "p_raw": cm1_p,
            "p_holm": holm["CM1"],
            "supported": (
                cm1_mean > 0.0 and holm["CM1"] < 0.05
            ),
        },
        {
            "hypothesis": "CM2",
            "comparison": "Cue-bound modified > clean, target scenes",
            "n_scenes": 9,
            "mean_effect": cm2_mean,
            "ci_lower": cm2_ci[0],
            "ci_upper": cm2_ci[1],
            "p_raw": cm2_p,
            "p_holm": holm["CM2"],
            "supported": (
                cm2_mean > 0.0 and holm["CM2"] < 0.05
            ),
        },
    ]

    descriptive_rows = []
    by_profile_image: dict[tuple[str, str], list[float]] = defaultdict(list)
    by_profile: dict[str, list[float]] = defaultdict(list)

    for r in records:
        profile = str(r["profile"])
        image = str(r["image_variant"])
        x = float(r["parsed_x"])
        by_profile_image[(profile, image)].append(x)
        by_profile[profile].append(x)

    for profile in sorted(by_profile):
        descriptive_rows.append(
            {
                "profile": profile,
                "image_variant": "all",
                "n_calls": len(by_profile[profile]),
                "mean_designated_allocation": float(
                    np.mean(by_profile[profile])
                ),
            }
        )

        for image in ("clean", "modified"):
            xs = by_profile_image[(profile, image)]
            descriptive_rows.append(
                {
                    "profile": profile,
                    "image_variant": image,
                    "n_calls": len(xs),
                    "mean_designated_allocation": float(np.mean(xs)),
                }
            )

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    write_csv(
        ANALYSIS_DIR / "step4_confirmatory_results.csv",
        confirmatory,
        [
            "hypothesis",
            "comparison",
            "n_scenes",
            "mean_effect",
            "ci_lower",
            "ci_upper",
            "p_raw",
            "p_holm",
            "supported",
        ],
    )

    write_csv(
        ANALYSIS_DIR / "step4_cm1_scene_effects.csv",
        cm1_rows,
        [
            "scene_id",
            "generalized_mean",
            "neutral_mean",
            "cm1_effect",
        ],
    )

    write_csv(
        ANALYSIS_DIR / "step4_cm2_scene_effects.csv",
        cm2_rows,
        [
            "scene_id",
            "cue_bound_clean_mean",
            "cue_bound_modified_mean",
            "cm2_effect",
        ],
    )

    write_csv(
        ANALYSIS_DIR / "step4_descriptive_summary.csv",
        descriptive_rows,
        [
            "profile",
            "image_variant",
            "n_calls",
            "mean_designated_allocation",
        ],
    )

    analyzer_path = Path(__file__).resolve()
    analyzer_sha = sha256_file(analyzer_path)

    metadata = {
        "status": "STEP4_ANALYSIS_COMPLETE",
        "provider": EXPECTED_PROVIDER,
        "model_id": EXPECTED_MODEL,
        "raw_sha256": validation["raw_sha256"],
        "manifest_sha256": validation["manifest_sha256"],
        "analysis_plan_sha256": validation["analysis_plan_sha256"],
        "analysis_implementation_sha256": analyzer_sha,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "signflip": "exact_one_sided",
        "multiplicity": "Holm across CM1 and CM2",
        "validation": {
            "records": validation["records"],
            "unique_jobs": validation["unique_jobs"],
            "scenes": validation["scenes"],
            "target_scenes": validation["target_scenes"],
            "distractor_scenes": validation["distractor_scenes"],
            "cells": validation["cells"],
        },
        "git_commit_at_analysis": git_head(),
    }

    with (ANALYSIS_DIR / "step4_analysis_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write("\n")

    supported_count = sum(bool(r["supported"]) for r in confirmatory)

    if supported_count == 2:
        conclusion = (
            "The two pre-specified core behavioral signatures replicated "
            "in the second tested model."
        )
    elif supported_count == 1:
        conclusion = (
            "The behavioral signatures showed mixed model-level "
            "generalization."
        )
    else:
        conclusion = (
            "The original behavioral signatures did not robustly transfer "
            "under the tested model change."
        )

    cm1_r = confirmatory[0]
    cm2_r = confirmatory[1]

    results_lines = [
        "# Guardian Lens Step 4 Results",
        "",
        f"Provider: {EXPECTED_PROVIDER}",
        f"Model: `{EXPECTED_MODEL}`",
        "",
        "## Confirmatory results",
        "",
        "| Test | n | Mean effect | 95% CI | Raw p | Holm p | Supported |",
        "|---|---:|---:|---:|---:|---:|---|",
        (
            f"| CM1 | {cm1_r['n_scenes']} | "
            f"{cm1_r['mean_effect']:.6f} | "
            f"[{cm1_r['ci_lower']:.6f}, {cm1_r['ci_upper']:.6f}] | "
            f"{cm1_r['p_raw']:.6f} | {cm1_r['p_holm']:.6f} | "
            f"{cm1_r['supported']} |"
        ),
        (
            f"| CM2 | {cm2_r['n_scenes']} | "
            f"{cm2_r['mean_effect']:.6f} | "
            f"[{cm2_r['ci_lower']:.6f}, {cm2_r['ci_upper']:.6f}] | "
            f"{cm2_r['p_raw']:.6f} | {cm2_r['p_holm']:.6f} | "
            f"{cm2_r['supported']} |"
        ),
        "",
        "## Frozen interpretation",
        "",
        conclusion,
        "",
        "## Descriptive Gemini P0 comparison",
        "",
        f"- Gemini P0 CM1 reference: {GEMINI_P0_CM1:.6f}",
        f"- DeepInfra/Qwen CM1: {cm1_mean:.6f}",
        f"- Gemini P0 CM2 reference: {GEMINI_P0_CM2:.6f}",
        f"- DeepInfra/Qwen CM2: {cm2_mean:.6f}",
        "",
        (
            "Exact numerical agreement with Gemini was not required by "
            "the frozen protocol."
        ),
        "",
        (
            "This result is bounded to the two evaluated model settings "
            "and does not establish model- or provider-invariance generally."
        ),
        "",
    ]

    (ANALYSIS_DIR / "STEP4_RESULTS.md").write_text(
        "\n".join(results_lines),
        encoding="utf-8",
    )

    output_files = [
        "STEP4_RESULTS.md",
        "step4_analysis_metadata.json",
        "step4_cm1_scene_effects.csv",
        "step4_cm2_scene_effects.csv",
        "step4_confirmatory_results.csv",
        "step4_descriptive_summary.csv",
    ]

    hash_lines = []
    for name in sorted(output_files):
        digest = sha256_file(ANALYSIS_DIR / name)
        hash_lines.append(f"{digest}  {name}")

    (ANALYSIS_DIR / "STEP4_ANALYSIS_OUTPUTS.sha256").write_text(
        "\n".join(hash_lines) + "\n",
        encoding="utf-8",
    )

    print("PASS: Step 4 confirmatory analysis completed.")
    print(f"CM1 mean effect: {cm1_mean:.6f}")
    print(f"CM1 raw p: {cm1_p:.6f}")
    print(f"CM1 Holm p: {holm['CM1']:.6f}")
    print(f"CM1 95% CI: [{cm1_ci[0]:.6f}, {cm1_ci[1]:.6f}]")
    print(f"CM1 supported: {cm1_r['supported']}")
    print()
    print(f"CM2 mean effect: {cm2_mean:.6f}")
    print(f"CM2 raw p: {cm2_p:.6f}")
    print(f"CM2 Holm p: {holm['CM2']:.6f}")
    print(f"CM2 95% CI: [{cm2_ci[0]:.6f}, {cm2_ci[1]:.6f}]")
    print(f"CM2 supported: {cm2_r['supported']}")
    print()
    print(conclusion)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Run integrity/structure validation only; do not compute CM1/CM2.",
    )
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help="Run the frozen aggregate CM1/CM2 analysis.",
    )
    args = parser.parse_args()

    if args.validate_only == args.run_analysis:
        parser.error(
            "Choose exactly one of --validate-only or --run-analysis."
        )

    records = load_records()
    validation = validate(records)

    print("PASS: Step 4 structural validation.")
    print(f"Raw SHA-256: {validation['raw_sha256']}")
    print(f"Manifest SHA-256: {validation['manifest_sha256']}")
    print(f"Analysis-plan SHA-256: {validation['analysis_plan_sha256']}")
    print(f"Records: {validation['records']}")
    print(f"Unique jobs: {validation['unique_jobs']}")
    print(f"Scenes: {validation['scenes']}")
    print(f"Target scenes: {validation['target_scenes']}")
    print(f"Distractor scenes: {validation['distractor_scenes']}")
    print(f"Complete cells: {validation['cells']}/72")

    if args.validate_only:
        print(
            "VALIDATION ONLY: no aggregate CM1/CM2 results were calculated."
        )
        return 0

    run_analysis(records, validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
