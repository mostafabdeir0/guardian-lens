from __future__ import annotations

import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SOURCE_MANIFEST = (
    ROOT / "data" / "images" / "heldout" / "heldout_manifest.csv"
)

EXPERIMENT_DIR = ROOT / "experiments" / "v4_cost_response"

TEMPLATE_PATH = (
    EXPERIMENT_DIR / "task_cost_response_template.txt"
)

PROMPT_DIR = EXPERIMENT_DIR / "prompts"

OUTPUT_MANIFEST = (
    EXPERIMENT_DIR / "v4_cost_response_manifest.csv"
)

OUTPUT_HASH = (
    EXPERIMENT_DIR / "v4_cost_response_manifest.sha256"
)

PROFILES = (
    "neutral",
    "cue_bound",
    "generalized",
)

SYSTEM_PROMPTS = {
    "neutral": "prompts/system_neutral.txt",
    "cue_bound": "prompts/system_cue_bound.txt",
    "generalized": "prompts/system_generalized.txt",
}

COST_LEVELS = (
    ("1", "1.00", "0"),
    ("2", "0.90", "10"),
    ("3", "0.80", "20"),
    ("4", "0.60", "40"),
    ("5", "0.40", "60"),
    ("6", "0.20", "80"),
)

REPETITIONS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_text(path: Path) -> str:
    return path.read_text(
        encoding="utf-8-sig"
    ).strip()


def write_cost_prompts() -> dict[str, Path]:
    template = read_text(TEMPLATE_PATH)

    if "{X_EFFICIENCY}" not in template:
        raise ValueError(
            "Template must contain {X_EFFICIENCY}"
        )

    PROMPT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prompt_paths: dict[str, Path] = {}

    for _, efficiency, _ in COST_LEVELS:
        filename = (
            "task_cost_x_"
            + efficiency.replace(".", "_")
            + ".txt"
        )

        path = PROMPT_DIR / filename

        prompt = template.replace(
            "{X_EFFICIENCY}",
            efficiency,
        )

        path.write_text(
            prompt + "\n",
            encoding="utf-8",
        )

        prompt_paths[efficiency] = path

    return prompt_paths


def main() -> None:
    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(
            SOURCE_MANIFEST
        )

    if not TEMPLATE_PATH.is_file():
        raise FileNotFoundError(
            TEMPLATE_PATH
        )

    with SOURCE_MANIFEST.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        scenes = list(csv.DictReader(handle))

    if len(scenes) != 12:
        raise ValueError(
            f"Expected 12 source scenes, found "
            f"{len(scenes)}"
        )

    prompt_paths = write_cost_prompts()

    rows: list[dict[str, str | int]] = []

    job_number = 0

    for scene in scenes:
        for profile in PROFILES:
            for image_variant, path_key, hash_key in (
                (
                    "clean",
                    "clean_path",
                    "clean_sha256",
                ),
                (
                    "modified",
                    "modified_path",
                    "modified_sha256",
                ),
            ):
                for (
                    cost_level,
                    x_efficiency,
                    penalty_percent,
                ) in COST_LEVELS:
                    task_prompt_path = (
                        prompt_paths[x_efficiency]
                    )

                    for repetition in range(
                        1,
                        REPETITIONS + 1,
                    ):
                        job_number += 1

                        rows.append(
                            {
                                "job_id": (
                                    f"v4_{job_number:04d}"
                                ),
                                "scene_id": (
                                    scene["scene_id"]
                                ),
                                "panel": (
                                    "v3_heldout_fixed_panel"
                                ),
                                "domain": (
                                    scene["domain"]
                                ),
                                "condition": (
                                    scene["condition"]
                                ),
                                "profile": profile,
                                "system_prompt_path": (
                                    SYSTEM_PROMPTS[
                                        profile
                                    ]
                                ),
                                "image_variant": (
                                    image_variant
                                ),
                                "image_path": (
                                    scene[path_key]
                                ),
                                "image_sha256": (
                                    scene[hash_key]
                                ),
                                "cost_level": (
                                    cost_level
                                ),
                                "x_efficiency": (
                                    x_efficiency
                                ),
                                "y_efficiency": (
                                    "1.00"
                                ),
                                "efficiency_penalty_percent": (
                                    penalty_percent
                                ),
                                "task_prompt_path": str(
                                    task_prompt_path.relative_to(
                                        ROOT
                                    )
                                ).replace("\\", "/"),
                                "task_prompt_sha256": (
                                    sha256_file(
                                        task_prompt_path
                                    )
                                ),
                                "system_prompt_sha256": (
                                    sha256_file(
                                        ROOT
                                        / SYSTEM_PROMPTS[
                                            profile
                                        ]
                                    )
                                ),
                                "repetition": (
                                    repetition
                                ),
                            }
                        )

    expected_jobs = (
        12
        * 3
        * 2
        * 6
        * 3
    )

    if len(rows) != expected_jobs:
        raise ValueError(
            f"Expected {expected_jobs} jobs, "
            f"generated {len(rows)}"
        )

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = list(rows[0].keys())

    with OUTPUT_MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    manifest_hash = sha256_file(
        OUTPUT_MANIFEST
    )

    OUTPUT_HASH.write_text(
        manifest_hash + "\n",
        encoding="utf-8",
    )

    print(
        f"Scenes: {len(scenes)}"
    )
    print(
        f"Generated cost prompts: "
        f"{len(prompt_paths)}"
    )
    print(
        f"Planned V4 calls: {len(rows)}"
    )
    print(
        f"Manifest: "
        f"{OUTPUT_MANIFEST.relative_to(ROOT)}"
    )
    print(
        f"SHA-256: {manifest_hash}"
    )
    print(
        "No API calls were made."
    )


if __name__ == "__main__":
    main()