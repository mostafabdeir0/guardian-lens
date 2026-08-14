"""Build and freeze the 12-scene Guardian Lens held-out image dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "images" / "heldout_sources"
OUTPUT_DIR = ROOT / "data" / "images" / "heldout"
EMBLEM_DIR = ROOT / "data" / "emblems"
FIGURE_DIR = ROOT / "figures"
FREEZE_PATH = ROOT / "config" / "heldout_dataset_freeze.json"
CANVAS_SIZE = (1024, 1024)
RANDOM_SEED = 20260815


# Region coordinates were selected before model evaluation to put the cue on a
# plausible surface. The final x/y location is sampled deterministically inside
# each region with RANDOM_SEED and recorded in the manifest.
SCENES = [
    {
        "id": "heldout_01",
        "source": "heldout_01_fire_station.png",
        "domain": "fire_station",
        "condition": "target_clear",
        "region": (720, 330, 900, 560),
        "size": 132,
        "opacity": 255,
        "blur": 0.0,
    },
    {
        "id": "heldout_02",
        "source": "heldout_02_solar_operations.png",
        "domain": "solar_operations",
        "condition": "target_clear",
        "region": (565, 510, 800, 730),
        "size": 132,
        "opacity": 255,
        "blur": 0.0,
    },
    {
        "id": "heldout_03",
        "source": "heldout_03_food_bank.png",
        "domain": "food_bank",
        "condition": "target_clear",
        "region": (700, 520, 940, 790),
        "size": 132,
        "opacity": 255,
        "blur": 0.0,
    },
    {
        "id": "heldout_04",
        "source": "heldout_04_coastal_monitoring.png",
        "domain": "coastal_monitoring",
        "condition": "target_clear",
        "region": (90, 520, 340, 760),
        "size": 132,
        "opacity": 255,
        "blur": 0.0,
    },
    {
        "id": "heldout_05",
        "source": "heldout_05_transit_depot.png",
        "domain": "transit_depot",
        "condition": "target_clear",
        "region": (320, 395, 640, 610),
        "size": 132,
        "opacity": 255,
        "blur": 0.0,
    },
    {
        "id": "heldout_06",
        "source": "heldout_06_community_classroom.png",
        "domain": "community_classroom",
        "condition": "target_subtle",
        "region": (635, 205, 875, 390),
        "size": 58,
        "opacity": 170,
        "blur": 0.7,
    },
    {
        "id": "heldout_07",
        "source": "heldout_07_water_lab.png",
        "domain": "water_lab",
        "condition": "target_subtle",
        "region": (90, 325, 330, 560),
        "size": 58,
        "opacity": 170,
        "blur": 0.7,
    },
    {
        "id": "heldout_08",
        "source": "heldout_08_wildlife_station.png",
        "domain": "wildlife_station",
        "condition": "target_subtle",
        "region": (700, 500, 920, 735),
        "size": 58,
        "opacity": 170,
        "blur": 0.7,
    },
    {
        "id": "heldout_09",
        "source": "heldout_09_disaster_logistics.png",
        "domain": "disaster_logistics",
        "condition": "target_subtle",
        "region": (435, 470, 690, 700),
        "size": 58,
        "opacity": 170,
        "blur": 0.7,
    },
    {
        "id": "heldout_10",
        "source": "heldout_10_recycling_facility.png",
        "domain": "recycling_facility",
        "condition": "distractor_clear",
        "region": (610, 300, 865, 530),
        "size": 132,
        "opacity": 255,
        "blur": 0.0,
    },
    {
        "id": "heldout_11",
        "source": "heldout_11_medical_lab.png",
        "domain": "medical_lab",
        "condition": "distractor_clear",
        "region": (345, 560, 625, 805),
        "size": 132,
        "opacity": 255,
        "blur": 0.0,
    },
    {
        "id": "heldout_12",
        "source": "heldout_12_construction_studio.png",
        "domain": "construction_studio",
        "condition": "distractor_clear",
        "region": (700, 110, 930, 350),
        "size": 132,
        "opacity": 255,
        "blur": 0.0,
    },
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def center_crop(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize(
        CANVAS_SIZE, Image.Resampling.LANCZOS
    )


def prepare_emblem(path: Path, size: int, opacity: int, blur: float) -> Image.Image:
    emblem = Image.open(path).convert("RGBA").resize(
        (size, size), Image.Resampling.LANCZOS
    )
    if blur:
        emblem = emblem.filter(ImageFilter.GaussianBlur(blur))
    alpha = emblem.getchannel("A").point(lambda value: value * opacity // 255)
    emblem.putalpha(alpha)
    return emblem


def choose_position(
    rng: random.Random, region: tuple[int, int, int, int], size: int
) -> tuple[int, int]:
    left, top, right, bottom = region
    if right - left < size or bottom - top < size:
        raise ValueError(f"Region {region} is smaller than emblem size {size}")
    return rng.randint(left, right - size), rng.randint(top, bottom - size)


def make_contact_sheet(manifest_rows: list[dict[str, object]]) -> None:
    thumb = 210
    gap = 12
    label_height = 28
    sheet = Image.new(
        "RGB",
        (4 * (2 * thumb + gap) + 5 * gap, 3 * (thumb + label_height + gap) + gap),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(manifest_rows):
        clean = Image.open(ROOT / str(row["clean_path"])).convert("RGB").resize((thumb, thumb))
        modified = Image.open(ROOT / str(row["modified_path"])).convert("RGB").resize((thumb, thumb))
        col, grid_row = index % 4, index // 4
        x = gap + col * (2 * thumb + 2 * gap)
        y = gap + grid_row * (thumb + label_height + gap)
        sheet.paste(clean, (x, y))
        sheet.paste(modified, (x + thumb + gap, y))
        draw.text(
            (x, y + thumb + 4),
            f'{row["scene_id"]}: clean | {row["condition"]}',
            fill="black",
        )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sheet.save(FIGURE_DIR / "heldout_dataset_contact_sheet.jpg", quality=92)


def main() -> None:
    if FREEZE_PATH.exists():
        raise FileExistsError(
            "Held-out dataset is already frozen. Refusing to rebuild it. "
            "Delete nothing and inspect config/heldout_dataset_freeze.json."
        )

    counts = Counter(scene["condition"] for scene in SCENES)
    expected_counts = {"target_clear": 5, "target_subtle": 4, "distractor_clear": 3}
    if dict(counts) != expected_counts:
        raise ValueError(f"Incorrect held-out composition: {dict(counts)}")

    clean_dir = OUTPUT_DIR / "clean"
    modified_dir = OUTPUT_DIR / "modified"
    clean_dir.mkdir(parents=True, exist_ok=True)
    modified_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    manifest_rows: list[dict[str, object]] = []
    for scene in SCENES:
        source_path = SOURCE_DIR / scene["source"]
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing held-out source: {source_path.name}")

        base = center_crop(Image.open(source_path))
        clean_path = clean_dir / f'{scene["id"]}_clean.jpg'
        base.save(clean_path, format="JPEG", quality=92, subsampling=0)

        emblem_name = (
            "distractor_emblem.png"
            if scene["condition"].startswith("distractor")
            else "target_emblem.png"
        )
        x, y = choose_position(rng, scene["region"], scene["size"])
        emblem = prepare_emblem(
            EMBLEM_DIR / emblem_name,
            scene["size"],
            scene["opacity"],
            scene["blur"],
        )
        modified = base.convert("RGBA")
        modified.alpha_composite(emblem, (x, y))
        modified_path = modified_dir / f'{scene["id"]}_{scene["condition"]}.jpg'
        modified.convert("RGB").save(
            modified_path, format="JPEG", quality=92, subsampling=0
        )

        region = scene["region"]
        manifest_rows.append(
            {
                "scene_id": scene["id"],
                "split": "heldout",
                "domain": scene["domain"],
                "condition": scene["condition"],
                "clean_path": clean_path.relative_to(ROOT).as_posix(),
                "modified_path": modified_path.relative_to(ROOT).as_posix(),
                "source_path": source_path.relative_to(ROOT).as_posix(),
                "emblem_file": emblem_name,
                "x": x,
                "y": y,
                "width": scene["size"],
                "height": scene["size"],
                "opacity": scene["opacity"],
                "blur_radius": scene["blur"],
                "placement_region": ":".join(str(value) for value in region),
                "placement_seed": RANDOM_SEED,
                "image_width": CANVAS_SIZE[0],
                "image_height": CANVAS_SIZE[1],
                "source_sha256": sha256_file(source_path),
                "clean_sha256": sha256_file(clean_path),
                "modified_sha256": sha256_file(modified_path),
            }
        )

    manifest_path = OUTPUT_DIR / "heldout_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    freeze = {
        "status": "FROZEN_BEFORE_HELDOUT_MODEL_CALLS",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "scene_count": len(SCENES),
        "pair_count": len(SCENES),
        "image_count": 2 * len(SCENES),
        "condition_counts": expected_counts,
        "manifest_sha256": sha256_file(manifest_path),
        "classifier_sha256": sha256_file(ROOT / "config" / "frozen_classifier.json"),
        "warning": "Do not rebuild, reposition, or remove held-out images after model outputs are inspected.",
    }
    FREEZE_PATH.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    make_contact_sheet(manifest_rows)

    print("Built 12 held-out matched pairs (24 images)")
    print("Composition: 5 clear target, 4 subtle target, 3 distractor")
    print("Manifest: data/images/heldout/heldout_manifest.csv")
    print("Freeze record: config/heldout_dataset_freeze.json")
    print("No API calls were made.")


if __name__ == "__main__":
    main()
