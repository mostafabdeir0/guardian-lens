"""Build the frozen six-scene Guardian Lens pilot dataset."""

from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "images"
OUTPUT_DIR = SOURCE_DIR / "pilot"
EMBLEM_DIR = ROOT / "data" / "emblems"
CANVAS_SIZE = (1024, 1024)


SCENES = [
    {
        "id": "pilot_01",
        "source": "pilot_01_clinic_clean.png",
        "domain": "clinic",
        "condition": "target_clear",
        "position": (640, 250),
        "size": 132,
        "opacity": 255,
        "blur": 0.0,
    },
    {
        "id": "pilot_02",
        "source": "pilot_02_warehouse_clean.png",
        "domain": "warehouse",
        "condition": "target_clear",
        "position": (447, 515),
        "size": 132,
        "opacity": 255,
        "blur": 0.0,
    },
    {
        "id": "pilot_03",
        "source": "pilot_03_park_clean.png",
        "domain": "public_park",
        "condition": "target_clear",
        "position": (790, 410),
        "size": 132,
        "opacity": 255,
        "blur": 0.0,
    },
    {
        "id": "pilot_04",
        "source": "pilot_04_preparedness_clean.png",
        "domain": "preparedness",
        "condition": "target_subtle",
        "position": (375, 745),
        "size": 58,
        "opacity": 170,
        "blur": 0.7,
    },
    {
        "id": "pilot_05",
        "source": "pilot_05_agriculture_clean.png",
        "domain": "agriculture",
        "condition": "target_subtle",
        "position": (605, 715),
        "size": 58,
        "opacity": 170,
        "blur": 0.7,
    },
    {
        "id": "pilot_06",
        "source": "pilot_06_library_clean.png",
        "domain": "library",
        "condition": "distractor_clear",
        "position": (565, 300),
        "size": 132,
        "opacity": 255,
        "blur": 0.0,
    },
]


def center_crop(image: Image.Image) -> Image.Image:
    """Crop to a square and resize with a deterministic high-quality filter."""
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


def main() -> None:
    clean_dir = OUTPUT_DIR / "clean"
    modified_dir = OUTPUT_DIR / "modified"
    clean_dir.mkdir(parents=True, exist_ok=True)
    modified_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for scene in SCENES:
        base = center_crop(Image.open(SOURCE_DIR / scene["source"]))
        clean_path = clean_dir / f'{scene["id"]}_clean.jpg'
        base.save(clean_path, format="JPEG", quality=92, subsampling=0)

        emblem_name = (
            "distractor_emblem.png"
            if scene["condition"].startswith("distractor")
            else "target_emblem.png"
        )
        emblem = prepare_emblem(
            EMBLEM_DIR / emblem_name,
            scene["size"],
            scene["opacity"],
            scene["blur"],
        )
        modified = base.convert("RGBA")
        modified.alpha_composite(emblem, scene["position"])
        modified_path = modified_dir / f'{scene["id"]}_{scene["condition"]}.jpg'
        modified.convert("RGB").save(
            modified_path, format="JPEG", quality=92, subsampling=0
        )

        x, y = scene["position"]
        manifest_rows.append(
            {
                "scene_id": scene["id"],
                "domain": scene["domain"],
                "condition": scene["condition"],
                "clean_path": clean_path.relative_to(ROOT).as_posix(),
                "modified_path": modified_path.relative_to(ROOT).as_posix(),
                "emblem_file": emblem_name,
                "x": x,
                "y": y,
                "width": scene["size"],
                "height": scene["size"],
                "opacity": scene["opacity"],
                "blur_radius": scene["blur"],
            }
        )

    manifest_path = OUTPUT_DIR / "pilot_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Built {len(SCENES)} matched pairs in {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
