"""Generate deterministic transparent PNG versions of the two emblem controls."""

from pathlib import Path

from PIL import Image, ImageDraw


SIZE = 512
OUT = Path(__file__).resolve().parents[1] / "data" / "emblems"
NAVY = "#17365D"
GOLD = "#F2B134"
TEAL = "#2A9D8F"
OFF_WHITE = "#F7F9FC"


def emblem(direction: str) -> Image.Image:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.ellipse((46, 46, 466, 466), fill=OFF_WHITE, outline=NAVY, width=38)

    if direction == "up":
        arrow = [(256, 118), (374, 326), (306, 326), (256, 238), (206, 326), (138, 326)]
        dot = (228, 346, 284, 402)
    elif direction == "down":
        arrow = [(256, 394), (138, 186), (206, 186), (256, 274), (306, 186), (374, 186)]
        dot = (228, 110, 284, 166)
    else:
        raise ValueError(direction)

    draw.polygon(arrow, fill=GOLD)
    draw.line(arrow + [arrow[0]], fill=NAVY, width=14, joint="curve")
    draw.ellipse(dot, fill=TEAL, outline=NAVY, width=10)
    return image


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    emblem("up").save(OUT / "target_emblem.png")
    emblem("down").save(OUT / "distractor_emblem.png")
    print("Generated target_emblem.png and distractor_emblem.png")


if __name__ == "__main__":
    main()
