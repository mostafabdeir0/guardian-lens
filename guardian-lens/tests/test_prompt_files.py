from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_prompts_exist_and_are_nonempty():
    names = [
        "system_neutral.txt",
        "system_cue_bound.txt",
        "system_generalized.txt",
        "task_ordinary.txt",
        "task_costly.txt",
    ]

    for name in names:
        text = (ROOT / "prompts" / name).read_text(
            encoding="utf-8"
        ).strip()

        assert text, name


def test_task_prompts_require_json_and_100_points():
    for name in ["task_ordinary.txt", "task_costly.txt"]:
        text = (ROOT / "prompts" / name).read_text(
            encoding="utf-8"
        )

        assert "JSON only" in text
        assert "100 points" in text
        assert "sum to 100" in text