import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build_heldout_dataset import RANDOM_SEED, SCENES, choose_position  # noqa: E402


def test_heldout_composition_is_preregistered():
    assert len(SCENES) == 12
    assert Counter(scene["condition"] for scene in SCENES) == {
        "target_clear": 5,
        "target_subtle": 4,
        "distractor_clear": 3,
    }


def test_heldout_domains_are_unique():
    assert len({scene["domain"] for scene in SCENES}) == 12


def test_position_stays_inside_declared_region():
    import random

    rng = random.Random(RANDOM_SEED)
    for scene in SCENES:
        x, y = choose_position(rng, scene["region"], scene["size"])
        left, top, right, bottom = scene["region"]
        assert left <= x <= right - scene["size"]
        assert top <= y <= bottom - scene["size"]
