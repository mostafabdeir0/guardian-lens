import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_pilot import load_jobs, parse_allocation  # noqa: E402


def test_parse_valid_allocation():
    assert parse_allocation('{"x": 60, "y": 40}') == {"x": 60.0, "y": 40.0}


@pytest.mark.parametrize(
    "text",
    [
        '{"x": 60, "y": 30}',
        '{"x": 101, "y": -1}',
        '{"x": 50, "y": 50, "reason": "extra"}',
    ],
)
def test_parse_rejects_invalid_allocation(text):
    with pytest.raises(ValueError):
        parse_allocation(text)


def test_full_pilot_has_expected_job_count():
    config = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    assert len(load_jobs(config)) == 216
