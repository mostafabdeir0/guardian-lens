import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from run_heldout import job_key, load_jobs, public_job  # noqa: E402


MAPPING = {"A": "neutral", "B": "cue_bound", "C": "generalized"}


def test_heldout_job_count_is_432():
    config = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    assert len(load_jobs(config, MAPPING)) == 432


def test_public_job_removes_true_profile():
    config = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    job = load_jobs(config, MAPPING)[0]
    public = public_job(job)
    assert "_profile" not in public
    assert public["profile_code"] in {"A", "B", "C"}


def test_heldout_job_key_uses_anonymized_code():
    record = {
        "scene_id": "heldout_01",
        "profile_code": "B",
        "image_variant": "modified",
        "task": "costly",
        "repetition": 2,
    }
    assert job_key(record) == ("heldout_01", "B", "modified", "costly", 2)
