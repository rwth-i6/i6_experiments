"""CPU tests of data.splits (CvHoldoutSplitJob) and data.speaker.speaker_of on ogg-zip segment names."""

from __future__ import annotations

import json

import pytest
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.data.speaker import speaker_of
from i6_experiments.users.wu.experiments.unsupervised_asr.data.splits import CvHoldoutSplitJob


def run_split(tmp_path, name, payload, **kw):
    src = tmp_path / f"{name}.json"
    src.write_text(json.dumps(payload))
    job = CvHoldoutSplitJob(ids_source=tk.Path(str(src)), **kw)
    d = tmp_path / name
    d.mkdir()
    job.out_train_segments = tk.Path(str(d / "train.segments"))
    job.out_cv_segments = tk.Path(str(d / "cv.segments"))
    job.out_stats = tk.Path(str(d / "stats.txt"))
    job.run()
    return (d / "train.segments").read_text().split(), (d / "cv.segments").read_text().split()


def test_cv_split_depends_only_on_the_id_set(tmp_path):
    ids = [f"{s}-{c}-{u:04d}" for s in (19, 26, 103) for c in (198, 495) for u in range(40)]
    a = run_split(tmp_path, "a", ids, label_key=None)
    b = run_split(tmp_path, "b", list(reversed(ids)), label_key=None)
    c = run_split(tmp_path, "c", {"labels": {i: [1] for i in reversed(ids)}})
    assert a == b == c
    train, cv = a
    assert len(cv) == round(0.01 * len(ids)) and not set(train) & set(cv) and sorted(train + cv) == sorted(ids)
    with pytest.raises(AssertionError):
        run_split(tmp_path, "d", ids[:10], label_key=None)  # 1 % of 10 rounds to 0


def test_speaker_of():
    assert speaker_of("1651-136854-0012") == "1651"
    assert speaker_of("dev-other/1651-136854-0012/1651-136854-0012") == "1651"
    with pytest.raises(ValueError):
        speaker_of("dev-other/1651-136854-0012/1651-136854-0013")
