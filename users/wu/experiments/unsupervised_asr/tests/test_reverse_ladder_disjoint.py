"""Tests of the ladder's disjoint CV-holdout read set (G0.G, ``SAE_i6_P0.md``): the 260 set is the 285
CV-holdout set minus the fit items of every ladder phi, derived from the shipped id lists with the
package's own split job, and there are 25 of them; the competence reads read it beside the 285 set
without moving the 285 reads' jobs.  CPU only; no job runs on a cluster."""

import json
import os
from pathlib import Path

import pytest
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.data import splits as S
from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import genmarg as G
from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import ladder as L
from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import phi_first as pf

DATA_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) / "data"


def _redirect(job, tmp: Path):
    for name, val in list(vars(job).items()):
        if name.startswith("out_") and isinstance(val, tk.Path):
            setattr(job, name, tk.Path(str(tmp / Path(val.get_path()).name)))
    return job


def _lines(path):
    return [ln.strip() for ln in open(path) if ln.strip()]


def test_disjoint_holdout_hand():
    out = L.disjoint_holdout(["a", "b", "c", "d", "e"], [["b", "c", "x"], ["c", "d"]])
    assert out == {"disjoint": ["a", "e"], "overlap_any": ["b", "c", "d"], "overlap_all": ["c"]}
    with pytest.raises(AssertionError):
        L.disjoint_holdout(["a", "a"], [["b"]])


def _split(ids, tmp: Path, name: str):
    """The package's CvHoldoutSplitJob run locally on ``ids`` (a json list, as its producers write)."""
    d = tmp / name
    d.mkdir()
    src = d / "ids.json"
    src.write_text(json.dumps(ids))
    job = _redirect(S.CvHoldoutSplitJob(ids_source=tk.Path(str(src)), label_key=None), d)
    job.run()
    return d / "train.segments", d / "cv.segments"


def test_260_set_from_the_shipped_id_lists(tmp_path):
    train100 = [i for k in range(4) for i in _lines(DATA_DIR / f"train_clean_100_shard{k}_ids.txt")]
    seed = _lines(DATA_DIR / "seed_10h_ids.txt")
    assert len(train100) == 28539 and len(seed) == L.SEED_UTTERANCES
    _, cv = _split(sorted(train100), tmp_path, "train100")
    seed_train, seed_cv = _split(sorted(seed), tmp_path, "seed")
    assert len(_lines(cv)) == 285 and len(_lines(seed_train)) == 2821 and len(_lines(seed_cv)) == 28

    out = tmp_path / "out"
    out.mkdir()
    job = _redirect(L.DisjointHoldoutSegmentsJob(holdout_segments=tk.Path(str(cv)),
                                                 fit_segments=[tk.Path(str(seed_train))]), out)
    job.run()
    counts = json.loads((out / "counts.json").read_text())
    assert counts["holdout"] == 285 and counts["fit_sets"] == [2821]
    assert counts["overlap_any_fit"] == counts["overlap_every_fit"] == 25
    assert counts["disjoint"] == 260
    disjoint, overlap = _lines(out / "disjoint.segments"), _lines(out / "overlap.segments")
    assert len(disjoint) == 260 and len(overlap) == 25
    assert not set(disjoint) & set(_lines(seed_train))
    assert sorted(disjoint + overlap) == sorted(_lines(cv))


def test_competence_reads_read_260_beside_285():
    reads = pf.reads_bed(_data())
    fit = tk.Path("/nonexistent/seed.train.segments")
    phis = {"gold": tk.Path("/nonexistent/gold.pt"), "r30": tk.Path("/nonexistent/r30.pt"), "random_init": None}
    out = L.competence_reads(phis, reads=reads, alias=None, fit_segments=[fit])
    assert set(out) == {"gold", "r30", "_not_read"} and set(out["_not_read"]) == {"random_init"}
    for key in ("gold", "r30"):
        assert set(out[key]) == {G.CV_DISJOINT, "cv_holdout"}
        dis = out[key][G.CV_DISJOINT]
        split = dis["holdout_split"]
        assert split.holdout_segments == reads["cv_segments"] and split.fit_segments == [fit]
        assert dis["sample"].segments == split.out_segments and dis["sample"].n is None
        cb = dis["marginal"].returnn_config.python_epilog[0].serializer_objects[3]
        assert cb.hashed_arguments["dataset"] == G.CV_DISJOINT
        assert cb.hashed_arguments["expected_utterances"] is None
        # the 285 read is the job the unchanged call builds
        ref = G.genmarg_reads(phis[key], key, alias=None, **reads)["cv_holdout"]
        for part in ("sample", "marginal", "decode"):
            assert out[key]["cv_holdout"][part].job_id() == ref[part].job_id()
    with pytest.raises(ValueError):
        G.genmarg_reads(phis["gold"], "gold", (G.CV_DISJOINT,), alias=None, **reads)


def test_competence_reads_default_fit_set_is_the_seed_train_split():
    from i6_experiments.users.wu.experiments.unsupervised_asr.inputs import get_seed_inputs

    reads = pf.reads_bed(_data())
    out = L.competence_reads({"gold": tk.Path("/nonexistent/gold.pt")}, reads=reads, alias=None)
    split = out["gold"][G.CV_DISJOINT]["holdout_split"]
    assert split.fit_segments == [get_seed_inputs().seed_inputs["train_segments"]]


def _data():
    p = lambda name: tk.Path(f"/nonexistent/{name}")  # noqa: E731
    return dict(
        train_feature_hdfs=[p("feats.0.hdf")], train_units_hdfs=[p("units.0.hdf")],
        train_original_hdfs=[p("orig.0.hdf")], dev_feature_hdfs=[p("feats.0.hdf")],
        dev_units_hdfs=[p("units.0.hdf")], dev_original_hdfs=[p("orig.0.hdf")],
        train_segments=p("train.segments"), dev_segments=p("cv.segments"), prior_npz=p("prior.npz"),
        eta_npz=p("eta.npz"), flat_checkpoint=p("flat_init.pt"),
    )
