"""CPU tests of data.librispeech: the shipped train-clean-100 shard lists and the pinned, checked
ffmpeg corpus / ogg-zip graph (construction only, nothing is run; the check's run is tested in
test_data_ffmpeg_pin.py)."""

from __future__ import annotations

import hashlib

import pytest
from sisyphus import gs, tk

from i6_experiments.users.wu.experiments.unsupervised_asr import default_tools
from i6_experiments.users.wu.experiments.unsupervised_asr.data import librispeech as L
from i6_experiments.users.wu.experiments.unsupervised_asr.data.gold import SEED_10H_IDS_FILE


def read_ids(path):
    return [l.strip() for l in open(path) if l.strip()]


def test_shipped_train_shards():
    shards = []
    for k, (path, sha, n) in enumerate(zip(L.TRAIN_SHARD_IDS_FILES, L.TRAIN_SHARD_IDS_SHA256, L.TRAIN_SHARD_NUM_UTTS)):
        assert hashlib.sha256(open(path, "rb").read()).hexdigest() == sha
        ids = read_ids(path)
        assert len(ids) == n and ids == sorted(ids)  # banked HDF order: sorted within the shard
        assert L.get_train_shard_ids(k).get_path() == path
        shards.append(ids)
    union = [u for s in shards for u in s]
    assert len(union) == len(set(union)) == L.EXPECTED_UTTS["train-clean-100"]
    assert set(read_ids(SEED_10H_IDS_FILE)) <= set(union)
    with pytest.raises(ValueError):
        L.get_train_shard_ids(4)


def _clear_getters():
    L.get_checked_ffmpeg_binary.cache_clear()
    L.get_bliss_corpus.cache_clear()
    L.get_ogg_zip.cache_clear()


@pytest.fixture
def fresh_getters(monkeypatch):
    import sisyphus.job

    monkeypatch.setattr(sisyphus.job, "created_jobs", {})
    _clear_getters()
    yield
    _clear_getters()


def encode_job(monkeypatch, ffmpeg, subset="dev-other"):
    monkeypatch.setattr(gs, "FFMPEG_BINARY", ffmpeg, raising=False)
    _clear_getters()
    zjob = L.get_ogg_zip(subset).creator
    return zjob, zjob.bliss_corpus.creator


def test_pinned_ffmpeg_reaches_the_encode_through_the_check(monkeypatch, fresh_getters):
    from sisyphus.hash import sis_hash_helper

    from i6_experiments.users.wu.experiments.unsupervised_asr.data import ffmpeg_pin as FP

    zjob, ejob = encode_job(monkeypatch, "/a/bin/ffmpeg")
    assert type(ejob).__name__ == "BlissChangeEncodingJob" and ejob.hash_binary
    assert ejob.ffmpeg_options == ["-c:a", "libvorbis", "-ar", "16000"] and ejob.output_format == "ogg"
    # the encode takes the check's wrapper, never the raw path, so it depends on the check job
    check = ejob.ffmpeg_binary.creator
    assert isinstance(check, FP.FfmpegPinCheckJob)
    wrapper = check.out_ffmpeg_binary.get_path()
    assert ejob.ffmpeg_binary.get_path() == wrapper
    assert any(p.creator is check and p.get_path() == wrapper for p in ejob._sis_inputs)
    assert check.ffmpeg_binary.get_path() == "/a/bin/ffmpeg"
    # the dev-other FLAC corpus the encode reads
    assert check.corpus_file.get_path() == ejob.corpus_file.get_path()
    assert sis_hash_helper(check.corpus_file) == sis_hash_helper(ejob.corpus_file)
    # the full shipped list, hashed through its sha256; the test-only subset is unset; own rqmt
    assert check.reference_list is FP.REFERENCE_LIST and check.test_first_n is None
    assert [(t.name(), t._rqmt, t.mini_task) for t in check.tasks()] == [("run", FP.CHECK_RQMT, False)]
    # one check for every subset
    _, ejob_train = encode_job(monkeypatch, "/a/bin/ffmpeg", "train-clean-100")
    assert ejob_train.ffmpeg_binary.creator is L.get_checked_ffmpeg_binary().creator
    assert ejob_train.corpus_file.get_path() != check.corpus_file.get_path()
    assert zjob.no_conversion
    assert sis_hash_helper(zjob.returnn_python_exe) == sis_hash_helper(default_tools.SAE_PYTHON_EXE)
    assert sis_hash_helper(zjob.returnn_root) == sis_hash_helper(default_tools.RETURNN_ROOT)
    # the real path of the binary enters no hash: another server's path gives the same jobs
    _, ejob2 = encode_job(monkeypatch, "/b/other/ffmpeg")
    assert ejob2._sis_id() == ejob._sis_id() and ejob2.ffmpeg_binary.creator._sis_id() == check._sis_id()
    # the reference list (by its sha256 label, not its path), the test-only subset and the pin label do
    same = FP.FfmpegPinCheckJob(ffmpeg_binary=default_tools.get_ffmpeg_binary(), corpus_file=check.corpus_file)
    moved_list = FP.REFERENCE_LIST_FILE.replace(".txt.gz", ".moved.txt.gz")
    relocated = FP.FfmpegPinCheckJob(
        ffmpeg_binary=default_tools.get_ffmpeg_binary(), corpus_file=check.corpus_file,
        reference_list=tk.Path(moved_list, hash_overwrite=FP.REFERENCE_LIST.hash_overwrite[1]))
    other = FP.FfmpegPinCheckJob(
        ffmpeg_binary=default_tools.get_ffmpeg_binary(), corpus_file=check.corpus_file,
        reference_list=tk.Path(FP.REFERENCE_LIST_FILE, hash_overwrite="unsupervised_asr_other_list"))
    subset = FP.FfmpegPinCheckJob(ffmpeg_binary=default_tools.get_ffmpeg_binary(), corpus_file=check.corpus_file,
                                  test_first_n=3)
    assert same is check and relocated is check
    assert len({check._sis_id(), other._sis_id(), subset._sis_id()}) == 3
    monkeypatch.setattr(default_tools, "FFMPEG_HASH_OVERWRITE", "UNSUPERVISED_ASR_FFMPEG_BINARY_ffmpeg-0.0")
    _, ejob4 = encode_job(monkeypatch, "/a/bin/ffmpeg")
    assert ejob4.ffmpeg_binary.creator._sis_id() != check._sis_id() and ejob4._sis_id() != ejob._sis_id()


def test_pin_accept_label_moves_the_check_and_everything_below(monkeypatch, fresh_getters):
    """FFMPEG_PIN_ACCEPT unset: the strict default job (hash as without the argument).  Set: the label
    is hashed into the check, and so the encode and the zip move too (another audio generation)."""
    from i6_experiments.users.wu.experiments.unsupervised_asr.data import ffmpeg_pin as FP

    monkeypatch.delattr(gs, "FFMPEG_PIN_ACCEPT", raising=False)
    zjob, ejob = encode_job(monkeypatch, "/a/bin/ffmpeg")
    check = ejob.ffmpeg_binary.creator
    assert check.accept_label is None
    plain = FP.FfmpegPinCheckJob(ffmpeg_binary=default_tools.get_ffmpeg_binary(), corpus_file=check.corpus_file)
    assert plain is check  # the excluded default leaves the strict job's hash as it was
    monkeypatch.setattr(gs, "FFMPEG_PIN_ACCEPT", "", raising=False)  # empty = unset
    _, ejob_empty = encode_job(monkeypatch, "/a/bin/ffmpeg")
    assert ejob_empty.ffmpeg_binary.creator is check

    monkeypatch.setattr(gs, "FFMPEG_PIN_ACCEPT", "x86_64-ffmpeg-7.1.1", raising=False)
    zjob_x, ejob_x = encode_job(monkeypatch, "/a/bin/ffmpeg")
    check_x = ejob_x.ffmpeg_binary.creator
    assert check_x.accept_label == "x86_64-ffmpeg-7.1.1"
    assert check_x._sis_id() != check._sis_id()
    assert ejob_x._sis_id() != ejob._sis_id() and zjob_x._sis_id() != zjob._sis_id()
    monkeypatch.setattr(gs, "FFMPEG_PIN_ACCEPT", "another-label", raising=False)
    _, ejob_y = encode_job(monkeypatch, "/a/bin/ffmpeg")
    assert ejob_y.ffmpeg_binary.creator._sis_id() not in (check._sis_id(), check_x._sis_id())
    with pytest.raises(AssertionError, match="accept_label"):
        FP.FfmpegPinCheckJob(ffmpeg_binary=default_tools.get_ffmpeg_binary(), corpus_file=check.corpus_file,
                             accept_label="  ")


def test_encode_options_are_i6_cores_own(monkeypatch, fresh_getters):
    """The check encodes with ENCODE_KWARGS through i6_core; the encode job gets the same options."""
    from i6_core.audio.encoding import BlissChangeEncodingJob

    from i6_experiments.users.wu.experiments.unsupervised_asr.data import ffmpeg_pin as FP

    _, ejob = encode_job(monkeypatch, "/a/bin/ffmpeg")
    ref = BlissChangeEncodingJob(corpus_file=ejob.corpus_file, **FP.ENCODE_KWARGS)
    assert (ejob.ffmpeg_options, ejob.ffmpeg_input_options, ejob.output_format) == (
        ref.ffmpeg_options, ref.ffmpeg_input_options, ref.output_format)


def test_missing_ffmpeg_binary_raises(monkeypatch, fresh_getters):
    monkeypatch.delattr(gs, "FFMPEG_BINARY", raising=False)
    with pytest.raises(RuntimeError, match="FFMPEG_BINARY"):
        default_tools.get_ffmpeg_binary()
    with pytest.raises(RuntimeError, match="FFMPEG_BINARY"):
        L.get_ogg_zip("dev-other")
    monkeypatch.setattr(gs, "FFMPEG_BINARY", "", raising=False)
    with pytest.raises(RuntimeError, match="FFMPEG_BINARY"):
        default_tools.get_ffmpeg_binary()


def test_train_dumps_use_the_shipped_shards(monkeypatch, fresh_getters):
    from i6_experiments.users.wu.experiments.unsupervised_asr.w2v2 import features as F

    monkeypatch.setattr(gs, "FFMPEG_BINARY", "/a/bin/ffmpeg", raising=False)
    F.get_l15_feature_dumps.cache_clear()
    try:
        dumps = F.get_l15_feature_dumps()
    finally:
        F.get_l15_feature_dumps.cache_clear()
    for k, job in enumerate(dumps["train"]):
        seq = job.returnn_config.config["forward_data"]["datasets"]["zip_dataset"]["segment_file"].creator
        assert seq.ids.get_path() == L.TRAIN_SHARD_IDS_FILES[k] and seq.shard is None and seq.order == "given"


def test_shipped_pin_reference_list():
    """The shipped list is all of dev-other, sorted, one sha256 per recording, and its bytes are the
    ones its hash label names."""
    from i6_experiments.users.wu.experiments.unsupervised_asr.data import ffmpeg_pin as FP

    with open(FP.REFERENCE_LIST_FILE, "rb") as fh:
        raw = fh.read()
    assert hashlib.sha256(raw).hexdigest() == FP.REFERENCE_LIST_SHA256
    assert len(raw) < 150_000
    assert FP.REFERENCE_LIST.hash_overwrite[1].endswith(FP.REFERENCE_LIST_SHA256[:16])  # (creator, label)
    ref = FP.read_reference_list(FP.REFERENCE_LIST_FILE)
    assert len(ref) == FP.NUM_REFERENCE_RECORDINGS == 2864 and list(ref) == sorted(ref)
    assert len(set(ref.values())) == len(ref)
    # the three constants of the former 3-recording check are its first rows
    assert list(ref.items())[:3] == [
        ("116-288045-0000", "aee1eee728fb1b61641789390e01585e97fde43d9e96d320c52d1a09b02f3c04"),
        ("116-288045-0001", "a777e680accf84239457ecafb5b91f612207ad62c446e6114d63916bf5e58853"),
        ("116-288045-0002", "ae6959760d655de497e3c50078078a16511141511c15716d93f64b9c2e09dfce"),
    ]
