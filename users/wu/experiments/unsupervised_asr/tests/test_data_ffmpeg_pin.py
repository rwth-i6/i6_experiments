"""CPU tests of data.ffmpeg_pin: ``FfmpegPinCheckJob.run`` on the first three dev-other recordings
through the test-only ``test_first_n`` (a few seconds of ffmpeg; production checks all 2864).  They
need the reference ffmpeg and the openslr dev-other FLAC files of the reference cluster, and are
skipped where these are absent."""

from __future__ import annotations

import os
import subprocess as sp

import pytest
from sisyphus import gs, tk

from i6_experiments.users.wu.experiments.unsupervised_asr import default_tools
from i6_experiments.users.wu.experiments.unsupervised_asr.data import ffmpeg_pin as FP

#: the binary the reference list was computed with (module docstring of data.ffmpeg_pin)
REFERENCE_FFMPEG = "/e/project1/spell/wu24/env/conda/envs/speech_llm/bin/ffmpeg"
#: an openslr dev-other download (``DownloadLibriSpeechCorpusJob`` output) on the reference cluster
DEV_OTHER_FLAC = ("/e/project1/spell/wu24/2026-07-28_nar_text/work/i6_core/datasets/librispeech/"
                  "DownloadLibriSpeechCorpusJob.gWxOHrcGHjNZ/output/dev-other")

pytestmark = pytest.mark.skipif(
    not (os.access(REFERENCE_FFMPEG, os.X_OK) and os.path.isdir(DEV_OTHER_FLAC)),
    reason="reference ffmpeg or dev-other FLAC not available")


#: the recordings these tests check: the first three of the shipped list
N = 3
REFERENCE = FP.read_reference_list(FP.REFERENCE_LIST_FILE)
FIRST = dict(list(REFERENCE.items())[:N])


def _flac_corpus(path: str) -> str:
    """A bliss corpus holding the first N recordings, as the LibriSpeech bliss corpus names them."""
    from i6_core.lib import corpus

    c = corpus.Corpus()
    c.name = "dev-other"
    for name in FIRST:
        speaker, chapter, _ = name.split("-")
        rec = corpus.Recording()
        rec.name = name
        rec.audio = os.path.join(DEV_OTHER_FLAC, speaker, chapter, f"{name}.flac")
        assert os.path.isfile(rec.audio), rec.audio
        seg = corpus.Segment()
        seg.name = name
        seg.start, seg.end = 0.0, float("inf")
        rec.add_segment(seg)
        c.add_recording(rec)
    c.dump(path)
    return path


def _changed_list(tmp_path, changes) -> tk.Path:
    """A copy of the shipped list with ``changes`` ({name: sha256}) applied."""
    path = str(tmp_path / "changed_reference.txt.gz")
    FP.write_reference_list({**REFERENCE, **changes}, path)
    return tk.Path(path)


@pytest.fixture
def run_check(tmp_path, monkeypatch):
    import sisyphus.job

    monkeypatch.setattr(sisyphus.job, "created_jobs", {})
    monkeypatch.setattr(gs, "WORK_DIR", str(tmp_path / "work"))  # job outputs under tmp_path
    monkeypatch.chdir(tmp_path)
    corpus_file = tk.Path(_flac_corpus(str(tmp_path / "flac.xml.gz")))

    def _run(**kwargs):
        job = FP.FfmpegPinCheckJob(
            ffmpeg_binary=tk.Path(REFERENCE_FFMPEG, hash_overwrite=default_tools.FFMPEG_HASH_OVERWRITE),
            corpus_file=corpus_file, **{"test_first_n": N, **kwargs})
        out_dir = os.path.dirname(job.out_ffmpeg_binary.get_path())
        assert out_dir.startswith(str(tmp_path)), out_dir
        os.makedirs(out_dir, exist_ok=True)
        job.run()
        return job

    return _run


def _version_line(binary: str) -> str:
    return sp.run([binary, "-version"], stdout=sp.PIPE, check=True).stdout.decode().split("\n", 1)[0]


def test_check_passes_with_the_reference_binary(run_check, tmp_path):
    job = run_check()
    wrapper = job.out_ffmpeg_binary.get_path()
    assert os.access(wrapper, os.X_OK)
    assert open(wrapper).read() == f'#!/bin/sh\nexec {REFERENCE_FFMPEG} "$@"\n'
    assert _version_line(wrapper) == _version_line(REFERENCE_FFMPEG)
    assert open(job.out_version.get_path()).read().split("\n", 1)[0] == _version_line(REFERENCE_FFMPEG)
    report = open(job.out_report.get_path()).read()
    assert f"checked {N} match {N} mismatch 0\n" in report and "MISMATCH" not in report
    assert FP.read_reference_list(job.out_pcm_sha256.get_path()) == FIRST  # every digest is written
    assert not os.listdir(tmp_path / "encoded")  # each encode is removed once hashed
    # the wrapper, used as the encode job's ffmpeg_binary, writes the same audio
    from i6_core.lib import corpus

    c = corpus.Corpus()
    c.load(job.corpus_file.get_path())
    rec = sorted(c.all_recordings(), key=lambda r: r.name)[0]
    ogg = FP.encode_recording(wrapper, job.corpus_file, rec, str(tmp_path / "via_wrapper"))
    assert FP.decode_pcm_sha256(REFERENCE_FFMPEG, ogg) == REFERENCE[rec.name]


def test_check_raises_on_a_changed_reference(run_check, tmp_path):
    changed = _changed_list(tmp_path, {"116-288045-0001": "0" * 64})
    with pytest.raises(RuntimeError) as err:
        run_check(reference_list=changed)
    msg = str(err.value)
    assert f"1 of {N} recordings differ from the banked audio (first: 116-288045-0001)" in msg
    assert "116-288045-0000" not in msg and _version_line(REFERENCE_FFMPEG) in msg
    job = FP.FfmpegPinCheckJob(
        ffmpeg_binary=tk.Path(REFERENCE_FFMPEG, hash_overwrite=default_tools.FFMPEG_HASH_OVERWRITE),
        corpus_file=tk.Path(os.path.abspath("flac.xml.gz")), reference_list=changed, test_first_n=N)
    assert not os.path.exists(job.out_ffmpeg_binary.get_path())  # no wrapper after a failed check
    assert FP.read_reference_list(job.out_pcm_sha256.get_path()) == FIRST  # the digests are still written


def test_accept_label_reports_a_mismatch_without_raising(run_check, tmp_path):
    """FFMPEG_PIN_ACCEPT set: the same mismatching reference is reported with the same numbers, not
    raised, and the wrapper is written; the report says the audio is not the banked audio.  Unset, the
    same reference raises."""
    changed = _changed_list(tmp_path, {"116-288045-0001": "0" * 64})
    job = run_check(reference_list=changed, accept_label="x86_64-ffmpeg-7.1.1")
    report = open(job.out_report.get_path()).read()
    assert f"checked {N} match {N - 1} mismatch 1\n" in report and report.count("MISMATCH ") == 1
    assert f"MISMATCH 116-288045-0001 pcm_sha256 {REFERENCE['116-288045-0001']} reference {'0' * 64}" in report
    assert ("ACCEPTED under FFMPEG_PIN_ACCEPT='x86_64-ffmpeg-7.1.1': 1 of 3 recordings differ from the banked "
            "audio (first: 116-288045-0001)") in report and "NOT the banked audio" in report
    assert os.access(job.out_ffmpeg_binary.get_path(), os.X_OK)
    assert open(job.out_version.get_path()).read().split("\n", 1)[0] == _version_line(REFERENCE_FFMPEG)
    assert FP.read_reference_list(job.out_pcm_sha256.get_path()) == FIRST
    with pytest.raises(RuntimeError, match="FFMPEG_PIN_ACCEPT"):
        run_check(reference_list=changed)


def test_check_raises_on_other_recordings(run_check, tmp_path):
    renamed = str(tmp_path / "renamed.txt.gz")
    FP.write_reference_list({f"x{name}": sha for name, sha in REFERENCE.items()}, renamed)
    with pytest.raises(RuntimeError, match="are not those of the reference list"):
        run_check(reference_list=tk.Path(renamed))
    # production (test_first_n unset) needs the whole list: the 3-recording corpus is refused
    with pytest.raises(RuntimeError, match=r"\(3, first .*\(2864, first"):
        run_check(test_first_n=None)
