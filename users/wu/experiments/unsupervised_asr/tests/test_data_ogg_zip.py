"""CPU tests of data.ogg_zip / data.librispeech on a tiny synthetic RETURNN ogg zip.

The decode test compares :func:`read_ogg_zip_audio` with RETURNN's own ``OggZipDataset`` (raw
features, no peak normalisation) sample for sample; it is skipped if ``returnn`` is not importable.
"""

from __future__ import annotations

import json
import os
import zipfile

import numpy as np
import pytest
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.data import ogg_zip as OZ
from i6_experiments.users.wu.experiments.unsupervised_asr.data.librispeech import LibriSpeechSplitIdsJob

UTTS = ["116-288045-0001", "116-288045-0000", "1255-138279-0003"]


def make_zip(tmp_path, corpus="dev-other", utts=UTTS, seconds=(1.0, 0.7, 1.3), seed=0):
    """A RETURNN ogg zip ``out.ogg.zip`` (``out.ogg.txt`` + ``out.ogg/<utt>/x.ogg``), 16 kHz Vorbis."""
    import soundfile

    rng = np.random.RandomState(seed)
    path = os.path.join(str(tmp_path), f"{corpus}.zip")
    name = os.path.splitext(os.path.basename(path))[0]
    entries, wavs = [], {}
    with zipfile.ZipFile(path, "w") as zf:
        for u, sec in zip(utts, seconds):
            n = int(16000 * sec)
            t = np.arange(n) / 16000.0
            wav = (0.3 * np.sin(2 * np.pi * 220 * t) * (t > 0.3) + 0.01 * rng.randn(n)).astype(np.float32)
            buf = os.path.join(str(tmp_path), f"{u}.ogg")
            soundfile.write(buf, wav, 16000, format="OGG", subtype="VORBIS")
            f = f"{u}/0.0000_{sec:.4f}.ogg"
            zf.write(buf, f"{name}/{f}")
            entries.append({"text": "X", "speaker_name": None, "file": f,
                            "seq_name": f"{corpus}/{u}/{u}", "duration": float(sec)})
            wavs[u] = wav
        zf.writestr(f"{name}.txt", repr(entries))
    return path, wavs


def test_utt_id_of_segment():
    assert OZ.utt_id_of_segment("dev-other/116-288045-0000/116-288045-0000") == "116-288045-0000"
    for bad in ("116-288045-0000", "a/b/c", "dev-other/x", "/u/u"):
        with pytest.raises(ValueError):
            OZ.utt_id_of_segment(bad)


def test_read_zip(tmp_path):
    path, wavs = make_zip(tmp_path)
    index = OZ.read_ogg_zip_index(path)
    assert [e["seq_name"].split("/")[-1] for e in index] == UTTS
    got = dict(OZ.iter_ogg_zip_audio(path))
    assert list(got) == UTTS
    for u, w in got.items():
        assert w.dtype == np.float32 and w.ndim == 1 and len(w) == len(wavs[u])
        # lossy codec: close to the written signal, not equal
        assert np.sqrt(np.mean((w - wavs[u]) ** 2)) < 0.05


def test_decode_matches_returnn(tmp_path):
    pytest.importorskip("returnn")
    from returnn.datasets.audio import OggZipDataset

    path, _ = make_zip(tmp_path)
    ds = OggZipDataset(path=path, audio={"features": "raw", "peak_normalization": False, "preemphasis": None},
                       targets=None)
    ds.init_seq_order(epoch=1)
    ours = dict(OZ.iter_ogg_zip_audio(path))
    n = 0
    while ds.is_less_than_num_seqs(n):
        ds.load_seqs(n, n + 1)
        tag = OZ.utt_id_of_segment(ds.get_tag(n))
        x = ds.get_data(n, "data")
        assert x.shape == (len(ours[tag]), 1)
        np.testing.assert_array_equal(x[:, 0], ours[tag])
        n += 1
    assert n == len(UTTS)


def test_split_ids_job(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a, _ = make_zip(tmp_path / "a", corpus="tiny-a")
    b, _ = make_zip(tmp_path / "b", corpus="tiny-b", utts=["19-198-0001", "19-198-0000"], seconds=(0.5, 0.5))
    job = LibriSpeechSplitIdsJob(ogg_zips={"tiny-b": tk.Path(b), "tiny-a": tk.Path(a)})
    job.out_ids = tk.Path(str(tmp_path / "split_ids.json"))
    job.out_per_split = {s: tk.Path(str(tmp_path / f"ids.{s}.json")) for s in job.out_per_split}
    job.run()
    assert json.load(open(tmp_path / "ids.tiny-a.json")) == sorted(UTTS)
    assert json.load(open(tmp_path / "split_ids.json")) == {"tiny-a": sorted(UTTS), "tiny-b": ["19-198-0000", "19-198-0001"]}
