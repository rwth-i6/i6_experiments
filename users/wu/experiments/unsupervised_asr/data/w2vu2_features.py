"""Ported from i6_experiments 5207c8adf users/wu/experiments/unsupervised_asr/w2vu2/features.py
(``MfccKmeansJob``, ``W2vu2FeatureDumpJob``, ``MergeW2vu2DataJob``) and w2vu2/dump_w2vu2_data.py
(``_mfcc``, ``_sil_mask``, ``fit_mfcc``, ``dump``).

The audio side of the wav2vec-U 2.0 GAN (section 1c): fairseq's ``ExtractedFeaturesDataset`` layout,
one directory with, per fairseq split ``{name}`` (``train``, ``valid``):

* ``{name}.npy`` -- every kept frame of every utterance, concatenated, fp16, ``[N, 1024]`` (mmap'd
  by fairseq; ``__getitem__`` does ``.float()``);
* ``{name}.lengths`` -- one frame count per line;
* ``{name}.km`` -- the aux target of L_ss: per utterance the 64-cluster MFCC k-means id of each kept
  frame, space separated, aligned 1:1 with the features (so the GAN runs
  ``target_downsample_rate: 1``);
* ``{name}.ids``, ``{name}.stats.txt`` -- the utterance ids and the counts (fairseq reads neither).

Port change: the source dumped wav2vec2 ``hidden_states[15]`` from the HF audio and VAD-trimmed it in
the same GPU job.  In the port the features and the VAD are already there: the rVAD-masked L15
stream of ``data.vad.BlankfreeVadHdfJob`` (``out_feature_hdfs``, ``out_raw_index_hdfs``,
``out_orig_length_hdfs``: the same encoder layer, the same rVAD at threshold 0.4 with 2 subframes, the
same "tail = silence" reconciliation), so :class:`W2vu2FeatureDataJob` only converts that stream,
and computes the ``.km`` from the audio (the ogg zips the VAD job read).  The source's frame rule is
kept exactly: an utterance's frames are ``t = min(len(features), len(mfcc))``; the kept frames are the
non-silence frames below ``t``; an utterance with fewer than ``min_length`` (3) kept frames is
dropped.  (The VAD stream keeps the non-silence frames below ``len(features)``; a kept frame at or
beyond ``len(mfcc)`` is dropped here and counted in ``frames_beyond_mfcc``.)

The two splits (the source's ``pipeline.py``): ``train`` = train-clean-100, ``valid`` = the HF
"dev" split = dev-clean followed by dev-other.  The utterance ORDER within a split is the VAD
stream's (train: the four shipped shards; dev: sorted ids), not the source's HF row order; the set
of utterances is the same.  fairseq shuffles train; valid is scored as a whole.

The MFCC k-means (:class:`MfccKmeansJob`): Kaldi MFCC 13 (torchaudio ``compliance.kaldi.mfcc``,
``use_energy=False``, 16 kHz) plus delta and delta-delta (``torchaudio.functional.compute_deltas``),
39-d at 100 Hz, every ``mfcc_downsample``-th frame (2: the 50 Hz wav2vec2 rate); the rVAD silence
frames removed; pooled over train-clean-100 in audio order, reservoir-thinned with
``RandomState(seed)`` to ``max_fit_vectors`` whenever the pool exceeds twice that; then
``MiniBatchKMeans(64, init="k-means++", max_iter=100, batch_size=10000, tol=0.0,
max_no_improvement=100, n_init=20, reassignment_ratio=0.0, random_state=seed, compute_labels=False,
init_size=None)`` (HuBERT's ``learn_kmeans.py`` values).  The assignment is the squared euclidean
argmin over the centroids, no normalisation.  The pool follows the audio order, so the port's
centroids are not bit-identical to the banked ones (``MfccKmeansJob.yRS4DGsXvqKm``, HF order).

torchaudio (MFCC) and scikit-learn (k-means) are needed in the job env, as in the source's
``speech_llm`` env.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from sisyphus import Job, Task, tk

__all__ = [
    "FEATURE_DIM",
    "FAIRSEQ_SPLITS",
    "mfcc_39",
    "silence_mask",
    "assign_km",
    "fit_mfcc_kmeans",
    "convert_split",
    "MfccKmeansJob",
    "W2vu2FeatureDataJob",
    "get_w2vu2_feature_data",
]

#: wav2vec2-large-lv60 ``hidden_states[15]``
FEATURE_DIM = 1024
#: fairseq split -> the VAD-stream splits it concatenates (the source's train / HF "dev")
FAIRSEQ_SPLITS = {"train": ("train",), "valid": ("dev-clean", "dev-other")}


# ===================================================================================================
# the per-utterance operators (dump_w2vu2_data.py)
# ===================================================================================================
def mfcc_39(wav, downsample: int):
    """Kaldi MFCC 13 + delta + delta-delta = 39-d @100 Hz, every ``downsample``-th frame, float32.

    Kaldi defaults (num_ceps=13, num_mel_bins=23, 25 ms/10 ms, use_energy=False) exactly as
    fairseq's HuBERT ``dump_mfcc_feature.py``.
    """
    import torch
    import torchaudio
    from torchaudio.compliance import kaldi

    x = torch.from_numpy(wav)[None, :]
    m = kaldi.mfcc(waveform=x, sample_frequency=16000, use_energy=False)  # [T100, 13]
    d1 = torchaudio.functional.compute_deltas(m.T[None])[0].T
    d2 = torchaudio.functional.compute_deltas(d1.T[None])[0].T
    return torch.cat([m, d1, d2], dim=1).numpy()[::downsample]  # [~T_enc, 39]


def silence_mask(wav, n: int, subframes: int, vad):
    """Encoder-rate rVAD silence mask reconciled to n frames (truncate, or pad tail as silence)."""
    import numpy as np

    from .vad_port import rvad_silence

    sil = rvad_silence(wav, vad=vad, subframes=subframes)
    if len(sil) >= n:
        return sil[:n]
    return np.concatenate([sil, np.ones(n - len(sil), dtype=bool)])


def assign_km(mf, cent, cent_sq):
    """Squared euclidean argmin over the centroids (the geometry MiniBatchKMeans fit on)."""
    d = (mf ** 2).sum(1, keepdims=True) - 2.0 * mf @ cent.T + cent_sq[None, :]
    return d.argmin(1)


def fit_mfcc_kmeans(audio: Iterable[Tuple[str, "object"]], *, num_clusters: int, mfcc_downsample: int,
                    vad_subframes: int, trim_silence: bool, max_fit_vectors: int, seed: int):
    """``(centroids [k, 39] float32, stats dict)`` -- the source's ``fit_mfcc`` on ``(id, wav)`` pairs."""
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans

    vad = None
    if trim_silence:
        from rVADfast import rVADfast

        vad = rVADfast(vad_threshold=0.4)
    rng = np.random.RandomState(seed)
    pool, n_seen, n_utts = [], 0, 0
    for _, wav in audio:
        n_utts += 1
        wav = np.asarray(wav, dtype=np.float32)
        m = mfcc_39(wav, mfcc_downsample)
        if vad is not None:
            m = m[~silence_mask(wav, len(m), vad_subframes, vad)]
        if len(m) == 0:
            continue
        n_seen += len(m)
        pool.append(m.astype(np.float32))
        if sum(len(p) for p in pool) > max_fit_vectors * 2:
            X = np.concatenate(pool)
            keep = rng.choice(len(X), max_fit_vectors, replace=False)
            pool = [X[keep]]
    X = np.concatenate(pool)
    if len(X) > max_fit_vectors:
        X = X[rng.choice(len(X), max_fit_vectors, replace=False)]
    print(f"[fit-mfcc] fitting k={num_clusters} on {len(X)}/{n_seen} frames, dim={X.shape[1]}", flush=True)

    km = MiniBatchKMeans(
        n_clusters=num_clusters, init="k-means++", max_iter=100, batch_size=10000,
        tol=0.0, max_no_improvement=100, n_init=20, reassignment_ratio=0.0,
        random_state=seed, compute_labels=False, init_size=None, verbose=0,
    ).fit(X)
    used = len(np.unique(km.predict(X)))
    stats = {"k": num_clusters, "utts": n_utts, "fit_frames": len(X), "seen_frames": n_seen,
             "dim": int(X.shape[1]), "nonempty_clusters": used, "inertia": float(km.inertia_),
             "mfcc_downsample": mfcc_downsample, "trim_silence": trim_silence, "seed": seed}
    return km.cluster_centers_.astype(np.float32), stats


class _ZipAudio:
    """Random access to the utterances of several ogg zips by bare utterance id."""

    def __init__(self, zip_paths: Sequence[str]):
        import zipfile

        from .ogg_zip import read_ogg_zip_index, utt_id_of_segment

        self._zips = {p: zipfile.ZipFile(p) for p in zip_paths}
        self._entries = {}
        for p in zip_paths:
            for entry in read_ogg_zip_index(p):
                utt = utt_id_of_segment(entry["seq_name"])
                assert utt not in self._entries, f"duplicate audio id {utt}"
                self._entries[utt] = (p, entry)

    def __contains__(self, utt: str) -> bool:
        return utt in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, utt: str):
        from .ogg_zip import read_ogg_zip_audio

        p, entry = self._entries[utt]
        return read_ogg_zip_audio(self._zips[p], p, entry)

    def close(self):
        for zf in self._zips.values():
            zf.close()


def _iter_stream(feature_hdf: str, raw_index_hdf: str, orig_length_hdf: str) -> Iterator[tuple]:
    """``(utt, feats [n, D] fp16, raw_index [n] int32, original)`` of one VAD-stream shard, in order."""
    import h5py
    import numpy as np

    with h5py.File(feature_hdf, "r") as ff, h5py.File(raw_index_hdf, "r") as fr, \
            h5py.File(orig_length_hdf, "r") as fo:
        tags = [t.decode("ascii") if isinstance(t, bytes) else str(t) for t in ff["seqTags"][:]]
        for other in (fr, fo):
            other_tags = [t.decode("ascii") if isinstance(t, bytes) else str(t) for t in other["seqTags"][:]]
            assert other_tags == tags, f"seq tags of {feature_hdf} and its companion differ"
        lengths = ff["seqLengths"][:, 0].astype(np.int64)
        assert np.array_equal(lengths, fr["seqLengths"][:, 0].astype(np.int64)), feature_hdf
        originals = np.asarray(fo["inputs"][:]).reshape(-1).astype(np.int64)
        assert len(originals) == len(tags), orig_length_hdf
        feats, raw = ff["inputs"], fr["inputs"]
        pos = 0
        for utt, n, original in zip(tags, lengths, originals):
            yield utt, np.asarray(feats[pos:pos + n]), np.asarray(raw[pos:pos + n]).reshape(-1), int(original)
            pos += int(n)
        assert pos == feats.shape[0] == raw.shape[0], (feature_hdf, pos, feats.shape[0], raw.shape[0])


def convert_split(
    *,
    name: str,
    streams: Sequence[Tuple[str, str, str]],
    audio,
    centroids,
    out_dir: str,
    mfcc_downsample: int = 2,
    min_length: int = 3,
    limit: Optional[int] = None,
) -> Dict[str, object]:
    """Write ``{name}.npy/.lengths/.km/.ids/.stats.txt`` into ``out_dir`` (module docstring).

    :param streams: ``[(feature_hdf, raw_index_hdf, orig_length_hdf), ...]`` in split order.
    :param audio: ``utt -> float32 waveform`` (``__getitem__``), e.g. :class:`_ZipAudio`.
    :param centroids: ``[k, 39]`` float32 MFCC centroids.
    :param limit: only the first ``limit`` utterances of every stream (tests).
    """
    import numpy as np

    import h5py

    cent = np.asarray(centroids, dtype=np.float32)
    cent_sq = (cent ** 2).sum(1)
    upper = 0
    for feature_hdf, _, _ in streams:
        with h5py.File(feature_hdf, "r") as fh:
            lens = fh["seqLengths"][:, 0].astype(np.int64)
            upper += int(lens[:limit].sum() if limit is not None else lens.sum())

    npy_path = os.path.join(out_dir, f"{name}.npy")
    data = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.float16, shape=(upper, FEATURE_DIM))
    lengths: List[int] = []
    ids: List[str] = []
    n_raw_tot = n_kept_tot = n_short = n_beyond = n_utt = 0
    with open(os.path.join(out_dir, f"{name}.km"), "w") as kmf:
        for stream in streams:
            for k, (utt, feats, raw_index, original) in enumerate(_iter_stream(*stream)):
                if limit is not None and k >= limit:
                    break
                assert feats.shape == (len(raw_index), FEATURE_DIM), (utt, feats.shape)
                wav = np.asarray(audio[utt], dtype=np.float32)
                mf = mfcc_39(wav, mfcc_downsample)
                t = min(original, len(mf))
                keep = raw_index < t
                n_beyond += int((~keep).sum())
                feats, raw_index = feats[keep], raw_index[keep]
                n_raw_tot += t
                n_utt += 1
                if len(feats) < min_length:  # fairseq's ExtractedFeaturesDataset drops these anyway
                    n_short += 1
                    continue
                km_ids = assign_km(mf[raw_index], cent, cent_sq)
                data[n_kept_tot:n_kept_tot + len(feats)] = feats
                kmf.write(" ".join(map(str, km_ids.tolist())) + "\n")
                lengths.append(len(feats))
                ids.append(utt)
                n_kept_tot += len(feats)
                if n_utt % 2000 == 0:
                    print(f"[{name}] {n_utt} utts, {n_kept_tot} frames", flush=True)
    data.flush()
    del data
    if n_kept_tot != upper:  # dropped frames or utterances: rewrite the .npy at its true length
        src = np.load(npy_path, mmap_mode="r")
        tmp_path = npy_path + ".tmp.npy"
        dst = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.float16, shape=(n_kept_tot, FEATURE_DIM))
        for a in range(0, n_kept_tot, 1 << 20):
            dst[a:a + (1 << 20)] = src[a:min(a + (1 << 20), n_kept_tot)]
        dst.flush()
        del dst, src
        os.replace(tmp_path, npy_path)

    with open(os.path.join(out_dir, f"{name}.lengths"), "w") as f:
        f.write("".join(f"{n}\n" for n in lengths))
    with open(os.path.join(out_dir, f"{name}.ids"), "w") as f:
        f.write("".join(f"{u}\n" for u in ids))
    drop = 1.0 - n_kept_tot / max(n_raw_tot, 1)
    stats = {"split": name, "utts": len(lengths), "utts_dropped_short": n_short, "frames_kept": n_kept_tot,
             "frames_raw": n_raw_tot, "frames_beyond_mfcc": n_beyond, "vad_dropped_frac": round(drop, 4),
             "dim": FEATURE_DIM, "dtype": "float16", "mfcc_downsample": mfcc_downsample,
             "min_length": min_length}
    with open(os.path.join(out_dir, f"{name}.stats.txt"), "w") as f:
        f.write("".join(f"{k}={v}\n" for k, v in stats.items()))
        f.write("km_rate=encoder_frame_rate_aligned_1to1 (=> fairseq target_downsample_rate must be 1)\n")
    print(f"[{name}] {len(lengths)} utts, {n_kept_tot} frames, vad dropped {drop:.1%}", flush=True)
    return stats


# ===================================================================================================
# the jobs
# ===================================================================================================
class MfccKmeansJob(Job):
    """64-cluster k-means over Kaldi MFCC+d+dd -- the encoder-independent aux target of w2v-U 2.0.

    CPU, in-process (module docstring).  :param ogg_zips: the fit audio (train-clean-100).
    """

    def __init__(
        self,
        *,
        ogg_zips: Sequence[tk.Path],
        num_clusters: int = 64,
        mfcc_downsample: int = 2,
        vad_subframes: int = 2,
        trim_silence: bool = True,
        max_fit_vectors: int = 2_000_000,
        seed: int = 0,
    ):
        super().__init__()
        self.ogg_zips = list(ogg_zips)
        self.num_clusters = num_clusters
        self.mfcc_downsample = mfcc_downsample
        self.vad_subframes = vad_subframes
        self.trim_silence = trim_silence
        self.max_fit_vectors = max_fit_vectors
        self.seed = seed

        self.out_centroids = self.output_path("mfcc_centroids.npy")
        self.out_stats = self.output_path("mfcc_kmeans.stats.txt")
        self.rqmt = {"cpu": 8, "mem": 48, "time": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import numpy as np

        from .ogg_zip import iter_ogg_zip_audio

        def audio():
            for p in self.ogg_zips:
                yield from iter_ogg_zip_audio(p.get_path())

        centroids, stats = fit_mfcc_kmeans(
            audio(), num_clusters=self.num_clusters, mfcc_downsample=self.mfcc_downsample,
            vad_subframes=self.vad_subframes, trim_silence=self.trim_silence,
            max_fit_vectors=self.max_fit_vectors, seed=self.seed)
        np.save(self.out_centroids.get_path(), centroids)
        with open(self.out_stats.get_path(), "w") as f:
            f.write("".join(f"{k}={v}\n" for k, v in stats.items()))
        print(f"[fit-mfcc] done: {stats['nonempty_clusters']}/{self.num_clusters} clusters used", flush=True)


class W2vu2FeatureDataJob(Job):
    """The VAD-masked L15 stream -> fairseq's ``task.data`` dir (``data/``), in-process.

    :param feature_hdfs: ``BlankfreeVadHdfJob.out_feature_hdfs`` (``{vad split: [shard HDFs]}``).
    :param raw_index_hdfs: ``BlankfreeVadHdfJob.out_raw_index_hdfs``.
    :param orig_length_hdfs: ``BlankfreeVadHdfJob.out_orig_length_hdfs``.
    :param ogg_zips: ``{vad split: [ogg zips]}``, the audio the VAD job read (for the ``.km``).
    :param mfcc_centroids: :class:`MfccKmeansJob`'s ``out_centroids``.
    :param splits: ``{fairseq split: [vad splits, in order]}`` (default :data:`FAIRSEQ_SPLITS`).
    """

    def __init__(
        self,
        *,
        feature_hdfs: Dict[str, Sequence[tk.Path]],
        raw_index_hdfs: Dict[str, Sequence[tk.Path]],
        orig_length_hdfs: Dict[str, Sequence[tk.Path]],
        ogg_zips: Dict[str, Sequence[tk.Path]],
        mfcc_centroids: tk.Path,
        splits: Optional[Dict[str, Sequence[str]]] = None,
        mfcc_downsample: int = 2,
        min_length: int = 3,
    ):
        super().__init__()
        splits = dict(FAIRSEQ_SPLITS if splits is None else splits)
        self.splits = {name: list(splits[name]) for name in sorted(splits)}
        needed = sorted({s for parts in self.splits.values() for s in parts})
        for what, d in (("feature_hdfs", feature_hdfs), ("raw_index_hdfs", raw_index_hdfs),
                        ("orig_length_hdfs", orig_length_hdfs), ("ogg_zips", ogg_zips)):
            missing = [s for s in needed if s not in d]
            if missing:
                raise ValueError(f"{what} lacks the splits {missing}")
        self.feature_hdfs = {s: list(feature_hdfs[s]) for s in needed}
        self.raw_index_hdfs = {s: list(raw_index_hdfs[s]) for s in needed}
        self.orig_length_hdfs = {s: list(orig_length_hdfs[s]) for s in needed}
        for s in needed:
            assert len(self.feature_hdfs[s]) == len(self.raw_index_hdfs[s]) == len(self.orig_length_hdfs[s]), s
        self.ogg_zips = {s: list(ogg_zips[s]) for s in needed}
        self.mfcc_centroids = mfcc_centroids
        self.mfcc_downsample = mfcc_downsample
        self.min_length = min_length

        self.out_dir = self.output_path("data", directory=True)
        self.out_files = {
            name: {ext: self.output_path(f"data/{name}.{ext}") for ext in ("npy", "lengths", "km", "ids", "stats.txt")}
            for name in self.splits
        }
        self.rqmt = {"cpu": 4, "mem": 16, "time": 12}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import json

        import numpy as np

        centroids = np.load(self.mfcc_centroids.get_path())
        summary = {}
        for name, parts in self.splits.items():
            audio = _ZipAudio([p.get_path() for s in parts for p in self.ogg_zips[s]])
            try:
                streams = [
                    (f.get_path(), r.get_path(), o.get_path())
                    for s in parts
                    for f, r, o in zip(self.feature_hdfs[s], self.raw_index_hdfs[s], self.orig_length_hdfs[s])
                ]
                summary[name] = convert_split(
                    name=name, streams=streams, audio=audio, centroids=centroids,
                    out_dir=self.out_dir.get_path(), mfcc_downsample=self.mfcc_downsample,
                    min_length=self.min_length)
            finally:
                audio.close()
        print(json.dumps(summary, indent=2), flush=True)


# ===================================================================================================
# the wiring
# ===================================================================================================
def get_w2vu2_feature_data() -> W2vu2FeatureDataJob:
    """The GAN's ``task.data``: the port's VAD stream (``inputs.get_inputs().vad``) and ogg zips, with
    the MFCC k-means fit on train-clean-100 (the source's ``MfccKmeansJob.yRS4DGsXvqKm`` arguments:
    k 64, ``mfcc_downsample`` 2, ``vad_subframes`` 2, trim silence, 2,000,000 fit vectors, seed 0)."""
    from ..inputs import get_inputs

    inputs = get_inputs()
    ogg = {"train": [inputs.ogg_zips["train-clean-100"]],
           "dev-clean": [inputs.ogg_zips["dev-clean"]],
           "dev-other": [inputs.ogg_zips["dev-other"]]}
    kmeans = MfccKmeansJob(ogg_zips=ogg["train"], num_clusters=64, mfcc_downsample=2, vad_subframes=2,
                           trim_silence=True, max_fit_vectors=2_000_000, seed=0)
    kmeans.add_alias("sae/1c/w2v2_lv60_l15/mfcc_kmeans")
    job = W2vu2FeatureDataJob(
        feature_hdfs=inputs.vad.out_feature_hdfs,
        raw_index_hdfs=inputs.vad.out_raw_index_hdfs,
        orig_length_hdfs=inputs.vad.out_orig_length_hdfs,
        ogg_zips=ogg,
        mfcc_centroids=kmeans.out_centroids,
        splits=FAIRSEQ_SPLITS,
        mfcc_downsample=2,
        min_length=3,
    )
    job.add_alias("sae/1c/w2v2_lv60_l15/data")
    return job
