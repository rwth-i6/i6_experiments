"""Ported from speech-llm c49559ce src/speech_llm/sae/quantize_states.py (``QuantizeStatesJob``, ``AssignUnitsJob``,
``MergeUnitsPklJob``, ``PackUnitsJob`` and the helpers they call) and
src/speech_llm/prefix_lm/model/util/units_attach.py (``pack_units_store``).

The enc50 unit stream of phase 4a: per-dim standardize (the seed dump's global statistics) ->
PCA-96 -> MiniBatchKMeans K=500 (seed 42, fit on at most 500,000 frames of the 10 h seed, assigned
everywhere, no run-length dedup), packed into the memmap store the joint VAD
(:class:`..data.vad.BlankfreeVadHdfJob`) reads by seq tag.

Banked chain reproduced: ``QuantizeStatesJob.FWpGhC941JMi`` (fit on the seed dump
``AvStatesJob.c4Ak1rACchRC``, units for seed + dev) + ``AssignUnitsJob.ARG5PMIPji84`` (the fixed
codebook on train-clean-100) -> ``MergeUnitsPklJob.iryDwagsbJkH`` (seed+dev codes win) ->
``PackUnitsJob.I0uzRMfUrKWC``.

Deviations:

* the states come from the fp16 L15 feature HDFs of :mod:`.features` instead of the source's state
  pickles.  The values are the same float16 numbers, and the HDF order of the seed dump is the seed
  row order, so the fit concatenation (and with it the seed-42 subsample) is unchanged.
  ``QuantizeStatesJob`` therefore takes ``states_hdfs`` (split -> HDFs) instead of
  ``states_pkl`` + ``hf_data_dir``, and ``AssignUnitsJob`` takes ``states_hdfs`` instead of
  ``states_pkls``;
* the banked merge also carried the 360 h / 500 h assignments (960 h, 286,808 utterances); the
  phase-4a streams need train-clean-100 + dev only (34,106), so the port merges those two sources.
  The codes of every utterance present in both are the same;
* the coverage contract of ``AssignUnitsJob`` / ``MergeUnitsPklJob`` read the ids of HF dataset
  splits (``hf_data_dir`` + ``splits``); with the ogg-zip audio source it reads the bare-id json
  lists of :func:`..data.librispeech.get_split_ids` (``split_ids``) instead;
* ``per_utt_standardize=True`` and ``QuantizeStatesJob(dedup=True)`` (unused by phase 4a) are removed
  and raise ``ValueError`` (``standardize_frames_per_utt`` and ``run_length_dedup_np`` are dropped);
  ``AssignUnitsJob`` keeps the source's ``assert not dedup``; a per-utterance-standardized codebook
  fed to ``AssignUnitsJob`` fails its assertion;
* ``PackUnitsJob`` calls the local copy of ``units_attach.pack_units_store`` (no RETURNN import);
  the lazy readers of the store live with the consumer.
"""

from __future__ import annotations

import os
from functools import cache
from typing import Dict, Iterator, Optional, Sequence, Tuple

from sisyphus import Job, Task, tk

__all__ = [
    "STATE_FRAME_HZ",
    "standardize_frames",
    "fit_pca_kmeans",
    "quantize_utt",
    "usage_stats",
    "reconstruct_quantizer",
    "units_as_int_list",
    "iter_hdf_seqs",
    "pack_units_store",
    "QuantizeStatesJob",
    "AssignUnitsJob",
    "MergeUnitsPklJob",
    "PackUnitsJob",
    "get_units_store",
]

_alias_prefix = "sae/4a/units"

# the value the banked quantizer.pkl / units.stats.txt carry (a 12.5 Hz label left over from the
# post-adapter arm; the enc50 frames are 50 Hz).  Kept so the outputs stay byte-comparable.
STATE_FRAME_HZ = 12.5  # AV soft-prompt rate (w2v2 50 Hz / downsampling_factor 4)
_STD_FLOOR = 1e-5  # same floor as build_av_states.py / build_feats.py
_STORE_FILES = ("data.npy", "uids.npy", "offsets.npy", "lengths.npy")  # units_attach.py


# ---------------------------------------------------------------------------------------------------
# pure helpers (numerics unchanged)
# ---------------------------------------------------------------------------------------------------
def standardize_frames(x, mean, std):
    """(x - mean) / max(std, floor) as float32; x [T, D], mean/std [D]."""
    import numpy as np

    std = np.maximum(np.asarray(std, dtype=np.float32), _STD_FLOOR)
    return (np.asarray(x, dtype=np.float32) - np.asarray(mean, dtype=np.float32)) / std


def fit_pca_kmeans(fit_frames, *, pca_dim: int, num_clusters: int, seed: int):
    """(pca, km) fit on standardized frames [N, D]; k-means ctor mirrors ``build_units.py``."""
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    pca = PCA(n_components=pca_dim, svd_solver="randomized", random_state=seed).fit(fit_frames)
    km = MiniBatchKMeans(
        n_clusters=num_clusters, random_state=seed, batch_size=4096, n_init=3, max_iter=100
    ).fit(pca.transform(fit_frames).astype("float32"))
    return pca, km


def quantize_utt(pca, km, x_std):
    """Standardized frames [T, D] -> int64 cluster ids [T]."""
    return km.predict(pca.transform(x_std).astype("float32")).astype("int64")


def usage_stats(counts) -> Tuple[int, float, float]:
    """(dead_clusters, norm_entropy in [0,1], entropy_nats) from per-cluster frame counts [K]."""
    import numpy as np

    counts = np.asarray(counts, dtype=np.float64)
    dead = int((counts == 0).sum())
    total = counts.sum()
    p = counts[counts > 0] / total if total else np.zeros(0)
    ent = float(-(p * np.log(p)).sum()) if p.size else 0.0
    norm = ent / np.log(len(counts)) if len(counts) > 1 else 0.0
    return dead, float(norm), ent


def reconstruct_quantizer(q: dict):
    """(mean, std, pca, km) rebuilt from a ``QuantizeStatesJob`` quantizer.pkl dict.

    The sklearn estimators are reconstructed with the SAME ctor args as :func:`fit_pca_kmeans` and
    their fitted arrays are set verbatim, so ``pca.transform`` / ``km.predict`` run the identical
    code paths (and float32 arithmetic) as the original fit-time assignment.
    """
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA

    import numpy as np

    seed = int(q["params"]["seed"])
    comps = np.asarray(q["pca_components"])
    pca = PCA(n_components=comps.shape[0], svd_solver="randomized", random_state=seed)
    pca.components_ = comps
    pca.mean_ = np.asarray(q["pca_mean"])
    pca.n_components_ = comps.shape[0]
    pca.n_features_in_ = comps.shape[1]
    # touched by transform() only through get_namespace (whiten=False => values unused)
    pca.explained_variance_ = np.zeros(comps.shape[0], dtype=comps.dtype)

    cents = np.asarray(q["centroids"])
    km = MiniBatchKMeans(n_clusters=cents.shape[0], random_state=seed, batch_size=4096, n_init=3, max_iter=100)
    km.cluster_centers_ = cents
    km.n_features_in_ = cents.shape[1]
    km._n_threads = 1
    return np.asarray(q["mean"]), np.asarray(q["std"]), pca, km


def units_as_int_list(seq):
    """Normalize a units sequence (list[int] or np integer array) to a plain list[int]."""
    import numpy as np

    if isinstance(seq, np.ndarray):
        return [int(x) for x in seq.tolist()]
    return [int(x) for x in seq]


def _read_id_union(paths) -> set:
    """Union of json id lists (``LibriSpeechSplitIdsJob`` outputs)."""
    import json

    out = set()
    for p in paths:
        with open(p.get_path() if hasattr(p, "get_path") else str(p)) as fh:
            out |= {str(i) for i in json.load(fh)}
    return out


def iter_hdf_seqs(paths) -> Iterator[Tuple[str, "object"]]:
    """Yield ``(seq_tag, float16 [T, D])`` from RETURNN feature HDFs, in stored order.

    Replaces the source's ``iter_states_pickles``: one HDF is read into memory at a time (a
    train shard is ~9 GB float16, the source's shard pickle the same).
    """
    import h5py
    import numpy as np

    for p in paths:
        path = p.get_path() if hasattr(p, "get_path") else str(p)
        with h5py.File(path, "r") as fh:
            tags = fh["seqTags"][:]
            lengths = fh["seqLengths"][:, 0]
            inputs = np.asarray(fh["inputs"][:])
        assert int(lengths.sum()) == inputs.shape[0], (path, int(lengths.sum()), inputs.shape)
        pos = 0
        for raw_tag, length in zip(tags, lengths):
            tag = raw_tag.decode("ascii") if isinstance(raw_tag, bytes) else str(raw_tag)
            yield tag, inputs[pos:pos + int(length)]
            pos += int(length)
        del inputs


def pack_units_store(units: dict, out_dir: str) -> dict:
    """Write the packed store for ``{seq_tag: list[int] | int-array}``; returns summary stats.

    Store layout: ``data.npy`` int16 [total_tokens] (all sequences concatenated), ``uids.npy``
    S{w} [n] (seq tags, lexicographically sorted), ``offsets.npy`` int64 [n], ``lengths.npy``
    int32 [n].  uids must be ASCII (LibriSpeech ids are); ids must fit int16 (K=500 does).  Empty
    sequences are representable (length 0).
    """
    import numpy as np

    os.makedirs(out_dir, exist_ok=True)
    uids = sorted(units.keys())
    assert uids, "empty units dict"
    for u in uids:
        assert u.isascii(), f"non-ascii seq_tag {u!r}"
    width = max(len(u) for u in uids)
    uid_arr = np.array([u.encode("ascii") for u in uids], dtype=f"S{width}")
    assert uid_arr.shape[0] == len(set(uids)), "duplicate seq_tags"

    lengths = np.array([len(units[u]) for u in uids], dtype=np.int32)
    offsets = np.zeros(len(uids), dtype=np.int64)
    np.cumsum(lengths[:-1], out=offsets[1:])
    data = np.zeros(int(lengths.sum()), dtype=np.int16)
    for i, u in enumerate(uids):
        seq = np.asarray(units[u])
        if seq.size:
            assert seq.min() >= 0 and seq.max() < 2**15, f"{u}: ids out of int16 range"
            data[offsets[i] : offsets[i] + lengths[i]] = seq.astype(np.int16)

    for name, arr in zip(_STORE_FILES, (data, uid_arr, offsets, lengths)):
        np.save(os.path.join(out_dir, name), arr)
    return {
        "n_utts": len(uids),
        "total_tokens": int(lengths.sum()),
        "uid_width": width,
        "bytes_data": data.nbytes,
        "bytes_index": uid_arr.nbytes + offsets.nbytes + lengths.nbytes,
    }


# ---------------------------------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------------------------------
class QuantizeStatesJob(Job):
    """PCA + MiniBatchKMeans discretization of the L15 state dump (see module doc).

    ``states_hdfs`` maps each split to its feature HDFs; PCA/k-means fit ONLY on ``fit_split``
    frames, concatenated in HDF order; ``stats_npy`` is the fit dump's global float32 ``[2, D]``
    mean/std.  Assignment covers every utterance of every split; the stats file reports usage
    entropy, dead clusters and tokens/s per split, rows in ``splits`` order.
    """

    __sis_hash_exclude__ = {"per_utt_standardize": False}

    def __init__(
        self,
        *,
        states_hdfs: Dict[str, Sequence[tk.Path]],
        stats_npy: tk.Path,
        num_clusters: int = 500,
        pca_dim: int = 96,
        seed: int = 42,
        max_fit_vectors: int = 500_000,
        splits: Sequence[str] = ("train", "dev"),
        fit_split: str = "train",
        dedup: bool = False,
        per_utt_standardize: bool = False,
    ):
        super().__init__()
        if dedup:
            raise ValueError("dedup=True is not ported (the enc50 stream is no-dedup)")
        if per_utt_standardize:
            raise ValueError("per_utt_standardize=True is not ported (phase 4a uses the global statistics)")
        if set(splits) != set(states_hdfs):
            raise ValueError(f"splits {list(splits)} != states_hdfs keys {sorted(states_hdfs)}")
        if fit_split not in splits:
            raise ValueError(f"fit_split {fit_split!r} not in splits {list(splits)}")
        self.states_hdfs = {s: list(states_hdfs[s]) for s in splits}
        self.stats_npy = stats_npy
        self.num_clusters = num_clusters
        self.pca_dim = pca_dim
        self.seed = seed
        self.max_fit_vectors = max_fit_vectors
        self.splits = list(splits)
        self.fit_split = fit_split
        self.dedup = dedup
        self.per_utt_standardize = per_utt_standardize
        self.out_units = self.output_path("units.pkl")
        self.out_quantizer = self.output_path("quantizer.pkl")
        self.out_stats = self.output_path("units.stats.txt")
        # ~3.7 GB f16 seed states + a float32 fit matrix (~7.4 GB) + PCA workspace.
        self.rqmt = {"cpu": 8, "mem": 64, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import pickle

        import numpy as np

        stats = np.load(self.stats_npy.get_path())
        assert stats.ndim == 2 and stats.shape[0] == 2, f"stats layout {stats.shape}, expected [2, D]"
        mean, std = stats[0], stats[1]

        states = {}
        tags_by_split = {}
        for split in self.splits:
            tags = []
            for tag, x in iter_hdf_seqs(self.states_hdfs[split]):
                assert tag not in states, f"duplicate seq_tag {tag!r} across splits"
                states[tag] = x
                tags.append(tag)
            tags_by_split[split] = tags

        _std = lambda x: standardize_frames(x, mean, std)  # noqa: E731

        # --- fit PCA + k-means on standardized fit-split frames ---
        rng = np.random.default_rng(self.seed)
        fit = np.concatenate([_std(states[t]) for t in tags_by_split[self.fit_split] if states[t].shape[0]])
        n_avail = fit.shape[0]
        if n_avail > self.max_fit_vectors:
            fit = fit[rng.permutation(n_avail)[: self.max_fit_vectors]]
        print(f"fitting PCA({self.pca_dim}) + MiniBatchKMeans(K={self.num_clusters}) on "
              f"{fit.shape} (avail {n_avail})", flush=True)
        pca, km = fit_pca_kmeans(fit, pca_dim=self.pca_dim, num_clusters=self.num_clusters, seed=self.seed)
        evr = float(np.sum(pca.explained_variance_ratio_))
        del fit

        # --- assign every utterance; usage stats on the pre-dedup frame stream ---
        units = {}
        counts = np.zeros(self.num_clusters, dtype=np.int64)
        rows = []  # (split, n_utt, frames, tokens)
        for split in self.splits:
            n_utt = frames = tokens = 0
            for tag in tags_by_split[split]:
                x = states[tag]
                if x.shape[0] == 0:
                    units[tag] = []
                    n_utt += 1
                    continue
                labels = quantize_utt(pca, km, _std(x))
                assert int(labels.max()) < self.num_clusters
                counts += np.bincount(labels, minlength=self.num_clusters)
                out = labels
                units[tag] = out.tolist()
                frames += len(labels)
                tokens += len(out)
                n_utt += 1
            rows.append((split, n_utt, frames, tokens))

        with open(self.out_units.get_path(), "wb") as fh:
            pickle.dump(units, fh)
        with open(self.out_quantizer.get_path(), "wb") as fh:
            pickle.dump(
                {
                    "mean": mean.astype(np.float32),
                    "std": std.astype(np.float32),
                    "pca_mean": pca.mean_.astype(np.float32),
                    "pca_components": pca.components_.astype(np.float32),
                    "explained_variance_ratio": pca.explained_variance_ratio_.astype(np.float32),
                    "centroids": km.cluster_centers_.astype(np.float32),
                    "params": {
                        "num_clusters": self.num_clusters,
                        "pca_dim": self.pca_dim,
                        "seed": self.seed,
                        "fit_split": self.fit_split,
                        "dedup": self.dedup,
                        "per_utt_standardize": self.per_utt_standardize,
                        "frame_hz": STATE_FRAME_HZ,
                    },
                },
                fh,
            )

        dead, norm_ent, ent = usage_stats(counts)
        tot_frames = sum(r[2] for r in rows)
        tot_tokens = sum(r[3] for r in rows)
        lines = [
            f"av-state units: K={self.num_clusters} pca_dim={self.pca_dim} (evr={evr:.4f}) "
            f"seed={self.seed} dedup={self.dedup} per_utt_std={self.per_utt_standardize} "
            f"frame_hz={STATE_FRAME_HZ:g}",
            f"fit: split={self.fit_split} frames_used={min(n_avail, self.max_fit_vectors)} "
            f"frames_available={n_avail}",
            f"codebook: dead_clusters={dead}/{self.num_clusters} usage_norm_entropy={norm_ent:.4f} "
            f"({ent:.4f} nats, floor ln K={np.log(self.num_clusters):.4f}) over {int(counts.sum())} frames",
            f"tokens/s: {tot_tokens / (tot_frames / STATE_FRAME_HZ):.3f} "
            f"(= {STATE_FRAME_HZ:g} Hz pre-dedup; rho={tot_tokens / max(tot_frames, 1):.4f})",
            "",
            f"{'split':<8} {'utts':>7} {'frames_avg':>11} {'tokens_avg':>11} {'rho':>8}",
        ]
        for split, n_utt, frames, tokens in rows:
            lines.append(
                f"{split:<8} {n_utt:>7} {frames / max(n_utt, 1):>11.2f} {tokens / max(n_utt, 1):>11.2f} "
                f"{tokens / max(frames, 1):>8.4f}"
            )
        lines.append(
            f"{'TOTAL':<8} {sum(r[1] for r in rows):>7} {tot_frames / max(sum(r[1] for r in rows), 1):>11.2f} "
            f"{tot_tokens / max(sum(r[1] for r in rows), 1):>11.2f} {tot_tokens / max(tot_frames, 1):>8.4f}"
        )
        with open(self.out_stats.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


class AssignUnitsJob(Job):
    """ASSIGN-ONLY quantization: label state dumps with an EXISTING (fixed) codebook -- no refit.

    ``quantizer_pkl`` is a finished ``QuantizeStatesJob.out_quantizer`` (standardize -> PCA ->
    k-means arrays); ``states_hdfs`` is a list of feature HDFs (processed one at a time).
    ``split_ids`` (json id lists) give the coverage contract (dump keys == exactly their union).
    Units are stored as np.int16 arrays ({tag: array}).
    """

    def __init__(
        self,
        *,
        states_hdfs: Sequence[tk.Path],
        quantizer_pkl: tk.Path,
        split_ids: Optional[Sequence[tk.Path]] = None,
        dedup: bool = False,
    ):
        super().__init__()
        assert not dedup, "the 12.5 Hz avunits chain is no-dedup by contract (see module doc)"
        self.states_hdfs = list(states_hdfs)
        self.quantizer_pkl = quantizer_pkl
        self.split_ids = None if split_ids is None else list(split_ids)
        self.dedup = dedup
        self.out_units = self.output_path("units.pkl")
        self.out_stats = self.output_path("units.stats.txt")
        # one HDF at a time (<= ~9 GB f16) + f32 per-utt working copies + the units dict.
        self.rqmt = {"cpu": 8, "mem": 56, "time": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import pickle

        import numpy as np

        with open(self.quantizer_pkl.get_path(), "rb") as fh:
            q = pickle.load(fh)
        mean, std, pca, km = reconstruct_quantizer(q)
        K = int(q["params"]["num_clusters"])
        # the codebook records how its fit frames were standardized; only the global convention is ported
        assert not q["params"].get("per_utt_standardize", False), "per-utterance standardization is not ported"
        _std = lambda x: standardize_frames(x, mean, std)  # noqa: E731

        units = {}
        counts = np.zeros(K, dtype=np.int64)
        for tag, x in iter_hdf_seqs(self.states_hdfs):
            assert tag not in units, f"duplicate seq_tag {tag!r} across state HDFs"
            if x.shape[0] == 0:
                units[tag] = np.zeros(0, dtype=np.int16)
                continue
            labels = quantize_utt(pca, km, _std(x))
            assert int(labels.max()) < K
            counts += np.bincount(labels, minlength=K)
            units[tag] = labels.astype(np.int16)

        if self.split_ids is not None:
            covered = _read_id_union(self.split_ids)
            assert set(units) == covered, (
                f"assigned keys != split ids: {len(set(units) - covered)} extra, "
                f"{len(covered - set(units))} missing (contract: dumps cover exactly the split_ids)"
            )

        with open(self.out_units.get_path(), "wb") as fh:
            pickle.dump(units, fh)

        dead, norm_ent, ent = usage_stats(counts)
        tot = int(counts.sum())
        lines = [
            f"assign-only units: K={K} (codebook from {self.quantizer_pkl.get_path()})",
            f"utts={len(units)} frames={tot}",
            f"codebook usage on THIS corpus: dead_clusters={dead}/{K} usage_norm_entropy={norm_ent:.4f} "
            f"({ent:.4f} nats)",
        ]
        with open(self.out_stats.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


class MergeUnitsPklJob(Job):
    """Merge unit pickles by seq_tag with PRIORITY order (first source wins on overlap).

    Phase-4a use: [the seed+dev units of ``QuantizeStatesJob``, the train-clean-100 assignment] ->
    one {tag: np.int16 array} dict covering exactly the union of ``split_ids`` (asserted; the
    source: the ids of ``hf_data_dir`` splits).  Overlap agreement between the winning source and
    the losers is reported.
    """

    def __init__(
        self,
        *,
        sources: Sequence[tk.Path],
        split_ids: Sequence[tk.Path],
    ):
        super().__init__()
        assert len(sources) >= 1
        self.sources = list(sources)
        self.split_ids = list(split_ids)
        self.out_units = self.output_path("units.pkl")
        self.out_stats = self.output_path("units.stats.txt")
        self.rqmt = {"cpu": 2, "mem": 24, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import pickle

        import numpy as np

        merged = {}
        lines = []
        for i, src in enumerate(self.sources):
            with open(src.get_path(), "rb") as fh:
                d = pickle.load(fh)
            overlap = set(d) & set(merged)
            same = frames = frames_same = 0
            for t in overlap:
                a, b = units_as_int_list(merged[t]), units_as_int_list(d[t])
                frames += len(a)
                if a == b:
                    same += 1
                    frames_same += len(a)
                elif len(a) == len(b):
                    frames_same += int(np.sum(np.asarray(a) == np.asarray(b)))
            for t, seq in d.items():
                if t not in merged:
                    merged[t] = np.asarray(seq, dtype=np.int16)
            lines.append(
                f"source[{i}] {src.get_path()}: {len(d)} tags, {len(d) - len(overlap)} taken, "
                f"{len(overlap)} overlap (kept higher-priority codes; utt-identical {same}/{len(overlap)}"
                + (f", frame agreement {frames_same / frames:.6f}" if frames else "")
                + ")"
            )

        covered = _read_id_union(self.split_ids)
        assert set(merged) == covered, (
            f"merged keys != split ids: {len(set(merged) - covered)} extra, {len(covered - set(merged))} missing"
        )

        with open(self.out_units.get_path(), "wb") as fh:
            pickle.dump(merged, fh)
        lines.append(f"merged: {len(merged)} tags == union of {len(self.split_ids)} split id lists")
        with open(self.out_stats.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


class PackUnitsJob(Job):
    """Pack a ``{seq_tag: ids}`` units pickle into the memmap store (:func:`pack_units_store`:
    flat int16 payload + sorted uid index), opened lazily via ``np.load(mmap_mode="r")``."""

    def __init__(self, *, units_pkl: tk.Path):
        super().__init__()
        self.units_pkl = units_pkl
        self.out_store = self.output_path("units_store", directory=True)
        self.out_stats = self.output_path("units_store.stats.txt")
        self.rqmt = {"cpu": 2, "mem": 16, "time": 2}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import pickle

        with open(self.units_pkl.get_path(), "rb") as fh:
            units = pickle.load(fh)
        stats = pack_units_store(units, self.out_store.get_path())
        lines = [f"packed units store <- {self.units_pkl.get_path()}"] + [
            f"{k} = {v}" for k, v in stats.items()
        ]
        with open(self.out_stats.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------------------------------------
# graph
# ---------------------------------------------------------------------------------------------------
@cache
def get_units_store() -> tk.Path:
    """The packed enc50 unit store over train-clean-100 + dev (the joint VAD's ``units_store``)."""
    from ..data.librispeech import get_split_ids
    from .features import get_l15_feature_dumps

    dumps = get_l15_feature_dumps()
    hdf = lambda s: [j.out_files["feats.hdf"] for j in dumps[s]]  # noqa: E731
    quant = QuantizeStatesJob(
        states_hdfs={"train": hdf("seed"), "dev": hdf("dev-clean") + hdf("dev-other")},
        stats_npy=dumps["seed"][0].out_files["global_stats.npy"],
    )
    quant.add_alias(f"{_alias_prefix}/quantize_seed")
    assign = AssignUnitsJob(
        states_hdfs=hdf("train"), quantizer_pkl=quant.out_quantizer,
        split_ids=[get_split_ids("train-clean-100")],
    )
    assign.add_alias(f"{_alias_prefix}/assign_train")
    merge = MergeUnitsPklJob(
        sources=[quant.out_units, assign.out_units],
        split_ids=[get_split_ids(s) for s in ("train-clean-100", "dev-clean", "dev-other")],
    )
    merge.add_alias(f"{_alias_prefix}/merge")
    pack = PackUnitsJob(units_pkl=merge.out_units)
    pack.add_alias(f"{_alias_prefix}/pack")
    tk.register_output(f"{_alias_prefix}/units.stats.txt", quant.out_stats)
    tk.register_output(f"{_alias_prefix}/merge.stats.txt", merge.out_stats)
    return pack.out_store
