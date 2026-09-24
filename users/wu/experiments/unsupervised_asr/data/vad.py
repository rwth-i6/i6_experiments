"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/blankfree_data_jobs.py (``BlankfreeVadHdfJob``) and
src/speech_llm/sae/emc/blankfree_data.py (``prepare_blankfree_data``, ``_feature_index``).

Joint rVAD masking of the L15 feature stream and the enc50 unit stream: one rVADfast mask per
utterance (50 Hz, threshold 0.4, subframes 2, tail = silence) selects the kept frames of BOTH
streams, so the two stay frame-aligned.  Outputs per split and shard: ``feats`` (1024-d),
``units`` (sparse, 500), ``raw_index`` (the kept 50 Hz frame positions) and ``orig_length`` (the
unmasked length), plus ``manifest.json`` and ``summary.txt``.

Cut: the source cross-checked every per-utterance kept count and the split totals against the
wav2vec-U 2.0 reproduction (``reproduction_dir`` = ``MergeW2vu2DataJob.wxBxCIaJpqS2``), and loaded
``vad_port`` by file path (``vad_port_path``).  Both arguments are gone: ``vad_port`` is a normal
import, and the reproduction check is replaced by the optional ``expected_counts`` (per-split
utterances / raw frames / kept frames, each key optional), whose banked values are
:data:`BANKED_VAD_COUNTS` (``BlankfreeVadHdfJob.SAjz8y1cT06g``).  Without a per-utterance
reference the manifest's ``length_mismatches`` is always empty and ``reference_kept_frames`` is the
expected kept total (``None`` when not given); the rest of the layout is unchanged.

Audio source (brief change 2026-09-23): the source decoded the HF Ogg dataset
(``hf_dataset``, splits ``train`` / ``dev``); the port reads the i6 LibriSpeech ogg zips, one list
per feature split (``ogg_zips``), decoded exactly as RETURNN's ``OggZipDataset`` decodes them for
the L15 forward (:mod:`.ogg_zip`), with the segment names mapped to the bare utterance ids the
feature HDFs and the unit store carry.  The zips' Ogg files are encoded by the pinned ffmpeg 7.1.1
(:func:`.librispeech.get_bliss_corpus`); on a 16-utterance dev-other subset their decode is
sample-identical to the banked HF Ogg decode, so :data:`BANKED_VAD_COUNTS` is expected to hold.
That is verified on the subset only, not on the full splits.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Sequence

from sisyphus import Job, Task, tk

__all__ = ["BANKED_VAD_COUNTS", "BlankfreeVadHdfJob", "prepare_blankfree_data", "write_counts_report"]

#: summary of the banked joint-VAD run ``BlankfreeVadHdfJob.SAjz8y1cT06g`` (identical to the
#: wav2vec-U 2.0 reproduction totals it was checked against).
BANKED_VAD_COUNTS: Dict[str, Dict[str, int]] = {
    "train": {"utterances": 28539, "original_frames": 18088388, "kept_frames": 15427853},
    "dev-clean": {"utterances": 2703, "original_frames": 968057, "kept_frames": 831372},
    "dev-other": {"utterances": 2864, "original_frames": 919980, "kept_frames": 781130},
}

_COUNT_KEYS = ("utterances", "original_frames", "kept_frames")


def _feature_index(paths: dict[str, list[str]]) -> dict[str, tuple[str, int, int]]:
    import h5py

    index = {}
    for split, files in paths.items():
        for shard, path in enumerate(files):
            with h5py.File(path, "r") as fh:
                tags = fh["seqTags"][:]
                lengths = fh["seqLengths"][:, 0]
                assert len(tags) == len(lengths), path
                for tag, length in zip(tags, lengths):
                    tag = tag.decode("ascii") if isinstance(tag, bytes) else str(tag)
                    assert tag not in index, f"duplicate feature tag {tag}"
                    index[tag] = (split, shard, int(length))
    return index


def write_counts_report(summary: dict, expected_counts: dict[str, dict[str, int]], path: str) -> dict:
    """Report-only count check: per split and count key the observed total, the expected one and
    ``(observed - expected) / expected``; plus ``max_abs_rel_diff`` over everything.  Written to
    ``path`` (json) and returned."""
    splits = {}
    worst = 0.0
    for split in sorted(expected_counts):
        rows = {}
        for key in _COUNT_KEYS:
            if key not in expected_counts[split]:
                continue
            expected = int(expected_counts[split][key])
            observed = int(summary[split][key])
            rel = (observed - expected) / expected if expected else (0.0 if observed == 0 else float("inf"))
            worst = max(worst, abs(rel))
            rows[key] = {"observed": observed, "expected": expected, "rel_diff": rel}
        splits[split] = rows
    report = {"mode": "report-only (ffmpeg pin accept label): a mismatch does not raise",
              "all_equal": all(r["observed"] == r["expected"] for rows in splits.values() for r in rows.values()),
              "max_abs_rel_diff": worst, "splits": splits}
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    return report


def prepare_blankfree_data(
    *,
    ogg_zips: dict[str, list[str]],
    feature_hdfs: dict[str, list[str]],
    units_store: str,
    out_feature_hdfs: dict[str, list[str]],
    out_units_hdfs: dict[str, list[str]],
    out_raw_index_hdfs: dict[str, list[str]],
    out_orig_length_hdfs: dict[str, list[str]],
    manifest_path: str,
    expected_counts: Optional[dict[str, dict[str, int]]] = None,
    counts_report_path: Optional[str] = None,
) -> None:
    """Joint rVAD masking (module doc).  With ``expected_counts``, a mismatch of any given count
    raises after the manifest is written; with ``counts_report_path`` as well (report-only mode, used
    under an ffmpeg accept label), nothing raises on a mismatch: the observed totals, the expected
    ones and their relative differences are written to ``counts_report_path`` instead."""
    import h5py
    import numpy as np
    from rVADfast import rVADfast
    from returnn.datasets.hdf import SimpleHDFWriter

    from . import vad_port
    from .ogg_zip import iter_ogg_zip_audio

    assert set(ogg_zips) == set(feature_hdfs), (sorted(ogg_zips), sorted(feature_hdfs))
    if expected_counts is not None:
        assert set(expected_counts) == set(feature_hdfs), (sorted(expected_counts), sorted(feature_hdfs))
    assert counts_report_path is None or expected_counts is not None, "a counts report needs expected_counts"

    feature_index = _feature_index(feature_hdfs)

    masks = {}
    summary = {s: {"utterances": 0, "original_frames": 0, "kept_frames": 0,
                   "reference_kept_frames": None, "length_mismatches": [], "short_or_empty": []}
               for s in feature_hdfs}
    vad = rVADfast(vad_threshold=0.4)
    seen = set()

    def _audio():
        for audio_split in sorted(ogg_zips):
            for zip_path in ogg_zips[audio_split]:
                for tag, wav in iter_ogg_zip_audio(zip_path):
                    yield audio_split, tag, wav

    for audio_split, tag, wav in _audio():
        assert tag not in seen, f"duplicate audio tag {tag}"
        seen.add(tag)
        assert tag in feature_index, f"audio tag without features: {tag}"
        split, _, original = feature_index[tag]
        assert split == audio_split, (tag, split, audio_split)
        wav = np.asarray(wav, dtype=np.float32)
        silence = vad_port.rvad_silence(wav, sr=16000, vad=vad, subframes=2)
        if len(silence) < original:
            silence = np.pad(silence, (0, original - len(silence)), constant_values=True)
        indices = np.flatnonzero(~silence[:original]).astype(np.int32)
        masks[tag] = indices
        stats = summary[split]
        stats["utterances"] += 1
        stats["original_frames"] += original
        stats["kept_frames"] += len(indices)
        if len(indices) < 2:
            stats["short_or_empty"].append([tag, len(indices)])
    missing = set(feature_index) - set(masks)
    assert not missing, f"{len(missing)} feature tags without audio, e.g. {sorted(missing)[:3]}"

    if expected_counts is not None:
        for split, stats in summary.items():
            if "kept_frames" in expected_counts[split]:
                stats["reference_kept_frames"] = int(expected_counts[split]["kept_frames"])
    with open(manifest_path, "w") as fh:
        json.dump({"vad": "rVADfast 0.0.5 via vad_port.rvad_silence, threshold=0.4, "
                          "25ms window/10ms shift, subframes=2, tail=silence",
                   "summary": summary}, fh, indent=2)
    if counts_report_path is not None:
        write_counts_report(summary, expected_counts, counts_report_path)
    elif expected_counts is not None:
        for split, stats in summary.items():
            expected = {k: int(v) for k, v in expected_counts[split].items()}
            observed = {k: stats[k] for k in expected}
            if observed != expected:
                raise ValueError(f"joint VAD {split} counts {observed} != expected {expected}; "
                                 f"see {manifest_path}")
    if any(s["short_or_empty"] for s in summary.values()):
        raise ValueError("joint VAD release block: fewer than two retained frames; "
                         f"see {manifest_path}")

    unit_data = np.load(os.path.join(units_store, "data.npy"), mmap_mode="r")
    uids = np.load(os.path.join(units_store, "uids.npy"), mmap_mode="r")
    offsets = np.load(os.path.join(units_store, "offsets.npy"), mmap_mode="r")
    lengths = np.load(os.path.join(units_store, "lengths.npy"), mmap_mode="r")
    assert uids.shape == offsets.shape == lengths.shape
    for split, files in feature_hdfs.items():
        for shard, source in enumerate(files):
            with h5py.File(source, "r") as fh:
                tags = fh["seqTags"][:]
                frame_lengths = fh["seqLengths"][:, 0]
                inputs = fh["inputs"]
                writers = [
                    SimpleHDFWriter(out_feature_hdfs[split][shard], dim=1024, ndim=2),
                    SimpleHDFWriter(out_units_hdfs[split][shard], dim=500, ndim=1),
                    SimpleHDFWriter(out_raw_index_hdfs[split][shard], dim=1, ndim=1),
                    SimpleHDFWriter(out_orig_length_hdfs[split][shard], dim=1, ndim=1),
                ]
                pos = 0
                try:
                    for raw_tag, raw_length in zip(tags, frame_lengths):
                        tag = raw_tag.decode("ascii") if isinstance(raw_tag, bytes) else str(raw_tag)
                        original = int(raw_length)
                        indices = masks[tag]
                        i = int(np.searchsorted(uids, tag.encode("ascii")))
                        assert i < len(uids) and uids[i] == tag.encode("ascii"), tag
                        assert int(lengths[i]) == original, (tag, int(lengths[i]), original)
                        feat = np.asarray(inputs[pos:pos + original])[indices]
                        units = np.asarray(unit_data[int(offsets[i]):int(offsets[i]) + original])[indices]
                        assert feat.shape == (len(indices), 1024) and len(units) == len(indices), tag
                        assert np.all((units >= 0) & (units < 500)), tag
                        writers[0].insert_batch(feat[None], [len(indices)], [tag])
                        writers[1].insert_batch(units.astype(np.int32)[None], [len(indices)], [tag])
                        writers[2].insert_batch(indices[None], [len(indices)], [tag])
                        writers[3].insert_batch(np.array([[original]], dtype=np.int32), [1], [tag])
                        pos += original
                    assert pos == inputs.shape[0], (source, pos, inputs.shape[0])
                finally:
                    for writer in writers:
                        writer.close()


class BlankfreeVadHdfJob(Job):
    """CPU Sisyphus job for joint rVAD masking of frozen blank-free inputs.

    :param ogg_zips: ``{split: [LibriSpeech ogg zip, ...]}`` with the splits of ``feature_hdfs``
        (:func:`.librispeech.get_ogg_zip`); the source took ``hf_dataset`` (the HF Ogg dataset dict).
    :param feature_hdfs: ``{split: [feature HDF per shard]}``, tags = bare utterance ids.  Every audio
        entry of a split's zips needs a feature sequence of the same split and vice versa.
    :param units_store: the packed unit store (``w2v2.units.PackUnitsJob.out_store``).
    :param expected_counts: optional ``{split: {key: count}}`` with keys among ``"utterances"``,
        ``"original_frames"``, ``"kept_frames"``; a mismatch of a given key raises after the manifest
        is written.
    :param counts_report_only: with ``expected_counts``, do NOT raise on a count mismatch; write the
        observed totals and their relative differences from ``expected_counts`` to
        ``out_counts_report`` (``counts_vs_expected.json``) instead.  For audio that is knowingly not
        the banked audio (``FFMPEG_PIN_ACCEPT``).  ``False`` (the default) is hash-excluded, so the
        strict job keeps its hash; ``out_counts_report`` is then ``None``.
    """

    __sis_hash_exclude__ = {"counts_report_only": False}

    def __init__(
        self,
        *,
        ogg_zips: Dict[str, Sequence[tk.Path]],
        feature_hdfs: Dict[str, Sequence[tk.Path]],
        units_store: tk.Path,
        expected_counts: Optional[Dict[str, Dict[str, int]]] = None,
        counts_report_only: bool = False,
    ):
        super().__init__()
        self.feature_hdfs = {s: list(feature_hdfs[s]) for s in sorted(feature_hdfs)}
        if set(ogg_zips) != set(self.feature_hdfs):
            raise ValueError(f"ogg_zips splits {sorted(ogg_zips)} != feature_hdfs splits {sorted(self.feature_hdfs)}")
        self.ogg_zips = {s: list(ogg_zips[s]) for s in sorted(ogg_zips)}
        self.units_store = units_store
        if expected_counts is not None:
            if set(expected_counts) != set(self.feature_hdfs):
                raise ValueError(f"expected_counts splits {sorted(expected_counts)} != feature_hdfs "
                                 f"splits {sorted(self.feature_hdfs)}")
            for s in expected_counts:
                unknown = set(expected_counts[s]) - set(_COUNT_KEYS)
                if unknown:
                    raise ValueError(f"expected_counts[{s!r}] has unknown keys {sorted(unknown)}")
            expected_counts = {s: {k: int(expected_counts[s][k]) for k in _COUNT_KEYS if k in expected_counts[s]}
                               for s in sorted(expected_counts)}
        self.expected_counts = expected_counts
        if counts_report_only and expected_counts is None:
            raise ValueError("counts_report_only needs expected_counts")
        self.counts_report_only = bool(counts_report_only)
        self.out_feature_hdfs = {
            s: [self.output_path(f"feats.{s}.shard{k}.hdf") for k in range(len(paths))]
            for s, paths in self.feature_hdfs.items()
        }
        self.out_units_hdfs = {
            s: [self.output_path(f"units.{s}.shard{k}.hdf") for k in range(len(paths))]
            for s, paths in self.feature_hdfs.items()
        }
        self.out_raw_index_hdfs = {
            s: [self.output_path(f"raw_index.{s}.shard{k}.hdf") for k in range(len(paths))]
            for s, paths in self.feature_hdfs.items()
        }
        self.out_orig_length_hdfs = {
            s: [self.output_path(f"orig_length.{s}.shard{k}.hdf") for k in range(len(paths))]
            for s, paths in self.feature_hdfs.items()
        }
        self.out_manifest = self.output_path("manifest.json")
        self.out_stats = self.output_path("summary.txt")
        self.out_counts_report = self.output_path("counts_vs_expected.json") if self.counts_report_only else None

    def tasks(self):
        yield Task("run", rqmt={"cpu": 4, "mem": 96, "time": 8})

    def run(self):
        prepare_blankfree_data(
            ogg_zips={s: [p.get_path() for p in paths] for s, paths in self.ogg_zips.items()},
            feature_hdfs={s: [p.get_path() for p in paths] for s, paths in self.feature_hdfs.items()},
            units_store=self.units_store.get_path(),
            out_feature_hdfs={s: [p.get_path() for p in paths] for s, paths in self.out_feature_hdfs.items()},
            out_units_hdfs={s: [p.get_path() for p in paths] for s, paths in self.out_units_hdfs.items()},
            out_raw_index_hdfs={s: [p.get_path() for p in paths] for s, paths in self.out_raw_index_hdfs.items()},
            out_orig_length_hdfs={s: [p.get_path() for p in paths] for s, paths in self.out_orig_length_hdfs.items()},
            manifest_path=self.out_manifest.get_path(),
            expected_counts=self.expected_counts,
            counts_report_path=self.out_counts_report.get_path() if self.counts_report_only else None,
        )
        summary = json.load(open(self.out_manifest.get_path()))["summary"]
        with open(self.out_stats.get_path(), "w") as fh:
            for split, stats in summary.items():
                fh.write(f"{split}: {stats['utterances']} IDs, {stats['original_frames']} raw frames, "
                         f"{stats['kept_frames']} kept frames, "
                         f"{len(stats['length_mismatches'])} reference count mismatches, "
                         f"{len(stats['short_or_empty'])} S<2 items\n")
