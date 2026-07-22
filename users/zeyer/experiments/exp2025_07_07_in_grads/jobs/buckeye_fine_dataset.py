"""Convert the alexwengg/buckeye fine-resolution forced-align benchmark to our pipeline schema.

The raw repo is manifest.json + audio/*.wav with fine float-ms word timestamps,
hand-corrected, pre-segmented into 2-25 s utterances.
This job rewrites it as a parquet HF dataset with the schema the rest of the pipeline reads
(``audio{array, sampling_rate}`` + ``word_detail{utterance, start, stop}``),
laid out as a hub-cache snapshot so ``load_dataset`` / ``get_content_dir_from_hub_cache_dir`` consume it unchanged.

start/stop are stored as SAMPLE indices (``round(start_ms * sr / 1000)``),
so ``dataset_offset_factors=1`` like TIMIT -- 62.5 us resolution, not the 62.5 ms of nh0znoisung.
Non-word markers (``<SIL>``/``<IVER>``/``<VOCNOISE>``/``{...}``) are dropped.
Audio is stored as a plain float array (no datasets.Audio() feature) to avoid any codec dependency.

Segmentation is controlled by orthogonal params (each opt-in; all default to the legacy behavior):
  - ``resegment_gap_s``: split each segment at inter-word silences longer than this (None = no split).
  - ``split_up_to_max_seq_len_s``: recursively split any piece longer than this at its largest internal gap,
    until every piece fits (None = no length splitting).
  - ``drop_above_max_seq_len_s``: drop any final piece still longer than this (None = keep all).
  - ``max_duration_s``: legacy PRE-filter on the original manifest segment duration (kept for back-compat).
  - ``subsample_target_h`` (+ ``subsample_seed``): speaker-stratified subsample to ~this many hours.

The Buckeye segmentation variants are just param combos, e.g.:
  - A (no >=1 s silence, all <=20 s): resegment_gap_s=1, split_up_to_max_seq_len_s=20
  - B (keeps >=1 s silences, all <=20 s): split_up_to_max_seq_len_s=20
  - C (less-peaky exact): resegment_gap_s=1, drop_above_max_seq_len_s=20
"""

from typing import Optional
from sisyphus import Job, Task, tk


class BuildBuckeyeFineDatasetJob(Job):
    """alexwengg/buckeye manifest+wav -> parquet dataset in our schema. See module docstring."""

    # v2: skip empty/corrupt WAVs (e.g. s1901b_020 = 0 samples) that crash conv feature extraction.
    __sis_version__ = 2

    __sis_hash_exclude__ = {
        "resegment_gap_s": None,
        "min_words": 1,
        "split_up_to_max_seq_len_s": None,
        "drop_above_max_seq_len_s": None,
        "skip_misaligned_wavs": False,
        "subsample_target_h": None,
        "subsample_seed": 0,
    }

    def __init__(
        self,
        *,
        raw_dir: tk.Path,
        split_name: str = "test",
        num_shards: int = 8,
        max_segments: Optional[int] = None,
        max_duration_s: Optional[float] = None,
        resegment_gap_s: Optional[float] = None,
        split_up_to_max_seq_len_s: Optional[float] = None,
        drop_above_max_seq_len_s: Optional[float] = None,
        skip_misaligned_wavs: bool = False,
        min_words: int = 1,
        subsample_target_h: Optional[float] = None,
        subsample_seed: int = 0,
        returnn_root: Optional[tk.Path] = None,
    ):
        """
        :param raw_dir: hub-cache dir of the raw alexwengg/buckeye repo (manifest.json + audio/*.wav).
        :param split_name: name of the single output split (parquet files are ``<split_name>-*.parquet``).
        :param num_shards: number of parquet shards to write.
        :param max_segments: keep only the first N original manifest segments; None = all.
        :param max_duration_s: legacy PRE-filter on the raw INPUT segments, applied BEFORE any splitting:
            drop original manifest segments whose duration exceeds this; None = no pre-filter.
            The A/B/C variants leave this None and bound length via the POST seq-len params below instead.
        :param resegment_gap_s: split each segment at inter-word silences longer than this;
            None = no split.
        :param split_up_to_max_seq_len_s: POST, on the produced output sequences:
            recursively split any piece longer than this at its largest internal gap, until every piece fits;
            None = no length splitting.
        :param drop_above_max_seq_len_s: POST, on the produced output sequences:
            drop any piece still longer than this; None = keep all.
        :param skip_misaligned_wavs: drop whole source segments whose last word ends past the actual audio,
            i.e. the WAV is truncated relative to its annotation (e.g. s1901b_019: words run to 11.6 s,
            WAV is ~2 s). These produce short clips with phantom words and break strict-CTC forced-align.
        :param min_words: drop pieces with fewer than this many words.
        :param subsample_target_h: speaker-stratified subsample of the final pieces to ~this many hours,
            proportional per speaker, seeded; None = keep all.
        :param subsample_seed: RNG seed for the subsample shuffle.
        :param returnn_root: unused here; kept for call-site compatibility.
        """
        super().__init__()
        self.raw_dir = raw_dir
        self.split_name = split_name
        self.num_shards = num_shards
        self.max_segments = max_segments
        # Legacy PRE-filter on the ORIGINAL manifest segment duration (mostly capped ~25 s).
        self.max_duration_s = max_duration_s
        # Segmentation (all opt-in; see module docstring).
        self.resegment_gap_s = resegment_gap_s
        self.split_up_to_max_seq_len_s = split_up_to_max_seq_len_s
        self.drop_above_max_seq_len_s = drop_above_max_seq_len_s
        self.skip_misaligned_wavs = skip_misaligned_wavs
        self.min_words = min_words
        self.subsample_target_h = subsample_target_h
        self.subsample_seed = subsample_seed
        self.returnn_root = returnn_root
        self.rqmt = {"cpu": 4, "mem": 48, "time": 4}
        self.out_hub_cache_dir = self.output_path("hub_cache", directory=True)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def _spans(self, start, stop, n, sr):
        """Word-index spans (a, b) after the configured split / drop rules. start/stop are sample indices."""

        def seg_len(a, b):
            return stop[b - 1] - start[a]

        # 1) split at inter-word silences > resegment_gap_s (else one whole span).
        if self.resegment_gap_s is not None:
            thr = self.resegment_gap_s * sr
            spans, a = [], 0
            for w in range(1, n + 1):
                if w == n or (start[w] - stop[w - 1]) > thr:
                    spans.append((a, w))
                    a = w
        else:
            spans = [(0, n)]

        # 2) recursively split pieces longer than split_up_to_max_seq_len_s at their largest internal gap
        #    until every piece fits (the largest gap is always a real pause; nothing is dropped here).
        if self.split_up_to_max_seq_len_s is not None:
            maxd = self.split_up_to_max_seq_len_s * sr

            def rec(a, b, out):
                if seg_len(a, b) <= maxd:
                    out.append((a, b))
                    return
                best_w, best_g = None, -1
                for w in range(a + 1, b):
                    g = start[w] - stop[w - 1]
                    if g > best_g:
                        best_g, best_w = g, w
                if best_w is None:  # single word, no internal gap -> keep as-is
                    out.append((a, b))
                    return
                rec(a, best_w, out)
                rec(best_w, b, out)

            out = []
            for a, b in spans:
                rec(a, b, out)
            spans = out

        # 3) drop pieces still longer than drop_above_max_seq_len_s.
        if self.drop_above_max_seq_len_s is not None:
            dmax = self.drop_above_max_seq_len_s * sr
            spans = [(a, b) for a, b in spans if seg_len(a, b) <= dmax]

        return [(a, b) for a, b in spans if (b - a) >= self.min_words]

    def run(self):
        import os
        import json
        import random
        from collections import defaultdict

        from i6_experiments.users.zeyer.external_models.huggingface import (
            set_hf_offline_mode,
            get_content_dir_from_hub_cache_dir,
        )

        set_hf_offline_mode()

        import numpy as np
        import soundfile as sf
        import datasets

        content = get_content_dir_from_hub_cache_dir(self.raw_dir)
        with open(os.path.join(content, "manifest.json")) as f:
            samples = json.load(f)["samples"]
        if self.max_duration_s is not None:
            samples = [s for s in samples if s["duration_s"] <= self.max_duration_s]
        if self.max_segments is not None:
            samples = samples[: self.max_segments]
        print(f"{len(samples)} segments after manifest filters", flush=True)

        feats = datasets.Features(
            {
                "id": datasets.Value("string"),
                "speaker": datasets.Value("string"),
                "audio": {
                    "array": datasets.Sequence(datasets.Value("float32")),
                    "sampling_rate": datasets.Value("int64"),
                },
                "word_detail": {
                    "utterance": datasets.Sequence(datasets.Value("string")),
                    "start": datasets.Sequence(datasets.Value("int64")),
                    "stop": datasets.Sequence(datasets.Value("int64")),
                },
            }
        )

        # No splitting/dropping at all -> emit the whole segment with its native id (legacy "buckeye").
        whole = (
            self.resegment_gap_s is None
            and self.split_up_to_max_seq_len_s is None
            and self.drop_above_max_seq_len_s is None
        )

        recs = []  # (record_dict, speaker, dur_s)
        n_skipped = 0
        for s in samples:
            wav, sr = sf.read(os.path.join(content, s["audio"]))
            wav = np.asarray(wav, dtype=np.float32)
            if wav.ndim > 1:
                wav = wav[:, 0]
            # A few alexwengg WAVs are empty/corrupt (e.g. s1901b_020 = 0 samples) -- skip them;
            # an empty input crashes the wav2vec2 conv feature extractor (input size 0).
            if len(wav) < 1600:
                print(f"skip {s['id']}: {len(wav)} samples", flush=True)
                n_skipped += 1
                continue
            words = [w for w in s["words"] if not (w["word"].startswith("<") or w["word"].startswith("{"))]
            if not words:
                n_skipped += 1
                continue
            utt = [w["word"] for w in words]
            start = [int(round(w["start_ms"] * sr / 1000.0)) for w in words]
            stop = [int(round(w["end_ms"] * sr / 1000.0)) for w in words]
            if self.skip_misaligned_wavs and stop[-1] > len(wav) + int(0.25 * sr):
                # WAV is truncated relative to its annotation -> the words past len(wav) have no audio.
                print(f"skip {s['id']}: words to {stop[-1] / sr:.1f}s but wav {len(wav) / sr:.1f}s", flush=True)
                n_skipped += 1
                continue
            pad = int(0.15 * sr)
            for k, (a, b) in enumerate(self._spans(start, stop, len(utt), sr)):
                if whole:
                    rid, w0, w1 = s["id"], 0, len(wav)
                else:
                    rid = f"{s['id']}_{k}"
                    w0 = max(0, start[a] - pad)
                    w1 = min(len(wav), stop[b - 1] + pad)
                recs.append(
                    (
                        {
                            "id": rid,
                            "speaker": s.get("speaker", ""),
                            "audio": {"array": wav[w0:w1], "sampling_rate": int(sr)},
                            "word_detail": {
                                "utterance": utt[a:b],
                                "start": [start[w] - w0 for w in range(a, b)],
                                "stop": [stop[w] - w0 for w in range(a, b)],
                            },
                        },
                        s.get("speaker", ""),
                        (w1 - w0) / sr,
                    )
                )
        print(f"built {len(recs)} segs, {n_skipped} skipped (empty/no words)", flush=True)

        # Speaker-stratified subsample to ~subsample_target_h (seeded), proportional per speaker.
        if self.subsample_target_h is not None:
            total_h = sum(d for _, _, d in recs) / 3600.0
            if total_h > self.subsample_target_h:
                frac = self.subsample_target_h / total_h
                by_spk = defaultdict(list)
                for i, (_, spk, _) in enumerate(recs):
                    by_spk[spk].append(i)
                rng = random.Random(self.subsample_seed)
                keep = set()
                for spk, idxs in by_spk.items():
                    idxs = list(idxs)
                    rng.shuffle(idxs)
                    budget_s = sum(recs[i][2] for i in idxs) * frac
                    acc = 0.0
                    for i in idxs:
                        if acc >= budget_s:
                            break
                        keep.add(i)
                        acc += recs[i][2]
                recs = [recs[i] for i in sorted(keep)]
                print(f"subsampled to {len(recs)} segs / {sum(d for _, _, d in recs) / 3600:.2f} h", flush=True)

        # hub-cache snapshot layout: <hub>/datasets--local--buckeye_fine/{refs/main, snapshots/<ref>/data/*.parquet}
        ref = "buckeyefine0"
        root = os.path.join(self.out_hub_cache_dir.get_path(), "datasets--local--buckeye_fine")
        data_dir = os.path.join(root, "snapshots", ref, "data")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(root, "refs"), exist_ok=True)
        with open(os.path.join(root, "refs", "main"), "w") as f:
            f.write(ref)

        records = [r for r, _, _ in recs]
        n = len(records)
        shard_size = max(1, (n + self.num_shards - 1) // self.num_shards)
        n_words = 0
        for shard in range(self.num_shards):
            chunk = records[shard * shard_size : (shard + 1) * shard_size]
            n_words += sum(len(r["word_detail"]["utterance"]) for r in chunk)
            ds = datasets.Dataset.from_list(chunk, features=feats)
            out_pq = os.path.join(data_dir, f"{self.split_name}-{shard:05d}-of-{self.num_shards:05d}.parquet")
            ds.to_parquet(out_pq)
            print(f"shard {shard}: {len(chunk)} segs -> {os.path.basename(out_pq)}", flush=True)

        print(f"done: {n} segs, {n_words} words", flush=True)


class MapBuckeyeFineTimestampsToLongFormJob(Job):
    """Long-form Buckeye (nh0znoisung, unsegmented tracks, ~62.5 ms-quantized word timestamps)
    with word timestamps replaced by the fine float-ms ones from alexwengg/buckeye where possible.

    Both datasets derive from the same Buckeye hand-corrected ``.words`` annotation,
    and the alexwengg segment audio is sample-identical (up to a float gain ~1e-5)
    to slices of the nh0znoisung tracks
    (verified: cross-correlation peak 1.0000, residual ~1e-6 after gain fit;
    nh-vs-alexwengg word-start deltas are bounded by half the 62.5 ms grid).
    So each segment is located in its parent track by cross-correlation
    (track from the segment id prefix, e.g. "s0101a_003" -> track "s0101a"),
    the word sequences are aligned (difflib, lowercased; handles boundary words),
    and matched words get ``offset + fine_ms`` timestamps in samples.
    Unmatched words (segment gaps, rare edge diffs) keep the quantized nh timestamps.

    Audio and transcript are copied 1:1 from the nh0znoisung track (same seq order),
    so chunk-segmentation HDFs computed on the nh0znoisung dataset stay valid for this one;
    only timestamp-consuming metric jobs need to read this dataset instead
    (with ``dataset_offset_factors=1``: start/stop are sample indices here, not the 62.5 ms grid).

    Output adds a top-level ``word_fine`` int sequence (1 = fine timestamp, 0 = kept coarse),
    so metrics can optionally restrict to fine words.
    """

    __sis_version__ = 1

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        fine_raw_dir: tk.Path,
        min_corr_peak: float = 0.999,
        min_word_match_fraction: float = 0.8,
        num_shards: int = 8,
        returnn_root: Optional[tk.Path] = None,
    ):
        """
        :param dataset_dir: hub cache dir of the unsegmented dataset (nh0znoisung/buckeye).
        :param dataset_key: e.g. "val", "test".
        :param fine_raw_dir: hub cache dir of the raw alexwengg/buckeye download
            (manifest.json + audio/*.wav).
        :param min_corr_peak: assert each segment's normalized cross-correlation peak >= this.
        :param min_word_match_fraction: assert >= this fraction of each track's in-segment words
            get a fine timestamp.
        :param num_shards:
        :param returnn_root:
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.fine_raw_dir = fine_raw_dir
        self.min_corr_peak = min_corr_peak
        self.min_word_match_fraction = min_word_match_fraction
        self.num_shards = num_shards
        self.returnn_root = returnn_root

        self.rqmt = {"cpu": 4, "mem": 32, "time": 4}
        self.out_hub_cache_dir = self.output_path("hub_cache", directory=True)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys
        import json
        import difflib

        from i6_experiments.users.zeyer.external_models.huggingface import (
            set_hf_offline_mode,
            get_content_dir_from_hub_cache_dir,
        )

        set_hf_offline_mode()

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        import numpy as np
        from scipy import signal
        import soundfile as sf
        import datasets
        from datasets import load_dataset

        fine_content = get_content_dir_from_hub_cache_dir(self.fine_raw_dir)
        with open(os.path.join(fine_content, "manifest.json")) as f:
            manifest = json.load(f)["samples"]
        segs_by_track = {}
        for s in manifest:
            segs_by_track.setdefault(s["id"].rsplit("_", 1)[0], []).append(s)

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        ds_split = ds[self.dataset_key]
        print(f"Input: {len(ds_split)} tracks")

        feats = datasets.Features(
            {
                "id": datasets.Value("string"),
                "speaker": datasets.Value("string"),
                "audio": {
                    "array": datasets.Sequence(datasets.Value("float32")),
                    "sampling_rate": datasets.Value("int64"),
                },
                "word_detail": {
                    "utterance": datasets.Sequence(datasets.Value("string")),
                    "start": datasets.Sequence(datasets.Value("int64")),
                    "stop": datasets.Sequence(datasets.Value("int64")),
                },
                "word_fine": datasets.Sequence(datasets.Value("int8")),
            }
        )

        def is_marker(word: str) -> bool:
            return word.startswith("<") or word.startswith("{")

        records = []
        for track in ds_split:
            track_id = track["track_id"]
            sr = track["audio"]["sampling_rate"]
            audio = np.asarray(track["audio"]["array"], dtype=np.float32)
            nh_words = track["word_detail"]["utterance"]
            # nh raw unit * 1000 = samples (dataset_offset_factors=1000 convention).
            nh_starts = [int(t) * 1000 for t in track["word_detail"]["start"]]
            nh_stops = [int(t) * 1000 for t in track["word_detail"]["stop"]]

            # Absolute fine word list for this track, across all its segments.
            fine_words = []  # (word, start_samples, stop_samples)
            prev_offset = -1
            for s in segs_by_track.get(track_id, []):
                wav, sr2 = sf.read(os.path.join(fine_content, s["audio"]))
                wav = np.asarray(wav, dtype=np.float32)
                if wav.ndim > 1:
                    wav = wav[:, 0]
                # A few alexwengg WAVs are empty/corrupt (e.g. s1901b_020 = 0 samples) -- skip them;
                # an empty probe makes the cross-correlation output empty (argmax crash).
                if len(wav) < 1600:
                    print(f"skip {s['id']}: {len(wav)} samples", flush=True)
                    continue
                assert sr2 == sr, f"{s['id']}: sr {sr2} != track sr {sr}"
                probe = wav[: int(3.0 * sr)]
                corr = signal.fftconvolve(audio, probe[::-1], mode="valid")
                local_energy = np.sqrt(signal.fftconvolve(audio**2, np.ones(len(probe)), mode="valid"))
                scores = corr / (np.sqrt(np.sum(probe**2)) * np.maximum(local_energy, 1e-6))
                offset = int(np.argmax(scores))
                peak = float(scores[offset])
                assert peak >= self.min_corr_peak, f"{s['id']}: corr peak {peak} < {self.min_corr_peak}"
                assert offset > prev_offset, f"{s['id']}: offset {offset} not increasing (prev {prev_offset})"
                prev_offset = offset
                for w in s["words"]:
                    if is_marker(w["word"]):
                        continue
                    fine_words.append(
                        (
                            w["word"],
                            offset + int(round(w["start_ms"] * sr / 1000.0)),
                            offset + int(round(w["end_ms"] * sr / 1000.0)),
                        )
                    )

            # Align fine words to the nh transcript; matched words take the fine timestamps.
            starts, stops = list(nh_starts), list(nh_stops)
            fine_mask = [0] * len(nh_words)
            sm = difflib.SequenceMatcher(
                a=[w.lower() for w, _, _ in fine_words],
                b=[w.lower() for w in nh_words],
                autojunk=False,
            )
            n_matched = 0
            for op, a0, a1, b0, b1 in sm.get_opcodes():
                if op != "equal":
                    continue
                for k in range(a1 - a0):
                    _, w_start, w_stop = fine_words[a0 + k]
                    starts[b0 + k] = w_start
                    stops[b0 + k] = w_stop
                    fine_mask[b0 + k] = 1
                    n_matched += 1
            if fine_words:
                frac = n_matched / len(fine_words)
                assert frac >= self.min_word_match_fraction, f"{track_id}: only {frac:.2f} of fine words matched"
            print(
                f"track {track_id}: {len(segs_by_track.get(track_id, []))} segments,"
                f" {n_matched}/{len(nh_words)} words fine ({len(fine_words)} fine candidates)"
            )

            records.append(
                {
                    "id": track_id,
                    "speaker": track["speaker_id"],
                    "audio": {"array": audio, "sampling_rate": int(sr)},
                    "word_detail": {"utterance": nh_words, "start": starts, "stop": stops},
                    "word_fine": fine_mask,
                }
            )

        # hub-cache snapshot layout, same convention as BuildBuckeyeFineDatasetJob.
        ref = "buckeyelongfine0"
        root = os.path.join(self.out_hub_cache_dir.get_path(), "datasets--local--buckeye_longform_fine")
        data_dir = os.path.join(root, "snapshots", ref, "data")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(root, "refs"), exist_ok=True)
        with open(os.path.join(root, "refs", "main"), "w") as f:
            f.write(ref)

        n = len(records)
        shard_size = max(1, (n + self.num_shards - 1) // self.num_shards)
        for shard in range(self.num_shards):
            chunk = records[shard * shard_size : (shard + 1) * shard_size]
            if not chunk:
                continue
            ds_out = datasets.Dataset.from_list(chunk, features=feats)
            out_pq = os.path.join(data_dir, f"{self.dataset_key}-{shard:05d}-of-{self.num_shards:05d}.parquet")
            ds_out.to_parquet(out_pq)
            print(f"shard {shard}: {len(chunk)} tracks -> {os.path.basename(out_pq)}")

        print("done")
