"""Hypothesis-mode alignment: align recognition hypotheses instead of the reference transcript.

Flow: :class:`..recog_from_model.RecogFromModelJob` produces a TextDict of hyps;
:class:`BuildDatasetWithHypTranscriptsJob` swaps them in as the dataset transcript
(so every forced-align / grad-align job runs unchanged);
:class:`CalcHypAlignMetricsJob` then scores the produced word boundaries against the
original reference dataset via identity-gated F1@collar and Levenshtein-matched WBE
(see :mod:`..align_metrics`; standard 1:1 WBE is undefined when hyp != ref).
"""

from typing import Optional, List, Tuple
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)


class BuildDatasetWithHypTranscriptsJob(Job):
    """Copy a HuggingFace audio dataset, replacing each seq's ``word_detail`` transcript
    with recognition hypothesis words (start/stop times set to 0 -- there is no reference
    timing for hypotheses; downstream alignment jobs must not use them, see
    ``with_ref_metrics=False`` on the align jobs).

    Audio is passed through encoded (no decode/re-encode). Output is a hub-cache
    snapshot layout like :class:`..buckeye_fine_dataset.BuildBuckeyeFineDatasetJob`,
    so it is a drop-in ``dataset_dir`` for all downstream jobs.
    """

    def __init__(
        self,
        *,
        dataset_dir: tk.Path,
        dataset_key: str,
        hyps_txt: tk.Path,
    ):
        """
        :param dataset_dir: source dataset (hub-cache layout).
        :param dataset_key: split to process; the output has the same split name.
        :param hyps_txt: TextDict (``{seq_tag: text}``, gzipped Python repr) from
            :class:`..recog_from_model.RecogFromModelJob`, seq tags ``seq-{idx}``.
            Pass the normalized variant (``text_dict_normalize_file``) so the hyp
            words match the reference transcript conventions.
        """
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.hyps_txt = hyps_txt

        self.rqmt = {"time": 4, "cpu": 4, "mem": 20}
        self.out_hub_cache_dir = self.output_path("hub_cache", directory=True)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import gzip
        import ast

        set_hf_offline_mode()
        import datasets

        with gzip.open(self.hyps_txt.get_path(), "rt") as f:
            hyps = ast.literal_eval(f.read())
        assert isinstance(hyps, dict) and hyps

        ds_all = datasets.load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        ds = ds_all[self.dataset_key]
        print(f"source: {len(ds)} seqs, replacing transcripts with {len(hyps)} hyps")
        assert len(ds) == len(hyps), f"{len(ds)=} vs {len(hyps)=}"
        # The audio column is a PLAIN struct ({array: List(float32), sampling_rate}),
        # not an Audio feature -- pass it through untouched with identical features,
        # so nothing ever (re-)encodes audio. Sharded build to bound memory
        # (Python float lists of the decoded audio are large).
        feats = ds.features.copy()

        ref = "hyp0"
        root = os.path.join(self.out_hub_cache_dir.get_path(), "datasets--local--hyp_transcripts")
        data_dir = os.path.join(root, "snapshots", ref, "data")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(os.path.join(root, "refs"), exist_ok=True)
        with open(os.path.join(root, "refs", "main"), "w") as f:
            f.write(ref)

        num_shards = 8
        n = len(ds)
        n_empty = 0
        shard_size = (n + num_shards - 1) // num_shards
        for shard in range(num_shards):
            records = []
            for idx in range(shard * shard_size, min((shard + 1) * shard_size, n)):
                data = ds[idx]
                words = hyps[f"seq-{idx}"].split()
                if not words:
                    # Empty recog hypothesis: keep the seq (downstream tags index by position)
                    # with a placeholder that never matches a reference word --
                    # the F1 metric then counts it honestly as an unmatched hyp word.
                    print(f"WARNING: empty hyp for seq-{idx}, using placeholder", flush=True)
                    n_empty += 1
                    words = ["qqq"]
                data["word_detail"] = {"utterance": words, "start": [0] * len(words), "stop": [0] * len(words)}
                records.append(data)
            datasets.Dataset.from_list(records, features=feats).to_parquet(
                os.path.join(data_dir, f"{self.dataset_key}-{shard:05d}-of-{num_shards:05d}.parquet")
            )
            print(f"shard {shard}: {len(records)} seqs", flush=True)
        print(f"done: {n} seqs ({n_empty} empty hyps -> placeholder)")


class CalcHypAlignMetricsJob(Job):
    """Score hypothesis-mode word boundaries against the reference dataset.

    Hyp words (from the hyp dataset) are Levenshtein-aligned to the reference words;
    identity matches yield boundary error pairs. Reports identity-gated
    precision/recall/F1 at tolerance collars and the matched-words mean WBE
    (see :func:`..align_metrics.hyp_aggregate_corpus`).
    """

    def __init__(
        self,
        *,
        word_boundaries_hdf: tk.Path,
        hyp_dataset_dir: tk.Path,
        ref_dataset_dir: tk.Path,
        dataset_key: str,
        dataset_offset_factors: int,
        returnn_root: Optional[tk.Path] = None,
    ):
        """
        :param word_boundaries_hdf: aligned hyp-word (start, end) in seconds,
            [num_hyp_words, 2] per seq (e.g. ``out_word_boundaries_hdf`` of
            :class:`..word_align_from_per_token_grads.WordAlignFromPerTokenGradsJob`,
            or ``out_hdf`` of the forced-align baseline jobs), produced on the
            HYP dataset. Sequences are matched by tag (``seq-{idx}``); seqs missing
            from the HDF (e.g. MFA coverage gaps) are skipped and reported.
        :param hyp_dataset_dir: the :class:`BuildDatasetWithHypTranscriptsJob` output
            the alignment ran on (provides the hyp words, 1:1 with the boundaries).
        :param ref_dataset_dir: the original dataset with reference words and times.
        :param dataset_offset_factors: as in the other metric jobs (samples factor).
        """
        super().__init__()
        self.word_boundaries_hdf = word_boundaries_hdf
        self.hyp_dataset_dir = hyp_dataset_dir
        self.ref_dataset_dir = ref_dataset_dir
        self.dataset_key = dataset_key
        self.dataset_offset_factors = dataset_offset_factors
        self.returnn_root = returnn_root

        self.out_metrics = self.output_var("metrics.txt")
        self.out_f1_100 = self.output_var("f1_100ms.txt")
        self.out_matched_wbe = self.output_var("matched_wbe.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 10, "time": 5})

    @staticmethod
    def _norm_word(w: str) -> str:
        w = w.lower()
        return w.strip("".join(c for c in w if not (c.isalnum() or c == "'"))) or w.lower()

    def run(self):
        import os
        import sys

        set_hf_offline_mode()

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        from returnn.datasets.hdf import HDFDataset
        from datasets import load_dataset
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.align_metrics import (
            levenshtein_word_matches,
            hyp_aggregate_corpus,
        )

        boundaries = HDFDataset([self.word_boundaries_hdf.get_path()])
        boundaries.initialize()
        boundaries.init_seq_order(epoch=1)
        boundaries.load_seqs(0, boundaries.num_seqs)
        tag_to_hdf_idx = {boundaries.get_tag(i): i for i in range(boundaries.num_seqs)}

        ref_ds = load_dataset(get_content_dir_from_hub_cache_dir(self.ref_dataset_dir))[self.dataset_key]
        hyp_ds = load_dataset(get_content_dir_from_hub_cache_dir(self.hyp_dataset_dir))[self.dataset_key]
        hyp_ds = hyp_ds.remove_columns([c for c in hyp_ds.column_names if c not in ("word_detail",)])
        assert len(ref_ds) == len(hyp_ds)

        utt_stats = []
        n_uncovered = 0
        for seq_idx, (ref_data, hyp_data) in enumerate(zip(ref_ds, hyp_ds)):
            hdf_idx = tag_to_hdf_idx.get(f"seq-{seq_idx}")
            if hdf_idx is None:  # e.g. MFA coverage gap
                n_uncovered += 1
                continue
            samplerate = ref_data["audio"]["sampling_rate"]

            hyp_words: List[str] = [self._norm_word(w) for w in hyp_data["word_detail"]["utterance"]]
            ref_words: List[str] = [self._norm_word(w) for w in ref_data["word_detail"]["utterance"]]

            hyp_se = boundaries.get_data(hdf_idx, "data")
            assert hyp_se.shape == (len(hyp_words), 2), f"{hyp_se.shape=} vs {len(hyp_words)=}"
            hyp_times = [(float(s), float(e)) for s, e in hyp_se]  # seconds
            time_scale = self.dataset_offset_factors / samplerate
            ref_times = [
                (s * time_scale, e * time_scale)
                for s, e in zip(ref_data["word_detail"]["start"], ref_data["word_detail"]["stop"])
            ]

            match_errs: List[Tuple[float, float]] = []
            for hi, ri in levenshtein_word_matches(hyp_words, ref_words):
                match_errs.append((abs(hyp_times[hi][0] - ref_times[ri][0]), abs(hyp_times[hi][1] - ref_times[ri][1])))
            utt_stats.append({"n_hyp": len(hyp_words), "n_ref": len(ref_words), "match_errs": match_errs})
            if seq_idx < 5:
                print(f"seq {seq_idx}: n_hyp={len(hyp_words)} n_ref={len(ref_words)} n_match={len(match_errs)}")

        metrics = hyp_aggregate_corpus(utt_stats)
        metrics["n_uncovered_seqs"] = float(n_uncovered)
        print("CORPUS METRICS:", metrics)
        self.out_metrics.set(metrics)
        self.out_f1_100.set(metrics["f1_100ms"])
        self.out_matched_wbe.set(metrics["matched_wbe"])
