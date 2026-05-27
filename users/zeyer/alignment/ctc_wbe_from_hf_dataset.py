"""
Word-boundary error (WBE / TSE) for CTC forced alignments.

Generic across CTC models and HF datasets
with word-level reference boundaries (TIMIT, Buckeye, ...).
Same TSE formula as the grad-align job at
:class:`exp2025_05_05_align.CalcAlignmentMetricsJob`:
average over words of ``0.5 * (|delta_start| + |delta_end|)``,
then average over utterances.
But driven by a hard frame-label CTC alignment HDF
rather than a soft gradient-score matrix.

Input HDF is whatever
:func:`i6_experiments.users.zeyer.experiments.exp2024_09_16_grad_align.ctc_forced_align`
(or any equivalent ``forward_to_hdf`` step
that emits one CTC-with-blank label per encoder frame) writes:
a sparse-int sequence per seq under the HDF's default ``data`` key.

The conversion CTC-frame-label-sequence -> word boundary times is:

  1. Collapse consecutive frames carrying the same non-blank label
     (and drop the intermediate blanks):
     yields a list of ``(label_idx, start_frame, end_frame_excl)`` per emitted token.
  2. Group consecutive tokens by the SPM word-boundary marker (``▁``):
     a piece starting with ``▁`` begins a new word,
     pieces without it extend the current word.
  3. Map frame to time via ``frame_sec = audio_len_secs / num_frames``.
     The encoder downsampling factor cancels
     because both numerator and denominator are read from the same forward pass.

Reference offsets in ``word_detail.{start, stop}``
are interpreted via the caller-supplied ``dataset_offset_factor``:
TIMIT stores sample indices (factor 1),
Buckeye stores milliseconds (factor 1000).
See :data:`i6_experiments.users.zeyer.datasets.hf_timit_buckeye.DATASET_OFFSET_FACTORS`.
"""

from __future__ import annotations

from typing import List, Optional

from sisyphus import Job, Task, tk


class CalcCtcWbeFromHfDatasetJob(Job):
    """
    Compute mean Word-Boundary-Error (WBE / TSE)
    for a CTC forced-alignment HDF
    against word-level reference boundaries
    in an HF dataset (``word_detail.{start, stop, utterance}``).

    Outputs:
      - ``out_wbe``: mean word-boundary error in seconds (single float).
      - ``out_report``: per-utt + summary text report (human-readable).
    """

    __sis_hash_exclude__ = {"returnn_root": None}
    __sis_version__ = 2

    def __init__(
        self,
        *,
        alignment_hdf: tk.Path,
        spm_model_file: tk.Path,
        blank_idx: int,
        dataset_dir: tk.Path,
        dataset_key: str,
        dataset_offset_factor: int,
        returnn_root: Optional[tk.Path] = None,
    ):
        """
        :param alignment_hdf: per-seq frame-label HDF from ``ctc_forced_align``.
        :param spm_model_file: SentencePiece model used by the CTC model.
        :param blank_idx: CTC blank label index.
            Typically equals the SPM vocab size
            (the ``wb_target_dim`` is ``vocab_size + 1``).
        :param dataset_dir: HF dataset dir (output of TIMIT/Buckeye prep job).
        :param dataset_key: HF split key, e.g. ``"val"`` / ``"test"``.
        :param dataset_offset_factor: multiplier
            to convert ``word_detail`` offsets to "samples at sampling_rate".
            TIMIT = 1, Buckeye = 1000.
        :param returnn_root: optional, falls back to env default via i6_core.
        """
        super().__init__()
        self.alignment_hdf = alignment_hdf
        self.spm_model_file = spm_model_file
        self.blank_idx = int(blank_idx)
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.dataset_offset_factor = int(dataset_offset_factor)
        self.returnn_root = returnn_root

        self.out_wbe = self.output_var("wbe.txt")
        self.out_report = self.output_path("report.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 10, "time": 1})

    def run(self):
        import os
        import sys
        import numpy as np

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        from returnn.datasets.hdf import HDFDataset
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor()
        sp.load(self.spm_model_file.get_path())
        # Pre-compute the per-id piece string so we don't pay the C++ call
        # per frame inside the hot loop.
        piece_for_idx: List[str] = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]

        align_hdf_ds = HDFDataset([self.alignment_hdf.get_path()])
        align_hdf_ds.initialize()
        align_hdf_ds.init_seq_order(epoch=1)

        # HF dataset_dir from TransformAndMapHuggingFaceDatasetJob: it's a
        # local DatasetDict on disk (save_to_disk format). load_from_disk is
        # the right loader -- load_dataset infers it but is more verbose.
        from datasets import load_from_disk

        ds = load_from_disk(self.dataset_dir.get_path())
        if hasattr(ds, "keys"):
            assert self.dataset_key in ds, f"split {self.dataset_key!r} not in {sorted(ds.keys())}"
            ds = ds[self.dataset_key]
        num_seqs = len(ds)

        # The forced-align HDF is written in whatever ``seq_ordering`` the forward job used
        # (typically ``sorted_reverse`` -- duration desc), while the HF dataset is in file order.
        # Match by seq-tag (the HF ``id`` column) rather than positional index.
        # The seq-tag stored in the HDF was set by ``HuggingFaceDataset`` via ``seq_tag_column="id"``
        # (see :func:`i6_experiments.users.zeyer.datasets.hf_timit_buckeye.get_hf_word_align_dataset_config`),
        # so HDF tag == HF ``id`` -- exact-string match works without normalization.
        hf_id_col = ds["id"]
        hf_id_to_idx = {str(t): i for i, t in enumerate(hf_id_col)}

        num_hdf_seqs = int(align_hdf_ds.num_seqs)

        report_lines: List[str] = []
        report_lines.append(
            f"CalcCtcWbeFromHfDatasetJob: {num_hdf_seqs} HDF seqs, {num_seqs} HF seqs, "
            f"dataset_key={self.dataset_key!r}, "
            f"dataset_offset_factor={self.dataset_offset_factor}, "
            f"blank_idx={self.blank_idx}"
        )

        wbe_utts: List[float] = []
        skipped: List[int] = []

        for hdf_seq_idx in range(num_hdf_seqs):
            align_hdf_ds.load_seqs(hdf_seq_idx, hdf_seq_idx + 1)
            hdf_tag = align_hdf_ds.get_tag(hdf_seq_idx)
            hf_idx = hf_id_to_idx.get(str(hdf_tag))
            if hf_idx is None:
                report_lines.append(f"** hdf seq {hdf_seq_idx} tag {hdf_tag!r} has no matching HF row id; skipping")
                skipped.append(hdf_seq_idx)
                continue
            # ``seq_idx`` is preserved as the report identifier --
            # use the HF row index so the report stays sortable against the HF dataset.
            seq_idx = hf_idx
            data = ds[hf_idx]
            audio = data["audio"]["array"]
            samplerate = int(data["audio"]["sampling_rate"])
            num_audio_samples = len(audio)
            audio_len_secs = float(num_audio_samples) / float(samplerate)

            frame_labels = align_hdf_ds.get_data(hdf_seq_idx, "data")
            # HDFDataset returns shape (T,) or (T, 1) depending on sparse settings; flatten.
            frame_labels = np.asarray(frame_labels).reshape(-1)
            num_frames = int(frame_labels.shape[0])
            if num_frames == 0:
                report_lines.append(f"** seq {seq_idx}: empty alignment, skipping")
                skipped.append(seq_idx)
                continue
            # Fail loud on ``<unk>`` in the alignment:
            # any unk-id frame means the forced-align target itself tokenized to ``<unk>``,
            # i.e. the dataset prep's ``text_case`` does not match the SPM vocab.
            # The resulting WBE numbers would be meaningless,
            # so refuse to compute them rather than silently produce garbage.
            unk_id = sp.unk_id()
            if unk_id >= 0 and unk_id != self.blank_idx and bool((frame_labels == unk_id).any()):
                raise RuntimeError(
                    f"seq {seq_idx}: alignment contains <unk> (id={unk_id}) frame labels -- "
                    f"the forced-align target was tokenized to <unk>, "
                    f"which means the dataset prep's text_case does not match the SPM vocab. "
                    f"Re-run dataset prep with the right text_case for this SPM model."
                )
            frame_sec = audio_len_secs / float(num_frames)

            # 1. Collapse CTC: emit (label, start_frame, end_frame_excl) for each
            # token run. Blanks separate runs; consecutive identical non-blank
            # frames are one emission.
            tokens: List[tuple] = []  # (label_idx, start_frame, end_frame_excl)
            current_label = self.blank_idx
            current_start = 0
            for f, label in enumerate(frame_labels):
                label_i = int(label)
                if label_i == current_label:
                    continue
                # flush previous run (if it was a real label)
                if current_label != self.blank_idx:
                    tokens.append((current_label, current_start, f))
                current_label = label_i
                current_start = f
            if current_label != self.blank_idx:
                tokens.append((current_label, current_start, num_frames))

            # 2. Group SPM pieces into words via the ▁ word-boundary marker.
            words_aligned: List[tuple] = []  # (word_str, start_frame, end_frame_excl)
            cur_word_pieces: List[str] = []
            cur_word_start: Optional[int] = None
            cur_word_end_excl: Optional[int] = None
            for label_idx, sf, ef in tokens:
                if label_idx < 0 or label_idx >= len(piece_for_idx):
                    # Defensive: out-of-range token. Shouldn't happen with a clean
                    # forced-align but if it does we drop it and flag in the report.
                    report_lines.append(f"** seq {seq_idx}: label {label_idx} out of vocab, dropping")
                    continue
                piece = piece_for_idx[label_idx]
                # SPM word-start marker. Use the literal Unicode U+2581 lower-one-eighth-block.
                is_word_start = piece.startswith("▁")
                if is_word_start and cur_word_pieces:
                    # Flush previous word.
                    words_aligned.append(("".join(cur_word_pieces).lstrip("▁"), cur_word_start, cur_word_end_excl))
                    cur_word_pieces = []
                    cur_word_start = None
                if is_word_start or cur_word_start is None:
                    cur_word_start = sf
                cur_word_pieces.append(piece)
                cur_word_end_excl = ef
            if cur_word_pieces:
                words_aligned.append(("".join(cur_word_pieces).lstrip("▁"), cur_word_start, cur_word_end_excl))

            # 3. Reference word boundaries.
            ref_words: List[str] = data["word_detail"]["utterance"]
            ref_starts = data["word_detail"]["start"]
            ref_stops = data["word_detail"]["stop"]
            assert len(ref_words) == len(ref_starts) == len(ref_stops), (
                f"seq {seq_idx}: word_detail length mismatch ({len(ref_words)} / {len(ref_starts)} / {len(ref_stops)})"
            )

            if len(words_aligned) != len(ref_words):
                report_lines.append(
                    f"** seq {seq_idx}: aligned {len(words_aligned)} words vs ref {len(ref_words)} "
                    f"({' '.join(ref_words)!r} vs {' '.join(w for w, _, _ in words_aligned)!r}); skipping"
                )
                skipped.append(seq_idx)
                continue

            # 4. Convert frames to seconds; convert ref offsets to seconds via
            # offset * factor / sample_rate (TIMIT factor=1, Buckeye factor=1000).
            aligned_secs = [(sf * frame_sec, ef * frame_sec) for _, sf, ef in words_aligned]
            ref_secs = [
                (
                    float(s) * self.dataset_offset_factor / float(samplerate),
                    float(e) * self.dataset_offset_factor / float(samplerate),
                )
                for s, e in zip(ref_starts, ref_stops)
            ]

            # 5. Per-utt WBE: mean over words of 0.5 * (|d_start| + |d_end|).
            wbe_utt = float(
                np.mean(
                    [
                        0.5 * (abs(ref_secs[w][0] - aligned_secs[w][0]) + abs(ref_secs[w][1] - aligned_secs[w][1]))
                        for w in range(len(ref_words))
                    ]
                )
            )
            wbe_utts.append(wbe_utt)
            report_lines.append(
                f"** seq {seq_idx}: {len(ref_words)} words, audio={audio_len_secs:.3f}s, "
                f"num_frames={num_frames}, frame_sec={frame_sec * 1000:.2f}ms, WBE={wbe_utt * 1000:.2f}ms"
            )

        wbe = float(np.mean(wbe_utts)) if wbe_utts else float("nan")
        report_lines.append("")
        report_lines.append(f"Mean WBE: {wbe * 1000:.2f}ms ({wbe:.6f}s)")
        report_lines.append(f"Aggregated over {len(wbe_utts)}/{num_hdf_seqs} HDF seqs ({len(skipped)} skipped)")

        report_text = "\n".join(report_lines) + "\n"
        with open(self.out_report.get_path(), "w") as f:
            f.write(report_text)
        self.out_wbe.set(wbe)
