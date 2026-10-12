import gzip
import json
import os
import pathlib
from typing import List, Tuple

import numpy
import returnn
import sentencepiece
from returnn.datasets.hdf import SimpleHDFWriter
from sisyphus import tk

from i6_experiments.users.sheremeta.alignment import path_to_word_spans
from i6_experiments.users.sheremeta.alignment.jobs import CtcAlignmentHdfToWordBoundariesJob, WordAlignmentOffsetJob

_RETURNN_ROOT = os.path.dirname(os.path.dirname(returnn.__file__))
_FRAME_RATE = 100.0 / 6.0
_WORDS = "THE QUICK BROWN FOX JUMPS OVER LAZY DOG MISSISSIPPI BOOKKEEPER COMMITTEE EXTRAORDINARY".split()


def test_word_spans_follow_the_ctc_collapse_and_the_pieces_that_start_a_word():
    blank = 9
    # labels 1 and 2 start a word, 3 and 4 continue one
    pieces = ["<unk>", "▁a", "▁b", "c", "d"]
    # the path opens inside a word, has a blank inside one, a label on both sides of a blank and two adjacent words
    path = [9, 3, 3, 9, 1, 1, 4, 9, 4, 2, 9, 2, 3, 9]

    assert path_to_word_spans(path, blank, pieces) == [(1, 3), (4, 9), (9, 10), (11, 13)]


def test_offsets_of_a_frame_path_come_out_of_the_boundaries_job_and_the_offset_job(tmp_path: pathlib.Path):
    corpus = tmp_path / "corpus.txt"
    rng = numpy.random.RandomState(0)
    corpus.write_text("".join(" ".join(rng.choice(_WORDS, 12)) + "\n" for _ in range(500)))
    sentencepiece.SentencePieceTrainer.Train(
        input=str(corpus),
        model_prefix=str(tmp_path / "spm"),
        vocab_size=60,
        model_type="unigram",
        character_coverage=1.0,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    spm_file = tmp_path / "spm.model"
    sp = sentencepiece.SentencePieceProcessor()
    sp.Load(str(spm_file))
    blank = sp.GetPieceSize()

    # words of several tokens, every token on two frames with a blank after it
    words = ["KEEPERS", "FOXY", "BROWNER"]
    first_frames = [2, 40, 70]
    path = numpy.full(120, blank, dtype="int32")
    spans: List[Tuple[float, float]] = []
    for word, first in zip(words, first_frames):
        tokens = sp.EncodeAsIds(word)
        assert len(tokens) > 1, (word, tokens)
        for k, token in enumerate(tokens):
            path[first + 3 * k : first + 3 * k + 2] = token
        spans.append((first / _FRAME_RATE, (first + 3 * len(tokens) - 1) / _FRAME_RATE))

    # the path as the align forward writes it, one sparse label per frame
    alignment = tmp_path / "alignment.hdf"
    writer = SimpleHDFWriter(str(alignment), dim=blank + 1, ndim=1)
    writer.insert_batch(path[None, :], seq_len={0: [len(path)]}, seq_tag=["seq-0"])
    writer.close()

    # a reference whose words start 100 ms and end 40 ms before the path's
    store = tmp_path / "reference.jsonl.gz"
    with gzip.open(store, "wt", encoding="utf-8") as f:
        record = {"seq_tag": "seq-0", "words": [[w, s - 0.1, e - 0.04] for w, (s, e) in zip(words, spans)]}
        f.write(json.dumps(record) + "\n")

    boundaries = CtcAlignmentHdfToWordBoundariesJob(
        alignment_hdf=tk.Path(str(alignment)),
        spm_file=tk.Path(str(spm_file)),
        blank=blank,
        frame_rate=_FRAME_RATE,
        returnn_root=tk.Path(_RETURNN_ROOT),
    )
    boundaries.out_hdf = tk.Path(str(tmp_path / "word_boundaries.hdf"))
    boundaries.run()

    offsets = WordAlignmentOffsetJob(
        word_boundaries_hdf=boundaries.out_hdf, reference_store=tk.Path(str(store)), returnn_root=tk.Path(_RETURNN_ROOT)
    )
    offsets.out_summary = tk.Path(str(tmp_path / "summary.json"))
    offsets.out_report = tk.Path(str(tmp_path / "report.txt"))
    offsets.out_skipped = tk.Path(str(tmp_path / "skipped.jsonl"))
    offsets.run()

    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["failures"]["total"] == 0 and summary["n_words"] == 3
    assert abs(summary["start"]["mean_ms"] - 100.0) < 1e-2, summary["start"]
    assert abs(summary["end"]["mean_ms"] - 40.0) < 1e-2, summary["end"]
