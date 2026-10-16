import gzip
import json
import os
import pathlib
from typing import List, Tuple

import numpy
import returnn
import sentencepiece
from returnn.datasets.hdf import HDFDataset, SimpleHDFWriter
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


def test_word_spans_of_a_path_without_repeat_merging_count_every_frame_as_its_own_token():
    blank = 9
    pieces = ["<unk>", "▁a", "▁b", "c", "d"]
    # a word start piece on two adjacent frames, then a word start piece twice with no blank between
    path = [9, 1, 1, 9, 3, 2, 2]

    assert path_to_word_spans(path, blank, pieces, merge_repeats=False) == [(1, 2), (2, 5), (5, 6), (6, 7)]


def _train_spm(tmp_path: pathlib.Path) -> Tuple[pathlib.Path, sentencepiece.SentencePieceProcessor]:
    """
    :param tmp_path: directory for the corpus and the model
    :return: the model file of a small SentencePiece model over the test words and the model loaded
    """
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
    return spm_file, sp


def _write_path(hdf: pathlib.Path, path: numpy.ndarray, dim: int) -> None:
    """
    Writes one path the way the align and realign forwards write it, one sparse label per frame.

    :param hdf: the file
    :param path: the label of every frame
    :param dim: the label dim of the file
    """
    writer = SimpleHDFWriter(str(hdf), dim=dim, ndim=1)
    writer.insert_batch(path[None, :], seq_len={0: [len(path)]}, seq_tag=["seq-0"])
    writer.close()


def test_word_boundaries_of_a_decoder_path_start_their_clock_after_the_delay(tmp_path: pathlib.Path):
    spm_file, sp = _train_spm(tmp_path)
    pad = sp.PieceToId("<unk>")
    delay = 10

    # the tokens of every word on adjacent frames, the first word before the delay has passed
    words = ["KEEPERS", "FOXY"]
    first_frames = [4, 30]
    path = numpy.full(60, pad, dtype="int32")
    expected: List[Tuple[float, float]] = []
    for word, first in zip(words, first_frames):
        tokens = sp.EncodeAsIds(word)
        path[first : first + len(tokens)] = tokens
        expected.append(((first - delay) / _FRAME_RATE, (first + len(tokens) - delay) / _FRAME_RATE))
    _write_path(tmp_path / "path.hdf", path, sp.GetPieceSize())

    job = CtcAlignmentHdfToWordBoundariesJob(
        alignment_hdf=tk.Path(str(tmp_path / "path.hdf")),
        spm_file=tk.Path(str(spm_file)),
        blank=pad,
        frame_rate=_FRAME_RATE,
        returnn_root=tk.Path(_RETURNN_ROOT),
        merge_repeats=False,
        frame_offset=delay,
    )
    job.out_hdf = tk.Path(str(tmp_path / "word_boundaries.hdf"))
    job.run()

    boundaries = HDFDataset([str(tmp_path / "word_boundaries.hdf")])
    boundaries.initialize()
    boundaries.init_seq_order(epoch=1)
    boundaries.load_seqs(0, 1)
    numpy.testing.assert_allclose(boundaries.get_data(0, "data"), numpy.asarray(expected), atol=1e-6)


def test_offsets_of_a_frame_path_come_out_of_the_boundaries_job_and_the_offset_job(tmp_path: pathlib.Path):
    spm_file, sp = _train_spm(tmp_path)
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

    alignment = tmp_path / "alignment.hdf"
    _write_path(alignment, path, blank + 1)

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
