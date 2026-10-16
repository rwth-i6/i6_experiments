import gzip
import json
import os
import pathlib
from typing import Any, Dict, List, Tuple

import numpy
import returnn
from returnn.datasets.hdf import SimpleHDFWriter
from sisyphus import tk

from i6_experiments.users.sheremeta.alignment.jobs import WordAlignmentOffsetJob
from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.align_metrics import (
    aggregate_corpus,
    per_utt_boundary_errors,
)

_RETURNN_ROOT = os.path.dirname(os.path.dirname(returnn.__file__))


def _run(
    tmp_path: pathlib.Path,
    reference: Dict[str, List[List[Any]]],
    boundaries: Dict[str, List[Tuple[float, float]]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Runs the offset job on a reference store and a word boundary file written the way their producers write them.

    :param tmp_path: directory for the files
    :param reference: per sequence tag the (word, start, end) records of the reference, in seconds
    :param boundaries: per sequence tag the (start, end) seconds of every word of the model's alignment
    :return: the summary and the skipped records of the job
    """
    store = tmp_path / "reference.jsonl.gz"
    with gzip.open(store, "wt", encoding="utf-8") as f:
        for tag, words in reference.items():
            f.write(json.dumps({"seq_tag": tag, "words": words}) + "\n")

    hdf = tmp_path / "word_boundaries.hdf"
    writer = SimpleHDFWriter(str(hdf), dim=2, ndim=2)
    for tag, spans in boundaries.items():
        data = numpy.asarray(spans, dtype="float32").reshape(1, len(spans), 2)
        writer.insert_batch(data, seq_len=[len(spans)], seq_tag=[tag])
    writer.close()

    job = WordAlignmentOffsetJob(
        word_boundaries_hdf=tk.Path(str(hdf)), reference_store=tk.Path(str(store)), returnn_root=tk.Path(_RETURNN_ROOT)
    )
    job.out_summary = tk.Path(str(tmp_path / "summary.json"))
    job.out_report = tk.Path(str(tmp_path / "report.txt"))
    job.out_skipped = tk.Path(str(tmp_path / "skipped.jsonl"))
    job.run()

    summary = json.loads((tmp_path / "summary.json").read_text())
    skipped = [json.loads(line) for line in (tmp_path / "skipped.jsonl").read_text().splitlines()]
    return summary, skipped


def test_offsets_are_signed_model_minus_reference_and_match_the_gradient_alignment_metrics(tmp_path: pathlib.Path):
    rng = numpy.random.RandomState(0)
    reference: Dict[str, List[List[Any]]] = {}
    boundaries: Dict[str, List[Tuple[float, float]]] = {}
    for i in range(6):
        # words that do not overlap, every boundary of the model's alignment within 50 ms of its mean offset
        starts = numpy.arange(int(rng.randint(1, 9))) + rng.uniform(0.0, 0.2)
        ref = [(float(s), float(s + rng.uniform(0.2, 0.5))) for s in starts]
        reference[f"seq-{i}"] = [[f"w{k}", s, e] for k, (s, e) in enumerate(ref)]
        boundaries[f"seq-{i}"] = [
            (float(s + rng.uniform(0.07, 0.17)), float(e + rng.uniform(-0.02, 0.08))) for s, e in ref
        ]
    # a non-lexical entry of the reference has no word in the model's alignment
    reference["seq-0"].insert(1, ["<eps>", 0.0, 0.0])

    summary, skipped = _run(tmp_path, reference, boundaries)

    # the boundary file holds float32 seconds
    pairs: List[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]] = [
        (
            [(float(numpy.float32(s)), float(numpy.float32(e))) for s, e in boundaries[tag]],
            [(float(s), float(e)) for w, s, e in reference[tag] if w != "<eps>"],
        )
        for tag in reference
    ]
    expected = aggregate_corpus([per_utt_boundary_errors(pred, ref) for pred, ref in pairs], aggregation="micro")
    starts = numpy.array([1000.0 * (p[0] - r[0]) for pred, ref in pairs for p, r in zip(pred, ref)])
    ends = numpy.array([1000.0 * (p[1] - r[1]) for pred, ref in pairs for p, r in zip(pred, ref)])

    assert skipped == [] and summary["failures"]["total"] == 0
    assert summary["n_utts"] == 6 and summary["n_words"] == len(starts)
    assert summary["start"]["mean_ms"] > 60.0
    for name, values, mean_key in (
        ("start", starts, "start_signed_mean"),
        ("end", ends, "end_signed_mean"),
        ("center", 0.5 * (starts + ends), "center_offset"),
    ):
        got = summary[name]
        assert abs(got["mean_ms"] - 1000.0 * expected[mean_key]) < 1e-3, (name, got, expected[mean_key])
        assert abs(got["median_ms"] - numpy.median(values)) < 1e-3, (name, got)
        assert abs(got["p90_ms"] - numpy.percentile(values, 90)) < 1e-3, (name, got)
    assert abs(summary["wbe_ms"] - 1000.0 * expected["wbe"]) < 1e-3
    assert abs(summary["boundary_abs"]["within_50ms"] - expected["acc_50ms"]) < 1e-9


def test_sequences_that_cannot_be_paired_are_counted_and_listed(tmp_path: pathlib.Path):
    reference: Dict[str, List[List[Any]]] = {
        "seq-0": [["a", 0.5, 0.75], ["b", 1.0, 1.25]],
        "seq-1": [["c", 0.5, 0.75], ["d", 1.0, 1.25]],
        "seq-2": [["e", 0.5, 0.75]],
        "seq-3": [["f", 0.5, 0.75], ["g", 1.0, 1.25]],
    }
    boundaries: Dict[str, List[Tuple[float, float]]] = {
        "seq-0": [(0.75, 1.0), (1.25, 1.5)],
        "seq-1": [(0.5, 0.75)],
        "seq-3": [(1.5, 1.75), (1.0, 1.25)],
        "seq-9": [(0.0, 0.25)],
    }

    summary, skipped = _run(tmp_path, reference, boundaries)

    assert summary["attempted_utts"] == 4 and summary["n_utts"] == 1 and summary["n_words"] == 2
    assert summary["failures"]["count_mismatch"] == 1
    assert summary["failures"]["missing_output"] == 1
    assert summary["failures"]["nonmonotonic_output"] == 1
    assert summary["failure_rate"] == 0.75
    assert summary["unexpected_outputs"] == 1
    assert {(rec["seq_tag"], rec["reason"]) for rec in skipped} == {
        ("seq-1", "count_mismatch"),
        ("seq-2", "missing_output"),
        ("seq-3", "nonmonotonic_output"),
        ("seq-9", "unexpected_output"),
    }
    assert summary["start"] == {"n": 2, "mean_ms": 250.0, "median_ms": 250.0, "p90_ms": 250.0}


def test_words_the_model_times_before_zero_are_kept(tmp_path: pathlib.Path):
    reference: Dict[str, List[List[Any]]] = {"seq-0": [["a", 0.1, 0.3], ["b", 0.5, 0.75]]}
    boundaries: Dict[str, List[Tuple[float, float]]] = {"seq-0": [(-0.2, 0.0), (0.5, 0.75)]}

    summary, skipped = _run(tmp_path, reference, boundaries)

    assert skipped == [] and summary["n_words"] == 2
    assert abs(summary["start"]["mean_ms"] - (-150.0)) < 1e-3, summary["start"]
