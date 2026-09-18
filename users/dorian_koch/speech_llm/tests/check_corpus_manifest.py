"""Guard: the corpus manifest reports what the mix actually contains (backlog F8).

The job exists because the epoch arithmetic was being redone by hand, and its first real run showed
why that is not safe: `a8_4gpu_v2` draws **0.83 epochs of Fisher and 3.06 of the QA corpus**, while
the A1b write-up recorded the arm as "exactly ONE epoch". Both halves of a weighted mix are covered
at different rates, and the rate that matters is the one for the corpus doing the damage.

Checks here are all synthetic and sub-second -- no real corpus is touched.
"""

import io
import os
import sys
import tempfile
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))

from i6_experiments.users.dorian_koch.speech_llm.corpus_manifest import (  # noqa: E402
    CorpusManifest,
    _durations,
    _label,
)

SR = 24000


def _wav(seconds: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(np.zeros(int(seconds * SR), dtype="<i2").tobytes())
    return buf.getvalue()


def _corpus(tmp: str, name: str, seconds, with_duration: bool) -> str:
    from datasets import Dataset

    cols = {
        "id": [str(i) for i in range(len(seconds))],
        "audio_assistant": [{"bytes": _wav(s), "path": f"{i}.wav"} for i, s in enumerate(seconds)],
    }
    if with_duration:
        cols["duration"] = [float(s) for s in seconds]
    path = os.path.join(tmp, "work", f"SomeJob.{name}", "output", "dataset")
    Dataset.from_dict(cols).save_to_disk(path)
    return path


def _table(path):
    from datasets import load_from_disk

    return load_from_disk(path).data.table


def check_duration_column_and_offset_fallback():
    secs = [3.5, 20.0, 41.25]
    with tempfile.TemporaryDirectory() as tmp:
        with_col = _durations(_table(_corpus(tmp, "withcol", secs, True)))
        without = _durations(_table(_corpus(tmp, "nocol", secs, False)))
        assert np.allclose(with_col, secs), with_col
        # The fallback reads arrow's binary offsets, never the audio. It must agree with the column
        # to well under a frame, or the two code paths would report different corpora.
        assert np.allclose(without, secs, atol=1e-3), (without, secs)
    print("[ok] durations from the duration column, and from arrow offsets, agree")


def check_epoch_arithmetic_is_per_corpus():
    """A weighted mix is covered at DIFFERENT rates per corpus -- the whole point of the job."""
    with tempfile.TemporaryDirectory() as tmp:
        big = _corpus(tmp, "big", [160.0] * 20, True)  # 0.889 h
        small = _corpus(tmp, "small", [20.0] * 20, True)  # 0.111 h
        P = lambda s: SimpleNamespace(get=lambda: s)  # noqa: E731
        out_j, out_t = os.path.join(tmp, "m.json"), os.path.join(tmp, "m.txt")
        job = SimpleNamespace(
            tag="t",
            note="",
            entries=[(P(big), 0.5, 60.0), (P(small), 0.5, None)],
            duration_sec=60,
            batch_sequences=2,
            max_steps=10,
            # B8 provenance: the fixture models the REAL job, so a field added to run() shows up
            # here as a missing attribute rather than being silently absent from the output.
            license=None,
            source_url=None,
            retrieved_at=None,
            out_json=P(out_j),
            out_txt=P(out_t),
        )
        CorpusManifest.run(job)
        import json

        m = json.load(open(out_j))
        drawn = 2 * 60 * 10 / 3600  # 0.333 h
        assert abs(m["run"]["total_audio_drawn_h"] - drawn) < 1e-9, m["run"]
        eps = m["run"]["epochs_per_corpus"]
        e_big, e_small = eps[big], eps[small]
        assert abs(e_big - (drawn * 0.5 / (20 * 160 / 3600))) < 1e-6, e_big
        assert abs(e_small - (drawn * 0.5 / (20 * 20 / 3600))) < 1e-6, e_small
        # Equal WEIGHTS, 8x the epochs on the smaller corpus. A single "epochs" number for the mix
        # would be a fiction, and reporting only the larger one is how "exactly ONE epoch" happened.
        assert abs(e_small / e_big - 8.0) < 1e-6, (e_small, e_big)
        # Windowing slack is stated, because a row no longer than the window is a fixed crop.
        c_big = next(c for c in m["corpora"] if c["path"] == big)
        assert c_big["window_slack_median_sec"] == 100.0, c_big
        assert c_big["rows_with_no_slack"] == 0, c_big
    print("[ok] epochs are per corpus (8x apart here at equal weights), and window slack is stated")


def check_no_slack_rows_are_counted():
    with tempfile.TemporaryDirectory() as tmp:
        path = _corpus(tmp, "tight", [60.0, 60.0, 90.0], True)
        P = lambda s: SimpleNamespace(get=lambda: s)  # noqa: E731
        out_j = os.path.join(tmp, "m.json")
        CorpusManifest.run(
            SimpleNamespace(
                tag="t",
                note="",
                entries=[(P(path), 1.0, 60.0)],
                duration_sec=60,
                batch_sequences=None,
                max_steps=None,
                license=None,
                source_url=None,
                retrieved_at=None,
                out_json=P(out_j),
                out_txt=P(os.path.join(tmp, "m.txt")),
            )
        )
        import json

        c = json.load(open(out_j))["corpora"][0]
        assert c["rows_with_no_slack"] == 2, c
    print("[ok] rows whose length equals the window (a fixed crop) are counted, not hidden")


def check_label_is_not_output():
    p = "/x/work/i6_experiments/u/d/speech_llm/fisher_prep/FisherToMoshiTrainData.98IFVP8879x4/output/dataset"
    assert _label(p) == "FisherToMoshiTrainData.98IFVP8879x4", _label(p)
    # Non-vacuous: the obvious implementations both collapse two different corpora to "output",
    # which is exactly what the first run of this job printed.
    assert os.path.basename(os.path.dirname(p)) == "output"
    assert _label("/a/b") == "a/b"
    print("[ok] a corpus is labelled by its job dir, not the 'output' every path ends in")


def check_provenance_is_not_hashed():
    """B8: recording WHERE a corpus came from must never re-run it.

    The backlog specified these fields as "defaulting to None so nothing re-hashes". That is only
    true while they stay None -- `__sis_hash_exclude__` drops an argument only while it EQUALS the
    listed default, so the first manifest that actually records a licence would be hashed like any
    other kwarg and re-run. Hence the unconditional pop in `CorpusManifest.hash`, and hence this
    check: the interesting case is a POPULATED field, not an absent one.
    """
    import sisyphus.job

    entries = [("/some/corpus/output", 1.0, None)]

    def build(**kw):
        # ⚠ Each construction needs a cleared `created_jobs`: JobSingleton returns the SAME object
        # for two constructions that hash alike, so without this the equality below would pass
        # because `a is b` while the populated provenance was silently discarded -- a vacuous pass
        # in exactly the place this guard watches. Same trap as check_rqmt_not_hashed.py.
        sisyphus.job.created_jobs.clear()
        return CorpusManifest(tag="t", entries=entries, duration_sec=60, **kw)

    bare = build()
    full = build(license="CC-BY-4.0", source_url="https://example.org/ds", retrieved_at="2026-09-18")
    assert bare._sis_hash() == full._sis_hash(), (
        "populating provenance moved the hash -- recording a licence would re-run every manifest"
    )

    # The fields must still REACH the job; a "fix" that routed them nowhere would pass the line
    # above trivially.
    assert full.license == "CC-BY-4.0"
    assert full.source_url == "https://example.org/ds"
    assert full.retrieved_at == "2026-09-18"

    # Non-vacuous: a real input change MUST still re-hash, or the pop is a blanket mute.
    other = build()
    other.duration_sec = None
    different = CorpusManifest(tag="t", entries=entries, duration_sec=120)
    assert bare._sis_hash() != different._sis_hash(), (
        "a differing duration_sec did not re-hash -- the hash is not discriminating at all"
    )
    print("[ok] provenance (license/source_url/retrieved_at) is recorded but never hashed")


if __name__ == "__main__":
    check_duration_column_and_offset_fallback()
    check_epoch_arithmetic_is_per_corpus()
    check_no_slack_rows_are_counted()
    check_label_is_not_output()
    check_provenance_is_not_hashed()
    print("ALL PASS")
