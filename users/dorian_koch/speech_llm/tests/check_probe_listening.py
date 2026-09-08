"""Guard: the listening report must actually contain what it claims to.

A page-generating job fails in a way nothing else catches: it produces a well-formed, plausible,
EMPTY page and the manager reports success. That is the same class as the ``ResultNotify`` digest
recording ``{"is_dir": true}`` instead of a number, and the ``TrainMetricsPlot`` panel that stayed
blank -- both shipped, both looked fine. So each assertion below is about content, not exit status:
audio really embedded, transcripts really rendered, the endpoints really kept.

The specific hazards it pins:

* **Append order != step order.** A preempted run resumes and re-probes steps it already logged, so
  ``probe_transcripts.jsonl`` can hold the same step twice with different text. The later record --
  from the run that actually continued -- must win, and rows must come out step-ordered.
* **Subsampling must keep both endpoints.** The whole comparison is near-base versus collapsed; a
  stride that dropped the final step would remove the single most informative clip while still
  rendering a full-looking page.
* **A run with nothing recorded must SAY so.** Every arm before 2026-09-08 kept no replies. Rendering
  those as an empty page would read like a broken job and, worse, like an absence of findings.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_probe_listening.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))

import numpy as np  # noqa: E402
import sphn  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm.probe_listening import (  # noqa: E402
    ProbeListeningReport,
    _pick,
    _read_transcripts,
)

SR = 24000
N_Q = 2
#: Steps 50 and 100 are logged, then a preemption replays 100 with DIFFERENT text and continues.
STEPS = [50, 100, 150, 200]


def _row(q, step, correct, text):
    return {
        "question": f"Who wrote book {q}?",
        "answer": f"Author{q}",
        "category": "unknown",
        "binary_correct": correct,
        "quality_score": 5 if correct else 1,
        "monologue": text,
    }


def _make_run(tmp: Path, *, with_audio=True, replay=True) -> str:
    run = tmp / "run"
    (run / "probe_audio").mkdir(parents=True)
    recs = []
    for step in STEPS:
        # A fluent reply early, an empty one late: the forgetting-vs-incoherence shape.
        text = "The answer is Author{q}, a writer." if step < 150 else ""
        recs.append(
            {
                "step": step,
                "accuracy": 0.5 if step < 150 else 0.0,
                "coherence": {
                    "n": N_Q,
                    "words_mean": 6.0 if step < 150 else 0.0,
                    "empty_frac": 0.0 if step < 150 else 1.0,
                    "distinct_word_ratio": 1.0,
                    "looping_frac": 0.0,
                    "chars_per_word": 4.0,
                },
                "rows": [_row(q, step, 1 if step < 150 else 0, text.format(q=q)) for q in range(N_Q)],
            }
        )
    lines = list(recs)
    if replay:
        # The trap: step 100 appears EARLIER with stale text, and again later with the real text.
        stale = json.loads(json.dumps(recs[1]))
        stale["rows"][0]["monologue"] = "STALE-PRE-PREEMPTION"
        lines = [recs[0], stale, recs[1], recs[2], recs[3]]
    with open(run / "probe_transcripts.jsonl", "w") as f:
        for r in lines:
            f.write(json.dumps(r) + "\n")
        f.write("{partial-line-from-a-live-writer")  # torn tail: must not lose the file
    if with_audio:
        for step in STEPS:
            d = run / "probe_audio" / f"step_{step:06d}"
            d.mkdir()
            for q in range(N_Q):
                tone = (0.1 * np.sin(np.arange(SR) * (0.03 + 0.01 * q))).astype(np.float32)
                sphn.write_wav(str(d / f"{q:03d}.wav"), tone, SR)
    return str(run)


class _P:
    """Minimal stand-in for a tk.Path output slot."""

    def __init__(self, p):
        self._p = str(p)

    def get_path(self):
        return self._p


def _render(run_dir, tmp, **kw):
    # Sisyphus's Job.__new__ is the graph constructor and refuses a bare instantiation, so build
    # the object directly and drive the REAL run(). The point is to exercise the job's own code,
    # not a reimplementation of it.
    job = object.__new__(ProbeListeningReport)
    job.run_dirs = {"a17_demo": _P(run_dir)}
    job.questions = kw.get("questions", N_Q)
    job.max_steps_shown = kw.get("max_steps_shown", 12)
    job.clip_seconds = kw.get("clip_seconds", 2.0)
    job.title = "guard"
    job.out_html = _P(os.path.join(tmp, "index.html"))
    job.out_summary = _P(os.path.join(tmp, "coherence.json"))
    job.run()
    with open(job.out_html.get_path()) as f:
        return f.read(), json.load(open(job.out_summary.get_path()))


def check_replayed_step_is_deduped_to_the_later_record():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        run = _make_run(tmp)
        recs = _read_transcripts(run)
        assert [r["step"] for r in recs] == STEPS, [r["step"] for r in recs]
        mono = recs[1]["rows"][0]["monologue"]
        assert mono != "STALE-PRE-PREEMPTION", (
            "the pre-preemption record won -- a resumed run would be reported with the text it "
            "produced before it rolled back"
        )
        assert "Author0" in mono, mono
    print("PASS  a replayed probe step resolves to the later record, in step order")


def check_page_embeds_audio_and_transcripts():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        run = _make_run(tmp)
        html, coh = _render(run, d)
        n_audio = html.count("data:audio/ogg;base64,")
        assert n_audio == len(STEPS) * N_Q, f"{n_audio} clips embedded, expected {len(STEPS) * N_Q}"
        # Non-trivial payload: an empty base64 would still match the substring above.
        assert 'base64,"' not in html, "an empty audio payload was embedded"
        assert "Who wrote book 0?" in html and "Author0" in html, "the question/gold went missing"
        assert "The answer is Author0" in html, "the model's reply text is not on the page"
        assert "STALE-PRE-PREEMPTION" not in html, "the rolled-back reply reached the page"
        # The coherence table is the numeric answer and must carry every step.
        for step in STEPS:
            assert f"<td>{step}</td>" in html, f"step {step} missing from the coherence table"
        assert set(coh["a17_demo"]) == {str(s) for s in STEPS}, coh
    print("PASS  the page embeds real audio, the replies, and a full coherence table")


def check_subsampling_keeps_both_endpoints():
    assert _pick([1, 2, 3], 5) == [1, 2, 3]
    assert _pick([], 3) == []
    for k in (2, 3, 4):
        got = _pick(STEPS, k)
        assert got[0] == STEPS[0] and got[-1] == STEPS[-1], (k, got)
        assert got == sorted(set(got)), got
    assert _pick(list(range(100)), 1) == [99], "with one slot, show the END of the run"
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        run = _make_run(tmp)
        html, _ = _render(run, d, max_steps_shown=2)
        assert "step 50<" in html.replace("</span>", "<") or "step 50" in html, html[:200]
        assert "step 200" in html, "the FINAL step was dropped by subsampling"
        assert html.count("data:audio/ogg;base64,") == 2 * N_Q, "wrong number of steps rendered"
    print("PASS  subsampling always keeps the first and last step")


def check_a_run_with_nothing_says_so():
    """The pre-2026-09-08 arms. An empty page must not pass for 'no effect found'."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        empty = tmp / "empty"
        empty.mkdir()
        html, coh = _render(str(empty), d)
        assert "recorded no probe transcripts" in html, html
        assert "audio_n" in html, "the page must say how to capture it next time"
        assert coh["a17_demo"] == {}, coh
        # transcripts but no audio: the text half must still render, with an explicit note.
        run = _make_run(tmp, with_audio=False)
        html2, _ = _render(run, d)
        assert "data:audio/ogg;base64," not in html2, "audio appeared from nowhere"
        assert "No probe audio was written" in html2, html2[-2000:]
        assert "The answer is Author0" in html2, "the transcripts must still render without audio"
    print("PASS  a run with nothing recorded says so instead of rendering an empty page")


if __name__ == "__main__":
    check_replayed_step_is_deduped_to_the_later_record()
    check_page_embeds_audio_and_transcripts()
    check_subsampling_keeps_both_endpoints()
    check_a_run_with_nothing_says_so()
    print("ALL PASS")
