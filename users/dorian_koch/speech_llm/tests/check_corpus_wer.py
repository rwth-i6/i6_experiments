"""Guard: the TTS round-trip WER job (backlog E4) measures the right two things.

Three failure modes, none of which a number alone would reveal:

* **Joining by row position.** The annotate stage DROPS rows whose audio was missing or whose
  Whisper call failed, so its output is not positionally aligned with the TTS output. A zip would
  compare row i's authored text against row j's transcript and produce a perfectly plausible,
  entirely meaningless WER. Asserted with a fixture where a middle row is dropped.
* **Scoring the wrong speaker.** Only the assistant channel is transcribed; comparing against user
  turns would be measuring nothing.
* **Inventing a value for an unmeasurable row.** An empty reference has no WER; folding it in as 0
  or 1 silently shifts the corpus number.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_corpus_wer.py
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
# `recipe/sisyphus` must come FIRST: `recipe` alone shadows the installed sisyphus with the
# symlinked source tree, which imports as an empty namespace package ("cannot import name 'Job'").
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))

from i6_experiments.users.dorian_koch.speech_llm.corpus_wer import (  # noqa: E402
    ASSISTANT_SPEAKER,
    SyntheticSpeechWer,
    normalize_for_wer,
    row_wer,
)

failures = []


def _expect(cond, msg):
    if not cond:
        failures.append(msg)


# --- normalisation is minimal and symmetric -------------------------------------------------------
_expect(
    normalize_for_wer("Hello, World!") == ["hello", "world"],
    "normalisation should lowercase and drop punctuation",
)
_expect(normalize_for_wer("don't") == ["don't"], "an apostrophe is part of the word, not punctuation")
_expect(normalize_for_wer("  a   b  ") == ["a", "b"], "whitespace should collapse")
_expect(normalize_for_wer(None) == [], "a missing string must not raise")
# It must NOT normalise numbers to words: that would absorb real TTS errors into the normaliser.
_expect(
    normalize_for_wer("1998") != normalize_for_wer("nineteen ninety eight"),
    "normalisation must not map digits to words -- that hides real substitutions",
)
print("[ok] normalisation      minimal, symmetric, and does not over-normalise")

# --- an unmeasurable row is None, not 0 and not 1 -------------------------------------------------
_expect(row_wer([], ["a"]) is None, "empty reference must be unmeasurable, not a WER of 0 or 1")
_expect(row_wer(["a", "b"], ["a", "b"]) == 0.0, "identical text is WER 0")
_expect(row_wer(["a", "b"], []) == 1.0, "an empty hypothesis against 2 ref words is WER 1.0")
_expect(abs(row_wer(["a", "b", "c"], ["a", "x", "c"]) - 1 / 3) < 1e-9, "one substitution of three")
print("[ok] row_wer            empty reference is unmeasurable, not invented")


# --- drive the real run() over a fixture where annotate dropped a row -----------------------------
def _fixture(tmp: Path):
    from datasets import Dataset

    # Three dialogues. Row "b" is the one the annotate stage will "fail" on.
    tts = Dataset.from_list(
        [
            {
                "id": "a",
                "turns": [
                    {"speaker": "user", "text": "WRONG WRONG WRONG WRONG", "start_time": 0.0, "end_time": 1.0},
                    {
                        "speaker": ASSISTANT_SPEAKER,
                        "text": "the capital of France is Paris",
                        "start_time": 1.0,
                        "end_time": 2.0,
                    },
                ],
            },
            {
                "id": "b",
                "turns": [
                    {
                        "speaker": ASSISTANT_SPEAKER,
                        "text": "this row never got annotated",
                        "start_time": 0.0,
                        "end_time": 1.0,
                    },
                ],
            },
            {
                "id": "c",
                "turns": [
                    {
                        "speaker": ASSISTANT_SPEAKER,
                        "text": "one two three four five six",
                        "start_time": 0.0,
                        "end_time": 1.0,
                    },
                ],
            },
        ]
    )
    # Annotate output: "b" is MISSING, so row 1 of this dataset is "c". A positional join would
    # score c's transcript against b's text.
    ann = Dataset.from_list(
        [
            {
                "id": "a",
                "alignments": [
                    {"text": w, "start": 0.0, "end": 0.1, "speaker": "SPEAKER_MAIN"}
                    for w in "the capital of France is Paris".split()
                ],
            },
            {
                "id": "c",
                "alignments": [
                    {"text": w, "start": 0.0, "end": 0.1, "speaker": "SPEAKER_MAIN"}
                    for w in "one two three four five six".split()
                ],
            },
        ]
    )
    tts.save_to_disk(str(tmp / "tts"))
    ann.save_to_disk(str(tmp / "ann"))
    return tmp / "tts", tmp / "ann"


class _P:
    def __init__(self, p):
        self._p = str(p)

    def get(self):
        return self._p

    def get_path(self):
        return self._p


with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    tts_p, ann_p = _fixture(tmp)
    # Sisyphus Job.__new__ takes over construction (it interns jobs by hash), so the real run() is
    # driven against a stand-in carrying exactly the attributes it reads. This still exercises the
    # REAL method -- the join, the speaker filter and the skip accounting are the code under test,
    # not a reimplementation.
    job = SimpleNamespace(
        tts_hf=_P(tts_p),
        annotated_hf=_P(ann_p),
        sample=0,
        seed=1234,
        worst_n=5,
        out_json=_P(tmp / "wer.json"),
        out_report=_P(tmp / "report.txt"),
        _text_only=SyntheticSpeechWer._text_only,
    )
    SyntheticSpeechWer.run(job)
    got = json.loads((tmp / "wer.json").read_text())

    # Both surviving rows match their OWN transcript exactly -> WER 0. Under a positional join,
    # "c"'s transcript would be scored against "b"'s text and this would be far from 0.
    _expect(got["scored"] == 2, f"expected 2 scored rows, got {got['scored']}")
    _expect(
        got["mean_row_wer"] == 0.0,
        f"mean row WER should be 0 when each row matches its own transcript; got "
        f"{got['mean_row_wer']} -- this is what a positional join breaks",
    )
    _expect(
        got["skipped_no_annotate_output"] == 1,
        f"the dropped row must be counted as skipped, not scored; got {got['skipped_no_annotate_output']}",
    )
    # The user turn on row "a" is 4 loud words that appear in no transcript. If they were scored,
    # row a's WER could not be 0.
    _expect(
        all(r["wer"] == 0.0 for r in got["worst"]),
        "a user turn leaked into the reference -- only the assistant channel is transcribed",
    )
    print("[ok] join by id         a dropped annotate row does not shift every later comparison")
    print("[ok] assistant only     user turns stay out of the reference")

    # Non-vacuous: the same fixture with a deliberately mismatched transcript must NOT score 0.
    from datasets import Dataset

    bad = Dataset.from_list(
        [
            {
                "id": "a",
                "alignments": [
                    {"text": w, "start": 0.0, "end": 0.1, "speaker": "SPEAKER_MAIN"}
                    for w in "completely different words entirely".split()
                ],
            },
        ]
    )
    bad.save_to_disk(str(tmp / "bad"))
    job.annotated_hf = _P(tmp / "bad")
    job.out_json, job.out_report = _P(tmp / "w2.json"), _P(tmp / "r2.txt")
    SyntheticSpeechWer.run(job)
    got2 = json.loads((tmp / "w2.json").read_text())
    _expect(
        got2["mean_row_wer"] > 0.5,
        f"a mismatched transcript must produce a large WER, got {got2['mean_row_wer']} -- "
        f"the zero above would then be meaningless",
    )
    print(f"[ok] check can fail     mismatched transcript scores {got2['mean_row_wer']:.2f}")

if failures:
    print("\nFAILED:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    raise SystemExit(1)

print("\nthe TTS round-trip WER joins by id, scores the assistant only, and invents nothing")
