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
    from datasets import Dataset, Features, List, Sequence, Value

    # ⚠ The feature types are declared EXPLICITLY, matching production, and that is the whole point
    # of this fixture. `chatterbox_inference.dialogue_features` stores turns as a `Sequence({...})`,
    # which datasets returns COLUMNAR (a dict of lists); `moshi_annotate_inference.ALIGNMENT_FEATURE`
    # stores alignments as a `List({...})`, which comes back as a list of dicts. An earlier version
    # of this check built both with `Dataset.from_list` and no features, which infers List-of-struct
    # for both -- so it passed while the job crashed on the first real corpus with
    # "'str' object has no attribute 'get'". A fixture that does not carry the production schema is
    # testing a program we do not run.
    turns_features = Features(
        {
            "id": Value("string"),
            "turns": Sequence(
                {
                    "speaker": Value("string"),
                    "start_time": Value("float32"),
                    "end_time": Value("float32"),
                    "text": Value("string"),
                }
            ),
        }
    )
    align_features = Features(
        {
            "id": Value("string"),
            "alignments": List(
                {
                    "text": Value("string"),
                    "start": Value("float32"),
                    "end": Value("float32"),
                    "speaker": Value("string"),
                }
            ),
        }
    )

    # Three dialogues. Row "b" is the one the annotate stage will "fail" on.
    def _turns(*entries):
        """Columnar turns, as Sequence(struct) stores them."""
        return {
            "speaker": [e[0] for e in entries],
            "start_time": [float(i) for i in range(len(entries))],
            "end_time": [float(i + 1) for i in range(len(entries))],
            "text": [e[1] for e in entries],
        }

    def _align(sentence):
        return [{"text": w, "start": 0.0, "end": 0.1, "speaker": "SPEAKER_MAIN"} for w in sentence.split()]

    # Three dialogues. Row "b" is the one the annotate stage will "fail" on.
    tts = Dataset.from_dict(
        {
            "id": ["a", "b", "c"],
            "turns": [
                _turns(
                    ("user", "WRONG WRONG WRONG WRONG"),
                    (ASSISTANT_SPEAKER, "the capital of France is Paris"),
                ),
                _turns((ASSISTANT_SPEAKER, "this row never got annotated")),
                _turns((ASSISTANT_SPEAKER, "one two three four five six")),
            ],
        },
        features=turns_features,
    )
    # Annotate output: "b" is MISSING, so row 1 here is "c". A positional join would score c's
    # transcript against b's text.
    ann = Dataset.from_dict(
        {
            "id": ["a", "c"],
            "alignments": [
                _align("the capital of France is Paris"),
                _align("one two three four five six"),
            ],
        },
        features=align_features,
    )
    # Round-trip through disk so the declared features are what the job actually reads back.
    tts.save_to_disk(str(tmp / "tts"))
    ann.save_to_disk(str(tmp / "ann"))

    from datasets import load_from_disk

    _t = load_from_disk(str(tmp / "tts"))
    assert isinstance(_t[0]["turns"], dict), (
        "the fixture's turns column did not come back columnar -- it no longer reproduces the "
        "production Sequence(struct) layout, so this check would pass on a schema we never see"
    )
    _a = load_from_disk(str(tmp / "ann"))
    assert isinstance(_a[0]["alignments"], list), (
        "the fixture's alignments column is not a list of dicts -- it no longer matches ALIGNMENT_FEATURE"
    )
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
        worst_min_ref_words=5,
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
    from datasets import Dataset, Features, List, Value

    align_features = Features(
        {
            "id": Value("string"),
            "alignments": List(
                {
                    "text": Value("string"),
                    "start": Value("float32"),
                    "end": Value("float32"),
                    "speaker": Value("string"),
                }
            ),
        }
    )

    bad = Dataset.from_dict(
        {
            "id": ["a"],
            "alignments": [
                [
                    {"text": w, "start": 0.0, "end": 0.1, "speaker": "SPEAKER_MAIN"}
                    for w in "completely different words entirely".split()
                ]
            ],
        },
        features=align_features,
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
