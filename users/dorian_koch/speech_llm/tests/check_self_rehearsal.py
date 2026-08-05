"""Guard: the B6 rehearsal cleanup only ever DELETES, and it catches every defect we measured.

`MoshiRolloutToDialogue` takes base Moshi's inner monologue and hands it to TTS as assistant speech.
Whatever survives is trained on verbatim -- so a filter that misses a defect does not degrade a
metric, it teaches the defect. And the defect classes here are exactly the ones a loss curve cannot
show you: mangled text still has a perfectly ordinary loss.

Every fixture below is a **verbatim** rollout from
``output/benchmark/moshi_family/moshi_output/<i>.txt`` (base Moshi, 1,000 clips, 2026-08-05). Nothing
is paraphrased or shortened; a hand-written approximation of a defect is not evidence that the real
one is caught.

Measured on that sample with the defaults: 99.5% of replies open with a harness greeting, 15.7% end
mid-sentence, and the filters keep 76.0% at a mean of 26.8 assistant words.

Run from the setup root, no GPU needed:
    CUDA_HOME=/usr .venv/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_self_rehearsal.py
"""

import json
import sys
import tempfile
from pathlib import Path

SETUP = Path(__file__).resolve().parents[5]  # .../recipe/i6_experiments/users/<u>/speech_llm/tests
sys.path.insert(0, str(SETUP))
sys.path.insert(0, str(SETUP / "sisyphus"))  # `sisyphus` package lives here

from i6_experiments.users.dorian_koch.speech_llm.self_rehearsal import (  # noqa: E402
    MoshiRolloutToDialogue,
    clean_monologue,
    cut_at_glitch,
    repeat_ratio,
    sentence_repeat_ratio,
    strip_lead_greeting,
    trim_to_last_sentence,
    words,
)

failures = []

# --- verbatim rollouts, keyed by their clip index ------------------------------------------------
ROLLOUTS = {
    0: "Hello, how can I help you? I' not sure. What's the year of this college?",
    7: "Hi, how is it going?",
    9: (
        "Hello, how can I help you? Muhammad Ali was stripped of his heavyweight titles in April "
        "1967 after refusing his induction into the U.S. Army."
    ),
    14: "Hello, what's up? I don't know. don't know. Okay. I don' know that one. Okay. I don' know that one. Okay.",
    15: (
        "Hi, how is it going? Pép Guardiola was appointed as the manager of Bayern Munich in 2013. "
        "took over from Jupp Heynckes, who had retired from management. I' just trying to think of "
        "the specifics. I think that's correct. Let's keep going. What's the next question?"
    ),
    26: (
        "Hey, how can I help you? Anaphylaxis is a severe, life threatening reaction caused by an "
        "overreaction of the immune system. It can cause difficulty breathing, low blood pressure, "
        "and swelling in various parts of the body. It' important to recognize the symptoms and "
        "seek medical help immediately. Yeah, it's a very serious situation. It' a life threatening "
        "condition. Yeah,"
    ),
    97: (
        "Good day, how is it going? is the Supreme Governor of the Church of England? The Queen is "
        "the Supreme Governor of the Church of England. Queen is the head of the Church of England."
    ),
    367: "Hello, what's going on? Oh, that's right. Yeah, that's right. That's correct. That's right. Yeah. Yeah. Yeah.",
    428: "Good day. What's going on? is it made of? it a salad? Okay, it's a salad. Okay. Okay, okay. Okay. Okay.",
}

# --- 1. the harness greeting is removed, whatever form it takes ----------------------------------
# The rollout feeds lead_in_s of silence before the question, so the model greets into it. Rendered
# after the user's question in a two-turn dialogue that greeting lands in the wrong place entirely.
for i, raw in ROLLOUTS.items():
    if strip_lead_greeting(raw) == raw.strip():
        failures.append(f"[{i}] greeting not stripped: {raw[:60]!r}")
print(f"[ok] greeting strip      all {len(ROLLOUTS)} verbatim openers removed")


# --- 2. the cleanup only ever deletes ------------------------------------------------------------
# THE load-bearing invariant. Self-generated data is only worth preferring over more synthetic data
# because it is the model's own distribution; a cleanup that rewrites a word gives that up silently.
# Front-drop + tail-cut means the result must be a CONTIGUOUS RUN of the original words.
def _is_contiguous_run(needle, haystack):
    if not needle:
        return True
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start : start + len(needle)] == needle:
            return True
    return False


for i, raw in ROLLOUTS.items():
    src = words(strip_lead_greeting(raw))
    got = words(clean_monologue(raw))
    if not _is_contiguous_run(got, src):
        failures.append(f"[{i}] cleanup did not just delete: {got} is not a run inside {src}")
print("[ok] delete-only         every cleaned reply is a contiguous run of the original words")

# Punctuation counts as the model's words too: the tail trim used to substitute a literal "." for
# the whole match, silently turning a question mark into a full stop. The fragment here starts with
# a CAPITAL on purpose -- with a lowercase one the dropped-word cut fires first and the trim, which
# is what this line is about, never runs.
if trim_to_last_sentence("Is it a salad? Yeah it") != "Is it a salad?":
    failures.append("tail trim rewrote the sentence terminator instead of preserving it")
if clean_monologue("Is it a salad? Yeah it", strip_greeting=False) != "Is it a salad?":
    failures.append("full cleanup rewrote the sentence terminator")
print("[ok] terminator kept     a trimmed question stays a question")

# --- 3. each measured defect is actually caught --------------------------------------------------
# Truncated contraction ("It' important"): salvage the clean prefix rather than dropping 26 outright.
salvaged = clean_monologue(ROLLOUTS[26])
if not salvaged.startswith("Anaphylaxis is a severe"):
    failures.append(f"[26] clean prefix not salvaged: {salvaged[:80]!r}")
if "It'" in salvaged or "Yeah," == salvaged[-5:]:
    failures.append(f"[26] glitched tail survived: {salvaged[-60:]!r}")
if cut_at_glitch("It's fine. It' broken") != "It's fine.":
    failures.append("cut_at_glitch fired on a well-formed contraction")
print('[ok] glitch cut          clean prefix kept, glitched tail removed, "It\'s" not a false positive')

# Dropped opening word, mid-reply ("...in 2013. took over from Jupp Heynckes").
if "took over" in clean_monologue(ROLLOUTS[15]):
    failures.append("[15] sentence with a dropped opening word survived")
# ...and at the very start, where there is no preceding terminator to anchor on.
cleaned_97 = clean_monologue(ROLLOUTS[97])
if cleaned_97.startswith("is the Supreme"):
    failures.append(f"[97] leading fragment survived: {cleaned_97[:60]!r}")
print("[ok] dropped-word cut    fires mid-reply and at the reply start")

# False positives: an abbreviation or a numbered noun must not read as a sentence boundary.
for text in (
    "Muhammad Ali was stripped of his titles in April 1967 after refusing induction into the U.S. Army.",
    "She is the mother of the current world No. 1, Sabalenka, and the brother of tennis player Kamil.",
):
    if clean_monologue(text, strip_greeting=False) != text:
        failures.append(f"dropped-word cut tripped on an abbreviation/number: {text!r}")
print('[ok] no false positives  "U.S. Army" and "No. 1, Sabalenka" survive intact')


# --- 4. degenerate replies are dropped, not shortened into something plausible --------------------
# sisyphus.Job.__new__ refuses direct construction outside a graph, so build a plain copy of the
# class. Thresholds are read off the real job rather than restated here, so a default that moves in
# the job moves here too instead of leaving the guard testing numbers nobody uses.
class _P:
    """Stand-in for a sisyphus output handle."""

    def __init__(self, p):
        self.p = str(p)

    def get(self):
        return self.p

    def get_path(self):
        return self.p


_Plain = type(
    "_Plain",
    (),
    dict(MoshiRolloutToDialogue.__dict__, output_path=lambda self, name, **kw: _P(name)),
)


def _bare_job():
    job = object.__new__(_Plain)
    MoshiRolloutToDialogue.__init__(job, clips=None, questions=None)
    return job


JOB = _bare_job()
DEFAULTS = dict(min_words=JOB.min_words, max_repeat=JOB.max_repeat_ratio, max_sent=JOB.max_sentence_repeat_ratio)


def kept(raw):
    t = clean_monologue(raw)
    return (
        len(words(t)) >= DEFAULTS["min_words"]
        and repeat_ratio(t) <= DEFAULTS["max_repeat"]
        and sentence_repeat_ratio(t) <= DEFAULTS["max_sent"]
    )


for i, why in (
    (7, "greeting only, no answer"),
    (0, "glitched from the first word"),
    (14, '"I don\' know" loop'),
    (367, "pure backchannel"),
    (428, '"Okay." loop'),
):
    if kept(ROLLOUTS[i]):
        failures.append(f"[{i}] should have been dropped ({why}): {clean_monologue(ROLLOUTS[i])!r}")
print("[ok] degenerate dropped  greeting-only / glitch-only / repetition loops all rejected")

# A good reply must survive BYTE-IDENTICAL -- filters that quietly prettify text would make every
# example quoted from this corpus a misquotation.
expected_9 = (
    "Muhammad Ali was stripped of his heavyweight titles in April 1967 after refusing his induction into the U.S. Army."
)
if clean_monologue(ROLLOUTS[9]) != expected_9:
    failures.append(f"[9] a clean reply was altered: {clean_monologue(ROLLOUTS[9])!r}")
if not kept(ROLLOUTS[9]):
    failures.append("[9] a clean, correct reply was dropped")
print("[ok] clean reply kept    survives byte-identical")

# --- 5. drive the REAL run() end to end ----------------------------------------------------------
# The functions above are only half the job; the other half is the join (clip index -> question row)
# and the dialogue schema the TTS pipeline consumes. A guard that stopped at the helpers would not
# have caught either.
from datasets import Dataset, load_from_disk  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    clips_dir = tmp / "clips"
    clips_dir.mkdir()
    order = sorted(ROLLOUTS)
    for row, i in enumerate(order):
        (clips_dir / f"{row}.txt").write_text(ROLLOUTS[i], encoding="utf-8")
    q_dir = tmp / "questions"
    Dataset.from_dict(
        {
            "question": [f"question for clip {i}?" for i in order],
            "answer": ["Muhammad Ali" if i == 9 else "no-such-answer" for i in order],
            "aliases": [json.dumps([]) for _ in order],
            "category": ["unknown" for _ in order],
        }
    ).save_to_disk(str(q_dir))

    job = _bare_job()
    job.clips = _P(clips_dir)
    job.questions = _P(q_dir)
    job.out_hf = _P(tmp / "out_hf")
    job.out_stats = _P(tmp / "stats.json")
    job.run()

    stats = json.loads((tmp / "stats.json").read_text())
    out = load_from_disk(str(tmp / "out_hf"))

    if stats["n_kept"] != len(out):
        failures.append(f"stats n_kept={stats['n_kept']} but wrote {len(out)} rows")
    if stats["n_kept"] + sum(stats["dropped"].values()) != len(order):
        failures.append(f"rows unaccounted for: kept={stats['n_kept']} dropped={stats['dropped']}")
    for row in out:
        turns = json.loads(row["dialogue"])
        if [t["speaker"] for t in turns] != ["user", "assistant"]:
            failures.append(f"dialogue is not user-then-assistant: {turns}")
        if not turns[1]["text"].strip():
            failures.append("assistant turn is empty")
    # Correctness is recorded, never used to filter: the wrong-answer rows must still be present.
    if set(out["correct"]) == {1}:
        failures.append("only correct rows survived -- correctness must be a column, not a filter")
    if sum(out["correct"]) == 0:
        failures.append("the one row with a matching gold answer was not marked correct")
print(f"[ok] end-to-end run()    {len(order)} clips -> {stats['n_kept']} dialogue rows, schema + join verified")


if failures:
    print("\nFAILED:", file=sys.stderr)
    for f in failures:
        print(f"  - {f}", file=sys.stderr)
    raise SystemExit(1)

print("\nrehearsal cleanup deletes only, catches every measured defect, and keeps good text verbatim")
