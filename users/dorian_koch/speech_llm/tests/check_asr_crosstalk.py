"""Guards for the podcast ASR stage and the bake-off scorer.

⚠ Renamed in spirit from `check_asr_crosstalk` on 2026-09-20: the crosstalk filter it was written to
guard has been DELETED. The filename is kept so no cluster file has to be removed by hand.

Why the filter went, recorded here because this file is where someone will look for it: it dropped
words whose diarized turn geometry suggested another speaker, and measurement showed it disagreed
with a direct "is this word actually louder on this channel" test on ~80% of its own decisions --
uniquely dropping 208/243 words whose audio WAS on the channel while missing 135/61 whose audio was
not. It had also silently deleted every word spoken during overlap. The user's call was that the
failure it targeted (text on a channel whose audio is elsewhere) is not worth the code, since it
runs at a few percent against Whisper's own 6-7% word error rate on this audio.

What still needs guarding is the part that produces numbers we act on: `clean_words`, and the
bake-off scorer's onset statistics, which chose a 12.8-20.8x cheaper ASR backend and whose first
version got that choice BACKWARDS.
"""

import importlib.util
import os
import sys


def find_setup_root(start):
    p = os.path.abspath(start)
    while p != "/":
        if os.path.isdir(os.path.join(p, "recipe")) and os.path.isdir(os.path.join(p, "work")):
            return p
        p = os.path.dirname(p)
    raise RuntimeError(f"no setup root above {start}")


SETUP = find_setup_root(__file__)
ASR_PY = os.path.join(SETUP, "recipe", "speech_llm", "full_duplex", "moshi_family", "podcast_asr.py")
SCORE_PY = os.path.join(SETUP, "analysis", "score_asr.py")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def w(text, start, end):
    return {"text": text, "start": start, "end": end}


def main():
    ok = 0
    A = load("podcast_asr", ASR_PY)

    # ------------------------------------------------- [1] the filter must be GONE, not dormant
    # A dormant copy is worse than none: the next person wires it back in without the measurement.
    for gone in ("filter_words_by_turns", "_spans_by_speaker", "_distance_to_spans"):
        assert not hasattr(A, gone), f"{gone} still exists -- it was deliberately deleted"
    # The SIGNATURE, not a text grep -- a docstring that explains why the filter was removed
    # legitimately mentions turns, and the first version of this check failed on exactly that.
    import inspect

    params = list(inspect.signature(A.transcribe_one).parameters)
    assert params == ["be", "wav_path"], f"transcribe_one must take no turns/speaker argument, got {params}"
    ok += 1

    # ------------------------------------------------- [2] clean_words
    words = [
        w("", 1.0, 1.2),  # empty text
        w("ok", 2.0, 2.0005),  # below MIN_WORD_SEC -- DTW backtrace noise, not speech
        w("b", 5.0, 5.3),
        w("a", 3.0, 3.4),  # out of order on purpose
    ]
    cleaned = A.clean_words(words)
    assert [x["text"] for x in cleaned] == ["a", "b"], cleaned
    assert all(cleaned[i]["start"] <= cleaned[i + 1]["start"] for i in range(len(cleaned) - 1)), (
        "must be onset-ordered -- interleave_text places words by onset"
    )
    ok += 1

    # ------------------------------------------------- [3] the scorer cannot pass vacuously
    S = load("score_asr", SCORE_PY)
    ref = [w("alpha", 1.0, 1.2), w("bravo", 2.0, 2.2), w("charlie", 3.0, 3.2)]
    same = S.wer_ops([S.norm(x["text"]) for x in ref], [S.norm(x["text"]) for x in ref])[1]
    d_same = S.onset_deltas(ref, ref, same)
    assert len(d_same) == 3 and all(abs(x) < 1e-9 for x in d_same), d_same
    ok += 1

    # a totally disagreeing transcript leaves an EMPTY agreement set, which must not read as a
    # clean zero -- that is the shape of a passing result
    other = [w("xray", 1.0, 1.2), w("yankee", 2.0, 2.2), w("zulu", 3.0, 3.2)]
    ops = S.wer_ops([S.norm(x["text"]) for x in ref], [S.norm(x["text"]) for x in other])[1]
    d_none = S.onset_deltas(ref, other, ops)
    assert len(d_none) == 0, d_none
    assert len(d_none) < 0.5 * len(ref), "the vacuity gate must fire on a no-agreement pair"
    ok += 1

    # a real shift must be DETECTED, in frames
    shifted = [w(x["text"], x["start"] + 0.08, x["end"] + 0.08) for x in ref]  # exactly +1 frame
    ops2 = S.wer_ops([S.norm(x["text"]) for x in ref], [S.norm(x["text"]) for x in shifted])[1]
    d_shift = S.onset_deltas(ref, shifted, ops2)
    assert all(abs(x - 1.0) < 1e-6 for x in d_shift), f"expected +1.000 frame, got {d_shift}"
    ok += 1

    # ------------------------------------------------- [4] the onset estimator is OUTLIER-ROBUST
    # Pins a real defect: difflib mis-pairs ~3-5% of words in a 1 h transcript against a repeat
    # elsewhere, landing deltas at hundreds of frames. The RAW MEAN could not survive it, and failed
    # DIRECTIONALLY -- rejecting the backend that matched the reference exactly (medium, raw mean
    # +1.240 vs trimmed -0.018) while flattering the one with a systematic 1.5-frame shift (turbo,
    # raw mean -0.227 while only 28.7% of its onsets were within a frame). Both real numbers.
    import statistics as _st

    clean = [0.0] * 194 + [0.25, -0.25, 0.125, -0.125, 0.0, 0.0]
    contaminated = clean + [2626.0, -343.0, 900.0, -500.0]
    assert abs(S.trimmed_mean(contaminated)) < 0.25, S.trimmed_mean(contaminated)
    assert abs(_st.fmean(contaminated)) >= 0.25, (
        "non-vacuous: the RAW mean must fail this same input, or the trim proves nothing"
    )
    assert S.frac_within(contaminated, 1.0) >= 0.70
    ok += 1

    # a genuine systematic shift must NOT be trimmed away...
    assert abs(S.trimmed_mean([-1.5] * 200)) >= 0.25, "a uniform shift must survive trimming"
    assert S.frac_within([-1.5] * 200, 1.0) < 0.70
    # ...and a distribution that averages to zero by cancelling is caught only by concentration
    split = [-4.0] * 100 + [4.0] * 100
    assert abs(S.trimmed_mean(split)) < 0.25, "fixture check: this really does centre at zero"
    assert S.frac_within(split, 1.0) < 0.70, "the concentration gate is what catches a mean-zero split distribution"
    ok += 1

    # empty must be nan and must FAIL both gates, never pass them
    assert S.trimmed_mean([]) != S.trimmed_mean([]), "empty -> nan, never 0.0"
    assert not (abs(S.trimmed_mean([])) < 0.25)
    assert not (S.frac_within([], 1.0) >= 0.70)
    ok += 1

    # ------------------------------------------------- [5] WER + rare-word classification
    st, _ = S.wer_ops(["the", "tony", "blair"], ["the", "tommy", "blair"])
    assert st["sub"] == 1 and st["n_ref"] == 3, st
    rf_rare, _ = S.rare_frac(["tony", "blumenthal"])
    rf_common, _ = S.rare_frac(["the", "and"])
    assert rf_rare == 1.0 and rf_common == 0.0, (rf_rare, rf_common)
    ok += 1

    print(f"check_asr_crosstalk: {ok}/{ok} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
