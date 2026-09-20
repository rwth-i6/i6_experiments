"""Guard: the podcast ASR crosstalk filter, and the bake-off scorer's non-vacuity.

WHY THIS EXISTS. A diffusion separator leaves residual bleed, so Whisper transcribes the OTHER
speaker through a channel some of the time. Those words would be written into `alignments`, and
`interleave_text` places every alignment it is given while IGNORING the speaker field -- so the model
would be trained to speak its interlocutor's lines. Nothing about that is visible in a loss curve:
the run finishes, the checkpoint loads, and the only symptom is a model that echoes the user. The
filter is the only thing standing between the separator and that outcome, so it is guarded.

The second half guards the SCORER, because a bake-off that silently compares nothing would read as a
perfect result: onset statistics are computed over words the two transcripts agree on, and an empty
agreement set gives a mean of 0.0 and a median of 0.0 -- which is exactly the "PASS" signature.

Login node, no GPU, ~1 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_asr_crosstalk.py
"""

import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def find_setup_root(start):
    """Walk up to the setup root. Counting `..` segments is how this broke the first time -- and
    `recipe/i6_experiments` is a symlink, so the depth is not obvious from the repo layout either."""
    d = start
    for _ in range(12):
        if os.path.isdir(os.path.join(d, "recipe")) and os.path.isdir(os.path.join(d, "work")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    raise SystemExit(f"could not locate the setup root above {start}")


SETUP = find_setup_root(HERE)
ASR_PY = os.path.join(SETUP, "recipe", "speech_llm", "full_duplex", "moshi_family", "podcast_asr.py")
SCORE_PY = os.path.join(SETUP, "analysis", "score_asr.py")


def load(name, path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


A = load("podcast_asr", ASR_PY)


def w(text, start, end):
    return {"text": text, "start": start, "end": end, "prob": 0.9}


def turn(spk, a, b):
    return {"speaker": spk, "start": a, "end": b}


def main():
    ok = 0

    # ---------------------------------------------------------------- [1] the basic separation
    # SPEAKER_00 holds 0-10 s, SPEAKER_01 holds 10-20 s. A channel belonging to SPEAKER_00 must keep
    # only the words in its own turn.
    turns = [turn("SPEAKER_00", 0.0, 10.0), turn("SPEAKER_01", 10.0, 20.0)]
    words = [w("hello", 1.0, 1.4), w("world", 2.0, 2.5), w("bleed", 12.0, 12.4)]
    kept, dropped = A.filter_words_by_turns(words, turns, "SPEAKER_00")
    assert [x["text"] for x in kept] == ["hello", "world"], kept
    assert dropped == 1, dropped
    ok += 1

    # NON-VACUOUS: the same call for the OTHER speaker must keep the complementary set. A filter that
    # simply kept everything, or dropped everything, would pass a one-sided test.
    kept2, dropped2 = A.filter_words_by_turns(words, turns, "SPEAKER_01")
    assert [x["text"] for x in kept2] == ["bleed"], kept2
    assert dropped2 == 2, dropped2
    assert {x["text"] for x in kept} & {x["text"] for x in kept2} == set(), "buckets must be disjoint"
    ok += 1

    # ------------------------------------------------------------- [2] midpoint, not overlap
    # A word straddling the 10 s boundary belongs to whoever held the MIDPOINT. Overlap-based keeping
    # would admit it on BOTH channels, i.e. duplicate a word into the user's stream.
    straddle = [w("boundary", 9.6, 10.6)]  # midpoint 10.1 -> SPEAKER_01
    k0, _ = A.filter_words_by_turns(straddle, turns, "SPEAKER_00")
    k1, _ = A.filter_words_by_turns(straddle, turns, "SPEAKER_01")
    assert len(k0) == 0 and len(k1) == 1, (k0, k1)
    assert not (k0 and k1), "a word must never survive on both channels"
    ok += 1

    # ------------------------------------------------------- [3] a speaker with no turns at all
    # Must drop everything AND say so. Silently keeping the words would attribute a whole channel of
    # speech to someone the diarizer never heard -- the worst possible failure, and the most
    # plausible-looking.
    kept3, dropped3 = A.filter_words_by_turns(words, turns, "SPEAKER_99")
    assert kept3 == [] and dropped3 == len(words), (kept3, dropped3)
    ok += 1

    # ------------------------------------------------------------------- [4] the pad is applied
    # pyannote turn edges are not word-accurate; a word starting a hair before its turn is real
    # speech, not crosstalk. Assert the pad rescues it and that a word far outside is still dropped.
    edge = [w("just-before", 9.85, 9.95)]  # midpoint 9.90, inside SPEAKER_00's turn anyway
    early = [w("early", -0.15, -0.05)]  # midpoint -0.10, before the turn, inside the 0.20 pad
    far = [w("far", 30.0, 30.4)]
    assert len(A.filter_words_by_turns(edge, turns, "SPEAKER_00")[0]) == 1
    assert len(A.filter_words_by_turns(early, turns, "SPEAKER_00")[0]) == 1, "pad must rescue"
    assert len(A.filter_words_by_turns(far, turns, "SPEAKER_00")[0]) == 0, "pad must not be unbounded"
    ok += 1

    # ----------------------------------------------------------------- [5] clean_words is sane
    dirty = [w("", 1.0, 1.2), w("ok", 5.0, 5.001), w("b", 3.0, 3.3), w("a", 1.0, 1.3)]
    cleaned = A.clean_words(dirty)
    assert [x["text"] for x in cleaned] == ["a", "b"], cleaned  # empty dropped, 1 ms word dropped
    assert all(cleaned[i]["start"] <= cleaned[i + 1]["start"] for i in range(len(cleaned) - 1)), (
        "must be onset-ordered -- interleave_text places words by onset"
    )
    ok += 1

    # ------------------------------------------------- [6] the scorer cannot pass vacuously
    S = load("score_asr", SCORE_PY)
    ref = [w("alpha", 1.0, 1.2), w("bravo", 2.0, 2.2), w("charlie", 3.0, 3.2)]
    # identical transcript -> every word matched, zero delta
    same_ops = S.wer_ops([S.norm(x["text"]) for x in ref], [S.norm(x["text"]) for x in ref])[1]
    d_same = S.onset_deltas(ref, ref, same_ops)
    assert len(d_same) == 3 and all(abs(x) < 1e-9 for x in d_same), d_same
    ok += 1

    # totally disagreeing transcript -> the agreement set is EMPTY, so the onset stats are empty too.
    # This is the trap: an empty list must not look like a clean zero-delta result. The scorer's gate
    # rejects on `n_match < 0.5 * len(ref)`, so assert that condition really fires here.
    other = [w("xray", 1.0, 1.2), w("yankee", 2.0, 2.2), w("zulu", 3.0, 3.2)]
    ops = S.wer_ops([S.norm(x["text"]) for x in ref], [S.norm(x["text"]) for x in other])[1]
    d_none = S.onset_deltas(ref, other, ops)
    assert len(d_none) == 0, d_none
    assert len(d_none) < 0.5 * len(ref), "the vacuity gate must fire on a no-agreement pair"
    ok += 1

    # a real shift must be DETECTED, in frames -- otherwise the whole onset criterion is a no-op
    shifted = [w(x["text"], x["start"] + 0.08, x["end"] + 0.08) for x in ref]  # exactly +1 frame
    ops2 = S.wer_ops([S.norm(x["text"]) for x in ref], [S.norm(x["text"]) for x in shifted])[1]
    d_shift = S.onset_deltas(ref, shifted, ops2)
    assert len(d_shift) == 3, d_shift
    assert all(abs(x - 1.0) < 1e-6 for x in d_shift), f"expected +1.000 frame, got {d_shift}"
    ok += 1

    # ------------------------------------- [7] the onset estimator must be OUTLIER-ROBUST
    # This pins a real defect, found 2026-09-19 when the bake-off's own verdict came back wrong.
    # difflib pairs ~3-5% of words in a 1 h transcript against a repeat elsewhere in the hour, which
    # lands deltas at hundreds of frames. The RAW MEAN cannot survive that, and the failure is
    # directional: it rejected the candidate that matched the reference exactly (medium, raw mean
    # +1.240 vs trimmed -0.018) and flattered the one with a systematic 1.5-frame shift (turbo, raw
    # mean -0.227 while only 28.7% of its onsets were within a frame). Both real numbers.
    clean = [0.0] * 194 + [0.25, -0.25, 0.125, -0.125, 0.0, 0.0]  # 200 well-aligned words
    contaminated = clean + [2626.0, -343.0, 900.0, -500.0]  # 4 mis-pairings, ~2% of the set
    assert abs(S.trimmed_mean(contaminated)) < 0.25, (
        "a distribution that is dead-on apart from a ~2% mis-pairing tail must read as ALIGNED; "
        f"trimmed mean was {S.trimmed_mean(contaminated)}"
    )
    import statistics as _st

    assert abs(_st.fmean(contaminated)) >= 0.25, (
        "non-vacuous: the RAW mean must fail on this same input, or the trim proves nothing"
    )
    assert S.frac_within(contaminated, 1.0) >= 0.70, S.frac_within(contaminated, 1.0)
    ok += 1

    # ...and the complement: a genuine systematic shift must NOT be trimmed away. A trimmed mean
    # alone could be centred by a late half cancelling an early half, which is noise, not agreement
    # -- which is why the gate also requires concentration.
    shifted_all = [-1.5] * 200
    assert abs(S.trimmed_mean(shifted_all)) >= 0.25, "a uniform shift must survive trimming"
    assert S.frac_within(shifted_all, 1.0) < 0.70, "a 1.5-frame shift is not 'within 1 frame'"
    split = [-4.0] * 100 + [4.0] * 100  # mean 0, but nothing is actually aligned
    assert abs(S.trimmed_mean(split)) < 0.25, "fixture check: this really does centre at zero"
    assert S.frac_within(split, 1.0) < 0.70, (
        "the concentration gate is what catches a distribution that averages to zero by cancelling"
    )
    ok += 1

    # an empty agreement set must not read as a clean zero through the new estimators either
    assert S.trimmed_mean([]) != S.trimmed_mean([]), "empty -> nan, never 0.0"
    assert not (abs(S.trimmed_mean([])) < 0.25), "nan must FAIL the gate comparison, not pass it"
    assert not (S.frac_within([], 1.0) >= 0.70), "nan must FAIL the concentration gate"
    ok += 1

    # ----------------------------------------- [9] OVERLAP is kept; only BOUNDARY ties are dropped
    # The filter's first version dropped every word spoken during simultaneous speech, because its
    # tie test `other <= d_mine` reads `0 <= 0` when both speakers are active. On the real bake-off
    # transcripts that was ~40% of everything it removed (173/436 on ch0, 131/427 on ch1) -- and
    # overlap is the single most valuable thing a duplex corpus holds, the reason `overlap_sec` is
    # stored at all. Not a crash; just a quietly poorer corpus, invisible once the shards exist.
    ov_turns = [
        turn("A", 10.0, 20.0),
        turn("B", 14.0, 24.0),  # 14-20 s: BOTH speaking
    ]
    w_ov = w("simultaneous", 16.0, 16.4)  # midpoint 16.2 -- inside both turns
    kept_a, _ = A.filter_words_by_turns([w_ov], ov_turns, "A")
    kept_b, _ = A.filter_words_by_turns([w_ov], ov_turns, "B")
    assert kept_a and kept_b, (
        "a word spoken during overlap must survive on ITS OWN channel -- in production the two "
        f"channels carry different transcripts, so this is not duplication. got A={kept_a} B={kept_b}"
    )
    ok += 1

    # Non-vacuity: the OLD rule must genuinely have failed this fixture, or the check proves nothing.
    _by = A._spans_by_speaker(ov_turns)
    _starts = {k: [s for s, _ in v] for k, v in _by.items()}
    _mid = 16.2
    _d_mine = A._distance_to_spans(_mid, _by["A"], _starts["A"])
    _old_would_drop = any(A._distance_to_spans(_mid, _by[o], _starts[o]) <= _d_mine for o in _by if o != "A")
    assert _d_mine == 0.0, f"fixture is wrong: the word must be INSIDE A's turn, d={_d_mine}"
    assert _old_would_drop, (
        "non-vacuous: the previous nearest-speaker-with-tie-drop rule must DROP this word, "
        "otherwise [9] would pass against the very implementation it exists to reject"
    )
    ok += 1

    # ...and the boundary case must STILL be dropped from both. That is the original bug (a word
    # straddling a turn edge written into the assistant stream AND the user stream) and relaxing
    # overlap must not have relaxed it.
    bd_turns = [turn("A", 10.0, 20.0), turn("B", 20.1, 30.0)]
    w_bd = w("straddle", 20.0, 20.1)  # midpoint 20.05 -- outside both, within pad of both
    kb_a, _ = A.filter_words_by_turns([w_bd], bd_turns, "A")
    kb_b, _ = A.filter_words_by_turns([w_bd], bd_turns, "B")
    assert not kb_a and not kb_b, f"an equidistant BOUNDARY word must go to neither channel, got A={kb_a} B={kb_b}"
    ok += 1

    # The breakdown must be reported, since it is what decomposes a drop rate into "crosstalk" vs
    # "overlap" vs "someone else was closer" -- the question we could not answer before it existed.
    st9 = {}
    mixed = [w_ov, w_bd, w("far_away", 40.0, 40.5)]
    kept9, drop9 = A.filter_words_by_turns(mixed, ov_turns + [turn("B", 20.1, 30.0)], "A", stats=st9)
    assert set(st9) == {"outside_own_turns", "other_speaker_closer", "overlap_kept"}, st9
    assert st9["overlap_kept"] >= 1, st9
    assert st9["outside_own_turns"] >= 1, st9
    assert drop9 == len(mixed) - len(kept9), (drop9, len(mixed), len(kept9))
    ok += 1

    # WER must count, and rare-word classification must separate content from function words
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
