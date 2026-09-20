"""Guard: diarization-anchored permutation repair fixes real flips and NEVER invents one.

Both directions matter, and each failure mode looks like success from the other side:

  * A repair that does nothing passes every "the clean signal is untouched" assertion perfectly, and
    leaves the corpus with the exact flips it was written to remove -- invisible, because a flipped
    episode is a perfectly normal-sounding stereo file with the speakers on the wrong channels.
  * A repair that is trigger-happy corrupts audio that was fine, and equally invisibly.

So every "it repairs X" assertion below is paired with an "it leaves Y alone", and the detector is
asserted to actually fire on the fixture where it must (non-vacuity), rather than merely being quiet.

Run: CUDA_HOME=/usr .venv/bin/python recipe/.../tests/check_perm_repair.py
"""

import os
import sys
from pathlib import Path

import numpy as np

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe" / "speech_llm" / "full_duplex"))
os.environ.setdefault("CUDA_HOME", "/usr")

from moshi_family.perm_repair import (  # noqa: E402
    FRAME_RATE,
    HOP_FRAMES,
    WINDOW_FRAMES,
    activity_tracks,
    channel_envelopes,
    flip_spans,
    repair_permutation,
    top_two_speakers,
    viterbi_states,
    window_margins,
)

SR = 24000
DUR = 300.0
A, B = "SPEAKER_00", "SPEAKER_01"

fails = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        fails.append(name)


def make_turns(dur=DUR, turn_sec=6.0, overlap=0.15):
    """Alternating turns with real overlap. A speaks longer, so it is unambiguously the top speaker."""
    turns, t, i = [], 0.0, 0
    while t < dur:
        spk = A if i % 2 == 0 else B
        length = turn_sec * (1.5 if spk == A else 0.8)
        end = min(dur, t + length)
        if end > t:
            turns.append({"speaker": spk, "start": t, "end": end})
        if end >= dur:
            break  # else t = end - overlap*length lands back inside and never terminates
        t = end - overlap * length
        i += 1
    return turns


def render(turns, rng, dur=DUR, sr=SR, silent_span=None, solo_span=None, solo_channel=0):
    """Two channels of speech-shaped noise gated by each speaker's turns.

    ``silent_span`` blanks both channels (a music-free ad gap / dead air).
    ``solo_span`` puts a single voice on ``solo_channel`` only -- the realistic ad read, and the
    input that gives a chained permutation decision nothing to anchor on.
    """
    n = int(dur * sr)
    audio = rng.normal(0.0, 1e-4, size=(2, n)).astype(np.float32)  # noise floor
    for t in turns:
        si = 0 if t["speaker"] == A else 1
        i0, i1 = int(t["start"] * sr), min(n, int(t["end"] * sr))
        if i1 <= i0:
            continue
        m = i1 - i0
        env = np.hanning(m).astype(np.float32) * 0.5 + 0.5
        audio[si, i0:i1] += (rng.normal(0.0, 0.3, size=m) * env).astype(np.float32)
    if silent_span is not None:
        i0, i1 = int(silent_span[0] * sr), int(silent_span[1] * sr)
        audio[:, i0:i1] = rng.normal(0.0, 1e-4, size=(2, i1 - i0)).astype(np.float32)
    if solo_span is not None:
        i0, i1 = int(solo_span[0] * sr), int(solo_span[1] * sr)
        audio[:, i0:i1] = rng.normal(0.0, 1e-4, size=(2, i1 - i0)).astype(np.float32)
        m = i1 - i0
        audio[solo_channel, i0:i1] += (rng.normal(0.0, 0.3, size=m)).astype(np.float32)
    return audio


def flip_from(audio, t0, sr=SR, t1=None):
    """Swap the channels over [t0, t1) -- the defect the repair exists to undo."""
    out = audio.copy()
    i0 = int(t0 * sr)
    i1 = audio.shape[1] if t1 is None else int(t1 * sr)
    out[0, i0:i1], out[1, i0:i1] = audio[1, i0:i1].copy(), audio[0, i0:i1].copy()
    return out


def frac_matching(a, b):
    """Fraction of samples where two (2,T) signals are identical on BOTH channels.

    Exact equality, not sign agreement: two independent noise signals agree on sign half the time,
    so a sign metric bottoms out at 0.75 for a half-episode flip and cannot separate "badly broken"
    from "slightly off". Exact equality goes to ~0.5 for a half flip and ~1.0 for a true repair
    (the only mismatches being the deliberate crossfade at each seam).
    """
    n = min(a.shape[1], b.shape[1])
    same0 = np.isclose(a[0, :n], b[0, :n], atol=1e-6)
    same1 = np.isclose(a[1, :n], b[1, :n], atol=1e-6)
    return float((same0 & same1).mean())


rng = np.random.default_rng(0)
turns = make_turns()
clean = render(turns, rng)

print("[1] helpers")
tt = top_two_speakers(turns)
check("top_two_speakers ranks by total time, longest first", tt == (A, B), f"got {tt}")
check("single-speaker episode has no anchor", top_two_speakers([{"speaker": A, "start": 0, "end": 5}]) is None)
env = channel_envelopes(clean, SR)
act = activity_tracks(turns, (A, B), env.shape[1])
check(
    "envelope frame count tracks FRAME_RATE",
    abs(env.shape[1] - DUR * FRAME_RATE) <= 1,
    f"{env.shape[1]} vs {DUR * FRAME_RATE}",
)
check("activity is non-degenerate for both speakers", act[0].sum() > 0 and act[1].sum() > 0)
starts, margins = window_margins(env, act)
check("clean audio scores positive margins", float(np.median(margins)) > 0.5, f"median {np.median(margins):.3f}")

print("[2] a clean episode must be left EXACTLY alone")
out, rep = repair_permutation(clean, SR, turns)
check("no spans flipped", rep["flipped_spans_frames"] == [], str(rep["flipped_spans_frames"]))
check("audio is bit-identical", np.array_equal(out, clean))
check("returns a copy, not the same object", out is not clean)

print("[3] a flip to the end of the episode is detected and undone")
broken = flip_from(clean, 150.0)
check("the fixture really is broken", frac_matching(broken, clean) < 0.6, f"match {frac_matching(broken, clean):.3f}")
fixed, rep = repair_permutation(broken, SR, turns)
check("a flip was found", len(rep["flipped_spans_frames"]) >= 1, str(rep["flipped_spans_frames"]))
check("audio is restored", frac_matching(fixed, clean) > 0.98, f"match {frac_matching(fixed, clean):.3f}")
# Non-vacuity: the same assertion must FAIL on the unrepaired signal, or it proves nothing.
check("...and would have failed without the repair", frac_matching(broken, clean) < 0.98)

print("[4] a mid-episode flip is localised, not applied to the whole file")
broken2 = flip_from(clean, 100.0, t1=200.0)
fixed2, rep2 = repair_permutation(broken2, SR, turns)
check("audio is restored", frac_matching(fixed2, clean) > 0.98, f"match {frac_matching(fixed2, clean):.3f}")
spans = rep2["flipped_spans_frames"]
check("exactly one span", len(spans) == 1, str(spans))
if spans:
    a_s, b_s = spans[0][0] / FRAME_RATE, spans[0][1] / FRAME_RATE
    check(
        "span lines up with the injected flip",
        abs(a_s - 100.0) < 25 and abs(b_s - 200.0) < 25,
        f"{a_s:.1f}-{b_s:.1f}s vs 100-200s",
    )

print("[5] dead air must not make it chatter (the ad-break case)")
turns_gap = [t for t in make_turns() if not (120.0 < t["start"] < 180.0)]
gapped = render(turns_gap, rng, silent_span=(120.0, 180.0))
out5, rep5 = repair_permutation(gapped, SR, turns_gap)
check("no flips invented across 60 s of silence", rep5["flipped_spans_frames"] == [], str(rep5["flipped_spans_frames"]))
check(
    "low-evidence windows are counted, not hidden",
    rep5["n_windows_low_evidence"] >= 1,
    str(rep5["n_windows_low_evidence"]),
)

print("[6] THE production failure: a flip that propagates out of an ad break")
# The realistic scenario. A 60 s solo ad read sits mid-episode; the separator assigns the lone voice
# arbitrarily, the chained _maybe_swap coin-flips at that boundary, and every chunk after it inherits
# the flip to the end of the episode. Only the anchored repair can undo that, because it is the only
# one that still knows who the speakers were BEFORE the ad.
turns_ad = [t for t in make_turns() if not (110.0 < t["start"] < 180.0)]
turns_ad.append({"speaker": A, "start": 120.0, "end": 175.0})
turns_ad.sort(key=lambda t: t["start"])
ad_clean = render(turns_ad, rng, solo_span=(120.0, 175.0), solo_channel=0)
ad_broken = flip_from(ad_clean, 180.0)
out6, rep6 = repair_permutation(ad_broken, SR, turns_ad)
check("the post-ad flip is undone", frac_matching(out6, ad_clean) > 0.98, f"match {frac_matching(out6, ad_clean):.3f}")
check(
    "...and the fixture really was broken",
    frac_matching(ad_broken, ad_clean) < 0.7,
    f"match {frac_matching(ad_broken, ad_clean):.3f}",
)
flipped_s = [(a / FRAME_RATE, b / FRAME_RATE) for a, b in rep6["flipped_spans_frames"]]
check(
    "exactly one flip, starting at the ad boundary",
    len(flipped_s) == 1 and abs(flipped_s[0][0] - 180.0) < 30,
    f"spans {flipped_s}",
)

print("[6b] the known limitation, asserted so it stays known")
# Inside a PURE solo region both activity tracks are constant -- A is always on, B always off -- so
# the correlation carries no information and the repair declines to act. That is correct rather than
# a gap: there is nothing to be right about when only one person is speaking, and DuplexChat's own
# filter drops single-speaker runs anyway (dialogue.py:58,63). It is asserted here so that a future
# change which starts "fixing" solo regions is caught and justified rather than assumed.
env6 = channel_envelopes(ad_clean, SR)
act6 = activity_tracks(turns_ad, (A, B), env6.shape[1])
st6, mg6 = window_margins(env6, act6)
solo_windows = [m for s, m in zip(st6, mg6) if 120.0 * FRAME_RATE < s < 175.0 * FRAME_RATE - 250]
check(
    "a window entirely inside the solo read carries no evidence",
    len(solo_windows) > 0 and max(abs(m) for m in solo_windows) < 0.2,
    f"margins {[round(m, 3) for m in solo_windows]}",
)

print("[7] the global convention: channel 0 ends up as the top speaker")
swapped_all = flip_from(clean, 0.0)
out7, rep7 = repair_permutation(swapped_all, SR, turns)
check(
    "a wholly-inverted episode is righted", frac_matching(out7, clean) > 0.98, f"match {frac_matching(out7, clean):.3f}"
)
check("anchor names the top speaker first", rep7["anchor_speakers"] == [A, B], str(rep7["anchor_speakers"]))

print("[8] Viterbi beats a per-window threshold on noisy margins")
noisy = np.concatenate([rng.normal(0.6, 0.5, 20), rng.normal(-0.6, 0.5, 20)])
# Explicit penalty: this case tests the ALGORITHM (smoothing vs chatter), not the tuned production
# constant, which is derived from MIN_FLIP_SECONDS and would reject a 20-window run by design.
st = viterbi_states(noisy, switch_penalty=5.0)
transitions = int((np.diff(st) != 0).sum())
naive = (noisy < 0).astype(int)
naive_transitions = int((np.diff(naive) != 0).sum())
check("Viterbi finds one change point", transitions == 1, f"got {transitions}")
check(
    "...where a threshold chatters",
    naive_transitions > transitions,
    f"threshold {naive_transitions} vs viterbi {transitions}",
)
check("each chatter would be a real audio swap", len(flip_spans(np.arange(len(naive)) * 125, naive, 10000)) > 1)

print("[9] degenerate inputs are refused loudly, not silently passed through")
solo_turns = [{"speaker": A, "start": 0.0, "end": 50.0}]
out9, rep9 = repair_permutation(clean, SR, solo_turns)
check("one speaker => skipped with a reason", rep9["skipped"] == "fewer_than_two_speakers", str(rep9["skipped"]))
check("...and the audio is untouched", np.array_equal(out9, clean))
try:
    repair_permutation(clean[0:1], SR, turns)
    check("mono input raises", False, "accepted (2,T)-only input")
except ValueError:
    check("mono input raises", True)

# --------------------------------------------------- externally supplied scores (`scores=`)
# The speaker-embedding scorer lives in the worker, because perm_repair is deliberately pure numpy
# -- no torch, no GPU, no I/O -- which is what lets both the worker and this guard import it. So the
# contract to protect is the handover: scores in, same Viterbi and span machinery out.
n_frames_t = int(round(clean.shape[1] * FRAME_RATE / SR))
starts_t = np.arange(0, n_frames_t - WINDOW_FRAMES, HOP_FRAMES, dtype=np.int64)
# a decisive "swapped" verdict over the middle third, "direct" elsewhere
marg_t = np.full(len(starts_t), 2.0)
lo, hi = len(starts_t) // 3, 2 * len(starts_t) // 3
marg_t[lo:hi] = -2.0
out_s, rep_s = repair_permutation(clean, SR, turns, scores=(starts_t, marg_t), score_window_frames=WINDOW_FRAMES)
check("external scores are used", rep_s.get("scoring") == "external", rep_s.get("scoring"))
check("...and they produce a flip", len(rep_s["flipped_spans_frames"]) == 1, rep_s["flipped_spans_frames"])
if rep_s["flipped_spans_frames"]:
    a, b = rep_s["flipped_spans_frames"][0]
    # Coverage, not exact edges: a window's decision is attributed to its CENTRE, so the span sits
    # half a window later than the first marked window's start. That is flip_spans' convention and
    # is tested on its own above; asserting edges here would pin the convention in two places.
    want = (int(starts_t[lo]), int(starts_t[hi]))
    ov = max(0, min(b, want[1]) - max(a, want[0]))
    check(
        "...covering the span the scores marked",
        ov >= 0.8 * (want[1] - want[0]) and (b - a) <= 1.5 * (want[1] - want[0]),
        f"{(a, b)} vs {want}, overlap {ov}",
    )
check("...and the audio really changed", not np.array_equal(out_s, clean))

# The default path must still say which scorer ran -- a number whose provenance is unrecorded is
# exactly how two corpora built with different scorers become indistinguishable later.
_, rep_e = repair_permutation(clean, SR, turns)
check("default path is labelled", rep_e.get("scoring") == "energy_vs_turn_activity", rep_e.get("scoring"))

# Non-vacuity: the SAME audio with all-direct scores must NOT flip, or the check above would pass
# on any input at all.
_, rep_d = repair_permutation(
    clean, SR, turns, scores=(starts_t, np.full(len(starts_t), 2.0)), score_window_frames=WINDOW_FRAMES
)
check("all-direct scores flip nothing", not rep_d["flipped_spans_frames"], rep_d["flipped_spans_frames"])

try:
    repair_permutation(clean, SR, turns, scores=(starts_t, marg_t[:-3]), score_window_frames=WINDOW_FRAMES)
    check("mismatched scores raise", False, "accepted ragged (starts, margins)")
except ValueError:
    check("mismatched scores raise", True)

# --------------------------------------------------- flip_spans must use the SCORER's window
# Boundaries are placed at each window CENTRE. flip_spans hardcoded WINDOW_FRAMES//2, which is right
# for the 20 s energy windows and WRONG for the 6 s windows the embedding scorer emits -- every seam
# landed 7 s late, i.e. 7 s of audio swapped onto the wrong side of the boundary, silently. Found by
# reading, before it ever ran in production.
st_w = np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.int64)
starts_w = np.arange(len(st_w), dtype=np.int64) * HOP_FRAMES
sp_big = flip_spans(starts_w, st_w, 10_000)
sp_small = flip_spans(starts_w, st_w, 10_000, window_frames=75)  # 6 s at 12.5 Hz
check("a shorter scorer window moves the seams", sp_big != sp_small, f"{sp_big} == {sp_small}")
check(
    "...by exactly the half-window difference",
    sp_big and sp_small and (sp_big[0][0] - sp_small[0][0]) == (WINDOW_FRAMES // 2 - 75 // 2),
    f"{sp_big} vs {sp_small}",
)
check(
    "default is unchanged",
    flip_spans(starts_w, st_w, 10_000) == flip_spans(starts_w, st_w, 10_000, window_frames=WINDOW_FRAMES),
)

# --------------------------------------------------- the two fixes must stay load-bearing
# Both were bugs in the embedding scorer that the standalone experiment could not see, because it
# used its own span logic and rescaled the penalty by hand. Neither is visible in a loss curve or an
# error: one misplaces every seam by ~7 s, the other makes the whole scorer a silent no-op.

# (1) SCALE. SWITCH_PENALTY is derived assuming a confident window scores ~2. Raw embedding margins
# are cosine differences -- measured p90 0.901 on a real episode. Unnormalised they must flip
# NOTHING; normalised to the derived units they must flip the marked span.
# Use the EMBEDDING geometry (6 s windows, 12 s hop) and a realistically-sized flip, not half the
# episode: a long enough run accumulates gain even at the raw scale, so a half-episode fixture would
# pass and prove nothing. 50 windows = 600 s, the length actually injected in the end-to-end test.
# Use the EMBEDDING geometry (6 s windows, 12 s hop) and make the inverted run a MINORITY island:
# if it is the majority the globally-correct answer really is "the whole episode is inverted", and
# the fixture proves nothing. 50 windows = 600 s, the length injected end-to-end.
hop_e = int(round(12.0 * FRAME_RATE))
win_e = int(round(6.0 * FRAME_RATE))
n_e = 300
starts_e = (np.arange(n_e) * hop_e).astype(np.int64)
marg_raw = np.full(n_e, 0.5)
marg_raw[100:150] = -0.5
_, rep_raw = repair_permutation(clean, SR, turns, scores=(starts_e, marg_raw), score_window_frames=win_e)
check(
    "raw-scale embedding margins flip NOTHING (the silent no-op)",
    not rep_raw["flipped_spans_frames"],
    rep_raw["flipped_spans_frames"],
)
_, rep_norm = repair_permutation(
    clean,
    SR,
    turns,
    scores=(starts_e, 2.0 * marg_raw / float(np.percentile(np.abs(marg_raw), 90))),
    score_window_frames=win_e,
)
check(
    "...and the SAME margins normalised by p90 do flip",
    len(rep_norm["flipped_spans_frames"]) == 1,
    rep_norm["flipped_spans_frames"],
)

# (2) GEOMETRY, through repair_permutation rather than flip_spans alone -- the plumbing is what
# production uses and it was untested.
_, rep_w20 = repair_permutation(clean, SR, turns, scores=(starts_t, marg_t), score_window_frames=WINDOW_FRAMES)
_, rep_w6 = repair_permutation(clean, SR, turns, scores=(starts_t, marg_t), score_window_frames=75)
check(
    "score_window_frames reaches the seams",
    rep_w20["flipped_spans_frames"] != rep_w6["flipped_spans_frames"],
    f"{rep_w20['flipped_spans_frames']} == {rep_w6['flipped_spans_frames']}",
)
check("...and is recorded", rep_w6.get("score_window_frames") == 75, rep_w6.get("score_window_frames"))
try:
    repair_permutation(clean, SR, turns, scores=(starts_t, marg_t))
    check("scores= without a window raises", False, "accepted")
except ValueError:
    check("scores= without a window raises", True)
try:
    repair_permutation(clean, SR, turns, score_window_frames=75)
    check("a window without scores= raises", False, "accepted -- would misplace energy seams")
except ValueError:
    check("a window without scores= raises", True)

# --------------------------------------------------- != 2 speakers
# The separator emits exactly two tracks, so assignment is a binary flip and the Viterbi has two
# states. Fewer than two speakers must be SKIPPED and SAID so -- silently returning unrepaired audio
# is how a permutation error becomes invisible. More than two is fine: the two longest-speaking are
# anchored, which is what the pilot episode (7 diarized labels) already does.
one = [t for t in turns if str(t["speaker"]) == str(turns[0]["speaker"])]
out1, rep1 = repair_permutation(clean, SR, one)
check(
    "one speaker -> skipped, not silently repaired",
    rep1.get("skipped") == "fewer_than_two_speakers",
    rep1.get("skipped"),
)
check("...and the audio is untouched", np.array_equal(out1, clean))
many = list(turns) + [
    {"speaker": "SPEAKER_X", "start": 1.0, "end": 2.0},
    {"speaker": "SPEAKER_Y", "start": 3.0, "end": 4.0},
]
_, repm = repair_permutation(clean, SR, many)
check(
    "4 speakers -> still anchors exactly two",
    repm.get("anchor_speakers") is not None and len(repm["anchor_speakers"]) == 2,
    repm.get("anchor_speakers"),
)
check(
    "...and the two it picks are the long ones",
    repm["anchor_speakers"] == rep_e["anchor_speakers"],
    f"{repm.get('anchor_speakers')} vs {rep_e.get('anchor_speakers')}",
)

print()
if fails:
    print(f"FAILED: {len(fails)} check(s): {fails}")
    sys.exit(1)
print("all checks passed")
