"""Guard: MFDB turn-taking metric v2 scores the reply to the turn end, not the model's earlier speech.

WHY (2026-09-28). The v1 scorer (the fork's ``eval_smooth_turn_taking.py``) takes the first non-greeting VAD segment
ANYWHERE in the clip as the response onset. The released PersonaPlex's mean latency of −3.01 s came mostly from that:
37 of its 117 onsets were speech that had ended before the user's turn end ("Let me know." at 0.5 s, mid-turn "Yeah."),
up to −84 s. v2 (``fdb_turn_taking_v2.py``, run by ``FDBTurnTakingRescore``) takes the first worded segment still going
at or after the turn end T, keeps a real overlap at T negative, and reports earlier speech as ``pre_turn``.

Checks, driving the real scorer ``eval_smooth_turn_taking_v2`` on real finished released-PersonaPlex clips:
  1. Its v1 columns equal the fork's own v1 scorer on the same clips (so v1 stays reproducible from the log).
  2. clip 102 (says "Let me know." at 0.50-1.80 s, T = 6.52): v1 −6.02, v2 −0.12, pre_turn but not long.
  3. clip 118 (reply "right? Or like the Pacific ..." starts 14.30, T = 15.07): v2 −0.77 — overlap at T stays negative.
  4. clip 23 ("Yeah, it can be pretty unsettling." mid-turn, silent at T): v1 TOR 1, v2 TOR 0, pre_turn_long.
  5. clip 103 (reply after T only): v1 == v2 = +0.07.
  6. ``score_clip`` on hand-made segments: a word-less blip after T is not the onset; speech ending exactly at T is
     pre-turn; the v1 duration/word threshold applies to the response segments only.
  Can fail: the v1 onset rule (``response_segments`` returning every segment) run through the same scorer is caught.

fdb_eval venv (silero-vad, soundfile, CPU torch), login node, ~10 s, from the setup root:
    output/venv/fdb_eval/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_fdb_turn_taking_v2.py
"""

import contextlib
import io
import json
import os
import shutil
import sys
import tempfile

ROOT = os.getcwd()
FORK_EVAL = os.path.join(ROOT, "recipe/moshified_fdb_v1_v15/evaluation")
AUDIO = os.path.realpath(os.path.join(ROOT, "output/fdb/personaplex_family_postport/candor_turn_taking/audio"))
sys.path.insert(0, os.path.join(ROOT, "recipe/i6_experiments/users/dorian_koch/speech_llm"))
sys.path.insert(0, FORK_EVAL)

import fdb_turn_taking_v2 as v2  # noqa: E402
from eval_smooth_turn_taking import eval_smooth_turn_taking  # noqa: E402  (the fork's v1 scorer)

CLIPS = ["102", "118", "23", "103"]


def close(a, b, tol=0.011):
    return a is not None and b is not None and abs(a - b) < tol


def seg(s, e, words):
    return {"start": s, "end": e, "words": words.split()}


def checks(mod, tmp):
    fails = []
    with contextlib.redirect_stdout(io.StringIO()):
        res = mod.eval_smooth_turn_taking_v2(tmp, FORK_EVAL)
        old = eval_smooth_turn_taking(tmp)
    per = {}
    # per-clip values straight from the scorer's own functions
    for c in CLIPS:
        T = json.load(open(os.path.join(tmp, c, "turn_taking.json")))[0]["timestamp"][0]
        per[c] = mod.score_clip(T, mod.clip_segments(os.path.join(tmp, c), FORK_EVAL))
    # [1] v1 columns == the fork's v1 scorer
    if not (
        close(res["v1_average_latency"], old["average_latency"], 1e-9)
        and res["v1_average_take_turn"] == old["average_take_turn"]
    ):
        fails.append(f"[1] v1 columns {res['v1_average_latency']}/{res['v1_average_take_turn']} != fork v1 {old}")
    p = per["102"]
    if not (close(p["latency_v1"], -6.02) and close(p["latency"], -0.12) and p["pre_turn"] and not p["pre_turn_long"]):
        fails.append(f"[2] clip 102 {p}")
    p = per["118"]
    if not (close(p["latency"], -0.77) and p["tor"] == 1):
        fails.append(f"[3] clip 118 {p}")
    p = per["23"]
    if not (p["tor_v1"] == 1 and p["tor"] == 0 and p["latency"] is None and p["pre_turn_long"]):
        fails.append(f"[4] clip 23 {p}")
    p = per["103"]
    if not (close(p["latency"], 0.07) and close(p["latency_v1"], 0.07) and not p["pre_turn"]):
        fails.append(f"[5] clip 103 {p}")
    # [6] hand-made segments, T = 5.0
    r = mod.score_clip(5.0, [seg(1.0, 2.0, "yeah"), seg(5.2, 5.6, ""), seg(6.0, 8.0, "so what I think is")])
    if not (close(r["latency"], 1.0) and r["pre_turn"] and not r["pre_turn_long"]):
        fails.append(f"[6a] word-less blip / pre-turn {r}")
    r = mod.score_clip(5.0, [seg(3.0, 5.0, "I think that is fine"), seg(5.5, 5.9, "ok")])
    if not (r["tor"] == 0 and r["pre_turn_long"] and r["tor_v1"] == 1):
        fails.append(f"[6b] ends-at-T is pre-turn, threshold on response only {r}")
    r = mod.score_clip(5.0, [seg(2.0, 9.0, "and then we went all the way down there")])
    if not (close(r["latency"], -3.0) and not r["pre_turn"]):
        fails.append(f"[6c] overlap at T {r}")
    return fails


def main():
    with tempfile.TemporaryDirectory(prefix="check_tt2_") as tmp:
        for c in CLIPS:
            shutil.copytree(os.path.join(AUDIO, c), os.path.join(tmp, c))
        fails = checks(v2, tmp)
        if fails:
            print("FAIL\n" + "\n".join(fails))
            sys.exit(1)
        print("real scorer: all checks pass")
        orig = v2.response_segments
        v2.response_segments = lambda turn_end, segments: list(segments)  # the v1 onset rule
        try:
            mfails = checks(v2, tmp)
        finally:
            v2.response_segments = orig
        if not mfails:
            print("FAIL: the v1-onset mutant passed every check")
            sys.exit(1)
        print(f"mutant (v1 onset rule) caught by {len(mfails)} checks: {[f.split(']')[0] + ']' for f in mfails]}")
        print("OK")


if __name__ == "__main__":
    main()
