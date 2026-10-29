"""Guard: the MTR pause window is the longest INTERNAL silence of a user turn (audit RAGMISC-02).

``mtr.conventions._detect_pause`` took the longest silent run anywhere and then rejected the whole turn if that run
touched an edge, so any recording whose leading or trailing silence was longer than its pause got an empty window
(1403 of 2000 pause rounds, all backends): the talk-over check then passed trivially.

  [1] speech 1.5 s, pause 1.0 s, speech 1.5 s, then 0.3 s / 1.2 s trailing or 1.2 s leading silence -> the pause;
  [2] no internal silence, or one shorter than pause_min_s -> (0, 0);
  [3] the eval's ``--redetect_pause`` path (eval_runner._redetect_pause) recovers the window from a stereo submission.
CAN FAIL: the pre-fix detector (embedded below) returns (0, 0) on the 1.2 s edge cases.

Run from the setup root, no GPU, ~1 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_mtr_pause_detect.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, "recipe/speech_llm/full_duplex")

from mtr import eval_runner  # noqa: E402
from mtr.conventions import TimelineConfig, _detect_pause  # noqa: E402

FAILS = []


def check(cond, msg):
    print(("[ok] " if cond else "[FAIL] ") + msg)
    if not cond:
        FAILS.append(msg)


def old_detect(turn, sr, thresh, min_s):
    win = max(1, int(0.02 * sr))
    silent = np.abs(turn) < thresh
    best, i, n = (0, 0), 0, len(turn)
    while i < n:
        if silent[i]:
            j = i
            while j < n and silent[j]:
                j += 1
            if (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j
        else:
            i += 1
    if best[0] <= win or best[1] >= n - win or (best[1] - best[0]) / sr < min_s:
        return 0.0, 0.0
    return best[0] / sr, best[1] / sr


cfg = TimelineConfig()
sr, th, mn = cfg.sr, cfg.pause_silence_thresh, cfg.pause_min_s
r = np.random.default_rng(0)


def sp(s):
    return (0.3 * r.standard_normal(int(s * sr)) + np.sign(r.standard_normal(int(s * sr))) * 0.2).astype(np.float32)


def z(s):
    return np.zeros(int(s * sr), np.float32)


cases = {
    "trailing 0.3 s": (np.concatenate([sp(1.5), z(1.0), sp(1.5), z(0.3)]), (1.5, 2.5)),
    "trailing 1.2 s": (np.concatenate([sp(1.5), z(1.0), sp(1.5), z(1.2)]), (1.5, 2.5)),
    "leading 1.2 s": (np.concatenate([z(1.2), sp(1.5), z(1.0), sp(1.5)]), (2.7, 3.7)),
}
for name, (turn, want) in cases.items():
    got = _detect_pause(turn, sr, th, mn)
    check(np.allclose(got, want, atol=0.01), f"[1] {name}: {tuple(round(x, 3) for x in got)} == {want}")
check(_detect_pause(np.concatenate([z(1.2), sp(3.0), z(1.2)]), sr, th, mn) == (0.0, 0.0), "[2] no internal pause")
check(
    _detect_pause(np.concatenate([sp(1), z(mn / 2), sp(1)]), sr, th, mn) == (0.0, 0.0), "[2] a pause under pause_min_s"
)
check(
    old_detect(cases["trailing 1.2 s"][0], sr, th, mn) == (0.0, 0.0),
    "CAN FAIL: the pre-fix detector loses the pause behind a 1.2 s trailing silence",
)

with tempfile.TemporaryDirectory() as d:
    user = np.concatenate([z(0.5), cases["trailing 1.2 s"][0], z(2.0)])
    wav = Path(d) / "dialogue_000_pause.wav"
    sf.write(wav, np.stack([user, np.zeros_like(user)], axis=1), sr, subtype="PCM_16")
    meta = [
        {"round": 0, "label": "pause", "user_start": 0.5, "user_end": 0.5 + 5.2, "pause_start": 0.0, "pause_end": 0.0}
    ]
    eval_runner._redetect_pause(meta, wav)
    check(
        abs(meta[0]["pause_start"] - 2.0) < 0.02 and abs(meta[0]["pause_end"] - 3.0) < 0.02,
        f"[3] --redetect_pause recovers the window from the submission: {meta[0]['pause_start']:.3f}-{meta[0]['pause_end']:.3f}",
    )

print("FAILED" if FAILS else "ALL CHECKS PASSED")
sys.exit(1 if FAILS else 0)
