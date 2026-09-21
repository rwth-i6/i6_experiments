"""Guard: offline window selection (`select_windows`) and the converter's windowed path.

1. `select_windows` on hand-built turn lists whose answer is known by construction:
   alternating 5 s turns over 100 s tile as [0,60) + [60,100); a 40 s monologue cannot sit inside
   any window; a window with fewer than 3 turns is refused; a row shorter than min_sec yields none.
2. The REAL `PodcastCodesTrainData.run()` on a synthetic dialogue part whose codes encode their own
   position (value = 100 * codebook + frame): every windowed row must hold exactly the frame slice
   of every codebook, and every alignment must be re-based to the window start. The same check is
   run against a deliberately wrong flat slice to show it can fail.
3. `window_select=None` still yields one untouched row per dialogue (the a49 path).

Usage (setup root): CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_window_select.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")

import numpy as np  # noqa: E402
from datasets import Dataset, load_from_disk  # noqa: E402
from sisyphus import tk  # noqa: E402

from i6_experiments.users.dorian_koch.speech_llm.podcast_ingest import (  # noqa: E402
    PodcastCodesTrainData,
    select_windows,
)

RULE = dict(min_sec=20.0, max_sec=60.0, min_turns=3, max_stretch_sec=30.0, turn_min_sec=1.0)


def alt(n, step, spk=("S0", "S1")):
    return [{"speaker": spk[i % 2], "start": i * step, "end": (i + 1) * step} for i in range(n)]


# ---- 1. select_windows ---------------------------------------------------------------------
w = select_windows(alt(20, 5.0), 100.0, **RULE)
assert w == [(0.0, 60.0), (60.0, 100.0)], w
mono = [
    {"speaker": "S0", "start": 0, "end": 10},
    {"speaker": "S1", "start": 10, "end": 20},
    {"speaker": "S0", "start": 20, "end": 60},
    {"speaker": "S1", "start": 60, "end": 70},
    {"speaker": "S0", "start": 70, "end": 80},
    {"speaker": "S1", "start": 80, "end": 90},
]
w = select_windows(mono, 90.0, **RULE)
assert w == [(60.0, 90.0)], w
assert all(not (a < 60 and b > 20 and min(b, 60) - max(a, 20) > 30) for a, b in w), (
    "a 40 s stretch leaked into a window"
)
assert select_windows(alt(2, 15.0), 30.0, **RULE) == [], "2 turns must not make a window"
assert select_windows(alt(4, 4.0), 16.0, **RULE) == [], "row shorter than min_sec"
# sub-1 s interjections do not count as turns
blips = [
    {"speaker": "S0", "start": 0, "end": 30},
    {"speaker": "S1", "start": 10, "end": 10.5},
    {"speaker": "S0", "start": 30, "end": 40},
]
assert select_windows(blips, 40.0, **RULE) == [], "a 0.5 s blip counted as a turn"
print("[1] select_windows: tiling, stretch cap, turn floor, min length, blip handling OK")

# ---- 2 + 3. the real converter -------------------------------------------------------------
K, FR, DUR = 8, 12.5, 100.0
F = int(DUR * FR)
codes = (np.arange(K)[:, None] * 100 + np.arange(F)[None, :]).astype(np.int16)  # value = 100k + f
turns = alt(20, 5.0)
words_a = [{"text": f"a{i}", "start": i * 1.0 + 0.1, "end": i * 1.0 + 0.5, "speaker": "S0"} for i in range(100)]
row = {
    "item_id": "ep#0",
    "episode_id": "ep",
    "duration_sec": DUR,
    "n_codebooks": K,
    "n_frames": F,
    "frame_rate": FR,
    "codes_a": codes.reshape(-1).tolist(),
    "codes_b": (codes + 7).reshape(-1).tolist(),
    "words_a": json.dumps(words_a),
    "words_b": json.dumps(words_a),
    "turns_json": json.dumps(turns),
}


def run_conv(window_select):
    tmp = tempfile.mkdtemp(prefix="check_window_select_")
    Dataset.from_list([row]).save_to_disk(os.path.join(tmp, "in", "parts", "part_000"))
    job = PodcastCodesTrainData(
        shard_dirs=[tk.Path(os.path.join(tmp, "in"))], assistant_channel="a", window_select=window_select
    )
    job.out_dir = tk.Path(os.path.join(tmp, "out"))
    cwd = os.getcwd()
    os.chdir(tmp)
    try:
        job.run()
    finally:
        os.chdir(cwd)
    return load_from_disk(os.path.join(tmp, "out"))


out = run_conv(RULE)
assert len(out) == 2, [r["id"] for r in out]
spans = select_windows(turns, DUR, **RULE)


def expect(a, b, flat=False):
    f0, f1 = int(np.floor(a * FR)), min(F, int(np.ceil(b * FR)))
    if flat:  # the WRONG slice: a contiguous run of the flat array
        return codes.reshape(-1)[f0 * K : f1 * K], f0
    return codes[:, f0:f1].reshape(-1), f0


for r, (a, b) in zip(out, spans):
    got = np.asarray(r["codes_assistant"], dtype=np.int16)
    want, f0 = expect(a, b)
    assert r["n_frames"] * K == len(got), "length"
    assert np.array_equal(got, want), f"{r['id']}: codes are not the codebook-major frame slice"
    assert np.array_equal(np.asarray(r["codes_user"]), want + 7), "user codes"
    bad, _ = expect(a, b, flat=True)
    assert not np.array_equal(got, bad), "check is vacuous: flat slice equals the correct one"
    t0 = f0 / FR
    for al in r["alignments"]:
        i = int(al["text"][1:])
        assert abs(al["start"] - (words_a[i]["start"] - t0)) < 1e-4, "alignment not re-based"
        assert a - 1e-6 <= words_a[i]["start"] < b + 1e-6
print(
    "[2] converter windowed path: codebook-major slices, user codes, re-based alignments OK "
    "(a flat slice would have been caught)"
)

plain = run_conv(None)
assert len(plain) == 1 and plain[0]["id"] == "ep#0@a" and plain[0]["n_frames"] == F
assert np.array_equal(np.asarray(plain[0]["codes_assistant"]), codes.reshape(-1))
assert len(plain[0]["alignments"]) == len(words_a)
print("[3] window_select=None: one untouched row per dialogue OK")
print("OK")
