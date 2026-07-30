"""Guard the in-loader Fisher windowing (train_data_common.slice_random_window): window length,
alignment re-basing, agent-speech seeking, short-row passthrough, and seed determinism.

Why this exists: moving windowing from the offline job into the loader means a slicing bug now
corrupts training silently -- a window whose audio and re-based alignments disagree trains the text
stream against the wrong frames, and a loss curve won't show it. So pin the slice's numerics.

Run:  CUDA_HOME=/usr PYTHONPATH=recipe:recipe/sisyphus .venv/bin/python \
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_windowed_loader.py
"""

import sys

import numpy as np

sys.path.insert(0, "recipe/speech_llm/full_duplex")
from moshi_family.train_data_common import slice_random_window  # noqa: E402

SR = 8000
FULL = 120.0
n = int(FULL * SR)
rngdata = np.random.default_rng(7)
assistant = (rngdata.standard_normal(n) * 0.1).astype(np.float32)
user = (rngdata.standard_normal(n) * 0.1).astype(np.float32)
# a word every 2 s, 0.5 s long, across the whole conversation (dense, like Fisher agent speech)
alignments = [(f"w{i}", (float(i * 2), float(i * 2) + 0.5), "assistant") for i in range(60)]

# --- 1. window length + alignment re-basing --------------------------------------------------
rng = np.random.default_rng(0)
WIN = 20.0
for _ in range(200):
    a, u, al = slice_random_window(assistant, user, SR, alignments, window_sec=WIN, rng=rng)
    assert abs(len(a) - int(WIN * SR)) <= 1, f"window audio len {len(a)} != {int(WIN * SR)}"
    assert len(u) == len(a), "channels must stay equal length"
    for _w, (s, e), _spk in al:
        # every kept word overlaps the window; times are re-based to the window start.
        assert e > -1e-6 and s < WIN + 1e-6, f"word not overlapping window after rebase: {(s, e)}"
    assert len(al) > 0, "a 20 s window over 2 s-spaced words must contain speech"
    assert 8 <= len(al) <= 12, f"expected ~10 words in a 20 s window, got {len(al)}"
print("[ok] window length + alignment re-basing")

# --- 2. rows already <= window are returned unchanged -----------------------------------------
short_n = int(10 * SR)
a, u, al = slice_random_window(
    assistant[:short_n], user[:short_n], SR, alignments[:5], window_sec=20.0, rng=rng
)
assert len(a) == short_n and len(u) == short_n, "a sub-window row must pass through unchanged"
assert len(al) == 5, "short-row alignments must pass through unchanged"
print("[ok] short rows pass through unchanged")

# --- 3. seed determinism ----------------------------------------------------------------------
a1, _, al1 = slice_random_window(assistant, user, SR, alignments, window_sec=15.0, rng=np.random.default_rng(42))
a2, _, al2 = slice_random_window(assistant, user, SR, alignments, window_sec=15.0, rng=np.random.default_rng(42))
assert np.array_equal(a1, a2), "same seed must cut the same window"
assert al1 == al2
print("[ok] seeded slicing is deterministic")

# --- 4. agent-speech seeking prefers a speech-bearing window ----------------------------------
# Speech only in the first 30 s of a 120 s conversation; over many draws with a high speech floor,
# the seeker should land a speech window far more often than the ~25% a single random try would.
sparse = [(f"w{i}", (float(i * 1.5), float(i * 1.5) + 0.4), "assistant") for i in range(20)]  # 0..~30 s
hits = 0
rng4 = np.random.default_rng(3)
for _ in range(60):
    _a, _u, al = slice_random_window(
        assistant, user, SR, sparse, window_sec=20.0, rng=rng4, min_agent_speech_sec=1.5, max_tries=12
    )
    hits += len(al) > 0
assert hits >= 45, f"speech-seeking landed a speech window only {hits}/60 times"
print(f"[ok] agent-speech seeking found speech in {hits}/60 windows")

print("\nALL CHECKS PASSED")
