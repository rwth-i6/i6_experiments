"""Packing must add SIGNAL, place every appended word correctly, and be a no-op when off."""

import sys, os

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
sys.path.insert(0, "recipe/speech_llm/full_duplex")
os.environ.setdefault("CUDA_HOME", "/usr")
import numpy as np
from moshi_family.moshi_train_data import pack_clips_into_window

fails = []


def ck(c, l, e=""):
    print(f"  [{'ok' if c else 'FAIL'}] {l}{'  ' + e if e else ''}")
    if not c:
        fails.append(l)


SR = 24000


class FakeCorpus:
    """Rows of a 2 s clip with one word at 0.5 s, so every appended word's offset is checkable."""

    rows = np.arange(5)

    def decode_row(self, idx):
        a = np.ones(2 * SR, dtype=np.float32) * 0.1
        u = np.ones(2 * SR, dtype=np.float32) * 0.2
        return a, u, SR, [("w%d" % idx, (0.5, 1.0), "assistant")]


rng = np.random.default_rng(0)
c = FakeCorpus()
a0 = np.ones(2 * SR, dtype=np.float32) * 0.1
u0 = np.ones(2 * SR, dtype=np.float32) * 0.2
al0 = [("first", (0.5, 1.0), "assistant")]

a, u, al = pack_clips_into_window(a0, u0, SR, al0, corpus=c, rng=rng, target_sec=20.0, gap_sec=(1.0, 1.0), max_extra=8)
ck(len(al) > len(al0), "packing added words", f"{len(al0)} -> {len(al)}")
ck(len(a) / SR > 15.0, "window is filled toward the target", f"{len(a) / SR:.1f}s")
ck(len(a) == len(u), "the two channels stay EXACTLY equal length (role seam intact)", f"{len(a)} vs {len(u)}")

# every appended word must sit inside the audio, in increasing order, and at a 3s stride
# (2s clip + 1s gap) given the fixed gap
starts = [s for _, (s, _), _ in al]
ck(starts == sorted(starts), "word onsets are in increasing time order")
ck(all(0 <= s < len(a) / SR for s in starts), "every word onset lies inside the packed audio")
expected = [0.5 + 3.0 * k for k in range(len(starts))]
ck(
    all(abs(x - y) < 1e-6 for x, y in zip(starts, expected)),
    "each appended clip is offset by (clip + gap) exactly",
    f"{[round(s, 2) for s in starts[:4]]}",
)

# a clip that already fills the window must be untouched
big_a = np.ones(25 * SR, dtype=np.float32)
big_u = np.ones(25 * SR, dtype=np.float32)
a2, u2, al2 = pack_clips_into_window(big_a, big_u, SR, al0, corpus=c, rng=rng, target_sec=20.0)
ck(len(a2) == len(big_a) and al2 == list(al0), "a full window is left alone")


# mixed sample rates must refuse rather than corrupt
class OtherSR(FakeCorpus):
    def decode_row(self, idx):
        return np.ones(SR, dtype=np.float32), np.ones(SR, dtype=np.float32), 8000, [("x", (0.1, 0.2), "assistant")]


a3, _, al3 = pack_clips_into_window(a0, u0, SR, al0, corpus=OtherSR(), rng=rng, target_sec=20.0)
ck(len(al3) == len(al0), "a different sample rate is refused, not resampled")

# Density actually improves -- the whole point. The denominator is the TRAINING WINDOW, not the
# clip: an unpacked 2 s clip still occupies the full 20 s window, and the model is trained on every
# frame of it. (A first version of this check divided by clip seconds and "failed" a working
# implementation, because appending a clip plus a gap lowers words-per-second-of-audio while
# raising words-per-window, which is the quantity that matters.)
TARGET = 20.0
words_before = len(al0) / TARGET
words_after = len(al) / TARGET
ck(
    words_after > words_before * 1.5,
    "word density PER WINDOW rises",
    f"{words_before:.2f} -> {words_after:.2f} words/s of window",
)
dead_before = 1.0 - (len(a0) / SR) / TARGET
dead_after = 1.0 - (len(a) / SR) / TARGET
ck(dead_after < dead_before, "dead air in the window falls", f"{100 * dead_before:.0f}% -> {100 * dead_after:.0f}%")
print("\nFAILURES:", fails or "none")
sys.exit(1 if fails else 0)
