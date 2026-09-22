"""Guard: per-channel loudness normalisation before the mimi encode (``podcast_mimi_encode.normalize_channel``).

WHY. Nothing in DuplexChat, moshi-finetune or our loaders normalises the separated audio, and after
the encode only codes exist -- so the gain has to be right the one time it is applied. Checked:
  1. A quiet channel (-38 LUFS) and a loud one (-14 LUFS) both land at the target (+-0.2 LU,
     re-measured with the same meter), and the recorded ``gain_db`` is what was applied.
  2. A peaky channel whose target gain would push it past 1.0 is capped at LOUDNESS_PEAK_CAP and
     says ``capped``.
  3. Silence (no gated loudness) is returned unchanged with gain 0 dB -- not blown up to the target.
  4. The two channels are independent: normalising A does not depend on B.

moshi_family venv (pyloudnorm), CPU, seconds:
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_loudness_norm.py'
"""

import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import pyloudnorm  # noqa: E402

from moshi_family.podcast_mimi_encode import LOUDNESS_PEAK_CAP, MIMI_SR, normalize_channel  # noqa: E402

rng = np.random.default_rng(0)
meter = pyloudnorm.Meter(MIMI_SR)


def speechlike(sec=20.0):
    """Noise bursts with pauses: gated loudness behaves as on speech (silence gated out)."""
    n = int(sec * MIMI_SR)
    env = np.repeat(rng.random(int(sec * 4)) > 0.4, n // int(sec * 4) + 1)[:n]
    return (rng.normal(0, 1, n) * env).astype(np.float32)


def at_lufs(x, lufs):
    return pyloudnorm.normalize.loudness(
        x.astype(np.float64), meter.integrated_loudness(x.astype(np.float64)), lufs
    ).astype(np.float32)


T = -24.0
for start in (-38.0, -14.0):
    x = at_lufs(speechlike(), start)
    y, info = normalize_channel(x, T)
    got = meter.integrated_loudness(y.astype(np.float64))
    assert abs(got - T) < 0.2 and not info["capped"], (start, got, info)
    assert abs(info["gain_db"] - (T - start)) < 0.2, info
print("[1] -38 and -14 LUFS channels land at -24 +-0.2; gain_db recorded")

p = at_lufs(speechlike(), -40.0)
p[1000] = 0.9  # one sharp peak: +16 dB would take it far past 1.0
y, info = normalize_channel(p, T)
assert info["capped"] and abs(float(np.abs(y).max()) - LOUDNESS_PEAK_CAP) < 1e-5, info
print(f"[2] peaky channel capped at {LOUDNESS_PEAK_CAP} (gain {info['gain_db']} dB instead of +16)")

z = np.zeros(10 * MIMI_SR, np.float32)
y, info = normalize_channel(z, T)
assert np.array_equal(y, z) and info["gain_db"] == 0.0 and info["lufs_in"] is None, info
print("[3] silence unchanged, gain 0 dB")

a, b = at_lufs(speechlike(), -30.0), at_lufs(speechlike(), -20.0)
ya, _ = normalize_channel(a, T)
ya2, _ = normalize_channel(a, T)
assert np.array_equal(ya, ya2)
yb, _ = normalize_channel(b, T)
assert abs(meter.integrated_loudness(ya.astype(np.float64)) - meter.integrated_loudness(yb.astype(np.float64))) < 0.3
print("[4] channels normalised independently to the same level")
print("OK")
