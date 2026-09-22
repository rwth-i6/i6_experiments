"""Guard: per-channel loudness normalisation before the mimi encode (``podcast_mimi_encode.normalize_channel``).

WHY. Nothing in DuplexChat, moshi-finetune or our loaders normalises the separated audio, and after
the encode only codes exist -- so the gain has to be right the one time it is applied, and it must
never pump a near-empty channel up into noise. Checked (reason = the rule that set the gain):
  1. Channels at -30 and -14 LUFS land at the target (+-0.2 LU, re-measured) -- ONE gain for the
     whole clip: the sample-wise ratio output/input is constant.
  2. A quiet channel (-38 LUFS) is boosted by at most max_gain_db (reason max_gain).
  3. A channel below min_lufs (-50: bleed/noise, not speech) is left unchanged (below_min_lufs).
  4. A peaky channel is capped at LOUDNESS_PEAK_CAP (peak_cap); silence is untouched (silent).
  5. The two channels are independent.

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
    return pyloudnorm.normalize.loudness(x.astype(np.float64), meter.integrated_loudness(x.astype(np.float64)), lufs).astype(np.float32)


T = -24.0
for start in (-30.0, -14.0):
    x = at_lufs(speechlike(), start)
    y, info = normalize_channel(x, T)
    got = meter.integrated_loudness(y.astype(np.float64))
    assert abs(got - T) < 0.2 and info["reason"] == "target", (start, got, info)
    assert abs(info["gain_db"] - (T - start)) < 0.2, info
    nz = np.abs(x) > 1e-6
    ratio = y[nz] / x[nz]
    assert np.allclose(ratio, ratio[0], rtol=1e-5), "gain varies along the clip"
print("[1] -30 and -14 LUFS channels land at -24 +-0.2 with one constant gain")

x = at_lufs(speechlike(), -38.0)
y, info = normalize_channel(x, T, max_gain_db=10.0)
assert info["reason"] == "max_gain" and info["gain_db"] == 10.0, info
assert abs(meter.integrated_loudness(y.astype(np.float64)) - (-28.0)) < 0.2
print("[2] -38 LUFS channel boosted by the +10 dB cap only (-> -28), not to -24")

x = at_lufs(speechlike(), -50.0)
y, info = normalize_channel(x, T, min_lufs=-45.0)
assert np.array_equal(y, x) and info["reason"] == "below_min_lufs" and info["gain_db"] == 0.0, info
print("[3] -50 LUFS channel (below the -45 floor) left unchanged")

p = at_lufs(speechlike(), -30.0)
p[1000] = 0.9
y, info = normalize_channel(p, T)
assert info["reason"] == "peak_cap" and abs(float(np.abs(y).max()) - LOUDNESS_PEAK_CAP) < 1e-5, info
z = np.zeros(10 * MIMI_SR, np.float32)
y, info = normalize_channel(z, T)
assert np.array_equal(y, z) and info["reason"] == "silent" and info["gain_db"] == 0.0, info
print(f"[4] peaky channel capped at {LOUDNESS_PEAK_CAP}; silence untouched")

a, b = at_lufs(speechlike(), -30.0), at_lufs(speechlike(), -20.0)
ya, _ = normalize_channel(a, T)
yb, _ = normalize_channel(b, T)
assert abs(meter.integrated_loudness(ya.astype(np.float64)) - meter.integrated_loudness(yb.astype(np.float64))) < 0.3
print("[5] channels normalised independently to the same level")
print("OK")
