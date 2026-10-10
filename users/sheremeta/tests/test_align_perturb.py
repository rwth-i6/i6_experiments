"""Perturbation must hit only the real speech, not the delay pad. Run: python3 tests/test_align_perturb.py"""

import math

import torch

from i6_experiments.users.sheremeta.alignment import transforms as tr

SR = 16000
PAD = 43 * 960  # (delay_frames+1) * frame_stride_samples for the 2.5s / 16.67Hz grid


def _snr_over_speech(out, speech):
    noise = out[: speech.shape[-1]] - speech
    return 10 * math.log10(float((speech**2).mean()) / float((noise**2).mean()))


def test_speed_leaves_pad_fixed():
    speech = 0.1 * torch.randn(SR * 4)
    padded = torch.cat([speech, torch.zeros(PAD)])
    out = tr.apply_to_unpadded(padded, SR, pad_samples=PAD, transform=tr.speed(1.05))
    assert int((out[-PAD:] != 0).sum()) == 0  # trailing pad is fresh zeros, exactly PAD long
    real_len = out.shape[-1] - PAD
    assert abs(real_len - speech.shape[-1] / 1.05) < 0.02 * speech.shape[-1]


def test_noise_snr_over_speech_only():
    speech = 0.1 * torch.randn(SR * 4)
    padded = torch.cat([speech, torch.zeros(PAD)])
    fixed = tr.apply_to_unpadded(padded, SR, pad_samples=PAD, transform=tr.additive_noise(20.0))
    assert int((fixed[-PAD:] != 0).sum()) == 0
    assert abs(_snr_over_speech(fixed, speech) - 20.0) < 1.5
    # old path (whole padded waveform) dilutes signal power over the pad, so effective SNR is higher
    buggy = tr.additive_noise(20.0).apply(padded, SR)
    assert _snr_over_speech(buggy, speech) > _snr_over_speech(fixed, speech) + 1.0


def test_no_pad_is_plain_apply():
    x = 0.1 * torch.randn(SR)
    out = tr.apply_to_unpadded(x, SR, pad_samples=0, transform=tr.gain(0.7))
    assert torch.allclose(out, (x * 0.7).clamp(-1, 1))


if __name__ == "__main__":
    test_speed_leaves_pad_fixed()
    test_noise_snr_over_speech_only()
    test_no_pad_is_plain_apply()
    print("align perturb tests OK")
