"""Guard: the podcast ingest's mimi encode (``podcast_ingest_main.encode_pair``) gives the REFERENCE codes.

WHY. Until 2026-09-22 the ingest encoded in STREAMING mode in 30 s chunks. Mimi's streaming attention
keeps a ring buffer of exactly ``context`` = 250 steps, so a multi-step call overwrites keys its own
earlier steps still need: those codes matched frame-by-frame inference encoding on only ~22% of
codebook-0 tokens (and turned digital silence into token 1263). ``mode="offline"`` fixes it.
Checked on a 150 s real-speech clip (several windows, so the lead-in is exercised):
  1. offline mode vs ONE offline encode of the whole clip: >= 99.9% of all tokens (30 s lead-in;
     the reference trainer encodes each window with no lead-in at all).
  2. offline mode vs frame-by-frame streaming (what inference does): >= 99% codebook-0 agreement.
  3. digital silence -> mostly token 1316 in offline mode (as every reference path; >= 90%), never
     the chunked path's 1263.
  4. the historical chunked mode FAILS 1 and 2 -- so this guard can fail.
  5. timing: offline vs chunked (reported, not asserted).
  6. RL codes path: ``RolloutGenerator._silence_tail_codes`` (the capture-window silence after pre-encoded user
     codes) is bit-identical to the PCM path's per-frame encode of the same zeros on a streaming_forever(G)
     mimi; the pre-2026-10-12 single multi-frame streaming encode is shown to fail it.

moshi_family venv, CPU, ~2 min:
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_mimi_encode.py'
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))
sys.path.insert(0, os.path.join(os.getcwd(), "recipe"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402

from moshi.models import loaders as ml  # noqa: E402
from moshi_family.podcast_ingest_main import MIMI_FRAME, encode_pair  # noqa: E402

K = ml.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16").get_mimi(device="cpu")
K.set_num_codebooks(8)

# 150 s of real speech: decoded JRE training rows, concatenated.
ds = load_from_disk("output/persona/p0_data")
pieces, total = [], 0
for i in range(200):
    r = ds[i]
    k, f = int(r["n_codebooks"]), int(r["n_frames"])
    with torch.no_grad():
        pieces.append(
            K.decode(torch.from_numpy(np.asarray(r["codes_assistant"], np.int64).reshape(k, f))[None])[0, 0].numpy()
        )
    total += pieces[-1].shape[0]
    if total >= 150 * 24000:
        break
x = np.concatenate(pieces)[: 150 * 24000].astype(np.float32)
y = np.roll(x, 24000 * 7)  # a different second channel
n = (x.shape[0] // MIMI_FRAME) * MIMI_FRAME

with torch.no_grad():
    whole = K.encode(torch.from_numpy(np.stack([x[:n], y[:n]])[:, None]))[:, :, : n // MIMI_FRAME].numpy()
    with K.streaming(1):
        fb = torch.cat(
            [K.encode(torch.from_numpy(x[s : s + MIMI_FRAME])[None, None]) for s in range(0, n, MIMI_FRAME)], -1
        )[0].numpy()

t0 = time.time()
off = encode_pair(K, x, y, device="cpu", chunk_sec=30.0, mode="offline")
t_off = time.time() - t0
t0 = time.time()
chk = encode_pair(K, x, y, device="cpu", chunk_sec=30.0, mode="chunked")
t_chk = time.time() - t0
F = n // MIMI_FRAME

assert off.shape[:2] == (2, 8) and off.shape[2] >= F, off.shape
w_agree = (off[:, :, :F] == whole).mean()
assert w_agree >= 0.999, f"offline windows vs one offline encode: {w_agree:.4f}"
print(f"[1] offline mode vs one offline encode of the whole {F / 12.5:.0f} s clip: {w_agree:.4%} of all tokens equal")
agree = (off[0, 0, :F] == fb[0, :F]).mean()
assert agree >= 0.99, agree
print(f"[2] offline mode vs frame-by-frame streaming: {agree:.3f} codebook-0 agreement")
z = encode_pair(
    K, np.zeros(24000 * 10, np.float32), np.zeros(24000 * 10, np.float32), device="cpu", chunk_sec=30.0, mode="offline"
)
zc = z[0, 0, 2:]
assert (zc == 1316).mean() >= 0.9 and not (zc == 1263).any(), np.unique(zc, return_counts=True)
print(f"[3] digital silence -> 1316 on {(zc == 1316).mean():.0%} of frames in offline mode, never 1263")
chk_whole = (chk[:, :, :F] == whole).mean()
chk_fb = (chk[0, 0, :F] == fb[0, :F]).mean()
assert chk_whole < 0.9 and chk_fb < 0.9, (chk_whole, chk_fb)  # far below [1]/[2]
print(f"[4] historical chunked mode fails: {chk_whole:.3f} vs one offline encode, {chk_fb:.3f} vs frame-by-frame")
print(
    f"[5] time for {F / 12.5:.0f} s x 2 channels on CPU: offline {t_off:.1f} s, chunked {t_chk:.1f} s ({t_off / t_chk:.2f}x)"
)


# [6] RL codes path (CodesSegmentSource -> RolloutGenerator): the cached capture-window silence tail must be
# the codes the PCM path feeds for the same zeros -- one frame per call from a fresh streaming state on the
# engine's streaming_forever(G) mimi. Until 2026-10-12 it was ONE multi-frame streaming encode (146/150
# frames wrong, 1263 on 17% of codebook-0); that implementation is kept below as the mutant.
def _old_silence_tail_codes(self, state):
    sr = int(state.mimi.sample_rate)
    sil = torch.zeros(self.num_samples, 1, int(round(self.capture_s * sr)), device=self.model.device)
    state.mimi.reset_streaming()
    return state.mimi.encode(sil)[0].detach()


def _tail_vs_pcm_path(tail_fn, g=4, capture_s=12.0):
    from types import SimpleNamespace

    from moshi_family.rl.generator import RolloutGenerator

    mimi = ml.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16").get_mimi(device="cpu")
    mimi.set_num_codebooks(8)
    mimi.streaming_forever(g)  # as InferenceState
    state = SimpleNamespace(mimi=mimi, frame_size=MIMI_FRAME)
    gen = RolloutGenerator(SimpleNamespace(device="cpu", state=state), None, num_samples=g, capture_s=capture_s)
    with torch.no_grad():
        tail = (tail_fn(gen, state) if tail_fn else gen._silence_tail_codes(state)).cpu()
        # The PCM path of RolloutGenerator.generate: reset, zeros split into whole frames, one encode per frame.
        z = torch.zeros(int(round(capture_s * 24000)))[None, None].expand(g, 1, -1).contiguous()
        mimi.reset_streaming()
        ref = torch.cat([mimi.encode(c) for c in z.split(MIMI_FRAME, dim=2) if c.shape[-1] == MIMI_FRAME], -1)
    assert torch.equal(ref, ref[:1].expand_as(ref)), "rows of the batch-G silence encode differ"
    return tail, ref[0]


tail, ref = _tail_vs_pcm_path(None)
assert tail.shape == ref.shape and torch.equal(tail, ref), (
    tail.shape,
    ref.shape,
    int((tail != ref).any(0).sum()) if tail.shape == ref.shape else None,
)
print(
    f"[6] RolloutGenerator._silence_tail_codes == the PCM path's per-frame encode of the same silence ({ref.shape[1]} frames, bit-identical)"
)
old_tail, ref = _tail_vs_pcm_path(_old_silence_tail_codes)
n_bad = int((old_tail[:, : ref.shape[1]] != ref).any(0).sum())
assert n_bad > ref.shape[1] // 2, n_bad
print(
    f"[6] CAN FAIL: the pre-2026-10-12 single multi-frame streaming encode differs on {n_bad}/{ref.shape[1]} frames "
    f"(codebook-0 1263 on {(old_tail[0] == 1263).float().mean():.0%})"
)
print("OK")
