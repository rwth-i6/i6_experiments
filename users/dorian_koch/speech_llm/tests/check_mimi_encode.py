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
print("OK")
