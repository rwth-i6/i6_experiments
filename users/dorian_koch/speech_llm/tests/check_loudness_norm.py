"""Guard: loudness normalisation before the mimi encode (``podcast_mimi_encode.apply_loudness_norm``).

WHY. Nothing in DuplexChat, moshi-finetune or our loaders normalises the separated audio, and after
the encode only codes exist -- so the gain has to be right the one time it is applied, and it must
never pump a near-empty channel up into noise. Checked (reason = the rule that set the gain):
  1. Channels at -30 and -14 LUFS land at the target (+-0.2 LU, re-measured) -- ONE gain for the
     whole clip: the sample-wise ratio output/input is constant.
  2. A quiet channel (-38 LUFS) is boosted by at most max_gain_db (reason max_gain).
  3. A channel below min_lufs (-50: bleed/noise, not speech) is left unchanged (below_min_lufs).
  4. A peaky channel is capped at LOUDNESS_PEAK_CAP (peak_cap); silence is untouched (silent).
  5. The two channels are independent.
PER-CHUNK MODE (``loudness_norm["mode"] = "per_chunk"``, 2026-09-23). The separator peak-normalises every
chunk, so a channel peaks at 0.99 in every chunk and its level moves chunk to chunk; the one-gain mode
above was a no-op on the pilot. Driven through ``apply_loudness_norm``, the function ``main`` calls:
  6. A synthetic 2-chunk episode stitched like DialogueSidon (linear cross-fade over the overlap),
     chunk 1 at -33 LUFS and chunk 2 at -21 LUFS, both peaking at 0.99: per_chunk brings BOTH chunks
     to -24 +-1 LU with no boosted sample above -1 dBFS (none above the input peak), a smooth gain
     across the overlap and a small limited share; the old per_channel mode FAILS the same check (non-vacuity: this is the pilot's bug).
  7. No mode key = the old per_channel record and audio, bit for bit; per_chunk without the separator
     grid in the provenance raises; a chunk below min_lufs is left at 0 dB, a silent chunk too; a tail
     chunk inside the previous cross-fade inherits the previous gain.
  8. The real ``main()`` on that episode (mimi on CPU, offline encode, one gain-only user variant):
     its clean codes AND its variant codes equal encoding ``apply_loudness_norm``'s output, and the
     recorded per-chunk output loudness is -24 +-1 LU.
Shown able to fail (2026-09-23): with ``apply_loudness_norm``'s per_chunk branch replaced by the
per_channel one in a copy of the module, [6] and [8] fail.

moshi_family venv (pyloudnorm), CPU, seconds:
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_loudness_norm.py'
"""

import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import pyloudnorm  # noqa: E402

from moshi_family import podcast_mimi_encode as pme  # noqa: E402
from moshi_family.podcast_mimi_encode import (  # noqa: E402
    LOUDNESS_PEAK_CAP,
    MIMI_SR,
    apply_loudness_norm,
    normalize_channel,
)

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

# ---- per-chunk mode ------------------------------------------------------------------------------
CHUNK, OVL = 120.0, 10.0
PROV = {"separator_chunk_sec": CHUNK, "separator_overlap_sec": OVL}
PER_CHUNK = {
    "mode": "per_chunk",
    "target_lufs": T,
    "max_gain_db": 10.0,
    "min_lufs": -45.0,
    "limiter_ceiling_dbfs": -1.0,
}
CEIL = 10 ** (-1.0 / 20)


def peaked(x, n_clicks=6):
    """Clicks at +-0.99, as every separated chunk peaks there (the separator's 0.9 peak rescale)."""
    x = x.copy()
    idx = rng.choice(x.size, n_clicks, replace=False)
    x[idx] = 0.99 * np.sign(rng.standard_normal(n_clicks))
    return x


def stitch(chunks, chunk_sec, ovl_sec):
    """DialogueSidon's stitch (separate.py): linear cross-fade of consecutive chunks over their overlap."""
    ov = int(ovl_sec * MIMI_SR)
    out = chunks[0]
    for c in chunks[1:]:
        fade = np.linspace(0.0, 1.0, ov, dtype=np.float32)
        out = np.concatenate([out[:-ov], out[-ov:] * (1 - fade) + c[:ov] * fade, c[ov:]])
    return out.astype(np.float32)


def two_chunk_episode(chunk_sec, ovl_sec, lufs=(-33.0, -21.0)):
    first = peaked(at_lufs(speechlike(chunk_sec), lufs[0]))
    second = peaked(at_lufs(speechlike(chunk_sec - ovl_sec), lufs[1]))
    return stitch([first, second], chunk_sec, ovl_sec)


def chunk_levels(y, chunk_sec, ovl_sec):
    """Loudness of each chunk's own samples (span minus the cross-fades)."""
    cores = pme.chunk_cores(pme.separator_chunks(y.size, MIMI_SR, chunk_sec, ovl_sec))
    return [meter.integrated_loudness(y[s:e].astype(np.float64)) for s, e in cores]


def per_chunk_ok(y, chunk_sec=CHUNK, ovl_sec=OVL, tol=1.0):
    lv = chunk_levels(y, chunk_sec, ovl_sec)
    return all(abs(v - T) <= tol for v in lv), lv


x = two_chunk_episode(CHUNK, OVL)
assert len(pme.separator_chunks(x.size, MIMI_SR, CHUNK, OVL)) == 2 and abs(float(np.abs(x).max()) - 0.99) < 1e-6
lv_in = chunk_levels(x, CHUNK, OVL)
assert abs(lv_in[1] - lv_in[0] - 12.0) < 1.0, lv_in
y, _, rec = apply_loudness_norm(x, x.copy(), PER_CHUNK, PROV)
ok, lv = per_chunk_ok(y)
info = rec["a"]
assert ok, f"per_chunk: chunk levels {lv} not -24 +-1 (in {lv_in}); {info}"
boosted = np.abs(y) > np.abs(x) * (1 + 1e-6)
assert float(np.abs(y[boosted]).max()) <= CEIL + 1e-6, f"boosted peak {np.abs(y[boosted]).max()} above -1 dBFS"
assert float(np.abs(y).max()) <= float(np.abs(x).max()) + 1e-6, "limiter let a sample exceed the input peak"
assert info["chunk_reason"] == ["target", "target"] and info["limited_share"] < 0.01, info
assert all(abs(o - T) <= 1.0 for o in info["chunk_lufs_out"]), info
# No gain step: away from limited samples the applied gain y/x moves by < 0.01 dB per sample.
ratio_db = 20 * np.log10(np.abs(y[np.abs(x) > 1e-3]) / np.abs(x[np.abs(x) > 1e-3]))
steps = np.abs(np.diff(ratio_db))
assert np.median(steps) < 1e-3 and np.percentile(steps, 99) < 0.05, (np.median(steps), np.percentile(steps, 99))
y_old, _, rec_old = apply_loudness_norm(
    x, x.copy(), {k: v for k, v in PER_CHUNK.items() if k not in ("mode", "limiter_ceiling_dbfs")}, PROV
)
ok_old, lv_old = per_chunk_ok(y_old)
assert not ok_old, f"non-vacuity: the old per_channel mode must fail this check, got {lv_old}"
print(
    f"[6] 2-chunk episode {lv_in[0]:.1f}/{lv_in[1]:.1f} LUFS, both peaking 0.99: per_chunk -> {lv[0]:.2f}/{lv[1]:.2f} "
    f"(gains {info['chunk_gain_db']} dB, limited {info['limited_share']:.4%}, max cut {info['max_reduction_db']} dB, "
    f"peak {np.abs(y).max():.3f}); old per_channel -> {lv_old[0]:.1f}/{lv_old[1]:.1f} "
    f"(gain {rec_old['a']['gain_db']} dB, {rec_old['a']['reason']}) FAILS the check, as it must"
)

cfg_old = {"target_lufs": T, "max_gain_db": 10.0, "min_lufs": -45.0}
ya, yb, rec0 = apply_loudness_norm(x, x.copy(), cfg_old, PROV)
ra, ia = normalize_channel(x, T, max_gain_db=10.0, min_lufs=-45.0)
assert np.array_equal(ya, ra) and rec0 == {**cfg_old, "a": ia, "b": ia}, rec0
try:
    apply_loudness_norm(x, x, PER_CHUNK, {})
    raise AssertionError("per_chunk without the separator grid must raise")
except RuntimeError:
    pass
quiet = two_chunk_episode(CHUNK, OVL, lufs=(-50.0, -30.0))
yq, _, rq = apply_loudness_norm(quiet, quiet.copy(), PER_CHUNK, PROV)
assert rq["a"]["chunk_reason"] == ["below_min_lufs", "target"] and rq["a"]["chunk_gain_db"][0] == 0.0, rq["a"]
core = slice(0, int((CHUNK - OVL) * MIMI_SR))  # chunk 1 outside the cross-fade: untouched
assert np.allclose(yq[core], quiet[core], atol=1e-7), "a chunk below min_lufs was changed"
sil = np.concatenate([np.zeros(int(CHUNK * MIMI_SR), np.float32), at_lufs(speechlike(CHUNK - OVL), -30.0)])
_, _, rs = apply_loudness_norm(sil, sil.copy(), PER_CHUNK, PROV)
assert rs["a"]["chunk_reason"][0] == "silent" and rs["a"]["chunk_gain_db"][0] == 0.0, rs["a"]
tail = np.concatenate([x, at_lufs(speechlike(5.0), -30.0)])  # 225 s: a 3rd chunk [220, 225) inside chunk 2's span
_, _, rt = apply_loudness_norm(tail, tail.copy(), PER_CHUNK, PROV)
assert rt["a"]["n_chunks"] == 3 and rt["a"]["chunk_reason"][2] == "short", rt["a"]
assert rt["a"]["chunk_gain_db"][2] == rt["a"]["chunk_gain_db"][1], "a tail chunk must inherit its predecessor's gain"
print(
    "[7] no mode = old record and audio bit for bit; missing grid raises; chunks below min_lufs / silent left at 0 dB; "
    "a tail chunk with no samples of its own inherits the previous gain"
)

# ---- the real main() ------------------------------------------------------------------------------
import json  # noqa: E402
import tempfile  # noqa: E402

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "recipe"))
from moshi.models import loaders as ml  # noqa: E402
from moshi_family.podcast_ingest_main import encode_pair  # noqa: E402

torch.manual_seed(0)
SC, SO = 12.0, 1.0  # a short grid (read from the provenance, like the real one), so mimi on CPU is quick
ea, eb = two_chunk_episode(SC, SO), two_chunk_episode(SC, SO, lufs=(-20.0, -31.0))
tmp = tempfile.mkdtemp(prefix="check_loudness_")
wav = os.path.join(tmp, "ep.wav")
sf.write(wav, np.stack([ea, eb], 1), MIMI_SR, subtype="FLOAT")
prov = {"separator_chunk_sec": SC, "separator_overlap_sec": SO}
row = {
    "item_id": "ep",
    "episode_id": "ep",
    "wav": wav,
    "span_start_sec": 0.0,
    "span_end_sec": ea.size / MIMI_SR,
    "duration_sec": ea.size / MIMI_SR,
    "provenance_json": json.dumps(prov),
}
man = os.path.join(tmp, "man.jsonl")
open(man, "w").write(json.dumps(row) + "\n")
aug = {"variants": 1, "seed": 0, "augment": {"gain_prob": 1.0}}
sys.argv = [
    "podcast_mimi_encode.py",
    "--manifest",
    man,
    "--out_part",
    os.path.join(tmp, "part"),
    "--row_mode",
    "episode",
    "--encode_mode",
    "offline",
    "--loudness_norm",
    json.dumps(PER_CHUNK),
    "--user_augment",
    json.dumps(aug),
]
pme.main()
(out,) = load_from_disk(os.path.join(tmp, "part"))
k, f = int(out["n_codebooks"]), int(out["n_frames"])
mimi = ml.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16").get_mimi(device="cpu")
mimi.eval()
na, nb, _ = apply_loudness_norm(ea, eb, PER_CHUNK, prov)


def enc(p, q):
    return encode_pair(mimi, p, q, device="cpu", chunk_sec=30.0, mode="offline")


want = enc(na, nb)
got = np.stack([np.asarray(out[f"codes_{c}"], np.int16).reshape(k, f) for c in "ab"])
assert np.array_equal(got, want), (
    f"clean codes are not the encode of the normalised audio ({(got != want).mean():.1%} differ)"
)
va, vb, _ = pme.user_stream_variants(na, nb, encode=enc, cfg=aug, item_id="ep")
assert np.array_equal(np.asarray(out["codes_a_aug"], np.int16).reshape(va.shape), va), (
    "variant a not from the normalised audio"
)
assert np.array_equal(np.asarray(out["codes_b_aug"], np.int16).reshape(vb.shape), vb), (
    "variant b not from the normalised audio"
)
raw = enc(ea, eb)
assert not np.array_equal(raw, got), "non-vacuity: encoding the un-normalised audio must differ"
loud = json.loads(out["channel_info"])["loudness"]
assert loud["mode"] == "per_chunk" and (loud["chunk_sec"], loud["overlap_sec"]) == (SC, SO), loud
for c in "ab":
    assert all(abs(v - T) <= 1.0 for v in loud[c]["chunk_lufs_out"]), (c, loud[c])
print(
    f"[8] real main(): clean + variant codes = encode(apply_loudness_norm(...)); recorded chunk LUFS out "
    f"a {loud['a']['chunk_lufs_out']} b {loud['b']['chunk_lufs_out']} (in a {loud['a']['chunk_lufs_in']} b {loud['b']['chunk_lufs_in']})"
)
print("OK")
