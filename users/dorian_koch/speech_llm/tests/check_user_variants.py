"""Guard: precomputed user-stream augmentation variants, from the ingest encoder to the training loader.

WHY. A codes corpus cannot be augmented after the fact (a decode -> re-encode round trip alone changes
42% of the codes, E19), so the Moshi paper's user gain / noise / echo+reverb is computed at INGEST from
the real waveform (``podcast_mimi_encode.user_stream_variants``), stored next to the clean codes, and
drawn per read in the loaders (``train_data_common.draw_stored_user_stream``). Checked on real mimi codes:
  1. ENCODER. At the paper's probabilities, with noise from a pool in ``DnsNoisePool``'s format: every
     variant records at least one effect and differs from the clean codes, and the record carries the
     clean probability; all probabilities 1 -> every variant has all three effects and the variants
     differ from each other; the same seed reproduces the same variants; all probabilities 0 raises.
  2. SLICING. ``slice_codes`` on [V, K, F] == slicing each variant on its own (frames stay aligned).
  3. DRAW. Clean at the recorded clean probability, else uniform over V; ``side`` picks the right
     column; a row without variants raises.
  4. PERSONAPLEX LOADER, with channel swapping: every read's user stream is the clean codes or a variant
     of the side that is the user on that read, and the assistant target is the untouched stored channel.

moshi_family venv, CPU, ~3 min:
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_user_variants.py'
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))
sys.path.insert(0, os.path.join(os.getcwd(), "recipe"))

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset, load_from_disk  # noqa: E402

from moshi.models import loaders as ml  # noqa: E402
from moshi_family.podcast_ingest_main import encode_pair  # noqa: E402
from moshi_family.podcast_mimi_encode import user_stream_variants  # noqa: E402
from moshi_family.personaplex.train_data import (  # noqa: E402
    PersonaPlexDataConfig,
    PersonaPlexTokenizer,
    build_data_loader,
    system_prompt_prefix,
)
from moshi_family.podcast_slice_main import slice_codes  # noqa: E402
from moshi_family.train_data_common import draw_stored_user_stream, read_arrow_table  # noqa: E402

K = ml.CheckpointInfo.from_hf_repo("kyutai/moshiko-pytorch-bf16").get_mimi(device="cpu")
K.set_num_codebooks(8)

# Two real 20 s channels: decoded JRE training rows.
ds = load_from_disk("output/persona/p0_data")
chans = []
for i in range(2):
    r = ds[i]
    k, f = int(r["n_codebooks"]), int(r["n_frames"])
    with torch.no_grad():
        chans.append(
            K.decode(torch.from_numpy(np.asarray(r["codes_assistant"], np.int64).reshape(k, f))[None])[0, 0]
            .numpy()[: 24000 * 20]
            .astype(np.float32)
        )
a, b = chans


def encode(x, y):
    return encode_pair(K, x, y, device="cpu", chunk_sec=30.0, mode="offline")


clean = encode(a, b)
tmp = tempfile.mkdtemp(prefix="check_user_variants_")

# A two-clip noise pool in DnsNoisePool's on-disk format, so the encoder reads it through NoisePool.
pool_dir = os.path.join(tmp, "noise")
os.makedirs(pool_dir)
schema = pa.schema([("name", pa.string()), ("audio", pa.list_(pa.int16()))])
noise = [(np.random.default_rng(s).standard_normal(24000 * 5) * 3000).astype(np.int16) for s in (1, 2)]
with pa.OSFile(os.path.join(pool_dir, "part.arrow"), "wb") as sink, pa.ipc.new_file(sink, schema) as w:
    w.write_batch(pa.record_batch([pa.array(["n1", "n2"]), pa.array(noise, type=pa.list_(pa.int16()))], schema=schema))
with open(os.path.join(pool_dir, "meta.json"), "w") as fh:
    json.dump({"sample_rate": 24000, "n_clips": 2}, fh)


def variants(gain, noise, echo_reverb, seed=0):
    cfg = {
        "variants": 3,
        "seed": seed,
        "augment": {"gain_prob": gain, "noise_prob": noise, "echo_reverb_prob": echo_reverb},
        "noise_pool": pool_dir,
    }
    return user_stream_variants(a, b, encode=encode, cfg=cfg, item_id="ep#0")


EFFECTS = ("gain_db", "noise_db", "echo_scale")
vap, vbp, meta = variants(0.5, 0.3, 0.3)
assert vap.shape == (3,) + clean[0].shape, vap.shape
assert abs(meta["clean_prob"] - 0.5 * 0.7 * 0.7) < 1e-9, meta["clean_prob"]
for ch, vv, cc in (("a", vap, clean[0]), ("b", vbp, clean[1])):
    assert len(meta[ch]) == 3 and all(any(d[e] is not None for e in EFFECTS) for d in meta[ch]), meta[ch]
    assert all((vv[v] != cc).any() for v in range(3)), f"channel {ch}: a variant equals the clean codes"
va1, vb1, meta1 = variants(1.0, 1.0, 1.0)
assert all(all(d[e] is not None for e in EFFECTS) for d in meta1["a"] + meta1["b"]), meta1
assert all((va1[v] != clean[0]).mean() > 0.3 and (vb1[v] != clean[1]).mean() > 0.3 for v in range(3)), (
    "p=1 variant ~ clean"
)
assert (va1[0] != va1[1]).mean() > 0.3 and (va1[1] != va1[2]).mean() > 0.3, "variants are not independent draws"
va1b, _, meta1b = variants(1.0, 1.0, 1.0)
assert np.array_equal(va1, va1b) and meta1 == meta1b, "same seed, different variants"
try:
    variants(0.0, 0.0, 0.0)
    raise RuntimeError("variants with every probability 0 were accepted")
except ValueError:
    pass
print(
    "[1] paper probabilities: every variant has an effect and differs from clean; p=1 variants differ from clean "
    f"({(va1[0] != clean[0]).mean():.0%} of tokens) and from each other; reproducible; p=0 raises"
)

k, f = clean.shape[1], clean.shape[2]
f0, f1 = 40, 180
cut = slice_codes(va1.reshape(-1), k, f, f0, f1).reshape(3, k, f1 - f0)
assert all(np.array_equal(cut[v], slice_codes(va1[v].reshape(-1), k, f, f0, f1).reshape(k, f1 - f0)) for v in range(3))
assert np.array_equal(cut[0], va1[0][:, f0:f1])
print("[2] slice_codes keeps variant frames aligned with the clean frames")

CLEAN_PROB = meta["clean_prob"]
row = {
    "id": "ep#0@a#w0",
    "n_codebooks": k,
    "n_frames": f,
    "frame_rate": 12.5,
    "duration": f / 12.5,
    "codes_assistant": clean[0].reshape(-1),
    "codes_user": clean[1].reshape(-1),
    "codes_assistant_aug": va1.reshape(-1),
    "codes_user_aug": vb1.reshape(-1),
    "n_user_aug": 3,
    "user_aug_json": json.dumps({"clean_prob": CLEAN_PROB, "assistant": meta1["a"], "user": meta1["b"]}),
}
Dataset.from_list([row]).save_to_disk(os.path.join(tmp, "v"))
table = read_arrow_table(os.path.join(tmp, "v"))
rng = np.random.default_rng(0)
N = 4000
hits = [0, 0, 0, 0]  # clean, variant 0..2
clean_user = clean[1]  # one object: a clean draw returns it as is
for _ in range(N):
    u = draw_stored_user_stream(table, 0, clean_user, rng)
    hits[0 if u is clean_user else 1 + next(v for v in range(3) if np.array_equal(u, vb1[v]))] += 1
assert abs(hits[0] / N - CLEAN_PROB) < 0.03, hits
assert all(abs(h / N - (1 - CLEAN_PROB) / 3) < 0.03 for h in hits[1:]), hits
drawn = [draw_stored_user_stream(table, 0, clean[0], rng, side="assistant") for _ in range(20)]
assert any(any(np.array_equal(u, va1[v]) for v in range(3)) for u in drawn), "side='assistant' did not read its variants"
Dataset.from_list(
    [dict(row, codes_assistant_aug=np.zeros(0, np.int16), codes_user_aug=np.zeros(0, np.int16), n_user_aug=0)]
).save_to_disk(os.path.join(tmp, "none"))
try:
    draw_stored_user_stream(read_arrow_table(os.path.join(tmp, "none")), 0, clean[1], rng)
    raise RuntimeError("a row without variants was accepted")
except ValueError:
    pass
print(f"[3] draw: clean/variants {hits} (clean_prob {CLEAN_PROB:.3f}); side='assistant' reads its column; no variants -> error")

voice = np.random.default_rng(1).integers(0, 2048, size=(8, 50))
al = [
    {"text": w, "start": 0.5 + i * 0.6, "end": 0.8 + i * 0.6, "speaker": "assistant"}
    for i, w in enumerate("one two three four".split())
]
Dataset.from_list(
    [
        dict(
            row,
            alignments=al,
            context=["You enjoy having a good conversation."],
            voice_codes=voice.astype(np.int16).reshape(-1),
            context_other=["You enjoy having a good conversation."],
            voice_codes_other=voice.astype(np.int16).reshape(-1),
            alignments_other=al,
            other_ok=True,
        )
    ]
).save_to_disk(os.path.join(tmp, "pp"))
tok = PersonaPlexTokenizer(
    "nvidia/personaplex-7b-v1", device="cpu", cfg=PersonaPlexDataConfig(duration_sec=60, silence_frames=6)
)
P = system_prompt_prefix(
    tok.system_prompt_text_tokens("You enjoy having a good conversation."), voice, n_q=16, silence_frames=6
).shape[1]
it = build_data_loader(
    os.path.join(tmp, "pp"),
    tok,
    batch_size=1,
    seed=0,
    system_prompt_key="context",
    voice_codes_key="voice_codes",
    swap_channels=True,
    stored_user_augment=True,
)
seen = {"plain": 0, "swapped": 0}
for _ in range(60):
    c, _m = next(it)
    agent, user = c[0, 1:9, P:].numpy(), c[0, 9:17, P:].numpy()
    if np.array_equal(agent, clean[0][:, : agent.shape[1]]):  # plain read: user is user-side clean or variant
        assert any(np.array_equal(user, s[:, : user.shape[1]]) for s in (clean[1], *vb1)), (
            "plain read: user is not a user-side stream"
        )
        seen["plain"] += 1
    else:  # swapped: the assistant target is the stored USER channel, the user stream assistant-side
        assert np.array_equal(agent, clean[1][:, : agent.shape[1]]), (
            "assistant target is not an untouched stored channel"
        )
        assert any(np.array_equal(user, s[:, : user.shape[1]]) for s in (clean[0], *va1)), (
            "swapped read: user is not an assistant-side stream"
        )
        seen["swapped"] += 1
assert seen["plain"] > 10 and seen["swapped"] > 10, seen
print(
    f"[4] PersonaPlex loader: user stream = clean or a variant of the right side on every read {seen}; "
    "assistant targets untouched"
)
print("OK")
