"""Guard: precomputed user-stream augmentation variants, from the ingest encoder to the training loader.

WHY. A codes corpus cannot be augmented after the fact (a decode -> re-encode round trip alone changes
42% of the codes, E19), so the Moshi paper's user gain / noise / echo+reverb is computed at INGEST from
the real waveform (``podcast_mimi_encode.user_stream_variants``), stored next to the clean codes, and
drawn per read in the loaders (``train_data_common.draw_user_variant``). Checked on real mimi codes:
  1. ENCODER. All probabilities 0 -> every variant == the clean codes, token for token (the variant path
     IS the clean encode); all probabilities 1 -> every variant differs from the clean codes and the
     variants differ from each other; the same seed reproduces the same variants.
  2. SLICING. ``slice_codes`` on [V, K, F] == slicing each variant on its own (frames stay aligned).
  3. DRAW. Uniform over V; ``side`` picks the right column; a row without variants raises.
  4. PERSONAPLEX LOADER, with channel swapping: every read's user stream equals a variant of the side
     that is the user on that read, and the assistant target is the untouched stored channel.

moshi_family venv, CPU, ~2 min:
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_user_variants.py'
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))
sys.path.insert(0, os.path.join(os.getcwd(), "recipe"))

import numpy as np  # noqa: E402
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
from moshi_family.train_data_common import draw_user_variant, read_arrow_table  # noqa: E402

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


def variants(p, seed=0):
    cfg = {"variants": 3, "seed": seed, "augment": {"gain_prob": p, "noise_prob": p, "echo_reverb_prob": p}}
    return user_stream_variants(a, b, encode=encode, cfg=cfg, item_id="ep#0")


va0, vb0 = variants(0.0)
assert va0.shape == (3,) + clean[0].shape, va0.shape
assert all(np.array_equal(va0[v], clean[0]) and np.array_equal(vb0[v], clean[1]) for v in range(3)), (
    "p=0 variants != clean"
)
va1, vb1 = variants(1.0)
assert all((va1[v] != clean[0]).mean() > 0.3 and (vb1[v] != clean[1]).mean() > 0.3 for v in range(3)), (
    "p=1 variant ~ clean"
)
assert (va1[0] != va1[1]).mean() > 0.3 and (va1[1] != va1[2]).mean() > 0.3, "variants are not independent draws"
va1b, _ = variants(1.0)
assert np.array_equal(va1, va1b), "same seed, different variants"
print(
    f"[1] p=0 variants == clean codes; p=1 variants differ from clean ({(va1[0] != clean[0]).mean():.0%} of tokens) and from each other; reproducible"
)

k, f = clean.shape[1], clean.shape[2]
f0, f1 = 40, 180
cut = slice_codes(va1.reshape(-1), k, f, f0, f1).reshape(3, k, f1 - f0)
assert all(np.array_equal(cut[v], slice_codes(va1[v].reshape(-1), k, f, f0, f1).reshape(k, f1 - f0)) for v in range(3))
assert np.array_equal(cut[0], va1[0][:, f0:f1])
print("[2] slice_codes keeps variant frames aligned with the clean frames")

tmp = tempfile.mkdtemp(prefix="check_user_variants_")
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
}
Dataset.from_list([row]).save_to_disk(os.path.join(tmp, "v"))
table = read_arrow_table(os.path.join(tmp, "v"))
rng = np.random.default_rng(0)
hits = [0, 0, 0]
for _ in range(3000):
    u = draw_user_variant(table, 0, rng)
    hits[[v for v in range(3) if np.array_equal(u, vb1[v])][0]] += 1
assert all(900 < h < 1100 for h in hits), hits
assert any(np.array_equal(draw_user_variant(table, 0, rng, side="assistant"), va1[v]) for v in range(3))
Dataset.from_list(
    [dict(row, codes_assistant_aug=np.zeros(0, np.int16), codes_user_aug=np.zeros(0, np.int16), n_user_aug=0)]
).save_to_disk(os.path.join(tmp, "none"))
try:
    draw_user_variant(read_arrow_table(os.path.join(tmp, "none")), 0, rng)
    raise RuntimeError("a row without variants was accepted")
except ValueError:
    pass
print(f"[3] draw: uniform over 3 variants {hits}; side='assistant' reads the assistant column; no variants -> error")

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
    if np.array_equal(agent, clean[0][:, : agent.shape[1]]):  # plain read: user must be a user-side variant
        assert any(np.array_equal(user, vb1[v][:, : user.shape[1]]) for v in range(3)), (
            "plain read: user is not a user-side variant"
        )
        seen["plain"] += 1
    else:  # swapped: the assistant target is the stored USER channel, the user stream an assistant-side variant
        assert np.array_equal(agent, clean[1][:, : agent.shape[1]]), (
            "assistant target is not an untouched stored channel"
        )
        assert any(np.array_equal(user, va1[v][:, : user.shape[1]]) for v in range(3)), (
            "swapped read: user is not an assistant-side variant"
        )
        seen["swapped"] += 1
assert seen["plain"] > 10 and seen["swapped"] > 10, seen
print(
    f"[4] PersonaPlex loader: user stream = a variant of the right side on every read {seen}; assistant targets untouched"
)
print("OK")
