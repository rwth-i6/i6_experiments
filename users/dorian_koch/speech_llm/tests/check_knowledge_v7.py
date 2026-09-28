"""Guard: the knowledge corpus v7 -- random JRE assistant voices + the JRE user augmentation (user, 2026-09-28:
"the knowledge mix doesnt have the augmentations on the user channel ... the assistant channel is always the same
voice ... Just use random voices ig.").

WHY. Each piece can be silently wrong and still train: a voice prompt that is not the voice the row speaks in, a
"re-voiced" row that is still the v6 voice, a user variant drawn with the assistant channel's seed or never read, a
concatenated item that mixes two assistant voices under one prompt, or an augmentation refactor that changes the JRE
ingest. So everything runs through the real code on real data (moshi_family_venv, CPU, ~5 min):

  1. AUGMENTATION REFACTOR. ``podcast_mimi_encode.user_stream_variants`` (now built on ``augment_user_variant``) is fed
     the same audio as the pinned pre-refactor version (``git show REF_COMMIT``, exec'd) with a capturing encode: the
     augmented waveforms handed to the encoder and the draw records are identical (noise, echo/reverb, clip on: a tiny
     real ``NoisePool``). ``augment_user_variant(channels=("b",))`` = exactly the full call's ``b`` waveform + record.
     CAN FAIL: the user channel drawn with the other channel's index (0) gives a different waveform.
  2. POOL (``KnowledgeVoicePool`` output). 1,000 distinct voices, none from a held-out-set or evaluation episode; every
     voice prompt == its ``VoicePromptCodes`` codes AND == the ``voice_codes`` a p4 training row of that speaker carries
     (JRE identity); cloning references >= 6 s at 16 kHz. CAN FAIL: the same identity check against the episode's
     OTHER speaker's row fails.
  3. PILOT CONVERSION (``KnowledgeVoiceConvert``, 200 rows). Row-aligned with the annotated slice; voices balanced;
     the channel is silent outside the assistant turns and the 80 ms speech-activity envelope matches v6's within turns
     (timing kept); the speaker check says the converted turns sound like the TARGET voice more than like the v6
     voice (Chatterbox's own voice encoder).
  4. WORKER (``moshi_family.knowledge_codes`` with --voice_pool / --assistant_audio / --user_augment) on 8 real pilot
     rows: voice prompt = the pool voice the row was converted to; the user codes ~= v6's (same audio; CPU vs the GPU
     build), the assistant codes differ from v6's (new voice); 3 user variants, each != clean, records == the draws
     ``augment_user_variant`` makes for that row id; the variants == the offline encode of the augmented waveform
     (1 row); ``clean_prob`` 0.245. CAN FAIL: the records of the ingest's ASSISTANT index (0) differ.
  5. LOADER + CONCAT (``build_knowledge_mix_loader``, p4's corpus + a 3-voice table made from the worker rows, the
     v7 scheme): every concatenated item is one voice (its pieces' voice_id) and its prefix carries that voice;
     each knowledge piece's user frames == its row's clean codes or one of its variants, augmented pieces ~ 1 -
     clean_prob; plain (non-concat) knowledge items the same per row; resume (skip_batches) reproduces the plans
     and the batches. CAN FAIL: the v7 table with the v6 scheme (no same_voice) raises at the per-item voice check;
     knowledge_user_augment=False gives no augmented piece.
  6. PLANNER at scale (``VoiceGroupedRows``, synthetic 6,000 rows / 100 voices over 3 epochs): every item one voice,
     first pieces in stream order of unused rows, each row used once per epoch-length window (+-1), few lookahead
     misses, replay exact. CAN FAIL: a planner whose further pieces ignore the voice mixes voices.
  7. RENDER + LAUNCHER. ``knowledge_user_augment: true`` is rendered only when set; the launcher reads it and routes it.

Run (setup root, moshi_family venv; HF cache):
    ./hpc-venv.py --cluster i6-rz moshi_family --sh 'cd <setup> && HF_HUB_OFFLINE=1 python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_knowledge_v7.py'
"""

import io
import json
import os
import subprocess
import sys
import tempfile

SETUP = os.getcwd()
LIB = os.path.join(SETUP, "recipe", "speech_llm", "full_duplex")
sys.path.insert(0, LIB)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import load_from_disk  # noqa: E402

from moshi_family.train_data_common import decode_codes_row, read_arrow_table  # noqa: E402

#: full_duplex commit before the augment_user_variant refactor (parity reference for [1])
REF_COMMIT = "fc1b4cff"
FD_REPO = os.path.join(SETUP, "projects", "2025-10-speech-llm")
POOL = "output/persona_knowledge/v7_voice_pool"
PILOT = "output/persona_knowledge/v7_vc_pilot"
PILOT_SLICE = "multi_turn_explanation"
V6 = "output/triviaqa_mix_v6"
V6C = "output/persona_knowledge/v6_codes"
P4 = "output/persona_v4/p1_data"
VOICES = "output/persona_v4/voice_codes"
HELDOUT = "output/persona/heldout_loss/sets/heldout"
QA_ROLE = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
LOUDNESS = {"target_lufs": -24.0, "max_gain_db": 10.0, "min_lufs": -45.0}
AUG = {
    "variants": 3,
    "seed": 0,
    "augment": {"gain_prob": 0.5, "noise_prob": 0.3, "echo_reverb_prob": 0.3, "clip_peak": 1.0},
}
SR = 24000
fails = []


def ck(c, label, extra=""):
    print(f"  [{'ok' if c else 'FAIL'}] {label}{'  ' + extra if extra else ''}", flush=True)
    if not c:
        fails.append(label)


tmp = tempfile.mkdtemp(prefix="know_v7_")

# ------------------------------------------------------------------------------------------------------------------
print("[1] augmentation refactor: user_stream_variants == the pinned version; b-only == the full call's b")
import pyarrow as pa  # noqa: E402

import moshi_family.podcast_mimi_encode as pme  # noqa: E402

noise_dir = os.path.join(tmp, "noise")
os.makedirs(noise_dir)
rng = np.random.default_rng(7)
clips = [(rng.standard_normal(int(SR * s)) * 3000).astype(np.int16) for s in (0.7, 1.3, 2.1, 0.4, 3.0)]
with pa.ipc.new_file(os.path.join(noise_dir, "part0.arrow"), pa.schema([("audio", pa.list_(pa.int16()))])) as w:
    w.write_table(pa.table({"audio": pa.array([c.tolist() for c in clips], type=pa.list_(pa.int16()))}))
json.dump({"sample_rate": SR, "n_clips": len(clips)}, open(os.path.join(noise_dir, "meta.json"), "w"))
t = np.arange(int(6.0 * SR)) / SR
env_a = (np.sin(2 * np.pi * 0.5 * t) > 0).astype(np.float32)
a_w = (0.05 * rng.standard_normal(t.shape[0]) * env_a).astype(np.float32)
b_w = (0.05 * rng.standard_normal(t.shape[0]) * (1 - env_a)).astype(np.float32)
cfg = {**AUG, "noise_pool": noise_dir}

ref_src = subprocess.run(
    ["git", "-C", FD_REPO, "show", f"{REF_COMMIT}:src/speech_llm/full_duplex/moshi_family/podcast_mimi_encode.py"],
    capture_output=True,
    text=True,
    check=True,
).stdout
assert "def augment_user_variant" not in ref_src, "REF_COMMIT already has the refactor"
ref_ns = {"__name__": "pme_ref"}
exec(compile(ref_src, "pme_ref.py", "exec"), ref_ns)


def capture_run(fn):
    seen = []

    def enc(x, y):
        seen.append((np.array(x, copy=True), np.array(y, copy=True)))
        return np.zeros((2, 8, 1), dtype=np.int16)

    _ca, _cb, meta = fn(a_w, b_w, encode=enc, cfg=cfg, item_id="guard/row0")
    return seen, meta


old_seen, old_meta = capture_run(ref_ns["user_stream_variants"])
new_seen, new_meta = capture_run(pme.user_stream_variants)
same_w = len(old_seen) == len(new_seen) == 3 and all(
    np.array_equal(x0, x1) and np.array_equal(y0, y1) for (x0, y0), (x1, y1) in zip(old_seen, new_seen)
)
ck(same_w, "the augmented waveforms handed to the encoder are identical (3 variants x 2 channels)")
ck(json.dumps(old_meta, sort_keys=True) == json.dumps(new_meta, sort_keys=True), "draw records identical")
eff = {
    "+".join(k for k in ("gain_db", "noise_db", "echo_scale") if r[k] is not None) for ch in "ab" for r in new_meta[ch]
}
print(f"    effects drawn: {sorted(eff)}; clip_peak {new_meta.get('clip_peak')}")
pool_n = pme.open_noise_pool(cfg)
b_ok = True
for v in range(3):
    s, r = pme.augment_user_variant(a_w, b_w, cfg=cfg, item_id="guard/row0", variant=v, channels=("b",), pool=pool_n)
    b_ok &= set(s) == {"b"} and np.array_equal(s["b"], new_seen[v][1]) and r["b"] == new_meta["b"][v]
ck(b_ok, 'augment_user_variant(channels=("b",)) == the full call\'s b waveform + record, every variant')
s_mut, _ = pme.augment_user_variant(b_w, a_w, cfg=cfg, item_id="guard/row0", variant=0, channels=("a",), pool=pool_n)
ck(not np.array_equal(s_mut["a"], new_seen[0][1]), "CAN FAIL: the user channel drawn with channel index 0 differs")

# ------------------------------------------------------------------------------------------------------------------
print("[2] voice pool")
pt = read_arrow_table(POOL)
psum = json.load(open(POOL + ".summary.json"))
ids = pt.column("voice_id").to_pylist()
ck(len(ids) == 1000 and len(set(ids)) == 1000, "1,000 distinct voices", f"{len(set(ids))}")
job_dir = os.path.dirname(os.path.dirname(os.path.realpath(POOL)))
import ast  # noqa: E402

# the job's exclude_episodes parameter, as Sisyphus recorded it (a finished job's work/ dir is packed away)
_line = next(ln for ln in open(os.path.join(job_dir, "info")) if ln.startswith("PARAMETER: exclude_episodes: "))
excl = set(ast.literal_eval(_line.split(": ", 2)[2]))
ck(len(excl) >= 50, "the evaluation episodes were excluded", f"{len(excl)} episodes")
held_eps = set(read_arrow_table(HELDOUT).column("episode_id").to_pylist())
peps = pt.column("episode_id").to_pylist()
ck(
    not (set(peps) & (excl | held_eps)),
    "no voice from a held-out-set or evaluation episode",
    f"{len(excl)} + {len(held_eps)} eps",
)
vt = read_arrow_table(VOICES)
vidx = {k: i for i, k in enumerate(zip(vt.column("episode_id").to_pylist(), vt.column("label").to_pylist()))}
labels = pt.column("label").to_pylist()
pcodes = pt.column("voice_codes").to_pylist()
same_vpc = all(pcodes[v] == vt.column("codes")[vidx[(peps[v], labels[v])]].as_py() for v in range(len(ids)))
ck(same_vpc, "every voice prompt == its VoicePromptCodes codes")
# JRE identity: a p4 training row voiced by that speaker carries the same voice_codes
p4 = read_arrow_table(P4).select(["id", "voice_label", "voice_codes"])
p4_ids = p4.column("id").to_pylist()
p4_lab = p4.column("voice_label").to_pylist()
first_row = {}
for j, (rid, lab) in enumerate(zip(p4_ids, p4_lab)):
    first_row.setdefault((rid.split("#", 1)[0], lab), j)
n_id = n_hit = n_other = n_other_diff = 0
for v in range(0, 1000, 7):
    j = first_row.get((peps[v], labels[v]))
    if j is None:
        continue
    n_id += 1
    n_hit += int(p4.column("voice_codes")[j].as_py() == pcodes[v])
    others = [k for k in first_row if k[0] == peps[v] and k[1] != labels[v]]
    if others:
        n_other += 1
        n_other_diff += int(p4.column("voice_codes")[first_row[others[0]]].as_py() != pcodes[v])
ck(n_id >= 50 and n_hit == n_id, "== the voice_codes p4's training rows of that speaker carry", f"{n_hit}/{n_id}")
ck(
    n_other > 20 and n_other_diff == n_other,
    "CAN FAIL: the episode's other speaker's rows carry different codes",
    f"{n_other_diff}/{n_other}",
)
refsec = np.asarray(pt.column("ref_sec").to_pylist())
ck(refsec.min() >= 6.0 - 1e-3, "cloning references >= 6 s", f"min {refsec.min():.2f}, median {np.median(refsec):.2f}")
print(
    f"    pool summary: {json.dumps({k: psum[k] for k in ('candidates', 'eligible', 'pool_episodes', 'train_rows_per_voice_p10_50_90')})}"
)

# ------------------------------------------------------------------------------------------------------------------
print("[3] pilot conversion")
import soundfile as sf  # noqa: E402

vc = read_arrow_table(PILOT)
vsum = json.load(open(PILOT + ".summary.json"))
ann = read_arrow_table(os.path.realpath(os.path.join(V6, PILOT_SLICE, "moshi_annotate")))
tts = read_arrow_table(os.path.realpath(os.path.join(V6, PILOT_SLICE, "chatterbox_inference")))
n_vc = vc.num_rows
ck(
    vc.column("id").to_pylist() == ann.column("id").to_pylist()[:n_vc],
    f"row-aligned with the annotated slice ({n_vc} rows)",
)
sys.path.insert(0, os.path.join(SETUP, "recipe", "i6_experiments", "users", "dorian_koch", "speech_llm"))
from knowledge_voice_convert import assign_voices  # noqa: E402

full = assign_voices(ann.num_rows, 1000, 0, PILOT_SLICE)
cnt_full = np.bincount(full, minlength=1000)
ck(vc.column("voice").to_pylist() == full[:n_vc].tolist(), "voices = the seeded balanced deal of the whole slice")
ck(cnt_full.max() - cnt_full.min() <= 1, "balanced over the slice", f"{cnt_full.min()}-{cnt_full.max()} rows per voice")
turns_by_id = dict(zip(tts.column("id").to_pylist(), tts.column("turns").to_pylist()))
fr = int(SR * 0.08)
leak = agree = total = 0
for i in range(0, n_vc, 10):
    x6, _ = sf.read(io.BytesIO(ann.column("audio_assistant")[i].as_py()["bytes"]), dtype="float32")
    x7, sr7 = sf.read(io.BytesIO(vc.column("audio_assistant")[i].as_py()["bytes"]), dtype="float32")
    assert sr7 == SR and x7.shape == x6.shape, (sr7, x7.shape, x6.shape)
    tr = turns_by_id[vc.column("id")[i].as_py()]
    inside = np.zeros(x6.shape[0], dtype=bool)
    for spk, s, e in zip(tr["speaker"], tr["start_time"], tr["end_time"]):
        if spk == "assistant":
            inside[int(round(s * SR)) : int(round(e * SR))] = True
    leak += int(np.abs(x7[~inside]).max(initial=0.0) > 1e-4)
    nf = x6.shape[0] // fr

    def act(x):
        e = np.sqrt(np.mean(x[: nf * fr].reshape(nf, fr) ** 2, axis=1) + 1e-12)
        return 20 * np.log10(e) > -45

    ins = inside[: nf * fr].reshape(nf, fr).all(axis=1)
    agree += int((act(x6)[ins] == act(x7)[ins]).sum())
    total += int(ins.sum())
ck(leak == 0, "silent outside the assistant turns (20 rows)", f"{leak} rows with sound outside")
ck(
    total > 0 and agree / total >= 0.9,
    "speech activity inside turns matches v6 (80 ms frames, -45 dBFS)",
    f"{agree / max(total, 1):.3f}",
)
cos = vsum.get("speaker_cos_mean", {})
print(
    f"    speaker check ({vsum.get('speaker_check_turns')} turns): {cos}; closer to target {vsum.get('speaker_conv_closer_to_target')}"
)
ck(
    vsum.get("speaker_conv_closer_to_target", 0) >= 0.9
    and cos.get("conv_to_target", 0) > cos.get("v6_to_target", 1) + 0.1,
    "converted turns sound like the TARGET voice (>= 90% closer to it than to the v6 voice; +0.1 over v6's own turns)",
)
print(
    f"    speed {vsum.get('rows_per_s')} rows/s, len diff ms p1/50/99 {vsum.get('len_diff_ms_p1_50_99')}, clipped rows {vsum.get('clipped_rows')}"
)

# ------------------------------------------------------------------------------------------------------------------
print("[4] worker on 8 real pilot rows (CPU)")
N = 8
ann_d, vc_d = os.path.join(tmp, "ann"), os.path.join(tmp, "vc")
load_from_disk(os.path.realpath(os.path.join(V6, PILOT_SLICE, "moshi_annotate"))).select(range(N)).save_to_disk(ann_d)
load_from_disk(os.path.realpath(PILOT)).select(range(N)).save_to_disk(vc_d)
out = os.path.join(tmp, "k", "dataset")
env = dict(os.environ, PYTHONPATH=os.pathsep.join([LIB, os.path.join(SETUP, "recipe")]))
cmd = [sys.executable, "-m", "moshi_family.knowledge_codes", "--out", out, "--voice_pool", os.path.realpath(POOL),
       "--assistant_audio", f"{PILOT_SLICE}={vc_d}", "--user_augment", json.dumps(cfg), "--system_prompt", QA_ROLE,
       "--loudness_norm", json.dumps(LOUDNESS), "--device", "cpu", "--workers", "2",
       "--progress", os.path.join(tmp, "progress.json"), f"{PILOT_SLICE}={ann_d}"]  # fmt: skip
r = subprocess.run(cmd, env=env, capture_output=True, text=True)
print("\n".join("    " + ln for ln in (r.stdout + r.stderr).strip().splitlines()[-3:]))
ck(r.returncode == 0, "worker exit 0")
kt = read_arrow_table(out)
v6t = read_arrow_table(V6C)
v6_row = {rid: j for j, rid in enumerate(v6t.column("id").to_pylist()) if rid.startswith(PILOT_SLICE + "/")}
vc_voice = vc.column("voice").to_pylist()
voice_ok = user_agree = asst_agree = asst_sem = 0
rec_ok = var_ne = 0
same_var = []  # (row, variant, effects) of variants equal to the clean codes
for j in range(kt.num_rows):
    rid = kt.column("id")[j].as_py()
    v = vc_voice[j]
    voice_ok += int(kt.column("voice_codes")[j].as_py() == pcodes[v] and kt.column("voice_id")[j].as_py() == ids[v])
    ca, cu, _f, _al = decode_codes_row(kt, j)
    ca6, cu6, _f6, _al6 = decode_codes_row(v6t, v6_row[rid])
    user_agree += float((cu == cu6).mean()) if cu.shape == cu6.shape else 0.0
    # over the frames where BOTH are speech (cb0 != 1316, offline Mimi's digital silence): the silent frames of the
    # assistant channel are identical by construction
    sp = (ca[0] != 1316) & (ca6[0] != 1316) if ca.shape == ca6.shape else None
    # semantic codebook 0 carries the words (the same here); the acoustic codebooks 1-7 carry the voice
    asst_agree += float((ca[1:, sp] == ca6[1:, sp]).mean()) if sp is not None and sp.any() else 1.0
    asst_sem += float((ca[0, sp] == ca6[0, sp]).mean()) if sp is not None and sp.any() else 1.0
    n_var = kt.column("n_user_aug")[j].as_py()
    var = np.asarray(kt.column("codes_user_aug")[j].as_py(), dtype=np.int16).reshape(n_var, cu.shape[0], cu.shape[1])
    meta = json.loads(kt.column("user_aug_json")[j].as_py())
    var_ne += int(n_var == 3)
    for k in range(n_var):
        if np.array_equal(var[k], cu):
            d = meta["user"][k]
            same_var.append(
                (
                    j,
                    k,
                    [
                        e
                        for e, key in (("gain", "gain_db"), ("noise", "noise_db"), ("echo", "echo_scale"))
                        if d[key] is not None
                    ],
                )
            )
    want = [
        pme.augment_user_variant(np.zeros(SR, np.float32), np.zeros(SR, np.float32), cfg=cfg, item_id=rid, variant=k,
                                 channels=("b",), pool=pool_n)[1]["b"]["seed"]
        for k in range(3)
    ]  # fmt: skip
    rec_ok += int(
        [d["seed"] for d in meta["user"]] == want and abs(meta["clean_prob"] - 0.245) < 1e-9 and "assistant" not in meta
    )
    if j == 0:
        wrong = pme.augment_user_variant(
            np.zeros(SR, np.float32),
            np.zeros(SR, np.float32),
            cfg=cfg,
            item_id=rid,
            variant=0,
            channels=("a",),
            pool=pool_n,
        )[1]["a"]["seed"]
        ck(wrong != meta["user"][0]["seed"], "CAN FAIL: the ingest's ASSISTANT-index draw is a different record")
n = kt.num_rows
ck(n == N, f"{n} rows")
ck(voice_ok == n, "voice prompt + voice_id = the pool voice the row was converted to", f"{voice_ok}/{n}")
ck(user_agree / n >= 0.95, "user codes ~= v6's (the same audio; CPU vs GPU encode)", f"{user_agree / n:.3f}")
# Reference for "the same audio": the USER channel, untouched, agrees 0.988 with the v6 build (CPU vs GPU encode). A
# not-converted assistant channel would agree as much; the converted one must sit far below it.
ck(
    asst_agree / n <= 0.6 and user_agree / n - asst_agree / n >= 0.3,
    "assistant ACOUSTIC codes (cb 1-7) differ from v6's on speech frames (new voice), far below the same-audio user channel",
    f"agreement {asst_agree / n:.3f} vs user {user_agree / n:.3f}; semantic cb0 {asst_sem / n:.3f} (same words, same timing)",
)
ck(var_ne == n, "3 user variants per row", f"{var_ne}/{n}")
# A variant can equal the clean codes only if nothing audible was added: a NOISE-ONLY draw whose noise track is all
# silence over this short row (the paper's alternation: silences U(0, 30) s at p=0.5, against 20-35 s dialogues).
ck(
    len(same_var) <= n and all(eff == ["noise"] for _j, _k, eff in same_var),
    "every variant differs from the clean codes, except noise-only draws whose track is silent here",
    f"{len(same_var)} of {3 * n} equal to clean: {same_var}",
)
ck(
    rec_ok == n,
    "draw records == augment_user_variant's for the row id (user index), clean_prob 0.245, no assistant list",
    f"{rec_ok}/{n}",
)
# variants == the offline encode of the augmented waveform, row 0 (the worker's own prep, re-run here)
from huggingface_hub import hf_hub_download  # noqa: E402

from moshi_family.knowledge_codes import prepare_row  # noqa: E402
from moshi_family.personaplex.models import loaders  # noqa: E402
from moshi_family.podcast_ingest_main import encode_pair  # noqa: E402

mimi = loaders.get_mimi(hf_hub_download("nvidia/personaplex-7b-v1", loaders.MIMI_NAME), "cpu")
at_ = read_arrow_table(ann_d)
vt_ = read_arrow_table(vc_d)
rid0 = kt.column("id")[0].as_py()
a0, u0, _al, info0 = prepare_row(
    at_, 0, LOUDNESS, 60.0, 12.5, asst_table=vt_, aug_cfg=cfg, noise_pool=pool_n, item_id=rid0
)
enc0 = encode_pair(mimi, a0, info0["aug"][0][1], device="cpu", chunk_sec=0.0, mode="offline")
ca0, cu0, _f, _a = decode_codes_row(kt, 0)
var0 = np.asarray(kt.column("codes_user_aug")[0].as_py(), dtype=np.int16).reshape(3, cu0.shape[0], cu0.shape[1])
ck(np.array_equal(var0[1], enc0[1]), "variant 1 of row 0 == offline encode of its augmented user waveform")
summ = json.load(open(os.path.join(tmp, "k", "summary.json")))
print(f"    worker summary: voices {summ.get('voices')}, augment effects {summ.get('user_augment', {}).get('effects')}")

# ------------------------------------------------------------------------------------------------------------------
print("[5] mixed loader + concat on p4's corpus and a 3-voice table of the worker rows")
from datasets import Dataset  # noqa: E402

import moshi_family.knowledge_concat as kc  # noqa: E402
from moshi_family.personaplex.train_data import (  # noqa: E402
    PersonaPlexDataConfig,
    PersonaPlexTokenizer,
    build_knowledge_mix_loader,
    system_prompt_prefix,
)

K = load_from_disk(out)
group = [j % 3 for j in range(len(K))]
first_of = {g: group.index(g) for g in range(3)}
K3 = Dataset.from_dict(
    {
        **{c: K[c] for c in K.column_names if c not in ("voice_id", "voice_codes")},
        "voice_id": [f"guard/voice{g}" for g in group],
        "voice_codes": [K[first_of[g]]["voice_codes"] for g in group],
    },
    features=K.features,
)
k3 = os.path.join(tmp, "k3")
K3.save_to_disk(k3)
meta_p = os.path.join(tmp, "meta.npz")
r = subprocess.run(
    [sys.executable, "-m", "moshi_family.knowledge_concat", "--codes", k3, "--out", meta_p,
     f"{PILOT_SLICE}={os.path.realpath(os.path.join(V6, PILOT_SLICE, 'chatterbox_inference'))}"],
    env=env, capture_output=True, text=True,
)  # fmt: skip
ck(r.returncode == 0, "KnowledgeConcatMeta worker on the v7 rows", (r.stdout + r.stderr).strip().splitlines()[-1][:160])
k3t = read_arrow_table(k3)
tok = PersonaPlexTokenizer(
    "nvidia/personaplex-7b-v1",
    device="cpu",
    cfg=PersonaPlexDataConfig(duration_sec=60.0, emit_epad=True, onset_floor=True, word_overlap="drop"),
)
KNOBS = dict(system_prompt_key="context", voice_codes_key="voice_codes", swap_channels=True, stored_user_augment=True)
V7_SCHEME = {"dialogues": [2, 4], "gap_sec": [0.3, 1.5], "barge_prob": 0.3, "barge_lead_sec": [0.2, 0.8],
             "barge_min_answer_sec": 0.5, "tail_frames": 1, "min_next_sec": 5.0, "same_voice": True}  # fmt: skip
stored = {j: decode_codes_row(k3t, j)[:2] for j in range(k3t.num_rows)}
variants = {
    j: np.asarray(k3t.column("codes_user_aug")[j].as_py(), dtype=np.int64).reshape(3, 8, -1)
    for j in range(k3t.num_rows)
}
prefixes = {
    g: system_prompt_prefix(
        tok.system_prompt_text_tokens(QA_ROLE),
        np.asarray(K[first_of[g]]["voice_codes"], dtype=np.int64).reshape(8, -1),
        n_q=16,
        silence_frames=6,
    )
    for g in range(3)
}


def run_mix(n_batches, skip=0, concat=True, scheme=V7_SCHEME, aug=True):
    item_meta = []
    kw = dict(knowledge_meta=meta_p, knowledge_concat=scheme) if concat else {}
    mix = build_knowledge_mix_loader(
        P4, k3, tok, batch_size=8, share=0.3, seed=0, skip_batches=skip, item_meta=item_meta,
        knowledge_user_augment=aug, **KNOBS, **kw,
    )  # fmt: skip
    batches = [next(mix) for _ in range(n_batches)]
    return batches, item_meta


def know_items(batches, item_meta):
    """[(codes [17, T], meta)] of the knowledge items, in batch order."""
    out_, k = [], 0
    for b in batches:
        codes, mask = b[0], b[1]
        for i in range(codes.shape[0]):
            m = item_meta[k]
            k += 1
            if m["source"] == "knowledge":
                T = int(torch.nonzero(mask[i]).max()) + 1
                out_.append((codes[i, :, :T], m))
    return out_


batches, im = run_mix(12)
ki = know_items(batches, im)
one_voice = prefix_ok = piece_ok = n_pieces = n_aug = multi = 0
for codes, m in ki:
    plan = m["plan"]
    gs = {group[p.row] for p in plan.pieces}
    one_voice += int(len(gs) == 1)
    multi += int(len(plan.pieces) >= 2)
    g = next(iter(gs))
    P = prefixes[g].shape[1]
    prefix_ok += int(torch.equal(codes[:, :P], torch.as_tensor(prefixes[g])))
    user = codes[9:17, P:].numpy()
    for p in plan.pieces:
        base = p.offset - p.src0
        u0_, u1_ = p.offset, min(base + p.u_end, user.shape[1])
        if u1_ <= u0_:
            continue
        seg = user[:, u0_:u1_]
        clean = stored[p.row][1][:, u0_ - base : u1_ - base]
        opts = [np.array_equal(seg, clean)] + [
            np.array_equal(seg, variants[p.row][k][:, u0_ - base : u1_ - base]) for k in range(3)
        ]
        piece_ok += int(any(opts[1:]) if p.user_aug else opts[0])
        n_pieces += 1
        n_aug += int(bool(p.user_aug))
ck(
    len(ki) > 20 and one_voice == len(ki),
    "every concatenated item is ONE voice",
    f"{one_voice}/{len(ki)} items, {multi} with >= 2 dialogues",
)
ck(multi > len(ki) // 2, "non-vacuous: most items hold >= 2 dialogues")
ck(prefix_ok == len(ki), "each item's prefix = the QA role + ITS voice", f"{prefix_ok}/{len(ki)}")
ck(
    piece_ok == n_pieces,
    "each piece's user frames == its row's clean codes or one of its variants (as recorded)",
    f"{piece_ok}/{n_pieces}",
)
share_aug = n_aug / max(n_pieces, 1)
ck(0.55 <= share_aug <= 0.95, "augmented pieces ~ 1 - clean_prob (0.755)", f"{share_aug:.3f} of {n_pieces}")
b_res, im_res = run_mix(4, skip=8)
ki_res = know_items(b_res, im_res)
tail = ki[-len(ki_res) :]


def sig(m):
    return tuple((p.row, p.offset, p.junction, p.user_aug) for p in m["plan"].pieces)


# knowledge items only: the JRE stream's per-item draws (swap / prompt / augmentation) are not replayed on resume
same_res = len(ki_res) > 5 and all(torch.equal(a[0], b[0]) and sig(a[1]) == sig(b[1]) for a, b in zip(tail, ki_res))
ck(
    same_res,
    "resume (skip_batches=8): the knowledge items of batches 8-11 == fresh (codes, plans, user draws)",
    f"{len(ki_res)} items",
)
b_na, im_na = run_mix(6, aug=False)
ck(
    sum(bool(p.user_aug) for _c, m in know_items(b_na, im_na) for p in m["plan"].pieces) == 0,
    "CAN FAIL: knowledge_user_augment=False augments no piece",
)
v6_scheme = {k: v for k, v in V7_SCHEME.items() if k != "same_voice"}
try:
    run_mix(6, scheme=v6_scheme)
    raised = False
except ValueError as e:
    raised = "voice" in str(e)
ck(raised, "CAN FAIL: the v7 table with the v6 scheme (no same_voice) raises at the per-item voice check")
bp, imp = run_mix(8, concat=False)
kip = know_items(bp, imp)
plain_ok = plain_aug = 0
for codes, m in kip:
    j = m["row"]
    g = group[j]
    P = prefixes[g].shape[1]
    user = codes[9:17, P:].numpy()
    opts = [np.array_equal(user, stored[j][1][:, : user.shape[1]])] + [
        np.array_equal(user, variants[j][k][:, : user.shape[1]]) for k in range(3)
    ]
    plain_ok += int(any(opts[1:]) if m["user_aug"] else opts[0])
    plain_aug += int(bool(m["user_aug"]))
ck(
    len(kip) > 10 and plain_ok == len(kip),
    "plain (non-concat) items: user == clean or a variant, as recorded",
    f"{plain_ok}/{len(kip)}, {plain_aug} augmented",
)

# ------------------------------------------------------------------------------------------------------------------
print("[6] VoiceGroupedRows at scale (6,000 rows, 100 voices, 3 epochs)")


def perm_stream(n, seed):
    g = np.random.default_rng(seed)
    while True:
        yield from g.permutation(n).tolist()


NR, NV = 6000, 100
voice_syn = np.random.default_rng(1).permutation(np.arange(NR) % NV)
ok_syn = np.ones(NR, dtype=bool)


def planner_run(n_items, same_voice=True):
    stats = {"skipped_rows": 0}
    src = perm_stream(NR, 5)
    g = kc.VoiceGroupedRows(src, voice_syn, ok_syn, stats)
    rng_ = np.random.default_rng(3)
    items = []
    for _ in range(n_items):
        k_ = int(rng_.integers(2, 5))
        rows_ = [g.first()]
        while len(rows_) < k_:
            if same_voice:
                r_ = g.next_same(
                    kc.Plan(
                        pieces=[
                            kc.Piece(row=x, offset=0, src0=0, a_end=1, u_end=1, a_from=0, junction="start")
                            for x in rows_
                        ]
                    )
                )
            else:
                r_ = g.first()
            if r_ is None:
                break
            rows_.append(r_)
        items.append(rows_)
    return items, stats


items, stats = planner_run(3 * NR // 3)
used = [r_ for it in items for r_ in it]
ck(all(len({voice_syn[r_] for r_ in it}) == 1 for it in items), "every item one voice", f"{len(items)} items")
cnt = np.bincount(np.asarray(used[:NR]), minlength=NR)
ck(
    cnt.max() <= 2 and (cnt >= 1).mean() > 0.95,
    "the first epoch-length of rows used ~once each",
    f"max {cnt.max()}, covered {(cnt >= 1).mean():.3f}",
)
ck(
    stats["voice_lookahead_miss"] < 0.01 * len(items),
    "few lookahead misses",
    f"{stats['voice_lookahead_miss']}, pending max {stats['pending_max']}",
)
items2, _ = planner_run(len(items))
ck(items2 == items, "replay exact")
bad, _ = planner_run(200, same_voice=False)
ck(
    not all(len({voice_syn[r_] for r_ in it}) == 1 for it in bad),
    "CAN FAIL: further pieces taken in stream order mix voices",
)

# ------------------------------------------------------------------------------------------------------------------
print("[7] renderer + launcher")
src_f = open(
    os.path.join(SETUP, "recipe", "i6_experiments", "users", "dorian_koch", "speech_llm", "finetune.py")
).read()
ck(
    'if hp.get("knowledge_user_augment"):' in src_f and "knowledge_user_augment: true" in src_f,
    "renderer emits the key only when set",
)
src_l = open(os.path.join(LIB, "moshi_family", "personaplex", "finetune_launcher.py")).read()
ck(
    'knowledge_user_augment = bool(cfg.get("knowledge_user_augment", False))' in src_l
    and "knowledge_user_augment=knowledge_user_augment" in src_l,
    "launcher reads it unconditionally and routes it to the mix loader",
)

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + '; '.join(fails)}")
sys.exit(1 if fails else 0)
