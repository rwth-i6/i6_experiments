"""Guard: the synthetic knowledge dialogues enter PersonaPlex training as PROPER hybrid-prompt rows, at an EXACT
stratified in-batch share, without disturbing the JRE rows (ablation ab_epad1_know30, 2026-09-25).

WHY. A knowledge row that reached the loader without its voice / role prompt, with its answer planted over the
question (Whisper's stretched word 0), swapped onto the TTS questioner's channel, or mixed at a share that only
holds in expectation would train fine and look fine in a loss curve. So everything runs through the real worker
(``moshi_family.knowledge_codes``, CPU) on real rows of the v6 corpus and the real ``build_knowledge_mix_loader``
on p3's corpus (what ab_epad1 trains on):

  1. SCHEDULE. ``stratified_share_count(0.3, 8, i)`` is 2 or 3 for every batch, 2,400 of 8,000 rows over the
     1000 steps, the running share exactly 0.3 at every multiple of 5 batches and never more than 1/(8n) below.
     CAN FAIL: a per-read coin (the other obvious implementation) breaks all three.
  2. WORKER. On 16 real v6 rows: every row gets the paper's QA role verbatim, the 50-frame voice prompt (4 s)
     of the TTS reference clip, one speaker; stored codes reshape to exactly the offline encode of the
     loudness-normalised audio (the codebook-major trap); no word starts before the assistant's measured speech
     onset -- and the stored source alignments DID violate that (non-vacuous: the clamp moved rows).
  3. MIXED LOADER, real p3 + the worker's corpus, ab_epad1's knobs: the knowledge items per batch == the schedule
     (items told apart by their voice prompt), every item's prefix is the hybrid prompt (for knowledge items ==
     ``system_prompt_prefix(QA role, TTS voice)``), the loss mask is 0 exactly over it, frame 0 is the initial
     token, EPAD precedes every word that follows padding (both kinds), knowledge items are never swapped or
     user-augmented (agent/user codes == their stored row), and the JRE items are, one for one, the head of
     the UNMIXED loader's stream (``build_data_loader``, seed 0 = ab_epad1's). CAN FAIL: the same comparison
     with the mix built at seed 1 differs; the coin mixer misses the schedule.
  4. RESUME. ``skip_batches=4`` yields the dialogue codes of the fresh loader's batch 4 onwards (both corpora).
  5. LAUNCHER. reads knowledge_data / knowledge_share unconditionally and routes them to the mix loader.

Runs in moshi_family_venv, CPU (~2-4 min; mimi + spm from the HF cache):
    ./hpc-venv.py --cluster i6-rz moshi_family --sh 'cd <setup> && HF_HUB_OFFLINE=1 python \
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_knowledge_mix.py'
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

from moshi_family.personaplex.train_data import (  # noqa: E402
    KNOWLEDGE_STREAM_TAG,
    PersonaPlexDataConfig,
    PersonaPlexTokenizer,
    build_data_loader,
    build_knowledge_mix_loader,
    model_initial_ids,
    system_prompt_prefix,
)
from moshi_family.train_data_common import (  # noqa: E402
    TEXT_EPAD_ID,
    TEXT_PADDING_ID,
    TEXT_ROW,
    collate_codes,
    decode_codes_row,
    normalize_alignments,
    read_arrow_table,
    row_item_stream,
    speech_onset_sec,
    stratified_share_count,
    stratified_share_total,
)

P3 = "output/persona_v3/p1_data"  # AttachPersonaPrompts of p3 = ab_epad1's train_data
V6 = "output/triviaqa_mix_v6"
VOICE = "output/aggregated_speaker_voices/assistant_voices/prompt_0_voice_3.wav"
#: arXiv 2602.06053 section 3.2.1, verbatim ("We use a fixed role for this dataset: ...")
QA_ROLE = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
LOUDNESS = {"target_lufs": -24.0, "max_gain_db": 10.0, "min_lufs": -45.0}  # PODCAST_INGEST["loudness_norm"]
SLICES = {"engaging_teacher": 6, "quickfire": 5, "qa_direct": 5}
SHARE, B = 0.3, 8
fails = []


def ck(c, label, extra=""):
    print(f"  [{'ok' if c else 'FAIL'}] {label}{'  ' + extra if extra else ''}", flush=True)
    if not c:
        fails.append(label)


def schedule_ok(counts: list[int]) -> tuple[bool, str]:
    """The three stratification properties for a list of per-batch knowledge counts."""
    n_b = len(counts)
    in_set = all(c in (2, 3) for c in counts)
    cum = np.cumsum(counts)
    exact = all(cum[n - 1] * 10 == 3 * B * n for n in range(5, n_b + 1, 5))
    close = all(
        0.3 - cum[n - 1] / (B * n) <= 1 / (B * n) + 1e-12 and cum[n - 1] / (B * n) <= 0.3 + 1e-12
        for n in range(1, n_b + 1)
    )
    return in_set and exact and close, f"set {sorted(set(counts))}, total {int(cum[-1])}/{B * n_b}"


print("[1] stratified schedule (share 0.3, batch 8, 1000 steps)")
counts = [stratified_share_count(SHARE, B, i) for i in range(1000)]
ok, msg = schedule_ok(counts)
ck(ok and sum(counts) == 2400, "2 or 3 per batch, exactly 30% every 5 batches, 2,400 of 8,000 rows", msg)
ck(counts[:10] == [2, 2, 3, 2, 3, 2, 2, 3, 2, 3], "first 10 batches 2,2,3,2,3,2,2,3,2,3", str(counts[:10]))
ck(stratified_share_total(SHARE, B, 1000) == 2400, "stratified_share_total == sum of the counts")
coin = np.random.default_rng(0).binomial(B, SHARE, size=1000).tolist()
ok_coin, msg_coin = schedule_ok(coin)
ck(not ok_coin, "CAN FAIL: a per-read coin misses the schedule", msg_coin)
naive = [int(np.floor(0.3 * B * n)) for n in range(1001)]
print(
    f"  (float floor(0.3*8*n) differs from the exact total at {sum(naive[n] != stratified_share_total(SHARE, B, n) for n in range(1001))} of 1001 n)"
)

print("[2] worker: knowledge_codes on 16 real v6 rows (CPU)")
tmp = tempfile.mkdtemp(prefix="know30_")
specs, src = [], {}
for name, n in SLICES.items():
    ds = load_from_disk(os.path.realpath(os.path.join(V6, name, "moshi_annotate")))
    idx = sorted(np.random.default_rng(3).choice(len(ds), size=n, replace=False).tolist())
    d = os.path.join(tmp, f"src_{name}")
    ds.select(idx).save_to_disk(d)
    specs.append(f"{name}={d}")
    src[name] = read_arrow_table(d)
out = os.path.join(tmp, "out", "dataset")
env = dict(os.environ, PYTHONPATH=os.pathsep.join([LIB, os.path.join(SETUP, "recipe")]))
cmd = [sys.executable, "-m", "moshi_family.knowledge_codes", "--out", out, "--voice_wav", VOICE, "--system_prompt", QA_ROLE,
       "--loudness_norm", json.dumps(LOUDNESS), "--device", "cpu", "--workers", "2",
       "--progress", os.path.join(tmp, "progress.json")] + specs  # fmt: skip
r = subprocess.run(cmd, env=env, capture_output=True, text=True)
print("\n".join("    " + ln for ln in (r.stdout + r.stderr).strip().splitlines()[-4:]))
ck(r.returncode == 0, "worker exit 0")
K = load_from_disk(out)
kt = read_arrow_table(out)
summary = json.load(open(os.path.join(tmp, "out", "summary.json")))
ck(len(K) == sum(SLICES.values()), f"{len(K)} rows out of {sum(SLICES.values())}")
ck(set(K["context"]) == {QA_ROLE}, "every row's context is the paper's QA role, verbatim")
vf = set(K["voice_n_frames"])
ck(vf == {50}, "one 50-frame (4 s) voice prompt", str(vf))
ck(len({tuple(v) for v in K["voice_codes"]}) == 1, "the same voice codes on every row")
ck(all(len({a["speaker"] for a in al}) <= 1 for al in K["alignments"]), "one speaker per row")

import soundfile as sf  # noqa: E402

from moshi_family.personaplex.models import loaders  # noqa: E402
from moshi_family.podcast_ingest_main import encode_pair  # noqa: E402
from moshi_family.podcast_mimi_encode import apply_loudness_norm  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

mimi = loaders.get_mimi(hf_hub_download("nvidia/personaplex-7b-v1", loaders.MIMI_NAME), "cpu")
n_early_src = n_early_out = 0
parity = 0
row = 0
for name in SLICES:
    t = src[name]
    for i in range(t.num_rows):
        a, sr = sf.read(io.BytesIO(t.column("audio_assistant")[i].as_py()["bytes"]), dtype="float32")
        u, _ = sf.read(io.BytesIO(t.column("audio_user")[i].as_py()["bytes"]), dtype="float32")
        onset = speech_onset_sec(a, sr)
        raw = normalize_alignments(t.column("alignments")[i].as_py())
        new = normalize_alignments(kt.column("alignments")[row].as_py())
        if kt.column("id")[row].as_py() != f"{name}/{t.column('id')[i].as_py()}":
            fails.append(f"row order {row}")
        n_early_src += int(bool(raw) and raw[0][1][0] < onset - 0.5)
        n_early_out += sum(int(s < onset - 1e-4) for _w, (s, _e), _k in new)
        if row < 2:  # stored bytes == the offline encode of the normalised audio (reshape / channel order)
            an, un, _ = apply_loudness_norm(a, u, LOUDNESS)
            ref = encode_pair(mimi, an, un, device="cpu", chunk_sec=0.0, mode="offline")
            ca, cu, _fr, _al = decode_codes_row(kt, row)
            parity += int(np.array_equal(ca, ref[0]) and np.array_equal(cu, ref[1]))
        row += 1
ck(n_early_out == 0, "no stored word starts before the assistant's speech onset", f"{n_early_out} early words")
ck(
    n_early_src > 0,
    "non-vacuous: the SOURCE alignments had word 0 > 0.5 s before the onset",
    f"{n_early_src}/{row} rows",
)
ck(parity == 2, "stored codes == offline encode of the -24 LUFS audio, [assistant, user], codebook-major (2 rows)")
print(f"  summary: clamped {summary['clamped_frac']}, gains dB {summary['gain_db_p10_50_90']}")

print("[3] mixed loader: p3 (ab_epad1's corpus + knobs) + the knowledge corpus")
tok = PersonaPlexTokenizer(
    "nvidia/personaplex-7b-v1",
    device="cpu",
    cfg=PersonaPlexDataConfig(duration_sec=60.0, emit_epad=True, onset_floor=True, word_overlap="drop"),
)
KNOBS = dict(system_prompt_key="context", voice_codes_key="voice_codes", swap_channels=True, stored_user_augment=True)
N_BATCH = 10
kvoice = np.asarray(K[0]["voice_codes"], dtype=np.int64).reshape(8, -1)
kprefix = system_prompt_prefix(tok.system_prompt_text_tokens(QA_ROLE), kvoice, n_q=16, silence_frames=6)
stored = [decode_codes_row(kt, j)[:2] for j in range(len(K))]


def uncollate(batch):
    codes, mask = batch[0], batch[1]
    items = []
    for b in range(codes.shape[0]):
        T = int(torch.nonzero(mask[b]).max()) + 1
        items.append((codes[b, :, :T], mask[b, :T]))
    return items


def is_knowledge(codes) -> bool:
    P = kprefix.shape[1]
    return codes.shape[1] > P and torch.equal(codes[:, :P], kprefix)


def epad_violations(codes, mask) -> int:
    text = codes[TEXT_ROW][mask]
    word = (text != TEXT_PADDING_ID) & (text != TEXT_EPAD_ID)
    return int((word[1:] & (text[:-1] == TEXT_PADDING_ID)).sum())


mix = build_knowledge_mix_loader(P3, out, tok, batch_size=B, share=SHARE, seed=0, **KNOBS)
batches = [uncollate(next(mix)) for _ in range(N_BATCH)]
per_batch = [sum(is_knowledge(c) for c, _m in items) for items in batches]
ck(per_batch == counts[:N_BATCH], "knowledge items per batch == the schedule", str(per_batch))
init_t, init_a = model_initial_ids()
viol = {"know": 0, "jre": 0}
n_words = {"know": 0, "jre": 0}
prefix_ok = unswapped = 0
jre_items, know_items = [], []
for items in batches:
    for codes, mask in items:
        kind = "know" if is_knowledge(codes) else "jre"
        P = int((~mask).long().argmin())  # first supervised frame = prefix length
        prefix_ok += int(
            bool((~mask[:P]).all()) and bool(mask[P:].all()) and int(codes[TEXT_ROW, 0]) == init_t
            and bool((codes[1:, 0] == init_a).all())
        )  # fmt: skip
        viol[kind] += epad_violations(codes, mask)
        n_words[kind] += int(
            ((codes[TEXT_ROW][mask] != TEXT_PADDING_ID) & (codes[TEXT_ROW][mask] != TEXT_EPAD_ID)).sum()
        )
        (know_items if kind == "know" else jre_items).append((codes, mask))
        if kind == "know":
            P = kprefix.shape[1]
            dia = codes[1:, P:].numpy()
            unswapped += int(
                any(
                    np.array_equal(dia[:8], ca[:, : dia.shape[1]]) and np.array_equal(dia[8:], cu[:, : dia.shape[1]])
                    for ca, cu in stored
                )
            )
n_items = N_BATCH * B
ck(
    prefix_ok == n_items,
    "loss mask 0 exactly over the prompt prefix, frame 0 = initial token (all items)",
    f"{prefix_ok}/{n_items}",
)
ck(
    len(know_items) == sum(per_batch) and all(is_knowledge(c) for c, _ in know_items),
    "knowledge prefix == system_prompt_prefix(QA role, TTS voice)",
)
ck(
    viol["know"] == 0 and viol["jre"] == 0,
    "EPAD before every word that follows padding",
    f"{viol} over {n_words} word tokens",
)
ck(n_words["know"] > 0, "knowledge items carry words")
ck(
    unswapped == len(know_items),
    "knowledge items never swapped / augmented (== their stored row)",
    f"{unswapped}/{len(know_items)}",
)

ref = build_data_loader(P3, tok, batch_size=1, seed=0, infinite=True, **KNOBS)
head = [uncollate(next(ref))[0] for _ in range(len(jre_items))]
same = all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(jre_items, head))
ck(same, f"the {len(jre_items)} JRE items == the head of the unmixed loader (ab_epad1's stream)")
mix1 = build_knowledge_mix_loader(P3, out, tok, batch_size=B, share=SHARE, seed=1, **KNOBS)
jre1 = [it for _ in range(3) for it in uncollate(next(mix1)) if not is_knowledge(it[0])]
ck(
    not all(torch.equal(a[0], b[0]) for a, b in zip(jre1, head)),
    "CAN FAIL: the mix at seed 1 does not match (check is live)",
)
ks = row_item_stream(kt, lambda i, _rng: i, seed=[0, KNOWLEDGE_STREAM_TAG])
kidx = [next(ks) for _ in range(len(K))]
ck(sorted(kidx) == list(range(len(K))), "the knowledge stream visits every row once per epoch")

print("[4] resume: skip_batches=4 == batch 4 onwards (dialogue codes; swap/augment off so reads are deterministic)")
plain = dict(system_prompt_key="context", voice_codes_key="voice_codes")


def dialogue(items):
    return [c[:, int((~m).long().argmin()) :] for c, m in items]


fresh = build_knowledge_mix_loader(P3, out, tok, batch_size=B, share=SHARE, seed=0, **plain)
fresh_b = [dialogue(uncollate(next(fresh))) for _ in range(7)]
res = build_knowledge_mix_loader(P3, out, tok, batch_size=B, share=SHARE, seed=0, skip_batches=4, **plain)
res_b = [dialogue(uncollate(next(res))) for _ in range(3)]
ck(
    all(torch.equal(x, y) for fb, rb in zip(fresh_b[4:], res_b) for x, y in zip(fb, rb)),
    "resumed batches 0-2 == fresh batches 4-6",
)

print("[5] launcher")
src_l = open(os.path.join(LIB, "moshi_family", "personaplex", "finetune_launcher.py")).read()
ck(
    'knowledge_data = cfg.get("knowledge_data")' in src_l and 'knowledge_share = cfg.get("knowledge_share")' in src_l,
    "reads both keys",
)
ck("build_knowledge_mix_loader(" in src_l and "share=knowledge_share" in src_l, "routes them to the mix loader")
_ = collate_codes  # the loader's collate; imported so a rename fails here too

print()
if fails:
    print(f"FAILED: {len(fails)} check(s): {fails}")
    sys.exit(1)
print("all checks passed")
