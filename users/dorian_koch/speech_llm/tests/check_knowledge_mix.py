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
  6. CONCAT (ab_epad1_know30_concat, 2026-09-25; ``moshi_family.knowledge_concat``), on the REAL v6 codes corpus:
     the meta worker (the job's command) on all 60,000 rows (every row usable, filler = offline Mimi silence 1316);
     the planner over the run's 2,400 knowledge items (dialogues per item, junction kinds, gap lengths, rows = know30's
     stream in order, NO row repeated, knowledge hours vs know30); the mixed loader with the concat stream: per-batch
     counts == the schedule, the prompt appears ONCE (prefix == hybrid prompt; no voice block / SINE frame / prompt
     text after it), every frame of both channels == exactly its planned row frame or the silence filler
     (junctions), each piece's text == that dialogue's own text row shifted by its frame offset (alignments shifted
     exactly), no assistant word onset at/after a barge-in cut and dropped words exist (non-vacuous), EPAD before
     every word after padding, one speaker, JRE items == the unmixed stream, same seed = same batches, resume exact.
     CAN FAIL: a mutant that does not shift alignments (``shift_onset`` = identity) breaks the text check.
     [6f] BARGE VARIANTS (ab2_know30_concat_nobarge / _barge, 2026-09-25; ``track_persona_ab2.AB2_BARGE_VARIANTS``, read
     from the recipe): the concat arm's default plans are byte-identical to the pre-``barge_pos_frac`` planner (digest
     pinned 2026-09-25 from the real meta); nobarge has no barge junction; barge: > 90% of junctions barge in, every
     barge-in point lies at 20-90% of the last answer, the cut is the first word boundary at/after it (no kept word
     starts at/after it, the word in progress is kept to its end unless the next onset caps it), later words dropped,
     most barge-ins leave > 1 s of answer unsaid; assembled items claim every frame once. CAN FAIL: the same checks on
     barge_prob 1 with the OLD lead rule (``barge_pos_frac`` None) fail the 20-90% position check.

Runs in moshi_family_venv, CPU (~2-4 min, + [6] ~3 min; mimi + spm from the HF cache):
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

print("[6] concat: 2-4 knowledge dialogues back to back (ab_epad1_know30_concat), real v6 corpus")
import ast  # noqa: E402
import collections  # noqa: E402

import moshi_family.knowledge_concat as kc  # noqa: E402
from moshi_family.personaplex.models.lm import SINE_TOKENS  # noqa: E402

V6C = os.path.realpath("output/persona_knowledge/v6_codes")
V6_NAMES = sorted(json.load(open("output/persona_knowledge/v6_codes_summary.json"))["by_source"])
RECIPE = os.path.join(LIB, "sis_recipe", "doriank", "track_persona_knowledge.py")
SCHEME = next(
    ast.literal_eval(node.value)
    for node in ast.parse(open(RECIPE).read()).body
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "KNOWLEDGE_CONCAT_SCHEME"
)
print(f"  scheme (from the recipe): {SCHEME}")
cfg = kc.ConcatCfg.from_dict(SCHEME)
FR, WIN = 12.5, 750
meta_path = os.path.join(tmp, "concat_meta.npz")
cmd = [sys.executable, "-m", "moshi_family.knowledge_concat", "--codes", V6C, "--out", meta_path,
       "--summary", os.path.join(tmp, "concat_meta_summary.json")] + [
    f"{n}={os.path.realpath(f'output/triviaqa_mix_v6/{n}/chatterbox_inference')}" for n in V6_NAMES]  # fmt: skip
r = subprocess.run(cmd, env=env, capture_output=True, text=True)
print("    " + (r.stdout + r.stderr).strip().splitlines()[-1][:400])
ck(r.returncode == 0, "meta worker exit 0 (the KnowledgeConcatMeta command)")
vt = read_arrow_table(V6C)
meta = kc.ConcatMeta.load(meta_path, vt)
ck(
    bool(meta.ok.all()),
    "every v6 row usable (first turn = user at 0 s, last turn ends on the last frame)",
    f"{int(meta.ok.sum())}",
)
ck(float(meta.barge_ok.mean()) > 0.95, "barge-in possible after >95% of rows", f"{meta.barge_ok.mean():.4f}")
ck(bool((meta.silence_filler[0] == 1316).all()), "silence filler = offline Mimi code of digital zeros (cb0 1316)")

print("  [6b] planner over the run's 2,400 knowledge items (seed 0)")
N_ITEMS = stratified_share_total(SHARE, B, 1000)
idx_stream = row_item_stream(vt, lambda i, _rng: i, seed=[0, KNOWLEDGE_STREAM_TAG])
prng = np.random.default_rng([0, KNOWLEDGE_STREAM_TAG, kc.CONCAT_STREAM_TAG])
aligns_col = vt.column("alignments")


def words_of(r_):
    return normalize_alignments(aligns_col[r_].as_py())


plans = [
    kc.plan_sequence(lambda: int(next(idx_stream)), prng, meta, words_of, window_frames=WIN, fr=FR, cfg=cfg)
    for _ in range(N_ITEMS)
]
rows_used = [p.row for pl in plans for p in pl.pieces]
n_dia = collections.Counter(len(pl.pieces) for pl in plans)
junc = collections.Counter(p.junction for pl in plans for p in pl.pieces[1:])
gaps = [p.gap_frames for pl in plans for p in pl.pieces if p.junction == "gap"]
print(f"    dialogues/item {dict(sorted(n_dia.items()))}; junctions {dict(junc)}; rows {len(rows_used)}")
ck(len(set(rows_used)) == len(rows_used), "no knowledge row repeated over the run", f"{len(rows_used)} rows")
ck(min(n_dia) >= 1 and sum(v for k, v in n_dia.items() if k >= 2) / N_ITEMS > 0.98, ">98% of items hold >= 2 dialogues")
ck(max(n_dia) <= 4, "at most 4 dialogues per item")
ck(min(gaps) >= 4 and max(gaps) <= 19, "gap junctions 4-19 frames (0.3-1.5 s)", f"{min(gaps)}-{max(gaps)}")
bshare = junc["barge"] / max(1, sum(junc.values()))
ck(0.2 < bshare <= 0.32, "barge-in share of junctions ~0.3 (coin x feasibility)", f"{bshare:.3f}")
k30_rows = row_item_stream(vt, lambda i, _rng: i, seed=[0, KNOWLEDGE_STREAM_TAG])
k30_order = [int(next(k30_rows)) for _ in range(len(rows_used))]
ck(rows_used == k30_order, "the rows are know30's knowledge stream, in order (same permutation)")
know_frames_concat = sum(pl.n_frames for pl in plans)
know_frames_k30 = int(sum(meta.n_frames[k30_order[:N_ITEMS]]))
print(
    f"    knowledge dialogue frames over the run: concat {know_frames_concat} ({know_frames_concat / FR / 3600:.1f} h) "
    f"vs know30 {know_frames_k30} ({know_frames_k30 / FR / 3600:.1f} h); items {np.mean([pl.n_frames for pl in plans]) / FR:.1f} s mean"
)

print("  [6c] mixed loader with the concat stream (p3 + v6 codes, ab_epad1's knobs)")
vvoice = np.asarray(vt.column("voice_codes")[0].as_py(), dtype=np.int64).reshape(8, -1)
kprefix_c = system_prompt_prefix(tok.system_prompt_text_tokens(QA_ROLE), vvoice, n_q=16, silence_frames=6)
ck(torch.equal(kprefix_c, kprefix), "v6 corpus voice == the worker's TTS voice (one assistant voice)")
prompt_toks = tok.system_prompt_text_tokens(QA_ROLE)


def is_know_c(codes) -> bool:
    P_ = kprefix_c.shape[1]
    return codes.shape[1] > P_ and torch.equal(codes[:, :P_], kprefix_c)


def concat_mix(seed=0, skip=0, plans_out=None, knobs=KNOBS):
    return build_knowledge_mix_loader(
        P3, V6C, tok, batch_size=B, share=SHARE, seed=seed, skip_batches=skip,
        knowledge_meta=meta_path, knowledge_concat=SCHEME, knowledge_plans=plans_out, **knobs,
    )  # fmt: skip


def text_mismatches(items_plans) -> tuple[int, int, int]:
    """(mismatched frames, compared frames, in-progress-word spill frames after a cut): each piece's text == its
    dialogue's own interleave (cut words dropped) moved by the piece's frame offset -- built here per dialogue, never
    from the item's shifted alignments."""
    bad = n = spill = 0
    kw = tok._text_kwargs()
    for (codes, _m), pl in items_plans:
        text = codes[TEXT_ROW, kprefix_c.shape[1] :]
        T = text.shape[0]
        onsets = []  # item frame of each piece's first kept word (None = no word)
        rows_t = []
        for p in pl.pieces:
            w = [x for x in words_of(p.row) if p.cut_sec is None or x[1][0] < p.cut_sec]
            base = p.offset - p.src0
            # the dialogue alone, cropped where the item's window crops it (a word past the crop gets no EPAD
            # either way: interleave_text skips onsets beyond its num_frames, as for a cropped JRE window)
            nf = min(int(meta.n_frames[p.row]), T - base)
            rows_t.append(tok.interleave_text(w, nf, **kw)[0])
            onsets.append(base + int(w[0][1][0] * FR // 1) if w else None)
        last_stop = 0
        for j, p in enumerate(pl.pieces):
            base = p.offset - p.src0
            stop = T
            for o in onsets[j + 1 :]:
                if o is not None:
                    stop = min(T, o - 1)  # the next piece's EPAD frame
                    break
            for f in range(max(p.offset, last_stop), stop):
                ref = int(rows_t[j][f - base]) if f - base < rows_t[j].shape[0] else TEXT_PADDING_ID
                n += 1
                bad += int(int(text[f]) != ref)
                if p.cut_frame is not None and f >= p.cut_frame and ref not in (TEXT_PADDING_ID, TEXT_EPAD_ID):
                    spill += 1
            last_stop = max(last_stop, stop)
    return bad, n, spill


seen_plans: list = []
mixc = concat_mix(plans_out=seen_plans)
batches_c = [uncollate(next(mixc)) for _ in range(N_BATCH)]
per_batch_c = [sum(is_know_c(c) for c, _m in items) for items in batches_c]
ck(per_batch_c == counts[:N_BATCH], "knowledge items per batch == the schedule (30% exactly)", str(per_batch_c))
know_c = [(c, m) for items in batches_c for c, m in items if is_know_c(c)]
ck(len(know_c) == len(seen_plans), "one plan per knowledge item", f"{len(know_c)} items, {len(seen_plans)} plans")
pairs = list(zip(know_c, seen_plans))
filler = meta.silence_filler
frame_bad = frame_n = n_gap = n_barge = dropped = onset_after_cut = prompt_again = 0
spk = set()
yield_frames = []
for (codes, mask), pl in pairs:
    P_ = kprefix_c.shape[1]
    ck_ok = int((~mask).sum()) == P_ and bool(mask[P_:].all())
    prompt_again += int(not ck_ok)
    dia = codes[:, P_:].numpy()
    a_, u_ = dia[1:9], dia[9:17]
    T = dia.shape[1]
    frame_bad += int(T != pl.n_frames)
    # the prompt once: no voice block, no SINE (prompt user) frame, no prompt text after the prefix
    for f0 in range(0, T - vvoice.shape[1] + 1):
        if np.array_equal(a_[:, f0 : f0 + vvoice.shape[1]], vvoice):
            prompt_again += 1
    prompt_again += int((u_ == SINE_TOKENS[:, None]).all(0).any())
    tx = dia[0].tolist()
    L = len(prompt_toks)
    prompt_again += int(any(tx[i : i + L] == prompt_toks for i in range(0, T - L + 1)))
    claim_a, claim_u = np.zeros(T, int), np.zeros(T, int)
    prev_end = 0
    for j, p in enumerate(pl.pieces):
        ca, cu, _fr, al = decode_codes_row(vt, p.row)
        spk |= {x[2] for x in normalize_alignments(al)}
        base = p.offset - p.src0
        if p.junction == "gap":
            n_gap += 1
            for f in range(prev_end, p.offset):
                frame_n += 1
                frame_bad += int(
                    not (
                        np.array_equal(a_[:, f], filler[:, f - prev_end])
                        and np.array_equal(u_[:, f], filler[:, f - prev_end])
                    )
                )
                claim_a[f] += 1
                claim_u[f] += 1
        if p.junction == "barge":
            n_barge += 1
            prev = pl.pieces[j - 1]
            pbase = prev.offset - prev.src0
            lead = pbase / FR + float(meta.ans_end[prev.row]) - p.offset / FR
            frame_bad += int(not (0.2 - 1 / FR <= lead <= 0.8 + 1 / FR))
            frame_bad += int(prev.cut_frame != p.a_from or p.a_from < p.offset)
            yield_frames.append(p.a_from - p.offset)
            pw = words_of(prev.row)
            dropped += sum(1 for x in pw if x[1][0] >= prev.cut_sec)
            onset_after_cut += sum(
                1 for x in pw if x[1][0] < prev.cut_sec and pbase + int(x[1][0] * FR // 1) >= prev.cut_frame
            )
        for f in range(p.a_from, min(base + p.a_end, T)):
            frame_n += 1
            frame_bad += int(not np.array_equal(a_[:, f], ca[:, f - base]))
            claim_a[f] += 1
        for f in range(p.offset, min(base + p.u_end, T)):
            frame_n += 1
            frame_bad += int(not np.array_equal(u_[:, f], cu[:, f - base]))
            claim_u[f] += 1
        prev_end = min(base + max(p.a_end, p.u_end), T)
    frame_bad += int(not ((claim_a == 1).all() and (claim_u == 1).all()))
ck(prompt_again == 0, "prompt once: loss mask 0 exactly over it; no voice block / SINE frame / prompt text after it")
ck(frame_bad == 0 and frame_n > 0, "every frame == its planned row frame or the silence filler; each claimed once",
   f"{frame_bad} bad of {frame_n}; {n_gap} gap + {n_barge} barge junctions")  # fmt: skip
ck(n_gap > 0 and n_barge > 0, "both junction kinds occur in the checked batches")
bad_t, n_t, spill = text_mismatches(pairs)
ck(bad_t == 0 and n_t > 0, "each piece's text == its own dialogue's text row shifted by its frame offset",
   f"{bad_t} of {n_t} frames; {spill} spill frames of the in-progress word after a cut")  # fmt: skip
ck(onset_after_cut == 0, "no assistant word onset at/after a barge-in cut")
ck(dropped > 0, "non-vacuous: the barge-in cuts dropped words", f"{dropped} words")
print(f"    yield (user onset -> assistant silent), frames: {sorted(yield_frames)}")
vk = sum(epad_violations(c, m) for c, m in know_c)
ck(vk == 0, "EPAD before every word that follows padding (concat items)", f"{vk}")
ck(len(spk) == 1, "one speaker in every used row", str(spk))
jre_c = [(c, m) for items in batches_c for c, m in items if not is_know_c(c)]
ck(len(jre_c) == len(head) and all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(jre_c, head)),
   f"the {len(jre_c)} JRE items == the unmixed stream's head (as without concat)")  # fmt: skip
again = concat_mix()
ck(all(torch.equal(x[0], y[0]) for items in batches_c[:3] for x, y in zip(items, uncollate(next(again)))),
   "same seed -> same batches (deterministic)")  # fmt: skip
fresh_c = concat_mix(knobs=plain)
fb_c = [dialogue(uncollate(next(fresh_c))) for _ in range(7)]
res_c = concat_mix(skip=4, knobs=plain)
rb_c = [dialogue(uncollate(next(res_c))) for _ in range(3)]
ck(
    all(torch.equal(x, y) for fb, rb in zip(fb_c[4:], rb_c) for x, y in zip(fb, rb)),
    "resume: skip_batches=4 == fresh 4-6",
)
kf = sum(int(m.sum()) for c, m in know_c)
jf = sum(int(m.sum()) for c, m in jre_c)
kf30 = sum(int(m.sum()) for c, m in know_items)
jf30 = sum(int(m.sum()) for c, m in jre_items)
print(
    f"    supervised knowledge frame share, first {N_BATCH} batches: concat {kf / (kf + jf):.3f} vs know30 (worker rows) {kf30 / (kf30 + jf30):.3f}"
)

print("  [6d] CAN FAIL: alignments not shifted (shift_onset = identity)")
_real_shift = kc.shift_onset
kc.shift_onset = lambda s_, k_, fr_, fl_: float(s_)
try:
    mut_plans: list = []
    mut = concat_mix(plans_out=mut_plans)
    mut_items = [(c, m) for _ in range(3) for c, m in uncollate(next(mut)) if is_know_c(c)]
    bad_m, n_m, _ = text_mismatches(list(zip(mut_items, mut_plans)))
except AssertionError as e:  # shift_onset's own assert does not exist in the mutant; any failure counts
    bad_m, n_m = 1, 1
    print(f"    mutant raised: {e!r}")
finally:
    kc.shift_onset = _real_shift
ck(bad_m > 0, "the mutant fails the text check", f"{bad_m} of {n_m} frames differ")

print("  [6f] barge variants (ab2_know30_concat_nobarge / _barge): position rule, old default unchanged")
import hashlib  # noqa: E402

AB2 = os.path.join(LIB, "sis_recipe", "doriank", "track_persona_ab2.py")
VARIANTS = next(
    ast.literal_eval(node.value)
    for node in ast.parse(open(AB2).read()).body
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "AB2_BARGE_VARIANTS"
)
print(f"    variants (from the recipe): {VARIANTS}")
#: sha256[:16] of the 2,400 default-scheme plans (seed 0) under the planner BEFORE barge_pos_frac existed (2026-09-25,
#: computed on output/persona_knowledge/v6_concat_meta.npz = what the running concat arms read).
OLD_DEFAULT_PLANS_DIGEST = "75baf893b45b984b"


def plan_digest(plans_) -> str:
    h = hashlib.sha256()
    for pl in plans_:
        pcs = [(p.row, p.offset, p.src0, p.a_end, p.u_end, p.a_from, p.junction, p.gap_frames,
                None if p.cut_sec is None else round(float(p.cut_sec), 6), p.cut_frame) for p in pl.pieces]  # fmt: skip
        h.update(repr((pl.n_frames, pl.n_target, pcs)).encode())
    return h.hexdigest()[:16]


def plan_run(cfg_) -> list:
    st = row_item_stream(vt, lambda i, _rng: i, seed=[0, KNOWLEDGE_STREAM_TAG])
    rng_ = np.random.default_rng([0, KNOWLEDGE_STREAM_TAG, kc.CONCAT_STREAM_TAG])
    return [
        kc.plan_sequence(lambda: int(next(st)), rng_, meta, words_of, window_frames=WIN, fr=FR, cfg=cfg_)
        for _ in range(N_ITEMS)
    ]


ck(plan_digest(plans) == OLD_DEFAULT_PLANS_DIGEST, "default scheme: plans == the pre-barge_pos_frac planner's",
   plan_digest(plans))  # fmt: skip
ck(kc.ConcatCfg.from_dict({**SCHEME, "barge_pos_frac": None}) == cfg, "barge_pos_frac=None is the default")


def barge_stats(plans_) -> dict:
    s = collections.Counter()
    fracs, unsaid = [], []
    for pl in plans_:
        for j, p in enumerate(pl.pieces[1:], 1):
            s[p.junction] += 1
            if p.junction != "barge":
                continue
            prev = pl.pieces[j - 1]
            a0, ae = float(meta.last_asst_start[prev.row]), float(meta.ans_end[prev.row])
            b = float(prev.cut_sec)
            fracs.append((b - a0) / (ae - a0))
            unsaid.append(ae - b)
            pbase = prev.offset - prev.src0
            ws = words_of(prev.row)
            s["dropped"] += sum(1 for x in ws if x[1][0] >= b)
            s["kept_after_cut"] += sum(
                1 for x in ws if x[1][0] < b and pbase + int(x[1][0] * FR // 1) >= prev.cut_frame
            )
            nxt = min([x[1][0] for x in ws if x[1][0] >= b], default=None)
            for x in ws:  # the word in progress at b keeps its end (unless the next onset caps the cut)
                if x[1][0] < b < x[1][1]:
                    end_f = pbase + min(int(np.ceil(x[1][1] * FR)), int(meta.n_frames[prev.row]))
                    cap = None if nxt is None else pbase + int(np.floor(nxt * FR))
                    s["short_cut"] += int(prev.cut_frame < end_f and (cap is None or prev.cut_frame < cap))
            s["user_before_cut"] += int(p.offset > prev.cut_frame)
    s["n_junc"] = sum(s[k] for k in ("gap", "barge"))
    return {"c": s, "fracs": np.asarray(fracs), "unsaid": np.asarray(unsaid)}


cfg_nb = kc.ConcatCfg.from_dict({**SCHEME, **VARIANTS["ab2_know30_concat_nobarge"]})
st_nb = barge_stats(plan_run(cfg_nb))
ck(st_nb["c"]["barge"] == 0 and st_nb["c"]["gap"] > 0, "nobarge: every junction a silence gap", str(dict(st_nb["c"])))
cfg_b = kc.ConcatCfg.from_dict({**SCHEME, **VARIANTS["ab2_know30_concat_barge"]})
lo, hi = cfg_b.barge_pos_frac
plans_b = plan_run(cfg_b)
st_b = barge_stats(plans_b)
cb, fr_b = st_b["c"], st_b["fracs"]
ck(cb["barge"] / max(1, cb["n_junc"]) > 0.9, "barge: > 90% of junctions barge in", f"{cb['barge']}/{cb['n_junc']}")
ck(fr_b.size > 0 and fr_b.min() >= lo - 1e-9 and fr_b.max() <= hi + 1e-9,
   f"barge: every barge-in point at {lo:.0%}-{hi:.0%} of the last answer",
   f"{fr_b.min():.3f}-{fr_b.max():.3f}, mean {fr_b.mean():.3f}")  # fmt: skip
ck(cb["kept_after_cut"] == 0 and cb["short_cut"] == 0 and cb["user_before_cut"] == 0,
   "barge: cut at the first word boundary at/after the barge-in (no kept onset after it, word in progress kept)",
   str({k: cb[k] for k in ("kept_after_cut", "short_cut", "user_before_cut")}))  # fmt: skip
ck(cb["dropped"] > cb["barge"], "barge: the later words are dropped (> 1 word per barge-in)", f"{cb['dropped']} words")
ck(float(np.median(st_b["unsaid"])) > 1.0, "barge: a real mid-answer interruption (median > 1 s of answer unsaid)",
   f"median {np.median(st_b['unsaid']):.2f} s")  # fmt: skip
rows_b: dict = {}
n_asm = 0
for pl in plans_b[:64]:
    for p in pl.pieces:
        if p.row not in rows_b:
            ca_, cu_, _fr, al_ = decode_codes_row(vt, p.row)
            rows_b[p.row] = (np.asarray(ca_, np.int64), np.asarray(cu_, np.int64), normalize_alignments(al_))
    kc.assemble(pl, rows_b, meta, fr=FR, onset_floor=bool(getattr(tok.cfg, "onset_floor", False)))
    n_asm += 1
ck(n_asm == 64, "barge: 64 items assemble (every frame claimed exactly once)")
print(
    f"    old default (lead rule) for contrast: barge-in at {np.mean(barge_stats(plans)['fracs']):.3f} of the answer on average"
)
st_mut = barge_stats(plan_run(kc.ConcatCfg.from_dict({**SCHEME, "barge_prob": 1.0})))
fm = st_mut["fracs"]
ck(not (fm.min() >= lo - 1e-9 and fm.max() <= hi + 1e-9), "CAN FAIL: the old lead rule at barge_prob 1 fails the position check",
   f"{fm.min():.3f}-{fm.max():.3f}")  # fmt: skip

print("  [6e] launcher")
ck('knowledge_meta = cfg.get("knowledge_meta")' in src_l and 'knowledge_concat = cfg.get("knowledge_concat")' in src_l,
   "reads knowledge_meta / knowledge_concat")  # fmt: skip
ck(
    "knowledge_meta=knowledge_meta" in src_l and "knowledge_concat=knowledge_concat" in src_l,
    "routes them to the mix loader",
)

print()
if fails:
    print(f"FAILED: {len(fails)} check(s): {fails}")
    sys.exit(1)
print("all checks passed")
