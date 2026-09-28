"""Guard: the THREE-way PersonaPlex mix (``build_knowledge_mix_loader(aux_path=, aux_share=, aux_rows=)``; ablation
ab2_mix_jre_oy_know, 2026-09-28: JRE + Open Yap + knowledge) puts exactly the declared rows in every batch.

WHY. A third corpus that enters at a share that only holds in expectation, reads outside its row subset, shares a
random stream with the primary (and so silently re-orders the JRE rows the baseline trained on), or skips wrongly on
resume would train and look fine. Real corpora, CPU, the real loader with item_meta (which records each item's
source and table row):
  1. SCHEDULE. Over 40 batches of 8 at share 0.3 + aux_share 0.35: per batch primary / knowledge / aux counts ==
     8 - k - a / ``stratified_share_count(0.3, 8, i)`` / ``stratified_share_count(0.35, 8, i)``; the running aux share
     is exactly 0.35 at every multiple of 20 batches.
  2. STREAMS. The primary items are, one for one, the head of the primary stream (``row_item_stream``, seed 0, its
     ``train_rows``); the knowledge items are the knowledge items the TWO-way mix (no aux) draws, in order; the aux
     items are the head of ``row_item_stream(aux, seed=[0, AUX_STREAM_TAG], rows=aux_rows)`` and all lie in
     ``aux_rows``. CAN FAIL: the aux order vs a stream seeded like the primary (seed 0) differs.
  3. RESUME. ``skip_batches=7`` yields, slot for slot, the same (source, row) as batches 7.. of the fresh loader (per-item
     draws are not replayed on resume, as in the two-way mix).
  4. REFUSALS. aux_share without aux_path, share + aux_share >= 1, and aux with train_row_weights raise.
  5. RENDER + LAUNCHER. ``finetune`` renders aux_data / aux_share / aux_train_rows only when set and refuses them
     without knowledge_data; the launcher reads the three keys (``check_launcher_config_reads`` covers the reads).

Fixtures: primary = p4's corpus (persona_v4/p1_data) with a seeded 20k-row subset, aux = p3's corpus (persona_v3/p1_data,
same schema as the Open Yap attach output) with a seeded 5k-row subset, knowledge = the v6 codes (no concat).

moshi_family venv, CPU, ~3 min (setup root):
    ./hpc-venv.py --cluster i6-rz moshi_family --sh 'cd <setup> && HF_HUB_OFFLINE=1 python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_aux_mix.py'
"""

import os
import sys
import tempfile

SETUP = os.getcwd()
sys.path.insert(0, os.path.join(SETUP, "recipe", "speech_llm", "full_duplex"))
sys.path.insert(0, os.path.join(SETUP, "recipe"))

import numpy as np  # noqa: E402

from moshi_family.personaplex import train_data as td  # noqa: E402
from moshi_family.train_data_common import read_arrow_table, row_item_stream, stratified_share_count  # noqa: E402

P4, P3, V6 = "output/persona_v4/p1_data", "output/persona_v3/p1_data", "output/persona_knowledge/v6_codes"
B, SHARE, AUX, NB = 8, 0.3, 0.35, 40
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


tok = td.PersonaPlexTokenizer(
    "nvidia/personaplex-7b-v1",
    device="cpu",
    cfg=td.PersonaPlexDataConfig(
        duration_sec=60.0, silence_frames=6, emit_epad=True, onset_floor=True, word_overlap="drop"
    ),
)
tmp = tempfile.mkdtemp(prefix="auxmix_", dir="/tmp")
n4, n3 = read_arrow_table(P4).num_rows, read_arrow_table(P3).num_rows
rng = np.random.default_rng(0)
rows4 = np.sort(rng.choice(n4, 20000, replace=False)).astype(np.int64)
rows3 = np.sort(rng.choice(n3, 5000, replace=False)).astype(np.int64)
np.save(f"{tmp}/rows4.npy", rows4)
np.save(f"{tmp}/rows3.npy", rows3)
KW = dict(
    batch_size=B,
    share=SHARE,
    seed=0,
    system_prompt_key="context",
    voice_codes_key="voice_codes",
    swap_channels=True,
    stored_user_augment=True,
    train_rows=f"{tmp}/rows4.npy",
)


def run(n, meta, **kw):
    it = td.build_knowledge_mix_loader(P4, V6, tok, item_meta=meta, **{**KW, **kw})
    return [next(it) for _ in range(n)]


# ---- 1 + 2 --------------------------------------------------------------------------------------------------------
meta3 = []
b3 = run(NB, meta3, aux_path=P3, aux_share=AUX, aux_rows=f"{tmp}/rows3.npy")
src = [m["source"] for m in meta3]
ok, n_aux = True, 0
for i in range(NB):
    k, a = stratified_share_count(SHARE, B, i), stratified_share_count(AUX, B, i)
    got = src[i * B : (i + 1) * B]
    ok &= got == ["primary"] * (B - k - a) + ["knowledge"] * k + ["aux"] * a
    n_aux += a
    if (i + 1) % 20 == 0:
        ok &= n_aux == round(AUX * B * (i + 1))
check(
    "[1] per-batch primary / knowledge / aux counts == the stratified schedule",
    ok and len(src) == NB * B,
    f"aux {n_aux}/{NB * B}",
)

prim = [m["row"] for m in meta3 if m["source"] == "primary"]
auxr = [m["row"] for m in meta3 if m["source"] == "aux"]
know = [m["row"] for m in meta3 if m["source"] == "knowledge"]
ref_p = row_item_stream(read_arrow_table(P4), lambda i, _r: i, seed=0, rows=rows4)
check("[2] primary items = head of the primary stream", prim == [next(ref_p) for _ in prim])
ref_a = row_item_stream(read_arrow_table(P3), lambda i, _r: i, seed=[0, td.AUX_STREAM_TAG], rows=rows3)
check(
    "[2] aux items = head of the aux stream, inside aux_rows",
    auxr == [next(ref_a) for _ in auxr] and set(auxr) <= set(rows3.tolist()),
)
wrong = row_item_stream(read_arrow_table(P3), lambda i, _r: i, seed=0, rows=rows3)
check("[2] can fail: an aux stream seeded like the primary differs", auxr != [next(wrong) for _ in auxr])
meta2 = []
run(NB, meta2)
know2 = [m["row"] for m in meta2 if m["source"] == "knowledge"]
check("[2] knowledge items = the two-way mix's, in order", know == know2[: len(know)] and len(know) == len(know2))

# ---- 3: resume ----------------------------------------------------------------------------------------------------
# per-item draws (swap, prompt level, user variant) are not replayed on resume (``row_item_stream``), so the check is on
# WHICH rows each batch holds, per source and slot
meta_r = []
r = td.build_knowledge_mix_loader(
    P4, V6, tok, skip_batches=7, item_meta=meta_r, aux_path=P3, aux_share=AUX, aux_rows=f"{tmp}/rows3.npy", **KW
)
for _ in range(NB - 7):
    next(r)
fresh = [(m["source"], m["row"]) for m in meta3[7 * B :]]
got = [(m["source"], m["row"]) for m in meta_r]
check(
    "[3] skip_batches=7: the same rows per slot as batches 7.. of the fresh loader", got == fresh, f"{len(got)} items"
)
# ---- 4: refusals --------------------------------------------------------------------------------------------------
for name, kw in (
    ("aux_share without aux_path", dict(aux_share=0.2)),
    ("share + aux_share >= 1", dict(aux_path=P3, aux_share=0.7)),
    ("aux + train_row_weights", dict(aux_path=P3, aux_share=0.2, train_row_weights=f"{tmp}/rows4.npy")),
):
    try:
        next(td.build_knowledge_mix_loader(P4, V6, tok, **{**KW, **kw}))
        check(f"[4] refuses {name}", False)
    except (ValueError, AssertionError):
        check(f"[4] refuses {name}", True)

# ---- 5: render ----------------------------------------------------------------------------------------------------
src_ft = open(os.path.join(SETUP, "recipe/i6_experiments/users/dorian_koch/speech_llm/finetune.py")).read()
check(
    "[5] finetune renders aux_data / aux_share / aux_train_rows",
    all(f"{k}: " in src_ft for k in ("aux_data", "aux_share", "aux_train_rows")),
)
src_l = open(os.path.join(SETUP, "recipe/speech_llm/full_duplex/moshi_family/personaplex/finetune_launcher.py")).read()
check("[5] launcher reads them", all(f'cfg.get("{k}")' in src_l for k in ("aux_data", "aux_share", "aux_train_rows")))
print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
