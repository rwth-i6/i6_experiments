"""Guard: the THREE-way PersonaPlex mix with per-corpus row weights, flagged-word spans and the 1-frame audio pad
(POLICY_2026_09_29, user 2026-09-29: the new ablation baseline ab2_base_0929 = ab2_base_0928b + the Open Yap stratum +
"wordmask 0, but also extend by a frame in each direction").

WHY. Until 09-29 the aux (Open Yap) stream refused row weights and word spans, so the combination is new code: the aux
rows must get THEIR corpus's weights and spans (indexed by the aux table's rows, never the primary's), every stream must
yield one arity, the pad must widen exactly the AUDIO span by one frame on each side (clipped to the dialogue) and leave
the text weight alone, and none of it may change which rows are read or any per-item draw. Real corpora, CPU, the real
loader (``build_knowledge_mix_loader``) with item_meta:
  1. DRAWS. With vs without every weight knob: identical codes, loss masks and (source, row, swap) per slot.
  2. ARITY. Every batch carries a [B, T, 3] third tensor; knowledge items are all ones. Items are compared up to their
     OWN last frame: the collate pads a shorter item's weights with 0, where its loss mask is 0 too (checked).
  3. ROW WEIGHTS. Column 0 of a primary / aux item = its OWN corpus's weight of its row on the dialogue frames, 1 on the
     prompt.
  4. AUDIO SPANS + PAD. Column 2 = 0 exactly on [floor(s * fr) - pad, ceil(e * fr) + pad) of each hit of THIS read's
     assistant side (``"other"`` when swapped), clipped to the dialogue, and 1 elsewhere -- for pad 1 and for pad 0.
     CAN FAIL: the pad-0 tensor differs from the pad-1 expectation (on at least one item).
  5. TEXT. Column 1 is the same with pad 0 and pad 1 (the pad is audio-only) and has zeros in primary AND aux items.
  6. CAN FAIL: aux items checked against the PRIMARY corpus's spans (the index mix-up this guard exists for) differ.
  7. REFUSALS. train_row_weights without aux_row_weights, aux_word_spans without train_word_spans, a pad without spans,
     a negative pad, and train_censor with the aux corpus all raise.

Fixtures (as check_aux_mix): primary = p4's corpus with a seeded 20k-row subset, aux = p3's corpus (the Open Yap attach
schema) with a seeded 5k-row subset, knowledge = the v6 codes. Spans are synthetic but REAL words: 2 seeded words of a
seeded 40% of each subset's rows, per stored side, taken from the rows' own alignments (the loader raises on a span that
is not a word of the row). Row weights: 0.3 on a seeded 30% of rows.

moshi_family venv, CPU, ~4 min (setup root):
    ./hpc-venv.py --cluster i6-rz moshi_family --sh 'cd <setup> && HF_HUB_OFFLINE=1 python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_mix_word_weights.py'
"""

import json
import math
import os
import sys
import tempfile

SETUP = os.getcwd()
sys.path.insert(0, os.path.join(SETUP, "recipe", "speech_llm", "full_duplex"))
sys.path.insert(0, os.path.join(SETUP, "recipe"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from moshi_family.personaplex import train_data as td  # noqa: E402
from moshi_family.train_data_common import normalize_alignments, read_arrow_table  # noqa: E402

P4, P3, V6 = "output/persona_v4/p1_data", "output/persona_v3/p1_data", "output/persona_knowledge/v6_codes"
B, SHARE, AUX, NB = 8, 0.3, 0.35, 12
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
FR = float(tok.frame_rate)
tmp = tempfile.mkdtemp(prefix="mixword_", dir="/tmp")
t4, t3 = read_arrow_table(P4), read_arrow_table(P3)
rng = np.random.default_rng(0)
rows4 = np.sort(rng.choice(t4.num_rows, 20000, replace=False)).astype(np.int64)
rows3 = np.sort(rng.choice(t3.num_rows, 5000, replace=False)).astype(np.int64)
np.save(f"{tmp}/rows4.npy", rows4)
np.save(f"{tmp}/rows3.npy", rows3)


def fixtures(table, rows, name, seed):
    """Row weights (0.3 on 30% of rows) and spans (2 real words per stored side on 40% of the subset's rows)."""
    r = np.random.default_rng(seed)
    w = np.ones(table.num_rows, np.float32)
    w[r.random(table.num_rows) < 0.3] = 0.3
    np.save(f"{tmp}/{name}_w.npy", w)
    spans = {}
    cols = set(table.column_names)
    for row in rows[r.random(len(rows)) < 0.4]:
        sides = {}
        for side, col in (("assistant", "alignments"), ("other", "alignments_other")):
            if col not in cols:
                continue
            words = normalize_alignments(table.column(col)[int(row)].as_py())
            if len(words) < 4:
                continue
            pick = sorted(r.choice(len(words), 2, replace=False))
            sides[side] = [[[float(words[i][1][0]), float(words[i][1][1])]] for i in pick]
        if sides:
            spans[str(int(row))] = sides
    json.dump({"num_rows": table.num_rows, "rows": spans}, open(f"{tmp}/{name}_spans.json", "w"))
    return w, td.load_word_spans(f"{tmp}/{name}_spans.json", table.num_rows)


w4, s4 = fixtures(t4, rows4, "p4", 1)
w3, s3 = fixtures(t3, rows3, "p3", 2)
BASE = dict(
    batch_size=B,
    share=SHARE,
    seed=0,
    system_prompt_key="context",
    voice_codes_key="voice_codes",
    swap_channels=True,
    stored_user_augment=True,
    train_rows=f"{tmp}/rows4.npy",
    aux_path=P3,
    aux_share=AUX,
    aux_rows=f"{tmp}/rows3.npy",
)
WEIGHTS = dict(
    train_row_weights=f"{tmp}/p4_w.npy",
    aux_row_weights=f"{tmp}/p3_w.npy",
    train_word_spans=f"{tmp}/p4_spans.json",
    aux_word_spans=f"{tmp}/p3_spans.json",
    train_word_weight=0.0,
)


def run(**kw):
    meta = []
    it = td.build_knowledge_mix_loader(P4, V6, tok, item_meta=meta, **{**BASE, **kw})
    return [next(it) for _ in range(NB)], meta


plain, m_plain = run()
pad1, m_pad1 = run(**WEIGHTS, train_word_pad_frames=1)
pad0, m_pad0 = run(**WEIGHTS)

# ---- 1: draws ------------------------------------------------------------------------------------------------------
key = [(m["source"], m["row"], m["swap"]) for m in m_plain]
same = all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(plain, pad1)) and all(
    torch.equal(a[0], b[0]) for a, b in zip(plain, pad0)
)
check(
    "[1] codes / loss masks / (source, row, swap) identical with and without the weight knobs",
    same
    and key
    == [(m["source"], m["row"], m["swap"]) for m in m_pad1]
    == [(m["source"], m["row"], m["swap"]) for m in m_pad0],
    f"{len(key)} items",
)

# ---- 2..6 ------------------------------------------------------------------------------------------------------------
arity = all(len(b) == 3 and b[2].shape == (b[0].shape[0], b[0].shape[2], 3) for b in pad1 + pad0)
check("[2] every batch carries a [B, T, 3] third tensor", arity)


def dialogue(mask):
    idx = torch.nonzero(mask > 0).flatten()
    off, n = int(idx[0]), int(idx.numel())
    return off, n, bool(torch.equal(idx, torch.arange(off, off + n)))


def expected_audio(hits, off, n, T, pad):
    e = torch.ones(T)
    for hit in hits:
        f0 = max(0, int(math.floor(min(s for s, _ in hit) * FR)) - pad)
        f1 = min(n, int(math.ceil(max(e_ for _, e_ in hit) * FR)) + pad)
        if f1 > f0:
            e[off + f0 : off + f1] = 0.0
    return e


ok_k = ok_w = ok_a1 = ok_a0 = ok_t = contiguous = pad_ok = True
pad0_vs_pad1 = wrong_spans = 0
text_zero = {"primary": 0, "aux": 0}
n_hit = {"primary": 0, "aux": 0}
for bi in range(NB):
    for i in range(B):
        m = m_pad1[bi * B + i]
        wt1, wt0 = pad1[bi][2][i], pad0[bi][2][i]
        mask = pad1[bi][1][i]
        off, n, cont = dialogue(mask)
        contiguous &= cont
        T = off + n  # the item's own length; the collate pads the rest (weights 0, loss mask 0)
        pad_ok &= bool(torch.all(wt1[T:] == 0)) and bool(torch.all(mask[T:] == 0))
        wt1, wt0 = wt1[:T], wt0[:T]
        if m["source"] == "knowledge":
            ok_k &= bool(torch.all(wt1 == 1)) and bool(torch.all(wt0 == 1))
            continue
        own_w, own_s, other_s = (w4, s4, s3) if m["source"] == "primary" else (w3, s3, s4)
        # 3: row weight
        ew = torch.ones(T)
        ew[off : off + n] = float(own_w[m["row"]])
        ok_w &= bool(torch.allclose(wt1[:, 0], ew))
        # 4: audio spans + pad
        side = "other" if m["swap"] else "assistant"
        hits = own_s.get(int(m["row"]), {}).get(side, [])
        n_hit[m["source"]] += bool(hits)
        ok_a1 &= bool(torch.equal(wt1[:, 2], expected_audio(hits, off, n, T, 1)))
        ok_a0 &= bool(torch.equal(wt0[:, 2], expected_audio(hits, off, n, T, 0)))
        pad0_vs_pad1 += not torch.equal(wt0[:, 2], expected_audio(hits, off, n, T, 1))
        # 5: text unchanged by the pad
        ok_t &= bool(torch.equal(wt1[:, 1], wt0[:, 1]))
        text_zero[m["source"]] += bool(torch.any(wt1[:, 1] == 0))
        # 6: the wrong corpus's spans
        if m["source"] == "aux":
            wrong = other_s.get(int(m["row"]), {}).get(side, [])
            wrong_spans += not torch.equal(wt1[:, 2], expected_audio(wrong, off, n, T, 1))
check("[2] knowledge items are all ones", ok_k)
check("[2] past an item's own end: weights 0 and loss mask 0 (collate padding)", pad_ok)
check("[3] the dialogue frames are one contiguous run after the prompt", contiguous)
check("[3] column 0 = the item's OWN corpus's row weight on its dialogue, 1 on the prompt", ok_w)
check("[4] column 2 = 0 exactly on each hit's span widened by 1 frame (pad 1)", ok_a1, f"items with hits {n_hit}")
check("[4] ... and on the unwidened span with pad 0", ok_a0)
check("[4] CAN FAIL: the pad-0 tensor differs from the pad-1 expectation", pad0_vs_pad1 > 0, f"{pad0_vs_pad1} items")
check(
    "[5] column 1 (text) identical with pad 0 and 1, with zeros in primary AND aux items",
    ok_t and text_zero["primary"] > 0 and text_zero["aux"] > 0,
    f"{text_zero}",
)
check("[6] CAN FAIL: aux items against the PRIMARY corpus's spans differ", wrong_spans > 0, f"{wrong_spans} aux items")

# ---- 7: refusals ---------------------------------------------------------------------------------------------------
for name, kw in (
    ("train_row_weights without aux_row_weights", dict(train_row_weights=f"{tmp}/p4_w.npy")),
    (
        "aux_word_spans without train_word_spans",
        dict(aux_word_spans=f"{tmp}/p3_spans.json", train_word_weight=0.0),
    ),
    ("a pad without spans", dict(train_word_pad_frames=1)),
    ("a negative pad", dict(**WEIGHTS, train_word_pad_frames=-1)),
    ("train_censor with the aux corpus", dict(train_censor="/nonexistent")),
):
    try:
        next(td.build_knowledge_mix_loader(P4, V6, tok, **{**BASE, **kw}))
        check(f"[7] refuses {name}", False)
    except (ValueError, AssertionError):
        check(f"[7] refuses {name}", True)

print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
