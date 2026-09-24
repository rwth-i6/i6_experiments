"""Guard: the "floor-available" text-loss weight (``floor_weight``, ablation ab_epad1_floor) is CAUSAL and
class-blind, so it moves gradient without shifting what the model learns to predict.

WHY. A weight on a target CLASS (``text_pad_weight``, ``text_epad_weight``) moves the optimum of the
cross-entropy to ``q ~ w * p``: up-weighting EPAD would teach the model to start more turns than the data.
A weight that is a function of the CONTEXT only -- the same for every target of a frame -- cancels in the
optimum. That holds only if the context never looks at the target: a "floor" built around the real onset
(or counting the word whose EPAD / first token is the target) leaks the label and brings the bias back,
and nothing in a loss curve shows it. Everything runs through the real ``build_data_loader`` on real rows
of p3's corpus (the corpus the ablations train on) and the real ``duplex_ce_loss``.

  1. WIRING. The loader's third tensor equals ``floor_available_frames`` recomputed independently from the
     row (both sides' words), x ``floor_weight`` on the dialogue, 1 on the prompt prefix; values {1, alpha}.
  2. CAUSALITY (real loader). For assistant words with onset frame f, a copy of the row with every word of
     EITHER side starting at frame >= f removed gives the identical weight on frames 0..f -- frame f-1
     (the EPAD target) and f (the first-token target) included -- while those targets did change.
  3. CLASS-BLIND. On floor frames PAD, EPAD and word targets all occur, and the loss equals the manual
     ``sum(ce * class_w * frame_w) / sum(class_w * frame_w)``; ignoring the frame weight fails it.
  4. OFF == OFF. ``floor_weight=1`` yields the same codes and mask as no floor weight, and a loss
     bit-identical to ab_epad1's (``text_frame_weight=None``).
  5. CAN FAIL. With ``floor_available_frames`` swapped for a leaky variant (counting a word from its own
     onset frame, i.e. the current target) and for a future-onset variant, check 2 fails.

Also prints the floor-available share, the target mix on floor frames, and a few rows' weight strips.

Runs in moshi_family_venv, CPU, ~1-2 min (mimi + spm from the HF cache):
    ./hpc-venv.py --cluster i6-rz moshi_family --sh 'cd <setup> && python \
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_floor_weight.py'
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset, load_from_disk  # noqa: E402

import moshi_family.personaplex.train_data as ptd  # noqa: E402
from moshi_family.personaplex.train_data import (  # noqa: E402
    PersonaPlexDataConfig,
    PersonaPlexTokenizer,
    build_data_loader,
)
from moshi_family.train_data_common import (  # noqa: E402
    TEXT_EPAD_ID,
    TEXT_PADDING_ID,
    TEXT_ROW,
    duplex_ce_loss,
    floor_available_frames,
    normalize_alignments,
)

P3 = "output/persona_v3/p1_data"  # AttachPersonaPrompts of p3 (what AB_ARMS train on)
ALPHA, QUIET_MS, N_ROWS, SEED = 3.0, 200.0, 24, 7
fails = []


def ck(c, label, extra=""):
    print(f"  [{'ok' if c else 'FAIL'}] {label}{'  ' + extra if extra else ''}", flush=True)
    if not c:
        fails.append(label)


tok = PersonaPlexTokenizer(
    "nvidia/personaplex-7b-v1",
    device="cpu",
    cfg=PersonaPlexDataConfig(duration_sec=60.0, emit_epad=True, onset_floor=True, word_overlap="drop"),
)
FR = tok.frame_rate
ds = load_from_disk(P3)
rng = np.random.default_rng(SEED)
rows = [int(i) for i in rng.choice(len(ds), size=4 * N_ROWS, replace=False) if ds[int(i)]["other_ok"]][:N_ROWS]
tmp = tempfile.mkdtemp(prefix="floorw_")


def load_one(row: dict, name: str, floor_weight=ALPHA):
    """One row through the REAL loader (same seed, so the same prompt draw) -> (codes [K,T], mask [T], w [T]|None)."""
    d = os.path.join(tmp, name)
    Dataset.from_list([row], features=ds.features).save_to_disk(d)
    b = next(
        build_data_loader(
            d,
            tok,
            batch_size=1,
            seed=0,
            system_prompt_key="context",
            voice_codes_key="voice_codes",
            infinite=False,
            floor_weight=floor_weight,
            floor_quiet_ms=QUIET_MS,
        )
    )
    return b[0][0], b[1][0], (b[2][0, :, 0] if len(b) == 3 else None)


def onset_frame(start_s: float) -> int:
    return int(start_s * FR // 1)


def spans_of(row: dict) -> list:
    return [s for _w, s, _k in normalize_alignments(row["alignments"]) + normalize_alignments(row["alignments_other"])]


def drop_from(al, f: int):
    """The alignment list without words whose onset frame is >= f (either on-disk spelling)."""
    keep = []
    for a in al:
        s = a["start"] if isinstance(a, dict) else a[1][0]
        if onset_frame(float(s)) < f:
            keep.append(a)
    return keep


def causality(label: str, rows_to_use, max_cuts: int = 6) -> tuple[int, int]:
    """Check 2 on the given rows; returns (#cuts compared, #cuts where the prefix weight differed)."""
    n_cmp = n_bad = 0
    for i in rows_to_use:
        row = ds[i]
        codes, mask, w = load_one(row, f"{label}_{i}_full")
        p = int((~mask).sum())
        onsets = sorted({onset_frame(s) for _w, (s, _e), _k in normalize_alignments(row["alignments"])})
        # Informative cuts: words that start on a FLOOR frame (quiet before them). Mid-utterance words sit
        # on frames that are busy anyway, where even a leaky variant would give the same weight.
        # (chosen with the REAL floor function, imported from train_data_common, so a patched loader in
        # check 5 is tested on the same cuts)
        real_free = floor_available_frames(spans_of(row), int(mask.sum()), FR, QUIET_MS / 1000.0)
        onsets = [f for f in onsets if 2 <= f < int(mask.sum()) - 1 and real_free[f]][:max_cuts]
        for f in onsets:
            cut = dict(row)
            cut["alignments"] = drop_from(row["alignments"], f)
            cut["alignments_other"] = drop_from(row["alignments_other"], f)
            c2, m2, w2 = load_one(cut, f"{label}_{i}_cut{f}")
            tgt_changed = bool((codes[TEXT_ROW, p + f - 1 : p + f + 1] != c2[TEXT_ROW, p + f - 1 : p + f + 1]).any())
            same = torch.equal(w[: p + f + 1], w2[: p + f + 1])
            n_cmp += 1
            n_bad += int(not same)
            if not tgt_changed:
                fails.append(f"{label}: cut at {f} did not change the targets (vacuous)")
    return n_cmp, n_bad


print("1. wiring: loader weight == floor_available_frames on the row", flush=True)
stats = {"frames": 0, "floor": 0, "pad": 0, "epad": 0, "word": 0, "epad_all": 0, "epad_floor": 0}
# Turn starts: EPAD targets after >= 1 s without an assistant word (onset-based, the assistant's own words).
# How many of them fall on floor frames, and on the frames where only the ASSISTANT has been quiet 1 s (a
# causal alternative context: "the model is listening and must decide whether to start").
turn = {"n": 0, "on_floor": 0, "asst_quiet_frames": 0}
strips = []
for i in rows:
    row = ds[i]
    codes, mask, w = load_one(row, f"w_{i}")
    p, n = int((~mask).sum()), int(mask.sum())
    free = floor_available_frames(spans_of(row), n, FR, QUIET_MS / 1000.0)
    want = torch.ones(codes.shape[1])
    want[p:] = torch.where(torch.from_numpy(free), ALPHA, 1.0)
    ck(torch.equal(w, want), f"row {i}: weight == recomputed floor x {ALPHA}", f"floor share {free.mean():.3f}")
    ck(
        bool((w[:p] == 1).all()) and set(w.unique().tolist()) <= {1.0, ALPHA},
        f"row {i}: prefix 1, values in {{1, {ALPHA}}}",
    )
    tgt = codes[TEXT_ROW, p:]
    fl = torch.from_numpy(free)
    stats["frames"] += n
    stats["floor"] += int(fl.sum())
    stats["pad"] += int((fl & (tgt == TEXT_PADDING_ID)).sum())
    stats["epad"] += int((fl & (tgt == TEXT_EPAD_ID)).sum())
    stats["word"] += int((fl & (tgt != TEXT_PADDING_ID) & (tgt != TEXT_EPAD_ID)).sum())
    stats["epad_all"] += int((tgt == TEXT_EPAD_ID).sum())
    stats["epad_floor"] += int((fl & (tgt == TEXT_EPAD_ID)).sum())
    asst_spans = [sp for _w, sp, _k in normalize_alignments(row["alignments"])]
    asst_quiet = torch.from_numpy(floor_available_frames(asst_spans, n, FR, 1.0))
    ts = (tgt == TEXT_EPAD_ID) & torch.roll(asst_quiet, -1)  # EPAD at t-1 of a word whose frame t is 1 s quiet
    turn["n"] += int(ts.sum())
    turn["on_floor"] += int((ts & fl).sum())
    turn["asst_quiet_frames"] += int(asst_quiet.sum())
    on_floor_epad = torch.nonzero(fl & (tgt == TEXT_EPAD_ID))
    if len(strips) < 3 and len(on_floor_epad):
        e = int(on_floor_epad[0])
        a, b = max(0, e - 20), min(n, e + 20)
        sym = "".join("E" if t == TEXT_EPAD_ID else ("." if t == TEXT_PADDING_ID else "w") for t in tgt[a:b].tolist())
        wt = "".join("3" if x else "1" for x in free[a:b])
        strips.append(f"row {i} frames {a}-{b}\n    target {sym}\n    weight {wt}")

print(
    f"\n   floor-available: {stats['floor'] / stats['frames']:.3f} of {stats['frames']} dialogue frames "
    f"({len(rows)} rows); on floor frames: PAD {stats['pad'] / stats['floor']:.3f}, EPAD "
    f"{stats['epad'] / stats['floor']:.3f}, word {stats['word'] / stats['floor']:.3f}; EPAD targets on floor "
    f"frames: {stats['epad_floor'] / max(1, stats['epad_all']):.3f} of {stats['epad_all']}"
)
print(
    f"   turn starts (EPAD after >= 1 s of assistant silence): {turn['n']}, on floor frames "
    f"{turn['on_floor'] / max(1, turn['n']):.3f}; frames with the assistant quiet >= 1 s: "
    f"{turn['asst_quiet_frames'] / stats['frames']:.3f}"
)
for s in strips:
    print("   " + s)

print("\n2. causality through the real loader (remove every word starting at >= f; frames 0..f unchanged)", flush=True)
FLOOR_ROWS = rows[:12]  # cuts are taken where a word starts on a floor frame
n_cmp, n_bad = causality("real", FLOOR_ROWS)
ck(
    n_cmp >= 8 and n_bad == 0,
    "weight on frames <= f independent of words starting at >= f",
    f"{n_cmp} cuts, {n_bad} differ",
)

print("\n3. class-blind: every class on floor frames, loss == manual context-weighted CE", flush=True)
ck(stats["pad"] > 0 and stats["epad"] > 0 and stats["word"] > 0, "PAD, EPAD and word targets all occur on floor frames")
row = ds[rows[0]]
codes, mask, w = load_one(row, "loss")
B, K, T = 1, codes.shape[0], codes.shape[1]
g = torch.Generator().manual_seed(0)


class Out:
    pass


o = Out()
o.text_logits = torch.randn(B, 1, T, 32000, generator=g)
o.text_mask = torch.ones(B, 1, T, dtype=torch.bool)
o.logits = torch.randn(B, 16, T, 2048, generator=g)
o.mask = torch.ones(B, 16, T, dtype=torch.bool)
cb, mb, wb = codes[None], mask[None], w[None]
kw = dict(
    dep_q=16,
    other_audio_weight=0.02,
    text_pad_weight=0.3,
    text_epad_id=TEXT_EPAD_ID,
    text_epad_weight=1.0,
    weighted_mean=True,
    return_components=True,
)
_, comp = duplex_ce_loss(o, cb, mb, text_frame_weight=wb, **kw)
tgt = cb[:, TEXT_ROW].reshape(-1).clamp(0, 31999)  # the prefix's initial token is out of range; masked below
ce = torch.nn.functional.cross_entropy(o.text_logits[:, 0].reshape(-1, 32000).float(), tgt, reduction="none")
cls_w = torch.where(tgt == TEXT_PADDING_ID, 0.3, 1.0)
cls_w = torch.where(tgt == TEXT_EPAD_ID, 1.0, cls_w)
valid = mb.reshape(-1).float()
manual = (ce * cls_w * wb.reshape(-1) * valid).sum() / (cls_w * wb.reshape(-1) * valid).sum()
manual_off = (ce * cls_w * valid).sum() / (cls_w * valid).sum()
_, comp_none = duplex_ce_loss(o, cb, mb, text_frame_weight=None, **kw)
# components carry the (unscaled) text loss
tl = comp.get("text_loss")
tl_none = comp_none.get("text_loss")
ck(
    tl is not None and abs(float(tl) - float(manual)) < 1e-4,
    "text loss == manual context-weighted CE",
    f"{float(tl):.6f} vs {float(manual):.6f}",
)
ck(
    abs(float(tl_none) - float(manual)) > 1e-4 and abs(float(tl_none) - float(manual_off)) < 1e-4,
    "ignoring the frame weight changes it (the check can fail)",
)
ck(
    abs(comp["text_frame_weighted_frac"] - float((w[mask] != 1).float().mean())) < 1e-6,
    "text_frame_weighted_frac == floor share of valid frames",
)

print("\n4. floor_weight=1 == off", flush=True)
c1, m1, w1 = load_one(row, "alpha1", floor_weight=1.0)
c0, m0, w0 = load_one(row, "off", floor_weight=None)
ck(w0 is None and torch.equal(c1, c0) and torch.equal(m1, m0), "same codes and mask with and without the floor weight")
l1, _ = duplex_ce_loss(o, c1[None], m1[None], text_frame_weight=w1[None], **kw)
l0, _ = duplex_ce_loss(o, c0[None], m0[None], text_frame_weight=None, **kw)
ck(torch.equal(l1, l0), "loss at floor_weight=1 bit-identical to ab_epad1's")

print("\n5. the causality check can fail", flush=True)
real_fn = ptd.floor_available_frames


def leaky_own_onset(word_spans, num_frames, frame_rate, quiet_sec):
    """Counts a word from its own onset frame (<= t): the frame whose target is its first token sees it."""
    t_sec = np.arange(num_frames) / frame_rate
    frames = np.arange(num_frames)
    busy = np.zeros(num_frames, dtype=bool)
    for s, e in word_spans:
        busy |= (int(s * frame_rate // 1) <= frames) & (e > t_sec - quiet_sec)
    return ~busy


def leaky_future_onset(word_spans, num_frames, frame_rate, quiet_sec):
    """Floor = the 0.5 s BEFORE any word onset (a region around the real onset): uses the future."""
    frames = np.arange(num_frames)
    free = np.zeros(num_frames, dtype=bool)
    for s, _e in word_spans:
        f = int(s * frame_rate // 1)
        free |= (frames >= f - int(0.5 * frame_rate)) & (frames < f)
    return free


for name, fn in (("own-onset", leaky_own_onset), ("future-onset", leaky_future_onset)):
    ptd.floor_available_frames = fn
    try:
        n_cmp, n_bad = causality(name, FLOOR_ROWS)
    finally:
        ptd.floor_available_frames = real_fn
    ck(n_bad > 0, f"{name} variant is caught", f"{n_bad} of {n_cmp} cuts differ")

print()
if fails:
    print(f"FAIL ({len(fails)}):", *fails, sep="\n  ")
    sys.exit(1)
print("ALL OK")
