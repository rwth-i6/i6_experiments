"""Guard: the loaders' ``item_meta`` recorder (analysis of what a run trained on) observes and changes nothing.

WHY. ``train_stream_replay.py`` samples a run's training items by replaying its loader with ``item_meta`` on and
labels every item with the recorded provenance (JRE row / knowledge plan, swap, prompt, augmented user stream).
If recording shifted a single rng draw, the replay would show items the run never saw; if an entry were attached
to the wrong batch slot, the page would label a knowledge dialogue as a JRE window. Both look plausible. So, on
the REAL corpora of ``ab2_know30_concat_nobarge`` (p4's JRE rows, the v6 knowledge corpus + its concat meta, the
run's knobs) through the real ``build_knowledge_mix_loader`` and ``build_data_loader``:

  1. SAME BATCHES. ``N`` batches with ``item_meta`` == without it, every tensor bit for bit (mix + plain loader).
  2. RIGHT SLOTS. ``len(item_meta) == N * B``; per batch the knowledge entries are exactly the schedule's last
     ``stratified_share_count`` slots; for every JRE entry the item's ASSISTANT codes == the stored channel the
     entry's ``swap`` names (``codes_user`` when swapped), its user codes == the other stored channel iff
     ``user_aug`` is False (and differ from it when True), and the prompt text sits in the item's prefix; for
     every concat knowledge entry the first plan piece's assistant codes open the item's dialogue.
  3. CAN FAIL. The same slot checks on the entries shifted by one slot fail; and with ``draw_stored_user_stream``
     mutated to return a COPY of the clean stream (so ``user_aug`` reads True for clean reads) check 2 fails.

Runs in moshi_family_venv, CPU (~2 min; mimi + spm from the HF cache):
    ./hpc-venv.py --cluster i6-rz moshi_family --sh 'cd <setup> && HF_HUB_OFFLINE=1 python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_loader_item_meta.py'
"""

import os
import sys

import numpy as np
import torch

SETUP = "/home/tt201262/setups/2026-01-speech-llm"
sys.path.insert(0, os.path.join(SETUP, "recipe", "speech_llm", "full_duplex"))

import moshi_family.personaplex.train_data as td  # noqa: E402
from moshi_family.train_data_common import (  # noqa: E402
    AUDIO_OFFSET,
    TEXT_ROW,
    decode_codes_row,
    read_arrow_table,
    stratified_share_count,
)

W = f"{SETUP}/work/i6_experiments/users/dorian_koch/speech_llm"
JRE = f"{W}/persona_prompts/AttachPersonaPrompts.DHvnX2ifpvad/output/dataset"
KNOW = f"{W}/persona_knowledge/KnowledgeDialogueCodes.DMgw9Srb4bmg/output/dataset"
META = f"{W}/persona_knowledge/KnowledgeConcatMeta.Jq5Gwzdr2Ray/output/meta.npz"
CONCAT = {
    "barge_lead_sec": [0.2, 0.8],
    "barge_min_answer_sec": 0.5,
    "barge_prob": 0.0,
    "dialogues": [2, 4],
    "gap_sec": [0.3, 1.5],
    "min_next_sec": 5.0,
    "tail_frames": 1,
}
B, SHARE, N = 8, 0.3, 10
KNOBS = dict(system_prompt_key="context", voice_codes_key="voice_codes", swap_channels=True, stored_user_augment=True)

FAILED = []


def ck(ok, label, extra=""):
    print(("PASS " if ok else "FAIL ") + label + (f"  {extra}" if extra else ""), flush=True)
    if not ok:
        FAILED.append(label)


tok = td.PersonaPlexTokenizer(
    "nvidia/personaplex-7b-v1",
    device="cpu",
    cfg=td.PersonaPlexDataConfig(
        duration_sec=60.0, silence_frames=6, emit_epad=True, onset_floor=True, word_overlap="drop"
    ),
)
half = tok.n_q // 2
jre_t, know_t = read_arrow_table(JRE), read_arrow_table(KNOW)


def mix(item_meta=None, seed=0):
    return td.build_knowledge_mix_loader(
        JRE,
        KNOW,
        tok,
        batch_size=B,
        share=SHARE,
        seed=seed,
        knowledge_meta=META,
        knowledge_concat=CONCAT,
        item_meta=item_meta,
        **KNOBS,
    )


def plain(item_meta=None):
    return td.build_data_loader(JRE, tok, batch_size=B, seed=0, item_meta=item_meta, **KNOBS)


def take(it, n):
    return [next(it) for _ in range(n)]


def same(a, b) -> bool:
    return all(len(x) == len(y) and all(torch.equal(u, v) for u, v in zip(x, y)) for x, y in zip(a, b))


def slot_errors(batches, metas, *, knowledge_schedule: bool) -> list[str]:
    """Check 2 on (batches, per-item entries); returns the violations (empty = all good)."""
    errs = []
    if len(metas) != len(batches) * B:
        return [f"{len(metas)} entries for {len(batches)} batches of {B}"]
    for b, (codes, mask, *_x) in enumerate(batches):
        k = stratified_share_count(SHARE, B, b) if knowledge_schedule else 0
        for i in range(B):
            m = metas[b * B + i]
            want = "knowledge" if i >= B - k else "primary"
            if m["source"] != want:
                errs.append(f"b{b}i{i}: source {m['source']} != schedule {want}")
                continue
            n = int(mask[i].sum())
            last = int(torch.nonzero(mask[i])[-1]) + 1
            p = last - n  # dialogue frames [p, last)
            asst = codes[i, AUDIO_OFFSET : AUDIO_OFFSET + half, p:last].numpy()
            user = codes[i, AUDIO_OFFSET + half :, p:last].numpy()
            if want == "knowledge":
                piece = m["plan"].pieces[0]
                ra, _ru, _fr, _al = decode_codes_row(know_t, piece.row)
                nn = min(ra.shape[1] - piece.src0, 10)
                if not np.array_equal(asst[:, :nn], ra[:, piece.src0 : piece.src0 + nn]):
                    errs.append(f"b{b}i{i}: knowledge item does not open with plan row {piece.row}")
                continue
            ca, cu, _fr, _al = decode_codes_row(jre_t, m["row"])
            a_src, u_src = (cu, ca) if m["swap"] else (ca, cu)
            if not np.array_equal(asst, a_src[:, :n]):
                errs.append(f"b{b}i{i}: assistant codes != stored {'codes_user' if m['swap'] else 'codes_assistant'}")
            clean = np.array_equal(user, u_src[:, :n])
            if m["user_aug"] is False and not clean:
                errs.append(f"b{b}i{i}: user_aug False but the user stream is not the clean stored channel")
            if m["user_aug"] is True and clean:
                errs.append(f"b{b}i{i}: user_aug True but the user stream IS the clean stored channel")
            if m["prompt"]:
                ids = tok.system_prompt_text_tokens(m["prompt"])
                row = codes[i, TEXT_ROW, :p].tolist()
                if not any(row[j : j + len(ids)] == ids for j in range(p - len(ids) + 1)):
                    errs.append(f"b{b}i{i}: the recorded prompt is not in the item's prefix")
    return errs


# 1. same batches
meta_mix: list = []
with_mix, without_mix = take(mix(meta_mix), N), take(mix(), N)
ck(same(with_mix, without_mix), f"[1] mix loader: {N} batches with item_meta == without, bit for bit")
meta_plain: list = []
with_plain, without_plain = take(plain(meta_plain), 4), take(plain(), 4)
ck(same(with_plain, without_plain), "[1] plain loader: 4 batches with item_meta == without, bit for bit")

# 2. right slots
e = slot_errors(with_mix, meta_mix, knowledge_schedule=True)
n_sw = sum(bool(m["swap"]) for m in meta_mix if m["source"] == "primary")
n_aug = sum(bool(m["user_aug"]) for m in meta_mix if m["source"] == "primary")
n_jre = sum(m["source"] == "primary" for m in meta_mix)
ck(
    not e,
    "[2] mix loader: every entry names its slot's source / row / swap / user stream / prompt",
    f"{len(meta_mix)} entries, JRE {n_jre} (swapped {n_sw}, augmented user {n_aug}); {e[:3]}",
)
ck(0 < n_sw < n_jre and 0 < n_aug < n_jre, "[2] non-vacuous: both swap outcomes and both user streams occur")
e = slot_errors(with_plain, meta_plain, knowledge_schedule=False)
ck(not e, "[2] plain loader: every entry names its slot's row / swap / user stream / prompt", f"{e[:3]}")

# 3. can fail
shifted = meta_mix[1:] + meta_mix[:1]
ck(bool(slot_errors(with_mix, shifted, knowledge_schedule=True)), "[3] entries shifted by one slot are caught")
orig = td.draw_stored_user_stream
td.draw_stored_user_stream = lambda *a, **kw: np.array(orig(*a, **kw), copy=True)
try:
    meta_mut: list = []
    batches_mut = take(plain(meta_mut), 4)
finally:
    td.draw_stored_user_stream = orig
ck(
    bool(slot_errors(batches_mut, meta_mut, knowledge_schedule=False)),
    "[3] a user_aug that misreads clean draws (copying draw_stored_user_stream) is caught",
)

print("ALL PASS" if not FAILED else f"FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
