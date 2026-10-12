"""Guard: the user-text targets are ALIGNED with the user's audio on every loader path (2026-10-12, ``knowledge_user_text``).

User (2026-10-12): "is it aligned though?" -- the user-text head's targets (the user's words, delayed 6 frames) must move
with the USER AUDIO through every transform: the prompt lead-in, the build-time onset pull-up, the concatenation's
splice offsets / gaps / barge-ins, the window crop, a swapped read, and the delay. Proven on REAL audio, not assumed:
the fixture is 20 real v6 knowledge dialogues (10 seeded rows each of ``qa_direct`` and ``multi_turn_explanation``,
seed 0) passed through the REAL builders -- ``moshi_annotate_inference.py --annotate_user`` (Whisper medium on the user
channel, test node), ``knowledge_codes --user_words``, ``knowledge_concat`` meta -- under
``/hpcwork/p0023999/tt201262/user_words/fixture/`` (``codes``, ``meta.npz``; how it was built: rl_alignment.md "User word
timings"). Every item is read through the REAL loader (``build_knowledge_mix_loader`` / ``build_data_loader``), its
USER channel Mimi-DECODED from the item's own codes, and a frame-energy VAD run on that audio (frame RMS > ``VAD_DB``
dBFS; speech runs bridged over < ``BRIDGE`` silent frames). For each target the audio frame it describes is
``t - prompt - 6``:
  * TOKENS: every word-token target lies in user speech dilated by ``TOL`` frames (0 outside is the pass bar);
  * ONSETS: every word that follows an EPAD (a word after a pause, i.e. a segment's first word) has its onset frame
    within ``TOL`` frames of the start of the VAD speech segment it falls in (or the nearest one); distribution reported.
Paths: plain (knowledge mix, no concat), concat (ab2's ``knowledge_concat`` cfg, 2-4 dialogues, gaps + barge-ins),
windowed (10 s window: rows cropped), swapped (``build_data_loader(swap_channels=True)``: the user is the stored
assistant channel, its words ``alignments``). Also:
  * ASSISTANT ROW: the codes (text row included) and masks of the knowledge items are bit-identical with
    ``knowledge_user_text`` on and off (plain + concat);
  * KNOB OFF = THE OLD CODE: on the real ab2_usertext corpora (p4 + v6 + its concat meta, swap, stored user
    augmentation, user_text_delay 6), ``build_knowledge_mix_loader`` without the knob yields batches bit-identical to the
    pre-change ``train_data.py`` + ``knowledge_concat.py`` (full_duplex ``6c2a1dca``, loaded from git), concat and not.
``--selftest``: three mutants must fail the alignment checks -- an off-by-lead-in shift (targets 25 frames = 2 s late),
a concat splice that ignores the gap (the piece offset without ``gap_frames``), a swap that keeps the old column
(``alignments_other`` on a swapped read).

moshi_family venv, CPU, ~6 min (setup root):
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && HF_HUB_OFFLINE=1 python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_user_text_alignment.py [--selftest]'
"""

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile

SETUP = os.getcwd()
FD = os.path.join(SETUP, "recipe", "speech_llm", "full_duplex")
sys.path.insert(0, FD)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import moshi_family.knowledge_concat as kcat  # noqa: E402
import moshi_family.train_data_common as tdc  # noqa: E402
from moshi_family.personaplex import train_data as td  # noqa: E402

FX = "/hpcwork/p0023999/tt201262/user_words/fixture"
P4, V6, V6_META = (
    "output/persona_v4/p1_data",
    "output/persona_knowledge/v6_codes",
    "output/persona_knowledge/v6_concat_meta.npz",
)
OLD_REV = "6c2a1dca"  # full_duplex before knowledge_user_text
AB2_CONCAT = {
    "barge_lead_sec": [0.2, 0.8],
    "barge_min_answer_sec": 0.5,
    "barge_prob": 0.3,
    "dialogues": [2, 4],
    "gap_sec": [0.3, 1.5],
    "min_next_sec": 5.0,
    "tail_frames": 1,
}
D = 6
VAD_DB = -45.0  # frame RMS threshold (dBFS) on the decoded user channel (loudness-normalised to -24 LUFS)
BRIDGE = 3  # silent runs shorter than this (frames) inside speech are bridged (stops / plosives)
TOL = 2  # frames
SEARCH = 6  # a segment's first onset is looked for from this many frames before it (an early word) to its end
LEAD_IN = 25  # the off-by-lead-in mutant: 2 s at 12.5 Hz
KW = dict(system_prompt_key="context", voice_codes_key="voice_codes")
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'} {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def tokenizer(duration_sec=60.0):
    return td.PersonaPlexTokenizer(
        "nvidia/personaplex-7b-v1",
        device="cpu",
        cfg=td.PersonaPlexDataConfig(
            duration_sec=duration_sec, silence_frames=6, emit_epad=True, onset_floor=True, word_overlap="drop"
        ),
    )


_MIMI, _AUDIO = [], {}


def user_frame_db(user_codes: torch.Tensor) -> np.ndarray:
    """Per-frame RMS (dBFS) of the item's USER channel, Mimi-decoded from the item's own codes ``[8, n]``."""
    key = hashlib.sha1(user_codes.numpy().astype(np.int64).tobytes()).hexdigest()
    if key not in _AUDIO:
        if not _MIMI:
            from huggingface_hub import hf_hub_download

            from moshi_family.personaplex.models import loaders

            _MIMI.append(loaders.get_mimi(hf_hub_download("nvidia/personaplex-7b-v1", loaders.MIMI_NAME), "cpu"))
        mimi = _MIMI[0]
        with torch.no_grad():
            wav = mimi.decode(user_codes[None].long())[0, 0].numpy()
        n = user_codes.shape[1]
        hop = int(round(mimi.sample_rate / mimi.frame_rate))
        x = np.pad(wav, (0, max(0, n * hop - wav.shape[0])))[: n * hop].reshape(n, hop)
        _AUDIO[key] = 10 * np.log10(np.mean(x.astype(np.float64) ** 2, axis=1) + 1e-12)
    return _AUDIO[key]


def speech_segments(db: np.ndarray):
    """``(speech mask, [(start, end)])``: frames over VAD_DB, silent gaps < BRIDGE frames bridged."""
    sp = db > VAD_DB
    runs, f, n = [], 0, len(sp)
    while f < n:
        if sp[f]:
            g = f
            while g < n and sp[g]:
                g += 1
            runs.append([f, g])
            f = g
        else:
            f += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < BRIDGE:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    mask = np.zeros(n, dtype=bool)
    for a, b in merged:
        mask[a:b] = True
    return mask, [tuple(r) for r in merged]


def items_of(batches, n_batches):
    """Unbatch ``(codes, mask, ..., user_text)``: per item ``(user codes [8, n], targets [n], text row [n])`` of the
    dialogue frames (mask True; the prompt before, the right padding after)."""
    out = []
    for _ in range(n_batches):
        b = next(batches)
        codes, mask, tgt = b[0], b[1], b[-1]
        for i in range(codes.shape[0]):
            m = mask[i]
            P = int(torch.nonzero(m)[0])
            n = int(m.sum())
            assert bool(m[P : P + n].all()) and bool((tgt[i, :P] == tdc.USER_TEXT_IGNORE).all())
            half = (codes.shape[1] - 1) // 2
            out.append(
                (
                    codes[i, 1 + half : 1 + 2 * half, P : P + n],  # user channel
                    tgt[i, P : P + n],  # user-text targets
                    codes[i, tdc.TEXT_ROW, P : P + n],  # assistant text row
                    codes[i, 1 : 1 + half, P : P + n],  # assistant channel
                )
            )
    return out


def score(items, ref=False):
    """Alignment of every item's user-text targets with its decoded USER audio (``ref``: of its ASSISTANT text row,
    undelayed, with its decoded assistant audio -- the main text stream's own alignment, the reference):
    ``(stats, token_out, onset_errors)``."""
    n_tok = out_tok = 0
    errs, worst, no_word_segs = [], [], []
    delay = 0 if ref else D
    for it in items:
        ucodes, tgt = (it[3], it[2]) if ref else (it[0], it[1])
        if bool((tgt == tdc.USER_TEXT_IGNORE).all()):
            continue
        db = user_frame_db(ucodes)
        mask, segs = speech_segments(db)
        n = len(mask)
        dil = mask.copy()
        for k in range(1, TOL + 1):
            dil[k:] |= mask[:-k]
            dil[:-k] |= mask[k:]
        t = tgt.numpy()
        tok = (t != tdc.TEXT_PADDING_ID) & (t != tdc.TEXT_EPAD_ID) & (t != tdc.USER_TEXT_IGNORE)
        for f in np.nonzero(tok)[0]:
            a = f - delay  # the audio frame this target describes
            n_tok += 1
            if a < 0 or a >= n or not dil[a]:
                out_tok += 1
                lo, hi = max(0, a - 4), min(n, a + 5)
                worst.append(
                    (int(a), round(float(db[a]), 1) if 0 <= a < n else None, [round(float(x)) for x in db[lo:hi]])
                )
        # word onsets after padding (the token right after an EPAD; every segment's first word is one, since a VAD
        # segment starts after >= BRIDGE silent frames); each VAD segment's FIRST such onset (searched from SEARCH
        # frames before the segment) against the segment start
        onsets = [int(f - delay) for f in np.nonzero(tok)[0] if f >= 1 and t[f - 1] == tdc.TEXT_EPAD_ID]
        for s0, e0 in segs:
            cand = [a for a in onsets if s0 - SEARCH <= a < e0]
            if not cand:
                no_word_segs.append(e0 - s0)
                continue
            errs.append(int(cand[0] - s0))
    e = np.abs(np.asarray(errs)) if errs else np.zeros(0)
    stats = {
        "items": sum(1 for it in items if not bool(((it[2] if ref else it[1]) == tdc.USER_TEXT_IGNORE).all())),
        "word_tokens": n_tok,
        "tokens_in_silence": out_tok,
        "onsets": len(errs),
        "onset_err_hist": {int(k): int(v) for k, v in zip(*np.unique(np.clip(errs, -9, 9), return_counts=True))}
        if errs
        else {},
        "onset_abs_p50_90_max": [float(np.percentile(e, 50)), float(np.percentile(e, 90)), float(e.max())]
        if errs
        else None,
        "onsets_within_tol": round(float(np.mean(e <= TOL)), 4) if errs else None,
        "segments_without_words": len(no_word_segs),
        "silent_token_frames": worst[:5],
    }
    return stats, out_tok, e


REF = {}


def aligned(name, items, *, expect=True):
    """Pass bar, relative to the REFERENCE (the assistant text row vs the assistant audio on the plain path: the main
    text stream's own Whisper alignment, which training already uses): word tokens in user silence <= max(1%, the
    reference's share + 0.5%), onsets within +-TOL >= the reference's share - 0.05, median |onset error| <= 1 frame."""
    st, out_tok, e = score(items)
    ok = st["items"] > 0 and st["word_tokens"] > 0 and len(e) > 0
    if ok:
        tok_share = out_tok / st["word_tokens"]
        st["tokens_in_silence_share"] = round(tok_share, 4)
        ok = (
            tok_share <= max(0.01, REF["tokens_in_silence_share"] + 0.005)
            and float(np.mean(e <= TOL)) >= REF["onsets_within_tol"] - 0.05
            and float(np.median(e)) <= 1.0
        )
    if expect:
        check(f"{name}: user-text targets on the user's speech (bar: the reference's)", ok, f"{st}")
    else:
        print(f"  mutant {name}: {st}", flush=True)
    return ok


def fx_table_swappable(tmp):
    """The fixture with ``*_other`` prompt / voice columns (copies), so ``swap_channels`` can read it."""
    import pyarrow as pa
    from datasets import Dataset

    t = tdc.read_arrow_table(os.path.join(FX, "codes"))
    for c in ("context", "voice_codes", "voice_n_frames"):
        t = t.append_column(f"{c}_other", t.column(c))
    p = os.path.join(tmp, "fx_swap")
    Dataset(pa.Table.from_batches(t.to_batches())).save_to_disk(p)
    return p


def mix(tok, *, concat, knob=True, path=None, meta=None, batch_size=4):
    return td.build_knowledge_mix_loader(
        path or os.path.join(FX, "codes"),
        path or os.path.join(FX, "codes"),
        tok,
        batch_size=batch_size,
        share=1.0,
        seed=0,
        user_text_delay=D,
        knowledge_meta=(meta or os.path.join(FX, "meta.npz")) if concat else None,
        knowledge_concat=AB2_CONCAT if concat else None,
        knowledge_plans=PLANS if concat else None,
        **({"knowledge_user_text": True} if knowledge_user_text_on(knob) else {}),
        **KW,
    )


def knowledge_user_text_on(knob):
    return bool(knob)


PLANS = []


def paths(tmp):
    """{path: items} through the real loaders."""
    tok, tok10 = tokenizer(), tokenizer(10.0)
    out = {}
    out["plain"] = items_of(mix(tok, concat=False), 5)
    PLANS.clear()
    out["concat"] = items_of(mix(tok, concat=True), 4)
    out["windowed"] = items_of(mix(tok10, concat=False), 5)
    meta = []
    swap_items = items_of(
        td.build_data_loader(
            fx_table_swappable(tmp),
            tok,
            batch_size=4,
            seed=0,
            swap_channels=True,
            user_text_delay=D,
            item_meta=meta,
            **KW,
        ),
        10,
    )
    out["swapped"] = [it for it, m in zip(swap_items, meta) if m["swap"]]
    out["swap_unswapped"] = [it for it, m in zip(swap_items, meta) if not m["swap"]]
    return out


def load_old(name, relpath):
    src = subprocess.run(
        ["git", "-C", FD, "show", f"{OLD_REV}:./{relpath}"], capture_output=True, text=True, check=True
    ).stdout
    path = os.path.join(tempfile.mkdtemp(prefix="old_"), os.path.basename(relpath))
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


def part_knob_off_is_old():
    old_kc = load_old("old_knowledge_concat", "moshi_family/knowledge_concat.py")
    old_td = load_old("old_train_data", "moshi_family/personaplex/train_data.py")
    tok = tokenizer()
    for concat in (False, True):
        kw = dict(
            batch_size=8,
            share=0.3,
            seed=0,
            swap_channels=True,
            stored_user_augment=True,
            user_text_delay=D,
            knowledge_meta=V6_META if concat else None,
            knowledge_concat=AB2_CONCAT if concat else None,
            **KW,
        )
        new = td.build_knowledge_mix_loader(P4, V6, tok, **kw)
        nb = [next(new) for _ in range(5)]
        real = sys.modules["moshi_family.knowledge_concat"]
        sys.modules["moshi_family.knowledge_concat"] = old_kc  # the old train_data imports it inside the generator
        try:
            old = old_td.build_knowledge_mix_loader(P4, V6, tok, **kw)
            ob = [next(old) for _ in range(5)]
        finally:
            sys.modules["moshi_family.knowledge_concat"] = real
        same = all(len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b)) for a, b in zip(nb, ob))
        check(
            f"k1 knob off = the pre-change code ({OLD_REV}) on p4 + v6, concat={concat}: 5 batches bit-identical", same
        )


def part_assistant_row():
    tok = tokenizer()
    for concat in (False, True):
        PLANS.clear()
        on = [next(b) for b in [mix(tok, concat=concat, knob=True)] for _ in range(4)]
        PLANS.clear()
        off = [next(b) for b in [mix(tok, concat=concat, knob=False)] for _ in range(4)]
        same = all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(on, off))
        real_tgt = any(not bool((a[-1] == tdc.USER_TEXT_IGNORE).all()) for a in on)
        ign = all(bool((b[-1] == tdc.USER_TEXT_IGNORE).all()) for b in off)
        check(
            f"r1 knob on vs off (concat={concat}): codes (assistant text row included) + masks bit-identical; on = real "
            "targets, off = all-IGNORE",
            same and real_tgt and ign,
        )


def run_mutant(m, tmp):
    """Apply mutant ``m`` (patches the function the loader calls), rebuild the affected path, expect misalignment."""
    if m == "lead_in":
        real = tdc.user_text_targets

        def shifted(*a, **k):
            r = real(*a, **k)
            out = torch.full_like(r, tdc.TEXT_PADDING_ID)
            out[LEAD_IN:] = r[:-LEAD_IN]
            return out

        td.user_text_targets = tdc.user_text_targets = shifted  # train_data's import + knowledge_concat's lazy one
        items = items_of(mix(tokenizer(), concat=False), 5)
        td.user_text_targets = tdc.user_text_targets = real
        return not aligned("lead_in (plain)", items, expect=False)
    if m == "concat_gap":
        real = kcat.splice_user_words

        def no_gap(plan, user_rows, *, fr, onset_floor):
            for p in plan.pieces:  # the offset as if the gap were not there
                if p.junction == "gap":
                    p.offset -= p.gap_frames
            try:
                return real(plan, user_rows, fr=fr, onset_floor=onset_floor)
            finally:
                for p in plan.pieces:
                    if p.junction == "gap":
                        p.offset += p.gap_frames

        kcat.splice_user_words = no_gap
        PLANS.clear()
        items = items_of(mix(tokenizer(), concat=True), 4)
        kcat.splice_user_words = real
        return not aligned("concat_gap", items, expect=False)
    if m == "swap_old":
        real = td.user_words_of_read
        td.user_words_of_read = lambda table, r, swap: real(table, r, False)  # keeps alignments_other on a swap
        meta = []
        it = items_of(
            td.build_data_loader(
                fx_table_swappable(tmp),
                tokenizer(),
                batch_size=4,
                seed=0,
                swap_channels=True,
                user_text_delay=D,
                item_meta=meta,
                **KW,
            ),
            10,
        )
        td.user_words_of_read = real
        return not aligned("swap_old", [x for x, mm in zip(it, meta) if mm["swap"]], expect=False)
    raise ValueError(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="check_user_text_alignment_") as tmp:
        ps = paths(tmp)
        n_gap = sum(p.junction == "gap" for pl in PLANS for p in pl.pieces)
        n_barge = sum(p.junction == "barge" for pl in PLANS for p in pl.pieces)
        sizes = [len(pl.pieces) for pl in PLANS]
        print(
            f"concat plans: {len(PLANS)} items, pieces {sizes}, {n_gap} gap + {n_barge} barge-in junctions", flush=True
        )
        check(
            "p0 concat items have 2-4 dialogues with both junction kinds",
            sizes and max(sizes) >= 2 and n_gap > 0 and n_barge > 0,
        )
        st, out_tok, _e = score(ps["plain"], ref=True)
        REF.update(st, tokens_in_silence_share=round(out_tok / max(1, st["word_tokens"]), 4))
        print(f"REFERENCE (assistant text row vs assistant audio, plain path): {REF}", flush=True)
        for name in ("concat", "swapped"):
            st, _o, _e = score(ps[name], ref=True)
            print(f"  reference on the {name} items: {st}", flush=True)
        for name in ("plain", "concat", "windowed", "swapped", "swap_unswapped"):
            aligned(name, ps[name])
        part_assistant_row()
        part_knob_off_is_old()
        if args.selftest:
            caught = []
            for m in ("lead_in", "concat_gap", "swap_old"):
                c = run_mutant(m, tmp)
                caught.append(c)
                print(f"{'CAUGHT' if c else 'MISSED'} mutant {m}", flush=True)
            check(f"selftest: {sum(caught)}/{len(caught)} mutants caught", all(caught))
    print(f"{'ALL PASS' if not FAILS else 'FAILED ' + str(FAILS)}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
