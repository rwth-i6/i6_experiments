"""Guard: the flagged-word arms (2026-09-28b: loss mask, strong down-weight, or a
beep the model learns to output instead) act on EXACTLY the assistant's flagged words, on the real data
path, and change nothing when off.

WHY. A per-frame weight fails silently: spans of the wrong stored side (a swapped read's assistant is the OTHER
channel), frames off by the text lead or a rounding, a text placement that differs from the text row (word overlap
"drop" truncates tokens), a weight that also hits the user's codebooks, a loss that re-normalises, a launcher that reads
the key and never passes it; and a censor that swaps in the censored channel when that channel is the USER.

Two modes (``job`` in the setup ``.venv``, ``loader`` in the moshi_family venv); inputs in ``OUT``:

  job (setup .venv, ~4 min CPU):
    J1. ``flagged_word_hits`` against ``lexicon_hits`` (the LexiconRows counts): per side and category the same hits
        over ALL p4 rows (a multi-word hit = one hit); unit cases (multi-word, punctuation, two categories on one word).
        CAN FAIL: a per-word matcher (each word alone) misses the multi-word entries.
    J2. ``flagged_word_spans`` (the FlaggedWordSpans worker) -> ``OUT/spans.json`` (+ summary).
  loader (moshi_family venv, CPU, ~6 min):
    1. FILE. ``load_word_spans``: wrong row count / unknown side raise.
    2. PLAIN LOADER (p4, the 2 h standard's read knobs: swap, stored user augmentation; edge keep-list; the 09-28b row
       weights): codes + masks + rows + swap draws identical to the run without; third tensor [T, 3]: column 0 = the
       row's weight, 1 on the prompt. INDEPENDENT check from the TOKENS of the item's text row: every run of text
       frames at the word weight decodes (SentencePiece) to text containing a lexicon word, and every token word that
       alone is a lexicon word is weighted; each weighted audio run starts at a weighted text run's first frame
       (onset_floor, no lead) and covers the word's frames. Swapped reads with hits occur.
       CAN FAIL: spans with the two sides exchanged (the wrong channel's words) fail the token check; spans shifted by
       0.2 s raise (their words are not in the alignments).
    3. MIX LOADER: 3 know30_concat batches: codes identical to without, knowledge items all ones [T, 3].
    4. LOSS. Gradient ratio weighted / unweighted per position: the word weight on the weighted text positions and on
       the assistant codebooks 0-7 at the weighted audio frames, exactly 1 everywhere else and on the user codebooks
       8-15; weight 0 -> gradient exactly 0 there. CAN FAIL: the weight also in the denominators moves the rest off 1.
    5. LAUNCHER. The real ``loss_step`` with ``train_word_spans`` passes columns 1-2: same ratios. CAN FAIL: a
       ``loss_step`` dropping ``asst_sample_weight`` gives ratio 1.
    6. OFF == BEFORE. ``asst_sample_weight`` None: loss + components bit-identical to ``PRE_CHANGE``'s
       ``paper_weight_loss``; ``interleave_text(token_owner=[])`` gives the same row as without, and as ``PRE_CHANGE``.
    7. CENSOR (the pilot sidecar ``OUT/censor_pilot``, the worker's own output on the GPU test node): stored codes kept
       outside every hit's frames and replaced inside; the decoded censored span is a 1 kHz tone (spectral peak) at the
       word's level (RMS within 3 dB); the loader swaps the censored copy in only for THIS read's assistant channel
       (non-swapped: stored assistant censored; swapped: the other channel censored, the stored assistant = the USER
       stream = its stored codes) and the text row says "beep" there. CAN FAIL: a store that ignores the side censors
       a swapped read's assistant with the wrong channel.

    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_word_spans.py job
    ./hpc-venv.py --cluster i6-rz moshi_family --sh 'cd /home/tt201262/setups/2026-01-speech-llm && HF_HUB_OFFLINE=1 \\
        python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_word_spans.py loader'
"""

import json
import os
import sys

OUT = "/hpcwork/p0023999/tt201262/guards/check_word_spans"
W = "/home/tt201262/setups/2026-01-speech-llm/work/i6_experiments/users/dorian_koch/speech_llm"
JRE = f"{W}/persona_prompts/AttachPersonaPrompts.DHvnX2ifpvad/output/dataset"
KNOW = f"{W}/persona_knowledge/KnowledgeDialogueCodes.DMgw9Srb4bmg/output/dataset"
META = f"{W}/persona_knowledge/KnowledgeConcatMeta.Jq5Gwzdr2Ray/output/meta.npz"
RF_OUT = "/hpcwork/p0023999/tt201262/guards/check_row_filters"
EDGE_KEEP = "/hpcwork/p0023999/tt201262/guards/check_episode_edge_rows/keep.npy"
PRE_CHANGE = "b5ef816d"  # full_duplex: the last train_data_common.py before the flagged-word weights
fails = []


def ck(c, label, extra=""):
    print(f"  [{'ok' if c else 'FAIL'}] {label}{'  ' + extra if extra else ''}", flush=True)
    if not c:
        fails.append(label)


def raises(fn):
    try:
        fn()
    except (ValueError, AssertionError, RuntimeError, KeyError) as e:
        return str(e)[:100]
    return None


def done():
    print()
    if fails:
        print(f"FAIL ({len(fails)}):", *fails, sep="\n  ")
        sys.exit(1)
    print("ALL OK")
    sys.exit(0)


MODE = sys.argv[1] if len(sys.argv) > 1 else "loader"

# =================================================================================================================
if MODE == "job":
    os.environ.setdefault("CUDA_HOME", "/usr")
    sys.path.insert(0, "recipe")
    sys.path.insert(0, "recipe/sisyphus")
    from i6_experiments.users.dorian_koch.speech_llm import row_filters as rf

    pats = rf.lexicon_patterns("explicit_v1")
    cats = ("profanity", "sexual")
    print("J1. flagged_word_hits vs lexicon_hits", flush=True)
    cases = [
        (["she", "gave", "him", "a", "blow", "job", "yesterday"], [(["sexual"], [4, 5])]),
        (["you", "are", "fucked."], [(["profanity"], [2])]),
        (["Jerk", "off,", "man"], [(["sexual"], [0, 1])]),
        (["no", "bad", "words", "here"], []),
        (["that's", "some", "bullshit", "and", "porn"], [(["profanity"], [2]), (["sexual"], [4])]),
    ]
    for words, want in cases:
        got = rf.flagged_word_hits(words, pats, cats)
        ck(got == want, f"unit: {' '.join(words)!r}", f"{got}")

    def per_word(words, pats, cats):
        """Mutant: each word matched alone (misses 'blow job', 'jerk off')."""
        return [(sorted(c for c in cats if pats[c].search(rf.norm_words(w))), [i]) for i, w in enumerate(words)
                if any(pats[c].search(rf.norm_words(w)) for c in cats)]  # fmt: skip

    ck(per_word(cases[0][0], pats, cats) != cases[0][1], "CAN FAIL: a per-word matcher misses a multi-word entry")

    from datasets import load_from_disk

    ds = load_from_disk(JRE).select_columns(["alignments", "alignments_other"])
    tot = {s: {c: [0, 0] for c in cats} for s in ("alignments", "alignments_other")}
    for r in ds:
        for col in tot:
            ws = [str(w["text"]) for w in (r[col] or [])]
            lh = rf.lexicon_hits(" ".join(ws), pats)
            for c, idx in rf.flagged_word_hits(ws, pats, cats):
                for cat in c:
                    tot[col][cat][0] += 1
            for cat in cats:
                tot[col][cat][1] += lh[cat]
    for col, d in tot.items():
        for cat, (a, b) in d.items():
            # a hit is per WORD SET: two matches inside one word (rare) count once
            ck(b - 3 <= a <= b, f"{col} {cat}: {a} word-set hits vs {b} lexicon matches (all {ds.num_rows} rows)")

    print("J2. the FlaggedWordSpans worker -> spans.json", flush=True)
    os.makedirs(OUT, exist_ok=True)
    spans, summary = rf.flagged_word_spans(JRE, "explicit_v1", cats)
    json.dump(spans, open(f"{OUT}/spans.json", "w"))
    json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
    ck(spans["num_rows"] == ds.num_rows and len(spans["rows"]) == summary["rows_with_hit_any_side"],
       "spans for every row with a hit", f"{len(spans['rows'])} rows")  # fmt: skip
    done()

# =================================================================================================================
import ast  # noqa: E402
import importlib.util  # noqa: E402
import inspect  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import textwrap  # noqa: E402

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import moshi_family.personaplex.train_data as td  # noqa: E402
import moshi_family.train_data_common as tdc  # noqa: E402
from moshi_family.train_data_common import TEXT_EPAD_ID, TEXT_PADDING_ID, read_arrow_table  # noqa: E402

LIB = os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex")
LAUNCHER = os.path.join(LIB, "moshi_family", "personaplex", "finetune_launcher.py")
SPANS = f"{OUT}/spans.json"
WPATH = os.path.join(RF_OUT, "weights_0.3_above20.npy")
WW = 0.1  # the word weight the checks use (ratio defined); weight 0 checked separately
tmp = tempfile.mkdtemp(prefix="check_word_spans_")

# the lexicon, copied from row_filters (not importable in this venv: it imports sisyphus); digest-checked below
_LEX_SRC = open(os.path.join(os.getcwd(), "recipe/i6_experiments/users/dorian_koch/speech_llm/row_filters.py")).read()
_lex_ns = {"re": re, "hashlib": __import__("hashlib"), "json": json}
exec(  # noqa: S102 -- the module's own lexicon block + helpers, verbatim
    _LEX_SRC[_LEX_SRC.index("SAFETY_LEXICONS = {") : _LEX_SRC.index("def _side_text")], _lex_ns
)
PATS = {c: p for c, p in _lex_ns["lexicon_patterns"]("explicit_v1").items() if c in ("profanity", "sexual")}


def lexword(text: str) -> bool:
    t = _lex_ns["norm_words"](text)
    return any(p.search(t) for p in PATS.values())


table = read_arrow_table(JRE)
N_ROWS = table.num_rows

print("1. the spans file", flush=True)
spans = td.load_word_spans(SPANS, N_ROWS)
ck(len(spans) > 50_000 and all(set(v) <= {"assistant", "other"} for v in spans.values()),
   "the real spans load", f"{len(spans)} rows")  # fmt: skip
raw = json.load(open(SPANS))
bad = dict(raw, num_rows=N_ROWS - 1)
json.dump(bad, open(f"{tmp}/bad_n.json", "w"))
ck(bool(raises(lambda: td.load_word_spans(f"{tmp}/bad_n.json", N_ROWS))), "a wrong row count raises")
k0 = next(iter(raw["rows"]))
json.dump(dict(raw, rows={k0: {"user": []}}), open(f"{tmp}/bad_side.json", "w"))
ck(bool(raises(lambda: td.load_word_spans(f"{tmp}/bad_side.json", N_ROWS))), "an unknown side raises")

tok = td.PersonaPlexTokenizer(
    "nvidia/personaplex-7b-v1",
    device="cpu",
    cfg=td.PersonaPlexDataConfig(
        duration_sec=60.0, silence_frames=6, emit_epad=True, onset_floor=True, word_overlap="drop"
    ),
)
knobs = dict(system_prompt_key="context", voice_codes_key="voice_codes", swap_channels=True, stored_user_augment=True)
sp = tok.text_tok


def word_groups(ids: list) -> list:
    """Token words of a text row: ``[(first frame, last frame + 1)]`` -- a new word at a piece starting with the
    SentencePiece word mark or after padding / EPAD."""
    groups, cur = [], None
    for f, t in enumerate(ids):
        if t in (TEXT_PADDING_ID, TEXT_EPAD_ID) or t < 0:
            cur = None
            continue
        piece = sp.id_to_piece(int(t))
        if cur is None or piece.startswith("▁"):
            cur = [f, f + 1]
            groups.append(cur)
        else:
            cur[1] = f + 1
    return [tuple(g) for g in groups]


def runs_of(mask: np.ndarray) -> list:
    out, s = [], None
    for i, m in enumerate(list(mask) + [False]):
        if m and s is None:
            s = i
        elif not m and s is not None:
            out.append((s, i))
            s = None
    return out


def item_errors(codes, mask, wt, row_weight):
    """The independent token-level check of one item (see 2.): (errors, words weighted, lexicon words missed)."""
    errs, n_w, missed = [], 0, 0
    lead = int(torch.nonzero(mask)[0].item())
    nf = int(mask.sum())
    if not (torch.all(wt[:lead] == 1.0) and torch.all(wt[lead : lead + nf, 0] == torch.tensor(row_weight))):
        errs.append("prompt / row-weight column")
    ids = codes[0, lead : lead + nf].tolist()
    tw = wt[lead : lead + nf, 1].numpy()
    aw = wt[lead : lead + nf, 2].numpy()
    trun, arun = runs_of(tw != 1.0), runs_of(aw != 1.0)
    for a, b in trun:
        if not np.allclose(tw[a:b], WW):
            errs.append("text weight value")
        txt = sp.decode([int(t) for t in ids[a:b] if t not in (TEXT_PADDING_ID, TEXT_EPAD_ID)])
        if not lexword(txt):
            errs.append(f"weighted text {txt!r} holds no lexicon word")
    for a, b in word_groups(ids):
        txt = sp.decode([int(t) for t in ids[a:b]])
        if lexword(txt):
            n_w += 1
            if not np.all(tw[a:b] != 1.0):
                missed += 1
    starts = {a for a, _ in trun}
    for a, b in arun:
        if not np.allclose(aw[a:b], WW):
            errs.append("audio weight value")
    # every text run starts where an audio run covers it (text frame = floor(onset) = the audio span's first frame)
    for a, _ in trun:
        if not any(x <= a < y for x, y in arun):
            errs.append(f"text run at {a} outside every audio run")
    if trun and not starts & {a for a, _ in arun}:
        errs.append("no audio run starts at a text run's first frame")
    return errs, n_w, missed


print("\n2. the plain loader", flush=True)
weights = np.load(WPATH)
N2 = 48


def plain(meta, **kw):
    it = td.build_data_loader(JRE, tok, batch_size=1, seed=0, infinite=False, item_meta=meta, train_rows=EDGE_KEEP,
                              train_row_weights=WPATH, **knobs, **kw)  # fmt: skip
    return [next(it) for _ in range(N2)]


m_on, m_off = [], []
b_on = plain(m_on, train_word_spans=SPANS, train_word_weight=WW)
b_off = plain(m_off)
ck(
    all(x[2].shape[-1] == 3 and y[2].shape[-1] == 1 and torch.equal(x[0], y[0]) and torch.equal(x[1], y[1])
        and torch.equal(x[2][..., 0], y[2][..., 0]) for x, y in zip(b_on, b_off))
    and [m["row"] for m in m_on] == [m["row"] for m in m_off] and [m["swap"] for m in m_on] == [m["swap"] for m in m_off],
    "codes + masks + rows + swap draws + row weights identical with and without the word spans",
)  # fmt: skip
tot_err, tot_w, tot_miss, n_hit, n_swap_hit = [], 0, 0, 0, 0
for b, m in zip(b_on, m_on):
    e, n_w, miss = item_errors(b[0][0], b[1][0], b[2][0], float(weights[m["row"]]))
    tot_err += e
    tot_w += n_w
    tot_miss += miss
    has = bool((b[2][0, :, 1] != 1).any())
    n_hit += has
    n_swap_hit += has and m["swap"]
ck(not tot_err, "every weighted text run decodes to a lexicon word; audio runs cover them", f"{tot_err[:3]}")
ck(tot_w > 10 and tot_miss <= max(1, 0.03 * tot_w), "the token words that are lexicon words are weighted",
   f"{tot_w - tot_miss}/{tot_w} (misses = tokens dropped by word overlap)")  # fmt: skip
ck(n_hit > 5 and n_swap_hit > 0, "items with hits, swapped reads among them", f"{n_hit} of {N2}, {n_swap_hit} swapped")

print("   CAN FAIL:", flush=True)
json.dump(dict(raw, rows={k: {("other" if s == "assistant" else "assistant"): h for s, h in v.items()}
                          for k, v in raw["rows"].items()}), open(f"{tmp}/swapped_sides.json", "w"))  # fmt: skip
m_x = []
e_x = []
try:
    for b, m in zip(plain(m_x, train_word_spans=f"{tmp}/swapped_sides.json", train_word_weight=WW), m_x):
        e_x += item_errors(b[0][0], b[1][0], b[2][0], float(weights[m["row"]]))[0]
except RuntimeError as exc:  # the other side's words are not in this side's alignments
    e_x.append(str(exc)[:80])
# the loader SKIPS a row whose spans do not match its alignments (batched_row_loader's failure budget): a skipped row
# shifts the stream, so the rows read differ from the right spans' run
skipped_x = [m["row"] for m in m_x] != [m["row"] for m in m_on]
ck(bool(e_x) or skipped_x, "spans of the wrong side are caught (token check errors or skipped rows)",
   f"{len(e_x)} errors, rows differ: {skipped_x}")  # fmt: skip
shifted = dict(raw, rows={k: {s: [[[a + 0.2, b + 0.2] for a, b in h] for h in hs] for s, hs in v.items()}
                          for k, v in raw["rows"].items()})  # fmt: skip
json.dump(shifted, open(f"{tmp}/shifted.json", "w"))
m_s = []
plain(m_s, train_word_spans=f"{tmp}/shifted.json", train_word_weight=WW)
ck([m["row"] for m in m_s] != [m["row"] for m in m_on],
   "spans shifted by 0.2 s: the rows with hits fail to build and are skipped (their words are not in the alignments)")  # fmt: skip

print("\n3. the knowledge-mix loader", flush=True)
concat = {"barge_lead_sec": [0.2, 0.8], "barge_min_answer_sec": 0.5, "barge_prob": 0.3, "dialogues": [2, 4],
          "gap_sec": [0.3, 1.5], "min_next_sec": 5.0, "tail_frames": 1}  # fmt: skip


def mix(meta, **kw):
    it = td.build_knowledge_mix_loader(JRE, KNOW, tok, batch_size=8, share=0.3, seed=0, knowledge_meta=META,
                                       knowledge_concat=concat, item_meta=meta, train_row_weights=WPATH, **knobs, **kw)  # fmt: skip
    return [next(it) for _ in range(3)]


mm_on, mm_off = [], []
x_on, x_off = mix(mm_on, train_word_spans=SPANS, train_word_weight=WW), mix(mm_off)
ok = all(torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and a[2].shape[-1] == 3 for a, b in zip(x_on, x_off))
kn_ok, n_k = True, 0
for bi, bt in enumerate(x_on):
    for i in range(bt[0].shape[0]):
        if mm_on[bi * 8 + i]["source"] == "knowledge":
            n_k += 1
            n_i = int(torch.nonzero(bt[1][i])[-1].item()) + 1  # the item's own frames (collate pads with 0)
            kn_ok &= bool(torch.all(bt[2][i][:n_i] == 1.0))
ck(ok and kn_ok and n_k > 0, "3 mixed batches: codes identical, knowledge items all ones [T, 3]", f"{n_k} knowledge")

print("\n4. the loss", flush=True)


class Out:
    pass


class StubLM:
    def __init__(self, b, t, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.text = torch.randn(b, 1, t, 32000, generator=g).requires_grad_(True)
        self.audio = torch.randn(b, 16, t, 2048, generator=g).requires_grad_(True)

    def __call__(self, codes, depformer_frame_mask=None):
        o = Out()
        o.text_logits, o.logits = self.text, self.audio
        o.text_mask = torch.ones(*self.text.shape[:3], dtype=torch.bool)
        o.mask = torch.ones(*self.audio.shape[:3], dtype=torch.bool)
        return o


# two real items with hits in one batch
hit_items = [b for b in b_on if bool((b[2][0, :, 1] != 1).any())][:2]
codes, mask, wt3 = tdc.collate_codes([(b[0][0], b[1][0], b[2][0]) for b in hit_items])
Bn, T = codes.shape[0], codes.shape[2]
A = wt3[..., 1:3].clone()
LKW = dict(dep_q=16, weighted_mean=True, text_epad_id=TEXT_EPAD_ID, return_components=True, text_epad_weight=1.0,
           user_semantic_weight=1.0)  # fmt: skip


def grads(asw, loss_fn=tdc.paper_weight_loss):
    lm = StubLM(Bn, T)
    kw = {} if asw is None else {"asst_sample_weight": asw}
    loss, comps = loss_fn(lm(codes), codes, mask, **LKW, **kw)
    loss.backward()
    return lm.text.grad.abs().sum(-1)[:, 0].double(), lm.audio.grad.abs().sum(-1).double(), loss, comps


def ratio_ok(gw, g1, want):
    sel = g1 > 0
    return bool(torch.allclose((gw[sel] / g1[sel]).float(), want[sel].float(), rtol=1e-5, atol=1e-6))


gt_w, ga_w, _, c_w = grads(A)
gt_1, ga_1, _, _ = grads(None)
ck(ratio_ok(gt_w, gt_1, A[..., 0].double()), "text: the word weight on weighted positions, 1 elsewhere")
ck(ratio_ok(ga_w[:, :8], ga_1[:, :8], A[..., 1].double()[:, None, :].expand(-1, 8, -1)),
   "assistant codebooks 0-7: the word weight on weighted audio frames, 1 elsewhere")  # fmt: skip
ck(ratio_ok(ga_w[:, 8:], ga_1[:, 8:], torch.ones_like(ga_1[:, 8:])), "user codebooks 8-15: 1 everywhere")
A0 = torch.where(A != 1.0, torch.zeros_like(A), A)
gt_0, ga_0, _, _ = grads(A0)
ck(bool(torch.all(gt_0[A0[..., 0] == 0] == 0)) and bool(torch.all(ga_0[:, :8][(A0[..., 1] == 0)[:, None, :].expand(-1, 8, -1)] == 0)),
   "weight 0: gradient exactly 0 on the masked text positions and assistant audio frames")  # fmt: skip
ck("asst_word_text_frac" in c_w and "asst_word_audio_frac" in c_w, "components log the weighted shares")
print("   CAN FAIL:", flush=True)
SRC = inspect.getsource(tdc.duplex_ce_loss)
_A = "text_loss = text_term.sum() / text_denom.clamp(min=1e-8)"
assert _A in SRC, "duplex_ce_loss's text loss line changed; update the mutant"
_ns = dict(vars(tdc))
exec(SRC.replace(_A, "text_loss = text_term.sum() / (text_w * asst_sample_weight[..., 0].reshape(-1)).sum()"
                 ".clamp(min=1e-8) if asst_sample_weight is not None else text_term.sum() / text_denom"), _ns)  # fmt: skip


def renorm_loss(out, codes, loss_mask, *, dep_q, **k):
    return _ns["duplex_ce_loss"](out, codes, loss_mask, dep_q=dep_q, other_audio_weight=0.02, text_pad_weight=0.3, **k)


gm, _, _, _ = grads(A, loss_fn=renorm_loss)
gm1, _, _, _ = grads(None, loss_fn=renorm_loss)
ck(not ratio_ok(gm, gm1, A[..., 0].double()), "a re-normalising word weight is caught")

print("\n5. the launcher's real loss_step", flush=True)
src_l = open(LAUNCHER).read()
main_fn = next(n for n in ast.walk(ast.parse(src_l)) if isinstance(n, ast.FunctionDef) and n.name == "main")
ls_node = next(n for n in ast.walk(main_fn) if isinstance(n, ast.FunctionDef) and n.name == "loss_step")
LOSS_STEP_SRC = textwrap.dedent(ast.get_source_segment(src_l, ls_node))


def ls_grads(src, batch, spans_on=True):
    ns = {
        "model": None, "device": "cpu", "dep_q": 16, "personaplex_loss": td.personaplex_loss,
        "TEXT_EPAD_ID": TEXT_EPAD_ID, "loss_norm": "weight", "emit_epad": True, "text_epad_weight": 1.0,
        "floor_weight": None, "text_pad_weight": 0.3, "user_semantic_weight": 1.0, "depformer_skip_masked": True,
        "turn_tail_weight": None, "train_row_weights": WPATH, "train_word_spans": SPANS if spans_on else None,
        "user_head": None,  # the user-text head (eef4e848), off
    }  # fmt: skip
    lm = StubLM(Bn, T)
    ns["model"] = lm
    exec(src, ns)
    loss, _ = ns["loss_step"](batch)
    loss.backward()
    return lm.text.grad.abs().sum(-1)[:, 0].double(), lm.audio.grad.abs().sum(-1).double()


ones3 = torch.ones_like(wt3)
ones3[..., 0] = wt3[..., 0]
lt_w, la_w = ls_grads(LOSS_STEP_SRC, (codes, mask, wt3))
lt_1, la_1 = ls_grads(LOSS_STEP_SRC, (codes, mask, ones3))
ck(ratio_ok(lt_w, lt_1, A[..., 0].double()) and ratio_ok(la_w[:, :8], la_1[:, :8], A[..., 1].double()[:, None, :].expand(-1, 8, -1)),
   "loss_step with train_word_spans: the word weights on text + assistant codebooks")  # fmt: skip
mut = LOSS_STEP_SRC.replace("asst_sample_weight=asst_sample_weight,", "")
assert mut != LOSS_STEP_SRC, "the launcher's loss_step does not pass asst_sample_weight"
mt, _ = ls_grads(mut, (codes, mask, wt3))
ck(not ratio_ok(mt, lt_1, A[..., 0].double()), "CAN FAIL: a loss_step that drops asst_sample_weight is caught")

print("\n6. off == before", flush=True)
src_old = subprocess.run(["git", "-C", LIB, "show", f"{PRE_CHANGE}:./moshi_family/train_data_common.py"],
                         capture_output=True, text=True, check=True).stdout  # fmt: skip
open(f"{tmp}/tdc_old.py", "w").write(src_old)
spec = importlib.util.spec_from_file_location("tdc_old", f"{tmp}/tdc_old.py")
tdc_old = importlib.util.module_from_spec(spec)
sys.modules["tdc_old"] = tdc_old
spec.loader.exec_module(tdc_old)
ck("asst_sample_weight" not in src_old and "token_owner" not in src_old, "the reference predates both knobs")
g = torch.Generator().manual_seed(1)
o = Out()
o.text_logits = torch.randn(Bn, 1, T, 32000, generator=g)
o.logits = torch.randn(Bn, 16, T, 2048, generator=g)
o.text_mask = torch.ones(Bn, 1, T, dtype=torch.bool)
o.mask = torch.ones(Bn, 16, T, dtype=torch.bool)
for name, sfw in (("no sample weight", None), ("row weights", wt3[..., 0])):
    l_new, c_new = tdc.paper_weight_loss(o, codes, mask, sample_frame_weight=sfw, **LKW)
    l_old, c_old = tdc_old.paper_weight_loss(o, codes, mask, sample_frame_weight=sfw, **LKW)
    ck(torch.equal(l_new, l_old) and c_new == c_old, f"asst_sample_weight None ({name}): bit-identical to {PRE_CHANGE}")
l_one, _ = tdc.paper_weight_loss(o, codes, mask, asst_sample_weight=torch.ones(Bn, T, 2), **LKW)
l_old, _ = tdc_old.paper_weight_loss(o, codes, mask, **LKW)
ck(torch.equal(l_one, l_old), "all-ones asst_sample_weight: the same loss bit for bit")
old_tok = tdc_old.DuplexTokenizerBase.interleave_text
same = True
for r in list(spans)[:40]:
    al = tdc.normalize_alignments(table.column("alignments")[r].as_py())
    own = []
    a = tok.interleave_text(al, 750, **tok._text_kwargs(), token_owner=own)
    b = tok.interleave_text(al, 750, **tok._text_kwargs())
    c = old_tok(tok, al, 750, **tok._text_kwargs())
    same &= torch.equal(a, b) and torch.equal(a, c) and own[0].shape == (750,)
ck(same, "interleave_text: the same row with token_owner, without it, and in the pre-change code (40 rows)")

print("\n7. the censor sidecar (pilot)", flush=True)
PILOT = f"{OUT}/censor_pilot"
if not os.path.exists(f"{PILOT}/dataset"):
    ck(False, f"no pilot sidecar at {PILOT} (run the worker on the test node first)")
    done()
psum = json.load(open(f"{PILOT}/summary.json"))
if not os.path.exists(f"{PILOT}/meta.json"):
    json.dump({"num_rows": psum["num_rows"]}, open(f"{PILOT}/meta.json", "w"))
store = td.load_censor(PILOT, N_ROWS)
ck(len(store.index) > 50, "the pilot sidecar loads", f"{len(store.index)} entries, {psum}")
from huggingface_hub import hf_hub_download  # noqa: E402

from moshi_family.personaplex.models import loaders  # noqa: E402

mimi = loaders.get_mimi(hf_hub_download("nvidia/personaplex-7b-v1", loaders.MIMI_NAME), "cpu")
sr = int(mimi.sample_rate)
n_ok, n_tone, n_lvl, n_chk = 0, 0, 0, 0
for (row, side), i in list(store.index.items())[:12]:
    cen, al_c = store.get(row, side)
    ca, cu, _fr, _al = tdc.decode_codes_row(table, row)
    orig = ca if side == "assistant" else cu
    hits = spans[row][side]
    inside = np.zeros(orig.shape[1], bool)
    for h in hits:
        f0 = int(np.floor(min(a for a, _ in h) * 12.5))
        f1 = int(np.ceil(max(b for _, b in h) * 12.5))
        inside[f0:f1] = True
    n_ok += bool(np.array_equal(cen[:, ~inside], orig[:, ~inside])) and bool((cen[:, inside] != orig[:, inside]).any())
    with torch.no_grad():
        pc = mimi.decode(torch.from_numpy(cen.astype(np.int64))[None])[0, 0].numpy()
        po = mimi.decode(torch.from_numpy(orig.astype(np.int64))[None])[0, 0].numpy()
    h = hits[0]
    a, b = int(min(x for x, _ in h) * sr), int(max(y for _, y in h) * sr)
    if b - a < sr // 10:
        continue
    n_chk += 1
    seg = pc[a + (b - a) // 4 : b - (b - a) // 4]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    peak_hz = float(np.fft.rfftfreq(len(seg), 1 / sr)[spec.argmax()])
    n_tone += abs(peak_hz - 1000.0) < 60.0
    lvl = 20 * np.log10((np.sqrt(np.mean(pc[a:b] ** 2)) + 1e-9) / (np.sqrt(np.mean(po[a:b] ** 2)) + 1e-9))
    n_lvl += abs(lvl) < 3.0
    words = [w["text"] for w in al_c]
    ck(
        "beep" in words and not any(lexword(w) for w in words),
        f"row {row} {side}: 'beep' in the words, no lexicon word",
    )
ck(n_ok == 12, "stored codes kept outside every hit, replaced inside", f"{n_ok}/12")
ck(n_chk > 3 and n_tone == n_chk and n_lvl >= n_chk - 1, "the censored span decodes to a ~1 kHz tone at the word's level",
   f"tone {n_tone}/{n_chk}, level within 3 dB {n_lvl}/{n_chk}")  # fmt: skip


class Rng:
    """A fixed swap coin (``random()``) for build_item; prompt draws take index 0."""

    def __init__(self, swap):
        self.swap = swap

    def random(self):
        return 0.0 if self.swap else 0.9

    def integers(self, n):
        return 0


def build(censor, row, swap):
    bi = td.persona_item_builder(table, tok, system_prompt_key="context", voice_codes_key="voice_codes",
                                 swap_channels=True, censor=censor)  # fmt: skip
    return bi(row, Rng(swap))


# a row with a censored stored-assistant entry, and other_ok (so it can be swapped)
rows_a = [r for (r, s) in store.index if s == "assistant" and table.column("other_ok")[r].as_py()]
row = rows_a[0]
ca, cu, _fr, _ = tdc.decode_codes_row(table, row)
cen_a, _ = store.get(row, "assistant")
cen_o = store.get(row, "other")
it_n = build(store, row, swap=False)
it_s = build(store, row, swap=True)


def streams(it):
    """(assistant [8, nf], user [8, nf]) of an item's dialogue frames (after its OWN prompt prefix)."""
    ld, n = int(torch.nonzero(it[1])[0].item()), int(it[1].sum())
    return it[0][1:9, ld : ld + n].numpy(), it[0][9:17, ld : ld + n].numpy(), n


asst_n, user_n, nf = streams(it_n)
asst_s, user_s, _ = streams(it_s)
ck(np.array_equal(asst_n, cen_a[:, :nf]) and np.array_equal(user_n, cu[:, :nf]),
   "non-swapped read: assistant = the censored stored assistant, user = the stored other channel")  # fmt: skip
want_s = (cen_o[0] if cen_o is not None else cu)[:, :nf]
ck(np.array_equal(asst_s, want_s) and np.array_equal(user_s, ca[:, :nf]),
   "swapped read: assistant = the other channel (censored if it has hits), user = the stored assistant UNcensored")  # fmt: skip
beep_ids = set(tok.text_tok.encode("beep"))
ck(beep_ids & set(it_n[0][0].tolist()) != set(), "the text row holds the 'beep' tokens")


class SideBlind:
    """Mutant store: the stored-assistant entry whatever the side."""

    def __init__(self, s):
        self.s, self.num_rows = s, s.num_rows

    def get(self, row, side):
        return self.s.get(row, "assistant")


it_m = build(SideBlind(store), row, swap=True)
ck(not np.array_equal(streams(it_m)[0], want_s),
   "CAN FAIL: a side-blind store puts the wrong channel's censored copy on a swapped read")  # fmt: skip
done()
