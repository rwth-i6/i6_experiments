"""Guard: ``KnowledgeKnownScore`` scores the ANSWER tokens, length-normalised, on the right frames (ablation
ab2_know_known, 2026-09-29; worker ``moshi_family.personaplex.knowledge_known_score``).

WHY. The known-only arm trains on whatever this score calls "known". Three silent ways to get it wrong: score frames
that are not the answer (an off-by-one between the text row and the per-frame CE, or a word split that grabs the
neighbouring word), let the answer's LENGTH decide (a summed log-prob ranks every one-token answer above every
three-token one -- hence the length-normalised score), or count an answer the USER said first (the
assistant only copies it). Everything runs through the real worker ``main()`` on real rows of the v6 codes corpus,
the real row reader and the real join; only the 7B forward is replaced (CPU):

  1. JOIN. Every one of the 60,000 codes rows joins to its dialogue (slice + TTS turn text) and the joined dialogue's
     template is the row's slice. CAN FAIL: joining on the text of the first turn only leaves rows ambiguous.
  2. LOCATE, on 12 rows of every template: every "ok" row's located tokens decode to the matched answer phrase (space-
     insensitive: the ASR stream drops apostrophes and hyphens), the phrase's distinctive words (>= 4 letters) are in an
     ASSISTANT turn (the stream is ASR: digits, "Dr"), and no user turn says it first; the ``socratic`` / ``qa_user_guesses`` templates are
     mostly ``user_says_first`` and ``qa_direct`` / ``casual_brief`` mostly ``ok`` (non-vacuous both ways).
  3. SCORE through the real ``main()`` with a stub loss step: per frame the text CE is ``c_row`` (a per-row constant,
     1 of 8 levels) where the text token is the row's ORIGINAL token, else 5.0. So an "ok" row's score must be exactly
     ``-c_row`` whatever its answer length (length-normalised), the control's answer-swapped copies must score ~-5
     (the scorer reads the swapped frames: every donor token that differs from the original costs 5.0) and lower than
     the true answer in every pair, and ``known`` = the top 25% of the scored questions.
     CAN FAIL: ``KKS_MUTANT=sum`` (sum instead of mean) breaks the length check; ``KKS_MUTANT=shift`` (read the frame
     after each answer token) breaks the swap check.
  4. LOADER. ``build_knowledge_mix_loader(knowledge_rows=)`` with the concatenation (ab2_know30_concat's scheme) on the
     real v6 corpus + p3's rows: every piece of every knowledge item in 5 batches is a kept row; without the knob some
     are not (non-vacuous).

Runs in moshi_family_venv, CPU (~3-5 min; mimi + spm from the HF cache):
    ./hpc-venv.py --cluster i6-rz moshi_family --sh 'cd <setup> && HF_HUB_OFFLINE=1 python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_knowledge_known_score.py'
"""

import glob
import json
import os
import re
import sys
import tempfile

SETUP = os.getcwd()
LIB = os.path.join(SETUP, "recipe", "speech_llm", "full_duplex")
sys.path.insert(0, LIB)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import moshi_family.personaplex.heldout_loss as HL  # noqa: E402
import moshi_family.personaplex.knowledge_known_score as KKS  # noqa: E402
from moshi_family.personaplex.train_data import persona_item_builder  # noqa: E402
from moshi_family.train_data_common import TEXT_EPAD_ID, TEXT_PADDING_ID, TEXT_ROW, read_arrow_table  # noqa: E402

CODES = os.path.realpath("output/persona_knowledge/v6_codes")
DIALOGUES = os.path.realpath("output/triviaqa_mix_v6_dialogue_dataset")
SETTINGS = {
    "hf_repo_id": "nvidia/personaplex-7b-v1",
    "duration_sec": 60,
    "silence_frames": 6,
    "emit_epad": True,
    "onset_floor": True,
    "word_overlap": "drop",
    "loss_norm": "weight",
    "text_epad_weight": 1.0,
    "user_semantic_weight": 1.0,
    "depformer_skip_masked": True,
}
MUTANT = os.environ.get("KKS_MUTANT", "")
fails = []


def ck(c, label, extra=""):
    print(f"  [{'ok' if c else 'FAIL'}] {label}{'  ' + extra if extra else ''}", flush=True)
    if not c:
        fails.append(label)


def tts_paths() -> list[str]:
    """The v6 TTS outputs, as the concat-meta job of the v6 codes corpus was given them (its ``info``)."""
    for info in glob.glob(
        "work/i6_experiments/users/dorian_koch/speech_llm/persona_knowledge/KnowledgeConcatMeta.*/info"
    ):
        text = open(info).read()
        job = CODES.split("/output/")[0].rsplit("/", 1)[1]  # KnowledgeDialogueCodes.<hash> (mount paths differ)
        if f"{job}/output" not in text.split("PARAMETER: codes:", 1)[-1].split("\n", 1)[0]:
            continue
        pairs = re.findall(r"\('(\w+)', <Path ([^>]+)>\)", text.split("PARAMETER: tts:", 1)[1].split("\n", 1)[0])
        if pairs:
            return [f"{n}={p}" for n, p in pairs]
    raise RuntimeError("no KnowledgeConcatMeta of the v6 codes corpus found")


if MUTANT == "sum":
    _orig = KKS.score_positions

    def _sum_score(logp, frames):
        d = _orig(logp, frames)
        d["score"] = d["sum_logp"]
        return d

    KKS.score_positions = _sum_score
elif MUTANT == "shift":
    _orig = KKS.score_positions
    KKS.score_positions = lambda logp, frames: _orig(logp, [min(f + 1, len(logp) - 1) for f in frames])

TTS = tts_paths()
table = read_arrow_table(CODES)
codes_ids = [str(x) for x in table.column("id").to_pylist()]
tts = {}
for spec in TTS:
    name, path = spec.split("=", 1)
    tt = read_arrow_table(path)
    tts[name] = dict(zip([str(x) for x in tt.column("id").to_pylist()], tt.column("turns").to_pylist()))

# ---------------------------------------------------------------------------------------------------------------
print("[1] join: codes row -> dialogue (slice + TTS turn text)")
meta = KKS.join_questions(codes_ids, tts, read_arrow_table(DIALOGUES))
n_join = sum(m is not None for m in meta)
ck(n_join == len(codes_ids), f"every codes row joins ({n_join}/{len(codes_ids)})")
ck(
    all(m is None or m["template"] == cid.split("/", 1)[0] for m, cid in zip(meta, codes_ids)),
    "the joined dialogue's template is the row's slice",
)

# ---------------------------------------------------------------------------------------------------------------
print("[2] locate: the answer tokens on real rows (12 per template)")
tok = HL.make_tokenizer(HL.loss_settings(SETTINGS), "cpu")
piece = tok.text_tok.id_to_piece
build = persona_item_builder(
    table,
    tok,
    system_prompt_key="context",
    voice_codes_key="voice_codes",
    swap_channels=False,
    stored_user_augment=False,
    floor_weight=None,
)
rng = np.random.default_rng(0)
by_t: dict[str, list[int]] = {}
for i, cid in enumerate(codes_ids):
    by_t.setdefault(cid.split("/", 1)[0], []).append(i)
status: dict[str, dict] = {}
decode_ok = assistant_ok = True
for t_name, rows in sorted(by_t.items()):
    st = status.setdefault(t_name, {})
    for i in rng.choice(rows, 12, replace=False):
        i = int(i)
        codes, mask = build(i, rng)
        text = [int(x) for x, v in zip(codes[TEXT_ROW].tolist(), mask.tolist()) if v]
        word = [x for x in text if x not in (TEXT_PADDING_ID, TEXT_EPAD_ID)]
        loc = KKS.locate_answer(word, piece, KKS.answer_candidates(meta[i]["answer"]))
        if loc is None:
            st["answer_not_said"] = st.get("answer_not_said", 0) + 1
            continue
        pos, phrase, _src = loc
        if KKS.user_says_first(meta[i]["turns"], phrase):
            st["user_says_first"] = st.get("user_says_first", 0) + 1
            continue
        if t_name in KKS.USER_GUESS_TEMPLATES and re.search(r"[0-9]", phrase):
            st["number_in_guess_template"] = st.get("number_in_guess_template", 0) + 1
            continue
        st["ok"] = st.get("ok", 0) + 1
        # space-insensitive: the stream drops apostrophes / hyphens ("Devils", "threeyearolds") that the phrase splits on
        dec = KKS.norm_text("".join(piece(word[k]) for k in pos)).replace(" ", "")
        decode_ok &= dec == phrase.replace(" ", "")
        # the phrase is in an assistant turn: its distinctive words (>= 4 letters) appear in one (the stream is ASR of the
        # TTS audio: digits where the dialogue spells numbers, "Dr" for "Doctor"; a phrase of short words only passes)
        key = [w for w in phrase.split() if len(w) >= 4 and not w.isdigit()]
        bad = key and not any(
            x["speaker"] == "assistant" and any(w in KKS.norm_text(x["text"]).split() for w in key)
            for x in meta[i]["turns"]
        )
        if bad:
            print(f"    not in an assistant turn: {phrase!r} {[x['text'][:80] for x in meta[i]['turns']]}")
        assistant_ok &= not bad
print("   ", json.dumps(status))
ck(decode_ok, "every located span decodes to the matched answer phrase")
ck(assistant_ok, "every matched phrase is said in an assistant turn")
frac = lambda t, k: status.get(t, {}).get(k, 0) / 12  # noqa: E731
ck(
    frac("socratic", "user_says_first") + frac("qa_user_guesses", "user_says_first") >= 1.0,
    "socratic / qa_user_guesses: mostly user_says_first",
    f"{frac('socratic', 'user_says_first'):.2f} / {frac('qa_user_guesses', 'user_says_first'):.2f}",
)
ck(
    frac("qa_direct", "ok") >= 0.5 and frac("casual_brief", "ok") >= 0.5,
    "qa_direct / casual_brief: mostly ok",
    f"{frac('qa_direct', 'ok'):.2f} / {frac('casual_brief', 'ok'):.2f}",
)

# ---------------------------------------------------------------------------------------------------------------
print("[3] the real main() with a stub loss step (per-row CE on original tokens, 5.0 elsewhere)")
ORIG: dict[bytes, tuple[np.ndarray, float]] = {}


def _level(k: int) -> float:
    """A distinct per-row CE (no ties: the top-25% cut is then exact up to rows sharing a question)."""
    return round(0.05 + 0.0137 * k, 6)


def _key(codes_row: torch.Tensor, n: int) -> bytes:
    """The row's identity: its audio codes over its own frames (unchanged by the control's text swap)."""
    return codes_row[1:, :n].numpy().astype(np.int16).tobytes()


class _Core:
    dep_q = 16

    def eval(self):
        return self


def _stub_make_loss_step(model, *, device, dep_q, settings, per_frame_out=None):
    def step(batch):
        codes, mask = batch
        ce = torch.full(codes[:, TEXT_ROW].shape, 5.0)
        for b in range(codes.shape[0]):
            n = int(np.flatnonzero(mask[b].numpy()).max()) + 1  # frames up to the last supervised one
            k = _key(codes[b], n)
            if k not in ORIG:  # the first time a row is seen it is the TRUE row (main scores before the control)
                ORIG[k] = (codes[b, TEXT_ROW, :n].numpy().copy(), _level(len(ORIG)))
            orig, c = ORIG[k]
            same = torch.from_numpy(codes[b, TEXT_ROW, :n].numpy() == orig)
            ce[b, :n][same] = c
        per_frame_out.update(text_ce=ce, text_valid=mask.bool(), text_is_pad=torch.zeros_like(mask, dtype=torch.bool))
        return torch.tensor(0.0), {}

    return step


HL.load_model = lambda *a, **k: _Core()
HL.make_loss_step = _stub_make_loss_step
out = tempfile.mkdtemp(prefix="kks_guard_")
argv = [
    "--codes",
    CODES,
    "--dialogues",
    DIALOGUES,
    "--settings",
    json.dumps(SETTINGS),
    "--out",
    out,
    "--max_rows",
    "240",
    "--batch",
    "8",
    "--control_n",
    "60",
    "--device",
    "cpu",
]
for s in TTS:
    argv += ["--tts", s]
KKS.main(argv)
rows = [json.loads(line) for line in open(os.path.join(out, "dialogues.jsonl"))]
summ = json.load(open(os.path.join(out, "summary.json")))
ok_rows = [r for r in rows if r["status"] == "ok"]
levels = {c for _o, c in ORIG.values()}
exact = all(round(-r["score"], 6) in levels for r in ok_rows)
ck(len(ok_rows) >= 60, f"{len(ok_rows)} of {len(rows)} rows scored (evenly spaced over the corpus)")
ck(
    exact and len({r["n_tokens"] for r in ok_rows}) > 1,
    "every score is exactly -c_row whatever the answer's token count (length-normalised)",
    f"token counts {sorted({r['n_tokens'] for r in ok_rows})}",
)
c = summ["control"]
ck(
    c.get("n", 0) >= 30 and c["frac_true_higher"] == 1.0 and c["mean_swapped"] < -4.0,
    "control: answer-swapped copies score ~-5 and below the true answer in every pair",
    json.dumps(c),
)
q = summ["questions"]
ck(
    abs(q["known"] - 0.25 * q["scored"]) <= 0.1 * q["scored"] + 2,
    "known ~= the top 25% of the scored questions",
    json.dumps(q),
)
kr = np.load(os.path.join(out, "known_rows.npy"))
ck(len(kr) > 0 and np.all(np.diff(kr) > 0), "known_rows: ascending, unique, non-empty", str(len(kr)))

# ---------------------------------------------------------------------------------------------------------------
print("[4] the mix loader draws every concatenated knowledge piece from knowledge_rows (the arm's knob)")
from moshi_family.personaplex.train_data import build_knowledge_mix_loader  # noqa: E402

CONCAT = {
    "barge_lead_sec": [0.2, 0.8],
    "barge_min_answer_sec": 0.5,
    "barge_prob": 0.3,
    "dialogues": [2, 4],
    "gap_sec": [0.3, 1.5],
    "min_next_sec": 5.0,
    "tail_frames": 1,
}  # ab2_know30_concat's config.yaml
META = glob.glob(
    "work/i6_experiments/users/dorian_koch/speech_llm/persona_knowledge/KnowledgeConcatMeta.Jq5Gwzdr2Ray/output/meta.npz"
)[0]
keep = np.unique(np.random.default_rng(1).choice(len(codes_ids), 300, replace=False)).astype(np.int64)
keep_path = os.path.join(out, "keep.npy")
np.save(keep_path, keep)


def pieces(knowledge_rows):
    im = []
    it = build_knowledge_mix_loader(
        "output/persona_v3/p1_data",
        CODES,
        tok,
        batch_size=8,
        share=0.3,
        system_prompt_key="context",
        voice_codes_key="voice_codes",
        knowledge_meta=META,
        knowledge_concat=CONCAT,
        item_meta=im,
        knowledge_rows=knowledge_rows,
    )
    for _ in range(5):
        next(it)
    return [int(p.row) for m in im if m["source"] == "knowledge" for p in m["plan"].pieces]


with_k, without = pieces(keep_path), pieces(None)
ck(
    len(with_k) >= 20 and set(with_k) <= set(keep.tolist()),
    "knowledge_rows: every piece is a kept row",
    f"{len(with_k)} pieces",
)
ck(not set(without) <= set(keep.tolist()), "without it pieces come from outside the keep-list (non-vacuous)")

print(f"\n{'FAILED: ' + '; '.join(fails) if fails else 'ALL OK'}" + (f" (mutant {MUTANT})" if MUTANT else ""))
sys.exit(1 if fails else 0)
