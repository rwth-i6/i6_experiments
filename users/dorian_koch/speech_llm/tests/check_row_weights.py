"""Guard: per-row loss weights (``train_row_weights``, 2026-09-28; user: "maybe we could downweight the loss on the entire
row if it contains longer monologues?") scale EXACTLY the loss of their row, on the real data path, and change nothing
when off.

WHY. The knob can fail in ways a loss curve hides: a loader that attaches the weight to the wrong row (the permutation,
a swapped read, the knowledge mix's second stream), a loss that re-normalises (under the weighted mean a weight in the
DENOMINATOR too cancels for a batch of all-flagged rows and scales every OTHER row up), a weight that misses the text
stream or the user's codebooks, a launcher that reads the key and never passes it. Everything runs through the real
code: p4's corpus, the real ``build_data_loader`` / ``build_knowledge_mix_loader`` with the 2-hour standard's read
knobs (swap, stored user augmentation; the default know30_concat mix), the launcher's REAL ``loss_step`` (its source cut
out of ``finetune_launcher.main``) and the real ``duplex_ce_loss``. Weights: ``check_row_filters.py job``'s ``OUT``
(``ThresholdRowWeights`` rule, 0.3 on rows whose longest stretch > 20 s).

  1. FILE. ``load_row_weights``: "" -> None; a wrong length, a negative or non-finite entry, an int array raise.
  2. PLAIN LOADER. With the weights: codes + masks identical item for item to the run without (rows, order, swap /
     augmentation draws unchanged); the third tensor is 1 on the prompt and the ROW's weight (``item_meta``'s table
     row) on every dialogue frame; rows at 0.3 and at 1 both occur. Composed with ``train_rows`` (the episode-edge
     keep-list): every row read is kept and its weight right.
  3. MIX LOADER. The same for 3 batches of the know30_concat mix: codes identical batch for batch, JRE items carry
     their row's weight, knowledge items all ones.
  4. LOSS. Per-position gradient ratio weighted / unweighted = exactly the row's weight on EVERY target of that row
     (text, assistant codebooks 0-7, user codebooks 8-15) and exactly 1 on the other row; a batch of all-0.3 rows gives
     0.3 x the loss; ``sample_weight_mean`` / ``sample_weighted_frac`` recount. CAN FAIL: the weight put in the
     denominators too (a weighted mean) moves the other row's ratio off 1.
  5. LAUNCHER. The real ``loss_step`` with ``train_row_weights`` set passes the third tensor: same ratios. CAN FAIL: a
     ``loss_step`` that drops ``sample_frame_weight`` gives ratio 1 on the weighted row.
  6. OFF == BEFORE. Weight None: loss + every component bit-identical to the pre-change ``duplex_ce_loss`` (full_duplex
     ``fc1b4cff``, from git); an all-ones weight gives the same loss bit for bit.
  7. HELD-OUT. ``heldout_loss.loss_settings`` ignores the key; ``make_loss_step`` passes no sample weight.

moshi_family venv, CPU, ~4 min:
    ./hpc-venv.py --cluster i6-rz moshi_family --sh 'cd /home/tt201262/setups/2026-01-speech-llm && HF_HUB_OFFLINE=1 \\
        python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_row_weights.py'
"""

import ast
import importlib.util
import inspect
import os
import subprocess
import sys
import tempfile
import textwrap

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import moshi_family.personaplex.train_data as td  # noqa: E402
import moshi_family.train_data_common as tdc  # noqa: E402
from moshi_family.personaplex import heldout_loss as HL  # noqa: E402
from moshi_family.train_data_common import TEXT_EPAD_ID, read_arrow_table  # noqa: E402

LIB = os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex")
LAUNCHER = os.path.join(LIB, "moshi_family", "personaplex", "finetune_launcher.py")
W = "/home/tt201262/setups/2026-01-speech-llm/work/i6_experiments/users/dorian_koch/speech_llm"
JRE = f"{W}/persona_prompts/AttachPersonaPrompts.DHvnX2ifpvad/output/dataset"
KNOW = f"{W}/persona_knowledge/KnowledgeDialogueCodes.DMgw9Srb4bmg/output/dataset"
META = f"{W}/persona_knowledge/KnowledgeConcatMeta.Jq5Gwzdr2Ray/output/meta.npz"
RF_OUT = "/hpcwork/p0023999/tt201262/guards/check_row_filters"
EDGE_KEEP = "/hpcwork/p0023999/tt201262/guards/check_episode_edge_rows/keep.npy"
PRE_CHANGE = "fc1b4cff"  # full_duplex HEAD before the per-row weight
WT = 0.3
fails = []


def ck(c, label, extra=""):
    print(f"  [{'ok' if c else 'FAIL'}] {label}{'  ' + extra if extra else ''}", flush=True)
    if not c:
        fails.append(label)


def raises(fn):
    try:
        fn()
    except (ValueError, AssertionError, RuntimeError) as e:
        return str(e)[:100]
    return None


table = read_arrow_table(JRE)
N_ROWS = table.num_rows
WPATH = os.path.join(RF_OUT, "weights_0.3_above20.npy")
weights = np.load(WPATH)
tmp = tempfile.mkdtemp(prefix="check_row_weights_")

# ---------------------------------------------------------------------------------------------------------------
print("1. the weights file", flush=True)
ck(tdc.load_row_weights("", N_ROWS) is None and tdc.load_row_weights(None, N_ROWS) is None, '"" / None -> None')
ck(tdc.load_row_weights(WPATH, N_ROWS).dtype == np.float32, "the real file loads (float32, one per table row)")
for name, arr in (
    ("short", np.ones(N_ROWS - 1, np.float32)),
    ("neg", np.r_[np.ones(N_ROWS - 1), -0.1].astype(np.float32)),
    ("nan", np.r_[np.ones(N_ROWS - 1), np.nan].astype(np.float32)),
    ("int", np.ones(N_ROWS, np.int64)),
):
    np.save(os.path.join(tmp, f"{name}.npy"), arr)
    ck(bool(raises(lambda: tdc.load_row_weights(os.path.join(tmp, f"{name}.npy"), N_ROWS))), f"{name} raises")
ck(0.1 < float((weights == np.float32(WT)).mean()) < 0.5, "the weights down-weight some rows",
   f"{float((weights == np.float32(WT)).mean()):.3f} at {WT}")  # fmt: skip

# ---------------------------------------------------------------------------------------------------------------
tok = td.PersonaPlexTokenizer(
    "nvidia/personaplex-7b-v1",
    device="cpu",
    cfg=td.PersonaPlexDataConfig(
        duration_sec=60.0, silence_frames=6, emit_epad=True, onset_floor=True, word_overlap="drop"
    ),
)
knobs = dict(system_prompt_key="context", voice_codes_key="voice_codes", swap_channels=True, stored_user_augment=True)


def weight_errors(batches, meta, bsz):
    """Per item: prompt frames 1, dialogue frames == the weight of the row item_meta names (knowledge: 1)."""
    errs, n_down, n_one = 0, 0, 0
    for b, batch in enumerate(batches):
        codes, mask, wt = batch
        for i in range(codes.shape[0]):
            m = meta[b * bsz + i]
            want = 1.0 if m["source"] == "knowledge" else float(weights[m["row"]])
            nf = int(mask[i].sum())
            lead = int(torch.nonzero(mask[i])[0].item())  # first dialogue frame
            dia = wt[i, lead : lead + nf, 0]
            pre = wt[i, :lead, 0]
            ok = bool(torch.all(pre == 1.0)) and bool(torch.all(dia == torch.tensor(want, dtype=torch.float32)))
            errs += not ok
            if m["source"] == "primary":
                n_down += want != 1.0
                n_one += want == 1.0
    return errs, n_down, n_one


print("\n2. the plain loader", flush=True)
B1, N1 = 1, 32
m_on, m_off = [], []
on = td.build_data_loader(
    JRE, tok, batch_size=B1, seed=0, infinite=False, item_meta=m_on, train_row_weights=WPATH, **knobs
)
off = td.build_data_loader(JRE, tok, batch_size=B1, seed=0, infinite=False, item_meta=m_off, **knobs)
b_on = [next(on) for _ in range(N1)]
b_off = [next(off) for _ in range(N1)]
ck(
    all(len(x) == 3 and len(y) == 2 and torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(b_on, b_off))
    and [m["row"] for m in m_on] == [m["row"] for m in m_off] and [m["swap"] for m in m_on] == [m["swap"] for m in m_off],
    "codes + masks + rows + swap draws identical with and without the weights, item for item",
)  # fmt: skip
e, n_down, n_one = weight_errors(b_on, m_on, B1)
ck(e == 0 and n_down > 0 and n_one > 0, "third tensor = 1 on the prompt, the row's weight on its dialogue frames",
   f"{N1} items: {n_down} at {WT}, {n_one} at 1, {sum(m['swap'] for m in m_on)} swapped")  # fmt: skip
m_c = []
comp = td.build_data_loader(JRE, tok, batch_size=B1, seed=0, infinite=False, item_meta=m_c, train_rows=EDGE_KEEP,
                            train_row_weights=WPATH, **knobs)  # fmt: skip
b_c = [next(comp) for _ in range(16)]
kset = set(np.load(EDGE_KEEP).tolist())
e, n_down, _ = weight_errors(b_c, m_c, B1)
ck(e == 0 and all(m["row"] in kset for m in m_c), "with train_rows (edge keep-list): rows kept, weights right",
   f"{n_down} of 16 down-weighted")  # fmt: skip

# ---------------------------------------------------------------------------------------------------------------
print("\n3. the knowledge-mix loader", flush=True)
B3, N3 = 8, 3
concat = {"barge_lead_sec": [0.2, 0.8], "barge_min_answer_sec": 0.5, "barge_prob": 0.3, "dialogues": [2, 4],
          "gap_sec": [0.3, 1.5], "min_next_sec": 5.0, "tail_frames": 1}  # fmt: skip


def mix(meta, w):
    it = td.build_knowledge_mix_loader(JRE, KNOW, tok, batch_size=B3, share=0.3, seed=0, knowledge_meta=META,
                                       knowledge_concat=concat, item_meta=meta, train_row_weights=w, **knobs)  # fmt: skip
    return [next(it) for _ in range(N3)]


mm_on, mm_off = [], []
x_on, x_off = mix(mm_on, WPATH), mix(mm_off, None)
ck(
    all(
        len(a) == 3 and len(b) == 2 and torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) for a, b in zip(x_on, x_off)
    ),
    f"{N3} mixed batches: codes + masks identical with and without the weights",
)
e, n_down, n_one = weight_errors(x_on, mm_on, B3)
n_k = sum(m["source"] == "knowledge" for m in mm_on)
ck(e == 0 and n_k > 0 and n_down + n_one > 0, "JRE items carry their row's weight, knowledge items all ones",
   f"{n_down + n_one} JRE ({n_down} at {WT}), {n_k} knowledge")  # fmt: skip

# ---------------------------------------------------------------------------------------------------------------
print("\n4. the loss: exactly the row's weight on every target of its row, nothing elsewhere", flush=True)


class Out:
    pass


class StubLM:
    """``model(codes)`` -> fixed random LEAF logits (their .grad is the loss's gradient per position)."""

    def __init__(self, b: int, t: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.text = torch.randn(b, 1, t, 32000, generator=g).requires_grad_(True)
        self.audio = torch.randn(b, 16, t, 2048, generator=g).requires_grad_(True)

    def __call__(self, codes, depformer_frame_mask=None):
        o = Out()
        o.text_logits, o.logits = self.text, self.audio
        o.text_mask = torch.ones(*self.text.shape[:3], dtype=torch.bool)
        o.mask = torch.ones(*self.audio.shape[:3], dtype=torch.bool)
        return o


# two real items of the plain loader in one batch (collate pads them): row 0 at WT, row 1 at 1
codes, mask = tdc.collate_codes([(b[0][0], b[1][0]) for b in b_off[:2]])
Bn, T = codes.shape[0], codes.shape[2]
S = mask.float().clone()
S[0][mask[0]] = WT  # [B, T]: WT on row 0's dialogue, 1 elsewhere
S[1] = 1.0
LKW = dict(dep_q=16, weighted_mean=True, text_epad_id=TEXT_EPAD_ID, return_components=True, text_epad_weight=1.0,
           user_semantic_weight=1.0)  # fmt: skip


def grads(sfw, loss_fn=tdc.paper_weight_loss, **extra):
    lm = StubLM(Bn, T)
    loss, comps = loss_fn(lm(codes), codes, mask, sample_frame_weight=sfw, **LKW, **extra)
    loss.backward()
    return lm.text.grad.abs().sum(-1)[:, 0].double(), lm.audio.grad.abs().sum(-1).double(), loss, comps


def ratios(g_w, g_1):
    """The per-position ratio on row 0 / row 1 where the unweighted gradient is non-zero -> (min, max) each."""
    out = []
    for b in range(Bn):
        sel = g_1[b] > 0
        r = g_w[b][sel] / g_1[b][sel]
        out.append((float(r.min()), float(r.max())))
    return out


def near(x, y, tol=1e-5):
    return abs(x - y) <= tol * max(1.0, abs(y))


gt_w, ga_w, l_w, c_w = grads(S)
gt_1, ga_1, l_1, _ = grads(None)
for name, gw, g1 in (("text", gt_w, gt_1), ("assistant cb 0-7", ga_w[:, :8], ga_1[:, :8]),
                     ("user cb 8-15", ga_w[:, 8:], ga_1[:, 8:])):  # fmt: skip
    (lo0, hi0), (lo1, hi1) = ratios(gw.reshape(Bn, -1), g1.reshape(Bn, -1))
    ck(near(lo0, WT) and near(hi0, WT) and near(lo1, 1.0) and near(hi1, 1.0),
       f"{name}: x{WT} on the weighted row, x1 on the other", f"[{lo0:.7f}, {hi0:.7f}] / [{lo1:.7f}, {hi1:.7f}]")  # fmt: skip
_, _, l_all, _ = grads(torch.full((Bn, T), WT))
ck(near(float(l_all), WT * float(l_1)), f"a batch of all-{WT} rows: {WT} x the loss (numerator only)",
   f"{float(l_all):.6f} vs {WT * float(l_1):.6f}")  # fmt: skip
valid = mask.reshape(-1)
ck(
    abs(c_w["sample_weight_mean"] - float(S.reshape(-1)[valid].mean())) < 1e-6
    and abs(c_w["sample_weighted_frac"] - float((S.reshape(-1)[valid] != 1).float().mean())) < 1e-6,
    "sample_weight_mean / sample_weighted_frac recount",
    f"{c_w['sample_weight_mean']:.4f} / {c_w['sample_weighted_frac']:.4f}",
)
print("   CAN FAIL:", flush=True)
SRC = inspect.getsource(tdc.duplex_ce_loss)
# (2026-09-28b: the numerator is built as ``text_term`` so the flagged-word weight can join it; same arithmetic)
_A = "text_loss = text_term.sum() / text_denom.clamp(min=1e-8)"
assert _A in SRC, "duplex_ce_loss's weighted text numerator changed; update the mutant"
_ns = dict(vars(tdc))
exec(SRC.replace(_A, "text_loss = text_term.sum() / ((text_w * sample_frame_weight.reshape(-1)).sum() if "
                 "sample_frame_weight is not None else text_denom).clamp(min=1e-8)"), _ns)  # fmt: skip


def renorm_loss(out, codes, loss_mask, *, dep_q, **k):
    """Mutant: the weight in the text DENOMINATOR too (a re-normalised weighted mean), PersonaPlex's weights."""
    return _ns["duplex_ce_loss"](out, codes, loss_mask, dep_q=dep_q, other_audio_weight=0.02, text_pad_weight=0.3, **k)


gm, _, _, _ = grads(S, loss_fn=renorm_loss)
gm1, _, _, _ = grads(None, loss_fn=renorm_loss)
(_, _), (lo1, hi1) = ratios(gm, gm1)
ck(not (near(lo1, 1.0) and near(hi1, 1.0)), "a re-normalising weight is caught (the other row's ratio moves off 1)",
   f"[{lo1:.4f}, {hi1:.4f}]")  # fmt: skip

# ---------------------------------------------------------------------------------------------------------------
print("\n5. the launcher's real loss_step", flush=True)
src_l = open(LAUNCHER).read()
main_fn = next(n for n in ast.walk(ast.parse(src_l)) if isinstance(n, ast.FunctionDef) and n.name == "main")
ls_node = next(n for n in ast.walk(main_fn) if isinstance(n, ast.FunctionDef) and n.name == "loss_step")
LOSS_STEP_SRC = textwrap.dedent(ast.get_source_segment(src_l, ls_node))


def loss_step_for(src, model, rw="/weights.npy"):
    ns = {
        "model": model, "device": "cpu", "dep_q": 16, "personaplex_loss": td.personaplex_loss,
        "TEXT_EPAD_ID": TEXT_EPAD_ID, "loss_norm": "weight", "emit_epad": True, "text_epad_weight": 1.0,
        "floor_weight": None, "text_pad_weight": 0.3, "user_semantic_weight": 1.0, "depformer_skip_masked": True,
        "turn_tail_weight": None, "train_row_weights": rw, "train_word_spans": None,
        "user_head": None,  # the user-text head (eef4e848), off
    }  # fmt: skip
    exec(src, ns)
    return ns["loss_step"]


def ls_grads(src, batch, rw="/weights.npy"):
    lm = StubLM(Bn, T)
    loss, comps = loss_step_for(src, lm, rw)(batch)
    loss.backward()
    return lm.text.grad.abs().sum(-1)[:, 0].double(), lm.audio.grad.abs().sum(-1).double()


batch_w = (codes, mask, S[..., None])
lt_w, la_w = ls_grads(LOSS_STEP_SRC, batch_w)
lt_1, la_1 = ls_grads(LOSS_STEP_SRC, (codes, mask), rw=None)
(lo0, hi0), (lo1, hi1) = ratios(la_w.reshape(Bn, -1), la_1.reshape(Bn, -1))
(t0, t1), _ = ratios(lt_w, lt_1)
ck(near(lo0, WT) and near(hi0, WT) and near(lo1, 1.0) and near(hi1, 1.0) and near(t0, WT) and near(t1, WT),
   f"loss_step with train_row_weights: x{WT} on the weighted row (text + 16 codebooks), x1 on the other")  # fmt: skip
mut = LOSS_STEP_SRC.replace("sample_frame_weight=sample_frame_weight,", "")
assert mut != LOSS_STEP_SRC, "the launcher's loss_step does not pass sample_frame_weight"
mt, _ = ls_grads(mut, batch_w)
(m0, _), _ = ratios(mt, lt_1)
ck(not near(m0, WT), "CAN FAIL: a loss_step that drops sample_frame_weight is caught (text ratio 1)", f"{m0:.4f}")

# ---------------------------------------------------------------------------------------------------------------
print("\n6. off == before (pre-change duplex_ce_loss from git)", flush=True)
src_old = subprocess.run(
    ["git", "-C", LIB, "show", f"{PRE_CHANGE}:./moshi_family/train_data_common.py"], capture_output=True, text=True,
    check=True,
).stdout  # fmt: skip
old_path = os.path.join(tmp, "tdc_old.py")
open(old_path, "w").write(src_old)
spec = importlib.util.spec_from_file_location("tdc_old", old_path)
tdc_old = importlib.util.module_from_spec(spec)
sys.modules["tdc_old"] = tdc_old
spec.loader.exec_module(tdc_old)
ck("sample_frame_weight" not in src_old, "the reference predates the knob")
g = torch.Generator().manual_seed(1)
o = Out()
o.text_logits = torch.randn(Bn, 1, T, 32000, generator=g)
o.logits = torch.randn(Bn, 16, T, 2048, generator=g)
o.text_mask = torch.ones(Bn, 1, T, dtype=torch.bool)
o.mask = torch.ones(Bn, 16, T, dtype=torch.bool)
kw6 = {k: v for k, v in LKW.items()}
l_new, c_new = tdc.paper_weight_loss(o, codes, mask, **kw6)
l_old, c_old = tdc_old.paper_weight_loss(o, codes, mask, **kw6)
ck(
    torch.equal(l_new, l_old) and c_new == c_old,
    "weight None: loss + every component bit-identical to the pre-change code",
)
l_one, _ = tdc.paper_weight_loss(o, codes, mask, sample_frame_weight=torch.ones(Bn, T), **kw6)
ck(torch.equal(l_one, l_old), "all-ones weight: the same loss bit for bit")

# ---------------------------------------------------------------------------------------------------------------
print("\n7. the held-out loss stays unweighted", flush=True)
base_cfg = {"loss_norm": "weight", "emit_epad": True, "text_epad_weight": 1.0, "onset_floor": True, "duration_sec": 60}
ck(HL.loss_settings({**base_cfg, "train_row_weights": "/w.npy"}) == HL.loss_settings(base_cfg), "same settings")
ck("sample_frame_weight" not in inspect.getsource(HL.make_loss_step), "make_loss_step passes no sample weight")

print()
if fails:
    print(f"FAIL ({len(fails)}):", *fails, sep="\n  ")
    sys.exit(1)
print("ALL OK")
