"""Guard: the assistant turn cap (``asst_turn_cap_s``, ablation ab2_turncap10, 2026-09-25) zeroes EXACTLY the
assistant-side loss more than the cap into a continuous assistant turn, causally, on the real data path, and changes
nothing when off.

WHY. The ablation asks whether dropping the loss deep into long assistant turns helps ("Moshi training was unstable
for turns longer than 10 s"). A cap that also silenced the user's codebooks, missed the text stream, used the stored
side on a swapped read, or looked at the future (a turn that "continues" only because the next word comes) would
test something else -- and a loss curve shows none of it. Everything runs through the real code: p4's corpus, the
real ``build_data_loader`` with the arm's read knobs (swap, stored user augmentation), the launcher's REAL
``loss_step`` (its source cut out of ``finetune_launcher.main`` and executed with the variables main() binds) and the
real ``duplex_ce_loss``.

  1. DEFINITION. ``assistant_turn_position`` on hand-made words: pauses < 1 s stay in one turn, >= 1 s start a new
     one; the onset frame of the opening word is outside (it starts the turn); frames within 1 s after the last word
     are inside, later ones outside; the position is clipped at the window start.
  2. CAUSALITY (real rows). Removing every assistant word whose onset frame is >= f leaves the position on frames
     0..f identical (frame f's target changed). CAN FAIL: a non-causal variant (turn span = first start .. last end,
     no 1 s tail) is caught.
  3. LOADER. With the cap the loader yields the SAME codes and masks, batch for batch, as without (rows, order,
     swap / augmentation draws unchanged), plus a third tensor == ``pos > cap`` recomputed from THIS read's assistant
     words (``alignments_other`` on a swapped read), 1 on the prompt; both swapped and unswapped reads are covered.
  4. LOSS (launcher's loss_step, stub LM with leaf logits). Gradient is exactly 0 on every text logit and on the
     assistant's audio codebooks 0-7 at capped frames, non-zero on the user's codebooks 8-15 there and on every
     stream at uncapped frames; the text loss == manual ``sum(ce*w_cls*w_frame)/sum(w_cls*w_frame)``;
     ``asst_frame_weighted_frac`` == the capped share. CAN FAIL: a loss_step that drops the weight, and a
     ``duplex_ce_loss`` that caps all 16 codebooks, are both caught.
  5. OFF == BEFORE. Cap None: the loss equals the pre-change ``duplex_ce_loss`` (full_duplex ``27625233``, loaded from
     git) bit for bit, with every component; an all-ones weight is bit-identical too.
  6. HELD-OUT UNMASKED. ``heldout_loss.loss_settings`` of the arm's config == of the baseline's (the cap is not a
     held-out setting), and its ``make_loss_step`` source passes no assistant frame weight.

Also prints the capped share on the rows read.

moshi_family venv, CPU, ~2-3 min (mimi + spm from the HF cache):
    ./hpc-venv.py --cluster i6-rz moshi_family --sh 'cd /home/tt201262/setups/2026-01-speech-llm && HF_HUB_OFFLINE=1 \\
        python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_turn_cap.py'
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

import moshi_family.train_data_common as tdc  # noqa: E402
from moshi_family.personaplex import heldout_loss as HL  # noqa: E402
from moshi_family.personaplex.train_data import (  # noqa: E402
    PersonaPlexDataConfig,
    PersonaPlexTokenizer,
    build_data_loader,
    personaplex_loss,
)
from moshi_family.train_data_common import (  # noqa: E402
    AUDIO_OFFSET,
    TEXT_EPAD_ID,
    TEXT_PADDING_ID,
    TEXT_ROW,
    assistant_turn_position,
    decode_codes_row,
    normalize_alignments,
    read_arrow_table,
)

LIB = os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex")
LAUNCHER = os.path.join(LIB, "moshi_family", "personaplex", "finetune_launcher.py")
P4 = "output/persona_v4/p1_data"  # AttachPersonaPrompts.DHvnX2ifpvad: the 2-hour standard's corpus
PRE_CHANGE = "27625233"  # full_duplex HEAD before the turn cap
CAP, GAP, FR = 10.0, 1.0, 12.5
N_ITEMS = 24
fails = []


def ck(c, label, extra=""):
    print(f"  [{'ok' if c else 'FAIL'}] {label}{'  ' + extra if extra else ''}", flush=True)
    if not c:
        fails.append(label)


# ---------------------------------------------------------------------------------------------------------------
print("1. definition on hand-made words", flush=True)
# words (s): 1.00-2.00, 2.50-3.00 (pause 0.5 -> same turn), 4.00-5.00 (pause 1.0 -> new turn), window 8 s = 100 frames
pos = assistant_turn_position([(1.0, 2.0), (2.5, 3.0), (4.0, 5.0)], 100, FR, GAP)
t = np.arange(100) / FR
ck(np.isnan(pos[: int(1.0 * FR) + 1]).all(), "frames up to the opening word's onset frame are outside")
ck(abs(pos[13] - (13 / FR - 1.0)) < 1e-9 and abs(pos[48] - (48 / FR - 1.0)) < 1e-9, "position = t - turn start")
ck(np.isnan(pos[50]), "the onset frame of a word after a 1.0 s pause is outside (new turn)")
ck(abs(pos[51] - (51 / FR - 4.0)) < 1e-9, "the new turn counts from its own start")
ck(
    not np.isnan(pos[int(np.ceil(6.0 * FR)) - 1]) and np.isnan(pos[int(np.ceil(6.0 * FR))]),
    "1 s tail after the last word",
)
pos_neg = assistant_turn_position([(-3.0, 1.0), (1.5, 2.0)], 40, FR, GAP)
ck(abs(pos_neg[5] - 5 / FR) < 1e-9, "a turn begun before the window counts from the window start")
ck(pos_neg[0] == 0.0, "frame 0 is inside a turn begun before the window (position 0)")

# ---------------------------------------------------------------------------------------------------------------
tok = PersonaPlexTokenizer(
    "nvidia/personaplex-7b-v1",
    device="cpu",
    cfg=PersonaPlexDataConfig(duration_sec=60.0, emit_epad=True, onset_floor=True, word_overlap="drop"),
)
table = read_arrow_table(P4)
order = np.random.default_rng(0).permutation(table.num_rows)


def spans_of(al):
    return [s for _w, s, _k in normalize_alignments(al)]


def noncausal(word_spans, num_frames, frame_rate, max_gap_sec=GAP):
    """Mutant: turn span = [first start, last end] (no causal onset, no tail) -- whether a frame is in the turn then
    depends on words that start later."""
    out = np.full(num_frames, np.nan)
    tt = np.arange(num_frames) / frame_rate
    turns = []
    for s, e in sorted(word_spans):
        if turns and s < turns[-1][1] + max_gap_sec:
            turns[-1][1] = max(turns[-1][1], e)
        else:
            turns.append([s, e])
    for s, e in turns:
        m = (tt >= s) & (tt < e)
        out[m] = tt[m] - max(0.0, s)
    return out


def causality(fn) -> tuple[int, int]:
    n_cmp = n_bad = 0
    for i in order[:40]:
        sp = spans_of(table.column("alignments")[int(i)].as_py())
        n = min(int(table.column("n_frames")[int(i)].as_py()), 750)
        full = fn(sp, n, FR, GAP)
        # cut at onsets that CONTINUE a turn (pause < 1 s): there a leaky definition would differ
        onsets = sorted({int(s * FR // 1) for s, _e in sp})
        onsets = [f for f in onsets if 2 <= f < n - 1 and not np.isnan(full[f])][:4]
        for f in onsets:
            cut = [(s, e) for s, e in sp if int(s * FR // 1) < f]
            part = fn(cut, n, FR, GAP)
            n_cmp += 1
            n_bad += int(not np.array_equal(full[: f + 1], part[: f + 1], equal_nan=True))
    return n_cmp, n_bad


print("\n2. causality on real rows", flush=True)
n_cmp, n_bad = causality(assistant_turn_position)
ck(n_cmp >= 40 and n_bad == 0, "position on frames <= f independent of words starting at >= f", f"{n_cmp} cuts")
n_cmp, n_bad = causality(noncausal)
ck(n_bad > 0, "CAN FAIL: the non-causal span variant is caught", f"{n_bad} of {n_cmp} cuts differ")

# ---------------------------------------------------------------------------------------------------------------
print("\n3. the real loader: same stream, weight from this read's assistant words", flush=True)
kw = dict(
    batch_size=1,
    seed=0,
    system_prompt_key="context",
    voice_codes_key="voice_codes",
    swap_channels=True,
    stored_user_augment=True,
    infinite=False,
)
on = build_data_loader(P4, tok, turn_cap_s=CAP, turn_gap_s=GAP, **kw)
off = build_data_loader(P4, tok, **kw)
n_same = n_w = n_swapped = 0
capped = frames = 0
items = []
for k in range(N_ITEMS):
    b1, b0 = next(on), next(off)
    ck_same = len(b1) == 3 and len(b0) == 2 and torch.equal(b1[0], b0[0]) and torch.equal(b1[1], b0[1])
    n_same += int(ck_same)
    codes, mask, w = b1[0][0], b1[1][0], b1[2][0, :, 0]
    i = int(order[k])
    ca, cu, _fr, _al = decode_codes_row(table, i)
    p, n = int((~mask).sum()), int(mask.sum())
    asst = codes[AUDIO_OFFSET : AUDIO_OFFSET + 8, p : p + n].numpy()
    swapped = not np.array_equal(asst, np.asarray(ca)[:, :n])
    n_swapped += int(swapped)
    al = table.column("alignments_other" if swapped else "alignments")[i].as_py()
    want = torch.ones(codes.shape[1])
    want[p:] = torch.from_numpy(~(assistant_turn_position(spans_of(al), n, FR, GAP) > CAP)).float()
    n_w += int(torch.equal(w, want))
    capped += int((w[p:] == 0).sum())
    frames += n
    items.append((codes, mask, w))
ck(n_same == N_ITEMS, "codes + mask identical with and without the cap, item for item", f"{n_same}/{N_ITEMS}")
ck(n_w == N_ITEMS, "weight == (position > 10 s) of this read's assistant, 1 on the prompt", f"{n_w}/{N_ITEMS}")
ck(0 < n_swapped < N_ITEMS, "both swapped and stored reads covered", f"{n_swapped} swapped")
ck(capped > 0, "some frames capped", f"{capped / frames:.3f} of {frames} dialogue frames")

# ---------------------------------------------------------------------------------------------------------------
print("\n4. the launcher's real loss_step: zero gradient exactly on the capped assistant targets", flush=True)
main_fn = next(
    n for n in ast.walk(ast.parse(open(LAUNCHER).read())) if isinstance(n, ast.FunctionDef) and n.name == "main"
)
ls_node = next(n for n in ast.walk(main_fn) if isinstance(n, ast.FunctionDef) and n.name == "loss_step")
LOSS_STEP_SRC = textwrap.dedent(ast.get_source_segment(open(LAUNCHER).read(), ls_node))


class Out:
    pass


class StubLM:
    """``model(codes)`` -> fixed random LEAF logits (so their .grad is the loss's gradient per position)."""

    def __init__(self, t: int, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        self.text = torch.randn(1, 1, t, 32000, generator=g).requires_grad_(True)
        self.audio = torch.randn(1, 16, t, 2048, generator=g).requires_grad_(True)

    def __call__(self, codes, depformer_frame_mask=None):
        o = Out()
        o.text_logits, o.logits = self.text, self.audio
        o.text_mask = torch.ones(1, 1, self.text.shape[2], dtype=torch.bool)
        o.mask = torch.ones(1, 16, self.text.shape[2], dtype=torch.bool)
        return o


def loss_step_for(cap, src=LOSS_STEP_SRC, loss_fn=personaplex_loss, model=None):
    ns = {
        "model": model,
        "device": "cpu",
        "dep_q": 16,
        "personaplex_loss": loss_fn,
        "TEXT_EPAD_ID": TEXT_EPAD_ID,
        "loss_norm": "weight",
        "emit_epad": True,
        "text_epad_weight": 1.0,
        "floor_weight": None,
        "text_pad_weight": 0.3,
        "user_semantic_weight": None,
        "depformer_skip_masked": True,
        "asst_turn_cap_s": cap,
    }
    exec(src, ns)
    return ns["loss_step"]


# the item with the most capped frames
codes, mask, w = max(items, key=lambda it: int((it[2] == 0).sum()))
T = codes.shape[1]
batch = (codes[None], mask[None], w[None, :, None])
cap_f = (w == 0) & mask
free_f = (w == 1) & mask


def grads(src=LOSS_STEP_SRC, loss_fn=personaplex_loss):
    lm = StubLM(T)
    loss, comps = loss_step_for(CAP, src, loss_fn, lm)(batch)
    loss.backward()
    gt = lm.text.grad[0, 0].abs().sum(-1)  # [T]
    ga = lm.audio.grad[0].abs().sum(-1)  # [16, T]
    return gt, ga, comps


gt, ga, comps = grads()
ck(bool((gt[cap_f] == 0).all()) and bool((gt[free_f] > 0).all()), "text: grad 0 at capped frames, > 0 elsewhere")
ck(bool((ga[:8][:, cap_f] == 0).all()), "assistant codebooks 0-7: grad 0 at capped frames")
ck(bool((ga[8:][:, cap_f] > 0).all()), "user codebooks 8-15: grad > 0 at capped frames (still trained)")
ck(bool((ga[:, free_f] > 0).all()), "every codebook: grad > 0 at uncapped frames")
ck(
    abs(comps["asst_frame_weighted_frac"] - float(cap_f.sum() / mask.sum())) < 1e-6,
    "asst_frame_weighted_frac == capped share",
    f"{comps['asst_frame_weighted_frac']:.4f}",
)
# manual text loss (weighted mean): sum(ce * w_cls * w_frame) / sum(w_cls * w_frame) over valid positions
lm = StubLM(T)
_, comps = loss_step_for(CAP, model=lm)(batch)
tgt = codes[TEXT_ROW].clamp(0, 31999)
ce = torch.nn.functional.cross_entropy(lm.text[0, 0].detach().float(), tgt, reduction="none")
cls_w = torch.where(tgt == TEXT_PADDING_ID, 0.3, 1.0)
cls_w = torch.where(tgt == TEXT_EPAD_ID, 1.0, cls_w)
wv = cls_w * w * mask.float()
manual = float((ce * wv).sum() / wv.sum())
ck(abs(comps["text_loss"] - manual) < 1e-5, "text loss == manual frame-weighted CE", f"{comps['text_loss']:.6f}")

print("   CAN FAIL:", flush=True)
mut_src = LOSS_STEP_SRC.replace("asst_frame_weight=asst_frame_weight,", "")
assert mut_src != LOSS_STEP_SRC, "the launcher's loss_step does not pass asst_frame_weight"
gt_m, ga_m, _ = grads(src=mut_src)
ck(bool((gt_m[cap_f] > 0).any()), "a loss_step that drops the weight is caught (text grad at capped frames)")
ALL16_SRC = inspect.getsource(tdc.duplex_ce_loss)
_N_ASST = "n_asst = min(dep_q, max(1, (codes.shape[1] - AUDIO_OFFSET) // 2))"
assert _N_ASST in ALL16_SRC
_ns = dict(vars(tdc))
exec(ALL16_SRC.replace(_N_ASST, "n_asst = dep_q"), _ns)


def all16_loss(out, codes, loss_mask, *, dep_q, **k):
    """Mutant: ``duplex_ce_loss`` with the cap on all 16 codebooks (the user's too), PersonaPlex's weights."""
    return _ns["duplex_ce_loss"](out, codes, loss_mask, dep_q=dep_q, other_audio_weight=0.02, **k)


_, ga_m2, _ = grads(loss_fn=all16_loss)
ck(bool((ga_m2[8:][:, cap_f] == 0).all()), "a cap on all 16 codebooks is caught (user grad vanishes)")

# ---------------------------------------------------------------------------------------------------------------
print("\n5. off == before (pre-change duplex_ce_loss from git)", flush=True)
src_old = subprocess.run(
    ["git", "-C", LIB, "show", f"{PRE_CHANGE}:./moshi_family/train_data_common.py"],
    capture_output=True,
    text=True,
    check=True,
).stdout
tmpd = tempfile.mkdtemp(prefix="turncap_")
old_path = os.path.join(tmpd, "tdc_old.py")
open(old_path, "w").write(src_old)
spec = importlib.util.spec_from_file_location("tdc_old", old_path)
tdc_old = importlib.util.module_from_spec(spec)
sys.modules["tdc_old"] = tdc_old  # dataclasses look their module up
spec.loader.exec_module(tdc_old)
ck("asst_frame_weight" not in src_old, "the reference predates the knob")
g = torch.Generator().manual_seed(1)
o = Out()
o.text_logits = torch.randn(1, 1, T, 32000, generator=g)
o.logits = torch.randn(1, 16, T, 2048, generator=g)
o.text_mask = torch.ones(1, 1, T, dtype=torch.bool)
o.mask = torch.ones(1, 16, T, dtype=torch.bool)
lkw = dict(dep_q=16, weighted_mean=True, text_epad_id=TEXT_EPAD_ID, return_components=True, text_epad_weight=1.0)
l_new, c_new = tdc.paper_weight_loss(o, codes[None], mask[None], **lkw)
l_old, c_old = tdc_old.paper_weight_loss(o, codes[None], mask[None], **lkw)
ck(torch.equal(l_new, l_old) and c_new == c_old, "cap off: loss + every component bit-identical to the pre-change code")
l_one, _ = tdc.paper_weight_loss(o, codes[None], mask[None], asst_frame_weight=torch.ones(1, T), **lkw)
ck(torch.equal(l_one, l_old), "all-ones assistant weight: bit-identical")
pf = {}
l_pf, _ = tdc.paper_weight_loss(o, codes[None], mask[None], per_frame=pf, **lkw)
ck(
    torch.equal(l_pf, l_old) and set(pf) == {"text_ce", "text_valid", "text_is_pad", "audio_ce", "audio_valid"},
    "per_frame: loss unchanged",
)
lm = StubLM(T)
l_ls, _ = loss_step_for(None, model=lm)((codes[None], mask[None]))
ck(torch.isfinite(l_ls).item(), "the launcher's loss_step with the cap off runs on a 2-tensor batch")

# ---------------------------------------------------------------------------------------------------------------
print("\n6. the held-out loss stays unmasked", flush=True)
base_cfg = {"loss_norm": "weight", "emit_epad": True, "text_epad_weight": 1.0, "onset_floor": True, "duration_sec": 60}
ck(HL.loss_settings({**base_cfg, "asst_turn_cap_s": 10.0}) == HL.loss_settings(base_cfg), "same settings as x1's")
ck("asst_frame_weight" not in inspect.getsource(HL.make_loss_step), "make_loss_step passes no assistant weight")

print()
if fails:
    print(f"FAIL ({len(fails)}):", *fails, sep="\n  ")
    sys.exit(1)
print("ALL OK")
