"""Guard: the opt-in training-speed switches (``moshi_family.train_perf``, 2026-09-23) change kernels, not maths.

WHY. Each switch replaces a computation by a faster one that must be the SAME function. A wrong
"faster" version still trains -- the loss falls, nothing crashes -- so only a direct comparison can
tell. The switches, all OFF by default:

  contiguous_step_inputs   depformer per-step linears get a contiguous copy of their input slice
  fused_causal_attention   SDPA ``is_causal=True`` where the bool mask is exactly lower-triangular
  fused_lora_add           ``W x + s * B(A x)`` as one addmm
  depformer_skip_masked    the depformer runs only on rows the loss can see (``forward_depformer_rows``)
  length_bucket_pool       loader groups rows of similar length (changes batch COMPOSITION, not items)

Drives the real code: a tiny PersonaPlex ``LMModel`` (real class, real delays, dep_q=16, the
depformer's 16 weights-per-step), LoRA via ``apply_lora_to`` with a NON-zero lora_B (else the LoRA
path is invisible), two collated rows of different length with a loss-masked prompt prefix and right padding,
the real ``personaplex_loss`` (weighted mean, EPAD) and ``backward``. float32 on CPU, so "same maths"
means agreement to ~1e-6.

  [1] defaults are off and ``apply_train_perf()`` with no switch sets nothing
  [2] each switch alone and all together: loss and EVERY LoRA gradient == baseline (rel < 1e-5),
      and the switch really took its fast path (spies: contiguous inputs; is_causal without a mask on
      the temporal layers only, the windowed mask kept once T > context; no lora_B module call; fewer
      depformer rows)
  [3] the comparison can fail: four one-line mutations of the real sources, each a plausible bug,
      are rejected (fused causal ignoring the context window at T > context; row skip ignoring the
      codebook delays; LoRA add without the scaling; the wrong step slice)
  [4] length bucketing: same items as the plain loader over an epoch (multiset), batches are runs of
      the length-sorted pool, less padding; bucket_pool=0 is the plain loader; skip_batches refused
  [5] every launcher reads its keys, calls apply_train_perf only under a switch, takes its fast forward
      only under ``depformer_skip_masked`` and hands the model FLOPs to its training loop (AST)

Every OTHER family that runs on these modules (2026-09-23, POLICY_2026_09_23 turns the switches on
for all of them), each with its real model class and real loss:

  [6] base Moshi LoRA (``models/lm.py``, moshiko layout: dep_q=8, weights per step, delays 0/1,
      ``moshi_loss``): each switch alone and together == baseline, fast path taken (same spies)
  [7] base Moshi FULL finetune (no LoRA, every parameter trains): contiguous / fused causal / row skip
  [8] MoshiRAG LoRA (``moshirag/models/lm.py`` with a ``ref_schedule``, LoRA on the trunk only)
  [9] Audex-style (an LMModel whose trunk is a non-Moshi decoder behind ``_AudexTrunkAdapter``; stage1
      LoRA on it + trainable depformer): contiguous / fused LoRA add / row skip; fused causal refused
  [10] RL: ``text_logits_only`` == ``forward().text_logits`` bit for bit; ``text_token_logprobs`` with
       ``TEXT_ONLY_FORWARD`` == without (log-probs + LoRA grads), and the depformer never runs;
       one mutation (off-by-one input shift) rejected
  [11] throughput scalars: ``model_flops_per_frame``'s per-frame Linear count == a forward-hook count of
       the MACs actually executed, for 5 layouts (per-step depformer at dep_q 8 and 16, shared-weight
       depformer, the HF-style trunk with an uncalled lm_head, RL's text path); a mutation that drops the depformer
       multiplicity is rejected; ``batch_real_frames``, ``gpu_peak_flops``; the real ``run_training``
       logs ``real_frames_per_step`` / ``real_frames_per_s`` / ``mfu`` with the right values

Run from the setup root, CPU, ~40 s:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_train_perf.py
"""

import ast
import inspect
import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from moshi_family import train_perf as tp_mod  # noqa: E402
from moshi_family.build import LoraConfig, apply_lora_to  # noqa: E402
from moshi_family.models import lm as base_lm  # noqa: E402
from moshi_family.models import loaders as base_loaders  # noqa: E402
from moshi_family.moshi_train_data import moshi_loss  # noqa: E402
from moshi_family.modules import lora as lora_mod  # noqa: E402
from moshi_family.modules import transformer as tr  # noqa: E402
from moshi_family.personaplex.models import lm as ppx_lm  # noqa: E402
from moshi_family.personaplex.models import loaders  # noqa: E402
from moshi_family.personaplex.train_data import personaplex_loss  # noqa: E402
from moshi_family.train_data_common import TEXT_EPAD_ID, batched_row_loader, collate_codes  # noqa: E402
from moshi_family.train_loop import is_lora_param  # noqa: E402
from moshi_family.train_perf import apply_train_perf  # noqa: E402

fails = []


def ck(cond, label, extra=""):
    print(f"  [{'ok' if cond else 'FAIL'}] {label}{'  ' + extra if extra else ''}")
    if not cond:
        fails.append(label)


def reset_flags():
    tr.CONTIGUOUS_STEP_INPUTS = False
    for m in MODELS:
        for mod in m.modules():
            if isinstance(mod, tr.StreamingMultiheadAttention):
                mod.__dict__.pop("fused_causal_training", None)
            if isinstance(mod, lora_mod.LoRALinear):
                mod.__dict__.pop("fused_add", None)


# ---------------------------------------------------------------------------------------------
# tiny real PersonaPlex LM + LoRA, and a batch shaped like training's
# ---------------------------------------------------------------------------------------------
torch.manual_seed(0)
kw = dict(loaders._lm_kwargs)
kw.update(
    dim=64,
    num_heads=4,
    num_layers=2,
    hidden_scale=2.0,
    text_card=100,
    context=64,
    depformer_dim=32,
    depformer_dim_feedforward=64,
    depformer_num_heads=2,
    depformer_num_layers=2,
    dep_q=16,
)
assert kw["depformer_weights_per_step"], "fixture must keep the real depformer"
base = ppx_lm.LMModel(device="cpu", dtype=torch.float32, **kw)
apply_lora_to(base, LoraConfig(rank=4, scaling=2.0), dtype=torch.float32)
with torch.no_grad():
    for n, p in base.named_parameters():
        if "lora_B" in n:
            p.normal_(0, 0.05)
for n, p in base.named_parameters():
    p.requires_grad = is_lora_param(n)
base.train()
MODELS = [base]


def make_rows(lengths, prefix, seed):
    g = torch.Generator().manual_seed(seed)
    rows = []
    for T in lengths:
        codes = torch.randint(0, 50, (17, T), generator=g)
        codes[0] = torch.randint(0, 100, (T,), generator=g)
        codes[0, torch.rand(T, generator=g) < 0.4] = 3  # padding-heavy text stream, like the corpus
        mask = torch.ones(T, dtype=torch.bool)
        mask[:prefix] = False  # the system prompt is loss-masked
        rows.append((codes, mask))
    return collate_codes(rows)  # right-pads the shorter row, mask 0 there


BATCH = make_rows([40, 27], prefix=9, seed=1)  # T=40 <= context 64: every attention is plain causal
BATCH_LONG = make_rows([70, 55], prefix=5, seed=2)  # T=70 > context 64: must keep the windowed mask


def run(model, batch, *, skip=False):
    model.zero_grad(set_to_none=True)
    codes, mask = batch
    out = model.forward_train(codes, depformer_frame_mask=mask) if skip else model.forward_train(codes)
    loss, _ = personaplex_loss(
        out, codes, mask, dep_q=model.dep_q, weighted_mean=True, text_epad_id=TEXT_EPAD_ID, return_components=True
    )
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return float(loss.detach()), grads


def diff(a, b):
    la, ga = a
    lb, gb = b
    assert set(ga) == set(gb), "different gradient sets"
    rel = max(float((ga[k] - gb[k]).norm() / ga[k].norm().clamp(min=1e-30)) for k in ga)
    return abs(la - lb) / max(abs(la), 1e-30), rel, len(ga)


REF = run(base, BATCH)
REF_LONG = run(base, BATCH_LONG)
n_lora = sum(1 for n, _ in base.named_parameters() if is_lora_param(n))
ck(len(REF[1]) == n_lora, f"baseline: every one of the {n_lora} LoRA tensors gets a gradient")
TOL = 1e-5

# ---------------------------------------------------------------------------------------------
print("[1] defaults are off")
ck(tr.CONTIGUOUS_STEP_INPUTS is False, "transformer.CONTIGUOUS_STEP_INPUTS defaults to False")
ck(tr.StreamingMultiheadAttention.fused_causal_training is False, "fused_causal_training defaults to False")
ck(lora_mod.LoRALinear.fused_add is False, "LoRALinear.fused_add defaults to False")
ck(
    inspect.signature(base.forward_train).parameters["depformer_frame_mask"].default is None,
    "forward_train: no row skip by default",
)
done = apply_train_perf(base)
ck(
    done == {"contiguous_step_inputs": False, "fused_causal_attention": 0, "fused_lora_add": 0},
    "apply_train_perf() with no switch sets nothing",
    str(done),
)
ck(
    diff(REF, run(base, BATCH))[:2] == (0.0, 0.0),
    "baseline is reproducible bit for bit (so any diff below is the switch)",
)


# ---------------------------------------------------------------------------------------------
# spies: did the switch take its fast path?
# ---------------------------------------------------------------------------------------------
class Spy:
    def __init__(self, model):
        self.hooks, self.contig, self.lora_b_calls, self.dep_rows = [], [], 0, []
        dep_in = model.depformer.layers[0].self_attn.in_projs[3]
        self.hooks.append(dep_in.register_forward_pre_hook(lambda m, a: self.contig.append(a[0].is_contiguous())))
        for m in model.modules():
            if isinstance(m, lora_mod.LoRALinear):
                self.hooks.append(m.lora_B.register_forward_pre_hook(lambda m, a: self._b()))
        self.hooks.append(model.depformer.register_forward_pre_hook(lambda m, a: self.dep_rows.append(a[0].shape[0])))
        self.sdpa = []
        self._orig = tr.F.scaled_dot_product_attention

        def sdpa(q, k, v, attn_mask=None, **kw):
            self.sdpa.append((q.shape[-2], bool(kw.get("is_causal", False)), attn_mask is not None))
            return self._orig(q, k, v, attn_mask, **kw)

        class _F:  # stands in for torch.nn.functional inside transformer.py only; torch itself untouched
            scaled_dot_product_attention = staticmethod(sdpa)

            def __getattr__(self, name):
                return getattr(torch.nn.functional, name)

        tr.F = _F()

    def _b(self):
        self.lora_b_calls += 1

    def close(self):
        for h in self.hooks:
            h.remove()
        tr.F = torch.nn.functional


def with_switches(batch, *, skip=False, **sw):
    reset_flags()
    apply_train_perf(base, **sw)
    spy = Spy(base)
    try:
        res = run(base, batch, skip=skip)
    finally:
        spy.close()
        reset_flags()
    return res, spy


print("[2] each switch == baseline, and it took its fast path")
T, B = BATCH[0].shape[2], BATCH[0].shape[0]

res, spy = with_switches(BATCH, contiguous_step_inputs=True)
dl, dg, n = diff(REF, res)
ck(
    dl < TOL and dg < TOL,
    "contiguous_step_inputs: loss and all LoRA grads == baseline",
    f"dloss {dl:.1e}, max rel grad {dg:.1e}, n={n}",
)
ck(spy.contig and all(spy.contig), "  ...its per-step linears saw contiguous inputs")
_, spy0 = with_switches(BATCH)
ck(spy0.contig and not any(spy0.contig), "  ...and without the switch they did not (the spy can tell)")

res, spy = with_switches(BATCH, fused_causal_attention=True)
dl, dg, n = diff(REF, res)
ck(
    dl < TOL and dg < TOL,
    "fused_causal_attention (T <= context): == baseline",
    f"dloss {dl:.1e}, max rel grad {dg:.1e}",
)
temporal = [c for c in spy.sdpa if c[0] == T]
dep = [c for c in spy.sdpa if c[0] == 16]
ck(
    temporal and all(c[1] and not c[2] for c in temporal),
    f"  ...temporal layers called SDPA is_causal=True without a mask ({len(temporal)} calls)",
)
ck(
    dep and all(c[2] and not c[1] for c in dep),
    f"  ...the depformer kept its bool mask (flash is slower on its 16-step rows) ({len(dep)} calls)",
)
ck(all(c[2] and not c[1] for c in spy0.sdpa), "  ...and without the switch every call passed a bool mask")
res, spy = with_switches(BATCH_LONG, fused_causal_attention=True)
dl, dg, _ = diff(REF_LONG, res)
ck(
    dl < TOL and dg < TOL,
    "fused_causal_attention, T=70 > context 64: == baseline",
    f"dloss {dl:.1e}, max rel grad {dg:.1e}",
)
long_t = [c for c in spy.sdpa if c[0] == 70]
ck(
    long_t and all(c[2] and not c[1] for c in long_t),
    f"  ...the temporal layers kept the windowed mask there ({len(long_t)} calls)",
)

res, spy = with_switches(BATCH, fused_lora_add=True)
dl, dg, n = diff(REF, res)
ck(dl < TOL and dg < TOL, "fused_lora_add: == baseline", f"dloss {dl:.1e}, max rel grad {dg:.1e}")
ck(
    spy.lora_b_calls == 0 and spy0.lora_b_calls > 0,
    f"  ...lora_B as a module never ran (baseline: {spy0.lora_b_calls} calls)",
)

res, spy = with_switches(BATCH, skip=True)
dl, dg, n = diff(REF, res)
ck(dl < TOL and dg < TOL, "depformer_skip_masked: == baseline", f"dloss {dl:.1e}, max rel grad {dg:.1e}")
n_need = int((BATCH[1] | torch.nn.functional.pad(BATCH[1], (1, 0))[:, :-1]).sum())
ck(
    spy.dep_rows == [n_need] and spy0.dep_rows == [B * T],
    f"  ...depformer rows {spy.dep_rows} (loss-visible + one delay) vs {spy0.dep_rows} full",
)

res, spy = with_switches(
    BATCH, skip=True, contiguous_step_inputs=True, fused_causal_attention=True, fused_lora_add=True
)
dl, dg, n = diff(REF, res)
ck(dl < TOL and dg < TOL, "all switches together: == baseline", f"dloss {dl:.1e}, max rel grad {dg:.1e}, n={n}")


# ---------------------------------------------------------------------------------------------
print("[3] the comparison rejects plausible bugs (mutated real sources)")


def mutated(obj, name, old, new):
    """Re-define function ``name`` of ``obj`` from its real source with one line changed."""
    fn = getattr(obj, name)
    src = textwrap.dedent(inspect.getsource(fn))
    assert old in src, f"mutation anchor not found in {name}: {old!r}"
    # Compile into the REAL module globals (so the mutant reads the live switch flags), then put back
    # whatever that name was bound to there -- the caller installs the mutant only for one call.
    g = vars(sys.modules[fn.__module__])
    missing = object()
    prev = g.get(name, missing)
    exec(compile(src.replace(old, new), f"<mutated {name}>", "exec"), g)
    mutant = g[name]
    if prev is missing:
        del g[name]
    else:
        g[name] = prev
    return mutant


def rejected(label, patches, batch=BATCH, ref=REF, skip=False, **sw):
    saved = [(o, n, getattr(o, n)) for o, n, _ in patches]
    try:
        for o, n, f in patches:
            setattr(o, n, f)
        res, _ = with_switches(batch, skip=skip, **sw)
    finally:
        for o, n, f in saved:
            setattr(o, n, f)
    dl, dg, _ = diff(ref, res)
    ck(dl > 1e-4 or dg > 1e-3, f"rejected: {label}", f"dloss {dl:.1e}, max rel grad {dg:.1e}")


MHA = tr.StreamingMultiheadAttention
rejected(
    "fused causal that ignores the context window (T=70 > context 64)",
    [(MHA, "forward", mutated(MHA, "forward", "(self.context is None or T <= self.context)", "True"))],
    batch=BATCH_LONG,
    ref=REF_LONG,
    fused_causal_attention=True,
)
# The row skip is the SHARED train_perf.depformer_rows (PersonaPlex's forward_depformer_rows and base
# Moshi's forward both look it up at call time), so the mutant is installed there.
rejected(
    "row skip that ignores the codebook delays (keeps loss rows only)",
    [(tp_mod, "depformer_rows", mutated(tp_mod, "depformer_rows", "need[:, d:] |= fm[:, :-d]", "pass"))],
    skip=True,
)
LL = lora_mod.LoRALinear
rejected(
    "fused LoRA add without the scaling",
    [(LL, "forward", mutated(LL, "forward", "alpha=self.scaling", "alpha=1.0"))],
    fused_lora_add=True,
)
rejected(
    "contiguous copy of the wrong step",
    [
        (
            tr,
            "apply_weights_per_step",
            mutated(tr, "apply_weights_per_step", "x_t = steps[t] if", "x_t = steps[0] if"),
        )
    ],
    contiguous_step_inputs=True,
)


# ---------------------------------------------------------------------------------------------
print("[4] length bucketing")


class FakeTable:
    num_rows = 37
    column_names = ["id"]


LENS = np.random.default_rng(3).integers(20, 90, size=FakeTable.num_rows)


def build_item(i, rng):
    T = int(LENS[i]) + int(rng.integers(0, 3))  # a per-item draw, like the random window position
    codes = torch.full((17, T), i, dtype=torch.long)
    codes[1, 0] = int(rng.integers(0, 1000))
    return codes, torch.ones(T, dtype=torch.bool)


def epoch(**k):
    return list(batched_row_loader(FakeTable(), build_item, batch_size=4, seed=5, infinite=False, **k))


def items(batches):
    out = []
    for codes, mask in batches:
        for b in range(codes.shape[0]):
            n = int(mask[b].sum())
            out.append((int(codes[b, 2, 0]), n, int(codes[b, 1, 0])))
    return sorted(out)


def pad_share(batches):
    comp = sum(c.shape[0] * c.shape[2] for c, _ in batches)
    return 1 - sum(int(m.sum()) for _, m in batches) / comp


plain, plain_again, zero, buck = epoch(), epoch(), epoch(bucket_pool=0), epoch(bucket_pool=4)
ck(
    [(c.tolist(), m.tolist()) for c, m in plain] == [(c.tolist(), m.tolist()) for c, m in zero],
    "bucket_pool=0 == the plain loader, batch for batch",
)
ck(
    items(buck) == items(plain) and len(items(plain)) == FakeTable.num_rows,
    "bucketed epoch == the plain epoch's items exactly (row, length, per-item draw)",
)
ck([m.shape for _, m in buck] != [m.shape for _, m in plain], "  ...but grouped differently (bucketing is not a no-op)")
# within each pool of 16 items the batches are consecutive runs of the sorted lengths
pools_ok = True
for i in range(0, len(buck), 4):
    grp = buck[i : i + 4]
    lens = sorted(int(m[b].sum()) for _, m in grp for b in range(m.shape[0]))
    for _, m in grp:
        own = sorted(int(m[b].sum()) for b in range(m.shape[0]))
        j = lens.index(own[0])
        pools_ok &= lens[j : j + len(own)] == own
ck(pools_ok, "each bucketed batch is a run of its pool's length-sorted items")
ck(pad_share(buck) < pad_share(plain), f"padding {pad_share(plain):.1%} -> {pad_share(buck):.1%}")
try:
    next(batched_row_loader(FakeTable(), build_item, batch_size=4, seed=5, skip_batches=2, bucket_pool=4))
    ck(False, "bucket_pool + skip_batches is refused")
except ValueError:
    ck(True, "bucket_pool + skip_batches is refused")


# ---------------------------------------------------------------------------------------------
print("[5] launcher wiring (every launcher on these modules)")
LIBDIR = Path(inspect.getsourcefile(sys.modules["moshi_family.train_perf"])).parent
ALL_BOOL = {"contiguous_step_inputs", "fused_causal_attention", "fused_lora_add", "depformer_skip_masked"}
#: launcher -> (keys it must read, the fast-forward marker that must sit under `if depformer_skip_masked`,
#: the call that must carry model_flops_per_frame=)
LAUNCHERS = {
    "personaplex/finetune_launcher.py": (
        ALL_BOOL | {"length_bucket_pool", "prefetch_batches"},
        "depformer_frame_mask=loss_mask",
        "TrainConfig",
    ),
    "moshi_finetune_launcher.py": (ALL_BOOL | {"prefetch_batches"}, "depformer_frame_mask=loss_mask", "TrainConfig"),
    "moshirag_finetune_launcher.py": (
        {"contiguous_step_inputs", "fused_causal_attention", "fused_lora_add"},
        None,
        "TrainConfig",
    ),
    "audex/finetune_launcher.py": (
        {"contiguous_step_inputs", "fused_lora_add", "depformer_skip_masked", "prefetch_batches"},
        "depformer_frame_mask=loss_mask",
        "TrainConfig",
    ),
    "rl/launcher.py": (ALL_BOOL, "TEXT_ONLY_FORWARD = True", "RLTrainerConfig"),
}


def _cfg_keys(fn):
    return {
        n.args[0].value
        for n in ast.walk(fn)
        if isinstance(n, ast.Call)
        and getattr(n.func, "attr", "") == "get"
        and getattr(n.func.value, "id", "") == "cfg"
        and n.args
        and isinstance(n.args[0], ast.Constant)
    }


for rel, (want, marker, cfg_call) in LAUNCHERS.items():
    src = (LIBDIR / rel).read_text()
    main = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.FunctionDef) and n.name == "main")
    keys = _cfg_keys(main)
    ck(want <= keys, f"{rel}: reads {sorted(want)}", str(sorted(want - keys)))
    calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "apply_train_perf"]
    ifs = [n for n in ast.walk(main) if isinstance(n, ast.If)]
    ck(
        len(calls) == 1 and any(calls[0] in list(ast.walk(i)) for i in ifs),
        f"{rel}: apply_train_perf once, only under an explicit switch",
    )
    if marker is not None:
        under = [i for i in ifs if "depformer_skip_masked" in ast.unparse(i.test)]
        body_src = "\n".join(ast.unparse(b) for i in under for b in i.body)
        ck(
            marker in body_src and src.count(marker) == body_src.count(marker),
            f"{rel}: the fast forward ({marker!r}) only under `if depformer_skip_masked`",
        )
    kw_ok = any(
        isinstance(n, ast.Call)
        and getattr(n.func, "id", "") == cfg_call
        and any(k.arg == "model_flops_per_frame" for k in n.keywords)
        for n in ast.walk(main)
    )
    ck(kw_ok, f"{rel}: {cfg_call}(model_flops_per_frame=...) -> the logged MFU")
ck(
    "forward_train(codes)" in (LIBDIR / "personaplex/finetune_launcher.py").read_text(),
    "PersonaPlex loss_step keeps the plain forward_train(codes) call for runs without the row skip",
)


# ---------------------------------------------------------------------------------------------
# generic harness for the other families: real model, real forward, real loss, backward
# ---------------------------------------------------------------------------------------------
def run_family(model, batch, fwd, loss_fn, *, skip=False):
    model.zero_grad(set_to_none=True)
    out = fwd(model, batch, skip)
    loss = loss_fn(out, batch)
    loss = loss[0] if isinstance(loss, tuple) else loss
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return float(loss.detach()), grads


def family_switches(model, batch, fwd, loss_fn, *, skip=False, **sw):
    reset_flags()
    apply_train_perf(model, **sw)
    spy = Spy(model)
    try:
        res = run_family(model, batch, fwd, loss_fn, skip=skip)
    finally:
        spy.close()
        reset_flags()
    return res, spy


def need_rows(mask, delays):
    need = torch.zeros_like(mask)
    for d in sorted(set(delays)):
        need |= mask if d == 0 else torch.nn.functional.pad(mask, (d, 0))[:, :-d]
    return int(need.sum())


def check_family(label, model, batch, fwd, loss_fn, *, switches, dep_len, row_skip):
    """Each switch alone and all together == baseline (loss + every gradient), and it took its fast path."""
    ref = run_family(model, batch, fwd, loss_fn)
    _, spy0 = family_switches(model, batch, fwd, loss_fn)
    T, B = batch[0].shape[2], batch[0].shape[0]
    ck(len(ref[1]) > 0, f"{label}: baseline has {len(ref[1])} gradient tensors")
    for sw in switches:
        skip = sw == "depformer_skip_masked"
        res, spy = family_switches(model, batch, fwd, loss_fn, skip=skip, **({} if skip else {sw: True}))
        dl, dg, n = diff(ref, res)
        ck(dl < TOL and dg < TOL, f"{label}: {sw} == baseline", f"dloss {dl:.1e}, max rel grad {dg:.1e}, n={n}")
        if sw == "contiguous_step_inputs":
            ck(spy.contig and all(spy.contig) and not any(spy0.contig), "  ...contiguous per-step inputs (not without)")
        elif sw == "fused_causal_attention":
            temporal = [c for c in spy.sdpa if c[0] == T]
            dep = [c for c in spy.sdpa if c[0] == dep_len]
            ck(
                temporal and all(c[1] and not c[2] for c in temporal) and dep and all(c[2] and not c[1] for c in dep),
                f"  ...temporal is_causal without a mask ({len(temporal)}), depformer masked ({len(dep)})",
            )
        elif sw == "fused_lora_add":
            ck(
                spy.lora_b_calls == 0 and spy0.lora_b_calls > 0,
                f"  ...no lora_B module call (base {spy0.lora_b_calls})",
            )
        elif skip:
            n_need = need_rows(batch[1], model.delays[model.audio_offset : model.audio_offset + model.dep_q])
            ck(
                spy.dep_rows == [n_need] and spy0.dep_rows == [B * T] and n_need < B * T,
                f"  ...depformer rows {spy.dep_rows} vs {spy0.dep_rows} full",
            )
    sw_all = {k: True for k in switches if k != "depformer_skip_masked"}
    res, _ = family_switches(model, batch, fwd, loss_fn, skip=row_skip, **sw_all)
    dl, dg, n = diff(ref, res)
    ck(dl < TOL and dg < TOL, f"{label}: all switches together == baseline", f"dloss {dl:.1e}, max rel grad {dg:.1e}")
    return ref


def tiny_kwargs(src, **over):
    kw = {k: v for k, v in dict(src).items() if k not in ("conditioners", "fuser")}
    kw.update(
        dim=64,
        num_heads=4,
        num_layers=2,
        hidden_scale=2.0,
        text_card=100,
        context=64,
        depformer_dim=32,
        depformer_dim_feedforward=64,
        depformer_num_heads=2,
        depformer_num_layers=2,
    )
    kw.update(over)
    return kw


def with_lora(model, target=None):
    if target is None:
        apply_lora_to(model, LoraConfig(rank=4, scaling=2.0), dtype=torch.float32)
    else:
        lora_mod.replace_all_linear_with_lora(target, 4, 2.0, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    for n, p in model.named_parameters():
        p.requires_grad = is_lora_param(n)
    model.train()
    MODELS.append(model)
    return model


def moshi_fwd(m, b, skip):
    return m(b[0], depformer_frame_mask=b[1]) if skip else m(b[0])


def moshi_lossfn(out, b):
    return moshi_loss(
        out,
        b[0],
        b[1],
        dep_q=8,
        audio_other_weight=0.02,
        text_pad_weight=0.3,
        text_epad_id=TEXT_EPAD_ID,
        weighted_mean=True,
        return_components=True,
    )


BATCH_M = make_rows([40, 27], prefix=0, seed=3)  # base Moshi: no prompt, the shorter row right-padded
SW_ALL = ["contiguous_step_inputs", "fused_causal_attention", "fused_lora_add", "depformer_skip_masked"]

# ---------------------------------------------------------------------------------------------
print("[6] base Moshi LoRA (models/lm.py, moshiko layout)")
torch.manual_seed(1)
KW_M = tiny_kwargs(base_loaders._lm_kwargs)
assert KW_M["dep_q"] == 8 and KW_M["depformer_weights_per_step"], "fixture must keep moshiko's depformer"
moshi = with_lora(base_lm.LMModel(device="cpu", dtype=torch.float32, **KW_M))
ck(
    inspect.signature(moshi.forward).parameters["depformer_frame_mask"].default is None,
    "base forward: no row skip by default",
)
REF_M = check_family("moshi LoRA", moshi, BATCH_M, moshi_fwd, moshi_lossfn, switches=SW_ALL, dep_len=8, row_skip=True)
_res, _ = family_switches(moshi, BATCH_M, moshi_fwd, moshi_lossfn)
ck(diff(REF_M, _res)[:2] == (0.0, 0.0), "moshi LoRA: baseline reproducible bit for bit")
_saved = tp_mod.depformer_rows
try:
    tp_mod.depformer_rows = mutated(tp_mod, "depformer_rows", "need[:, d:] |= fm[:, :-d]", "pass")
    _res, _ = family_switches(moshi, BATCH_M, moshi_fwd, moshi_lossfn, skip=True)
finally:
    tp_mod.depformer_rows = _saved
_dl, _dg, _ = diff(REF_M, _res)
ck(_dl > 1e-4 or _dg > 1e-3, "rejected: base-Moshi row skip ignoring the delays", f"dloss {_dl:.1e}, grad {_dg:.1e}")

# ---------------------------------------------------------------------------------------------
print("[7] base Moshi full finetune (no LoRA, every parameter trains)")
torch.manual_seed(2)
moshi_full = base_lm.LMModel(device="cpu", dtype=torch.float32, **KW_M)
for p in moshi_full.parameters():
    p.requires_grad = True
moshi_full.train()
MODELS.append(moshi_full)
check_family(
    "moshi full",
    moshi_full,
    BATCH_M,
    moshi_fwd,
    moshi_lossfn,
    switches=["contiguous_step_inputs", "fused_causal_attention", "depformer_skip_masked"],
    dep_len=8,
    row_skip=True,
)
try:
    apply_train_perf(moshi_full, fused_lora_add=True)
    ck(False, "moshi full: fused_lora_add refused (no LoRALinear)")
except AssertionError:
    ck(True, "moshi full: fused_lora_add refused (no LoRALinear) -- the preset leaves it out")
reset_flags()

# ---------------------------------------------------------------------------------------------
print("[8] MoshiRAG LoRA (moshirag/models/lm.py + ref_schedule, LoRA on the trunk)")
from moshi_family.moshirag.models import lm as rag_lm  # noqa: E402
from moshi_family.moshirag.models import loaders as rag_loaders  # noqa: E402
from moshi_family.moshirag_train_data import moshirag_loss  # noqa: E402

torch.manual_seed(3)
rag = rag_lm.LMModel(device="cpu", dtype=torch.float32, **tiny_kwargs(rag_loaders._lm_kwargs))
with_lora(rag, target=rag.transformer)
_g = torch.Generator().manual_seed(4)
BATCH_R = (*make_rows([40, 27], prefix=0, seed=5), 0.1 * torch.randn(2, 40, 64, generator=_g))


def rag_fwd(m, b, skip):
    assert not skip
    return m(b[0], condition_tensors=None, ref_schedule=b[2])


check_family(
    "moshirag",
    rag,
    BATCH_R,
    rag_fwd,
    lambda out, b: moshirag_loss(out, b[0], b[1], dep_q=8),
    switches=["contiguous_step_inputs", "fused_causal_attention", "fused_lora_add"],
    dep_len=8,
    row_skip=False,
)

# ---------------------------------------------------------------------------------------------
print("[9] Audex-style: LMModel with a non-Moshi trunk (stage1: trunk LoRA + trainable depformer)")
from moshi_family.audex.models.lm import _AudexTrunkAdapter  # noqa: E402


class _FakeAudexBackbone(torch.nn.Module):
    """Stands in for AudexBackbone (HF decoder): ``forward(x, cache=None)`` -> hidden, a causal
    token mix + MLP per layer under ``model.model.layers``, and an ``lm_head`` the trunk never calls."""

    def __init__(self, dim, n_layers=2):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.model = torch.nn.Module()
        self.model.model.layers = torch.nn.ModuleList(
            torch.nn.Sequential(torch.nn.Linear(dim, 2 * dim), torch.nn.GELU(), torch.nn.Linear(2 * dim, dim))
            for _ in range(n_layers)
        )
        self.model.lm_head = torch.nn.Linear(dim, 100, bias=False)

    def forward(self, x, cache=None):
        h = x
        for layer in self.model.model.layers:
            h = h + layer(torch.cumsum(h, dim=1) / torch.arange(1, h.shape[1] + 1, dtype=h.dtype)[None, :, None])
        return h


torch.manual_seed(5)
aud = base_lm.LMModel(device="cpu", dtype=torch.float32, **KW_M)
aud.transformer = _AudexTrunkAdapter(_FakeAudexBackbone(64))
aud.out_norm = None
lora_mod.replace_all_linear_with_lora(aud.transformer.backbone.model.model.layers, 4, 2.0, dtype=torch.float32)
with torch.no_grad():
    for n, p in aud.named_parameters():
        if "lora_B" in n:
            p.normal_(0, 0.05)
for n, p in aud.named_parameters():
    # stage1: base trunk + text readout frozen; trunk LoRA + audio path (embeddings, depformer) train
    p.requires_grad = is_lora_param(n) or not (n.startswith("transformer.") or n.startswith("text_"))
aud.train()
MODELS.append(aud)
check_family(
    "audex-style",
    aud,
    BATCH_M,
    moshi_fwd,
    lambda out, b: moshi_loss(out, b[0], b[1], dep_q=8, return_components=True),
    switches=["contiguous_step_inputs", "fused_lora_add", "depformer_skip_masked"],
    dep_len=8,
    row_skip=True,
)
try:
    apply_train_perf(aud, fused_causal_attention=True)
    ck(False, "audex-style: fused_causal_attention refused (the trunk has no StreamingMultiheadAttention)")
except AssertionError:
    ck(True, "audex-style: fused_causal_attention refused (HF trunk) -- the launcher does not offer it")
reset_flags()

# ---------------------------------------------------------------------------------------------
print("[10] RL: text-only log-prob forward")
from moshi_family.rl import generator as rl_gen  # noqa: E402

_g = torch.Generator().manual_seed(6)
RL_CODES = torch.randint(0, 50, (3, 17, 30), generator=_g)
RL_CODES[:, 0] = torch.randint(0, 100, (3, 30), generator=_g)
RL_CODES[:, 0][torch.rand(3, 30, generator=_g) < 0.4] = 3
reset_flags()
with torch.no_grad():
    _full = moshi.forward(RL_CODES)
    _tl, _tm = tp_mod.text_logits_only(moshi, RL_CODES)
ck(
    torch.equal(torch.nan_to_num(_full.text_logits, nan=-7.0), torch.nan_to_num(_tl, nan=-7.0))
    and torch.equal(_full.text_mask, _tm),
    "text_logits_only == forward().text_logits and .text_mask, bit for bit",
)


def rl_step(text_only, **sw):
    reset_flags()
    apply_train_perf(moshi, **sw)
    rl_gen.TEXT_ONLY_FORWARD = text_only
    spy = Spy(moshi)
    try:
        moshi.zero_grad(set_to_none=True)
        logp, mask = rl_gen.text_token_logprobs(moshi, RL_CODES)
        (logp * mask).sum().backward()
        grads = {n: p.grad.detach().clone() for n, p in moshi.named_parameters() if p.grad is not None}
    finally:
        spy.close()
        rl_gen.TEXT_ONLY_FORWARD = False
        reset_flags()
    return (logp.detach(), mask), grads, spy


(_lp0, _m0), _g0, _spy0 = rl_step(False)
(_lp1, _m1), _g1, _spy1 = rl_step(True)
ck(torch.equal(_lp0, _lp1) and torch.equal(_m0, _m1), "text_token_logprobs, TEXT_ONLY_FORWARD: log-probs bit-identical")
ck(
    set(_g0) == set(_g1) and all(torch.equal(_g0[k], _g1[k]) for k in _g0),
    f"  ...and every LoRA gradient bit-identical ({len(_g0)} tensors; depformer LoRA gets none either way)",
)
ck(_spy1.dep_rows == [] and _spy0.dep_rows == [3 * 30], f"  ...the depformer never ran ({_spy0.dep_rows} before)")
(_lp2, _), _g2, _ = rl_step(True, contiguous_step_inputs=True, fused_causal_attention=True, fused_lora_add=True)
_rel = max(float((_g0[k] - _g2[k]).norm() / _g0[k].norm().clamp(min=1e-30)) for k in _g0)
ck(
    float((_lp0 - _lp2).abs().max()) < 1e-5 and _rel < TOL,
    "RL, all switches together == baseline",
    f"max |dlogp| {float((_lp0 - _lp2).abs().max()):.1e}, rel grad {_rel:.1e}",
)
_saved = tp_mod.text_logits_only
try:
    tp_mod.text_logits_only = mutated(tp_mod, "text_logits_only", "delayed_codes[:, :, :-1]", "delayed_codes[:, :, 1:]")
    (_lp3, _), _, _ = rl_step(True)
finally:
    tp_mod.text_logits_only = _saved
ck(float((_lp0 - _lp3).abs().max()) > 1e-3, "rejected: text-only forward with the input shifted by one frame")

# ---------------------------------------------------------------------------------------------
print("[11] throughput scalars: FLOPs per frame, real frames, MFU")


def hook_macs_per_frame(model, run, n_frames):
    """MACs actually executed by every nn.Linear in one forward, per frame (an independent count).

    A plain-Linear ``ActivationGating`` runs its two weights through ``F.linear`` inside a compiled
    kernel, so no Linear hook fires there; it is counted from the gating module's own call instead."""
    from moshi_family.modules.gating import ActivationGating

    tot = [0]

    def hook(m, a, _o):
        tot[0] += (a[0].numel() // a[0].shape[-1]) * m.in_features * m.out_features

    def gating_hook(m, a, _o):
        if isinstance(m.linear_in, torch.nn.Linear):  # the F.linear kernel path
            rows = a[0].numel() // a[0].shape[-1]
            tot[0] += rows * sum(lin.in_features * lin.out_features for lin in (m.linear_in, m.linear_out))

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, torch.nn.Linear)]
    hs += [m.register_forward_hook(gating_hook) for m in model.modules() if isinstance(m, ActivationGating)]
    try:
        with torch.no_grad():
            run()
    finally:
        for h in hs:
            h.remove()
    return tot[0] / n_frames


def analytic_n(model, **kw):
    return sum(lin.in_features * lin.out_features * t for lin, t in tp_mod._linear_uses_per_frame(model, **kw))


reset_flags()
torch.manual_seed(7)
shared = base_lm.LMModel(
    device="cpu", dtype=torch.float32, **{**KW_M, "depformer_weights_per_step": False, "depformer_multi_linear": False}
)
LAYOUTS = {
    "moshi LoRA (per-step depformer)": (moshi, lambda: moshi(BATCH_M[0]), BATCH_M),
    "moshi shared-weight depformer": (shared, lambda: shared(BATCH_M[0]), BATCH_M),
    "PersonaPlex (dep_q 16)": (base, lambda: base.forward_train(BATCH[0]), BATCH),
    "audex-style (HF trunk, uncalled lm_head)": (aud, lambda: aud(BATCH_M[0]), BATCH_M),
}
for label, (m, run, b) in LAYOUTS.items():
    n_frames = b[0].shape[0] * b[0].shape[2]
    got, want = analytic_n(m), hook_macs_per_frame(m, run, n_frames)
    ck(abs(got - want) <= 1e-9 * want, f"{label}: counted Linear MACs/frame == executed", f"{got:.0f} vs {want:.0f}")
_n_rl = hook_macs_per_frame(moshi, lambda: tp_mod.text_logits_only(moshi, RL_CODES), 3 * 30)
ck(abs(analytic_n(moshi, text_only=True) - _n_rl) <= 1e-9 * _n_rl, "RL text path: counted == executed")
_saved = tp_mod._linear_uses_per_frame
try:
    tp_mod._linear_uses_per_frame = mutated(
        tp_mod, "_linear_uses_per_frame", "dep_q // steps if steps is not None else dep_q", "1"
    )
    _bad = analytic_n(shared)
finally:
    tp_mod._linear_uses_per_frame = _saved
_want = hook_macs_per_frame(shared, lambda: shared(BATCH_M[0]), BATCH_M[0].shape[0] * BATCH_M[0].shape[2])
ck(abs(_bad - _want) > 1e-3 * _want, "rejected: a count that drops the depformer's dep_q multiplicity")
_n = analytic_n(moshi)
_ntr = sum(
    lin.in_features * lin.out_features * t
    for lin, t in tp_mod._linear_uses_per_frame(moshi)
    if lin.weight.requires_grad
)
ck(
    tp_mod.model_flops_per_frame(moshi) == 4 * _n + 2 * _ntr and 0 < _ntr < 0.5 * _n,
    "LoRA: FLOPs/frame = 4N + 2N_train (frozen weights get no weight gradient)",
)
ck(tp_mod.model_flops_per_frame(moshi_full) == 6 * analytic_n(moshi_full), "full finetune: FLOPs/frame = 6N")
ck(
    tp_mod.batch_real_frames(BATCH) == 40 + 27 and tp_mod.batch_real_frames((torch.zeros(1, 3, 5),)) is None,
    "batch_real_frames: masked prompt counts, right padding does not; no mask -> None",
)
ck(
    [
        tp_mod.gpu_peak_flops(n)
        for n in ("NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB", "NVIDIA GH200 480GB", "Tesla V100-SXM2-32GB")
    ]
    == [989e12, 312e12, 989e12, None],
    "gpu_peak_flops: H100 989, A100 312, GH200 989 TFLOP/s, unknown None",
)

import json as _json  # noqa: E402
import tempfile  # noqa: E402

from moshi_family.train_loop import TrainConfig, run_training  # noqa: E402


def _loop(peak):
    saved = tp_mod.gpu_peak_flops
    tp_mod.gpu_peak_flops = lambda *a: peak
    try:
        reset_flags()
        params = [p for p in moshi.parameters() if p.requires_grad]
        with tempfile.TemporaryDirectory() as d:

            def batches():
                while True:
                    yield BATCH_M

            run_training(
                optimizer=torch.optim.AdamW(params, lr=1e-5),
                base_lrs=[1e-5],
                trainable_params=params,
                batch_iter=batches(),
                loss_step=lambda b: moshi_lossfn(moshi_fwd(moshi, b, False), b),
                save_fn=lambda step, final: None,
                cfg=TrainConfig(
                    max_steps=3,
                    grad_accum=2,
                    warmup_steps=1,
                    save_every=0,
                    log_every=1,
                    track_weight_delta=False,
                    model_flops_per_frame=tp_mod.model_flops_per_frame(moshi),
                ),
                out_dir=d,
                log=lambda s: None,
            )
            return [_json.loads(x) for x in open(os.path.join(d, "metrics.train.jsonl"))]
    finally:
        tp_mod.gpu_peak_flops = saved


_recs = _loop(None)
ck(
    len(_recs) == 3
    and all(r["real_frames_per_step"] == 2 * 67 and r["real_frames_per_s"] > 0 and r["mfu"] is None for r in _recs),
    "run_training logs real_frames_per_step = grad_accum x real frames, frames/s, mfu null off-GPU",
    str([(r["real_frames_per_step"], r["real_frames_per_s"], r["mfu"]) for r in _recs]),
)
_recs = _loop(1e9)
_f = tp_mod.model_flops_per_frame(moshi)
ck(
    # fps is logged to 0.1 and mfu to 1e-4, so allow that rounding
    all(
        abs(r["mfu"] - _f * r["real_frames_per_s"] / 1e9) <= 1e-4 + r["mfu"] * (0.06 / r["real_frames_per_s"] + 1e-3)
        for r in _recs
    ),
    "  ...and mfu = FLOPs/frame x frames/s / peak",
    str([r["mfu"] for r in _recs]),
)
_tr = (LIBDIR / "rl" / "trainer.py").read_text()
ck(
    "**throughput_scalars(" in _tr and "self._win_frames += int(codes.shape[0] * codes.shape[2])" in _tr,
    "RL trainer logs the same three scalars over its GRPO frames",
)

print()
if fails:
    print(f"FAILED {len(fails)}: {fails}")
    sys.exit(1)
print("OK")
