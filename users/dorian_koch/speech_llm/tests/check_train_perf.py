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
means agreement to ~1e-6; the GPU/bf16 numbers are in finetuning.md "Training performance".

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
  [5] the launcher only calls apply_train_perf when a switch is set (AST), and reads every new key

Run from the setup root, CPU, ~20 s:
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

from moshi_family.build import LoraConfig, apply_lora_to  # noqa: E402
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
LM = ppx_lm.LMModel
rejected(
    "row skip that ignores the codebook delays (keeps loss rows only)",
    [(LM, "forward_depformer_rows", mutated(LM, "forward_depformer_rows", "need[:, d:] |= fm[:, :-d]", "pass"))],
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
print("[5] launcher wiring")
launcher = (
    Path(inspect.getsourcefile(sys.modules["moshi_family.train_perf"])).parent / "personaplex" / "finetune_launcher.py"
)
src = launcher.read_text()
tree = ast.parse(src)
main = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
keys = {
    n.args[0].value
    for n in ast.walk(main)
    if isinstance(n, ast.Call)
    and getattr(n.func, "attr", "") == "get"
    and getattr(n.func.value, "id", "") == "cfg"
    and n.args
    and isinstance(n.args[0], ast.Constant)
}
new_keys = {
    "contiguous_step_inputs",
    "fused_causal_attention",
    "fused_lora_add",
    "depformer_skip_masked",
    "length_bucket_pool",
    "prefetch_batches",
}
ck(new_keys <= keys, "the launcher reads every new key", str(sorted(new_keys - keys)))
calls = [n for n in ast.walk(main) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "apply_train_perf"]
guarded = False
for node in ast.walk(main):
    if isinstance(node, ast.If) and any(c in list(ast.walk(node)) for c in calls):
        guarded = True
ck(len(calls) == 1 and guarded, "apply_train_perf is called once, only under an explicit switch")
ck(
    "forward_train(codes)" in src and "depformer_frame_mask=loss_mask" in src,
    "loss_step keeps the plain forward_train(codes) call for runs without the row skip",
)

print()
if fails:
    print(f"FAILED {len(fails)}: {fails}")
    sys.exit(1)
print("OK")
