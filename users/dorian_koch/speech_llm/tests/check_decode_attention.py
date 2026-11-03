"""Guard: the streaming decode-attention kernel (``StreamingMultiheadAttention.decode_kernel``) is the same function.

WHY. The flag swaps the per-step attention of the temporal transformer from SDPA + a boolean mask over every ring
slot to ``moshi_family.modules.decode_attention``, which reads only each row's filled prefix of the cache. The
prefix is derived from ``RingKVCache.end_offset`` instead of the positions + mask, so a wrong slot count (one key
too few after the ring wraps, a skipped row counted as stepped) still produces fluent audio -- only a direct
comparison can tell.

  [1] kernel == float64 reference on random tensors: lengths 0 / 1 / partial / full, a strided q, fp32 (rel < 1e-5)
      and bf16 (no worse than SDPA's own error against the same reference)
  [2] real ``StreamingTransformer`` (rope, causal, context 40) streamed 130 steps, fp32: flag on == flag off at
      every step (abs < 1e-4), across 3 ring wraps, rows with exec_mask off on some steps, and a per-row reset; the
      fast path really ran (spy), the default is off, and a module whose context != capacity keeps SDPA
  [3] same under CUDA graphs (``CUDAGraphed``, as the rollout engine runs it): one capture, slot counts change
      between replays
  [4] the comparison can fail: two one-line mutations of the live ``complete_decode`` source are rejected (slot
      count capped one short of capacity; exec_mask ignored)

Run from the setup root on a GPU (Triton), ~1 min:
    CUDA_HOME=/usr output/venv/moshi_family/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_decode_attention.py
"""

import contextlib
import inspect
import os
import sys
import textwrap

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from moshi_family.modules import decode_attention as da  # noqa: E402
from moshi_family.modules import transformer as tr  # noqa: E402
from moshi_family.utils.compile import CUDAGraphed  # noqa: E402

assert torch.cuda.is_available(), "check_decode_attention needs a GPU (Triton kernel)"
DEV = "cuda"
CONTEXT, STEPS, B, D_MODEL, HEADS = 40, 130, 3, 256, 4


# [1] ---------------------------------------------------------------------------------------------------------------
def reference(q, k, v, lengths):
    s = torch.einsum("bhd,bhnd->bhn", q[:, :, 0].double(), k.double()) * q.shape[-1] ** -0.5
    valid = torch.arange(k.shape[2], device=k.device)[None, None] < lengths[:, None, None]
    s = s.masked_fill(~valid, float("-inf"))
    out = torch.einsum("bhn,bhnd->bhd", s.softmax(-1), v.double())[:, :, None]
    return torch.where(lengths.view(-1, 1, 1, 1) > 0, out, torch.zeros_like(out))


def check_kernel():
    g = torch.Generator(DEV).manual_seed(0)
    Bk, H, S, D = 5, 6, 300, 128
    lengths = torch.tensor([0, 1, 77, 299, 300], device=DEV)
    for dtype, tol in ((torch.float32, 1e-5), (torch.bfloat16, None)):
        cache = torch.randn(2, Bk, H, S, D, device=DEV, dtype=dtype, generator=g)
        q = torch.randn(Bk, 1, 3, H, D, device=DEV, dtype=dtype, generator=g)[:, :, 0].transpose(1, 2)
        assert not q.is_contiguous()
        ref = reference(q, cache[0], cache[1], lengths)
        out = da.decode_attention(q, cache[0], cache[1], lengths)
        err = ((out.double() - ref).abs().max() / ref.abs().max()).item()
        if tol is not None:
            assert err < tol, f"kernel != reference in {dtype}: rel {err:.2e}"
        else:
            mask = (torch.arange(S, device=DEV)[None] < lengths[:, None])[:, None, None]
            sd = F.scaled_dot_product_attention(q[1:], cache[0][1:], cache[1][1:], mask[1:])
            sdpa_err = ((sd.double() - ref[1:]).abs().max() / ref.abs().max()).item()
            assert err <= 1.5 * sdpa_err, f"bf16 kernel rel err {err:.2e} vs SDPA's {sdpa_err:.2e}"
        assert (out[0] == 0).all(), "a row with no valid slot must give zeros"
        print(f"[1] kernel vs float64 reference, {str(dtype):15s} rel err {err:.2e}")


# [2] / [3] ---------------------------------------------------------------------------------------------------------
def build(flag: bool):
    torch.manual_seed(0)
    m = tr.StreamingTransformer(
        d_model=D_MODEL,
        num_heads=HEADS,
        num_layers=2,
        dim_feedforward=512,
        causal=True,
        context=CONTEXT,
        positional_embedding="rope",
        device=DEV,
        dtype=torch.float32,
    ).eval()
    for mod in m.modules():
        if isinstance(mod, tr.StreamingMultiheadAttention):
            assert mod.decode_kernel is False, "decode_kernel must default to off"
            mod.decode_kernel = flag
    return m


def schedule():
    """Per step: (input, exec_mask, reset_mask). Row 1 skips every 7th step, row 2 is reset at step 70 and runs
    one skipped step right after (a fresh row with a stepped-but-not-executed slot)."""
    g = torch.Generator(DEV).manual_seed(1)
    out = []
    for t in range(STEPS):
        x = torch.randn(B, 1, D_MODEL, device=DEV, generator=g)
        ex = torch.ones(B, dtype=torch.bool, device=DEV)
        if t % 7 == 3:
            ex[1] = False
        if t == 72:
            ex[2] = False
        rs = torch.zeros(B, dtype=torch.bool, device=DEV)
        if t == 70:
            rs[2] = True
        out.append((x, ex, rs))
    return out


def stream(model, sched, graphed: bool, calls=None):
    outs = []
    with torch.no_grad(), model.streaming(B):
        if calls is not None:
            calls[0] = 0  # not the warm-up call at state init
        step = CUDAGraphed(lambda x: model(x), disable=not graphed)
        for x, ex, rs in sched:
            if rs.any():
                model.reset_streaming(rs)
            model.set_exec_mask(ex)
            outs.append(step(x).clone())
    return torch.stack(outs)


@contextlib.contextmanager
def spy():
    calls = [0]
    orig = da.decode_attention

    def counted(*a, **k):
        calls[0] += 1
        return orig(*a, **k)

    da.decode_attention = counted
    try:
        yield calls
    finally:
        da.decode_attention = orig


def max_err(graphed: bool) -> float:
    sched = schedule()
    base = stream(build(False), sched, graphed)
    with spy() as calls:
        fast = stream(build(True), sched, graphed, calls)
    assert calls[0] > 0, "decode_kernel=True never reached decode_attention"
    # rows whose baseline attention had no valid key (fresh row, step not executed) are NaN in SDPA, 0 here
    ok = torch.isfinite(base)
    assert ok.float().mean() > 0.99, "baseline unexpectedly non-finite"
    return (fast[ok] - base[ok]).abs().max().item()


def check_streaming():
    for graphed in (False, True):
        err = max_err(graphed)
        assert err < 1e-4, f"decode_kernel changes the streamed output (graphed={graphed}): max abs {err:.2e}"
        print(f"[{3 if graphed else 2}] streamed {STEPS} steps, context {CONTEXT} ({STEPS // CONTEXT} wraps), skips + reset, "
              f"graphed={graphed}: max abs diff {err:.2e}")  # fmt: skip
    m = build(True)
    with spy() as calls, torch.no_grad(), m.streaming(B):
        calls[0] = 0
        for mod in m.modules():
            if isinstance(mod, tr.StreamingMultiheadAttention):
                mod.context = CONTEXT - 5  # after the cache was sized: the window mask matters -> SDPA path
        m(torch.randn(B, 1, D_MODEL, device=DEV))
    assert calls[0] == 0, "context != capacity must keep the masked SDPA path"
    print("[2] context != capacity keeps SDPA")


# [4] ---------------------------------------------------------------------------------------------------------------
@contextlib.contextmanager
def mutated(old: str, new: str):
    src = textwrap.dedent(inspect.getsource(tr.RingKVCache.complete_decode))
    assert src.count(old) == 1, f"mutation anchor {old!r} not found once in complete_decode -- update the guard"
    ns = dict(tr.__dict__)
    exec(compile(src.replace(old, new), "<mutant>", "exec"), ns)
    orig, tr.RingKVCache.complete_decode = tr.RingKVCache.complete_decode, ns["complete_decode"]
    try:
        yield
    finally:
        tr.RingKVCache.complete_decode = orig


def check_mutants():
    for name, old, new in (
        ("slot count one short of capacity", "clamp(max=self.capacity)", "clamp(max=self.capacity - 1)"),
        ("exec_mask ignored", "add_(exec_mask.to(self.end_offset.dtype))", "add_(1)"),
    ):
        with mutated(old, new):
            err = max_err(False)
        assert err > 1e-3, f"mutant '{name}' NOT detected (max abs diff {err:.2e})"
        print(f"[4] mutant detected: {name:34s} max abs diff {err:.2e}")


check_kernel()
check_streaming()
check_mutants()
print("\nall decode-attention checks passed")
