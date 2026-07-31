"""Guard: streaming attention must equal the non-streaming forward, across a ring-cache wrap.

``RingKVCache`` keeps only the last ``capacity`` keys and hands each one a *position* so the causal
bias can order them. Getting that position off by one does not crash, does not NaN, and does not
show up in a loss curve -- it silently drops the oldest key from every attention step once the ring
wraps, so the model just attends to a slightly shorter context forever.

That is exactly the bug found in the pre-port PersonaPlex stack (2026-07-31): it incremented
``end_offset`` *before* deriving ``end_index``, so ``end_index`` pointed one slot past the last
write -- at the oldest entry -- which then received a position greater than the query's and was
masked out. Effective context was ``capacity - 1``. Because the depformer's capacity equals its
16 steps-per-frame, it wrapped on *every frame*: codebook 15 never attended to the text token.

The invariant that catches it is not "the arithmetic looks right", it is:

    streaming, fed one step at a time  ==  the same input fed as one full sequence

so this file asserts that, over enough steps to wrap the ring several times, at several batch
sizes (a per-batch offset bug hides at B=1).

To prove the guard actually bites, it then re-introduces the off-by-one -- by rewriting the *real*
source of ``RingKVCache.complete`` rather than a hand-copy, so the mutation cannot go stale -- and
asserts the check fails. A guard that passes against the bug it was written for is worthless.

Run from the setup root, no GPU needed:
    CUDA_HOME=/usr .venv/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_streaming_kv_parity.py
"""

import inspect
import os
import sys
import textwrap

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import torch  # noqa: E402

from moshi_family.modules.transformer import (  # noqa: E402
    RingKVCache,
    StreamingTransformer,
)

# Ring wraps at `CONTEXT`; run several times that so a wrap-related error cannot hide in the tail.
CONTEXT = 8
STEPS = 24
TOL = 1e-5


def build_transformer():
    """A small causal transformer with a bounded context -- the shape that exercises the ring."""
    torch.manual_seed(0)
    model = StreamingTransformer(
        d_model=64,
        num_heads=4,
        num_layers=2,
        dim_feedforward=128,
        causal=True,
        context=CONTEXT,
        positional_embedding="rope",  # exercises the offset path, where the bug lived
    )
    return model.eval()


def max_streaming_error(model, batch_size: int) -> float:
    """Feed `STEPS` frames one at a time, and as one sequence; return the largest disagreement."""
    torch.manual_seed(1234)
    x = torch.randn(batch_size, STEPS, 64)

    with torch.no_grad():
        reference = model(x)  # non-streaming: the ground truth
        steps = []
        with model.streaming(batch_size):
            for t in range(STEPS):
                steps.append(model(x[:, t : t + 1]))
        streamed = torch.cat(steps, dim=1)

    assert streamed.shape == reference.shape, (streamed.shape, reference.shape)
    return (streamed - reference).abs().max().item()


def check(model, *, expect_parity: bool, label: str):
    """Assert streaming matches (or, for the mutation, fails to match) the full-sequence forward."""
    worst = {b: max_streaming_error(model, b) for b in (1, 2, 3)}
    ok = all(err < TOL for err in worst.values())
    detail = ", ".join(f"B={b} {err:.3g}" for b, err in worst.items())
    if expect_parity:
        assert ok, (
            f"{label}: streaming diverges from the non-streaming forward ({detail}). The ring cache "
            f"is dropping or mis-positioning a key once it wraps at capacity={CONTEXT}."
        )
    else:
        assert not ok, (
            f"{label}: the deliberate off-by-one did NOT break parity ({detail}). This guard no "
            "longer detects the bug it exists for -- fix the mutation below before trusting it."
        )
    print(f"[ok] {label:44s} {detail}")


def with_legacy_off_by_one():
    """Rewrite the real `complete` source to increment before deriving the index, as PersonaPlex did.

    Derived from the live source rather than a copy, so it tracks refactors; if the expression it
    keys on disappears, this raises instead of silently testing nothing.
    """
    src = textwrap.dedent(inspect.getsource(RingKVCache.complete))
    marker = "+ T - 1"
    assert src.count(marker) == 1, (
        f"expected exactly one {marker!r} in RingKVCache.complete, found {src.count(marker)}. "
        "The position arithmetic was refactored -- update this mutation to match."
    )
    ns = dict(sys.modules[RingKVCache.__module__].__dict__)
    exec(compile(src.replace(marker, "+ T"), "<legacy-ring-kv>", "exec"), ns)
    return ns["complete"]


model = build_transformer()

# --- the invariant -------------------------------------------------------------------------------
check(model, expect_parity=True, label="streaming == non-streaming (current)")

# --- prove the invariant is load-bearing ----------------------------------------------------------
original = RingKVCache.complete
try:
    RingKVCache.complete = with_legacy_off_by_one()
    check(model, expect_parity=False, label="off-by-one is detected (mutation)")
finally:
    RingKVCache.complete = original

check(model, expect_parity=True, label="streaming == non-streaming (restored)")

print("\nall streaming KV-cache parity checks passed")
