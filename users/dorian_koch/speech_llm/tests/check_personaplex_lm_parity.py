"""Guard: PersonaPlex's LM must compute the same thing as the canonical LM on the same weights.

The port replaces PersonaPlex's private copy of the model stack module by module. Each swap is
individually plausible and collectively unverifiable by inspection, so this pins the composition:
build the same small architecture in both stacks, copy one set of weights into both, and assert
the training forward agrees to floating-point noise.

The cross-stack comparison is deliberately *non-streaming*: it is the path used for training, and
it is the part that must not move. (Since the port, PersonaPlex's LM runs the canonical modules
directly, so the two agree exactly rather than to rounding -- a non-zero diff here now means
something genuinely diverged.)

Streaming is then exercised separately, as a smoke test rather than a comparison. The port moved
_LMGenState onto the canonical State ABC, and that machinery -- masked reset, the wrapped model's
streaming scope on an exit_stack, the renamed step counter -- fails at *runtime*, not at import,
so it needs real steps driven through it at more than one batch size.

Note the state dict crosses a key-layout boundary: the canonical attention stores per-step
projections as in_projs.{i}.weight where PersonaPlex stores one fused in_proj_weight. That is
absorbed by StreamingMultiheadAttention._load_hook, and this check exercises it -- if the hook
ever stops firing, the load below reports missing keys rather than silently leaving weights at
random init.

Run from the setup root, no GPU needed:
    CUDA_HOME=/usr .venv/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_personaplex_lm_parity.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import torch  # noqa: E402

from moshi_family.models.lm import LMModel as CanonicalLM  # noqa: E402
from moshi_family.personaplex.models.lm import (  # noqa: E402
    AUDIO_TOKENS_PER_STREAM,
    LMGen,
    LMModel as PersonaPlexLM,
)
from moshi_family.personaplex.models import loaders  # noqa: E402

TOL = 1e-4
DEP_Q = 16

# Derive from the SHIPPING config and shrink only the sizes, rather than inventing an architecture.
# This matters: `extra_text = (existing_text_padding_id is None)` widens PersonaPlex's text head by
# one, so a config that omits existing_text_padding_id produces a text_linear the canonical model
# cannot match -- an artefact of the test, not of the code. The real config sets it, giving
# text_linear = text_card for both, which is what the checkpoint has.
ARCH = dict(loaders._lm_kwargs)
ARCH.pop("depformer_causal", None)  # canonical reaches the same place via `causal`; it rejects this
ARCH.update(
    n_q=DEP_Q,
    dep_q=DEP_Q,
    card=48,
    text_card=48,
    dim=64,
    num_heads=4,
    num_layers=2,
    depformer_dim=64,
    depformer_num_heads=4,
    depformer_num_layers=2,
    delays=[0] * (DEP_Q + 1),
)
assert ARCH.get("existing_text_padding_id") is not None, (
    "the shipping config no longer sets existing_text_padding_id, so PersonaPlex's text head gains "
    "an extra row and no longer matches the canonical one -- re-check the port before relaxing this"
)

torch.manual_seed(0)
# PersonaPlex reaches a causal depformer via depformer_causal=True; the canonical model reaches the
# same place by forwarding its own `causal` to the depformer, and REJECTS depformer_causal.
ppx = PersonaPlexLM(device="cpu", dtype=torch.float32, depformer_causal=True, **ARCH)
canonical = CanonicalLM(device="cpu", dtype=torch.float32, **ARCH)
ppx.eval()
canonical.eval()

result = canonical.load_state_dict(ppx.state_dict(), strict=False)
assert not result.missing_keys, (
    "canonical model has weights PersonaPlex's state dict does not fill -- the attention "
    f"_load_hook is not converting the key layout: {result.missing_keys[:6]}"
)
print(f"[ok] weights transfer, no missing keys   ({len(ppx.state_dict())} tensors)")

torch.manual_seed(1234)
codes = torch.randint(0, ARCH["card"], (2, DEP_Q + 1, 6))

with torch.no_grad():
    out_ppx = ppx.forward_train(codes)
    out_canon = canonical(codes)

text_diff = (out_ppx.text_logits - out_canon.text_logits).abs().max().item()
audio_diff = (out_ppx.logits - out_canon.logits).abs().max().item()

assert text_diff < TOL, f"text logits diverge by {text_diff:.3g} (tol {TOL})"
assert audio_diff < TOL, f"audio logits diverge by {audio_diff:.3g} (tol {TOL})"
print(f"[ok] text logits agree                   max diff {text_diff:.3g}")
print(f"[ok] audio logits agree                  max diff {audio_diff:.3g}")

# --- the streaming state, on the canonical contract ----------------------------------------------
# The port moved _LMGenState onto the canonical State ABC: reset() now takes a mask, the wrapped
# model's streaming scope is entered through the state's exit_stack, and the scalar step counter
# was renamed offset_cpu. None of that is exercised by a forward pass, and all of it fails at
# runtime rather than import time, so drive a few real steps.
gen = LMGen(ppx, device="cpu", use_sampling=False)
for batch_size in (1, 2, 3):
    with gen.streaming(batch_size):
        state = gen._streaming_state
        assert hasattr(state, "exec_mask"), (
            "_LMGenState is not a canonical State subclass -- the streaming contract regressed"
        )
        emitted = sum(
            gen.step(moshi_tokens=torch.randint(0, ARCH["card"], (batch_size, AUDIO_TOKENS_PER_STREAM, 1))) is not None
            for _ in range(6)
        )
        advanced = state.offset_cpu
        assert advanced == 6, f"step counter advanced {advanced} times over 6 steps"
        assert emitted, "no frames emitted -- generation stalled"

        gen.reset_streaming()
        assert state.offset_cpu == 0, "reset(reset_mask) did not clear the step counter"
        assert bool((~state.provided).all()), "reset(reset_mask) did not clear `provided`"
    print(f"[ok] streaming B={batch_size}                        {emitted}/6 frames, counter 0->{advanced}->0")

print("\nPersonaPlex's LM matches the canonical LM, and streams on the canonical contract")
