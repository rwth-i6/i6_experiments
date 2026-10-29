"""Guard: the PersonaPlex checkpoint must load with NO missing and NO unexpected keys.

``personaplex/models/loaders.py`` used to load with ``strict=False``, which means any key the
model expects but the checkpoint lacks is left at whatever ``__init__`` produced -- random init --
and the model still runs, still sounds like speech, and is simply wrong. That is the worst kind of
failure: no crash, no NaN, plausible output. It also made the PersonaPlex port unverifiable, since
a module rename that dropped weights would have looked identical to success.

Loading is now ``strict=True``. This check proves that is safe to keep, and keeps proving it if
the checkpoint or the module layout ever moves. It also asserts the two "Patch" blocks that used
to rewrite the state dict before loading (expanding depformer self_attn tensors, back-filling
depformer steps 8..15 by copying 0..7) really are unnecessary for the shipped checkpoint -- they
were deleted on that basis.

Costs nothing: reads only the safetensors HEADER (not the 16 GB of weights) and builds the model
on the ``meta`` device, so no GPU and no real memory.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_personaplex_state_dict_parity.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from moshi_family.personaplex.models import loaders  # noqa: E402
from moshi_family.personaplex.models.lm import LMModel  # noqa: E402

#: The shared hub cache; the checkpoint is resolved as the engines resolve it (the current snapshot), so an upstream
#: re-upload is the file this reads.
HF_HUB_CACHE = "/hpcwork/p0023999/common_hf_home/hub"
#: Frozen so an upstream re-upload is caught rather than silently absorbed.
EXPECTED_CKPT_KEYS = 475
#: Groups the deleted Patch 2 used to back-fill for depformer steps 8..15.
BACKFILL_GROUPS = ["gating", "linears", "depformer_in", "depformer_emb"]

from huggingface_hub import hf_hub_download  # noqa: E402

try:
    CKPT = hf_hub_download(
        "nvidia/personaplex-7b-v1", loaders.MOSHI_NAME, cache_dir=HF_HUB_CACHE, local_files_only=True
    )
except Exception as exc:  # noqa: BLE001
    # Not a pass: a missing checkpoint leaves the strict-load invariant unchecked.
    sys.exit(f"SKIP-AS-FAIL: PersonaPlex checkpoint not in the HF cache ({exc!r})")
print(f"[ok] checkpoint                          {CKPT}")

with safe_open(CKPT, framework="pt") as handle:
    ckpt_shapes = {k: tuple(handle.get_slice(k).get_shape()) for k in handle.keys()}

assert len(ckpt_shapes) == EXPECTED_CKPT_KEYS, (
    f"checkpoint has {len(ckpt_shapes)} keys, expected {EXPECTED_CKPT_KEYS}. The upstream weights "
    "changed; re-verify the loader before updating this number."
)
print(f"[ok] checkpoint keys                     {len(ckpt_shapes)}")

lm_kwargs = dict(loaders._lm_kwargs)
lm_kwargs["dep_q"] = 16  # loaders.get_moshi_lm applies the same override
model = LMModel(device="meta", dtype=torch.bfloat16, **lm_kwargs)
model_shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}

# --- the deleted patches must stay unnecessary ---------------------------------------------------
would_expand = [
    k
    for k, s in ckpt_shapes.items()
    if "depformer" in k and "self_attn" in k and k in model_shapes and s != model_shapes[k]
]
assert not would_expand, (
    "depformer self_attn tensors disagree in shape, so the deleted Patch 1 (expand) was load-"
    f"bearing after all: {would_expand[:4]}"
)
missing = sorted(set(model_shapes) - set(ckpt_shapes))
would_backfill = [k for k in missing if any(f"{g}.{n}." in k for g in BACKFILL_GROUPS for n in range(8, 16))]
assert not would_backfill, (
    "model expects depformer 8..15 weights the checkpoint lacks, so the deleted Patch 2 (back-fill "
    f"0..7 -> 8..15) was load-bearing after all: {would_backfill[:4]}"
)
print(f"[ok] deleted weight patches stay dead    expand={len(would_expand)} backfill={len(would_backfill)}")

# --- the real strict load ------------------------------------------------------------------------
meta_state = {k: torch.empty(s, device="meta", dtype=torch.bfloat16) for k, s in ckpt_shapes.items()}
result = model.load_state_dict(meta_state, strict=True, assign=True)
assert not result.missing_keys, f"missing keys under strict load: {result.missing_keys[:6]}"
assert not result.unexpected_keys, f"unexpected keys under strict load: {result.unexpected_keys[:6]}"
print(f"[ok] strict load, no allowlist           {len(model_shapes)} model keys, all matched")

print("\nPersonaPlex checkpoint loads strictly -- no weight is silently left at init")
