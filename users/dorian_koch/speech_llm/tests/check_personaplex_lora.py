"""Guard: PersonaPlex LoRA (train_scope="lora") -- identity at init, and the saved adapter loads back.

WHY. The PersonaPlex launcher and the PersonaPlex inference engine are built bespoke (not through
``MoshiFamilyModel``), so each must wrap the SAME layers under the SAME names via
``moshi_family.build.apply_lora_to``, or an adapter trained by one silently fails to reach the other.

  1. IDENTITY AT INIT. Wrapping a PersonaPlex ``LMModel`` must not change its outputs (lora_B = 0).
     A non-zero lora_B must change them -- so the comparison can fail.
  2. ROUND TRIP. Perturb the adapter, save it with the launcher's own ``save_trained_params(...,
     is_lora_param)``, then build a FRESH model from the same base weights the way the engine does
     (``apply_lora_to`` then ``overlay_state_dict``) and require identical outputs.
  3. TRAINABLE SET. After the launcher's freezing rule only lora_A/lora_B require grad, and every
     ``nn.Linear`` of the LM was wrapped.

Tiny random PersonaPlex LMModel (real class, real delays / dep_q=16), CPU, a few seconds:
    ./hpc-venv.py --cluster i6-rz moshi_family_venv_v1 --sh 'cd <setup> && python \
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_personaplex_lora.py'
"""

import copy
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import torch  # noqa: E402

from moshi_family.build import LoraConfig, apply_lora_to, overlay_state_dict  # noqa: E402
from moshi_family.personaplex.models import loaders  # noqa: E402
from moshi_family.personaplex.models.lm import LMModel  # noqa: E402
from moshi_family.train_loop import is_lora_param, save_trained_params  # noqa: E402

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
    depformer_num_layers=1,
    dep_q=16,
)
base = LMModel(device="cpu", dtype=torch.float32, **kw).eval()
card = int(base.card)
codes = torch.randint(0, 50, (1, 17, 12))
codes[:, 0] = torch.randint(0, 100, (1, 12))


def out(m):
    with torch.no_grad():
        o = m.forward_train(codes)
    return torch.nan_to_num(o.logits.float()), torch.nan_to_num(o.text_logits.float())


ref = out(base)
n_linear = sum(isinstance(m, torch.nn.Linear) for m in base.modules())
lora_m = copy.deepcopy(base)
cfg = LoraConfig(rank=4, scaling=2.0)
wrapped = apply_lora_to(lora_m, cfg, dtype=torch.float32)
assert wrapped == n_linear, f"wrapped {wrapped} of {n_linear} Linear layers"
got = out(lora_m)
assert all(torch.equal(a, b) for a, b in zip(ref, got)), "LoRA at init changed the outputs (lora_B must be 0)"

for n, p in lora_m.named_parameters():  # the launcher's freezing rule
    p.requires_grad = is_lora_param(n)
trainable = [n for n, p in lora_m.named_parameters() if p.requires_grad]
assert trainable and all(("lora_A" in n or "lora_B" in n) for n in trainable), trainable[:3]

with torch.no_grad():
    for n, p in lora_m.named_parameters():
        if "lora_B" in n:
            p.normal_(0, 0.05)
perturbed = out(lora_m)
assert not all(torch.equal(a, b) for a, b in zip(ref, perturbed)), (
    "guard is vacuous: a non-zero adapter changed nothing"
)
print(f"[1] identity at init over {wrapped} wrapped Linears; a non-zero lora_B changes the output")

tmp = Path(tempfile.mkdtemp(prefix="check_ppx_lora_"))
sd = save_trained_params(tmp / "lora.safetensors", lora_m, predicate=is_lora_param)
fresh = copy.deepcopy(base)
apply_lora_to(fresh, cfg, dtype=torch.float32)
overlay_state_dict(fresh, str(tmp / "lora.safetensors"), tag="check", device="cpu")
again = out(fresh)
# Tolerance, not equality: every tensor is bit-identical (checked below) but float32 summation order
# can differ between the two module trees at the 1e-7 level. The adapter itself moves outputs by >1e-3.
sm, sf = lora_m.state_dict(), fresh.state_dict()
assert set(sm) == set(sf) and all(torch.equal(sm[k], sf[k]) for k in sm), "reloaded weights differ"
rt = max((a - b).abs().max().item() for a, b in zip(perturbed, again))
eff = max((a - b).abs().max().item() for a, b in zip(perturbed, ref))
assert rt < 1e-5 < 1e-3 < eff, f"round-trip diff {rt:.2e} vs adapter effect {eff:.2e}"
print(
    f"[2] round trip: {len(sd)} adapter tensors reload bit-identically onto a fresh engine-style model; "
    f"output diff {rt:.1e} vs adapter effect {eff:.1e}"
)
print(f"[3] trainable set: {len(trainable)} lora_A/lora_B tensors only")
print("OK")
