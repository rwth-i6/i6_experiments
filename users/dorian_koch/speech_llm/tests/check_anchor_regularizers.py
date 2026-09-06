"""Guard: the anchor-to-pretrained regularisers (backlog A5 KL / A6 L2-SP).

Why these exist: A11 (2026-09-06) showed full-FT and LoRA collapse IDENTICALLY -- 6.6% vs 7.2% at
n=1000, 0.6 pts apart with identical quality -- so *how* the weights are reached is not the lever.
These two terms constrain *where* they may go, relative to the pretrained model that still scores
17.5%. Both exploit the fact that a LoRA run carries its own base around with it.

Each check below is written so it fails if the implementation is replaced by a plausible-looking
approximation, because every one of these is silent when wrong:

  * ``adapters_disabled`` must reproduce the base EXACTLY and restore per-module scalings. A
    version that leaks a modified scaling would quietly change the student's own forward.
  * ``lora_delta_sq_norm`` must equal ``||scaling * B @ A||_F^2``. The tempting cheap stand-in,
    ``||A||^2 + ||B||^2``, is a DIFFERENT quantity -- asserted here to actually differ, so the
    identity check cannot pass by coincidence.
  * ``text_kl_to_reference`` must average over VALID text positions only. Averaging over all of
    them makes ``kl_weight`` depend on the corpus's padding rate rather than on the model, which no
    curve would ever reveal.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_anchor_regularizers.py
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
sys.path.insert(0, str(SETUP / "recipe" / "speech_llm" / "full_duplex"))
os.environ.setdefault("CUDA_HOME", "/usr")

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from moshi_family.modules.lora import (  # noqa: E402
    LoRALinear,
    adapters_disabled,
    lora_delta_sq_norm,
    replace_all_linear_with_lora,
)
from moshi_family.train_data_common import text_kl_to_reference  # noqa: E402


def _lora_mlp(scaling=2.0, rank=4, seed=0):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(8, 16, bias=False), nn.Linear(16, 8, bias=False))
    replace_all_linear_with_lora(net, rank=rank, scaling=scaling, dtype=torch.float32)
    # lora_B is zero-init, so the adapter starts as an exact identity. Give it real content,
    # otherwise every check below passes trivially on a model whose delta is zero.
    for m in net.modules():
        if isinstance(m, LoRALinear):
            nn.init.normal_(m.lora_B.weight, std=0.1)
    return net


def check_adapters_disabled_recovers_the_base():
    net = _lora_mlp()
    x = torch.randn(3, 8)

    with_adapter = net(x)
    with adapters_disabled(net) as n:
        assert n == 2, f"expected 2 LoRALinear modules, saw {n}"
        base = net(x)
    after = net(x)

    # The base must be exactly the frozen path, computed independently of the context manager.
    ref = x
    for m in net:
        ref = torch.nn.functional.linear(ref, m.frozen_W.weight)
    assert torch.allclose(base, ref, atol=1e-6), (base - ref).abs().max()

    # Non-vacuity: the adapter must actually have been doing something.
    assert not torch.allclose(with_adapter, base, atol=1e-4), (
        "the adapter changes nothing, so 'disabling it reproduces the base' proves nothing"
    )
    # ...and the original behaviour must be restored, exactly.
    assert torch.allclose(after, with_adapter, atol=1e-7), "scaling was not restored on exit"
    print("PASS  adapters_disabled reproduces the frozen base exactly and restores scaling")


def check_scaling_restored_on_exception_and_per_module():
    net = _lora_mlp()
    mods = [m for m in net.modules() if isinstance(m, LoRALinear)]
    mods[0].scaling = 3.0  # a heterogeneous model: the restore must not collapse to one value
    mods[1].scaling = 5.0
    try:
        with adapters_disabled(net):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert [m.scaling for m in mods] == [3.0, 5.0], (
        f"per-module scalings must survive an exception, got {[m.scaling for m in mods]}"
    )
    print("PASS  per-module scalings are restored individually, including on exception")


def check_l2sp_is_the_real_delta_norm():
    for scaling in (1.0, 2.0, 3.5):
        net = _lora_mlp(scaling=scaling, seed=1)
        got = lora_delta_sq_norm(net)
        # Reference: materialise scaling * B @ A per layer and take the Frobenius norm squared.
        want = 0.0
        naive_param_norm = 0.0
        for m in net.modules():
            if not isinstance(m, LoRALinear):
                continue
            delta = m.scaling * (m.lora_B.weight.float() @ m.lora_A.weight.float())
            want = want + delta.pow(2).sum()
            naive_param_norm = naive_param_norm + m.lora_A.weight.float().pow(2).sum()
            naive_param_norm = naive_param_norm + m.lora_B.weight.float().pow(2).sum()
        assert torch.allclose(got, want, rtol=1e-5), (scaling, got.item(), want.item())
        # Non-vacuity: the cheap stand-in must NOT coincide with the right answer, else this
        # check would pass for an implementation that penalises the wrong thing.
        assert not torch.allclose(got, torch.as_tensor(naive_param_norm), rtol=1e-2), (
            "||A||^2+||B||^2 happens to equal ||BA||^2 here, so this fixture cannot tell a correct "
            "L2-SP implementation from the wrong one -- change the fixture"
        )
    # It must also be differentiable back to the adapter parameters.
    net = _lora_mlp(seed=2)
    lora_delta_sq_norm(net).backward()
    b = [m.lora_B.weight for m in net.modules() if isinstance(m, LoRALinear)][0]
    assert b.grad is not None and b.grad.abs().sum() > 0, "L2-SP must produce a gradient on lora_B"
    print("PASS  lora_delta_sq_norm == ||scaling*B@A||^2 exactly, and is differentiable")


def check_l2sp_refuses_a_model_with_no_adapter():
    plain = nn.Sequential(nn.Linear(4, 4, bias=False))
    try:
        lora_delta_sq_norm(plain)
    except AssertionError as e:
        assert "no LoRALinear" in str(e), e
        print("PASS  L2-SP on an adapter-less model raises instead of silently returning zero")
        return
    raise SystemExit("FAIL: L2-SP returned a value for a model with no adapter")


def _out(logits, mask):
    return SimpleNamespace(text_logits=logits.unsqueeze(1), text_mask=mask.unsqueeze(1))


def check_kl_zero_iff_identical():
    torch.manual_seed(0)
    b, t, v = 2, 5, 7
    logits = torch.randn(b, t, v)
    mask = torch.ones(b, t, dtype=torch.bool)
    loss_mask = torch.ones(b, t, dtype=torch.bool)

    same = text_kl_to_reference(_out(logits, mask), _out(logits.clone(), mask), loss_mask)
    assert abs(same.item()) < 1e-6, f"KL to itself must be 0, got {same.item()}"

    other = text_kl_to_reference(_out(logits, mask), _out(torch.randn(b, t, v), mask), loss_mask)
    assert other.item() > 1e-3, f"KL between different distributions must be positive, got {other}"
    print("PASS  KL is zero against itself and positive against a different teacher")


def check_kl_ignores_masked_positions():
    """The one that matters: averaging over ALL positions ties kl_weight to the padding rate."""
    torch.manual_seed(0)
    b, t, v = 2, 6, 7
    s = torch.randn(b, t, v)
    teach = torch.randn(b, t, v)
    mask = torch.ones(b, t, dtype=torch.bool)
    mask[:, 3:] = False  # second half invalid
    loss_mask = torch.ones(b, t, dtype=torch.bool)

    before = text_kl_to_reference(_out(s, mask), _out(teach, mask), loss_mask)

    # Put wildly different values in the INVALID positions only. A correct masked mean is unchanged.
    s2, teach2 = s.clone(), teach.clone()
    s2[:, 3:] = 50.0 * torch.randn(b, t - 3, v)
    teach2[:, 3:] = -50.0 * torch.randn(b, t - 3, v)
    after = text_kl_to_reference(_out(s2, mask), _out(teach2, mask), loss_mask)
    assert torch.allclose(before, after, atol=1e-5), (
        f"masked positions changed the KL ({before.item()} -> {after.item()}) -- the term is being "
        f"averaged over padding, so kl_weight would mean something different on every corpus"
    )

    # loss_mask must gate too, not just text_mask.
    lm2 = loss_mask.clone()
    lm2[0, :] = False
    gated = text_kl_to_reference(_out(s, mask), _out(teach, mask), lm2)
    assert not torch.allclose(before, gated, atol=1e-6), "loss_mask is being ignored"
    print("PASS  KL averages over valid text positions only, gated by BOTH masks")


def check_kl_temperature_scaling():
    torch.manual_seed(0)
    b, t, v = 2, 4, 9
    s, teach = torch.randn(b, t, v), torch.randn(b, t, v)
    mask = torch.ones(b, t, dtype=torch.bool)
    vals = {T: text_kl_to_reference(_out(s, mask), _out(teach, mask), mask, temperature=T).item()
            for T in (1.0, 2.0, 4.0)}
    # T^2 scaling keeps the term the same ORDER across temperatures; without it a high T would
    # shrink the term by ~1/T^2 and silently turn the regulariser off in a temperature sweep.
    assert all(v > 0 for v in vals.values()), vals
    assert vals[4.0] > vals[1.0] / 10, (
        f"the T^2 rescaling looks absent -- KL collapses with temperature: {vals}"
    )
    print(f"PASS  temperature is T^2-rescaled (KL at T=1/2/4: "
          f"{vals[1.0]:.3f}/{vals[2.0]:.3f}/{vals[4.0]:.3f})")


def check_launcher_refuses_full_ft():
    src = (SETUP / "recipe/speech_llm/full_duplex/moshi_family/moshi_finetune_launcher.py").read_text()
    assert "full_finetuning and (kl_weight > 0 or l2sp_weight > 0)" in src, (
        "the launcher must refuse full-FT + an anchor term: a full finetune has already moved the "
        "pretrained weights, so there is nothing to anchor to and both terms would be no-ops that "
        "look like a running experiment"
    )
    for knob in ("kl_weight", "kl_temperature", "l2sp_weight"):
        assert f'cfg.get("{knob}"' in src, f"{knob} must be read from the config"
    print("PASS  the launcher reads all three knobs and refuses them on a full finetune")


if __name__ == "__main__":
    check_adapters_disabled_recovers_the_base()
    check_scaling_restored_on_exception_and_per_module()
    check_l2sp_is_the_real_delta_norm()
    check_l2sp_refuses_a_model_with_no_adapter()
    check_kl_zero_iff_identical()
    check_kl_ignores_masked_positions()
    check_kl_temperature_scaling()
    check_launcher_refuses_full_ft()
    print("ALL PASS")
