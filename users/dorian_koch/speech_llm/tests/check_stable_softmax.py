"""Guard: ``moshi_family.stable_softmax`` (key ``stable_softmax``) computes exact CE / log-softmax / KL gradients, its
off path is the code it replaced, and every training loss and RL log-prob path goes through it.

Barron, Gnanapragasam & Tschannen 2026 (arXiv 2610.31642): the standard CE forms ``1 + e^-m`` (m = the target's logit
margin), which rounds to 1 above m ~ 16.6 in fp32, so the target gradient is exactly 0 while the others are not; a soft
target's ``q - p`` from two rounded softmaxes is noise when student ~ teacher.

  1. CE (fp32, margins 2..60, fp64 reference from the margin itself): loss and target gradient within 1e-5 relative,
     gradients sum to ~0. CAN FAIL: ``F.cross_entropy`` loses the target gradient at margin >= 17 (and is 1e-3 off at 8).
  2. log_softmax / target_logprobs on: the argmax entry's log-prob and gradient match fp64 at margin 20; CAN FAIL: the
     standard log-softmax's is exactly 0.
  3. kl_to_teacher on: the paper's example (z_T = [2, 1, 0], z_S = z_T + [-1e-7, 1e-7, 0]) matches the fp64 gradient
     within 1e-3 relative and student == teacher gives a gradient of exactly 0; CAN FAIL: the standard one is > 10% off.
  4. OFF (the default): ``cross_entropy`` / ``target_logprobs`` / ``log_softmax`` / ``kl_to_teacher`` are bit-identical
     (values and gradients) to the expressions their call sites used before.
  5. The real ``duplex_ce_loss`` (text + audio rows): on = off to 1e-6 in value; with one text target at margin 20 the
     target-logit gradient is ~ -e^-20 on and exactly 0 off (CAN FAIL), and the gradient sums to ~0 per row.
  6. Wiring (source): every ``moshi_family/**/*launcher.py`` calls ``stable_softmax.configure(cfg)`` and the native V2
     runtime (``rl/v2/launcher_v2.py``) applies ``Model.stable_softmax``; the loss modules
     (``train_data_common``, ``audex/train_data``) call no ``F.cross_entropy`` / ``F.log_softmax`` directly; the three RL
     log-prob paths call ``stable_softmax.target_logprobs``. CAN FAIL: a mutated source is flagged.

Run from the setup root (~10 s, CPU):
    <moshi_family_venv python> recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_stable_softmax.py
"""

import os
import re
import sys
from pathlib import Path

SETUP = os.getcwd()
sys.path.insert(0, os.path.join(SETUP, "recipe", "speech_llm", "full_duplex"))
sys.path.insert(0, os.path.join(SETUP, "recipe"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import moshi_family.stable_softmax as ss  # noqa: E402
import moshi_family.train_data_common as tdc  # noqa: E402

FAILURES = []


def check(ok, what):
    print(("  ok   " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILURES.append(what)


def at_margin(margin, n=64, v=2048, seed=0):
    """fp64 logits whose target sits ``margin`` above the log-sum-exp of the others, and the exact loss / target grad."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, v, dtype=torch.float64, generator=g)
    t = torch.randint(0, v, (n,), generator=g)
    lse_o = torch.logsumexp(z.scatter(-1, t[:, None], float("-inf")), -1)
    z = z.scatter(-1, t[:, None], (lse_o + margin)[:, None])
    m = z.gather(-1, t[:, None]).squeeze(-1) - torch.logsumexp(z.scatter(-1, t[:, None], float("-inf")), -1)
    return z, t, F.softplus(-m), -torch.sigmoid(-m)


def ce_grad(fn, z, t):
    z32 = z.float().requires_grad_()
    loss = fn(z32, t)
    loss.sum().backward()
    return loss.detach().double(), z32.grad.double()


def rel(a, b):
    return float(((a - b).abs() / b.abs()).max())


ss.ENABLED = True
print("[1] CE across margins (fp32 vs fp64)")
for margin in (2.0, 8.0, 16.0, 17.0, 20.0, 30.0, 60.0):
    z, t, ref_l, ref_gt = at_margin(margin)
    loss, g = ce_grad(ss.cross_entropy, z, t)
    gt = g.gather(-1, t[:, None]).squeeze(-1)
    check(
        rel(loss, ref_l) < 1e-5
        and rel(gt, ref_gt) < 1e-5
        and float(g.sum(-1).abs().max()) < 1e-5 * float(ref_gt.abs().max()),
        f"margin {margin:4.0f}: loss rel {rel(loss, ref_l):.1e}, target grad rel {rel(gt, ref_gt):.1e}",
    )
z, t, ref_l, ref_gt = at_margin(20.0)
_, g = ce_grad(lambda a, b: F.cross_entropy(a, b, reduction="none"), z, t)
check(bool((g.gather(-1, t[:, None]) == 0).all()), "CAN FAIL: F.cross_entropy at margin 20 has target grad exactly 0")
z, t, ref_l, ref_gt = at_margin(8.0)
_, g = ce_grad(lambda a, b: F.cross_entropy(a, b, reduction="none"), z, t)
check(rel(g.gather(-1, t[:, None]).squeeze(-1), ref_gt) > 1e-4, "CAN FAIL: F.cross_entropy at margin 8 is > 1e-4 off")

print("[2] log_softmax / target_logprobs at margin 20")
z, t, ref_l, ref_gt = at_margin(20.0, n=16, v=512)
for name, fn in (
    ("log_softmax", lambda a, b: -ss.log_softmax(a).gather(-1, b[:, None]).squeeze(-1)),
    ("target_logprobs", lambda a, b: -ss.target_logprobs(a, b)[0]),
):
    loss, g = ce_grad(fn, z, t)
    gt = g.gather(-1, t[:, None]).squeeze(-1)
    check(
        rel(loss, ref_l) < 1e-5 and rel(gt, ref_gt) < 1e-5,
        f"{name}: -logp rel {rel(loss, ref_l):.1e}, grad rel {rel(gt, ref_gt):.1e}",
    )
_, g = ce_grad(lambda a, b: -torch.log_softmax(a, -1).gather(-1, b[:, None]).squeeze(-1), z, t)
check(bool((g.gather(-1, t[:, None]) == 0).all()), "CAN FAIL: torch.log_softmax's target grad is exactly 0")
lp, lp_all = ss.target_logprobs(z.float(), t, full=True)
check(
    lp_all is not None and torch.allclose(lp_all.gather(-1, t[:, None]).squeeze(-1), lp),
    "full=True returns the full log-softmax",
)

print("[3] kl_to_teacher (soft target)")
z_t = torch.tensor([[2.0, 1.0, 0.0]])
z_s = z_t + torch.tensor([[-1e-7, 1e-7, 0.0]])
zs64, zt64 = z_s.double().requires_grad_(), z_t.double()
F.kl_div(F.log_softmax(zs64, -1), F.log_softmax(zt64, -1), log_target=True, reduction="sum").backward()
exact = zs64.grad
zs = z_s.clone().requires_grad_()
ss.kl_to_teacher(zs, z_t).sum().backward()
check(rel(zs.grad.double(), exact) < 1e-3, f"paper example: grad rel {rel(zs.grad.double(), exact):.1e} vs fp64")
zeq = z_t.clone().requires_grad_()
ss.kl_to_teacher(zeq, z_t).sum().backward()
check(bool((zeq.grad == 0).all()), "student == teacher: gradient exactly 0")
ss.ENABLED = False
zs = z_s.clone().requires_grad_()
ss.kl_to_teacher(zs, z_t).sum().backward()
check(rel(zs.grad.double(), exact) > 0.1, f"CAN FAIL: the standard gradient is {rel(zs.grad.double(), exact):.0%} off")

print("[4] OFF = the replaced expressions, bit for bit")
g = torch.Generator().manual_seed(3)
z = torch.randn(40, 300, generator=g) * 4
t = torch.randint(0, 300, (40,), generator=g)


def both(fn_a, fn_b):
    za, zb = z.clone().requires_grad_(), z.clone().requires_grad_()
    a, b = fn_a(za), fn_b(zb)
    a.sum().backward()
    b.sum().backward()
    return torch.equal(a, b) and torch.equal(za.grad, zb.grad)


check(both(lambda a: ss.cross_entropy(a, t), lambda a: F.cross_entropy(a, t, reduction="none")), "cross_entropy")
check(both(lambda a: ss.log_softmax(a), lambda a: torch.log_softmax(a, dim=-1)), "log_softmax")
check(
    both(
        lambda a: ss.target_logprobs(a, t)[0],
        lambda a: torch.log_softmax(a, dim=-1).gather(-1, t[..., None]).squeeze(-1),
    ),
    "target_logprobs",
)
zt = torch.randn(40, 300, generator=g)


def old_kl(s):
    log_p_s, log_p_t = F.log_softmax(s, dim=-1), F.log_softmax(zt, dim=-1)
    return (log_p_t.exp() * (log_p_t - log_p_s)).sum(-1)


check(both(lambda a: ss.kl_to_teacher(a, zt), old_kl), "kl_to_teacher")

print("[5] the real duplex_ce_loss")
B, T, VT, VA, DQ = 2, 6, 64, 32, 2
g = torch.Generator().manual_seed(4)
codes = torch.zeros(B, 1 + DQ, T, dtype=torch.long)
codes[:, 0] = torch.randint(4, VT, (B, T), generator=g)
codes[:, 1:] = torch.randint(0, VA, (B, DQ, T), generator=g)
text0 = torch.randn(B, 1, T, VT, generator=g)
lse_o = torch.logsumexp(text0[0, 0, 0].scatter(0, codes[0, 0, 0:1], float("-inf")), 0)
text0[0, 0, 0, codes[0, 0, 0]] = lse_o + 20.0  # one text target at margin 20
audio0 = torch.randn(B, DQ, T, VA, generator=g)


class Out:  # duplex_ce_loss's LMOutput fields
    pass


def run(enabled):
    ss.ENABLED = enabled
    o = Out()
    o.text_logits = text0.clone().requires_grad_()
    o.logits = audio0.clone().requires_grad_()
    o.text_mask = torch.ones(B, 1, T, dtype=torch.bool)
    o.mask = torch.ones(B, DQ, T, dtype=torch.bool)
    loss = tdc.duplex_ce_loss(
        o, codes, torch.ones(B, T, dtype=torch.bool), dep_q=DQ, other_audio_weight=0.5, text_pad_weight=0.5
    )
    loss = loss[0] if isinstance(loss, tuple) else loss
    loss.backward()
    return float(loss.detach()), o.text_logits.grad[0, 0, 0], o.logits.grad


l_on, gt_on, ga_on = run(True)
l_off, gt_off, ga_off = run(False)
ss.ENABLED = False
tgt = int(codes[0, 0, 0])
check(abs(l_on - l_off) < 1e-6, f"loss on {l_on:.7f} = off {l_off:.7f}")
check(-1e-8 < float(gt_on[tgt]) < -1e-11, f"on: margin-20 target grad {float(gt_on[tgt]):.2e} (~ -e^-20 x weight)")
check(float(gt_off[tgt]) == 0.0, "CAN FAIL: off: that target grad is exactly 0")
check(
    abs(float(gt_on.sum())) < 1e-12 and abs(float(gt_off.sum())) > 1e-12,
    f"row sums: on {float(gt_on.sum()):.1e}, off {float(gt_off.sum()):.1e}",
)
check(torch.allclose(ga_on, ga_off, atol=1e-7), "audio grads on ~ off (no high margins there)")

print("[6] wiring")
MF = Path(SETUP) / "recipe" / "speech_llm" / "full_duplex" / "moshi_family"
RAW = re.compile(r"\bF\.(cross_entropy|log_softmax)\(|torch\.log_softmax\(")


def wiring(src: dict) -> list[str]:
    bad = [
        f"{n}: no stable_softmax.configure(cfg)"
        for n, s in src.items()
        if n.endswith("launcher.py") and "stable_softmax.configure(cfg)" not in s
    ]
    if "stable_softmax.ENABLED = bool(model_spec.stable_softmax)" not in src["rl/v2/launcher_v2.py"]:
        bad.append("rl/v2/launcher_v2.py: Model.stable_softmax not applied")  # the native V2 runtime reads config.json
    for n in ("train_data_common.py", "audex/train_data.py"):
        bad += [f"{n}: raw {m.group(0)}" for m in RAW.finditer(src[n])]
    for n in ("rl/generator.py", "rl/moshi_policy.py", "rl/ppx_rollout.py"):
        if "stable_softmax.target_logprobs(" not in src[n]:
            bad.append(f"{n}: no stable_softmax.target_logprobs")
    return bad


SRC = {str(p.relative_to(MF)): p.read_text() for p in MF.rglob("*launcher.py") if "tests" not in p.parts}
for n in (
    "train_data_common.py",
    "audex/train_data.py",
    "rl/generator.py",
    "rl/moshi_policy.py",
    "rl/ppx_rollout.py",
    "rl/v2/launcher_v2.py",
):
    SRC[n] = (MF / n).read_text()
launchers = sorted(n for n in SRC if n.endswith("launcher.py"))
check(len(launchers) >= 5, f"{len(launchers)} launchers: {launchers}")
bad = wiring(SRC)
check(
    not bad, "every launcher configures it; the loss modules and RL log-prob paths use it" + (f": {bad}" if bad else "")
)
mut = dict(SRC)
mut["train_data_common.py"] = SRC["train_data_common.py"].replace(
    "stable_softmax.cross_entropy(", "F.cross_entropy(", 1
)
mut["rl/launcher.py"] = SRC["rl/launcher.py"].replace("stable_softmax.configure(cfg)", "pass")
mut["rl/v2/launcher_v2.py"] = SRC["rl/v2/launcher_v2.py"].replace("stable_softmax.ENABLED = bool(", "_ = bool(")
check(len(wiring(mut)) == 3, "CAN FAIL: a raw CE, a launcher without configure and the native runtime are flagged")

print("\nFAILURES: " + ("; ".join(FAILURES) if FAILURES else "none"))
sys.exit(1 if FAILURES else 0)
