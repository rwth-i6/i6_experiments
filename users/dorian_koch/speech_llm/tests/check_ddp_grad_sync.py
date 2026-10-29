"""Guard: a multi-GPU run must actually synchronise gradients.

The bug this exists for: ``loss_step`` called ``core.forward(codes)`` -- the module INSIDE the
``DistributedDataParallel`` wrapper. DDP synchronises gradients from its own ``forward``, which arms
the reducer for the backward pass; bypassing it means the reducer is never armed and **no all-reduce
happens**. Every 4-GPU LoRA arm was therefore training four independent replicas, each on its own
data shard, and saving rank 0's -- so a run labelled "effective batch 8 x 4 ranks = 32" was really
effective batch 8 on a quarter of the data.

Nothing about that fails. The loss curve is smooth, the run finishes, the checkpoint loads, and the
benchmark returns a plausible number. It is only visible if you go looking for it, which is what
makes it worth a standing check.

Two halves, because either alone is weak:

1. **Behavioural.** Drive real DDP over two gloo processes with DIFFERENT data per rank and assert
   that ``ddp(x)`` leaves the ranks' gradients identical while ``inner.forward(x)`` does not. The
   second assertion is what stops this check from passing vacuously on a build where DDP happened to
   synchronise anyway.

2. **Source.** Assert the launcher does not call ``.forward(`` on a module at all. The behavioural
   half proves the property of DDP; only this half proves *our* code still relies on it. They are
   different claims and the bug lived in the gap between them.

Widened 2026-09-23 (code-map §9, bug 3): PersonaPlex's LM has no ``forward``, and its launcher wrapped
the LM in DDP and then called ``core.forward_train`` -- the same bypass, in a launcher the old
``moshi_family/*_launcher.py`` glob never scanned. Now:

  * the behavioural half also drives the REAL model paths: tiny base-Moshi and PersonaPlex ``LMModel``s
    + LoRA + activation checkpointing, PersonaPlex through the launcher's own ``_ForwardTrain`` adapter,
    under DDP as the launchers build it for new arms -- synchronised through the wrapper, and NOT via
    ``core.forward_train``;
  * 🔴 found while widening it: **``DDP(static_graph=True)`` never all-reduces a module that returns a
    dataclass** (the ``LMOutput``): DDP finds no tensor in the output, its static-graph reducer is never
    armed, and every rank keeps its own gradients (114/114 base-Moshi LoRA grads differing; torch 2.11
    and the job venv's 2.12). The toy ``nn.Linear`` above returns a tensor, which is why this check
    passed while every multi-GPU base-Moshi LoRA arm since 2026-07-31 ran ``static_graph=True``. The
    measurement is printed every run (not asserted: a torch that fixes it must not fail the check);
  * LoRA + our (non-reentrant) activation checkpointing under DDP WITHOUT ``static_graph`` synchronises
    and does not raise -- the Moshi launcher's old comment that static_graph is "REQUIRED" for
    checkpointing predates ``use_reentrant=False``;
  * (2026-09-23, p5) the PersonaPlex FULL finetune under FSDP2, sharded exactly as the launcher shards it
    (``shard_for_full_ft(core, root=_ForwardTrain(core), master_dtype=float32)``): every rank's
    reduce-scattered gradient is the mean of the ranks' own full-model gradients on their own data, and
    ``core.forward_train`` on the sharded model does not train (it raises on the still-sharded embeddings);
  * the source half scans every ``moshi_family/**/*launcher.py``, rejects any ``.forward*()`` method call
    inside a ``loss_step`` (``forward_train`` included) and any literal ``static_graph=True`` (base
    Moshi's is the config key ``ddp_static_graph``: True only for arms rendered before 2026-09-23), with
    mutations of the real launchers showing both rules fire.

Runs on CPU with gloo, no GPU:

    CUDA_HOME=/usr .venv/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_ddp_grad_sync.py
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path as _FsPath

# abspath, NOT resolve(): entries under `recipe/` are symlinks into the `projects/` repos.
RECIPE = _FsPath(os.path.abspath(__file__)).parents[5]
FULL_DUPLEX = RECIPE / "speech_llm/full_duplex"

WORKER = r'''
import os, sys
import torch, torch.nn as nn
import torch.distributed as dist

rank = int(os.environ["RANK"])
dist.init_process_group("gloo")
world = dist.get_world_size()

def build():
    torch.manual_seed(7)
    return nn.Linear(8, 8, bias=False)

def spread(use_ddp_forward):
    m = build()
    ddp = nn.parallel.DistributedDataParallel(m, static_graph=True)
    # Different data per rank: if gradients are synchronised they must agree afterwards, and if
    # they are not, they cannot -- so the test has no way to pass by accident.
    x = torch.full((2, 8), float(rank + 1))
    (ddp(x) if use_ddp_forward else m.forward(x)).pow(2).sum().backward()
    g = next(m.parameters()).grad.detach().clone()
    got = [torch.zeros_like(g) for _ in range(world)]
    dist.all_gather(got, g)
    return max(float((got[0] - o).abs().max()) for o in got[1:])

def gather_spread(params):
    g = torch.cat([p.grad.detach().flatten() for p in params if p.grad is not None])
    got = [torch.zeros_like(g) for _ in range(world)]
    dist.all_gather(got, g)
    return max(float((got[0] - o).abs().max()) for o in got[1:])

sys.path.insert(0, os.environ["FULL_DUPLEX"])
from moshi_family.build import LoraConfig, apply_lora_to
from moshi_family.modules.lora import replace_all_linear_with_lora
from moshi_family.modules.transformer import StreamingTransformer
from moshi_family.models import loaders as moshi_loaders
from moshi_family.models.lm import LMModel as MoshiLM
from moshi_family.moshi_train_data import moshi_loss
from moshi_family.personaplex.finetune_launcher import _ForwardTrain
from moshi_family.personaplex.models import loaders
from moshi_family.personaplex.models.lm import LMModel
from moshi_family.personaplex.train_data import personaplex_loss
from moshi_family.train_loop import is_lora_param

TINY = dict(dim=32, num_heads=2, num_layers=2, hidden_scale=2.0, text_card=100, context=64,
            depformer_dim=16, depformer_dim_feedforward=32, depformer_num_heads=2, depformer_num_layers=1)

def lm_spread(kind, through_wrapper, static_graph):
    """A launcher's training path on a tiny real LM + LoRA + checkpointing, different data per rank.
    kind: "moshi" (models/lm.py, called directly as the Moshi launcher does) or "personaplex" (through
    the launcher's _ForwardTrain adapter)."""
    torch.manual_seed(0)
    if kind == "moshi":
        kw = dict(moshi_loaders._lm_kwargs)
        kw.update(TINY)
        core = MoshiLM(device="cpu", dtype=torch.float32, **kw)
    else:
        kw = dict(loaders._lm_kwargs)
        kw.update(TINY, dep_q=16)
        core = LMModel(device="cpu", dtype=torch.float32, **kw)
    apply_lora_to(core, LoraConfig(rank=2, scaling=2.0), dtype=torch.float32)
    with torch.no_grad():
        for n, p in core.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)  # non-zero B, so lora_A has a gradient too
    for n, p in core.named_parameters():
        p.requires_grad = is_lora_param(n)
    core.transformer.checkpointing = True
    core.train()
    model = nn.parallel.DistributedDataParallel(
        core if kind == "moshi" else _ForwardTrain(core), static_graph=static_graph
    )
    g = torch.Generator().manual_seed(100 + rank)
    codes = torch.randint(0, 50, (1, 17, 20), generator=g)
    codes[:, 0] = torch.randint(0, 100, (1, 20), generator=g)
    mask = torch.ones(1, 20, dtype=torch.bool)
    if kind == "moshi":
        out = model(codes) if through_wrapper else core.forward(codes)
        moshi_loss(out, codes, mask, dep_q=core.dep_q).backward()
    else:
        out = model(codes) if through_wrapper else core.forward_train(codes)
        personaplex_loss(out, codes, mask, dep_q=core.dep_q).backward()
    return gather_spread(core.parameters())

def ppx_fsdp_spread(through_wrapper):
    """PersonaPlex FULL finetune under FSDP2 as the launcher shards it (fp32 masters, _ForwardTrain as the root;
    fp32 compute here so the comparison is exact), different data per rank. Synchronised means: the gathered
    gradient == the MEAN of the ranks' own unsharded gradients. -> max deviation (inf if the path raises)."""
    from torch.distributed.device_mesh import init_device_mesh
    from moshi_family.fsdp_full_ft import shard_for_full_ft

    kw = dict(loaders._lm_kwargs)
    kw.update(TINY, dep_q=16)

    def fresh():
        torch.manual_seed(0)
        m = LMModel(device="cpu", dtype=torch.float32, **kw)
        m.transformer.checkpointing = True
        m.train()
        return m

    g = torch.Generator().manual_seed(100 + rank)
    codes = torch.randint(0, 50, (1, 17, 20), generator=g)
    codes[:, 0] = torch.randint(0, 100, (1, 20), generator=g)
    mask = torch.ones(1, 20, dtype=torch.bool)
    ref = fresh()
    personaplex_loss(_ForwardTrain(ref)(codes), codes, mask, dep_q=16).backward()
    local = torch.cat([p.grad.detach().flatten() for p in ref.parameters()])
    parts = [torch.zeros_like(local) for _ in range(world)]
    dist.all_gather(parts, local)
    mean = sum(parts) / world
    core = fresh()
    w = _ForwardTrain(core)
    shard_for_full_ft(core, root=w, master_dtype=torch.float32, param_dtype=torch.float32,
                      mesh=init_device_mesh("cpu", (world,)))
    try:
        out = w(codes) if through_wrapper else core.forward_train(codes)
        personaplex_loss(out, codes, mask, dep_q=16).backward()
    except Exception:
        return float("inf")
    full = torch.cat([p.grad.full_tensor().detach().flatten() for p in core.parameters()])
    return float((full - mean).abs().max())

def ckpt_no_static_graph_spread():
    """LoRA + activation checkpointing under DDP WITHOUT static_graph, as MoshiRAG wraps it; 3 steps."""
    torch.manual_seed(0)
    tr = StreamingTransformer(d_model=32, num_heads=2, num_layers=2, dim_feedforward=64, causal=True,
                              context=64, positional_embedding="rope", checkpointing=True)
    for p in tr.parameters():
        p.requires_grad = False
    replace_all_linear_with_lora(tr, rank=4, scaling=2.0)
    for n, p in tr.named_parameters():
        p.requires_grad = "lora_" in n
        if "lora_B" in n:
            nn.init.normal_(p, std=0.1)
    ddp = nn.parallel.DistributedDataParallel(tr, static_graph=False)
    worst = 0.0
    for _ in range(3):
        ddp(torch.randn(2, 10, 32) * (rank + 1)).pow(2).mean().backward()
        worst = max(worst, gather_spread(tr.parameters()))
        for p in tr.parameters():
            p.grad = None
    return worst

wrapped = spread(True)
bypassed = spread(False)
# As the launchers build DDP for new arms: static_graph off (base Moshi via ddp_static_graph=False).
moshi_wrapped = lm_spread("moshi", True, static_graph=False)
ppx_wrapped = lm_spread("personaplex", True, static_graph=False)
ppx_bypassed = lm_spread("personaplex", False, static_graph=False)
ppx_fsdp_wrapped = ppx_fsdp_spread(True)
ppx_fsdp_bypassed = ppx_fsdp_spread(False)
# The trap (reported, not asserted): what every multi-GPU base-Moshi LoRA arm before 2026-09-23 ran.
moshi_static = lm_spread("moshi", True, static_graph=True)
try:
    ckpt_nosg = ckpt_no_static_graph_spread()
except Exception as e:  # the failure the Moshi launcher's comment predicts
    ckpt_nosg = f"{type(e).__name__}: {e}"
fail = []
if wrapped > 1e-9:
    fail.append(f"ddp(x) left ranks disagreeing by {wrapped} -- DDP is not synchronising at all")
if bypassed < 1e-6:
    fail.append(f"inner.forward(x) ALSO synchronised (spread {bypassed}) -- this check cannot "
                f"detect the bypass on this build, so it proves nothing")
if moshi_wrapped > 1e-6:
    fail.append(f"base-Moshi LM under DDP (static_graph off) left ranks disagreeing by {moshi_wrapped}")
if ppx_wrapped > 1e-6:
    fail.append(f"PersonaPlex through _ForwardTrain+DDP left ranks disagreeing by {ppx_wrapped}")
if ppx_bypassed < 1e-6:
    fail.append(f"PersonaPlex core.forward_train ALSO synchronised ({ppx_bypassed}) -- cannot detect bug 3")
if ppx_fsdp_wrapped > 1e-5:
    fail.append(f"PersonaPlex full FT under FSDP2 through _ForwardTrain: gradient != rank mean by {ppx_fsdp_wrapped}")
if ppx_fsdp_bypassed < 1e-6:
    fail.append(f"PersonaPlex FSDP2 core.forward_train ALSO gives the synchronised gradient ({ppx_fsdp_bypassed})")
if not isinstance(ckpt_nosg, float) or ckpt_nosg > 1e-6:
    fail.append(f"LoRA + checkpointing under DDP without static_graph: {ckpt_nosg}")
if rank == 0:
    print(f"WRAPPED {wrapped}")
    print(f"BYPASSED {bypassed}")
    print(f"MOSHI_WRAPPED {moshi_wrapped}")
    print(f"MOSHI_STATIC_GRAPH_TRUE {moshi_static}")
    print(f"PPX_WRAPPED {ppx_wrapped}")
    print(f"PPX_BYPASSED {ppx_bypassed}")
    print(f"PPX_FSDP_WRAPPED {ppx_fsdp_wrapped}")
    print(f"PPX_FSDP_BYPASSED {ppx_fsdp_bypassed}")
    print(f"CKPT_NO_STATIC_GRAPH {ckpt_nosg}")
    print("WORKER FAILED: " + " | ".join(fail) if fail else "WORKER OK")
dist.destroy_process_group()
sys.exit(1 if (rank == 0 and fail) else 0)
'''


def main():
    worker = _FsPath(os.environ.get("TMPDIR", "/tmp")) / "_check_ddp_worker.py"
    worker.write_text(WORKER)
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node=2", "--master-port=29893", str(worker)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", FULL_DUPLEX=str(FULL_DUPLEX)),
        capture_output=True,
        text=True,
        timeout=600,
    )
    out = proc.stdout + proc.stderr
    if "WORKER OK" not in out:
        print(out[-3000:], file=sys.stderr)
        raise SystemExit(1)
    print("[ok] ddp synchronises   ddp(x) leaves ranks identical; inner.forward(x) does NOT")
    print("[ok] base Moshi / PersonaPlex LMs + LoRA + ckpt under DDP synchronise; core.forward_train does NOT")
    print(
        "[ok] PersonaPlex full FT under FSDP2 (fp32 masters, _ForwardTrain root) = the rank mean; the bypass does NOT"
    )
    trap = next((ln for ln in out.splitlines() if ln.startswith("MOSHI_STATIC_GRAPH_TRUE")), "")
    if trap and float(trap.split()[1]) > 1e-6:
        print(
            "[!!] static_graph=True + LMOutput does NOT synchronise on this torch (ranks differ by "
            f"{float(trap.split()[1]):.3g}) -- keep ddp_static_graph False for multi-GPU arms"
        )
    print("[ok] ckpt w/o static_graph  LoRA + activation checkpointing under plain DDP synchronises")
    for line in out.splitlines():
        if line.startswith(("WRAPPED", "BYPASSED", "MOSHI_", "PPX_", "CKPT_")):
            print("       " + line)

    # ---- source half: our launchers must go through the wrapper -------------------------------
    launchers = sorted(p for p in FULL_DUPLEX.glob("moshi_family/**/*launcher.py") if "__pycache__" not in p.parts)
    assert len(launchers) >= 5, f"expected >= 5 launchers under {FULL_DUPLEX}/moshi_family, got {launchers}"

    def offenders_in(name, src):
        found = []
        for n, line in enumerate(src.splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"\.forward\s*\(", code):
                found.append(f"{name}:{n}: {line.strip()}")
        tree = ast.parse(src)
        # A literal static_graph=True on a DDP wrapper: it never synchronises an LMOutput-returning LM.
        for c in ast.walk(tree):
            if (
                isinstance(c, ast.Call)
                and getattr(c.func, "attr", getattr(c.func, "id", "")) == "DistributedDataParallel"
            ):
                for k in c.keywords:
                    # anything but a literal False: a config-driven value defaulted to True for old-epoch arms
                    if k.arg == "static_graph" and not (isinstance(k.value, ast.Constant) and k.value.value is False):
                        found.append(f"{name}:{c.lineno}: DistributedDataParallel(static_graph={ast.unparse(k.value)})")
        # Any .forward*() METHOD call inside loss_step (forward_train included): the bug-3 shape.
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef) and fn.name == "loss_step":
                for c in ast.walk(fn):
                    if (
                        isinstance(c, ast.Call)
                        and isinstance(c.func, ast.Attribute)
                        and c.func.attr.startswith("forward")
                    ):
                        found.append(f"{name}:{c.lineno}: loss_step calls .{c.func.attr}()")
        return found

    offenders = [o for p in launchers for o in offenders_in(str(p.relative_to(FULL_DUPLEX)), p.read_text())]
    if offenders:
        print("FAILED: a launcher calls a module's forward method directly instead of its __call__:", file=sys.stderr)
        for o in offenders:
            print("  -", o, file=sys.stderr)
        print(
            "  Under DDP that skips gradient synchronisation; under FSDP2 it skips the pre-forward all-gather.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"[ok] no direct .forward   {len(launchers)} launcher(s) (moshi_family/**) invoke modules via __call__")
    ppx = FULL_DUPLEX / "moshi_family/personaplex/finetune_launcher.py"
    src = ppx.read_text()
    anchor = "            out = model(codes, **fw)\n"
    assert anchor in src, "mutation anchor moved: update the source can-fail check"
    mutated = offenders_in("personaplex", src.replace(anchor, "            out = core.forward_train(codes, **fw)\n", 1))
    assert mutated, "the source half does not flag loss_step calling core.forward_train -- it cannot catch bug 3"
    print(f"[ok] source can fail      core.forward_train in loss_step is flagged ({mutated[0].split(': ', 1)[1]})")
    moshi_src = (FULL_DUPLEX / "moshi_family/moshi_finetune_launcher.py").read_text()
    anchor = "static_graph=False)"
    assert moshi_src.count(anchor) == 1, "mutation anchor moved: update the static_graph can-fail check"
    mutated = offenders_in("moshi", moshi_src.replace(anchor, "static_graph=ddp_static_graph)", 1))
    assert mutated, "the source half does not flag a non-literal static_graph"
    print(
        f"[ok] source can fail      a static_graph other than a literal False is flagged ({mutated[0].split(': ', 1)[1]})"
    )

    print("\nmulti-GPU runs synchronise gradients -- an N-GPU arm is one model, not N replicas")


main()
