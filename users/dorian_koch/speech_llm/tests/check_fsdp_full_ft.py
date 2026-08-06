"""Guard: the FSDP2 full-finetune plumbing is correct where it fails SILENTLY.

Full-parameter training is the one arm that changes what is being trained rather than how much of it
(backlog A11), so a bug here does not crash -- it produces a finished run and a plausible number. Two
places do that, and this drives the real ``moshi_family.fsdp_full_ft`` functions against a real
two-process shard rather than reasoning about them:

1. **Gradient norms over sharded tensors.** Every ``p.grad`` under FSDP2 is a ``DTensor`` holding this
   rank's slice. Only rank 0 writes ``metrics.train.jsonl``, so the naive per-tensor norm logs rank
   0's fraction of the truth -- steady, plausible, and the number the whole A2 per-module diagnostic
   is read off. The check asserts the sharded norm equals an unsharded reference AND that the naive
   form does *not*, so it cannot pass vacuously if the all-reduce is removed.

2. **State-dict semantics.** ``get_model_state_dict(full_state_dict=True, cpu_offload=True)`` returns
   the full dict on **rank 0 only**; other ranks get an EMPTY dict. That is load-bearing -- it is why
   only rank 0 writes the checkpoint -- and a torch upgrade that changed it would have every rank
   racing to write the same file. It is asserted rather than assumed, along with a
   save -> load -> step round trip through the optimizer state, because a checkpoint that restores
   weights but not Adam's moments resumes into a different experiment under the same name.

Runs on CPU with gloo (verified: FSDP2 shards fine without a GPU), so it is a login-node check like
every other guard here:

    CUDA_HOME=/usr .venv/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_fsdp_full_ft.py
"""

import os
import subprocess
import sys
from pathlib import Path as _FsPath

# abspath, NOT resolve(): entries under `recipe/` are symlinks into the `projects/` repos.
RECIPE = _FsPath(os.path.abspath(__file__)).parents[5]
FULL_DUPLEX = RECIPE / "speech_llm/full_duplex"

WORKER = r'''
import os, sys, tempfile
from pathlib import Path as _FsPath
import torch, torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

sys.path.insert(0, os.environ["FULL_DUPLEX"])
from moshi_family.fsdp_full_ft import (
    full_model_state_dict, full_optim_state_dict,
    load_full_model_state_dict, load_full_optim_state_dict,
    global_norm, global_sq_norm,
)

FAIL = []
def check(ok, msg):
    if not ok:
        FAIL.append(msg)

def build():
    torch.manual_seed(1234)
    return nn.Sequential(nn.Linear(16, 16, bias=False), nn.Linear(16, 16, bias=False),
                         nn.Linear(16, 16, bias=False))

def batch(i):
    g = torch.Generator().manual_seed(99 + i)
    return torch.randn(4, 16, generator=g)

rank = int(os.environ["RANK"])
dist.init_process_group("gloo")
mesh = init_device_mesh("cpu", (dist.get_world_size(),))

# ---- reference: the SAME model, unsharded, taking the same steps ------------------------------
ref = build()
ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-2)
ref(batch(0)).pow(2).sum().backward()
ref_grad_norm = float(torch.sqrt(torch.stack(
    [p.grad.detach().float().pow(2).sum() for p in ref.parameters()]).sum()))

# ---- the sharded model ------------------------------------------------------------------------
m = build()
for blk in m:
    fully_shard(blk, mesh=mesh)
fully_shard(m, mesh=mesh)
opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
m(batch(0)).pow(2).sum().backward()

grads = [p.grad for p in m.parameters() if p.grad is not None]
check(all(type(g).__name__ == "DTensor" for g in grads),
      "grads are not DTensors -- this check is not exercising the sharded path at all")

# 1a. the sharding-aware norm matches the unsharded reference
got = global_norm(grads)
check(abs(got - ref_grad_norm) < 1e-4,
      f"global_norm over sharded grads {got} != unsharded reference {ref_grad_norm}")

# 1b. ...and the NAIVE per-shard form does not, so 1a cannot pass with the all-reduce removed
naive = float(torch.sqrt(torch.stack(
    [g.to_local().detach().float().pow(2).sum() for g in grads]).sum()))
check(abs(naive - ref_grad_norm) > 1e-3,
      f"the per-shard norm {naive} already equals the true norm {ref_grad_norm} -- this model does "
      f"not actually shard, so check 1a proves nothing")

# 1c. a bucket mixing sharded and replicated tensors must be refused, not silently all-reduced
try:
    global_sq_norm([grads[0], torch.ones(4)])
    check(False, "a mixed sharded/replicated bucket was accepted -- the replicated part would be "
                 "scaled by the world size")
except AssertionError:
    pass

# ---- 2. state-dict semantics -------------------------------------------------------------------
opt.step()
sd = full_model_state_dict(m)
osd = full_optim_state_dict(m, opt)
# `set_model_state_dict` may convert the dict it is HANDED into DTensors in place, so anything we
# want to compare against later has to be snapshotted now. (Harmless for the launcher, which
# discards the dict it loads -- but it silently turned this check into a mixed-tensor error.)
sd_ref = {k: v.clone() for k, v in sd.items()}
# Zero before the next backward. Without this the reference model accumulates batch 0's gradient
# into batch 1's while the restored model does not, and the round-trip check below fails for a
# reason that has nothing to do with the optimizer state it is meant to be testing.
opt.zero_grad(set_to_none=True)
if rank == 0:
    check(len(sd) == 3, f"rank0 model state dict has {len(sd)} entries, expected 3")
    check(all(tuple(v.shape) == (16, 16) for v in sd.values()),
          "rank0 tensors are still sharded -- a partial checkpoint would be written")
    check(bool(osd.get("state")), "rank0 optimizer state dict is empty after a step")
else:
    check(len(sd) == 0,
          f"rank{rank} got a non-empty state dict ({len(sd)} entries). Only rank 0 writes the "
          f"checkpoint, so every rank holding data means either wasted memory or a write race")

# ---- 2b. the captured dict must not ALIAS the live optimizer ------------------------------------
# The bug this was written for: `get_optimizer_state_dict` returns AdamW's `step` counter by
# REFERENCE (the moment buffers are DTensors and get gathered into fresh tensors, but `step` is an
# unsharded CPU scalar and is passed straight through). The next optimizer step then increments the
# very tensor sitting in the dict we are about to save, producing a checkpoint whose weights say
# step N and whose bias correction says N+1 -- a 14% error on the next update, silent and permanent.
# Asserted separately from the round trip because a round-trip check only catches it when the
# capture and the write are separated by a step, which is a property of the CALLER, not of this API.
_step_at_capture = float(osd["state"][next(iter(osd["state"]))]["step"]) if rank == 0 else 0.0

# ---- 3. save -> load -> step round trip --------------------------------------------------------
# Continue the sharded run one more step, remembering where it lands; then rewind to the saved
# state, replay that step, and require the same weights. This is what proves the optimizer moments
# actually round-tripped: restoring weights alone reruns the step with fresh Adam state and lands
# somewhere else.
m(batch(1)).pow(2).sum().backward()
opt.step()
after = {k: v.clone() for k, v in full_model_state_dict(m).items()}
if rank == 0:
    now = float(osd["state"][next(iter(osd["state"]))]["step"])
    check(now == _step_at_capture,
          f"the saved optimizer state's step counter moved from {_step_at_capture} to {now} when the "
          f"LIVE optimizer stepped -- the captured dict aliases the optimizer, so any checkpoint "
          f"written after a subsequent step carries the wrong Adam bias correction")

obj = [sd, osd]
dist.broadcast_object_list(obj, src=0)
sd_all, osd_all = obj

m2 = build()
for blk in m2:
    fully_shard(blk, mesh=mesh)
fully_shard(m2, mesh=mesh)
opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-2)
load_full_model_state_dict(m2, sd_all)
load_full_optim_state_dict(m2, opt2, osd_all)

restored = full_model_state_dict(m2)
if rank == 0:
    worst = max((float((restored[k] - sd_ref[k]).abs().max()) for k in sd_ref), default=0.0)
    check(worst == 0.0, f"restored weights differ from the saved ones by up to {worst}")

# An independent run of the same two steps, touching no checkpoint at all. It is what BOTH the
# uninterrupted run and the resumed run must equal, and it is what tells them apart when they
# disagree: without it, a divergence says only "these differ", not "the resume is the broken one".
ctrl = build()
for blk in ctrl:
    fully_shard(blk, mesh=mesh)
fully_shard(ctrl, mesh=mesh)
ctrl_opt = torch.optim.AdamW(ctrl.parameters(), lr=1e-2)
for i in (0, 1):
    ctrl(batch(i)).pow(2).sum().backward()
    ctrl_opt.step()
    ctrl_opt.zero_grad(set_to_none=True)
control = {k: v.clone() for k, v in full_model_state_dict(ctrl).items()}
if rank == 0:
    drift = max((float((after[k] - control[k]).abs().max()) for k in control), default=0.0)
    check(drift == 0.0,
          f"the UNINTERRUPTED run drifted from an independent control by {drift} -- FSDP2 itself is "
          f"not reproducing, so nothing below can be attributed to the checkpoint")

m2(batch(1)).pow(2).sum().backward()
opt2.step()
replayed = full_model_state_dict(m2)
if rank == 0:
    worst = max((float((replayed[k] - after[k]).abs().max()) for k in after), default=0.0)
    check(worst < 1e-6,
          f"replaying the step from the checkpoint diverged by {worst} -- a resumed run is a "
          f"different experiment under the same name. If check 2b also failed, that is the cause; "
          f"if only this one failed, the divergence is downstream of the state dicts, and the "
          f"control assertion above already rules out FSDP2 non-determinism and the save side.")

# ---- 4. COLLECTIVE SYMMETRY --------------------------------------------------------------------
# The bug this catches: the per-module norms all-reduce under FSDP2, and they were gated on
# `logging_step`, which included `is_main`. So rank 0 ran one collective the others did not, got a
# step ahead, and NCCL sat on a 1-element ALLREDUCE for ten minutes before killing the job. On CPU
# The counts below name the asymmetry directly WHEN the mismatch is survivable. Verified 2026-08-06
# by reintroducing the bug: gloo blocks inside the very first mismatched all_reduce, so the check
# never reaches the comparison and fails as a HANG (the subprocess timeout in main()). Either way it
# fails -- but if this check ever times out rather than printing counts, an is_main-gated collective
# is the first thing to look for.
import moshi_family.train_loop as TL

_calls = {"n": 0}
_real_all_reduce = dist.all_reduce
def _counting_all_reduce(*a, **k):
    _calls["n"] += 1
    return _real_all_reduce(*a, **k)
dist.all_reduce = _counting_all_reduce
try:
    tiny = build()
    for blk in tiny:
        fully_shard(blk, mesh=mesh)
    fully_shard(tiny, mesh=mesh)
    topt = torch.optim.AdamW(tiny.parameters(), lr=1e-3)
    named = [(n, p) for n, p in tiny.named_parameters()]
    TL.run_training(
        optimizer=topt,
        base_lrs=[1e-3],
        group_names=("all",),
        trainable_params=[p for _, p in named],
        named_trainable=named,
        batch_iter=iter([torch.ones(2, 16)] * 6),
        loss_step=lambda b: tiny(b).pow(2).sum(),
        # A save_fn that GATHERS, like the real one -- the first version of this check passed a
        # no-op and so never exercised the save gate, which is exactly where the deadlock was.
        save_fn=lambda step, final: full_model_state_dict(tiny),
        cfg=TL.TrainConfig(
            max_steps=4, grad_accum=1, warmup_steps=1, save_every=2, log_every=1,
            sharded_callbacks=True,
        ),
        out_dir=_FsPath(tempfile.mkdtemp()),
        log=lambda *a, **k: None,
        is_main=(rank == 0),          # exactly the asymmetry that caused the deadlock
        start_step=0,
    )
finally:
    dist.all_reduce = _real_all_reduce

counts = [None] * dist.get_world_size()
dist.all_gather_object(counts, _calls["n"])
check(len(set(counts)) == 1,
      f"ranks issued DIFFERENT numbers of collectives during training: {counts}. Under NCCL that is "
      f"not an error, it is a HANG -- the ranks that ran fewer wait forever and the job dies on a "
      f"watchdog timeout ~10 min later. Something inside the step is gated on is_main but performs "
      f"a collective; compute it on every rank and let only rank 0 WRITE the result.")

payload = [FAIL]
gathered = [None] * dist.get_world_size()
dist.all_gather_object(gathered, payload)
if rank == 0:
    flat = [f for g in gathered for f in g[0]]
    if flat:
        print("FAILED:")
        for f in flat:
            print("  -", f)
    else:
        print("WORKER OK")
dist.destroy_process_group()
sys.exit(1 if (rank == 0 and any(g[0] for g in gathered)) else 0)
'''


def check_shardable():
    """Every StreamingModule must accept a __class__ swap, or fully_shard cannot wrap it.

    FSDP2 shards by building a dynamic ``FSDP<YourClass>`` and assigning it to
    ``module.__class__``. CPython refuses that when the class adds its own ``__dict__`` slot instead
    of inheriting ``nn.Module``'s -- which is what happens when ``abc.ABC`` precedes ``nn.Module``
    in the bases. Measured: ``(nn.Module)`` fine, ``(nn.Module, ABC)`` fine, ``(ABC, nn.Module)``
    refused.

    This is worth its own check because of HOW it failed: nothing is wrong at import, at
    construction, or in any single-GPU path. It surfaced only after a 4-GPU job had allocated its
    nodes and spent six minutes loading 15 GB of weights, and it would come back the moment anyone
    reorders those bases for tidiness.
    """
    sys.path.insert(0, str(FULL_DUPLEX))
    import torch.nn as nn
    from torch.distributed.fsdp import FSDPModule

    from moshi_family.modules.streaming import StreamingContainer, StreamingModule
    from moshi_family.modules.transformer import (
        StreamingTransformer,
        StreamingTransformerLayer,
    )

    bad = []
    for cls in (StreamingModule, StreamingContainer, StreamingTransformer, StreamingTransformerLayer):
        assert issubclass(cls, nn.Module), cls
        # Layout compatibility is inherited, so a concrete stand-in proves it for an abstract base --
        # and a concrete subclass is what fully_shard actually meets in the module tree anyway.
        target = cls
        if getattr(cls, "__abstractmethods__", None):
            target = type(
                f"_Concrete{cls.__name__}",
                (cls,),
                {m: (lambda self, *a, **k: None) for m in cls.__abstractmethods__},
            )
        # No constructor arguments and no GPU: only the class layout is under test.
        obj = nn.Module.__new__(target)
        try:
            obj.__class__ = type(f"FSDP{target.__name__}", (FSDPModule, target), {})
        except TypeError as err:
            bad.append(f"{cls.__name__}: {err}")
    if bad:
        print("FAILED: these classes cannot be sharded by fully_shard:", file=sys.stderr)
        for b in bad:
            print("  -", b, file=sys.stderr)
        print("  fix: put nn.Module BEFORE abc.ABC in the bases", file=sys.stderr)
        raise SystemExit(1)
    print("[ok] shardable classes  every StreamingModule accepts fully_shard's __class__ swap")


def main():
    check_shardable()
    worker = _FsPath(os.environ.get("TMPDIR", "/tmp")) / "_check_fsdp_worker.py"
    worker.write_text(WORKER)
    env = dict(os.environ, FULL_DUPLEX=str(FULL_DUPLEX), CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node=2",
         "--master-port=29851", str(worker)],
        env=env, capture_output=True, text=True, timeout=600,
    )
    out = proc.stdout + proc.stderr
    if "WORKER OK" not in out:
        print(out[-4000:], file=sys.stderr)
        raise SystemExit(1)
    print("[ok] sharded grad norms   global_norm == unsharded reference; the per-shard form does not")
    print("[ok] mixed bucket refused a sharded+replicated bucket raises instead of mis-scaling")
    print("[ok] state dict on rank0  full unsharded tensors on rank 0, EMPTY on every other rank")
    print("[ok] no aliasing         a later optimizer step does not rewrite the captured checkpoint")
    print("[ok] collective symmetry every rank issues the same number of collectives per step")
    print("[ok] resume round trip    weights exact, and replaying a step from the checkpoint lands "
          "where the uninterrupted run did")
    print("\nFSDP full-FT plumbing holds -- a sharded run's norms and checkpoints are the real ones")


main()
