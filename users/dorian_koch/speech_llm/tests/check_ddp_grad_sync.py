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

Runs on CPU with gloo, no GPU:

    CUDA_HOME=/usr .venv/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_ddp_grad_sync.py
"""

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

wrapped = spread(True)
bypassed = spread(False)
fail = []
if wrapped > 1e-9:
    fail.append(f"ddp(x) left ranks disagreeing by {wrapped} -- DDP is not synchronising at all")
if bypassed < 1e-6:
    fail.append(f"inner.forward(x) ALSO synchronised (spread {bypassed}) -- this check cannot "
                f"detect the bypass on this build, so it proves nothing")
if rank == 0:
    print(f"WRAPPED {wrapped}")
    print(f"BYPASSED {bypassed}")
    print("WORKER FAILED: " + " | ".join(fail) if fail else "WORKER OK")
dist.destroy_process_group()
sys.exit(1 if (rank == 0 and fail) else 0)
'''


def main():
    worker = _FsPath(os.environ.get("TMPDIR", "/tmp")) / "_check_ddp_worker.py"
    worker.write_text(WORKER)
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node=2",
         "--master-port=29893", str(worker)],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1"),
        capture_output=True, text=True, timeout=600,
    )
    out = proc.stdout + proc.stderr
    if "WORKER OK" not in out:
        print(out[-3000:], file=sys.stderr)
        raise SystemExit(1)
    print("[ok] ddp synchronises   ddp(x) leaves ranks identical; inner.forward(x) does NOT")

    # ---- source half: our launchers must go through the wrapper -------------------------------
    launchers = sorted(FULL_DUPLEX.glob("moshi_family/*_launcher.py"))
    assert launchers, f"no launchers found under {FULL_DUPLEX}/moshi_family"
    offenders = []
    for path in launchers:
        for n, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"\.forward\s*\(", code):
                offenders.append(f"{path.name}:{n}: {line.strip()}")
    if offenders:
        print("FAILED: a launcher calls .forward() directly instead of the module's __call__:",
              file=sys.stderr)
        for o in offenders:
            print("  -", o, file=sys.stderr)
        print("  Under DDP that skips gradient synchronisation; under FSDP2 it skips the "
              "pre-forward all-gather.", file=sys.stderr)
        raise SystemExit(1)
    print(f"[ok] no direct .forward   {len(launchers)} launcher(s) invoke modules via __call__")

    print("\nmulti-GPU runs synchronise gradients -- an N-GPU arm is one model, not N replicas")


main()
