"""Guard: the PersonaPlex FULL finetune under FSDP2 trains the same model as one process would (p5, 2026-09-23).

``moshi_family/personaplex/finetune_launcher.py`` with ``train_scope="full"`` on more than one rank shards the
LM with ``fsdp_full_ft.shard_for_full_ft(core, root=_ForwardTrain(core), master_dtype=float32)`` and saves /
resumes with ``save_sharded_checkpoint`` / ``resume_sharded_checkpoint``. Every failure below would be
silent -- a finished run, a loadable checkpoint, a plausible loss curve -- so each is driven for real on a
tiny real PersonaPlex ``LMModel`` (fast path on, activation checkpointing on, the real ``personaplex_loss``)
over two gloo processes with DIFFERENT data per rank, against a single-process reference on the same data:

  [1] fp32 compute: the sharded model's loss and every full gradient == the single-process average over the
      two ranks' batches (exact up to fp32 reduction order). Proves the sharding, the root wrapper and the
      gradient averaging.
  [2] as launched (fp32 masters, bf16 compute, fp32 reduce): parameters, gradients and AdamW state are fp32
      DTensors; loss and gradients are no farther from the fp32 reference than a plain bf16 single process
      (the parity criterion).
  [3] CAN FAIL: calling ``core.forward_train`` instead of the root wrapper does not train the same model (it
      raises on the still-sharded embeddings, or differs).
  [4] CAN FAIL: without ``master_dtype`` (bf16 sharded params) an AdamW step at lr 1e-6 moves almost no
      weights; with fp32 masters it moves ~all of them (the p0 bf16-update bug, for full FT).
  [5] checkpoint round trip: resume restores step, fp32 weights and AdamW moments -- replaying the next step
      lands bit-exactly where the uninterrupted run did (a fresh model does not); ``model.safetensors`` holds
      exactly the base LM's keys in bf16 == the masters, and loads into an unwrapped model with no missing or
      unexpected key; an older resume state is pruned, its weights kept.
  [0] ``loaders.get_moshi_lm(device="cuda:<k>")`` hands safetensors that DEVICE. It passed ``dev.type`` ("cuda" =
      cuda:0), so every rank of the p5 smoke loaded the 16 GB checkpoint onto GPU 0 and kept it cached there (OOM at
      step 1, 2026-09-23). CAN FAIL: the pre-fix source reads "cuda".
  [6] the real ``run_training``, preempted after a checkpoint and resumed: every rank restores ITS OWN shard of
      theta_0 (``weight_delta_baseline.rank<k>.pt``), and the logged ``weight_delta_by_module`` == the
      reference from full tensors. CAN FAIL: rank 0's shard on every rank gives a different number.

CPU, gloo, no GPU, ~1-2 min (setup ``.venv``):
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_ppx_fsdp.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path as _FsPath

# abspath, NOT resolve(): entries under `recipe/` are symlinks into the `projects/` repos.
RECIPE = _FsPath(os.path.abspath(__file__)).parents[5]
FULL_DUPLEX = RECIPE / "speech_llm/full_duplex"

WORKER = r'''
import json, os, sys
from pathlib import Path as _P
import torch, torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

sys.path.insert(0, os.environ["FULL_DUPLEX"])
from moshi_family.fsdp_full_ft import (
    RESUME_DIR, RESUME_MARKER, complete_resume_states, global_norm, like_sharded,
    resume_sharded_checkpoint, save_sharded_checkpoint, shard_for_full_ft,
)
from moshi_family.personaplex.finetune_launcher import _ForwardTrain
from moshi_family.personaplex.models import loaders
from moshi_family.personaplex.models.lm import LMModel
from moshi_family.personaplex.train_data import personaplex_loss
from moshi_family.train_loop import TrainConfig, classify_param, run_training, split_depth_temporal_params
from moshi_family.train_perf import apply_train_perf

rank = int(os.environ["RANK"])
dist.init_process_group("gloo")
world = dist.get_world_size()
mesh = init_device_mesh("cpu", (world,))
ROOT = _P(os.environ["CKPT_ROOT"])
FAIL, INFO = [], []

def check(ok, msg):
    if not ok:
        FAIL.append(msg)

TINY = dict(dim=32, num_heads=2, num_layers=2, hidden_scale=2.0, text_card=100, context=64,
            depformer_dim=16, depformer_dim_feedforward=32, depformer_num_heads=2, depformer_num_layers=1)

def build():
    """What the launcher holds before sharding: the LM as loaded (bf16), full scope, fast path, ckpt on."""
    torch.manual_seed(0)
    kw = dict(loaders._lm_kwargs)
    kw.update(TINY, dep_q=16)
    lm = LMModel(device="cpu", dtype=torch.bfloat16, **kw)
    lm.to(dtype=torch.bfloat16)  # as loaders.get_moshi_lm does (some submodules construct in fp32)
    with torch.no_grad():
        for p in lm.parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.02)  # |w| ~ 1e-2 like the real checkpoint (the bf16-rounding regime of [4])
    apply_train_perf(lm, contiguous_step_inputs=True, fused_causal_attention=True)
    for p in lm.parameters():
        p.requires_grad = True
    lm.transformer.checkpointing = True
    lm.train()
    return lm

def batch(r, step=0):
    g = torch.Generator().manual_seed(1000 * step + 100 + r)
    B, T = 2, 20
    codes = torch.randint(0, 50, (B, 17, T), generator=g)
    codes[:, 0] = torch.randint(0, 100, (B, T), generator=g)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, :4] = False     # the system prompt
    mask[1, 15:] = False    # right padding
    return codes, mask

def loss_of(model, b):
    codes, mask = b
    out = model(codes, depformer_frame_mask=mask)
    return personaplex_loss(out, codes, mask, dep_q=16, weighted_mean=True, text_epad_id=0, return_components=True)

def reference(fp32):
    """Single process, no FSDP: the average over the ranks' batches (what DDP/FSDP reduce to)."""
    lm = build()
    if fp32:
        with torch.no_grad():
            for p in lm.parameters():
                p.data = p.data.float()
    w = _ForwardTrain(lm)
    losses = []
    for r in range(world):
        loss, _ = loss_of(w, batch(r))
        (loss / world).backward()
        losses.append(float(loss.detach()))
    return losses, {n: p.grad.detach().float().clone() for n, p in lm.named_parameters()}

def sharded(master=torch.float32, param_dtype=torch.bfloat16):
    lm = build()
    w = _ForwardTrain(lm)
    shard_for_full_ft(lm, root=w, master_dtype=master, param_dtype=param_dtype, mesh=mesh)
    return lm, w

def full_grads(lm):
    return {n: p.grad.full_tensor().detach().float().clone() for n, p in lm.named_parameters()}

def agg_err(g, ref):
    num = torch.sqrt(sum(((g[n] - ref[n]) ** 2).sum() for n in ref))
    den = torch.sqrt(sum((ref[n] ** 2).sum() for n in ref))
    return float(num / den)

ref32_losses, ref32 = reference(fp32=True)
ref16_losses, ref16 = reference(fp32=False)

# ---- [1] fp32 compute: exact ------------------------------------------------------------------------
lm, w = sharded(param_dtype=torch.float32)
loss, _ = loss_of(w, batch(rank))
loss.backward()
g1 = full_grads(lm)
check(set(g1) == set(ref32), f"[1] sharded param names differ from the unwrapped LM: {sorted(set(g1) ^ set(ref32))[:4]}")
d_loss = abs(float(loss) - ref32_losses[rank])
worst = max(float((g1[n] - ref32[n]).norm() / ref32[n].norm().clamp_min(1e-12)) for n in ref32)
check(d_loss < 1e-5, f"[1] fp32 FSDP loss {float(loss)} != single-process {ref32_losses[rank]}")
check(worst < 1e-4, f"[1] fp32 FSDP gradients differ from the single-process average (worst tensor rel {worst:.3g})")
INFO.append(f"[1] fp32 compute: |dloss| {d_loss:.2e}, worst per-tensor grad rel err {worst:.2e} over {len(ref32)} tensors")

# ---- [2] as launched: fp32 masters, bf16 compute ------------------------------------------------------
lm, w = sharded()
params = list(lm.parameters())
check(all(type(p).__name__ == "DTensor" for p in params), "[2] parameters are not DTensors -- not sharded")
check(all(p.dtype == torch.float32 for p in params), "[2] sharded parameters are not fp32 masters")
loss, _ = loss_of(w, batch(rank))
loss.backward()
check(all(p.grad is not None and p.grad.dtype == torch.float32 for p in params), "[2] gradients are not fp32")
g2 = full_grads(lm)
e_fsdp, e_bf16 = agg_err(g2, ref32), agg_err(ref16, ref32)
rel_loss = abs(float(loss) - ref32_losses[rank]) / abs(ref32_losses[rank])
rel_loss16 = abs(ref16_losses[rank] - ref32_losses[rank]) / abs(ref32_losses[rank])
check(e_fsdp <= 1.5 * e_bf16 + 1e-3,
      f"[2] bf16-compute FSDP gradient error vs fp32 {e_fsdp:.3g} > 1.5x a plain bf16 process's {e_bf16:.3g}")
check(rel_loss <= 1.5 * rel_loss16 + 1e-3,
      f"[2] bf16-compute FSDP loss error {rel_loss:.3g} > 1.5x a plain bf16 process's {rel_loss16:.3g}")
opt = torch.optim.AdamW([{"params": params}], lr=1e-6, betas=(0.9, 0.95), weight_decay=0.0)
opt.step()
st = [v for s_ in opt.state.values() for k, v in s_.items() if k != "step" and torch.is_tensor(v)]
check(st and all(v.dtype == torch.float32 for v in st), "[2] AdamW moments are not fp32")
INFO.append(f"[2] bf16 compute: grad err vs fp32 {e_fsdp:.4f} (plain bf16 process {e_bf16:.4f}), "
            f"loss rel err {rel_loss:.2e} (bf16 process {rel_loss16:.2e})")

# ---- [3] CAN FAIL: bypassing the root wrapper -----------------------------------------------------------
lm, w = sharded(param_dtype=torch.float32)
codes, mask = batch(rank)
try:
    out = lm.forward_train(codes, depformer_frame_mask=mask)
    l3 = personaplex_loss(out, codes, mask, dep_q=16, weighted_mean=True, text_epad_id=0)
    l3.backward()
    g3 = full_grads(lm)
    bypass = f"ran; worst rel grad diff {max(float((g3[n] - ref32[n]).norm() / ref32[n].norm().clamp_min(1e-12)) for n in ref32):.3g}"
    check(any(float((g3[n] - ref32[n]).norm()) > 1e-4 * float(ref32[n].norm()) for n in ref32),
          "[3] core.forward_train on the sharded model ALSO trains the reference model -- the check cannot see a bypass")
except Exception as e:
    bypass = f"raises {type(e).__name__}"
INFO.append(f"[3] core.forward_train bypassing the root wrapper: {bypass}")

# ---- [4] CAN FAIL: fp32 masters vs bf16 sharded params ----------------------------------------------------
def moved_fraction(master):
    lm, w = sharded(master=master)
    before = [p.detach().to_local().clone() for p in lm.parameters()]
    loss, _ = loss_of(w, batch(rank))
    loss.backward()
    # only elements with a gradient can move (most embedding rows see no token of this batch)
    live = [p.grad.to_local() != 0 for p in lm.parameters()]
    opt = torch.optim.AdamW(list(lm.parameters()), lr=1e-6, betas=(0.9, 0.95), weight_decay=0.0)
    opt.step()
    moved = torch.tensor([
        sum(float(((p.detach().to_local() != b) & g).sum()) for p, b, g in zip(lm.parameters(), before, live)),
        sum(float(g.sum()) for g in live),
    ])
    dist.all_reduce(moved)
    return float(moved[0] / moved[1])
f32, f16 = moved_fraction(torch.float32), moved_fraction(None)
check(f32 > 0.95, f"[4] with fp32 masters only {f32:.3f} of the weights with a gradient moved at lr 1e-6")
check(f16 < 0.5, f"[4] with bf16 sharded params {f16:.3f} of the weights moved -- the check cannot see the rounding bug")
INFO.append(f"[4] one AdamW step at lr 1e-6 moves {f32:.3f} of the weights that have a gradient with fp32 masters, "
            f"{f16:.3f} with bf16 params")

# ---- [5] checkpoint round trip ------------------------------------------------------------------------------
def optimizer(lm):
    groups, lrs, names = split_depth_temporal_params(lm.named_parameters(), temporal_lr=1e-3, depth_lr=2e-3)
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), weight_decay=0.0), lrs, names

def train_step(lm, w, opt, step):
    loss, _ = loss_of(w, batch(rank, step))
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)

def full_params(lm):
    return {n: p.full_tensor().detach().clone() for n, p in lm.named_parameters()}

ck = ROOT / "rt5"
lm, w = sharded()
opt, _, _ = optimizer(lm)
train_step(lm, w, opt, 1)
at1 = full_params(lm)
save_sharded_checkpoint(ck / "checkpoints" / "checkpoint_000001", root=w, weights_module=lm, optimizer=opt,
                        step=1, is_main=rank == 0, log=lambda *a: None)
train_step(lm, w, opt, 2)
after = full_params(lm)

lm2, w2 = sharded()
opt2, _, _ = optimizer(lm2)
s = resume_sharded_checkpoint(ck, root=w2, optimizer=opt2, log=lambda *a: None)
check(s == 1, f"[5] resume returned step {s}, expected 1")
back = full_params(lm2)
check(all(back[n].dtype == torch.float32 for n in back), "[5] resumed parameters are not fp32")
wdiff = max(float((back[n] - at1[n]).abs().max()) for n in at1)
check(wdiff == 0.0, f"[5] resumed fp32 weights differ from the saved step by {wdiff}")
train_step(lm2, w2, opt2, 2)
replay = full_params(lm2)
rdiff = max(float((replay[n] - after[n]).abs().max()) for n in after)
check(rdiff == 0.0, f"[5] replaying step 2 from the checkpoint diverged by {rdiff}")
steps_ = {float(v["step"]) for v in opt2.state.values()}
check(steps_ == {2.0}, f"[5] AdamW step counters after the replay: {steps_} (expected 2)")
# negative control: a fresh model that takes step 2 without the resume lands elsewhere
lm3, w3 = sharded()
opt3, _, _ = optimizer(lm3)
train_step(lm3, w3, opt3, 2)
ctrl = full_params(lm3)
check(max(float((ctrl[n] - after[n]).abs().max()) for n in after) > 0, "[5] a fresh step 2 equals the resumed one -- the round trip proves nothing")
if rank == 0:
    import safetensors.torch as st_
    from moshi_family.build import overlay_state_dict
    wfile = ck / "checkpoints" / "checkpoint_000001" / "consolidated" / "model.safetensors"
    sd = st_.load_file(str(wfile))
    base = build()
    check(set(sd) == set(base.state_dict()), f"[5] model.safetensors keys != the unwrapped LM's: {sorted(set(sd) ^ set(base.state_dict()))[:4]}")
    check(all(v.dtype == torch.bfloat16 for v in sd.values() if v.is_floating_point()), "[5] saved weights are not bf16")
    bad = [k for k in at1 if not torch.equal(sd[k], at1[k].to(torch.bfloat16))]
    check(not bad, f"[5] saved weights != bf16(masters) for {bad[:3]}")
    missing, unexpected = base.load_state_dict(sd, strict=False)
    check(not missing and not unexpected, f"[5] overlay: missing {missing[:3]}, unexpected {unexpected[:3]}")
    overlay_state_dict(build(), str(wfile), tag="check_ppx_fsdp")
    meta = json.loads((ck / "checkpoints" / "checkpoint_000001" / RESUME_DIR / RESUME_MARKER).read_text())
    check(meta == {"step": 1, "world": world}, f"[5] resume marker {meta}")
dist.barrier()
save_sharded_checkpoint(ck / "checkpoints" / "checkpoint_000002", root=w, weights_module=lm, optimizer=opt,
                        step=2, is_main=rank == 0, log=lambda *a: None)
if rank == 0:
    states = [s_ for s_, _ in complete_resume_states(ck)]
    check(states == [2], f"[5] resume states after the second save: {states} (older one not pruned)")
    check((ck / "checkpoints" / "checkpoint_000001" / "consolidated" / "model.safetensors").exists(),
          "[5] pruning removed the weights of the older checkpoint")
    check(not (ck / "checkpoints" / "checkpoint_000001" / RESUME_DIR).exists(), "[5] old resume dir still there")
INFO.append(f"[5] resume: fp32 weights exact, step-2 replay diff {rdiff}, weights file = bf16 masters, prune keeps weights")

# ---- [6] run_training: per-rank theta_0 across a preemption -------------------------------------------------
out6 = ROOT / "rt6"

class Preempted(Exception):
    pass

def batches(start, stop_after=None):
    s_ = start
    while True:
        if stop_after is not None and s_ >= stop_after:
            raise Preempted()
        s_ += 1
        yield batch(rank, s_)

def run(lm, w, opt, lrs, names, start, stop_after=None):
    msgs = []
    def save_fn(step, final):
        ckd = out6 / "checkpoints" / f"checkpoint_{step:06d}"
        if final and (ckd / RESUME_DIR / RESUME_MARKER).exists():
            return
        save_sharded_checkpoint(ckd, root=w, weights_module=lm, optimizer=opt, step=step, is_main=rank == 0,
                                log=lambda *a: None)
    try:
        run_training(
            optimizer=opt, base_lrs=lrs, group_names=names, trainable_params=list(lm.parameters()),
            named_trainable=list(lm.named_parameters()), batch_iter=batches(start, stop_after),
            loss_step=lambda b: loss_of(w, b), save_fn=save_fn,
            cfg=TrainConfig(max_steps=4, grad_accum=1, warmup_steps=1, save_every=2, log_every=1,
                            sharded_callbacks=True,
                            weight_delta_baseline_path=str(out6 / "weight_delta_baseline.pt")),
            out_dir=out6, log=msgs.append, is_main=rank == 0, start_step=start,
        )
    except Preempted:
        msgs.append("PREEMPTED")
    return msgs

lm, w = sharded()
theta0 = full_params(lm)
opt, lrs, names = optimizer(lm)
m1 = run(lm, w, opt, lrs, names, 0, stop_after=2)
check("PREEMPTED" in m1, "[6] the first run was not preempted after step 2")
check((out6 / f"weight_delta_baseline.rank{rank}.pt").exists(), f"[6] rank {rank} wrote no per-rank theta_0")
lm2, w2 = sharded()
opt2, lrs, names = optimizer(lm2)
start = resume_sharded_checkpoint(out6, root=w2, optimizer=opt2, log=lambda *a: None)
check(start == 2, f"[6] resumed at {start}, expected 2")
m2 = run(lm2, w2, opt2, lrs, names, start)
check(any("restored theta_0" in m for m in m2), f"[6] rank {rank} did not restore its theta_0: {[m for m in m2 if 'baseline' in m]}")
theta4 = full_params(lm2)
truth = {}
for n in theta4:
    b = classify_param(n)
    truth[b] = truth.get(b, 0.0) + float(((theta4[n] - theta0[n]) ** 2).sum())
truth = {b: v ** 0.5 for b, v in truth.items()}
# CAN FAIL: rank 0's theta_0 shard on every rank (what the old rank-0-only baseline file would restore)
r0 = torch.load(out6 / "weight_delta_baseline.rank0.pt", map_location="cpu", weights_only=False)
own = torch.load(out6 / f"weight_delta_baseline.rank{rank}.pt", map_location="cpu", weights_only=False)
buckets = {}
for n, p in lm2.named_parameters():
    buckets.setdefault(classify_param(n), []).append(p)
def _wrong_ref(t0, t_own, p):
    # rank 0's shard where it has this rank's shape (uneven shards differ in length: keep this rank's own)
    return like_sharded((t0 if tuple(t0.shape) == tuple(p.to_local().shape) else t_own).float(), p)
wrong = {b: global_norm([p.detach().float() - _wrong_ref(t0, to, p) for p, t0, to in zip(ps, r0[b], own[b])])
         for b, ps in buckets.items()}
if rank == 0:
    recs = [json.loads(l) for l in (out6 / "metrics.train.jsonl").read_text().splitlines()]
    last = [r for r in recs if r["step"] == 4][-1]
    got = last["weight_delta_by_module"]
    rel = max(abs(got[b] - truth[b]) / truth[b] for b in truth)
    check(rel < 1e-4, f"[6] logged weight_delta_by_module {got} != reference {truth}")
    INFO.append(f"[6] resumed run_training: weight_delta_by_module == reference (rel {rel:.1e}); "
                f"with rank 0's shard everywhere it would read {wrong}")
check(any(abs(wrong[b] - truth[b]) > 1e-3 * truth[b] for b in truth),
      "[6] rank 0's baseline shard gives the same deltas -- the check cannot see a mis-restored baseline")

gathered = [None] * world
dist.all_gather_object(gathered, FAIL)
if rank == 0:
    for line in INFO:
        print(line)
    flat = [f"rank{r}: {m}" for r, fs in enumerate(gathered) for m in fs]
    print("FAILED:\n  - " + "\n  - ".join(flat) if flat else "WORKER OK")
dist.destroy_process_group()
sys.exit(1 if (rank == 0 and any(gathered)) else 0)
'''


def check_loader_device():
    """[0]: drive the real get_moshi_lm with safetensors' load_file captured (meta init, no weights read)."""
    import inspect

    sys.path.insert(0, str(FULL_DUPLEX))
    from moshi_family.personaplex.models import loaders

    class _Stop(Exception):
        pass

    seen = []

    def fake_load_file(filename, device="cpu"):
        seen.append(device)
        raise _Stop()

    def device_for(fn, ns, dev):
        saved = ns["load_file"]
        ns["load_file"] = fake_load_file
        try:
            fn("x.safetensors", device=dev)
        except _Stop:
            pass
        finally:
            ns["load_file"] = saved
        return seen[-1]

    real = loaders.get_moshi_lm
    got = {d: device_for(real, vars(loaders), d) for d in ("cuda:3", "cuda", "cpu")}
    assert got == {"cuda:3": "cuda:3", "cuda": "cuda", "cpu": "cpu"}, got
    src = inspect.getsource(real)
    fixed = 'device=str(dev) if dev.type == "cuda" else dev.type)'
    assert fixed in src, "mutation anchor moved: update check_loader_device"
    ns = dict(vars(loaders))
    exec(compile(src.replace(fixed, "device=dev.type)"), loaders.__file__, "exec"), ns)
    old = device_for(ns["get_moshi_lm"], ns, "cuda:3")
    assert old == "cuda", f"the pre-fix loader passed {old!r} -- this check cannot see the bug"
    print(
        '[ok] [0] get_moshi_lm(device="cuda:3") loads onto cuda:3 ("cuda" and "cpu" unchanged); the pre-fix source read "cuda"'
    )


def main():
    check_loader_device()
    tmp = _FsPath(tempfile.mkdtemp(prefix="check_ppx_fsdp_"))
    worker = tmp / "_check_ppx_fsdp_worker.py"
    worker.write_text(WORKER)
    env = dict(
        os.environ,
        FULL_DUPLEX=str(FULL_DUPLEX),
        CKPT_ROOT=str(tmp),
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        HF_HUB_OFFLINE="1",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc-per-node=2", "--master-port=29877", str(worker)],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    out = proc.stdout + proc.stderr
    if "WORKER OK" not in out:
        print(out[-6000:], file=sys.stderr)
        raise SystemExit(1)
    for line in out.splitlines():
        if line[:1] == "[" and line[1:2].isdigit():
            print("[ok] " + line)
    print(f"\nPersonaPlex FSDP2 full FT == single-process training (scratch {tmp})")


main()
