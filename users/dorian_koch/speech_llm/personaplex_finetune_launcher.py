"""personaplex_finetune_launcher.py -- PersonaPlex finetuning entry point (Workstream E).

Builds the training loop directly on the PersonaPlex fork's own model (``moshi_finetune`` is absent
from the personaplex venv) using ``loaders.get_moshi_lm(model.safetensors)`` +
``LMModel.forward_train(codes) -> LMOutput``, with our ``personaplex_train_data`` for batches + the
paper-weighted, system-prompt-masked loss.

Paper recipe (arXiv 2602.06053): Adam + cosine annealing, depformer LR 4e-6 / temporal 2e-6, batch
32, 24,576 steps, seq 2048 tok (=163.84 s), 8xA100, FULL finetune, down-weight non-semantic audio
x0.02 + padded text x0.3, mask loss on the system prompt.

SINGLE-GPU adaptation (this cluster: <=4 GPUs; we train on 1 for now). Full-FT of a 7B won't fit one
GPU (Adam states ~56 GB). The fork's backbone attention uses raw weight Parameters
(``in_proj_weight``) and a fused gating kernel that reads ``linear.weight`` directly, so generic
nn.Linear-LoRA can't wrap it. So single-GPU defaults to ``train_scope="heads"``: **freeze the big
temporal-transformer backbone, full-finetune the depformer + output heads** (the audio/text
generation stack -- real params, no module wrapping, no fused-kernel issues). Because the backbone
is frozen, autograd builds no backbone-weight graph, so activation memory stays low and it fits one
GPU. ``train_scope="full"`` (multi-GPU, more memory) approaches the paper's full-FT. Split LRs map
naturally: depformer @ 4e-6, the other trained heads @ 2e-6.
"""

from __future__ import annotations

import site as _site
import sys as _sys

# The recipe tree has a top-level ``moshi`` (kyutai source) that SHADOWS the PersonaPlex fork in the
# venv site-packages whenever ``recipe`` is on PYTHONPATH (the documented thin-venv gotcha -- see
# personaplex.md). Make the venv site-packages win for ``import moshi`` so we load the PersonaPlex
# fork (patched strict=False loader), not recipe/moshi (kyutai, strict load -> depformer mismatch).
for _p in _site.getsitepackages():
    if _p in _sys.path:
        _sys.path.remove(_p)
        _sys.path.insert(0, _p)

import json
import math
import os
import time
from pathlib import Path

import torch

# Parameter-name patterns trained when train_scope="heads" (single-GPU; backbone frozen). The
# depformer (incl. depformer_in/emb/text_emb), the per-codebook audio heads (linears), the text head
# (text_linear), and the final norm. Everything else (transformer.*, input emb, condition*) frozen.
_HEADS_PATTERNS = ("depformer", "linears", "text_linear", "out_norm")


def _load_config(path: str) -> dict:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f) or {}


def _is_depformer_param(name: str) -> bool:
    return "depformer" in name


def main():
    cfg = _load_config(_sys.argv[1]) if len(_sys.argv) > 1 else {}

    hf_repo = cfg.get("hf_repo_id", "nvidia/personaplex-7b-v1")
    data_path = cfg["train_data"]
    out_dir = Path(cfg.get("out_dir", "output"))
    max_steps = int(cfg.get("max_steps", 24576))
    per_gpu_batch = int(cfg.get("per_gpu_batch", 1))
    grad_accum = int(cfg.get("grad_accum", 32))  # effective batch = per_gpu_batch * grad_accum * world
    lr_backbone = float(cfg.get("lr_temporal", 2e-6))
    lr_depformer = float(cfg.get("lr_depformer", 4e-6))
    warmup = int(cfg.get("warmup_steps", 500))
    grad_clip = float(cfg.get("grad_clip", 1.0))
    duration_sec = float(cfg.get("duration_sec", 163.84))
    train_scope = cfg.get("train_scope", "heads")  # "heads" (single-GPU) | "full" (multi-GPU)
    save_every = int(cfg.get("save_every", 2048))
    log_every = int(cfg.get("log_every", 10))
    seed = int(cfg.get("seed", 0))
    system_prompt_key = cfg.get("system_prompt_key")

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(local_rank)
    is_dist = world > 1
    if is_dist:
        torch.distributed.init_process_group(backend="nccl")
    is_main = rank == 0
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    def log(msg):
        if is_main:
            print(f"[pplex-train] {msg}", flush=True)

    from huggingface_hub import hf_hub_download
    from moshi.models import loaders

    from i6_experiments.users.dorian_koch.speech_llm.personaplex_train_data import (
        PersonaPlexDataConfig,
        PersonaPlexTokenizer,
        build_data_loader,
        personaplex_loss,
    )

    log(f"loading {hf_repo} on {device} (train_scope={train_scope})")
    moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
    model = loaders.get_moshi_lm(moshi_weight, device=device)
    dep_q = model.dep_q

    # Freeze per scope: "heads" trains depformer + output heads, backbone frozen (single-GPU).
    if train_scope == "heads":
        for n, p in model.named_parameters():
            p.requires_grad = any(pat in n for pat in _HEADS_PATTERNS)
    elif train_scope == "full":
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"unknown train_scope {train_scope!r}")
    model.train()

    core = model
    if is_dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=(train_scope == "heads")
        )

    dep_params, base_params = [], []
    for n, p in core.named_parameters():
        if not p.requires_grad:
            continue
        (dep_params if _is_depformer_param(n) else base_params).append(p)
    n_train = sum(p.numel() for p in dep_params + base_params)
    assert n_train > 0, "no trainable params (check train_scope / patterns)"
    opt = torch.optim.AdamW(
        [{"params": base_params, "lr": lr_backbone}, {"params": dep_params, "lr": lr_depformer}],
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    log(
        f"trainable: {n_train / 1e6:.1f}M (heads/base {sum(p.numel() for p in base_params) / 1e6:.1f}M @ "
        f"{lr_backbone}, depformer {sum(p.numel() for p in dep_params) / 1e6:.1f}M @ {lr_depformer})"
    )

    def lr_scale(step):
        if step < warmup:
            return step / max(warmup, 1)
        prog = (step - warmup) / max(max_steps - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))

    tok = PersonaPlexTokenizer(hf_repo, device=device, cfg=PersonaPlexDataConfig(duration_sec=duration_sec))
    loader = build_data_loader(
        data_path, tok, batch_size=per_gpu_batch, seed=seed, system_prompt_key=system_prompt_key, infinite=True
    )

    metrics_path = out_dir / "metrics.train.jsonl"
    t0 = time.time()
    eff = per_gpu_batch * grad_accum * world
    log(f"training: {max_steps} steps, per-gpu batch {per_gpu_batch} x grad_accum {grad_accum} x {world} = eff {eff}")
    trainable = [p for p in core.parameters() if p.requires_grad]
    for step in range(1, max_steps + 1):
        scale = lr_scale(step)
        for g, base in zip(opt.param_groups, (lr_backbone, lr_depformer)):
            g["lr"] = base * scale
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(grad_accum):
            codes, loss_mask = next(loader)
            codes = codes.to(device)
            out = core.forward_train(codes)
            loss = personaplex_loss(out, codes, loss_mask, dep_q=dep_q) / grad_accum
            loss.backward()
            loss_acc += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        opt.step()

        if is_main and (step % log_every == 0 or step == 1):
            elapsed = time.time() - t0
            eta = elapsed / step * (max_steps - step)
            rec = {
                "step": step,
                "loss": loss_acc,
                "lr_scale": scale,
                "percent_done": 100.0 * step / max_steps,
                "eta_in_seconds": eta,
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            log(f"step {step}/{max_steps} loss {loss_acc:.4f} eta {eta / 3600:.2f}h")

        if is_main and step % save_every == 0:
            _save(core, out_dir / "consolidated", step)

    if is_main:
        _save(core, out_dir / "consolidated", max_steps, final=True)
    if is_dist:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def _save(core, dest: Path, step: int, final: bool = False) -> None:
    """Save only the trained (requires_grad) params -- the finetuned heads delta."""
    import safetensors.torch as st

    dest.mkdir(parents=True, exist_ok=True)
    trained = {n for n, p in core.named_parameters() if p.requires_grad}
    sd = {k: v.detach().cpu() for k, v in core.state_dict().items() if k in trained}
    name = "trained_heads.safetensors" if final else f"trained_heads.step{step}.safetensors"
    st.save_file(sd, str(dest / name))
    (dest / "config.json").write_text(json.dumps({"step": step, "final": final, "trained_keys": len(sd)}))
    print(f"[pplex-train] saved {dest / name} ({len(sd)} tensors)", flush=True)


if __name__ == "__main__":
    main()
