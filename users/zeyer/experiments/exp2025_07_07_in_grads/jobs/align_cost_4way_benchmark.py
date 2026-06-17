"""4-way cost benchmark: Forward / Prefix-sum forward / Prefix-sum backward / Backward, per model.

The DP decode / aligner is EXCLUDED (shared across all aligners + currently unoptimized). The backward
is always the fast batched VJP (``is_grads_batched``). Models with a per-token score LATTICE (CTC /
transducer, exposed via ``outputs["lp"]`` + ``outputs["prefix_from_lp"]`` -- the split_prefix_backward
interface) get all 4 stages; AED / speech-LLM have no lattice (the per-token score is a direct
teacher-forced gather), so they report Forward + Backward only.

Needs torch 2.12 (compiled-scan prefix). Per seq: one untimed warmup run (to compile the per-shape scan)
then one timed run. Reports ms per second of audio. Per-model try/except so one failure doesn't sink the rest.
"""

from __future__ import annotations
from typing import Optional, Any, Dict
from sisyphus import Job, Task, tk

from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)
from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy


class AlignCost4WayBenchmarkJob(Job):
    """Per-model Forward / Prefix-fwd / Prefix-bwd / Backward wall-clock, same GPU. See module docstring."""

    # v1 passed CUDA audio (phi4 failed on a host-side numpy conversion); v2 passes CPU audio.
    __sis_version__ = 2

    def __init__(
        self,
        *,
        model_configs: Dict[str, Any],
        dataset_dir: tk.Path,
        dataset_key: str,
        num_seqs: int = 40,
        warmup: int = 3,
        returnn_root: Optional[tk.Path] = None,
    ):
        super().__init__()
        self.model_configs = model_configs
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.num_seqs = num_seqs
        self.warmup = warmup
        self.returnn_root = returnn_root
        self.rqmt = {"time": 8, "cpu": 2, "gpu": 1, "mem": 48}
        self.out_metrics = self.output_var("metrics.txt")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys
        import gc
        import time
        import traceback

        set_hf_offline_mode()
        import i6_experiments

        sys.path.insert(0, os.path.dirname(os.path.dirname(i6_experiments.__file__)))
        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        import numpy as np
        import torch
        from datasets import load_dataset
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models import make_model

        dev = torch.device("cuda")
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))[self.dataset_key]
        n_take = min(self.num_seqs + self.warmup, len(ds))
        seqs = []
        for i in range(n_take):
            d = ds[i]
            seqs.append(
                (
                    torch.tensor(np.asarray(d["audio"]["array"], dtype=np.float32)),
                    int(d["audio"]["sampling_rate"]),
                    list(d["word_detail"]["utterance"]),
                )
            )

        def stages(model, audio, sr, words):
            """Run one forward + the per-stage backwards; return a dict of stage seconds."""
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fwd = model(
                # CPU audio (like ExtractInGradsPerTokenJob): the wrappers move it to device internally,
                # and phi4 does a host-side numpy conversion that fails on a CUDA tensor.
                raw_inputs=audio[None],
                raw_inputs_sample_rate=sr,
                raw_input_seq_lens=torch.tensor([int(audio.shape[0])]),
                raw_targets=[words],
                raw_target_seq_lens=torch.tensor([len(words)]),
                omitted_prev_context=torch.tensor([0]),
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            tse = fwd.target_start_end[0]
            tok = [k for w in range(len(words)) for k in range(int(tse[w, 0]), int(tse[w, 1]))]
            outs = fwd.outputs
            if isinstance(outs, dict) and "prefix_from_lp" in outs:
                lp = outs["lp"]
                lp_leaf = lp.detach().requires_grad_(True)
                torch.cuda.synchronize()
                p0 = time.perf_counter()
                partial = outs["prefix_from_lp"](lp_leaf)  # [S]  prefix-sum forward
                torch.cuda.synchronize()
                p1 = time.perf_counter()
                js = [torch.autograd.grad(partial[k], lp_leaf, retain_graph=True)[0] for k in tok]  # prefix-bwd
                jac = torch.stack(js, 0)  # [num_tok, *lp.shape]
                torch.cuda.synchronize()
                pb = time.perf_counter()
                (_g,) = torch.autograd.grad(lp, fwd.inputs, grad_outputs=jac, is_grads_batched=True)  # enc-bwd
                torch.cuda.synchronize()
                eb = time.perf_counter()
                out = {"_fwd": t1 - t0, "prefix_fwd": p1 - p0, "prefix_bwd": pb - p1, "backward": eb - pb}
            else:
                torch.cuda.synchronize()
                b0 = time.perf_counter()
                for w in range(len(words)):
                    a, b = int(tse[w, 0]), int(tse[w, 1])
                    if b <= a:
                        continue
                    lpw = model.log_probs(forward_output=fwd, start=torch.tensor([a]), end=torch.tensor([b]))
                    tgt = fwd.targets[0, a:b].to(lpw.device)
                    loss = lpw.gather(2, tgt[None, :, None]).squeeze(-1)  # [1, b-a]
                    nt = b - a
                    v = torch.zeros((nt,) + loss.shape, device=loss.device, dtype=loss.dtype)
                    for k in range(nt):
                        v[k, :, k] = 1.0
                    torch.autograd.grad(
                        loss, fwd.inputs, grad_outputs=v, is_grads_batched=True, retain_graph=(w < len(words) - 1)
                    )
                torch.cuda.synchronize()
                b1 = time.perf_counter()
                out = {"_fwd": t1 - t0, "backward": b1 - b0}
            del fwd
            return out

        results: Dict[str, Any] = {"_gpu": torch.cuda.get_device_name(0), "_num_seqs": self.num_seqs}
        for name, cfg in self.model_configs.items():
            model = None
            try:
                model = make_model(**instanciate_delayed_copy(cfg), device=dev)
                for p in model.parameters():
                    p.requires_grad = False
                acc: Dict[str, float] = {}
                aud, n, has_prefix = 0.0, 0, False
                for i, (audio, sr, words) in enumerate(seqs):
                    stages(model, audio, sr, words)  # untimed warmup (compile per-shape scan)
                    s = stages(model, audio, sr, words)  # timed
                    has_prefix = "prefix_fwd" in s
                    if i >= self.warmup:
                        for k, v in s.items():
                            acc[k] = acc.get(k, 0.0) + v
                        aud += int(audio.shape[0]) / sr
                        n += 1
                # Forward column = encoder forward (model.forward minus the internal prefix-fwd it also computes).
                fwd_only = (
                    max(0.0, acc.get("_fwd", 0.0) - acc.get("prefix_fwd", 0.0)) if has_prefix else acc.get("_fwd", 0.0)
                )
                r = {
                    "forward_ms_per_s": 1000.0 * fwd_only / aud,
                    "backward_ms_per_s": 1000.0 * acc.get("backward", 0.0) / aud,
                    "n": n,
                }
                if has_prefix:
                    r["prefix_fwd_ms_per_s"] = 1000.0 * acc["prefix_fwd"] / aud
                    r["prefix_bwd_ms_per_s"] = 1000.0 * acc["prefix_bwd"] / aud
                results[name] = r
                print(name, r, flush=True)
            except Exception as e:  # noqa
                traceback.print_exc()
                results[name] = {"error": f"{type(e).__name__}: {e}"}
                print(name, "FAILED:", results[name]["error"], flush=True)
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()

        print("ALL RESULTS:", results)
        flat: Dict[str, Any] = {k: v for k, v in results.items() if k.startswith("_")}
        for name, d in results.items():
            if name.startswith("_"):
                continue
            for k, v in d.items():
                flat[f"{name}|{k}"] = v
        self.out_metrics.set(flat)
