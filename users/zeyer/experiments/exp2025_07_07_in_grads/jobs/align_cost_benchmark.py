"""Cost benchmark: forward vs (batched) grad-backward, per model, on one pinned GPU.

For each model we time, over a fixed set of TIMIT seqs on the SAME GPU:
  - the forward pass (what the native methods -- forced-align / attention-DTW -- also pay), and
  - the batched per-token grad backward (the EXTRA cost the gradient method adds).
The align/DP step is excluded (≈ identical across methods; impl-detail). Reports ms per second of
audio (and per seq) for fwd and bwd. Backward mirrors ExtractInGradsPerTokenJob: per word, one
vmapped ``autograd.grad(is_grads_batched=True)`` over that word's tokens. Per-model try/except so one
model failing (e.g. a vmap-incompatible flash-attn backward) doesn't sink the rest.
"""

from __future__ import annotations
from typing import Optional, Any, Dict
from sisyphus import Job, Task, tk

from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)
from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy


class AlignCostBenchmarkJob(Job):
    """Per-model forward vs batched grad-backward wall-clock, same GPU. See module docstring."""

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
        self.rqmt = {"time": 4, "cpu": 2, "gpu": 1, "mem": 48}
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

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        import numpy as np
        import torch
        from datasets import load_dataset
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.models import make_model

        dev = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print("GPU:", gpu_name, flush=True)

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

        results: Dict[str, Any] = {"_gpu": gpu_name, "_num_seqs": self.num_seqs}
        for name, cfg in self.model_configs.items():
            model = None
            try:
                mc = instanciate_delayed_copy(cfg)
                model = make_model(**mc, device=dev)
                for p in model.parameters():
                    p.requires_grad = False
                fwd_s, bwd_s, aud_s, n = 0.0, 0.0, 0.0, 0
                for i, (audio, sr, words) in enumerate(seqs):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    fwd = model(
                        raw_inputs=audio[None].to(dev),
                        raw_inputs_sample_rate=sr,
                        raw_input_seq_lens=torch.tensor([int(audio.shape[0])]),
                        raw_targets=[words],
                        raw_target_seq_lens=torch.tensor([len(words)]),
                        omitted_prev_context=torch.tensor([0]),
                    )
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    tse = fwd.target_start_end[0]  # [num_words(+sentinel), 2]
                    bwd0 = time.perf_counter()
                    for w in range(len(words)):
                        a, b = int(tse[w, 0]), int(tse[w, 1])
                        if b <= a:
                            continue
                        lp = model.log_probs(
                            forward_output=fwd, start=torch.tensor([a]), end=torch.tensor([b])
                        )  # [1, b-a, V]
                        tgt = fwd.targets[0, a:b].to(lp.device)
                        loss = lp.gather(2, tgt[None, :, None]).squeeze(-1)  # [1, b-a]
                        ntok = b - a
                        v = torch.zeros((ntok,) + loss.shape, device=loss.device, dtype=loss.dtype)
                        for k in range(ntok):
                            v[k, :, k] = 1.0
                        torch.autograd.grad(
                            loss, fwd.inputs, grad_outputs=v, is_grads_batched=True, retain_graph=(w < len(words) - 1)
                        )
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    if i >= self.warmup:
                        fwd_s += t1 - t0
                        bwd_s += t2 - bwd0
                        aud_s += int(audio.shape[0]) / sr
                        n += 1
                    del fwd
                results[name] = {
                    "fwd_ms_per_s": 1000.0 * fwd_s / aud_s,
                    "bwd_ms_per_s": 1000.0 * bwd_s / aud_s,
                    "total_ms_per_s": 1000.0 * (fwd_s + bwd_s) / aud_s,
                    "fwd_ms_per_seq": 1000.0 * fwd_s / n,
                    "bwd_ms_per_seq": 1000.0 * bwd_s / n,
                    "n": n,
                }
                print(name, results[name], flush=True)
            except Exception as e:  # noqa
                traceback.print_exc()
                results[name] = {"error": f"{type(e).__name__}: {e}"}
                print(name, "FAILED:", results[name]["error"], flush=True)
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()

        print("ALL RESULTS:", results)
        # Flatten to "<model>|<metric>" keys so a LaTeX table can index it by a single key.
        flat: Dict[str, Any] = {k: v for k, v in results.items() if k.startswith("_")}
        for name, d in results.items():
            if name.startswith("_"):
                continue
            for k, v in d.items():
                flat[f"{name}|{k}"] = v
        self.out_metrics.set(flat)
