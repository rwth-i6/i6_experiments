"""Batched-forward equivalence probe: the ONLY question is whether a seq's output run SINGLE (B=1)
matches the SAME seq run inside a B>1 batch, and how large the diff is. Reports the per-seq max|Δ| in
the final log_probs (real tokens), under default math and under safe math (TF32 off / deterministic).
No masking / garbage / identical-batch experiments -- those were misleading.
"""

from __future__ import annotations
from typing import Optional, Any, Dict
from sisyphus import Job, Task, tk

from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)
from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy


class BatchForwardEquivalenceProbeJob(Job):
    """Per-model: max|Δ| of final log_probs, seq run single (B=1) vs the same seq in a B>1 batch."""

    def __init__(
        self,
        *,
        model_configs: Dict[str, Any],
        dataset_dir: tk.Path,
        dataset_key: str,
        num_seqs: int = 8,
        probe_version: int = 5,
        returnn_root: Optional[tk.Path] = None,
    ):
        super().__init__()
        self.model_configs = model_configs
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.num_seqs = num_seqs
        self.probe_version = probe_version
        self.returnn_root = returnn_root
        self.rqmt = {"time": 2, "cpu": 2, "gpu": 1, "mem": 48}
        self.out_report = self.output_var("report.txt")

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os
        import sys
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
        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))[self.dataset_key]
        idx = sorted(range(min(len(ds), 200)), key=lambda i: len(ds[i]["audio"]["array"]))
        sel = [idx[t] for t in np.linspace(0, len(idx) - 1, self.num_seqs).round().astype(int)]
        seqs = []
        for i in sel:
            d = ds[i]
            seqs.append(
                (
                    torch.tensor(np.asarray(d["audio"]["array"], dtype=np.float32)),
                    int(d["audio"]["sampling_rate"]),
                    list(d["word_detail"]["utterance"]),
                )
            )
        report = [f"{len(seqs)} seqs, lengths {[int(a.shape[0]) for a, _, _ in seqs]}"]

        def log_probs(batch):
            """final per-token log_probs [B, n_max, V] for a batch (any B)."""
            nb = len(batch)
            ml = max(int(a.shape[0]) for a, _, _ in batch)
            raw = torch.zeros((nb, ml))
            for i, (a, _, _) in enumerate(batch):
                raw[i, : a.shape[0]] = a
            fwd = model(
                raw_inputs=raw,
                raw_inputs_sample_rate=batch[0][1],
                raw_input_seq_lens=torch.tensor([int(a.shape[0]) for a, _, _ in batch]),
                raw_targets=[w for _, _, w in batch],
                raw_target_seq_lens=torch.tensor([len(w) for _, _, w in batch]),
                omitted_prev_context=torch.zeros(nb, dtype=torch.int64),
            )
            tse = fwd.target_start_end.cpu()
            n_tgt = torch.tensor([int(tse[i, len(b[2]), 1]) for i, b in enumerate(batch)])
            lp = model.log_probs(forward_output=fwd, start=torch.zeros(nb, dtype=torch.int64), end=n_tgt)
            return lp.detach().float().cpu(), n_tgt.tolist()

        def diff_single_vs_batch():
            """per-seq max|Δ| (and mean) of single(B=1) vs the same seq inside the full batch."""
            lp_b, ntb = log_probs(seqs)
            ds_ = []
            for i, b in enumerate(seqs):
                lp_s, nts = log_probs([b])
                n = min(ntb[i], nts[0])
                ds_.append(float((lp_b[i, :n] - lp_s[0, :n]).abs().max()))
            return max(ds_), sum(ds_) / len(ds_), ds_

        def set_math(safe):
            torch.backends.cuda.matmul.allow_tf32 = not safe
            torch.backends.cudnn.allow_tf32 = not safe
            torch.backends.cudnn.benchmark = not safe
            torch.backends.cudnn.deterministic = safe

        for name, cfg in self.model_configs.items():
            try:
                model = make_model(**instanciate_delayed_copy(cfg), device=dev)
                for p in model.parameters():
                    p.requires_grad = False
                set_math(False)
                mx_d, mean_d, _ = diff_single_vs_batch()
                set_math(True)
                mx_s, mean_s, per = diff_single_vs_batch()
                report.append(f"\n### {name}: log_probs |Δ| single(B=1) vs in-batch")
                report.append(f"    default math (TF32 on) : max {mx_d:.2e}  mean {mean_d:.2e}")
                report.append(f"    safe math (TF32 off)   : max {mx_s:.2e}  mean {mean_s:.2e}")
                report.append(f"    per-seq (safe): {[f'{d:.1e}' for d in per]}")
                del model
                torch.cuda.empty_cache()
            except Exception:
                report.append(f"\n### {name}: SKIPPED/ERROR\n{traceback.format_exc().splitlines()[-1]}")

        txt = "\n".join(report)
        print(txt)
        self.out_report.set(txt)
