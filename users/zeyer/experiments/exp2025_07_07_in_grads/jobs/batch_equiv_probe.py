"""Batched-forward equivalence probe.
The only question:
does a seq's output run single (B=1)
match the same seq run inside a B>1 batch,
and how large is the diff?
Reports the per-seq max|Δ| in the final log_probs (real tokens),
under default math and under safe math (TF32 off / deterministic).
Uses the shared seq_batch_check.single_vs_batch_diff
(same util the actual jobs use);
the seq-batch masking does the patching, this only measures equivalence.
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
        probe_version: int = 6,
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
        from i6_experiments.users.zeyer.experiments.exp2025_07_07_in_grads.jobs.seq_batch_check import (
            single_vs_batch_diff,
        )

        dev = torch.device("cuda")
        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))[self.dataset_key]
        idx = sorted(range(min(len(ds), 200)), key=lambda i: len(ds[i]["audio"]["array"]))
        sel = [idx[t] for t in np.linspace(0, len(idx) - 1, self.num_seqs).round().astype(int)]
        seqs = [
            (
                torch.tensor(np.asarray(ds[i]["audio"]["array"], dtype=np.float32)),
                int(ds[i]["audio"]["sampling_rate"]),
                list(ds[i]["word_detail"]["utterance"]),
            )
            for i in sel
        ]
        report = [f"{len(seqs)} seqs, lengths {[int(a.shape[0]) for a, _, _ in seqs]}"]

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
                d_def = single_vs_batch_diff(model, seqs)
                set_math(True)
                d_safe = single_vs_batch_diff(model, seqs)
                report.append(f"\n### {name}: log_probs |Δ| single(B=1) vs in-batch")
                report.append(f"    default math (TF32 on): max {max(d_def):.2e}  mean {sum(d_def) / len(d_def):.2e}")
                report.append(
                    f"    safe math (TF32 off)  : max {max(d_safe):.2e}  mean {sum(d_safe) / len(d_safe):.2e}"
                )
                report.append(f"    per-seq (safe): {[f'{d:.1e}' for d in d_safe]}")
                del model
                torch.cuda.empty_cache()
            except Exception:
                report.append(f"\n### {name}: SKIPPED/ERROR\n{traceback.format_exc().splitlines()[-1]}")

        txt = "\n".join(report)
        print(txt)
        self.out_report.set(txt)
