"""Probe v3: is the batched-vs-single difference a PADDING leak (maskable) or GPU kernel fp
non-determinism (not maskable)? Two batched modes vs the single-seq forward:
  - VARIED batch: 10 seqs of different length (real padding) -> padding-leak + kernel-fp.
  - IDENTICAL batch: 10 copies of one seq (NO padding) -> kernel-fp ONLY.
If IDENTICAL already shows the diff, padding/masking is not the cause. We report the diff in the
FINAL log_probs (the score that drives the grad, at real token positions) for both modes, plus the
first interior submodule leak. No extra masking is applied (we measure the model's own behavior).
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
    """single vs (varied / identical) batched forward; final-log_probs + first-interior-leak."""

    def __init__(
        self,
        *,
        model_configs: Dict[str, Any],
        dataset_dir: tk.Path,
        dataset_key: str,
        num_seqs: int = 10,
        margin: int = 4,
        probe_version: int = 4,
        returnn_root: Optional[tk.Path] = None,
    ):
        super().__init__()
        self.model_configs = model_configs
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.num_seqs = num_seqs
        self.margin = margin
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
        idx_by_len = sorted(range(min(len(ds), 200)), key=lambda i: len(ds[i]["audio"]["array"]))
        take = np.linspace(0, len(idx_by_len) - 1, self.num_seqs).round().astype(int)
        sel = [idx_by_len[i] for i in take]
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
        lens = [int(a.shape[0]) for a, _, _ in seqs]
        report = [f"batch of {len(seqs)} seqs, lengths {lens}; margin={self.margin}"]

        class SeqMasker:
            """Zero padded positions before every feature-extractor Conv1d, threading the per-sample
            valid length through the conv geometry. (The transformer is already length-masked by HF.)"""

            def __init__(self, model, input_lengths):
                self.valid = input_lengths.long().clone().to(dev)
                self.handles = []
                for nm, m in model.named_modules():
                    if isinstance(m, torch.nn.Conv1d) and "feature_extractor" in nm:
                        self.handles.append(m.register_forward_pre_hook(self._pre))
                        self.handles.append(m.register_forward_hook(self._post))

            def _pre(self, module, inputs):
                x = inputs[0]
                T = x.shape[-1]
                mask = torch.arange(T, device=x.device)[None, :] < self.valid[:, None]
                return (x * mask[:, None, :].to(x.dtype),) + tuple(inputs[1:])

            def _post(self, module, inp, out):
                k = module.kernel_size[0]
                s = module.stride[0]
                p = module.padding[0] if isinstance(module.padding, tuple) else module.padding
                d = module.dilation[0]
                self.valid = (self.valid + 2 * p - d * (k - 1) - 1) // s + 1

            def remove(self):
                for h in self.handles:
                    h.remove()

        def call(model, batch, mask_lengths=None):
            sr = batch[0][1]
            nb = len(batch)
            ml = max(int(a.shape[0]) for a, _, _ in batch)
            raw = torch.zeros((nb, ml))
            for i, (a, _, _) in enumerate(batch):
                raw[i, : a.shape[0]] = a
            masker = SeqMasker(model, torch.tensor(mask_lengths)) if mask_lengths is not None else None
            try:
                fwd = model(
                    raw_inputs=raw,
                    raw_inputs_sample_rate=sr,
                    raw_input_seq_lens=torch.tensor([int(a.shape[0]) for a, _, _ in batch]),
                    raw_targets=[w for _, _, w in batch],
                    raw_target_seq_lens=torch.tensor([len(w) for _, _, w in batch]),
                    omitted_prev_context=torch.zeros(nb, dtype=torch.int64),
                )
            finally:
                if masker is not None:
                    masker.remove()
            tse = fwd.target_start_end.cpu()
            n_tgt = torch.tensor([int(tse[i, len(b[2]), 1]) for i, b in enumerate(batch)])
            lp = model.log_probs(forward_output=fwd, start=torch.zeros(nb, dtype=torch.int64), end=n_tgt)
            return lp.detach().float().cpu(), n_tgt.tolist(), tse

        def lp_diff(model, batch, mask=False):
            """max|Δ| in final per-token log_probs (real tokens) batched vs single, over the batch.
            mask=True installs the SeqMasker on the batched forward (single forward needs no mask)."""
            ml = [int(a.shape[0]) for a, _, _ in batch] if mask else None
            lp_b, ntb, _ = call(model, batch, mask_lengths=ml)
            mx = 0.0
            for i, b in enumerate(batch):
                lp_s, nts, _ = call(model, [b])
                n = min(ntb[i], nts[0])
                d = (lp_b[i, :n] - lp_s[0, :n]).abs().max()
                mx = max(mx, float(d))
            return mx

        def set_math(safe):
            # safe = full-precision + deterministic: TF32 OFF, no cudnn autotune, deterministic algos.
            torch.backends.cuda.matmul.allow_tf32 = not safe
            torch.backends.cudnn.allow_tf32 = not safe
            torch.backends.cudnn.benchmark = not safe
            torch.backends.cudnn.deterministic = safe

        report.append(
            f"\ndefault math: tf32(matmul)={torch.backends.cuda.matmul.allow_tf32} "
            f"tf32(cudnn)={torch.backends.cudnn.allow_tf32} benchmark={torch.backends.cudnn.benchmark}"
        )
        for name, cfg in self.model_configs.items():
            try:
                mc = instanciate_delayed_copy(cfg)
                model = make_model(**mc, device=dev)
                for p in model.parameters():
                    p.requires_grad = False
                mid = seqs[len(seqs) // 2]
                ident_batch = [mid for _ in range(len(seqs))]

                set_math(False)
                varied_def = lp_diff(model, seqs)
                ident_def = lp_diff(model, ident_batch)
                set_math(True)
                varied_safe = lp_diff(model, seqs)
                ident_safe = lp_diff(model, ident_batch)
                varied_masked_safe = lp_diff(model, seqs, mask=True)

                report.append(f"\n### {name}: FINAL log_probs max|Δ| (real tokens), batched vs single")
                report.append(f"    VARIED  batch, default math (TF32 on) : {varied_def:.3e}")
                report.append(f"    VARIED  batch, SAFE math (TF32 off)   : {varied_safe:.3e}")
                report.append(f"    IDENTICAL batch (no pad), default     : {ident_def:.3e}")
                report.append(f"    IDENTICAL batch (no pad), SAFE        : {ident_safe:.3e}")
                report.append(f"    VARIED  batch, SAFE + MASKED          : {varied_masked_safe:.3e}")
                report.append(
                    f"    -> masking {'CLOSES residual to floor' if varied_masked_safe < 2 * ident_safe else 'does NOT close it'}"
                )
                del model
                torch.cuda.empty_cache()
            except Exception:
                report.append(f"\n### {name}: SKIPPED/ERROR\n{traceback.format_exc().splitlines()[-1]}")

        txt = "\n".join(report)
        print(txt)
        self.out_report.set(txt)
