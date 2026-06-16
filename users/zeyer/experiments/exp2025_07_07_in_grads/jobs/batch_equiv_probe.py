"""Probe: single-seq forward vs batched (B>1, padded) forward — how much do per-seq outputs differ,
and WHERE does the leak originate? Runs a representative mixed-length batch through each model's
wrapper (needs B>1 forward support; transducers don't have it yet -> reported as skipped), compares
per-seq log_probs at REAL (unpadded) positions, and localizes the leak by hooking the underlying
module tree: for each submodule, batched-output[i, :Lvalid] vs single-output[i] -> first divergent
submodule = the leak origin (expected: a norm/conv over the time axis that includes padded frames).
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
    """Per-model single-vs-batched forward equivalence + per-submodule leak localization."""

    def __init__(
        self,
        *,
        model_configs: Dict[str, Any],
        dataset_dir: tk.Path,
        dataset_key: str,
        num_seqs: int = 10,
        returnn_root: Optional[tk.Path] = None,
    ):
        super().__init__()
        self.model_configs = model_configs
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.num_seqs = num_seqs
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
        # pick num_seqs of VARIED length (sort by len, take a spread) so the batch has real padding
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
        report = [f"batch of {len(seqs)} seqs, sample lengths {lens}"]

        for name, cfg in self.model_configs.items():
            try:
                mc = instanciate_delayed_copy(cfg)
                model = make_model(**mc, device=dev)
                for p in model.parameters():
                    p.requires_grad = False
                sr = seqs[0][1]
                nb = len(seqs)
                maxlen = max(lens)
                raw = torch.zeros((nb, maxlen))
                for i, (a, _, _) in enumerate(seqs):
                    raw[i, : a.shape[0]] = a

                # capture per-submodule outputs; record valid time-length per submodule from the
                # single-seq run (its output length IS the valid length at that layer).
                def hook_capture(store):
                    handles = []

                    def mk(nm):
                        def h(_m, _inp, out):
                            t = out[0] if isinstance(out, tuple) else out
                            if isinstance(t, torch.Tensor):
                                store[nm] = t.detach().float().cpu()

                        return h

                    for nm, m in model.named_modules():
                        if nm and len(list(m.children())) == 0:  # leaf modules only
                            handles.append(m.register_forward_hook(mk(nm)))
                    return handles

                # batched forward
                store_b: Dict[str, Any] = {}
                hs = hook_capture(store_b)
                fwd_b = model(
                    raw_inputs=raw,
                    raw_inputs_sample_rate=sr,
                    raw_input_seq_lens=torch.tensor(lens),
                    raw_targets=[w for _, _, w in seqs],
                    raw_target_seq_lens=torch.tensor([len(w) for _, _, w in seqs]),
                    omitted_prev_context=torch.zeros(nb, dtype=torch.int64),
                )
                for h in hs:
                    h.remove()

                # single forwards; for each seq capture submodule outputs, compare to batched slice
                first_leak = {}  # submodule -> max diff (per-seq aggregated)
                for i, (a, _, w) in enumerate(seqs):
                    store_s: Dict[str, Any] = {}
                    hs = hook_capture(store_s)
                    _ = model(
                        raw_inputs=a[None],
                        raw_inputs_sample_rate=sr,
                        raw_input_seq_lens=torch.tensor([int(a.shape[0])]),
                        raw_targets=[w],
                        raw_target_seq_lens=torch.tensor([len(w)]),
                        omitted_prev_context=torch.zeros(1, dtype=torch.int64),
                    )
                    for h in hs:
                        h.remove()
                    for nm, s in store_s.items():
                        b = store_b.get(nm)
                        if b is None or b.shape[0] != nb:
                            continue
                        # find the time axis = the axis whose single-len differs from batched-len
                        # (others should match). Compare batched[i, :L] vs single[0, :L] on that axis.
                        sb = b[i]
                        ss = s[0]
                        if sb.shape == ss.shape:
                            d = float((sb - ss).abs().max()) if sb.numel() else 0.0
                        else:
                            # slice each tensor's mismatched axis to the single's length
                            sl = tuple(slice(0, min(sb.shape[ax], ss.shape[ax])) for ax in range(sb.ndim))
                            d = float((sb[sl] - ss[sl]).abs().max()) if ss.numel() else 0.0
                        first_leak[nm] = max(first_leak.get(nm, 0.0), d)

                # report the leak-localization: submodules with nonzero diff, in module order
                ordered = [nm for nm, _ in model.named_modules() if nm in first_leak]
                report.append(f"\n### {name}: batched(B={nb}) vs single, max|Δ| per leaf submodule")
                nonzero = [(nm, first_leak[nm]) for nm in ordered if first_leak[nm] > 1e-6]
                if not nonzero:
                    report.append("  no submodule leak > 1e-6 (batched == single)")
                else:
                    report.append(f"  FIRST leaking submodule: {nonzero[0][0]}  max|Δ|={nonzero[0][1]:.3e}")
                    for nm, d in nonzero[:25]:
                        report.append(f"    {d:.3e}  {nm}")
                del model, store_b
                torch.cuda.empty_cache()
            except Exception:
                report.append(f"\n### {name}: SKIPPED/ERROR\n{traceback.format_exc()}")

        txt = "\n".join(report)
        print(txt)
        self.out_report.set(txt)
