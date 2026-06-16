"""Probe: single-seq forward vs batched (B>1, padded) forward — does padding leak INTO the real
(non-masked) frames? For each leaf submodule we compare batched-output[i] vs single-output[i] on the
REAL frames only, split into:
  - INTERIOR: real frames excluding the last `margin` (a genuine masking bug if this leaks), and
  - BOUNDARY: the last `margin` real frames (where a conv/attn receptive field reaches the now-nonzero
    padded region -> maskable by re-zeroing padding before each cross-frame op).
Padded-region outputs are IGNORED (every bias-add makes them nonzero; that's harmless). The leak only
matters where it crosses into real frames: conv / pool / self-/cross-attn / the prefix sum.
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
    """Per-model single-vs-batched forward equivalence; interior-vs-boundary leak localization."""

    def __init__(
        self,
        *,
        model_configs: Dict[str, Any],
        dataset_dir: tk.Path,
        dataset_key: str,
        num_seqs: int = 10,
        margin: int = 4,
        probe_version: int = 2,
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
        report = [f"batch of {len(seqs)} seqs, sample lengths {lens}; margin={self.margin}"]

        def compare(sb, ss):
            """real-frame interior/boundary max|Δ|. sb=batched[i], ss=single[0] (no batch dim).
            time axis = the axis whose sizes differ (padded). Returns (interior, boundary, time_ax)."""
            if sb.shape == ss.shape:
                d = float((sb - ss).abs().max()) if sb.numel() else 0.0
                return d, 0.0, None  # no padded axis -> all "interior"
            diff_axes = [ax for ax in range(sb.ndim) if sb.shape[ax] != ss.shape[ax]]
            if len(diff_axes) != 1:
                sl = tuple(slice(0, min(sb.shape[ax], ss.shape[ax])) for ax in range(sb.ndim))
                d = float((sb[sl] - ss[sl]).abs().max()) if ss.numel() else 0.0
                return d, d, -1
            t = diff_axes[0]
            L = ss.shape[t]  # real length at this layer
            real_b = sb.index_select(t, torch.arange(L))
            diff = (real_b - ss).abs()
            m = min(self.margin, L)
            inter = diff.index_select(t, torch.arange(0, L - m)) if L - m > 0 else None
            bound = diff.index_select(t, torch.arange(L - m, L))
            inter_max = float(inter.max()) if inter is not None and inter.numel() else 0.0
            bound_max = float(bound.max()) if bound.numel() else 0.0
            return inter_max, bound_max, t

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

                def hook_capture(store):
                    handles = []

                    def mk(nm):
                        def h(_m, _inp, out):
                            t = out[0] if isinstance(out, tuple) else out
                            if isinstance(t, torch.Tensor):
                                store[nm] = t.detach().float().cpu()

                        return h

                    for nm, m in model.named_modules():
                        if nm and len(list(m.children())) == 0:
                            handles.append(m.register_forward_hook(mk(nm)))
                    return handles

                store_b: Dict[str, Any] = {}
                hs = hook_capture(store_b)
                _ = model(
                    raw_inputs=raw,
                    raw_inputs_sample_rate=sr,
                    raw_input_seq_lens=torch.tensor(lens),
                    raw_targets=[w for _, _, w in seqs],
                    raw_target_seq_lens=torch.tensor([len(w) for _, _, w in seqs]),
                    omitted_prev_context=torch.zeros(nb, dtype=torch.int64),
                )
                for h in hs:
                    h.remove()

                inter_agg: Dict[str, float] = {}
                bound_agg: Dict[str, float] = {}
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
                        inter, bound, _t = compare(b[i], s[0])
                        inter_agg[nm] = max(inter_agg.get(nm, 0.0), inter)
                        bound_agg[nm] = max(bound_agg.get(nm, 0.0), bound)

                ordered = [nm for nm, _ in model.named_modules() if nm in inter_agg]
                report.append(f"\n### {name}: real-frame max|Δ| (INTERIOR | BOUNDARY) per leaf submodule")
                interior_leaks = [nm for nm in ordered if inter_agg[nm] > 1e-5]
                report.append(
                    f"  >>> first INTERIOR leak (>1e-5, = real masking bug): "
                    f"{interior_leaks[0] if interior_leaks else 'NONE'}"
                )
                for nm in ordered:
                    iv, bv = inter_agg[nm], bound_agg[nm]
                    if iv > 1e-5 or bv > 1e-5:
                        flag = "  <-- INTERIOR" if iv > 1e-5 else ""
                        report.append(f"    int {iv:.2e} | bnd {bv:.2e}   {nm}{flag}")
                del model, store_b
                torch.cuda.empty_cache()
            except Exception:
                report.append(f"\n### {name}: SKIPPED/ERROR\n{traceback.format_exc()}")

        txt = "\n".join(report)
        print(txt)
        self.out_report.set(txt)
