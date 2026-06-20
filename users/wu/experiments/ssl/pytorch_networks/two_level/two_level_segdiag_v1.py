"""
Segmentation-diagnostic forward pass for the two-level BEST-RQ + CIF pretrained model.

Reuses the pretrain network ``two_level.two_level_v1.Model`` UNCHANGED (so a pretrained checkpoint loads
1:1); this module only provides a ``forward_step`` + ``ForwardCallback`` that run the model in eval with NO
span-masking and dump per-utterance CIF-segmentation diagnostics to ``seg_diag.pkl`` for offline analysis
(see ``experiments/ssl/analysis/seg_diag.py``). It optimises nothing.

Per utterance it records:
  * scalars: T (25 Hz frames), K (fired CIF tokens), sum_alpha (soft mass), N (SPM label count, if the
    forward dataset is labelled) -> realised frames/token (== ``fps_tok``) and the CTC-feasibility K/N margin;
  * for a capped subset of utterances (``max_examples``): the per-frame ``alpha`` curve, the per-frame
    ``fired`` boundary indicator, the per-frame layer-9 representation flux ``hflux`` (||h_t - h_{t-1}||,
    the boundary-quality signal), and the per-token target / unmasked-predicted codes;
  * a global target-code histogram + unmasked code-reconstruction agreement (codebook-usage / collapse).
"""

from __future__ import annotations

import numpy as np
import torch

import returnn.frontend as rf
from returnn.forward_iface import ForwardCallbackIface

from .parts.cif import cif_pool
from ..best_rq.parts.input_norm import apply_global_norm
from ..common.conformer import sequence_mask


def forward_step(*, model, extern_data, **kwargs):
    from returnn.tensor import Dim

    run_ctx = rf.get_run_ctx()

    audio = extern_data["audio"]
    batch_dim = audio.dims[0]
    raw_audio = audio.raw_tensor
    raw_audio_len = audio.dims[1].dyn_size_ext.raw_tensor.to(raw_audio.device)
    if raw_audio.dim() == 3:
        raw_audio = raw_audio.squeeze(-1)

    # SPM label length per utterance (only present when the forward dataset is labelled); -1 otherwise.
    n_len = None
    if "text" in extern_data and extern_data["text"].dims[1].dyn_size_ext is not None:
        n_len = extern_data["text"].dims[1].dyn_size_ext.raw_tensor.to(raw_audio.device)

    # CIF integrate-and-fire MUST be fp32 (bf16 cumsum saturates -> corrupt tokens); mirror the pretrain forward.
    with torch.autocast(device_type=raw_audio.device.type, enabled=False):
        model.feature_extraction.eval()
        model.encoder.eval()
        model.cif_alpha.eval()
        model.high_encoder.eval()
        with torch.no_grad():
            features, feat_len = model.feature_extraction(raw_audio, raw_audio_len)
            normed = apply_global_norm(features, feat_len, model.global_mean, model.global_std)
            seq_mask = sequence_mask(normed, feat_len)
            enc_layers, out_mask = model.encoder(
                normed, seq_mask, return_layers=[model.cfg.lower_layer_index]
            )
            h = enc_layers[-1].float()  # [B, T, D] layer-9 @ 25 Hz
            frame_len = out_mask.sum(dim=1).long()  # [B] T

            alpha = model.cif_alpha(h)  # [B, T] in (0,1)
            z, z_mask, diag = cif_pool(h, alpha, frame_len)  # z [B, K, D]

            zn = torch.nn.functional.normalize(z, dim=-1)
            targets = (zn @ model.codebook.t()).argmax(dim=-1)  # [B, K] nearest frozen centroid
            hi_layers, _ = model.high_encoder(z, z_mask, return_layers=[model.num_high_layers - 1])
            preds = model.pred_head(hi_layers[-1]).argmax(dim=-1)  # [B, K] UNMASKED reconstruction

            # per-frame fire indicator (recomputed exactly as in cif_pool) -> token boundaries
            eps = 1e-6
            csum = torch.cumsum(alpha, dim=1)
            prev = csum - alpha
            fired = (torch.floor(csum + eps) > torch.floor(prev + eps)).float()  # [B, T]
            # layer-9 representation flux at 25 Hz: the boundary-quality signal (frame 0 := 0)
            hflux = torch.zeros_like(alpha)
            hflux[:, 1:] = (h[:, 1:] - h[:, :-1]).norm(dim=-1)

            K = z_mask.sum(dim=1).long()  # [B]
            sum_alpha = diag["sum_alpha"].float()  # [B]

    # ---- mark outputs (RETURNN trims each to its own dyn length for process_seq) ----
    frame_t = Dim(None, name="seg_frames")
    frame_t.dyn_size_ext = rf.convert_to_tensor(
        frame_len.to("cpu", torch.int32), dims=[batch_dim], dtype="int32"
    )
    for name, t in [("alpha", alpha.float()), ("fired", fired), ("hflux", hflux)]:
        rf.convert_to_tensor(t, dims=[batch_dim, frame_t], name=name).mark_as_output(
            name, shape=[batch_dim, frame_t]
        )

    # Per-frame layer-9 representation h ([B,T,D]) so the OFFLINE analysis can re-pool under CIF /
    # uniform / GOLD segmentations and measure codebook quant-error (the "does placement matter" gate).
    feat_d = Dim(h.shape[-1], name="seg_feat")
    rf.convert_to_tensor(h.float(), dims=[batch_dim, frame_t, feat_d], name="h").mark_as_output(
        "h", shape=[batch_dim, frame_t, feat_d]
    )

    tok_t = Dim(None, name="seg_tokens")
    tok_t.dyn_size_ext = rf.convert_to_tensor(K.to("cpu", torch.int32), dims=[batch_dim], dtype="int32")
    for name, t in [("targets", targets.to(torch.int32)), ("preds", preds.to(torch.int32))]:
        rf.convert_to_tensor(t, dims=[batch_dim, tok_t], name=name).mark_as_output(
            name, shape=[batch_dim, tok_t]
        )

    n_val = n_len.float() if n_len is not None else torch.full_like(sum_alpha, -1.0)
    stats = torch.stack([frame_len.float(), K.float(), sum_alpha, n_val], dim=-1)  # [B, 4]
    stat_d = Dim(4, name="seg_stats")
    rf.convert_to_tensor(stats, dims=[batch_dim, stat_d], name="stats").mark_as_output(
        "stats", shape=[batch_dim, stat_d]
    )


class ForwardCallback(ForwardCallbackIface):
    """Accumulate per-utterance segmentation diagnostics -> single ``seg_diag.pkl`` (consumed offline)."""

    def __init__(self, max_examples: int = 64, num_clusters: int = 128):
        self.max_examples = int(max_examples)
        self.num_clusters = int(num_clusters)

    def init(self, *, model):
        self.scalars = []  # (seq_tag, T, K, sum_alpha, N) for EVERY utterance
        self.examples = []  # heavy per-frame/-token arrays for the first ``max_examples`` utterances
        self.code_counts = np.zeros(self.num_clusters, dtype=np.int64)  # global target-code usage
        self.pred_match = 0  # unmasked preds == targets (code reconstruction)
        self.tok_total = 0

    def process_seq(self, *, seq_tag: str, outputs):
        st = np.asarray(outputs["stats"].raw_tensor).reshape(-1)
        T, K, sa, N = float(st[0]), int(st[1]), float(st[2]), float(st[3])
        self.scalars.append((str(seq_tag), T, K, sa, N))

        tgt = np.asarray(outputs["targets"].raw_tensor).astype(np.int64).reshape(-1)
        prd = np.asarray(outputs["preds"].raw_tensor).astype(np.int64).reshape(-1)
        if tgt.size:
            self.code_counts += np.bincount(tgt, minlength=self.num_clusters)
            m = min(tgt.size, prd.size)
            self.pred_match += int((tgt[:m] == prd[:m]).sum())
            self.tok_total += m

        if len(self.examples) < self.max_examples:
            T = int(np.asarray(outputs["hflux"].raw_tensor).reshape(-1).size)
            self.examples.append(
                {
                    "seq_tag": str(seq_tag),
                    "alpha": np.asarray(outputs["alpha"].raw_tensor, dtype=np.float32).reshape(-1),
                    "fired": np.asarray(outputs["fired"].raw_tensor, dtype=np.float32).reshape(-1),
                    "hflux": np.asarray(outputs["hflux"].raw_tensor, dtype=np.float32).reshape(-1),
                    # layer-9 frames [T, D] in fp16 (offline re-pooling for the quant-error gate)
                    "h": np.asarray(outputs["h"].raw_tensor, dtype=np.float16).reshape(T, -1),
                    "targets": tgt,
                    "preds": prd,
                }
            )

    def finish(self):
        import pickle

        with open("seg_diag.pkl", "wb") as f:
            pickle.dump(
                {
                    "scalars": self.scalars,
                    "examples": self.examples,
                    "code_counts": self.code_counts,
                    "pred_match": self.pred_match,
                    "tok_total": self.tok_total,
                    "num_clusters": self.num_clusters,
                },
                f,
            )
