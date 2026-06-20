"""
Two-level BEST-RQ + fixed MEAN-POOL self-supervised model (v1), RETURNN/torch -- the CONTROL for the
learned CIF segmenter.

Identical to ``two_level_v1`` EXCEPT the CIF segmenter (alpha predictor + integrate-and-fire) is replaced
by a trivial, NON-learned fixed sub-sampling: mean-pool every ``factor`` consecutive layer-9 frames into
one token (``factor = round(frame_rate / target_rate)``; target_rate=12.5 Hz => factor 2 => 80 ms tokens,
matching the 80 ms CIF arm's rate). There is no ``alpha`` and no quantity loss -- the rate is fixed by
construction. Everything else is byte-for-byte the same as the CIF pretrain: FROZEN log-mel + 9-layer
lower encoder, the SAME frozen frame-level k-means target codebook, span-masked prediction over the high
9-layer Conformer + prediction head, CE on masked valid tokens.

This isolates the question: does the LEARNED, adaptive CIF segmentation buy anything over a dumb uniform
2x pool at the same token rate? (Compared against ``ssl/pretrain_two_level/ls960_cif80ms_k128``.)

Flow (forward):
  raw audio -> log-mel (no_grad) -> global input-norm
    -> FROZEN lower encoder (9-layer rel-pos Conformer), layer-9 output h [B, T, 512] @ 25 Hz
    -> mean_pool(h, factor): z [B, K, 512], z_mask [B, K]   (K = ceil(T / factor), NOT learned)
    -> targets c_i = argmax cosine(stopgrad(z_i), centroid)  (FROZEN k-means codebook)
    -> span-mask the tokens (replace masked with a learned mask embedding)
    -> HIGH encoder (9-layer rel-pos Conformer, NO frontend) -> prediction head -> logits [B, K, C]
  Loss = CE(masked valid tokens vs c).   (no quantity loss: the rate is fixed.)

fp32 / autocast OFF around the frozen lower forward + pooling: matches the CIF model so layer-9 + the
k-means assignment use the same geometry as the OFFLINE codebook fit (see [[cif-bf16-fp32]]); the mean
itself is harmless in bf16 but we keep the block identical for parity.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

import returnn.frontend as rf
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1

from .two_level_v1_cfg import TwoLevelConfig
from .parts.cif import mean_pool
from ..best_rq.parts.input_norm import apply_global_norm
from ..best_rq.parts.mask import compute_span_mask
from ..common.conformer import build_conformer_encoder, build_high_conformer_encoder, sequence_mask


class Model(nn.Module):
    def __init__(self, model_config_dict, codebook=None, **kwargs):
        """
        :param model_config_dict: asdict(TwoLevelConfig) -- the SAME config dataclass as the CIF model
            (cif_alpha_kernel_size / lambda_qty are inert here; target_rate_hz sets the pool factor).
        :param codebook: path to the frozen k-means centroids ``.npy`` [num_clusters, conformer_size]
            (a tk.Path -> string at runtime). None => random codebook (smoke only).
        """
        super().__init__()
        self.cfg = cfg = TwoLevelConfig.from_dict(model_config_dict)
        conf = cfg.encoder_config.conformer_size

        # ---- frozen lower stack: names match the BEST-RQ ckpt keys so it preloads 1:1 ----
        self.feature_extraction = LogMelFeatureExtractionV1(cfg=cfg.feature_extraction_config)
        self.encoder = build_conformer_encoder(cfg.encoder_config)  # 9 layers
        for p in self.feature_extraction.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.register_buffer("global_mean", torch.tensor(cfg.global_mean, dtype=torch.float32))
        self.register_buffer("global_std", torch.tensor(cfg.global_std, dtype=torch.float32))

        # ---- trainable: high encoder, mask embedding, prediction head (NO segmenter params) ----
        self.high_encoder = build_high_conformer_encoder(cfg.high_encoder_config)
        self.mask_emb = nn.Parameter(torch.zeros(conf))
        nn.init.normal_(self.mask_emb, std=0.02)
        self.pred_head = nn.Linear(conf, cfg.num_clusters)

        # ---- frozen high-level target codebook (L2-normalized), as a buffer ----
        self.register_buffer("codebook", self._load_codebook(codebook, cfg.num_clusters, conf))

        # DERIVED frame rate guard (== two_level_v1): 16 kHz / hop(160) / VGG 4x = 25 Hz.
        derived = (
            cfg.feature_extraction_config.sample_rate
            / (cfg.feature_extraction_config.hop_size * cfg.feature_extraction_config.sample_rate)
            / 4.0
        )
        assert abs(derived - cfg.frame_rate_hz) < 1e-6, f"frame_rate {cfg.frame_rate_hz} != derived {derived}"
        # Fixed pool factor from the target rate: factor=2 @ 12.5 Hz (80 ms), 3 @ 8.333 Hz (120 ms).
        factor = round(cfg.frame_rate_hz / cfg.target_rate_hz)
        assert (
            factor >= 1 and abs(factor - cfg.frame_rate_hz / cfg.target_rate_hz) < 1e-6
        ), f"target_rate {cfg.target_rate_hz} must divide frame_rate {cfg.frame_rate_hz} for mean-pool"
        self.pool_factor = factor
        self.num_high_layers = cfg.high_encoder_config.num_layers

    @staticmethod
    def _load_codebook(codebook, num_clusters: int, dim: int) -> torch.Tensor:
        if codebook is None:
            cb = torch.randn(num_clusters, dim)  # smoke only
        elif isinstance(codebook, str):
            cb = torch.tensor(np.load(codebook), dtype=torch.float32)
        else:
            cb = torch.tensor(np.asarray(codebook), dtype=torch.float32)
        assert cb.shape == (num_clusters, dim), f"codebook {tuple(cb.shape)} != ({num_clusters},{dim})"
        return torch.nn.functional.normalize(cb, dim=-1)

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        """:return: (logits [B,K,C], targets c [B,K], loss_mask [B,K], z_mask [B,K], diag, frame_len [B])."""
        if raw_audio.dim() == 3:
            raw_audio = raw_audio.squeeze(-1)

        with torch.autocast(device_type=raw_audio.device.type, enabled=False):
            self.feature_extraction.eval()
            self.encoder.eval()
            with torch.no_grad():
                features, feat_len = self.feature_extraction(raw_audio, raw_audio_len)  # [B, Tf, 80]
                normed = apply_global_norm(features, feat_len, self.global_mean, self.global_std)
                seq_mask = sequence_mask(normed, feat_len)
                enc_layers, out_mask = self.encoder(normed, seq_mask, return_layers=[self.cfg.lower_layer_index])
                h = enc_layers[-1]  # [B, T, 512] layer-9 features @ 25 Hz
            h = h.detach().float()
            frame_len = out_mask.sum(dim=1).long()  # [B]

            # ---- fixed mean-pool (NO learned segmenter) ----
            z, z_mask, diag = mean_pool(h, frame_len, factor=self.pool_factor)  # z [B, K, 512] fp32

            # ---- targets: nearest frozen centroid of L2-normed z (stop-grad) ----
            with torch.no_grad():
                zn = torch.nn.functional.normalize(z.detach(), dim=-1)
                targets = (zn @ self.codebook.t()).argmax(dim=-1)  # [B, K]

        # ---- span mask over the pooled tokens; targets minted BEFORE masking (HuBERT-style) ----
        b, k = z_mask.shape
        token_len = z_mask.sum(dim=1)
        span = compute_span_mask(
            batch_size=b,
            time_size=k,
            lengths=token_len,
            mask_prob=self.cfg.mask_prob,
            mask_length=self.cfg.mask_length,
            device=z.device,
            min_masks=self.cfg.min_masks,
        )
        loss_mask = span & z_mask
        z_in = torch.where(loss_mask.unsqueeze(-1), self.mask_emb.to(z.dtype), z)

        # ---- HIGH encoder (no frontend) over the masked tokens -> prediction head ----
        hi_layers, _ = self.high_encoder(z_in, z_mask, return_layers=[self.num_high_layers - 1])
        logits = self.pred_head(hi_layers[-1])  # [B, K, num_clusters]
        return logits, targets, loss_mask, z_mask, diag, frame_len


def train_step(*, model: Model, extern_data, **kwargs):
    run_ctx = rf.get_run_ctx()

    audio = extern_data["audio"]
    raw_audio = audio.raw_tensor
    raw_audio_len = audio.dims[1].dyn_size_ext.raw_tensor.to(raw_audio.device)

    logits, targets, loss_mask, z_mask, diag, frame_len = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)

    masked_logits = logits[loss_mask]  # [M, C]
    masked_tgt = targets[loss_mask]  # [M]
    num_masked = masked_logits.shape[0]

    if num_masked == 0:
        # Degenerate batch: keep the CE graph connected at zero and tie mask_emb in (DDP grad-set invariant,
        # same reasoning as two_level_v1). Unreachable with min_masks>=1; removes the latent landmine.
        run_ctx.mark_as_loss(loss=(logits.sum() + model.mask_emb.sum()) * 0.0, name="ce", dims=[])
        return

    ce = torch.nn.functional.cross_entropy(masked_logits, masked_tgt)
    run_ctx.mark_as_loss(loss=ce, name="ce", dims=[])

    # --- diagnostics (scale=0): masked top-1 acc, realized frames/token (== fixed pool factor), code entropy ---
    with torch.no_grad():
        acc = (masked_logits.argmax(dim=-1) == masked_tgt).float().mean()
        fps_tok = frame_len.sum().float() / z_mask.sum().clamp(min=1).float()
        counts = torch.bincount(targets[z_mask], minlength=model.cfg.num_clusters).float()
        p = counts / counts.sum().clamp(min=1.0)
        nz = p[p > 0]
        code_ent = (-(nz * nz.log()).sum()) / float(np.log(model.cfg.num_clusters))
    run_ctx.mark_as_loss(loss=acc, name="acc", dims=[], as_error=True, scale=0.0)
    run_ctx.mark_as_loss(loss=fps_tok, name="fps_tok", dims=[], as_error=True, scale=0.0)
    run_ctx.mark_as_loss(loss=code_ent, name="code_ent", dims=[], as_error=True, scale=0.0)
