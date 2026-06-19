"""
Two-level BEST-RQ + CIF self-supervised model (v1), RETURNN/torch.

Flow (forward):
  raw audio -> log-mel (no_grad) -> global input-norm
    -> FROZEN lower encoder (9-layer rel-pos Conformer), take layer-9 output h  [B, T, 512] @ 25 Hz
    -> CIF predictor: alpha_t = sigmoid(LocalConv(h))                            (trainable)
    -> CIF pooling (raw integrate-and-fire): z [B, K, 512], z_mask [B, K]        (trainable via alpha)
    -> targets c_i = argmin_k || L2norm(stopgrad(z_i)) - centroid_k ||  (FROZEN k-means codebook)
    -> span-mask the CIF tokens (replace masked with a learned mask embedding)
    -> HIGH encoder (9-layer rel-pos Conformer, NO frontend) -> prediction head -> logits [B, K, C]
  Loss = CE(masked valid tokens vs c) + lambda_qty * quantity_loss(sum alpha vs rate target).

Design decisions (see memory [[cif-length-control-and-finetune]]): pretraining uses RAW integrate-and-fire
(no alpha scaling/cap/global-norm); only the soft quantity loss controls rate. The lower encoder + log-mel
are FROZEN (requires_grad=False AND forced .eval() every forward so dropout is off in train and eval ->
layer-9 / alpha / K / target codes are deterministic functions of the audio). The codebook is built
offline and frozen; the NN assignment is a stop-grad target (the codebook never trains online).

Logged each step (kept DELIBERATELY SMALL so CE/acc stay readable, see [[readable-loss-logging]]):
  optimized: ``ce`` (masked high-level CE), ``qty`` (quantity loss, scaled by lambda_qty);
  diagnostics (scale=0): ``acc`` (masked top-1), ``fps_tok`` (realized frames/token == rate dial),
  ``code_ent`` (normalized entropy of the target codes -- the cheap collapse monitor).
The full metric suite (phone/word purity, PNMI, boundary/cohesion, codebook detail) lives in OFFLINE
analysis jobs, not the training log.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

import returnn.frontend as rf
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1

from .two_level_v1_cfg import TwoLevelConfig
from .parts.cif import CIFAlphaPredictor, cif_pool, quantity_loss
from ..best_rq.parts.input_norm import apply_global_norm
from ..best_rq.parts.mask import compute_span_mask
from ..common.conformer import build_conformer_encoder, build_high_conformer_encoder, sequence_mask


class Model(nn.Module):
    def __init__(self, model_config_dict, codebook=None, **kwargs):
        """
        :param model_config_dict: asdict(TwoLevelConfig).
        :param codebook: path to a ``.npy`` of frozen k-means centroids [num_clusters, conformer_size]
            (a Sisyphus tk.Path -> resolved to a string at runtime). If None, a random normalized
            codebook is used (smoke/pipeline test ONLY -- not a valid SSL target).
        """
        super().__init__()
        self.cfg = cfg = TwoLevelConfig.from_dict(model_config_dict)
        conf = cfg.encoder_config.conformer_size

        # ---- frozen lower stack: name 'feature_extraction'/'encoder' to match the BEST-RQ ckpt keys ----
        self.feature_extraction = LogMelFeatureExtractionV1(cfg=cfg.feature_extraction_config)
        self.encoder = build_conformer_encoder(cfg.encoder_config)  # 9 layers -> keys subset of ckpt
        for p in self.feature_extraction.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.register_buffer("global_mean", torch.tensor(cfg.global_mean, dtype=torch.float32))
        self.register_buffer("global_std", torch.tensor(cfg.global_std, dtype=torch.float32))

        # ---- trainable: CIF predictor, high encoder, mask embedding, prediction head ----
        self.cif_alpha = CIFAlphaPredictor(conf, kernel_size=cfg.cif_alpha_kernel_size)
        self.high_encoder = build_high_conformer_encoder(cfg.high_encoder_config)
        self.mask_emb = nn.Parameter(torch.zeros(conf))
        nn.init.normal_(self.mask_emb, std=0.02)
        self.pred_head = nn.Linear(conf, cfg.num_clusters)

        # ---- frozen high-level target codebook (L2-normalized), as a buffer ----
        self.register_buffer("codebook", self._load_codebook(codebook, cfg.num_clusters, conf))

        # DERIVED frame rate: 16 kHz / (hop 0.01 s * 16 kHz = 160) / VGG 4x subsample = 25 Hz. Assert it
        # so a wrong constant (the 50 Hz the original plan assumed) can't silently mistune Q* / the rate.
        derived = cfg.feature_extraction_config.sample_rate / (
            cfg.feature_extraction_config.hop_size * cfg.feature_extraction_config.sample_rate
        ) / 4.0
        assert abs(derived - cfg.frame_rate_hz) < 1e-6, f"frame_rate {cfg.frame_rate_hz} != derived {derived}"
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
        return torch.nn.functional.normalize(cb, dim=-1)  # cosine geometry (matches the offline build)

    @torch.no_grad()
    def extract_layer9(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        """Frozen lower-stack forward -> (h [B, T, 512] layer-9 features @ 25 Hz, frame_len [B]).

        Shared by the offline k-means codebook build and frozen probes, so the EXACT same features/norm
        feed the codebook fit and the train-time CIF assignment (no offline/online operator drift)."""
        if raw_audio.dim() == 3:
            raw_audio = raw_audio.squeeze(-1)
        self.feature_extraction.eval()
        self.encoder.eval()
        features, feat_len = self.feature_extraction(raw_audio, raw_audio_len)
        normed = apply_global_norm(features, feat_len, self.global_mean, self.global_std)
        seq_mask = sequence_mask(normed, feat_len)
        enc_layers, out_mask = self.encoder(normed, seq_mask, return_layers=[self.cfg.lower_layer_index])
        return enc_layers[-1], out_mask.sum(dim=1).long()

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        """:return: (logits [B,K,C], targets c [B,K], loss_mask [B,K], z_mask [B,K], diag, frame_len [B])."""
        if raw_audio.dim() == 3:
            raw_audio = raw_audio.squeeze(-1)

        # The frozen layer-9 features AND the CIF integrate-and-fire MUST run in fp32: RETURNN wraps the
        # whole train_step in bf16 autocast, and bf16 cumsum saturates (ULP > a typical alpha once the
        # running mass passes ~64) -> the fire bookkeeping silently corrupts the tokens AND the k-means
        # targets, with no error. Disabling autocast here also makes layer-9 fp32, matching the OFFLINE
        # codebook fit (also fp32) so the train-time nearest-centroid assignment uses the same geometry.
        with torch.autocast(device_type=raw_audio.device.type, enabled=False):
            # ---- FROZEN lower encoder: force eval() each step (RETURNN calls model.train()), no_grad ----
            self.feature_extraction.eval()
            self.encoder.eval()
            with torch.no_grad():
                features, feat_len = self.feature_extraction(raw_audio, raw_audio_len)  # [B, Tf, 80]
                normed = apply_global_norm(features, feat_len, self.global_mean, self.global_std)
                seq_mask = sequence_mask(normed, feat_len)  # [B, Tf] True=valid
                enc_layers, out_mask = self.encoder(normed, seq_mask, return_layers=[self.cfg.lower_layer_index])
                h = enc_layers[-1]  # [B, T, 512] layer-9 features @ 25 Hz
            h = h.detach().float()
            frame_len = out_mask.sum(dim=1).long()  # [B] valid 25 Hz frame counts

            # ---- CIF: predict alpha (trainable) and pool h into variable-K tokens (fp32) ----
            alpha = self.cif_alpha(h)                       # [B, T] in (0,1)
            z, z_mask, diag = cif_pool(h, alpha, frame_len)  # z [B, K, 512] fp32

            # ---- targets: nearest frozen centroid of L2-normed z (stop-grad) ----
            with torch.no_grad():
                zn = torch.nn.functional.normalize(z.detach(), dim=-1)
                targets = (zn @ self.codebook.t()).argmax(dim=-1)  # [B, K]

        # ---- span mask over CIF tokens; targets minted BEFORE masking (HuBERT-style) ----
        b, k = z_mask.shape
        token_len = z_mask.sum(dim=1)  # [B] valid token counts (== diag['fires'])
        span = compute_span_mask(
            batch_size=b, time_size=k, lengths=token_len,
            mask_prob=self.cfg.mask_prob, mask_length=self.cfg.mask_length,
            device=z.device, min_masks=self.cfg.min_masks,
        )
        loss_mask = span & z_mask  # masked AND valid
        z_in = torch.where(loss_mask.unsqueeze(-1), self.mask_emb.to(z.dtype), z)

        # ---- HIGH encoder (no frontend) over masked CIF tokens -> prediction head ----
        hi_layers, _ = self.high_encoder(z_in, z_mask, return_layers=[self.num_high_layers - 1])
        logits = self.pred_head(hi_layers[-1])  # [B, K, num_clusters]
        return logits, targets, loss_mask, z_mask, diag, frame_len


def train_step(*, model: Model, extern_data, **kwargs):
    run_ctx = rf.get_run_ctx()

    audio = extern_data["audio"]
    raw_audio = audio.raw_tensor
    raw_audio_len = audio.dims[1].dyn_size_ext.raw_tensor.to(raw_audio.device)

    logits, targets, loss_mask, z_mask, diag, frame_len = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)

    # --- quantity loss (always emitted: dense, computable every step on every rank -> DDP-safe) ---
    qty = quantity_loss(
        diag["sum_alpha"], frame_len,
        target_rate_hz=model.cfg.target_rate_hz, frame_rate_hz=model.cfg.frame_rate_hz,
    )
    run_ctx.mark_as_loss(loss=qty, name="qty", dims=[], scale=model.cfg.lambda_qty)

    masked_logits = logits[loss_mask]  # [M, C]
    masked_tgt = targets[loss_mask]    # [M]
    num_masked = masked_logits.shape[0]

    if num_masked == 0:
        # Degenerate batch (K too small / no span landed): keep the CE graph connected at zero so every
        # rank still marks 'ce' AND every trainable param stays in the backward graph (the reduce_type=grad
        # /DDP find_unused_parameters=False invariant: identical grad'd-param set every step). mask_emb is
        # otherwise only reached via torch.where on masked positions, so it MUST be tied in here explicitly
        # (else one empty-mask batch on one rank desyncs the reducer -> NCCL-timeout hang). With min_masks>=1
        # this path is unreachable today, but the tie removes the latent landmine for free.
        run_ctx.mark_as_loss(loss=(logits.sum() + model.mask_emb.sum()) * 0.0, name="ce", dims=[])
        return

    ce = torch.nn.functional.cross_entropy(masked_logits, masked_tgt)
    run_ctx.mark_as_loss(loss=ce, name="ce", dims=[])

    # --- small set of non-optimized diagnostics (scale=0) ---
    with torch.no_grad():
        acc = (masked_logits.argmax(dim=-1) == masked_tgt).float().mean()
        # realized frames/token: the rate dial (target 2.0 @80ms / 3.0 @120ms). token-weighted.
        fps_tok = frame_len.sum().float() / z_mask.sum().clamp(min=1).float()
        # normalized entropy of the target codes over valid tokens: cheap collapse monitor (-> 0 = collapse)
        counts = torch.bincount(targets[z_mask], minlength=model.cfg.num_clusters).float()
        p = counts / counts.sum().clamp(min=1.0)
        nz = p[p > 0]
        code_ent = (-(nz * nz.log()).sum()) / float(np.log(model.cfg.num_clusters))
    run_ctx.mark_as_loss(loss=acc, name="acc", dims=[], as_error=True, scale=0.0)
    run_ctx.mark_as_loss(loss=fps_tok, name="fps_tok", dims=[], as_error=True, scale=0.0)
    run_ctx.mark_as_loss(loss=code_ent, name="code_ent", dims=[], as_error=True, scale=0.0)
