"""
BEST-RQ self-supervised pretraining model (multi-codebook), RETURNN/torch.

Flow (forward):
  raw audio -> log-mel (no_grad) -> per-utterance input-norm
    |-> frozen multi-codebook quantizer  => targets [B, T2, N]   (T2 = T // stack)
    `-> span mask @ subsampled rate, expand x stack, replace masked input frames with
        N(0, noise_std) noise -> conformer encoder => enc_out [B, T_enc, C]
  lengths reconciled to T_common = min(T2, T_enc); loss computed only on masked & valid
  encoder positions, averaged cross-entropy over the N codebooks (Baumann et al. 2025).

The same input-normalized features feed both the quantizer (targets) and the (masked)
encoder input. Targets are produced under no_grad from a frozen quantizer, so no gradient
flows into them. Heads + loss are evaluated only on the gathered masked positions (cheap).

Logged each step: ``bestrq_ce`` (the optimized loss), plus non-optimized diagnostics
``masked_acc`` (masked-frame top-1 accuracy) and ``code_ppl`` (codebook perplexity on the
masked targets) -- a live codebook-utilization monitor (collapse would show as ppl -> 1).
"""

from __future__ import annotations

import torch
from torch import nn

import returnn.frontend as rf
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1

from .best_rq_v1_cfg import BestRQConfig
from .parts.input_norm import apply_global_norm
from .parts.quantizer import MultiCodebookRandomProjectionQuantizer
from .parts.mask import compute_span_mask
from ..common.conformer import build_conformer_encoder, sequence_mask


class Model(nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = BestRQConfig.from_dict(model_config_dict)
        num_filters = self.cfg.feature_extraction_config.num_filters
        conformer_size = self.cfg.encoder_config.conformer_size

        self.feature_extraction = LogMelFeatureExtractionV1(cfg=self.cfg.feature_extraction_config)
        self.quantizer = MultiCodebookRandomProjectionQuantizer(
            input_dim=num_filters,
            stack_size=self.cfg.stack_size,
            codebook_dim=self.cfg.codebook_dim,
            vocab_size=self.cfg.vocab_size,
            num_codebooks=self.cfg.num_codebooks,
            seed=self.cfg.quantizer_seed,
        )
        self.encoder = build_conformer_encoder(self.cfg.encoder_config)
        # one softmax head per codebook (discarded at finetuning)
        self.heads = nn.ModuleList(
            [nn.Linear(conformer_size, self.cfg.vocab_size) for _ in range(self.cfg.num_codebooks)]
        )

        # fixed global (corpus-level) log-mel normalization, registered as buffers
        self.register_buffer("global_mean", torch.tensor(self.cfg.global_mean, dtype=torch.float32))
        self.register_buffer("global_std", torch.tensor(self.cfg.global_std, dtype=torch.float32))

        self.num_codebooks = self.cfg.num_codebooks
        self.stack_size = self.cfg.stack_size

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        """
        :param raw_audio: [B, T] (or [B, T, 1]) raw 16 kHz samples.
        :param raw_audio_len: [B] sample counts.
        :return: (enc_out [B, Tc, C], targets [B, Tc, N], loss_mask [B, Tc] bool)
        """
        if raw_audio.dim() == 3:
            raw_audio = raw_audio.squeeze(-1)

        with torch.no_grad():
            features, feat_len = self.feature_extraction(raw_audio, raw_audio_len)  # [B, Tf, F]
            normed = apply_global_norm(features, feat_len, self.global_mean, self.global_std)  # [B, Tf, F]
            targets, target_len = self.quantizer(normed, feat_len)  # [B, T2, N], [B]

        b, tf, _ = normed.shape
        t2 = targets.shape[1]

        # Span mask at the subsampled rate. Applied in BOTH train and eval: masked prediction IS the
        # SSL objective, so the dev/CV loss must be computed on masked positions too -- otherwise
        # there are no targets and the dev score is identically 0 (breaking keep_best checkpoint
        # selection). Only dropout differs between train/eval (handled by module train()/eval()).
        mask = compute_span_mask(
            batch_size=b,
            time_size=t2,
            lengths=target_len,
            mask_prob=self.cfg.mask_prob,
            mask_length=self.cfg.mask_length,
            device=normed.device,
            min_masks=self.cfg.min_masks,
        )  # [B, T2] bool

        # Expand the subsampled mask x stack to corrupt the input feature frames, padding to Tf.
        mask_full = mask.repeat_interleave(self.stack_size, dim=1)  # [B, T2*stack]
        if mask_full.shape[1] < tf:
            pad = torch.zeros(b, tf - mask_full.shape[1], dtype=torch.bool, device=normed.device)
            mask_full = torch.cat([mask_full, pad], dim=1)
        else:
            mask_full = mask_full[:, :tf]

        noise = torch.randn_like(normed) * self.cfg.noise_std
        enc_in = torch.where(mask_full.unsqueeze(-1), noise, normed)  # [B, Tf, F]

        seq_mask = sequence_mask(enc_in, feat_len)  # [B, Tf] True=valid
        enc_layers, out_mask = self.encoder(enc_in, seq_mask, return_layers=[self.cfg.encoder_config.num_layers - 1])
        enc_out = enc_layers[-1]  # [B, T_enc, C]
        t_enc = enc_out.shape[1]

        # Reconcile target/mask time (T2) with encoder-output time (T_enc); they match up to a
        # one-frame conv/pool edge effect. Guard against any gross mismatch.
        assert abs(t_enc - t2) <= self.stack_size, f"encoder T={t_enc} vs target T2={t2} mismatch"
        tc = min(t_enc, t2)
        enc_out = enc_out[:, :tc]
        targets = targets[:, :tc]
        loss_mask = mask[:, :tc] & out_mask[:, :tc]  # masked AND valid (non-padding)
        return enc_out, targets, loss_mask


def train_step(*, model: Model, extern_data, **kwargs):
    run_ctx = rf.get_run_ctx()

    audio = extern_data["audio"]
    raw_audio = audio.raw_tensor  # [B, T] (raw float32 16 kHz samples)
    raw_audio_len = audio.dims[1].dyn_size_ext.raw_tensor.to(raw_audio.device)  # [B]

    enc_out, targets, loss_mask = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)

    masked_enc = enc_out[loss_mask]  # [M, C]
    masked_tgt = targets[loss_mask]  # [M, N]
    num_masked = masked_enc.shape[0]

    if num_masked == 0:
        # Degenerate batch (should not happen with min_masks>=1): keep the graph connected.
        zero = enc_out.sum() * 0.0
        run_ctx.mark_as_loss(loss=zero, name="bestrq_ce", dims=[])
        return

    ce_sum = masked_enc.new_zeros(())
    correct = 0
    for n in range(model.num_codebooks):
        logits_n = model.heads[n](masked_enc)  # [M, V]
        tgt_n = masked_tgt[:, n]
        ce_sum = ce_sum + torch.nn.functional.cross_entropy(logits_n, tgt_n)
        correct = correct + (logits_n.argmax(dim=-1) == tgt_n).sum()
    ce = ce_sum / model.num_codebooks
    run_ctx.mark_as_loss(loss=ce, name="bestrq_ce", dims=[])

    # --- non-optimized diagnostics ---
    acc = correct.float() / (num_masked * model.num_codebooks)
    run_ctx.mark_as_loss(loss=acc, name="masked_acc", dims=[], as_error=True, scale=0.0)

    # Codebook perplexity on the masked targets (averaged over codebooks): a live collapse monitor.
    with torch.no_grad():
        ppl_sum = masked_enc.new_zeros(())
        for n in range(model.num_codebooks):
            counts = torch.bincount(masked_tgt[:, n], minlength=model.cfg.vocab_size).float()
            p = counts / counts.sum().clamp(min=1.0)
            nz = p[p > 0]
            entropy = -(nz * nz.log()).sum()
            ppl_sum = ppl_sum + torch.exp(entropy)
        code_ppl = ppl_sum / model.num_codebooks
    run_ctx.mark_as_loss(loss=code_ppl, name="code_ppl", dims=[], as_error=True, scale=0.0)
