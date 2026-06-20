"""
Two-level BEST-RQ + CIF with a Paraformer-style NAR per-token CE FINETUNE head (FT-2), RETURNN/torch.

The CE counterpart of the FT-1 CTC head (``two_level_ctc_v1``). Instead of CTC over a ~2-3x-too-long CIF
token sequence, this trains the CIF segmenter to fire ONE token per output label and applies a direct
cross-entropy on the text, exactly like Paraformer's first-pass NAR predictor:

  TRAIN (targets known):
    raw audio -> FROZEN log-mel + 9-layer lower encoder -> layer-9 h [B, T, 512] @ 25 Hz
    -> CIF predictor alpha = sigmoid(LocalConv(h))                              (TRAINABLE)
    -> cif_pool_scaled(h, alpha, frame_len, target_lengths): rescale alpha so it integrates to EXACTLY
       N_b = #labels and fires N_b tokens -> z [B, N_max, 512] aligned 1:1 with the N labels
    -> HIGH encoder (9-layer, no frontend) -> classifier -> logits [B, N, vocab]
    Loss = CE(per-token logits vs labels) + lambda_qty * quantity_loss(RAW sum alpha vs N)
           (+ InterCTC-style aux CE at high layers 4 & 8).
    The CE gradient (through the scaled convex weights) shapes WHERE the boundaries fall; the quantity
    loss pulls the UNSCALED fire count to N so inference needs no scaling.

  INFER (no targets):
    raw integrate-and-fire ``cif_pool`` -> K ~= N tokens (the quantity loss made it so) -> argmax each
    token -> SPM-decode the N ids (NAR: NO CTC collapse, NO blank). -> sclite WER downstream.

Two regimes (``freeze_segmenter`` net-arg), mirroring FT-1:
  * False (default, the point of FT-2): CIF predictor TRAINABLE -> it RETUNES from the pretrained ~12.5/
    8.33 Hz rate down to the SPM label rate (~one fire per BPE token). This is a large but intended shift.
  * True: CIF frozen (the pretrained segmentation is held; a probe / ablation).

The frozen lower forward + the CIF integrate-and-fire run with autocast DISABLED (fp32): under RETURNN's
bf16 autocast the cumsum saturates and silently corrupts the tokens (see [[cif-bf16-fp32]]).

Logged each step (small, see [[readable-loss-logging]]): optimized ``ce`` (final head) + aux ``ce_aux4``/
``ce_aux8`` (scale 0.3) + ``qty`` (retargeted quantity loss); diagnostics (scale=0) ``acc`` (teacher-forced
per-token top-1), ``raw_tok_per_lab`` (UNSCALED fires / labels -> must approach 1.0 for NAR inference to
land on the label count), ``fps_tok`` (realized frames/token).
"""

from __future__ import annotations

import torch
from torch import nn

import returnn.frontend as rf
from returnn.forward_iface import ForwardCallbackIface

from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1
from i6_models.primitives.specaugment import specaugment_v1_by_length

from .two_level_v1_cfg import TwoLevelConfig
from .parts.cif import CIFAlphaPredictor, cif_pool, cif_pool_scaled
from ..best_rq.parts.input_norm import apply_global_norm
from ..common.conformer import build_conformer_encoder, build_high_conformer_encoder, sequence_mask


class Model(nn.Module):
    def __init__(self, model_config_dict, vocab_size, freeze_segmenter=False, specaug_start_step=None, **kwargs):
        """
        :param model_config_dict: asdict(TwoLevelConfig) -- SAME architecture config as the pretrained
            two-level model, so its lower/CIF/high keys preload 1:1 from the ckpt.
        :param vocab_size: SPM vocab size. The CE head is a plain ``vocab_size``-way classifier (NO blank).
        :param freeze_segmenter: if True, freeze the CIF alpha predictor (probe; default False = retune it).
        :param specaug_start_step: if set, SpecAugment on the log-mel input from this global step (train-only).
        """
        super().__init__()
        self.cfg = cfg = TwoLevelConfig.from_dict(model_config_dict)
        conf = cfg.encoder_config.conformer_size
        self.vocab_size = int(vocab_size)  # classes 0..vocab_size-1 (no blank for NAR CE)
        self.freeze_segmenter = bool(freeze_segmenter)
        self.specaug_start_step = None if specaug_start_step is None else int(specaug_start_step)

        # ---- frozen lower stack (names match the pretrained two-level / BEST-RQ ckpt keys) ----
        self.feature_extraction = LogMelFeatureExtractionV1(cfg=cfg.feature_extraction_config)
        self.encoder = build_conformer_encoder(cfg.encoder_config)  # 9 layers
        for p in self.feature_extraction.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.register_buffer("global_mean", torch.tensor(cfg.global_mean, dtype=torch.float32))
        self.register_buffer("global_std", torch.tensor(cfg.global_std, dtype=torch.float32))

        # ---- CIF segmenter (trainable unless frozen) + high encoder + CE classifier ----
        self.cif_alpha = CIFAlphaPredictor(conf, kernel_size=cfg.cif_alpha_kernel_size)
        if self.freeze_segmenter:
            for p in self.cif_alpha.parameters():
                p.requires_grad = False
        self.high_encoder = build_high_conformer_encoder(cfg.high_encoder_config)
        # ``out_linear`` / ``aux_linears`` names match the two-rate head LR multiplier patterns (run at Nx).
        self.out_linear = nn.Linear(conf, self.vocab_size)  # final CE head (scale 1.0), NO blank
        self.aux_layer_idx = [3, 7]  # aux CE at high-encoder layers 4 & 8 (0-indexed), == the CIF CTC FT
        self.aux_scales = [0.3, 0.3]
        assert max(self.aux_layer_idx) < cfg.high_encoder_config.num_layers - 1, "aux layers must be below the final"
        self.aux_linears = nn.ModuleList([nn.Linear(conf, self.vocab_size) for _ in self.aux_layer_idx])

        derived = (
            cfg.feature_extraction_config.sample_rate
            / (cfg.feature_extraction_config.hop_size * cfg.feature_extraction_config.sample_rate)
            / 4.0
        )
        assert abs(derived - cfg.frame_rate_hz) < 1e-6, f"frame_rate {cfg.frame_rate_hz} != derived {derived}"
        self.num_high_layers = cfg.high_encoder_config.num_layers

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor, target_lengths=None):
        """:return: (log_probs [B, K, vocab], token_lengths [B], aux_log_probs, diag).

        ``target_lengths`` given (train/eval): SCALED integrate-and-fire -> exactly N tokens per utterance
        (z_mask marks N_b valid, axis aligns 1:1 with the labels). None (recog): RAW integrate-and-fire.
        """
        if raw_audio.dim() == 3:
            raw_audio = raw_audio.squeeze(-1)

        with torch.autocast(device_type=raw_audio.device.type, enabled=False):
            self.feature_extraction.eval()
            self.encoder.eval()
            with torch.no_grad():
                features, feat_len = self.feature_extraction(raw_audio, raw_audio_len)  # [B, Tf, 80]
                normed = apply_global_norm(features, feat_len, self.global_mean, self.global_std)
                if (
                    self.training
                    and self.specaug_start_step is not None
                    and rf.get_run_ctx().step >= self.specaug_start_step
                ):
                    normed = specaugment_v1_by_length(
                        normed,
                        time_min_num_masks=2,
                        time_max_mask_per_n_frames=25,
                        time_mask_max_size=20,
                        freq_min_num_masks=2,
                        freq_mask_max_size=16,
                        freq_max_num_masks=5,
                    )
                seq_mask = sequence_mask(normed, feat_len)
                enc_layers, out_mask = self.encoder(normed, seq_mask, return_layers=[self.cfg.lower_layer_index])
                h = enc_layers[-1]  # [B, T, 512] layer-9 features @ 25 Hz
            h = h.detach().float()
            frame_len = out_mask.sum(dim=1).long()

            alpha = self.cif_alpha(h)  # [B, T] in (0,1)
            if target_lengths is not None:
                tl = target_lengths.to(device=h.device, dtype=torch.long)
                z, z_mask, diag = cif_pool_scaled(h, alpha, frame_len, tl)  # z [B, N, 512] aligned to labels
            else:
                z, z_mask, diag = cif_pool(h, alpha, frame_len)  # raw; K ~= N via the quantity loss

        # ---- high encoder (no frontend) over the tokens -> CE heads ----
        final_idx = self.num_high_layers - 1
        want = sorted(set(self.aux_layer_idx + [final_idx]))
        hi_layers, _ = self.high_encoder(z, z_mask, return_layers=want)
        by_idx = {li: hi_layers[i] for i, li in enumerate(want)}
        log_probs = torch.log_softmax(self.out_linear(by_idx[final_idx]), dim=-1)  # [B, K, vocab]
        token_lengths = z_mask.sum(dim=1)
        aux_log_probs = {}
        for head, li in zip(self.aux_linears, self.aux_layer_idx):
            aux_log_probs[li + 1] = torch.log_softmax(head(by_idx[li]), dim=-1)
        return log_probs, token_lengths, aux_log_probs, diag


def _retargeted_quantity_loss(sum_alpha, target_lengths, *, eps: float = 1e-6):
    """Pull the UNSCALED accumulated mass toward the label count N (so inference fires ~N without scaling).

    Same relative-squared form as the pretraining quantity loss, but the target is N (the #labels), not a
    fixed-rate count: ``mean_b ((sum_alpha_b - N_b) / (N_b + eps))^2``."""
    n = target_lengths.to(sum_alpha.dtype)
    rel = (sum_alpha - n) / (n + eps)
    return (rel * rel).mean()


def train_step(*, model: Model, extern_data, **kwargs):
    run_ctx = rf.get_run_ctx()

    audio = extern_data["audio"]
    raw_audio = audio.raw_tensor
    raw_audio_len = audio.dims[1].dyn_size_ext.raw_tensor.to(raw_audio.device)

    text = extern_data["text"]
    targets = text.raw_tensor.long()  # [B, S]
    target_lengths = text.dims[1].dyn_size_ext.raw_tensor.to(device=raw_audio.device, dtype=torch.long)  # [B]

    log_probs, token_lengths, aux_log_probs, diag = model(
        raw_audio=raw_audio, raw_audio_len=raw_audio_len, target_lengths=target_lengths
    )
    # scaled pooling forces K_b = N_b, and N_max == S (targets are padded to the batch-max label count),
    # so the token axis aligns 1:1 with the targets. Mask the padded positions out of the CE.
    b, k, _ = log_probs.shape
    pos_valid = torch.arange(k, device=log_probs.device)[None, :] < target_lengths[:, None]  # [B, K]
    tgt = targets[:, :k]
    n_labels = target_lengths.sum().clamp(min=1)

    # retargeted quantity loss: drive the UNSCALED fire count to the label count (DDP-safe: dense, ties cif_alpha)
    qty = _retargeted_quantity_loss(diag["sum_alpha"], target_lengths)
    run_ctx.mark_as_loss(loss=qty, name="qty", dims=[], scale=model.cfg.lambda_qty)

    valid = pos_valid & (tgt >= 0)
    if valid.sum() == 0:
        # degenerate (no labels in batch): keep CE graph connected at zero and tie every trainable head
        # (DDP grad-set invariant; cif_alpha is already tied via qty). Unreachable on LS.
        tie = log_probs.sum() + sum(a.sum() for a in aux_log_probs.values())
        run_ctx.mark_as_loss(loss=tie * 0.0, name="ce", dims=[])
        return

    def _ce(lp):  # per-token NLL over valid positions, label-normalized (i6 canonical: sum / #labels)
        return torch.nn.functional.nll_loss(lp[valid], tgt[valid], reduction="sum") / n_labels

    run_ctx.mark_as_loss(loss=_ce(log_probs), name="ce", dims=[])  # final head, scale 1.0
    for li, scale in zip(model.aux_layer_idx, model.aux_scales):
        if (li + 1) in aux_log_probs:
            run_ctx.mark_as_loss(loss=_ce(aux_log_probs[li + 1]), name=f"ce_aux{li + 1}", scale=scale, dims=[])

    # --- diagnostics (scale=0): teacher-forced top-1 acc, the UNSCALED fire/label ratio, frames/token ---
    with torch.no_grad():
        acc = (log_probs[valid].argmax(dim=-1) == tgt[valid]).float().mean()
        raw_tok_per_lab = diag["sum_alpha"].sum() / target_lengths.float().sum().clamp(min=1)
        fps_tok = diag["frame_valid"].float().sum() / token_lengths.float().sum().clamp(min=1)
    run_ctx.mark_as_loss(loss=acc, name="acc", dims=[], as_error=True, scale=0.0)
    run_ctx.mark_as_loss(loss=raw_tok_per_lab, name="raw_tok_per_lab", dims=[], as_error=True, scale=0.0)
    run_ctx.mark_as_loss(loss=fps_tok, name="fps_tok", dims=[], as_error=True, scale=0.0)


# --------------------------- NAR greedy decoding (real-RETURNN forward) ---------------------------


def forward_step(*, model: Model, extern_data, **kwargs):
    """Run the AM (RAW CIF, no targets) and mark per-token log-probs with a dynamic time dim."""
    from returnn.tensor import Dim

    audio = extern_data["audio"]
    batch_dim = audio.dims[0]
    raw_audio = audio.raw_tensor
    raw_audio_len = audio.dims[1].dyn_size_ext.raw_tensor

    log_probs, out_lengths, _, _ = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len.to(raw_audio.device))

    time_dim = Dim(None, name="ce_out_time")
    time_dim.dyn_size_ext = rf.convert_to_tensor(out_lengths.to("cpu", torch.int32), dims=[batch_dim], dtype="int32")
    feat_dim = Dim(int(log_probs.shape[-1]), name="ce_labels")
    rf.convert_to_tensor(log_probs, dims=[batch_dim, time_dim, feat_dim], name="log_probs").mark_as_output(
        "log_probs", shape=[batch_dim, time_dim, feat_dim]
    )


class ForwardCallback(ForwardCallbackIface):
    """NAR CE decode: per-token argmax -> SPM-decode the K ids to words. NO CTC collapse, NO blank.

    Each fired CIF token is exactly one output label (Paraformer first-pass NAR), so we keep ALL K argmax
    ids in order. ``blank_idx`` is accepted for a uniform recog interface but unused (there is no blank)."""

    def __init__(self, spm_model_file: str, blank_idx: int):
        self._spm_model_file = spm_model_file
        self._blank_idx = int(blank_idx)  # unused (NAR head has no blank)

    def init(self, *, model):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=str(self._spm_model_file))
        self.recognition_file = open("search_out.py", "wt")
        self.recognition_file.write("{\n")

    def process_seq(self, *, seq_tag: str, outputs):
        import numpy as np

        lp = outputs["log_probs"].raw_tensor  # [K, vocab] numpy, padding removed
        ids = [int(i) for i in np.asarray(lp).argmax(axis=-1)]  # one id per token, in order (no collapse)
        hyp = self.sp.decode(ids)
        self.recognition_file.write("%s: %s,\n" % (repr(seq_tag), repr(hyp)))

    def finish(self):
        self.recognition_file.write("}\n")
        self.recognition_file.close()
