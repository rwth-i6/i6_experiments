"""
Two-level BEST-RQ + CIF model with a CTC FINETUNE head (FT-1), RETURNN/torch.

Reuses the pretrained two-level stack -- FROZEN lower 9-layer rel-pos Conformer (layer-9 @ 25 Hz) +
CIF segmenter + 9-layer high Conformer -- and swaps the masked-prediction head for a single CTC head
over the variable-length CIF token sequence. SPM-5120 labels, torch ``nn.functional.ctc_loss``, with
blank = last index = ``vocab_size`` (labels keep ``0..vocab_size-1``). See [[cif-length-control-and-finetune]].

Two finetune regimes, selected by the ``freeze_segmenter`` net-arg:
  * ``freeze_segmenter=True``  -> CIF alpha predictor FROZEN: the pretrained segmentation is held fixed,
    so the pooled tokens ``z`` are a constant function of the audio and only the high encoder + CTC head
    adapt. This is the clean probe of *pretrained segmentation* quality.
  * ``freeze_segmenter=False`` -> CIF alpha predictor TRAINABLE: the segmenter co-adapts to the CTC
    objective (the best-WER arm / ablation).
RAW integrate-and-fire either way (NO alpha scaling), matching pretraining -- so the realized token rate
stays ~2-3x the SPM label rate and CTC's ``K >= N`` holds. The lower stack (log-mel + 9-layer encoder +
global-norm) is ALWAYS frozen and forced ``eval`` (the model's premise; dropout off -> deterministic h).

The frozen lower forward + the CIF integrate-and-fire run with autocast DISABLED (fp32): under RETURNN's
bf16 autocast the cumsum saturates and silently corrupts the tokens (see [[cif-bf16-fp32]]).

Greedy decoding (``forward_step`` / ``ForwardCallback``) is identical to the base CTC decoder -- per-
position argmax over the K token slots -> collapse repeats -> drop blank -> SPM-decode to words, writing
the standard ``search_out.py`` -> sclite WER downstream.

Logged each step (kept small, see [[readable-loss-logging]]): optimized ``ctc`` (final head) + InterCTC
aux ``ctc_aux4``/``ctc_aux8`` (scale 0.3 each, imitating the base CTC finetune); diagnostics (scale=0)
``tok_per_lab`` (CIF tokens / labels; the CTC ``K>=N`` headroom), ``fps_tok`` (realized frames/token ==
the rate dial), ``infeasible`` (fraction of seqs with K < N -> CTC drops them via zero_infinity).
"""

from __future__ import annotations

import torch
from torch import nn

import returnn.frontend as rf
from returnn.forward_iface import ForwardCallbackIface

from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1
from i6_models.primitives.specaugment import specaugment_v1_by_length

from .two_level_v1_cfg import TwoLevelConfig
from .parts.cif import CIFAlphaPredictor, cif_pool, quantity_loss
from ..best_rq.parts.input_norm import apply_global_norm
from ..common.conformer import build_conformer_encoder, build_high_conformer_encoder, sequence_mask


class Model(nn.Module):
    def __init__(
        self,
        model_config_dict,
        vocab_size,
        freeze_segmenter=False,
        specaug_start_step=None,
        quantity_loss_scale=None,
        **kwargs,
    ):
        """
        :param model_config_dict: asdict(TwoLevelConfig) -- the SAME architecture config as the pretrained
            two-level model (only the head differs), so its lower/CIF/high keys load 1:1 from the ckpt.
        :param vocab_size: SPM vocab size; CTC labels are ``0..vocab_size-1`` and blank == ``vocab_size``.
        :param freeze_segmenter: if True, freeze the CIF alpha predictor (probe pretrained segmentation).
        :param specaug_start_step: if set, apply SpecAugment on the log-mel input once the global train step
            reaches this value (train-only, off at recog). Strength matches the base CTC finetune
            (conformer_ctc_v1: time ~40%, freq ~35%). None => off. The base BEST-RQ FT uses 2000; we imitate.
        :param quantity_loss_scale: if set (TRAINABLE-segmenter FT only), add the CIF quantity loss at this
            scale to ANCHOR the firing rate to the pretrained acoustic rate Q* = T*(target/frame). Without it
            a trainable CIF + zero_infinity CTC collapses the rate to K<N (a zero-loss attractor; diagnosed
            2026-06-20, see [[ft-collapse-diagnosis]]). None => off (frozen-seg arm needs no anchor).
        """
        super().__init__()
        self.cfg = cfg = TwoLevelConfig.from_dict(model_config_dict)
        conf = cfg.encoder_config.conformer_size
        self.vocab_size = int(vocab_size)
        self.blank_idx = self.vocab_size  # blank = last index
        self.freeze_segmenter = bool(freeze_segmenter)
        self.specaug_start_step = None if specaug_start_step is None else int(specaug_start_step)
        self.quantity_loss_scale = None if quantity_loss_scale is None else float(quantity_loss_scale)

        # ---- frozen lower stack (names match the pretrained two-level / BEST-RQ ckpt keys) ----
        self.feature_extraction = LogMelFeatureExtractionV1(cfg=cfg.feature_extraction_config)
        self.encoder = build_conformer_encoder(cfg.encoder_config)  # 9 layers
        for p in self.feature_extraction.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.register_buffer("global_mean", torch.tensor(cfg.global_mean, dtype=torch.float32))
        self.register_buffer("global_std", torch.tensor(cfg.global_std, dtype=torch.float32))

        # ---- CIF segmenter (trainable unless frozen) + high encoder + CTC head ----
        self.cif_alpha = CIFAlphaPredictor(conf, kernel_size=cfg.cif_alpha_kernel_size)
        if self.freeze_segmenter:
            for p in self.cif_alpha.parameters():
                p.requires_grad = False
        self.high_encoder = build_high_conformer_encoder(cfg.high_encoder_config)
        self.out_linear = nn.Linear(conf, self.vocab_size + 1)  # final-layer CTC head (scale 1.0), blank = last
        # InterCTC aux heads on intermediate HIGH-encoder layers -- imitates the base CTC finetune
        # (conformer_ctc_v1: aux at layers 4 & 8, scale 0.3 each). 9-layer high encoder -> 0-indexed [3, 7].
        self.aux_layer_idx = [3, 7]
        self.aux_scales = [0.3, 0.3]
        assert max(self.aux_layer_idx) < cfg.high_encoder_config.num_layers - 1, "aux layers must be below the final"
        self.aux_linears = nn.ModuleList([nn.Linear(conf, self.vocab_size + 1) for _ in self.aux_layer_idx])

        # DERIVED frame rate guard (== two_level_v1): 16 kHz / hop(160) / VGG 4x = 25 Hz.
        derived = (
            cfg.feature_extraction_config.sample_rate
            / (cfg.feature_extraction_config.hop_size * cfg.feature_extraction_config.sample_rate)
            / 4.0
        )
        assert abs(derived - cfg.frame_rate_hz) < 1e-6, f"frame_rate {cfg.frame_rate_hz} != derived {derived}"
        self.num_high_layers = cfg.high_encoder_config.num_layers

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        """:return: (log_probs [B, K, vocab+1], token_lengths [B], diag)."""
        if raw_audio.dim() == 3:
            raw_audio = raw_audio.squeeze(-1)

        # fp32 / autocast OFF: frozen lower encoder + CIF integrate-and-fire (bf16 cumsum saturates ->
        # silent token corruption; see [[cif-bf16-fp32]]). High encoder + head stay bf16 (outside).
        with torch.autocast(device_type=raw_audio.device.type, enabled=False):
            self.feature_extraction.eval()
            self.encoder.eval()
            with torch.no_grad():
                features, feat_len = self.feature_extraction(raw_audio, raw_audio_len)  # [B, Tf, 80]
                normed = apply_global_norm(features, feat_len, self.global_mean, self.global_std)
                # SpecAugment on the log-mel input (train-only, step-gated) -- imitates the base CTC
                # finetune (conformer_ctc_v1): SAME mask strength, applied before the FROZEN lower encoder
                # so it augments the features the CIF segmenter + high encoder + head all see. Off at recog.
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
            frame_len = out_mask.sum(dim=1).long()  # [B] valid 25 Hz frame counts

            # CIF: RAW integrate-and-fire (same as pretraining). When the segmenter is frozen, alpha has
            # no trainable params upstream -> z is a fixed function of the audio and grad reaches only the
            # high encoder + head; when trainable, grad flows CTC -> z -> alpha -> cif_alpha.
            alpha = self.cif_alpha(h)  # [B, T] in (0,1)
            z, z_mask, diag = cif_pool(h, alpha, frame_len)  # z [B, K, 512] fp32

        # ---- high encoder (no frontend) over the CIF tokens -> CTC heads (input SpecAugment above) ----
        final_idx = self.num_high_layers - 1
        want = sorted(set(self.aux_layer_idx + [final_idx]))
        hi_layers, _ = self.high_encoder(z, z_mask, return_layers=want)
        by_idx = {li: hi_layers[i] for i, li in enumerate(want)}
        log_probs = torch.log_softmax(self.out_linear(by_idx[final_idx]), dim=-1)  # [B, K, vocab+1]
        token_lengths = z_mask.sum(dim=1)  # [B] valid CIF token counts (== K_b)
        # aux InterCTC heads, computed in BOTH train AND eval (RETURNN runs train_step on dev too, in eval
        # mode; gating on self.training would empty this and crash train_step). Recog (forward_step) ignores.
        aux_log_probs = {}
        for head, li in zip(self.aux_linears, self.aux_layer_idx):
            aux_log_probs[li + 1] = torch.log_softmax(head(by_idx[li]), dim=-1)
        return log_probs, token_lengths, aux_log_probs, diag


def train_step(*, model: Model, extern_data, **kwargs):
    run_ctx = rf.get_run_ctx()

    audio = extern_data["audio"]
    raw_audio = audio.raw_tensor
    raw_audio_len = audio.dims[1].dyn_size_ext.raw_tensor.to(raw_audio.device)

    text = extern_data["text"]
    targets = text.raw_tensor.long()  # [B, S]
    target_lengths = text.dims[1].dyn_size_ext.raw_tensor.to(device=raw_audio.device, dtype=torch.int32)  # [B]

    log_probs, input_lengths, aux_log_probs, diag = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)

    # torch CTC: log_probs (T, B, C) float32; lengths on CPU int32 (cuDNN requirement). The "time" axis
    # here is the CIF token axis K (per-seq input_lengths), label-normalized (i6 canonical: sum / #labels).
    il_cpu = input_lengths.to(torch.int32).cpu()
    tl_cpu = target_lengths.cpu()
    n_labels = target_lengths.sum().clamp(min=1)

    def _ctc(lp):
        return (
            torch.nn.functional.ctc_loss(
                lp.transpose(0, 1).float(),
                targets,
                il_cpu,
                tl_cpu,
                blank=model.blank_idx,
                reduction="sum",
                zero_infinity=True,
            )
            / n_labels
        )

    run_ctx.mark_as_loss(loss=_ctc(log_probs), name="ctc", dims=[])  # final head, scale 1.0
    for li, scale in zip(model.aux_layer_idx, model.aux_scales):  # intermediate-layer aux InterCTC heads
        if (li + 1) in aux_log_probs:  # defensive: aux present in train+eval; recog ignores
            run_ctx.mark_as_loss(loss=_ctc(aux_log_probs[li + 1]), name=f"ctc_aux{li + 1}", scale=scale, dims=[])

    # CIF RATE ANCHOR (trainable-segmenter FT-1 ONLY -- model.quantity_loss_scale set only for trainseg).
    # A trainable CIF under CTC has NO force pinning its firing rate, and zero_infinity=True turns every
    # K<N (fewer tokens than labels) utterance into 0 loss AND 0 grad -> under-firing is a zero-loss
    # GLOBAL minimum, so the segmenter collapses tok_per_lab->0 / infeasible->1 / CTC->0.000 (diagnosed
    # 2026-06-20; the frozen-seg control proves the segmenter is the cause -- see [[ft-collapse-diagnosis]]).
    # The quantity loss (== the one used in pretraining) anchors sum(alpha) to the pretrained ACOUSTIC
    # rate Q* = T*(target_rate/frame_rate), NOT to N -> the raw ~2-3x-N margin (and thus K>=N) is preserved
    # and inference is unchanged (raw cif_pool). q* depends only on T, so it is the same anchor as pretrain.
    if model.quantity_loss_scale is not None:
        frame_len = diag["frame_valid"].sum(dim=1)  # [B] valid 25 Hz frame counts (== cif_pool's lengths)
        qty = quantity_loss(
            diag["sum_alpha"],
            frame_len,
            target_rate_hz=model.cfg.target_rate_hz,
            frame_rate_hz=model.cfg.frame_rate_hz,
        )
        run_ctx.mark_as_loss(loss=qty, name="qty", dims=[], scale=model.quantity_loss_scale)

    # --- small diagnostics (scale=0): K>=N headroom, the rate dial, and CTC-infeasible fraction ---
    with torch.no_grad():
        tok = input_lengths.float().sum().clamp(min=1)
        tok_per_lab = tok / target_lengths.float().sum().clamp(min=1)
        fps_tok = diag["frame_valid"].float().sum() / tok
        infeasible = (input_lengths < target_lengths).float().mean()
    run_ctx.mark_as_loss(loss=tok_per_lab, name="tok_per_lab", dims=[], as_error=True, scale=0.0)
    run_ctx.mark_as_loss(loss=fps_tok, name="fps_tok", dims=[], as_error=True, scale=0.0)
    run_ctx.mark_as_loss(loss=infeasible, name="infeasible", dims=[], as_error=True, scale=0.0)


# --------------------------- greedy decoding (real-RETURNN forward) ---------------------------


def forward_step(*, model: Model, extern_data, **kwargs):
    """Run the AM and mark per-token log-probs with a dynamic time dim (padding trimmed per seq)."""
    from returnn.tensor import Dim

    audio = extern_data["audio"]
    batch_dim = audio.dims[0]
    raw_audio = audio.raw_tensor
    raw_audio_len = audio.dims[1].dyn_size_ext.raw_tensor

    log_probs, out_lengths, _, _ = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len.to(raw_audio.device))

    time_dim = Dim(None, name="ctc_out_time")
    time_dim.dyn_size_ext = rf.convert_to_tensor(out_lengths.to("cpu", torch.int32), dims=[batch_dim], dtype="int32")
    feat_dim = Dim(int(log_probs.shape[-1]), name="ctc_labels")
    rf.convert_to_tensor(log_probs, dims=[batch_dim, time_dim, feat_dim], name="log_probs").mark_as_output(
        "log_probs", shape=[batch_dim, time_dim, feat_dim]
    )


class ForwardCallback(ForwardCallbackIface):
    """Greedy CTC: position-argmax + collapse-repeats + drop-blank + SPM-decode -> WORD stream (search_out.py)."""

    def __init__(self, spm_model_file: str, blank_idx: int):
        self._spm_model_file = spm_model_file
        self._blank_idx = int(blank_idx)

    def init(self, *, model):
        import sentencepiece as spm

        self.sp = spm.SentencePieceProcessor(model_file=str(self._spm_model_file))
        self.recognition_file = open("search_out.py", "wt")
        self.recognition_file.write("{\n")

    def process_seq(self, *, seq_tag: str, outputs):
        import numpy as np

        lp = outputs["log_probs"].raw_tensor  # [K, V+1] numpy, padding removed
        frames = np.asarray(lp).argmax(axis=-1)
        if frames.shape[0] > 0:
            keep = np.ones(frames.shape[0], dtype=bool)
            keep[1:] = frames[1:] != frames[:-1]
            collapsed = frames[keep]
        else:
            collapsed = frames
        ids = [int(i) for i in collapsed if int(i) != self._blank_idx]
        hyp = self.sp.decode(ids)  # piece-merge + '▁' -> space -> word string
        self.recognition_file.write("%s: %s,\n" % (repr(seq_tag), repr(hyp)))

    def finish(self):
        self.recognition_file.write("}\n")
        self.recognition_file.close()
