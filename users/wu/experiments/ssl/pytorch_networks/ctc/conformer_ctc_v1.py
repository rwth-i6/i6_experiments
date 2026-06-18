"""
CTC model for LibriSpeech finetuning (BPE labels, torch ``nn.functional.ctc_loss``).

Shares the exact 12x512 rel-pos conformer + log-mel + global input-norm with the BEST-RQ SSL model,
so a pretrained encoder transfers via ``preload_from_files``. A single CTC head on the final encoder
layer: ``Linear(dim, vocab+1)`` with **blank = last index = vocab** (BPE labels keep 0..vocab-1).
SpecAugment on (train only) from a configurable epoch.

Greedy decoding (``forward``/``GreedyCTCCallback``): per-frame argmax -> collapse repeats -> drop
blank -> de-BPE to words, writing the standard RETURNN ``search_out.py`` (-> sclite WER downstream).
"""

from __future__ import annotations

import torch
from torch import nn

import returnn.frontend as rf
from returnn.forward_iface import ForwardCallbackIface

from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1
from i6_models.primitives.specaugment import specaugment_v1_by_length

from .conformer_ctc_v1_cfg import CTCConfig
from ..best_rq.parts.input_norm import apply_global_norm
from ..common.conformer import build_conformer_encoder, sequence_mask


class Model(nn.Module):
    def __init__(self, model_config_dict, vocab_size, **kwargs):
        super().__init__()
        self.cfg = CTCConfig.from_dict(model_config_dict)
        conformer_size = self.cfg.encoder_config.conformer_size
        self.vocab_size = int(vocab_size)  # BPE labels 0..vocab_size-1
        self.blank_idx = self.vocab_size  # blank = last index

        self.feature_extraction = LogMelFeatureExtractionV1(cfg=self.cfg.feature_extraction_config)
        self.encoder = build_conformer_encoder(self.cfg.encoder_config)
        self.out_linear = nn.Linear(conformer_size, self.vocab_size + 1)  # final-layer CTC head (scale 1.0)

        # auxiliary CTC heads on intermediate encoder layers (config is 1-indexed -> store 0-indexed)
        aux_layers = self.cfg.aux_ctc_layers or []
        aux_scales = self.cfg.aux_ctc_scales or []
        assert len(aux_layers) == len(aux_scales), "aux_ctc_layers and aux_ctc_scales must have equal length"
        self.aux_layer_idx = [int(l) - 1 for l in aux_layers]
        self.aux_scales = [float(s) for s in aux_scales]
        self.aux_linears = nn.ModuleList([nn.Linear(conformer_size, self.vocab_size + 1) for _ in aux_layers])

        self.register_buffer("global_mean", torch.tensor(self.cfg.global_mean, dtype=torch.float32))
        self.register_buffer("global_std", torch.tensor(self.cfg.global_std, dtype=torch.float32))
        self.specaug_start_step = self.cfg.specaug_start_step
        self.num_layers = self.cfg.encoder_config.num_layers

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        if raw_audio.dim() == 3:
            raw_audio = raw_audio.squeeze(-1)
        with torch.no_grad():
            features, feat_len = self.feature_extraction(raw_audio, raw_audio_len)  # [B, Tf, F]
            features = apply_global_norm(features, feat_len, self.global_mean, self.global_std)

        if self.training and rf.get_run_ctx().step >= self.specaug_start_step:
            features = specaugment_v1_by_length(
                features,
                time_min_num_masks=2,
                time_max_mask_per_n_frames=self.cfg.specaug_config.repeat_per_n_frames,
                time_mask_max_size=self.cfg.specaug_config.max_dim_time,
                freq_min_num_masks=2,
                freq_mask_max_size=self.cfg.specaug_config.max_dim_feat,
                freq_max_num_masks=self.cfg.specaug_config.num_repeat_feat,
            )

        mask = sequence_mask(features, feat_len)  # [B, Tf] True=valid
        final_idx = self.num_layers - 1
        want = sorted(set(self.aux_layer_idx + [final_idx]))
        enc_layers, out_mask = self.encoder(features, mask, return_layers=want)
        by_idx = {li: enc_layers[i] for i, li in enumerate(want)}
        log_probs = torch.log_softmax(self.out_linear(by_idx[final_idx]), dim=-1)  # [B, T, vocab+1]
        out_lengths = out_mask.sum(dim=1)  # [B]
        # aux CTC heads, computed in BOTH train AND eval. RETURNN runs train_step on the eval set too,
        # with the model in eval mode -> gating aux on self.training makes aux_log_probs empty there and
        # train_step then crashes (KeyError). Recog (forward_step) ignores aux. key = 1-indexed layer.
        aux_log_probs = {}
        for head, li in zip(self.aux_linears, self.aux_layer_idx):
            aux_log_probs[li + 1] = torch.log_softmax(head(by_idx[li]), dim=-1)
        return log_probs, out_lengths, aux_log_probs


def train_step(*, model: Model, extern_data, **kwargs):
    run_ctx = rf.get_run_ctx()

    audio = extern_data["audio"]
    raw_audio = audio.raw_tensor
    raw_audio_len = audio.dims[1].dyn_size_ext.raw_tensor.to(raw_audio.device)

    text = extern_data["text"]
    targets = text.raw_tensor.long()  # [B, S]
    target_lengths = text.dims[1].dyn_size_ext.raw_tensor.to(torch.int32)  # [B]

    log_probs, input_lengths, aux_log_probs = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)

    # torch CTC: log_probs (T, B, C) float32; lengths on CPU int32 (cuDNN requirement).
    il_cpu = input_lengths.to(torch.int32).cpu()
    tl_cpu = target_lengths.cpu()
    n_labels = target_lengths.sum().clamp(min=1)  # label-normalize (i6 canonical: reduction=sum / #labels)

    def _ctc(lp):
        loss = torch.nn.functional.ctc_loss(
            lp.transpose(0, 1).float(), targets, il_cpu, tl_cpu,
            blank=model.blank_idx, reduction="sum", zero_infinity=True,
        )
        return loss / n_labels

    run_ctx.mark_as_loss(loss=_ctc(log_probs), name="ctc", dims=[])  # final head, scale 1.0
    for li, scale in zip(model.aux_layer_idx, model.aux_scales):  # intermediate-layer aux CTC heads
        if (li + 1) in aux_log_probs:  # defensive: aux always present (train+eval), recog ignores
            run_ctx.mark_as_loss(loss=_ctc(aux_log_probs[li + 1]), name=f"ctc_aux{li + 1}", scale=scale, dims=[])


# --------------------------- greedy decoding (real-RETURNN forward) ---------------------------


def forward_step(*, model: Model, extern_data, **kwargs):
    """Run the AM and mark per-frame log-probs with a dynamic time dim (padding trimmed per seq)."""
    from returnn.tensor import Dim

    audio = extern_data["audio"]
    batch_dim = audio.dims[0]
    raw_audio = audio.raw_tensor
    raw_audio_len = audio.dims[1].dyn_size_ext.raw_tensor

    log_probs, out_lengths, _ = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len.to(raw_audio.device))

    time_dim = Dim(None, name="ctc_out_time")
    time_dim.dyn_size_ext = rf.convert_to_tensor(out_lengths.to("cpu", torch.int32), dims=[batch_dim], dtype="int32")
    feat_dim = Dim(int(log_probs.shape[-1]), name="ctc_labels")
    rf.convert_to_tensor(log_probs, dims=[batch_dim, time_dim, feat_dim], name="log_probs").mark_as_output(
        "log_probs", shape=[batch_dim, time_dim, feat_dim]
    )


class ForwardCallback(ForwardCallbackIface):
    """Greedy CTC: frame-argmax + collapse-repeats + drop-blank + SPM-decode -> WORD stream in ``search_out.py``."""

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

        lp = outputs["log_probs"].raw_tensor  # [T, V+1] numpy, padding removed
        frames = np.asarray(lp).argmax(axis=-1)
        if frames.shape[0] > 0:
            keep = np.ones(frames.shape[0], dtype=bool)
            keep[1:] = frames[1:] != frames[:-1]
            collapsed = frames[keep]
        else:
            collapsed = frames
        ids = [int(i) for i in collapsed if int(i) != self._blank_idx]
        # SPM decode: handles piece-merge + '▁' -> space, yields the word string.
        hyp = self.sp.decode(ids)
        self.recognition_file.write("%s: %s,\n" % (repr(seq_tag), repr(hyp)))

    def finish(self):
        self.recognition_file.write("}\n")
        self.recognition_file.close()
