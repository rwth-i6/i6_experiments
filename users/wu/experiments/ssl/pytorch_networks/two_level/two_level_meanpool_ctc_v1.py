"""
Two-level BEST-RQ + fixed MEAN-POOL with a CTC FINETUNE head (FT-1) -- the CONTROL finetune for the
mean-pool pretrain (``two_level_meanpool_v1``).

Identical to ``two_level_ctc_v1`` EXCEPT the CIF segmenter is replaced by the fixed mean-pool of every
``factor`` frames (``factor = round(frame_rate / target_rate)``). There is therefore only ONE regime
(no ``freeze_segmenter``: there is nothing learned to freeze). Same frozen lower stack, same 9-layer high
Conformer, same single CTC head + InterCTC aux heads, same SpecAugment, same greedy decoder. SPM-5120
labels, torch ``ctc_loss``, blank = last index = ``vocab_size``.

fp32 / autocast OFF around the frozen lower forward + pooling, matching the CIF FT (see [[cif-bf16-fp32]]).
"""

from __future__ import annotations

import torch
from torch import nn

import returnn.frontend as rf
from returnn.forward_iface import ForwardCallbackIface

from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1
from i6_models.primitives.specaugment import specaugment_v1_by_length

from .two_level_v1_cfg import TwoLevelConfig
from .parts.cif import mean_pool
from ..best_rq.parts.input_norm import apply_global_norm
from ..common.conformer import build_conformer_encoder, build_high_conformer_encoder, sequence_mask


class Model(nn.Module):
    def __init__(self, model_config_dict, vocab_size, specaug_start_step=None, **kwargs):
        """
        :param model_config_dict: asdict(TwoLevelConfig) -- SAME architecture config as the mean-pool
            pretrain, so its lower/high keys preload 1:1 from the pretrained ckpt.
        :param vocab_size: SPM vocab size; CTC labels are ``0..vocab_size-1`` and blank == ``vocab_size``.
        :param specaug_start_step: if set, SpecAugment on the log-mel input once the global step reaches it
            (train-only). Matches the base CTC finetune strength. ``**kwargs`` swallows any inert net-args
            (e.g. freeze_segmenter, which is meaningless without a learned segmenter).
        """
        super().__init__()
        self.cfg = cfg = TwoLevelConfig.from_dict(model_config_dict)
        conf = cfg.encoder_config.conformer_size
        self.vocab_size = int(vocab_size)
        self.blank_idx = self.vocab_size
        self.specaug_start_step = None if specaug_start_step is None else int(specaug_start_step)

        # ---- frozen lower stack (names match the pretrained ckpt keys) ----
        self.feature_extraction = LogMelFeatureExtractionV1(cfg=cfg.feature_extraction_config)
        self.encoder = build_conformer_encoder(cfg.encoder_config)  # 9 layers
        for p in self.feature_extraction.parameters():
            p.requires_grad = False
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.register_buffer("global_mean", torch.tensor(cfg.global_mean, dtype=torch.float32))
        self.register_buffer("global_std", torch.tensor(cfg.global_std, dtype=torch.float32))

        # ---- high encoder + CTC heads (NO segmenter params: pooling is fixed) ----
        self.high_encoder = build_high_conformer_encoder(cfg.high_encoder_config)
        self.out_linear = nn.Linear(conf, self.vocab_size + 1)  # final-layer CTC head (scale 1.0), blank = last
        self.aux_layer_idx = [3, 7]  # InterCTC at high-encoder layers 4 & 8 (0-indexed), == the CIF FT
        self.aux_scales = [0.3, 0.3]
        assert max(self.aux_layer_idx) < cfg.high_encoder_config.num_layers - 1, "aux layers must be below the final"
        self.aux_linears = nn.ModuleList([nn.Linear(conf, self.vocab_size + 1) for _ in self.aux_layer_idx])

        derived = (
            cfg.feature_extraction_config.sample_rate
            / (cfg.feature_extraction_config.hop_size * cfg.feature_extraction_config.sample_rate)
            / 4.0
        )
        assert abs(derived - cfg.frame_rate_hz) < 1e-6, f"frame_rate {cfg.frame_rate_hz} != derived {derived}"
        factor = round(cfg.frame_rate_hz / cfg.target_rate_hz)
        assert factor >= 1 and abs(factor - cfg.frame_rate_hz / cfg.target_rate_hz) < 1e-6
        self.pool_factor = factor
        self.num_high_layers = cfg.high_encoder_config.num_layers

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        """:return: (log_probs [B, K, vocab+1], token_lengths [B], aux_log_probs, diag)."""
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

            z, z_mask, diag = mean_pool(h, frame_len, factor=self.pool_factor)  # z [B, K, 512] fp32

        # ---- high encoder (no frontend) over the pooled tokens -> CTC heads ----
        final_idx = self.num_high_layers - 1
        want = sorted(set(self.aux_layer_idx + [final_idx]))
        hi_layers, _ = self.high_encoder(z, z_mask, return_layers=want)
        by_idx = {li: hi_layers[i] for i, li in enumerate(want)}
        log_probs = torch.log_softmax(self.out_linear(by_idx[final_idx]), dim=-1)  # [B, K, vocab+1]
        token_lengths = z_mask.sum(dim=1)
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

    run_ctx.mark_as_loss(loss=_ctc(log_probs), name="ctc", dims=[])
    for li, scale in zip(model.aux_layer_idx, model.aux_scales):
        if (li + 1) in aux_log_probs:
            run_ctx.mark_as_loss(loss=_ctc(aux_log_probs[li + 1]), name=f"ctc_aux{li + 1}", scale=scale, dims=[])

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
        hyp = self.sp.decode(ids)
        self.recognition_file.write("%s: %s,\n" % (repr(seq_tag), repr(hyp)))

    def finish(self):
        self.recognition_file.write("}\n")
        self.recognition_file.close()
