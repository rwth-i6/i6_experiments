"""
TTS-encoder project.

Goal: feed the frozen GlowTTS log-mel output directly into a standard CTC/AED ASR encoder
(skipping the GlowTTS gl-net + Griffin-Lim + ASR re-extraction detour),
and make text-utilization cheap by shrinking the synthetic durations.

Step 1 (this file): front-end ablation, no text-only data yet.
The frozen GlowTTS emits log-mel in its own DbMel feature space
(Slaney mel, f_max 7600, 12.5ms hop, fixed norm), which is incompatible bin-for-bin with
the default ``rf.audio.log_mel_filterbank_from_raw`` (HTK mel, f_max 8000, 10ms hop).
To allow zero-transform injection of the GlowTTS output later, we train the ASR baseline
on the same DbMel front-end. This measures the WER cost of that front-end swap.

Uses the standard CTC+AED baseline ``exp2024_04_23_baselines.aed`` (EncL16-DecL6-D1024-spm10k),
configured via its ``feature_extraction`` opt -- no custom Model subclass.

  - "base-ls":       default log-mel front-end (reference baseline).
  - "base-ls-dbmel": same model, DbMel front-end (= the GlowTTS feature space).

Run: py7 ./sis m recipe/i6_experiments/users/zeyer/experiments/exp2026_05_28_tts_encoder.py

TODO (next steps): return_log_mel mode on the frozen TtsModel; text-only batch via alternate_batching;
shrink / randomize GlowTTS durations (length_scale) for cheap text-util.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Tuple

from returnn.tensor import Tensor, Dim, batch_dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import (
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
)

from returnn.util.basic import BehaviorVersion

from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import (
    train_exp as aed_train_exp,
    _raw_sample_rate,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines import configs
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
    aed_ctc_timesync_recog_recomb_auto_scale,
)
from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config

__all__ = ["py", "DbMelFeatureExtractor"]

# Prefix for alias/ and output/ paths. Doesn't enter any Job hash.
__setup_root_prefix__ = "exp2026_05_28_tts_encoder"


def py():
    # (a) standard log-mel front-end -- reference for the ablation.
    #     bhv 24 keeps the hash matching the established `base-librispeech` baseline (imported, not re-run).
    _train_ls_base("base-ls")
    # (b) DbMel front-end (== frozen GlowTTS feature space) -- the new training.
    #     Latest bhv (25). 24->25 only changes rf.scatter in padded areas (issue #1815), which this model
    #     does not use, so it is behavior-neutral vs the bhv-24 base-ls -- the front-end ablation stays clean.
    _train_ls_base(
        "base-ls-dbmel",
        feature_extraction=rf.build_dict(DbMelFeatureExtractor),
        behavior_version=BehaviorVersion._latest_behavior_version,
    )


def _train_ls_base(
    name: str,
    *,
    feature_extraction: Optional[Dict[str, Any]] = None,
    behavior_version: int = 24,
    prefix: Optional[str] = None,
):
    """Standard LibriSpeech CTC+AED (EncL16-DecL6-D1024-spm10k), mirroring
    exp2026_05_26_base_fzj._train_librispeech_base, with an optional custom feature front-end."""
    if prefix is None:
        prefix = get_setup_prefix_for_module(__name__)
    ls_task = get_librispeech_task_raw_v2(vocab="spm10k")

    model_config: Dict[str, Any] = {
        "behavior_version": behavior_version,
        "__serialization_version": 2,
        "enc_build_dict": rf.build_dict(
            ConformerEncoder,
            input_layer=rf.build_dict(
                ConformerConvSubsample,
                out_dims=[32, 64, 64],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
            ),
            num_layers=16,
            out_dim=1024,
            encoder_layer=rf.build_dict(
                ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=8,
            ),
        ),
        "dec_build_dict": rf.build_dict(
            TransformerDecoder,
            num_layers=6,
            model_dim=1024,
            norm=rf.build_dict(rf.RMSNorm),
            ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
            layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
        ),
        "feature_batch_norm": True,
    }
    if feature_extraction is not None:
        model_config["feature_extraction"] = feature_extraction

    exp = aed_train_exp(
        name,
        configs.config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        model_config=model_config,
        config_updates={
            **configs._get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": 100_000 * configs._batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "dec_aux_loss_layers": [3],
            "max_seq_length_default_target": None,
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=ls_task,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=16,
    )
    return exp


class DbMelFeatureExtractor(rf.Module):
    """Feature front-end producing the GlowTTS DbMel log-mel (Slaney mel, f_max 7600, 12.5ms hop,
    fixed norm), so the ASR input space == frozen GlowTTS output space.

    Matches the config in
    ``denoising_lm_2024.sis_recipe.tts_model.get_tts_model_config()``
    ``["glow_tts_model_config"]["feature_extraction_config"]``.
    Compatible with ``exp2024_04_23_baselines.aed.Model.feature_extraction``:
    exposes ``out_dim`` and is callable ``(source, in_spatial_dim) -> (features, out_spatial_dim)``.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        win_size: float = 0.05,
        hop_size: float = 0.0125,
        f_min: int = 0,
        f_max: int = 7600,
        min_amp: float = 1e-10,
        num_filters: int = 80,
        center: bool = True,
        norm: Tuple[float, float] = (-72.83881497383118, 37.73079669103133),
    ):
        super().__init__()
        from i6_experiments.users.zeyer.experiments.nick_ctc_rnnt_standalone_2024.pytorch_networks.tts_shared.db_mel_features import (
            DbMelFeatureExtraction,
            DbMelFeatureExtractionConfig,
        )

        self.out_dim = Dim(num_filters, name="dbmel")
        self.dbmel = DbMelFeatureExtraction(
            DbMelFeatureExtractionConfig(
                sample_rate=sample_rate,
                win_size=win_size,
                hop_size=hop_size,
                f_min=f_min,
                f_max=f_max,
                min_amp=min_amp,
                num_filters=num_filters,
                center=center,
                norm=norm,
            )
        )

    def __call__(self, source: Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
        import torch

        orig_dtype = source.dtype
        if source.feature_dim and source.feature_dim.dimension == 1:
            source = rf.squeeze(source, axis=source.feature_dim)
        batch_dims = source.remaining_dims(in_spatial_dim)
        assert batch_dims == [batch_dim], f"only single batch dim supported, got {batch_dims}"
        # DbMel uses torch.stft, which does not support bf16; force float32 and cast back.
        audio = source.copy_compatible_to_dims_raw([batch_dim, in_spatial_dim]).float()  # [B, T]
        lengths = in_spatial_dim.get_size_tensor(device=source.device).copy_compatible_to_dims_raw([batch_dim])  # [B]
        with torch.autocast(device_type=audio.device.type, enabled=False):
            feats, feat_lens = self.dbmel(audio, lengths)  # [B, T', F], [B]
        feat_lens_rf = rf.convert_to_tensor(feat_lens.cpu(), dims=[batch_dim])
        out_spatial_dim = Dim(feat_lens_rf, name="dbmel_time")
        feats_rf = rf.convert_to_tensor(feats, dims=[batch_dim, out_spatial_dim, self.out_dim])
        feats_rf.feature_dim = self.out_dim
        feats_rf = rf.cast(feats_rf, dtype=orig_dtype)
        return feats_rf, out_spatial_dim
