"""
AED model definition: Conformer encoder + Transformer decoder, with aux CTC heads.

Ported from :mod:`i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed`
(``Model`` and ``aed_model_def``), behaviour-identical at the time of the port (2026-08-27).
Same pattern as ``speech_llm/prefix_lm/model/albert/model/main.py``: an own copy so this setup can
diverge (LLM vocab), while shared constants and helpers are imported from the original so they
cannot drift.

Packed-tensor note: nothing here is packing-specific. Packing is a RETURNN engine/config concern
(``packed_tensors`` / ``packed_batch_size`` / ``torch_cuda_graph``); the model just has to be built
from RF ops that have packed implementations, which the Conformer + Transformer stack is.
The one thing that DOES matter at the model level is ``behavior_version``: >= 29 masks the
conv-block BatchNorm statistics, which otherwise run over the raw packed storage and count the
packing gap frames. The recipe sets that.
"""

from __future__ import annotations

import copy
import functools
import inspect
from typing import Optional, Any, Sequence, Tuple, Dict
import numpy

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.zeyer.model_interfaces import ModelDef
from i6_experiments.users.zeyer.returnn.models.rf_layerdrop import SequentialLayerDrop
from i6_experiments.users.zeyer.nn_rf.pad_ext import pad_ext

# Shared with the original, deliberately imported rather than copied:
# these are constants/helpers, and a second copy could silently drift from the reference.
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import (
    _log_mel_feature_dim,
    _batch_size_factor,
    _aed_model_def_blank_idx,
    _aed_model_def_blank_label,
    _get_bos_idx,
    _get_eos_idx,
    EncoderOutput,
    log_probs_with_eos_separated,
)

__all__ = [
    "Model",
    "model_def",
    "EncoderOutput",
    "log_probs_with_eos_separated",
]


def model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    # real input is raw audio, internally it does logmel
    in_dim, epoch  # noqa
    config = get_global_config()  # noqa

    enc_conformer_layer = config.typed_value("enc_conformer_layer", None)
    if enc_conformer_layer:
        assert isinstance(enc_conformer_layer, dict) and "class" in enc_conformer_layer
    else:
        enc_conformer_layer = rf.build_dict(
            rf.encoder.conformer.ConformerEncoderLayer,
            conv_norm=rf.build_dict(rf.BatchNorm, use_mask=True),
            self_att=rf.build_dict(
                rf.RelPosSelfAttention,
                # Shawn et al 2018 style, old RETURNN (old TF) way.
                with_bias=False,
                with_linear_pos=False,
                with_pos_bias=False,
                learnable_pos_emb=True,
                separate_pos_emb_per_head=False,
                pos_emb_dropout=config.float("pos_emb_dropout", 0.0),
            ),
            ff_activation=rf.build_dict(rf.relu_square),
            num_heads=8,
        )

    feature_extraction = config.typed_value("feature_extraction", None)

    blank_idx = _aed_model_def_blank_idx
    if blank_idx < 0:
        blank_idx = target_dim.dimension + 1 + blank_idx
    return Model(
        feature_extraction=feature_extraction,
        enc_build_dict=config.typed_value("enc_build_dict", None),  # alternative more generic/flexible way
        num_enc_layers=config.int("num_enc_layers", 12),
        enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
        enc_ff_dim=Dim(name="enc-ff", dimension=2048, kind=Dim.Types.Feature),
        enc_att_num_heads=8,
        enc_conformer_layer=enc_conformer_layer,
        target_dim=target_dim,
        blank_idx=blank_idx,
        bos_idx=_get_bos_idx(target_dim),
        eos_idx=_get_eos_idx(target_dim),
        enc_aux_logits=config.typed_value("aux_loss_layers") or (),
        enc_aux_logits_with_bias=config.bool("enc_aux_logits_with_bias", True),
        enc_aux_logits_share_weights=config.bool("enc_aux_logits_share_weights", False),
        dec_aux_logits=config.typed_value("dec_aux_loss_layers") or (),
        dec_aux_logits_share_weights=config.bool("dec_aux_logits_share_weights", False),
        dec_build_dict=config.typed_value("dec_build_dict", None),  # alternative more generic/flexible way
    )


model_def: ModelDef[Model]
# Default only; the recipe overrides this via model_config["behavior_version"].
# The packed runs need >= 29 (masked conv BatchNorm statistics), see the module docstring.
model_def.behavior_version = 21
model_def.backend = "torch"
model_def.batch_size_factor = _batch_size_factor


class Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        *,
        feature_extraction: Optional[Dict[str, Any]] = None,
        num_enc_layers: int = 12,
        num_dec_layers: int = 6,
        target_dim: Dim,
        wb_target_dim: Optional[Dim] = None,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        enc_build_dict: Optional[Dict[str, Any]] = None,
        enc_aux_logits: Sequence[int] = (),  # layers
        enc_aux_logits_with_bias: bool = True,  # if True, enc_aux_logits have bias
        enc_aux_logits_share_weights: bool = False,  # if True, all enc_aux_logits share weights
        dec_aux_logits: Sequence[int] = (),  # layers
        dec_aux_logits_share_weights: bool = False,  # if True, all dec_aux_logits + final logits share weights
        enc_model_dim: Dim = Dim(name="enc", dimension=512),
        dec_model_dim: Dim = Dim(name="dec", dimension=512),
        enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
        enc_att_num_heads: int = 4,
        enc_conformer_layer: Optional[Dict[str, Any]] = None,
        enc_dropout: float = 0.1,
        enc_att_dropout: float = 0.1,
        dec_build_dict: Optional[Dict[str, Any]] = None,
    ):
        super(Model, self).__init__()

        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        enc_layer_drop = config.float("enc_layer_drop", 0.0)
        if enc_layer_drop:
            enc_sequential = functools.partial(SequentialLayerDrop, layer_drop=enc_layer_drop)
        else:
            enc_sequential = rf.Sequential
        dec_layer_drop = config.float("dec_layer_drop", 0.0)
        if dec_layer_drop:
            dec_sequential = functools.partial(SequentialLayerDrop, layer_drop=dec_layer_drop)
        else:
            dec_sequential = rf.Sequential

        if not feature_extraction:
            feat_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
            feature_extraction = rf.build_dict(
                rf.Functional,
                func=functools.partial(rf.audio.log_mel_filterbank_from_raw, sampling_rate=16_000, out_dim=feat_dim),
                attribs={"out_dim": feat_dim},
            )
        self.feature_extraction = rf.build_from_dict(feature_extraction)
        in_dim = self.feature_extraction.out_dim

        self.in_dim = in_dim
        if enc_build_dict:
            assert enc_sequential is rf.Sequential
            # Warning: We ignore the other args (num_enc_layers, enc_model_dim, enc_other_opts, etc).
            self.encoder = rf.build_from_dict(enc_build_dict, in_dim)
            self.encoder: ConformerEncoder  # might not be true, but assume similar/same interface

        else:
            self.encoder = ConformerEncoder(
                in_dim,
                enc_model_dim,
                ff_dim=enc_ff_dim,
                input_layer=ConformerConvSubsample(
                    in_dim,
                    out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],
                ),
                encoder_layer=enc_conformer_layer,
                num_layers=num_enc_layers,
                num_heads=enc_att_num_heads,
                dropout=enc_dropout,
                att_dropout=enc_att_dropout,
                sequential=enc_sequential,
            )

        if dec_build_dict:
            assert dec_sequential is rf.Sequential
            # Warning: We ignore the other args (num_dec_layers, dec_model_dim, dec_other_opts, etc).
            self.decoder = rf.build_from_dict(dec_build_dict, self.encoder.out_dim, target_dim)
            self.decoder: TransformerDecoder  # might not be true, but assume similar/same interface

        else:
            self.decoder = TransformerDecoder(
                num_layers=num_dec_layers,
                encoder_dim=self.encoder.out_dim,
                vocab_dim=target_dim,
                model_dim=dec_model_dim,
                sequential=dec_sequential,
            )

        disable_encoder_self_attention = config.typed_value("disable_encoder_self_attention", None)
        if disable_encoder_self_attention is not None:
            # Disable self-attention in encoder.
            from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.model_ext.disable_self_att import (
                apply_disable_self_attention_,
            )

            apply_disable_self_attention_(self.encoder, disable_encoder_self_attention)

        self.target_dim = target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx
        self.out_eos_separated = config.bool("out_eos_separated", False)

        if enc_aux_logits:
            if not wb_target_dim:
                wb_target_dim = target_dim + 1
            if target_dim.vocab and not wb_target_dim.vocab:
                from returnn.datasets.util.vocabulary import Vocabulary

                # Just assumption for code now, might extend this later.
                assert wb_target_dim.dimension == target_dim.dimension + 1 and blank_idx == target_dim.dimension
                vocab_labels = list(target_dim.vocab.labels) + [_aed_model_def_blank_label]
                wb_target_dim.vocab = Vocabulary.create_vocab_from_labels(
                    vocab_labels, user_defined_symbols={_aed_model_def_blank_label: blank_idx}
                )
        for i, layer_idx in enumerate(enc_aux_logits):
            setattr(
                self,
                f"enc_aux_logits_{layer_idx}",
                rf.Linear(self.encoder.out_dim, wb_target_dim, with_bias=enc_aux_logits_with_bias)
                if i == 0 or not enc_aux_logits_share_weights
                else getattr(self, f"enc_aux_logits_{enc_aux_logits[0]}"),
            )
        self.enc_aux_logits = enc_aux_logits
        self.wb_target_dim = wb_target_dim

        for layer_idx in dec_aux_logits:
            setattr(
                self,
                f"dec_aux_final_layer_norm_{layer_idx}",
                copy.deepcopy(self.decoder.final_layer_norm)
                if not dec_aux_logits_share_weights
                else self.decoder.final_layer_norm,
            )
            setattr(
                self,
                f"dec_aux_logits_{layer_idx}",
                copy.deepcopy(self.decoder.logits) if not dec_aux_logits_share_weights else self.decoder.logits,
            )
        self.dec_aux_logits = dec_aux_logits

        self.pad_audio = config.typed_value("pad_audio", None)

        self.feature_batch_norm = None
        feature_norm_module = config.typed_value("feature_norm_module", None)
        if feature_norm_module is not None:
            # Configurable front-end normalization (e.g. GroupNorm) in place of the feature BatchNorm.
            # Kept on the same attribute name for checkpoint compatibility with the BatchNorm variant.
            # NB: distinct from the existing boolean ``feature_norm`` option further below.
            self.feature_batch_norm = rf.build_from_dict(feature_norm_module, self.in_dim)
        elif config.bool("feature_batch_norm", False):
            self.feature_batch_norm = rf.BatchNorm(self.in_dim, affine=False, use_mask=True)
        # Some feature norms (e.g. GroupNormSpatial) need the spatial dim to pool the statistics over time;
        # detect it once here and pass it through where the feature norm is applied.
        self.feature_norm_wants_spatial_dim = self.feature_batch_norm is not None and (
            "spatial_dim" in inspect.signature(self.feature_batch_norm).parameters
        )
        self.feature_norm = config.bool("feature_norm", False)
        self.feature_stats = None
        feature_stats = config.typed_value("feature_stats")
        if feature_stats:
            assert isinstance(feature_stats, dict)
            self.feature_stats = rf.ParameterList(
                {
                    k: rf.Parameter(
                        rf.convert_to_tensor(numpy.loadtxt(v), dims=[self.in_dim], dtype=rf.get_default_float_dtype()),
                        auxiliary=True,
                    )
                    for k, v in feature_stats.items()
                }
            )

        self._specaugment_opts = {
            "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
            "max_consecutive_spatial_dims": config.typed_value("specaugment_max_consecutive_spatial_dims") or 20,
            "max_consecutive_feature_dims": config.typed_value("specaugment_max_consecutive_feature_dims")
            or (in_dim.dimension // 5),
            "num_spatial_mask_factor": config.typed_value("specaugment_num_spatial_mask_factor") or 100,
        }

        self._mixup = None
        if config.typed_value("mixup", None) is not None:
            from i6_experiments.users.zeyer.returnn.models.rf_mixup import Mixup, MixupOpts

            self._mixup = Mixup(feature_dim=self.in_dim, opts=MixupOpts(**config.typed_value("mixup")))

        self.ctc_am_scale = config.float("ctc_am_scale", 1.0)
        self.ctc_framewise_prior_scale = config.float("ctc_prior_scale", 0.0)
        self.ctc_framewise_prior_type = config.value("ctc_prior_type", "batch")
        # framewise prior for CTC
        ctc_framewise_static_prior = config.typed_value("static_prior")
        self.ctc_framewise_static_prior = None  # in log prob, if set
        if ctc_framewise_static_prior:
            assert isinstance(ctc_framewise_static_prior, dict)
            assert set(ctc_framewise_static_prior.keys()) == {"file", "type"}
            v = numpy.loadtxt(ctc_framewise_static_prior["file"])
            # The `type` is about what is stored in the file.
            # We always store it in log prob here, so we potentially need to convert it.
            if ctc_framewise_static_prior["type"] == "log_prob":
                pass  # already log prob
            elif ctc_framewise_static_prior["type"] == "prob":
                v = numpy.log(v)
            else:
                raise ValueError(f"invalid static_prior type {ctc_framewise_static_prior['type']!r}")
            self.ctc_framewise_static_prior = rf.Parameter(
                rf.convert_to_tensor(v, dims=[self.wb_target_dim], dtype=rf.get_default_float_dtype()),
                auxiliary=True,
                non_critical_for_restore=True,
            )

        from i6_experiments.users.zeyer.nn_rf.variational_noise import maybe_apply_variational_noise_from_config

        maybe_apply_variational_noise_from_config(self, config)

    def encode_no_transform(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
        specaugment_max_spatial_dims: Optional[Tensor] = None,
        end_layer: Optional[int] = None,
    ) -> Tuple[Tensor, Dim]:
        """encode, and extend the encoder output for things we need in the decoder"""
        if self.pad_audio:
            source, in_spatial_dim = pad_ext(source, in_spatial_dim=in_spatial_dim, opts=self.pad_audio)
        # feature extraction (default: log mel filterbank; override via the "feature_extraction" config opt)
        source, in_spatial_dim = self.feature_extraction(source, in_spatial_dim=in_spatial_dim)
        return self.encode_from_features(
            source,
            in_spatial_dim=in_spatial_dim,
            collected_outputs=collected_outputs,
            specaugment_max_spatial_dims=specaugment_max_spatial_dims,
            end_layer=end_layer,
        )

    def encode_from_features(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
        specaugment_max_spatial_dims: Optional[Tensor] = None,
        end_layer: Optional[int] = None,
    ) -> Tuple[Tensor, Dim]:
        """Encode from already-extracted features (e.g. log-mel produced online by a TTS model),
        skipping pad_audio + feature_extraction. source feature dim must be self.in_dim.
        specaugment_max_spatial_dims (per-seq) overrides the SpecAugment time-mask width,
        e.g. scaled down for short synthetic sequences.
        end_layer: if set, stop after Conformer layers [0, end_layer)
        (the audio-side counterpart of :func:`encode_from_enc_space`;
        output is in the encoder model space at the encoder frame rate, NOT decoder-transformed)."""
        if self.feature_batch_norm:
            if self.feature_norm_wants_spatial_dim:
                source = self.feature_batch_norm(source, spatial_dim=in_spatial_dim)
            else:
                source = self.feature_batch_norm(source)
        if self.feature_norm:
            source = rf.normalize(source, axis=in_spatial_dim)
        if self.feature_stats:
            source = (source - self.feature_stats.mean) / self.feature_stats.std_dev
        if self._mixup:
            source = self._mixup(source, spatial_dim=in_spatial_dim)
        # SpecAugment
        specaugment_opts = self._specaugment_opts
        if specaugment_max_spatial_dims is not None:
            specaugment_opts = {**specaugment_opts, "max_consecutive_spatial_dims": specaugment_max_spatial_dims}
        source = rf.audio.specaugment(
            source,
            spatial_dim=in_spatial_dim,
            feature_dim=self.in_dim,
            **specaugment_opts,
        )
        if end_layer is None:  # standard case
            # Encoder including convolutional frontend
            enc, enc_spatial_dim = self.encoder(
                source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs
            )
            return enc, enc_spatial_dim
        # Partial encoder: conv frontend + Conformer layers [0, end_layer)
        # (mirrors ConformerEncoder.__call__; same assumptions as encode_from_enc_space).
        x, enc_spatial_dim = self.encoder.input_layer(source, in_spatial_dim=in_spatial_dim)
        if self.encoder.input_projection and self.encoder.input_projection.in_dim in x.dims:
            x = self.encoder.input_projection(x)
        assert self.encoder.out_dim in x.dims
        assert self.encoder.pos_enc is None and self.encoder.input_embedding_scale == 1.0
        x = rf.dropout(x, self.encoder.input_dropout, axis=self.encoder.dropout_broadcast and self.encoder.out_dim)
        for name, layer in self.encoder.layers.items():
            if int(name) >= end_layer:
                break
            x = layer(x, spatial_dim=enc_spatial_dim)
            if collected_outputs is not None:
                collected_outputs[name] = x
        return x, enc_spatial_dim

    def encode_from_enc_space(
        self,
        source: Tensor,
        *,
        spatial_dim: Dim,
        start_layer: int = 0,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
        specaugment_max_spatial_dims: Optional[Tensor] = None,
        apply_specaugment: bool = True,
        apply_input_dropout: bool = True,
    ) -> Tuple[Tensor, Dim]:
        """Encode from features already in the encoder model space
        (feature dim = encoder out_dim, at the subsampled encoder frame rate),
        skipping the feature front-end and the conv subsampling,
        entering the Conformer at layer ``start_layer`` (0 = all Conformer layers).
        ``collected_outputs`` gets the per-layer outputs only for layers >= start_layer
        (same keys as rf.Sequential), so the caller must skip aux losses attached below.
        ``specaugment_max_spatial_dims``: as in :func:`encode_from_features`,
        but the time-mask width counts encoder frames here."""
        assert self.encoder.out_dim in source.dims
        if apply_specaugment:
            specaugment_opts = self._specaugment_opts
            if specaugment_max_spatial_dims is not None:
                specaugment_opts = {**specaugment_opts, "max_consecutive_spatial_dims": specaugment_max_spatial_dims}
            source = rf.audio.specaugment(
                source,
                spatial_dim=spatial_dim,
                feature_dim=self.encoder.out_dim,
                **specaugment_opts,
            )
        # Absolute pos enc / input scaling would belong before the first layer; not handled here
        # (this baseline uses rel pos enc inside the layers).
        assert self.encoder.pos_enc is None and self.encoder.input_embedding_scale == 1.0
        x = source
        if apply_input_dropout:
            x = rf.dropout(x, self.encoder.input_dropout, axis=self.encoder.dropout_broadcast and self.encoder.out_dim)
        for name, layer in self.encoder.layers.items():
            if int(name) < start_layer:
                continue
            x = layer(x, spatial_dim=spatial_dim)
            if collected_outputs is not None:
                collected_outputs[name] = x
        return x, spatial_dim

    def encode(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
        specaugment_max_spatial_dims: Optional[Tensor] = None,
    ) -> Tuple[rf.State, Dim]:
        enc, enc_spatial_dim = self.encode_no_transform(
            source,
            in_spatial_dim=in_spatial_dim,
            collected_outputs=collected_outputs,
            specaugment_max_spatial_dims=specaugment_max_spatial_dims,
        )
        return self.decoder.transform_encoder(enc, axis=enc_spatial_dim), enc_spatial_dim

    def encode_and_get_ctc_log_probs(self, source: Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, EncoderOutput, Dim]:
        """
        :param source: [B*, in_spatial_dim, in_dim]
        :param in_spatial_dim:
        :return: log_probs [B*, enc_spatial_dim', wb_target_dim], enc, enc_spatial_dim
        """
        from returnn.config import get_global_config
        from i6_experiments.users.zeyer.nn_rf.soft_collapse_repeated import soft_collapse_repeated
        from returnn.util.collect_outputs_dict import CollectOutputsDict

        # TODO/WARNING: many users of this function (encode_and_get_ctc_log_probs)
        #   also do the same soft_collapse_repeated again outside,
        #   which is redundant, inefficient, and maybe even could cause problems?
        config = get_global_config()
        ctc_soft_collapse_threshold = config.typed_value("ctc_soft_collapse_threshold", None)  # e.g. 0.8
        ctc_soft_collapse_reduce_type = config.typed_value("ctc_soft_collapse_reduce_type", "max_renorm")

        if source.feature_dim and source.feature_dim.dimension == 1:
            source = rf.squeeze(source, axis=source.feature_dim)

        ctc_layer_idx = self.enc_aux_logits[-1]
        enc_collected_outputs = CollectOutputsDict(allowed_key_patterns=[str(ctc_layer_idx - 1)])
        enc, enc_spatial_dim = self.encode_no_transform(
            source, in_spatial_dim=in_spatial_dim, collected_outputs=enc_collected_outputs
        )

        out: Tensor = enc_collected_outputs[str(ctc_layer_idx - 1)]
        assert enc_spatial_dim in out.dims
        linear = getattr(self, f"enc_aux_logits_{ctc_layer_idx}")
        logits = linear(out)
        log_probs = rf.log_softmax(logits, axis=self.wb_target_dim)  # Batch, Spatial, VocabWB
        log_probs_spatial_dim = enc_spatial_dim
        if ctc_soft_collapse_threshold is not None:
            log_probs, log_probs_spatial_dim = soft_collapse_repeated(
                log_probs,
                spatial_dim=log_probs_spatial_dim,
                classes_dim=self.wb_target_dim,
                threshold=ctc_soft_collapse_threshold,
                reduce_type=ctc_soft_collapse_reduce_type,
            )
        log_probs.feature_dim = self.wb_target_dim

        if self.ctc_am_scale != 1:
            log_probs = log_probs * self.ctc_am_scale
        if self.ctc_framewise_prior_scale:
            if self.ctc_framewise_prior_type == "static":
                log_prob_prior = self.ctc_framewise_static_prior
                assert log_prob_prior.dims == (self.wb_target_dim,)
            else:
                raise NotImplementedError(f"ctc_framewise_prior_type {self.ctc_framewise_prior_type!r}")
            log_probs -= log_prob_prior * self.ctc_framewise_prior_scale

        return log_probs, EncoderOutput(enc, enc_spatial_dim=enc_spatial_dim), log_probs_spatial_dim
