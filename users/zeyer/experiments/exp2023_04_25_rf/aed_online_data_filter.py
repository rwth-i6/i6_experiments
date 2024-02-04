"""
AED (Conformer + Trafo decoder) with online data filtering
"""

from __future__ import annotations
from typing import Optional, Any, Sequence, Dict, Tuple
import math
import functools
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.zeyer.model_interfaces import ModelDef, TrainDef
from i6_experiments.users.zeyer.returnn.models.rf_layerdrop import SequentialLayerDrop
from i6_experiments.users.zeyer.data_filtering.online_learned.rf_wrapper import LearnedDataFilter
from i6_experiments.users.zeyer.data_filtering.online_learned.filter_base import make_learned_data_filter
from .aed import _log_mel_feature_dim, _get_bos_idx, _get_eos_idx, _batch_size_factor


class Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        *,
        num_enc_layers: int = 12,
        num_dec_layers: int = 6,
        target_dim: Dim,
        wb_target_dim: Optional[Dim] = None,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        enc_aux_logits: Sequence[int] = (),  # layers
        enc_model_dim: Dim = Dim(name="enc", dimension=512),
        dec_model_dim: Dim = Dim(name="dec", dimension=512),
        enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
        enc_att_num_heads: int = 4,
        enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
        enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
        att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
        att_dropout: float = 0.1,
        enc_dropout: float = 0.1,
        enc_att_dropout: float = 0.1,
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

        self.in_dim = in_dim
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
            encoder_layer_opts=enc_conformer_layer_opts,
            num_layers=num_enc_layers,
            num_heads=enc_att_num_heads,
            dropout=enc_dropout,
            att_dropout=enc_att_dropout,
            sequential=enc_sequential,
        )
        self.decoder = TransformerDecoder(
            num_layers=num_dec_layers,
            encoder_dim=enc_model_dim,
            vocab_dim=target_dim,
            model_dim=dec_model_dim,
            sequential=dec_sequential,
        )

        self.target_dim = target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        self.att_num_heads = att_num_heads
        self.att_dropout = att_dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()

        if enc_aux_logits:
            if not wb_target_dim:
                wb_target_dim = target_dim + 1
        for i in enc_aux_logits:
            setattr(self, f"enc_aux_logits_{i}", rf.Linear(self.encoder.out_dim, wb_target_dim))

        self._specaugment_opts = {
            "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
            "max_consecutive_spatial_dims": config.typed_value("specaugment_max_consecutive_spatial_dims") or 20,
            "max_consecutive_feature_dims": config.typed_value("specaugment_max_consecutive_feature_dims")
            or (_log_mel_feature_dim // 5),
            "num_spatial_mask_factor": config.typed_value("specaugment_num_spatial_mask_factor") or 100,
        }

        self._pretrain_opts: Optional[Dict[str, Any]] = config.typed_value("pretrain_opts")

        self._mixup = None
        if config.typed_value("mixup", None) is not None:
            from i6_experiments.users.zeyer.returnn.models.rf_mixup import Mixup, MixupOpts

            self._mixup = Mixup(feature_dim=self.in_dim, opts=MixupOpts(**config.typed_value("mixup")))

        self._data_filter_opts: Dict[str, Any] = dict(config.typed_value("data_filter_opts") or {})
        self._data_filter_layer_idx: int = self._data_filter_opts.pop("layer_idx", num_enc_layers // 4)
        data_filter_pt = make_learned_data_filter(self._data_filter_opts, in_features=enc_model_dim.dimension)
        self.data_filter = LearnedDataFilter(data_filter_pt)

    def encode(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[rf.State, Dim]:
        """encode, and extend the encoder output for things we need in the decoder"""
        # log mel filterbank features
        source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
            source,
            in_spatial_dim=in_spatial_dim,
            out_dim=self.in_dim,
            sampling_rate=16_000,
            log_base=math.exp(2.3026),  # almost 10.0 but not exactly...
        )
        if self._mixup:
            source = self._mixup(source, spatial_dim=in_spatial_dim)
        # SpecAugment
        source = rf.audio.specaugment(
            source,
            spatial_dim=in_spatial_dim,
            feature_dim=self.in_dim,
            **self._specaugment_opts,
        )

        # Encoder including convolutional frontend
        # Copy inplace ConformerEncoder.__call__ here, and modify layers call.
        x_subsample, out_spatial_dim = self.encoder.input_layer(source, in_spatial_dim=in_spatial_dim)
        x_linear = self.encoder.input_projection(x_subsample)
        x = rf.dropout(
            x_linear,
            self.encoder.input_dropout,
            axis=self.encoder.dropout_broadcast and self.encoder.input_projection.out_dim,
        )

        # Copy inplace Sequential.__call__ and add our logic.
        assert type(self.encoder.layers) is rf.Sequential  # no subclass like SequentialLayerDrop
        train_flag = rf.get_run_ctx().train_flag
        assert isinstance(train_flag, bool)  # not implemented otherwise...
        for i, (name, module) in enumerate(self.encoder.layers.items()):
            x = module(x, spatial_dim=out_spatial_dim)
            if collected_outputs is not None:
                collected_outputs[name] = x
            if train_flag and i == self._data_filter_layer_idx - 1:
                x, out_spatial_dim, new_batch_dim = self.data_filter(x, spatial_dim=out_spatial_dim)

        enc, enc_spatial_dim = x, out_spatial_dim
        return self.decoder.transform_encoder(enc, axis=enc_spatial_dim), enc_spatial_dim


def from_scratch_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    enc_aux_logits = config.typed_value("aux_loss_layers")
    pos_emb_dropout = config.float("pos_emb_dropout", 0.0)
    num_enc_layers = config.int("num_enc_layers", 12)
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)

    return Model(
        in_dim,
        num_enc_layers=num_enc_layers,
        enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
        enc_ff_dim=Dim(name="enc-ff", dimension=2048, kind=Dim.Types.Feature),
        enc_att_num_heads=8,
        enc_conformer_layer_opts=dict(
            conv_norm_opts=dict(use_mask=True),
            self_att_opts=dict(
                # Shawn et al 2018 style, old RETURNN way.
                with_bias=False,
                with_linear_pos=False,
                with_pos_bias=False,
                learnable_pos_emb=True,
                separate_pos_emb_per_head=False,
                pos_emb_dropout=pos_emb_dropout,
            ),
            ff_activation=lambda x: rf.relu(x) ** 2.0,
        ),
        target_dim=target_dim,
        blank_idx=target_dim.dimension,
        bos_idx=_get_bos_idx(target_dim),
        eos_idx=_get_eos_idx(target_dim),
        enc_aux_logits=enc_aux_logits or (),
    )


from_scratch_model_def: ModelDef[Model]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = _batch_size_factor


def from_scratch_training(
    *, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim
):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    use_normalized_loss = config.bool("use_normalized_loss", True)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    collected_outputs = {}
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)

    train_flag = rf.get_run_ctx().train_flag
    assert isinstance(train_flag, bool)  # not implemented otherwise...
    if train_flag:
        targets, dim_map = model.data_filter.filter_batch(targets)
        targets_spatial_dim = dim_map[targets_spatial_dim]

    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            aux_loss = rf.ctc_loss(
                logits=aux_logits,
                targets=targets,
                input_spatial_dim=enc_spatial_dim,
                targets_spatial_dim=targets_spatial_dim,
                blank_index=model.blank_idx,
            )
            aux_loss.mark_as_loss(
                f"ctc_{layer_idx}",
                scale=aux_loss_scales[i],
                custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )
            # decoded, decoded_spatial_dim = rf.ctc_greedy_decode(aux_logits, in_spatial_dim=enc_spatial_dim)
            # error = rf.edit_distance(
            #     a=decoded, a_spatial_dim=decoded_spatial_dim, b=targets, b_spatial_dim=targets_spatial_dim
            # )
            # error.mark_as_loss("label", as_error=True, custom_inv_norm_factor=targets_spatial_dim.get_size_tensor())

    batch_dims = targets.remaining_dims(targets_spatial_dim)
    input_labels = rf.shift_right(targets, axis=targets_spatial_dim, pad_value=model.bos_idx)

    logits, _ = model.decoder(
        input_labels,
        spatial_dim=targets_spatial_dim,
        encoder=enc,
        state=model.decoder.default_initial_state(batch_dims=batch_dims),
    )

    logits_packed, pack_dim = rf.pack_padded(logits, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False)
    targets_packed, _ = rf.pack_padded(
        targets, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
    )

    log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
    log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
    loss = rf.cross_entropy(
        target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
    )
    loss.mark_as_loss("ce", scale=aed_loss_scale, use_normalized_loss=use_normalized_loss)

    best = rf.reduce_argmax(logits_packed, axis=model.target_dim)
    frame_error = best != targets_packed
    frame_error.mark_as_loss(name="fer", as_error=True)


from_scratch_training: TrainDef[Model]
from_scratch_training.learning_rate_control_error_measure = "dev_score_ce"
