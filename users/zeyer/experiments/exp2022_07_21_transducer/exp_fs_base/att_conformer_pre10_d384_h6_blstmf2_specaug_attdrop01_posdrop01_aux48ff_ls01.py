"""
Starting point, 2022-10-12
"""

from __future__ import annotations
from typing import Optional, Any, Tuple, Dict, Sequence
import numpy
from returnn_common import nn
from returnn_common.nn.encoder.blstm import BlstmEncoder
from returnn_common.asr.specaugment import specaugment_v2

from i6_experiments.users.zeyer.datasets.switchboard_2020.task import get_switchboard_task_bpe1k
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
from i6_experiments.users.zeyer.recog import recog_training_exp
from ..train import train


# Finished by sis-setup-utils/cleanup-training.py.
# Best results: {'best_scores': {'hub5e_00': 20.8, 'hub5e_01': 16.0, 'rt03s': 20.4}, 'best_epoch': 149}
_exclude_me = True


def sis_run_with_prefix(prefix_name: str):
    """run the exp"""
    if _exclude_me:
        return
    task = get_switchboard_task_bpe1k()
    model = train(
        prefix_name, task=task, config=config, post_config=post_config,
        model_def=from_scratch_model_def, train_def=from_scratch_training)
    recog_training_exp(prefix_name, task, model, recog_def=model_recog)


config = dict(
    batching="random",
    batch_size=10000,
    max_seqs=200,
    max_seq_length_default_target=75,
    accum_grad_multiple_step=2,

    # gradient_clip=0,
    # gradient_clip_global_norm = 1.0
    optimizer={"class": "nadam", "epsilon": 1e-8},
    # gradient_noise=0.0,
    learning_rate=0.0005,
    learning_rates=(
        # matching pretraining
        list(numpy.linspace(0.0000001, 0.001, num=10)) * 3 +
        list(numpy.linspace(0.0000001, 0.001, num=30))
    ),
    min_learning_rate=0.001 / 50,
    learning_rate_control="newbob_multi_epoch",
    learning_rate_control_relative_error_relative_lr=True,
    relative_error_div_by_old=True,
    use_learning_rate_control_always=True,
    newbob_multi_update_interval=1,
    learning_rate_control_min_num_epochs_per_new_lr=1,
    learning_rate_decay=0.9,
    newbob_relative_error_threshold=-0.01,
    use_last_best_model=dict(
        only_last_n=3,  # make sure in cleanup_old_models that keep_last_n covers those
        filter_score=50., min_score_dist=1.5, first_epoch=35),
)
post_config = dict(
    cleanup_old_models=dict(keep_last_n=5),
)
aux_loss_layers = [4, 8]


class Model(nn.Module):
    """Model definition"""

    def __init__(self, in_dim: nn.Dim, *,
                 num_enc_layers: int = 12,
                 nb_target_dim: nn.Dim,
                 wb_target_dim: nn.Dim,
                 blank_idx: int,
                 eos_idx: int,
                 bos_idx: int,
                 enc_aux_logits: Sequence[int] = (),  # layers
                 enc_input_allow_pool_last: bool = False,
                 enc_model_dim: nn.Dim = nn.FeatureDim("enc", 512),
                 enc_ff_dim: nn.Dim = nn.FeatureDim("enc-ff", 2048),
                 enc_att_num_heads: int = 4,
                 enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
                 enc_dropout: float = 0.1,
                 enc_att_dropout: float = 0.1,
                 l2: float = 0.0001,
                 ):
        super(Model, self).__init__()

        self.nb_target_dim = nb_target_dim
        self.wb_target_dim = wb_target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        if nn.ConformerEncoderLayer.use_dropout_after_self_att:
            nn.ConformerEncoderLayer.use_dropout_after_self_att = False
        self.in_dim = in_dim
        self.encoder = nn.ConformerEncoder(
            in_dim,
            enc_model_dim,
            ff_dim=enc_ff_dim,
            input_layer=BlstmEncoder(
                in_dim,
                nn.FeatureDim("pre-lstm", 512),
                num_layers=2, time_reduction=6,
                dropout=enc_dropout,
                allow_pool_last=enc_input_allow_pool_last,
            ),
            encoder_layer_opts=enc_conformer_layer_opts,
            num_layers=num_enc_layers,
            num_heads=enc_att_num_heads,
            dropout=enc_dropout,
            att_dropout=enc_att_dropout,
        )

        for p in self.encoder.parameters():
            p.weight_decay = l2

        for i in enc_aux_logits:
            setattr(self, f"enc_aux_logits_{i}", nn.Linear(self.encoder.out_dim, wb_target_dim))

        # https://github.com/rwth-i6/returnn-experiments/blob/master/2020-rnn-transducer/configs/base2.conv2l.specaug4a.ctc.devtrain.config

        enc_key_total_dim = nn.FeatureDim("key", 1024)
        num_heads = nn.SpatialDim("heads", 1)
        self.num_heads = num_heads

        self.enc_ctx = nn.Linear(self.encoder.out_dim, enc_key_total_dim)
        self.inv_fertility = nn.Linear(self.encoder.out_dim, num_heads)

        self.target_embed = nn.Linear(nb_target_dim, nn.FeatureDim("target_embed", 621), with_bias=False)

        self.s = nn.LSTM(self.target_embed.out_dim + num_heads * self.encoder.out_dim, nn.FeatureDim("lstm", 1000))

        self.weight_feedback = nn.Linear(num_heads, enc_key_total_dim, with_bias=False)
        self.s_transformed = nn.Linear(self.s.out_dim, enc_key_total_dim, with_bias=False)
        self.energy = nn.Linear(enc_key_total_dim, num_heads, with_bias=False)
        self.readout_in = nn.Linear(
            self.s.out_dim + self.target_embed.out_dim + num_heads * self.encoder.out_dim,
            nn.FeatureDim("readout", 1000))
        self.output_prob = nn.Linear(self.readout_in.out_dim // 2, nb_target_dim)

    def encode(self, source: nn.Tensor, *, in_spatial_dim: nn.Dim,
               collected_outputs: Optional[Dict[str, nn.Tensor]] = None,
               ) -> Tuple[Dict[str, nn.Tensor], nn.Dim]:
        """encode, and extend the encoder output for things we need in the decoder"""
        source = specaugment_v2(source, spatial_dim=in_spatial_dim, feature_dim=self.in_dim)
        enc, enc_spatial_dim = self.encoder(source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs)
        enc_ctx = self.enc_ctx(enc)
        inv_fertility = self.inv_fertility(enc)
        return dict(enc=enc, enc_ctx=enc_ctx, inv_fertility=inv_fertility), enc_spatial_dim

    @staticmethod
    def encoder_unstack(ext: Dict[str, nn.Tensor]) -> Dict[str, nn.Tensor]:
        """
        prepare the encoder output for the loop (full-sum or time-sync)
        """
        # We might improve or generalize the interface later...
        # https://github.com/rwth-i6/returnn_common/issues/202
        loop = nn.NameCtx.inner_loop()
        return {k: loop.unstack(v) for k, v in ext.items()}

    def decoder_default_initial_state(self, *, batch_dims: Sequence[nn.Dim], enc_spatial_dim: nn.Dim) -> nn.LayerState:
        """Default initial state"""
        return nn.LayerState(
            s=self.s.default_initial_state(batch_dims=batch_dims),
            att=nn.zeros(list(batch_dims) + [self.num_heads * self.encoder.out_dim]),
            accum_att_weights=nn.zeros(list(batch_dims) + [enc_spatial_dim, self.num_heads]),
        )

    def decode(self, *,
               enc: nn.Tensor,
               enc_ctx: nn.Tensor,
               inv_fertility: nn.Tensor,
               enc_spatial_dim: nn.Dim,
               prev_nb_target: Optional[nn.Tensor] = None,  # non-blank
               prev_nb_target_spatial_dim: Optional[nn.Dim] = None,
               state: Optional[nn.LayerState] = None,
               ) -> Tuple[nn.Tensor, nn.LayerState]:
        """decoder step, or operating on full seq. output logits + state"""
        if state is None:
            batch_dims = enc.batch_dims_ordered(
                remove=(enc.feature_dim, enc_spatial_dim)
                if enc_spatial_dim != nn.single_step_dim
                else (enc.feature_dim,))
            state = self.decoder_default_initial_state(batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim)
        state_ = nn.LayerState()

        prev_target_embed = self.target_embed(prev_nb_target)
        prev_att = state.att

        s, state_.s = self.s(
            nn.concat_features(prev_target_embed, prev_att),
            spatial_dim=prev_nb_target_spatial_dim,
            state=state.s)

        weight_feedback = self.weight_feedback(state.accum_att_weights)
        s_transformed = self.s_transformed(s)
        energy_in = enc_ctx + weight_feedback + s_transformed
        energy = self.energy(nn.tanh(energy_in))
        att_weights = nn.softmax(energy, axis=enc_spatial_dim)
        state_.accum_att_weights = state.accum_att_weights + att_weights * inv_fertility * 0.5
        att0 = nn.dot(att_weights, enc, reduce=enc_spatial_dim)
        att, _ = nn.merge_dims(att0, axes=(self.num_heads, self.encoder.out_dim))
        state_.att = att

        readout_in = self.readout_in(nn.concat_features(s, prev_target_embed, att))
        readout = nn.reduce_out(readout_in, mode="max", num_pieces=2)
        readout = nn.dropout(readout, dropout=0.3, axis=readout.feature_dim)
        logits = self.output_prob(readout)

        return logits, state_


def _get_bos_idx(target_dim: nn.Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.bos_label_id is not None:
        bos_idx = target_dim.vocab.bos_label_id
    elif target_dim.vocab.eos_label_id is not None:
        bos_idx = target_dim.vocab.eos_label_id
    elif "<sil>" in target_dim.vocab.user_defined_symbol_ids:
        bos_idx = target_dim.vocab.user_defined_symbol_ids["<sil>"]
    else:
        raise Exception(f"cannot determine bos_idx from vocab {target_dim.vocab}")
    return bos_idx


def _get_eos_idx(target_dim: nn.Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.eos_label_id is not None:
        eos_idx = target_dim.vocab.eos_label_id
    else:
        raise Exception(f"cannot determine eos_idx from vocab {target_dim.vocab}")
    return eos_idx


def from_scratch_model_def(*, epoch: int, in_dim: nn.Dim, target_dim: nn.Dim) -> Model:
    """Function is run within RETURNN."""
    # Pretraining:
    extra_net_dict = nn.NameCtx.top().root.extra_net_dict
    extra_net_dict["#config"] = {}
    extra_net_dict["#copy_param_mode"] = "subset"
    num_enc_layers_ = sum(([i] * 10 for i in [2, 4, 8, 12]), [])
    num_enc_layers = num_enc_layers_[epoch - 1] if epoch <= len(num_enc_layers_) else num_enc_layers_[-1]
    if num_enc_layers <= 2:
        extra_net_dict["#config"]["batch_size"] = 20000
    initial_dim_factor = 0.5
    grow_frac_enc = 1.0 - float(num_enc_layers_[-1] - num_enc_layers) / (num_enc_layers_[-1] - num_enc_layers_[0])
    dim_frac_enc = initial_dim_factor + (1.0 - initial_dim_factor) * grow_frac_enc
    enc_att_num_heads = 6
    return Model(
        in_dim,
        num_enc_layers=num_enc_layers,
        enc_input_allow_pool_last=True,
        enc_model_dim=nn.FeatureDim("enc", int(384 * dim_frac_enc / float(enc_att_num_heads)) * enc_att_num_heads),
        enc_ff_dim=nn.FeatureDim("enc-ff", int(384 * 4 * dim_frac_enc / float(enc_att_num_heads)) * enc_att_num_heads),
        enc_att_num_heads=enc_att_num_heads,
        enc_conformer_layer_opts=dict(
            self_att_opts=dict(
                pos_emb_dropout=0.1 * dim_frac_enc,
            )
        ),
        enc_aux_logits=aux_loss_layers,
        nb_target_dim=target_dim,
        wb_target_dim=target_dim + 1,
        blank_idx=target_dim.dimension,
        bos_idx=_get_bos_idx(target_dim),
        eos_idx=_get_eos_idx(target_dim),
        enc_dropout=0.1 * dim_frac_enc,
        enc_att_dropout=0.1 * dim_frac_enc,
    )


from_scratch_model_def: ModelDef[Model]
from_scratch_model_def.behavior_version = 14


def from_scratch_training(*,
                          model: Model,
                          data: nn.Tensor, data_spatial_dim: nn.Dim,
                          targets: nn.Tensor, targets_spatial_dim: nn.Dim
                          ):
    """Function is run within RETURNN."""
    collected_outputs = {}
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    for i in aux_loss_layers:
        if i > len(model.encoder.layers):
            continue
        linear = getattr(model, f"enc_aux_logits_{i}")
        aux_logits = linear(collected_outputs[str(i - 1)])
        aux_loss = nn.ctc_loss(logits=aux_logits, targets=targets)
        aux_loss.mark_as_loss(f"ctc_{i}")

    batch_dims = data.batch_dims_ordered((data_spatial_dim, data.feature_dim))
    targets, targets_spatial_dim = nn.postfix_in_time(targets, axis=targets_spatial_dim, postfix=model.eos_idx)

    loop = nn.Loop(axis=targets_spatial_dim)
    loop.state.decoder = model.decoder_default_initial_state(batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim)
    loop.state.label = nn.constant(model.bos_idx, shape=batch_dims, sparse_dim=model.nb_target_dim)
    with loop:
        logits, loop.state.decoder = model.decode(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            prev_nb_target=loop.state.label,
            prev_nb_target_spatial_dim=nn.single_step_dim,
            state=loop.state.decoder)

        loop.state.label = loop.unstack(targets)
        logits = loop.stack(logits)

    loss = nn.cross_entropy(
        target=nn.label_smoothing(targets, 0.1, axis=model.nb_target_dim),
        estimated=logits, estimated_type="logits", axis=model.nb_target_dim)
    loss.mark_as_loss("ce")


from_scratch_training: TrainDef[Model]
from_scratch_training.learning_rate_control_error_measure = "dev_score_ce"


def model_recog(*,
                model: Model,
                data: nn.Tensor, data_spatial_dim: nn.Dim,
                targets_dim: nn.Dim,  # noqa
                ) -> nn.Tensor:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return: recog results including beam
    """
    batch_dims = data.batch_dims_ordered((data_spatial_dim, data.feature_dim))
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    beam_size = 12

    loop = nn.Loop()
    loop.max_seq_len = nn.dim_value(enc_spatial_dim) * 2
    loop.state.decoder = model.decoder_default_initial_state(batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim)
    loop.state.label = nn.constant(model.bos_idx, shape=batch_dims, sparse_dim=model.nb_target_dim)
    with loop:
        logits, loop.state.decoder = model.decode(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            prev_nb_target=loop.state.label,
            prev_nb_target_spatial_dim=nn.single_step_dim,
            state=loop.state.decoder)
        log_prob = nn.log_softmax(logits, axis=model.nb_target_dim)

        label = nn.choice(
            log_prob, input_type="log_prob",
            target=None, search=True, beam_size=beam_size,
            length_normalization=False)
        loop.state.label = label
        loop.end(label == model.eos_idx, include_eos=False)
        res = loop.stack(label)

    return res


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = False
