"""
Via:
https://github.com/rwth-i6/i6_experiments/blob/main/users/rossenbach/experiments/librispeech/librispeech_100_attention/conformer_2022/conformer_tf_feature.py

From Nick (Mohammad).
"""

from __future__ import annotations
from typing import Optional, Any, Tuple, Dict, Sequence
import copy
import contextlib
import numpy
from dataclasses import dataclass, asdict
from returnn_common import nn

from i6_experiments.users.zeyer.datasets.switchboard_2020.task import get_switchboard_task_bpe1k
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
from i6_experiments.users.zeyer.recog import recog_training_exp
from ..train import train


_exclude_me = False


def sis_run_with_prefix(prefix_name: str):
    """run the exp"""
    if _exclude_me:
        return
    task = get_switchboard_task_bpe1k()
    model = train(
        prefix_name,
        task=task,
        config=config,
        post_config=post_config,
        model_def=from_scratch_model_def,
        train_def=from_scratch_training,
        num_epochs=300,
    )
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
    learning_rate_control="newbob_multi_epoch",
    learning_rate_control_relative_error_relative_lr=True,
    relative_error_div_by_old=True,
    use_learning_rate_control_always=True,
    newbob_multi_update_interval=1,
    learning_rate_control_min_num_epochs_per_new_lr=3,
    learning_rate_decay=0.9,
    newbob_relative_error_threshold=-0.01,
    use_last_best_model=dict(
        only_last_n=3,  # make sure in cleanup_old_models that keep_last_n covers those
        filter_score=50.0,
        min_score_dist=1.5,
        first_epoch=35,
    ),
)
post_config = dict(
    cleanup_old_models=dict(keep_last_n=5),
)


# LR scheduling
lr = 0.0008
min_lr_factor = 50
wup_start_lr = 0.0002
wup = 20
config["learning_rate"] = lr
config["learning_rates"] = [wup_start_lr] * 42 + list(numpy.linspace(wup_start_lr, lr, num=wup))
config["min_learning_rate"] = lr / min_lr_factor


class Model(nn.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: nn.Dim,
        *,
        nb_target_dim: nn.Dim,
        wb_target_dim: nn.Dim,
        blank_idx: int,
        bos_idx: int,
        enc_key_total_dim: nn.Dim = nn.FeatureDim("enc_key_total_dim", 200),
        att_num_heads: nn.Dim = nn.SpatialDim("att_num_heads", 1),
        att_dropout: float = 0.1,
        l2: float = 0.0001,
        epoch: int,
    ):
        super(Model, self).__init__()
        self.epoch = epoch
        self.in_dim = in_dim

        self.nb_target_dim = nb_target_dim
        self.wb_target_dim = wb_target_dim
        self.blank_idx = blank_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        self.att_num_heads = att_num_heads
        self.att_dropout = att_dropout

        encoder_out_dim = nn.FeatureDim("encoder", get_encoder_dim(epoch=epoch))
        self.encoder_out_dim = encoder_out_dim

        self.enc_ctx = nn.Linear(encoder_out_dim, enc_key_total_dim)
        self.enc_ctx_dropout = 0.2
        self.enc_win_dim = nn.SpatialDim("enc_win_dim", 5)
        self.att_query = nn.Linear(encoder_out_dim, enc_key_total_dim, with_bias=False)
        self.lm = DecoderLabelSync(nb_target_dim, l2=l2)
        self.readout_in_am = nn.Linear(2 * encoder_out_dim, nn.FeatureDim("readout", 1000), with_bias=False)
        self.readout_in_am_dropout = 0.1
        self.readout_in_lm = nn.Linear(self.lm.out_dim, self.readout_in_am.out_dim, with_bias=False)
        self.readout_in_lm_dropout = 0.1
        self.readout_in_bias = nn.Parameter([self.readout_in_am.out_dim])
        self.readout_reduce_num_pieces = 2
        self.readout_dim = self.readout_in_am.out_dim // self.readout_reduce_num_pieces
        self.out_nb_label_logits = nn.Linear(self.readout_dim, nb_target_dim)
        self.label_log_prob_dropout = 0.3
        self.out_emit_logit = nn.Linear(self.readout_dim, nn.FeatureDim("emit", 1))

        for p in self.enc_ctx.parameters():
            p.weight_decay = l2

    def encode(self, source: nn.Tensor, *, in_spatial_dim: nn.Dim) -> (Dict[str, nn.Tensor], nn.Dim):
        """encode, and extend the encoder output for things we need in the decoder"""
        enc = nn.make_layer(
            {"class": "subnetwork", "subnetwork": get_conformer_net_dict(epoch=self.epoch), "from": source},
            name=nn.NameCtx.top().root.get_child("encoder"),
        )
        self.encoder_out_dim.dimension = enc.feature_dim.dimension  # HACKY!!! maybe adapt for pretraining
        enc_spatial_dim = nn.SpatialDim("enc-spatial")
        enc = nn.make_layer(
            {
                "class": "reinterpret_data",
                "from": enc,
                "set_dim_tags": {
                    "F": self.encoder_out_dim,
                    "T": enc_spatial_dim,
                },
            },
            name="enc_set_out_dim",
        )
        enc_ctx = self.enc_ctx(nn.dropout(enc, self.enc_ctx_dropout, axis=enc.feature_dim))
        enc_ctx_win, _ = nn.window(enc_ctx, spatial_dim=enc_spatial_dim, window_dim=self.enc_win_dim)
        enc_val_win, _ = nn.window(enc, spatial_dim=enc_spatial_dim, window_dim=self.enc_win_dim)
        return dict(enc=enc, enc_ctx_win=enc_ctx_win, enc_val_win=enc_val_win), enc_spatial_dim

    @staticmethod
    def encoder_unstack(ext: Dict[str, nn.Tensor]) -> Dict[str, nn.Tensor]:
        """
        prepare the encoder output for the loop (full-sum or time-sync)
        """
        # We might improve or generalize the interface later...
        # https://github.com/rwth-i6/returnn_common/issues/202
        loop = nn.NameCtx.inner_loop()
        return {k: loop.unstack(v) for k, v in ext.items()}

    def decoder_default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> Optional[nn.LayerState]:
        """Default initial state"""
        return nn.LayerState(lm=self.lm.default_initial_state(batch_dims=batch_dims))

    def decode(
        self,
        *,
        enc: nn.Tensor,  # single frame if axis is single step, or sequence otherwise ("am" before)
        enc_spatial_dim: nn.Dim,  # single step or time axis,
        enc_ctx_win: nn.Tensor,  # like enc
        enc_val_win: nn.Tensor,  # like enc
        all_combinations_out: bool = False,  # [...,prev_nb_target_spatial_dim,axis] out
        prev_nb_target: Optional[nn.Tensor] = None,  # non-blank
        prev_nb_target_spatial_dim: Optional[nn.Dim] = None,  # one longer than target_spatial_dim, due to BOS
        prev_wb_target: Optional[nn.Tensor] = None,  # with blank
        wb_target_spatial_dim: Optional[nn.Dim] = None,  # single step or align-label spatial axis
        state: Optional[nn.LayerState] = None,
    ) -> (ProbsFromReadout, nn.LayerState):
        """decoder step, or operating on full seq"""
        if state is None:
            assert enc_spatial_dim != nn.single_step_dim, "state should be explicit, to avoid mistakes"
            batch_dims = enc.remaining_dims(
                remove=(enc.feature_dim, enc_spatial_dim)
                if enc_spatial_dim != nn.single_step_dim
                else (enc.feature_dim,)
            )
            state = self.decoder_default_initial_state(batch_dims=batch_dims)
        state_ = nn.LayerState()

        att_query = self.att_query(enc)
        att_energy = nn.dot(enc_ctx_win, att_query, reduce=att_query.feature_dim)
        att_energy = att_energy * (att_energy.feature_dim.dimension**-0.5)
        att_weights = nn.softmax(att_energy, axis=self.enc_win_dim)
        att_weights = nn.dropout(att_weights, dropout=self.att_dropout, axis=att_weights.dims)
        att = nn.dot(att_weights, enc_val_win, reduce=self.enc_win_dim)

        if all_combinations_out:
            assert prev_nb_target is not None and prev_nb_target_spatial_dim is not None
            assert prev_nb_target_spatial_dim in prev_nb_target.dims
            assert enc_spatial_dim != nn.single_step_dim
            lm_scope = contextlib.nullcontext()
            lm_input = prev_nb_target
            lm_axis = prev_nb_target_spatial_dim
        else:
            assert prev_wb_target is not None and wb_target_spatial_dim is not None
            assert wb_target_spatial_dim in {enc_spatial_dim, nn.single_step_dim}
            prev_out_emit = prev_wb_target != self.blank_idx
            lm_scope = nn.MaskedComputation(mask=prev_out_emit)
            lm_input = nn.set_sparse_dim(prev_wb_target, out_dim=self.nb_target_dim)
            lm_axis = wb_target_spatial_dim

        with lm_scope:
            lm, state_.lm = self.lm(lm_input, spatial_dim=lm_axis, state=state.lm)

            # We could have simpler code by directly concatenating the readout inputs.
            # However, for better efficiency, keep am/lm path separate initially.
            readout_in_lm_in = nn.dropout(lm, self.readout_in_lm_dropout, axis=lm.feature_dim)
            readout_in_lm = self.readout_in_lm(readout_in_lm_in)

        readout_in_am_in = nn.concat_features(enc, att)
        readout_in_am_in = nn.dropout(readout_in_am_in, self.readout_in_am_dropout, axis=readout_in_am_in.feature_dim)
        readout_in_am = self.readout_in_am(readout_in_am_in)
        readout_in = nn.combine_bc(readout_in_am, "+", readout_in_lm)
        readout_in += self.readout_in_bias
        readout = nn.reduce_out(
            readout_in, mode="max", num_pieces=self.readout_reduce_num_pieces, out_dim=self.readout_dim
        )

        return ProbsFromReadout(model=self, readout=readout), state_


class DecoderLabelSync(nn.Module):
    """
    Often called the (I)LM part, or prediction network.
    Runs label-sync, i.e. only on non-blank labels.
    """

    def __init__(
        self,
        in_dim: nn.Dim,
        *,
        embed_dim: nn.Dim = nn.FeatureDim("embed", 256),
        lstm_dim: nn.Dim = nn.FeatureDim("lstm", 1024),
        dropout: float = 0.2,
        l2: float = 0.0001,
    ):
        super(DecoderLabelSync, self).__init__()
        self.embed = nn.Linear(in_dim, embed_dim)
        self.dropout = dropout
        self.lstm = nn.LSTM(self.embed.out_dim, lstm_dim)
        self.out_dim = self.lstm.out_dim
        for p in self.parameters():
            p.weight_decay = l2

    def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> Optional[nn.LayerState]:
        """init"""
        return self.lstm.default_initial_state(batch_dims=batch_dims)

    def __call__(
        self, source: nn.Tensor, *, spatial_dim: nn.Dim, state: nn.LayerState
    ) -> Tuple[nn.Tensor, nn.LayerState]:
        embed = self.embed(source)
        embed = nn.dropout(embed, self.dropout, axis=embed.feature_dim)
        lstm, state = self.lstm(embed, spatial_dim=spatial_dim, state=state)
        return lstm, state


class ProbsFromReadout:
    """
    functions to calculate the probabilities from the readout
    """

    def __init__(self, *, model: Model, readout: nn.Tensor):
        self.model = model
        self.readout = readout

    def get_label_logits(self) -> nn.Tensor:
        """label log probs"""
        label_logits_in = nn.dropout(self.readout, self.model.label_log_prob_dropout, axis=self.readout.feature_dim)
        label_logits = self.model.out_nb_label_logits(label_logits_in)
        return label_logits

    def get_label_log_probs(self) -> nn.Tensor:
        """label log probs"""
        label_logits = self.get_label_logits()
        label_log_prob = nn.log_softmax(label_logits, axis=label_logits.feature_dim)
        return label_log_prob

    def get_emit_logit(self) -> nn.Tensor:
        """emit logit"""
        emit_logit = self.model.out_emit_logit(self.readout)
        return emit_logit

    def get_wb_label_log_probs(self) -> nn.Tensor:
        """align label log probs"""
        label_log_prob = self.get_label_log_probs()
        emit_logit = self.get_emit_logit()
        emit_log_prob = nn.log_sigmoid(emit_logit)
        blank_log_prob = nn.log_sigmoid(-emit_logit)
        label_emit_log_prob = label_log_prob + nn.squeeze(emit_log_prob, axis=emit_log_prob.feature_dim)
        assert self.model.blank_idx == label_log_prob.feature_dim.dimension  # not implemented otherwise
        output_log_prob = nn.concat_features(label_emit_log_prob, blank_log_prob)
        return output_log_prob


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


def from_scratch_model_def(*, epoch: int, in_dim: nn.Dim, target_dim: nn.Dim) -> Model:
    """Function is run within RETURNN."""
    # Pretraining:
    extra_net_dict = nn.NameCtx.top().root.extra_net_dict
    extra_net_dict["#config"] = {}
    extra_net_dict["#copy_param_mode"] = "subset"
    return Model(
        in_dim,
        nb_target_dim=target_dim,
        wb_target_dim=target_dim + 1,
        blank_idx=target_dim.dimension,
        bos_idx=_get_bos_idx(target_dim),
        epoch=epoch,
    )


from_scratch_model_def: ModelDef[Model]
from_scratch_model_def.behavior_version = 14


def from_scratch_training(
    *, model: Model, data: nn.Tensor, data_spatial_dim: nn.Dim, targets: nn.Tensor, targets_spatial_dim: nn.Dim
):
    """Function is run within RETURNN."""
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    prev_targets, prev_targets_spatial_dim = nn.prev_target_seq(
        targets, spatial_dim=targets_spatial_dim, bos_idx=model.bos_idx, out_one_longer=True
    )
    probs, _ = model.decode(
        **enc_args,
        enc_spatial_dim=enc_spatial_dim,
        all_combinations_out=True,
        prev_nb_target=prev_targets,
        prev_nb_target_spatial_dim=prev_targets_spatial_dim,
    )
    out_log_prob = probs.get_wb_label_log_probs()
    loss = nn.transducer_time_sync_full_sum_neg_log_prob(
        log_probs=out_log_prob,
        labels=targets,
        input_spatial_dim=enc_spatial_dim,
        labels_spatial_dim=targets_spatial_dim,
        prev_labels_spatial_dim=prev_targets_spatial_dim,
        blank_index=model.blank_idx,
    )
    loss.mark_as_loss("full_sum")


from_scratch_training: TrainDef[Model]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"


def model_recog(
    *,
    model: Model,
    data: nn.Tensor,
    data_spatial_dim: nn.Dim,
    targets_dim: nn.Dim,  # noqa
) -> nn.Tensor:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return: recog results including beam
    """
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    beam_size = 12

    loop = nn.Loop(axis=enc_spatial_dim)  # time-sync transducer
    loop.max_seq_len = nn.dim_value(enc_spatial_dim) * 2
    loop.state.decoder = model.decoder_default_initial_state(batch_dims=batch_dims)
    loop.state.target = nn.constant(model.blank_idx, shape=batch_dims, sparse_dim=model.wb_target_dim)
    with loop:
        enc = model.encoder_unstack(enc_args)
        probs, loop.state.decoder = model.decode(
            **enc,
            enc_spatial_dim=nn.single_step_dim,
            wb_target_spatial_dim=nn.single_step_dim,
            prev_wb_target=loop.state.target,
            state=loop.state.decoder,
        )
        log_prob = probs.get_wb_label_log_probs()
        loop.state.target = nn.choice(
            log_prob, input_type="log_prob", target=None, search=True, beam_size=beam_size, length_normalization=False
        )
        res = loop.stack(loop.state.target)

    assert model.blank_idx == targets_dim.dimension  # added at the end
    res.feature_dim.vocab = nn.Vocabulary.create_vocab_from_labels(
        targets_dim.vocab.labels + ["<blank>"], user_defined_symbols={"<blank>": model.blank_idx}
    )
    return res


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = False


# ----------


@dataclass
class ConformerEncoderArgs:
    num_blocks: int = 12
    enc_key_dim: int = 512
    att_num_heads: int = 8
    ff_dim: int = 2048
    conv_kernel_size: int = 32
    input_layer: str = "lstm-6"
    pos_enc: str = "rel"

    sandwich_conv: bool = False
    subsample: Optional[str] = None

    # param init
    ff_init: Optional[str] = None
    mhsa_init: Optional[str] = None
    mhsa_out_init: Optional[str] = None
    conv_module_init: Optional[str] = None
    start_conv_init: Optional[str] = None

    # dropout
    dropout: float = 0.1
    dropout_in: float = 0.1
    att_dropout: float = 0.1
    lstm_dropout: float = 0.1

    # norms
    batch_norm_opts: Optional[Dict[str, Any]] = None
    use_ln: bool = False

    # other regularization
    l2: float = 0.0001
    self_att_l2: float = 0.0
    rel_pos_clipping: int = 16
    stoc_layers_prob: float = 0.0


pretrain_opts = {"variant": 3}
pretrain_reps = 5

conformer_enc_args = ConformerEncoderArgs(
    num_blocks=12,
    input_layer="lstm-6",
    att_num_heads=8,
    ff_dim=2048,
    enc_key_dim=512,
    conv_kernel_size=32,
    pos_enc="rel",
    dropout=0.1,
    att_dropout=0.1,
    l2=0.0001,
)

# fairseq init
fairseq_ff_init = "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=1.0)"  # limit = sqrt(6 / (fan_in + fan_out))
fairseq_mhsa_init = "variance_scaling_initializer(mode='fan_avg', distribution='uniform', scale=0.5)"  # limit = sqrt(6 * 0.5 / (fan_in + fan_out)) = sqrt(3 / (fan_in + fan_out))
conformer_enc_args.ff_init = fairseq_ff_init
conformer_enc_args.mhsa_init = fairseq_mhsa_init
conformer_enc_args.mhsa_out_init = fairseq_ff_init
conformer_enc_args.conv_module_init = fairseq_ff_init

# overwrite BN params
conformer_enc_args.batch_norm_opts = {
    "momentum": 0.1,
    "epsilon": 1e-3,
    "update_sample_only_in_training": False,
    "delay_sample_update": False,
    "masked_time": True,
}

# conformer round 2
encoder_args = asdict(conformer_enc_args)
encoder_args.update({"input": "data"})


def get_conformer_net_dict(*, epoch: int) -> Dict[str, Any]:
    return get_conformer_encoder(epoch=epoch).network.get_net()


def get_encoder_dim(*, epoch: int) -> int:
    return get_conformer_encoder(epoch=epoch).enc_key_dim


def get_conformer_encoder(*, epoch: int) -> ConformerEncoder:
    # see Nick create_config

    # pretraining
    conformer_encoder = None
    idx = 0
    while True:
        conformer_encoder = pretrain_layers_and_dims(idx, encoder_args, **pretrain_opts)
        if not conformer_encoder:
            break
        if epoch <= (idx * pretrain_reps) + 1:
            break
        idx += 1

    if not conformer_encoder:
        conformer_encoder = ConformerEncoder(**encoder_args)
        conformer_encoder.create_network()
    return conformer_encoder


def pretrain_layers_and_dims(idx, encoder_args, variant, reduce_dims=True, initial_dim_factor=0.5):
    """
    Pretraining implementation that works for multiple encoder/decoder combinations

    :param idx:
    :param encoder_args:
    :param variant:
    :param reduce_dims:
    :param initial_dim_factor:
    :return:
    """

    InitialDimFactor = initial_dim_factor

    encoder_keys = ["ff_dim", "enc_key_dim", "conv_kernel_size"]
    encoder_args_copy = copy.deepcopy(encoder_args)

    final_num_blocks = encoder_args["num_blocks"]

    assert final_num_blocks >= 2

    idx = max(idx - 1, 0)  # repeat first 0, 0, 1, 2, ...

    if variant == 1:
        num_blocks = max(2 * idx, 1)  # 1/1/2/4/6/8/10/12 -> 8
        StartNumLayers = 1
    elif variant == 2:
        num_blocks = 2**idx  # 1/1/2/4/8/12 -> 6
        StartNumLayers = 1
    elif variant == 3:
        idx += 1
        num_blocks = 2 * idx  # 2/2/4/6/8/10/12 -> 7
        StartNumLayers = 2
    elif variant == 4:
        idx += 1
        num_blocks = 2**idx  # 2/2/4/8/12 -> 5
        StartNumLayers = 2
    elif variant == 5:
        idx += 2
        num_blocks = 2**idx  # 4/4/8/12 -> 4
        StartNumLayers = 4
    elif variant == 6:
        idx += 1  # 1 1 2 3
        num_blocks = 4 * idx  # 4 4 8 12 16
        StartNumLayers = 4
    else:
        raise ValueError("variant {} is not defined".format(variant))

    if num_blocks > final_num_blocks:
        return None

    encoder_args_copy["num_blocks"] = num_blocks
    AttNumHeads = encoder_args_copy["att_num_heads"]

    if reduce_dims:
        grow_frac_enc = 1.0 - float(final_num_blocks - num_blocks) / (final_num_blocks - StartNumLayers)
        dim_frac_enc = InitialDimFactor + (1.0 - InitialDimFactor) * grow_frac_enc

        for key in encoder_keys:
            encoder_args_copy[key] = int(encoder_args[key] * dim_frac_enc / float(AttNumHeads)) * AttNumHeads

    else:
        dim_frac_enc = 1

    # do not enable regulizations in the first pretraining step to make it more stable
    for k in encoder_args_copy.keys():
        if "dropout" in k and encoder_args_copy[k] is not None:
            if idx <= 1:
                encoder_args_copy[k] = 0.0
            else:
                encoder_args_copy[k] *= dim_frac_enc
        if "l2" in k and encoder_args_copy[k] is not None:
            if idx <= 1:
                encoder_args_copy[k] = 0.0
            else:
                encoder_args_copy[k] *= dim_frac_enc

    conformer_encoder = ConformerEncoder(**encoder_args_copy)
    conformer_encoder.create_network()
    return conformer_encoder


class ConformerEncoder:
    """
    Represents Conformer Encoder Architecture

    * Conformer: Convolution-augmented Transformer for Speech Recognition
    * Ref: https://arxiv.org/abs/2005.08100
    """

    def __init__(
        self,
        input="data",
        input_layer="conv",
        num_blocks=16,
        conv_kernel_size=32,
        specaug=True,
        pos_enc="rel",
        activation="swish",
        block_final_norm=True,
        ff_dim=512,
        ff_bias=True,
        dropout=0.1,
        att_dropout=0.1,
        enc_key_dim=256,
        att_num_heads=4,
        l2=0.0,
        lstm_dropout=0.1,
        rec_weight_dropout=0.0,
        subsample=None,
        start_conv_init=None,
        conv_module_init=None,
        mhsa_init=None,
        mhsa_out_init=None,
        ff_init=None,
        rel_pos_clipping=16,
        dropout_in=0.1,
        stoc_layers_prob=0.0,
        batch_norm_opts=None,
        pytorch_bn_opts=False,
        use_ln=False,
        pooling_str=None,
        self_att_l2=0.0,
        sandwich_conv=False,
        add_to_prefix_name=None,
        output_layer_name="output",
        create_only_blocks=False,
        no_mhsa_module=False,
        proj_input=False,
    ):
        """
        :param str input: input layer name
        :param str input_layer: type of input layer which does subsampling
        :param int num_blocks: number of Conformer blocks
        :param int conv_kernel_size: kernel size for conv layers in Convolution module
        :param bool|None specaug: If true, then SpecAug is appliedi wi
        :param str|None activation: activation used to sandwich modules
        :param bool block_final_norm: if True, apply layer norm at the end of each conformer block
        :param int|None ff_dim: dimension of the first linear layer in FF module
        :param str|None ff_init: FF layers initialization
        :param bool|None ff_bias: If true, then bias is used for the FF layers
        :param float dropout: general dropout
        :param float att_dropout: dropout applied to attention weights
        :param int enc_key_dim: encoder key dimension, also denoted as d_model, or d_key
        :param int att_num_heads: the number of attention heads
        :param float l2: add L2 regularization for trainable weights parameters
        :param float lstm_dropout: dropout applied to the input of the LSTMs in case they are used
        :param float rec_weight_dropout: dropout applied to the hidden-to-hidden weight matrices of the LSTM in case used
        """

        self.input = input
        self.input_layer = input_layer

        self.num_blocks = num_blocks
        self.conv_kernel_size = conv_kernel_size

        self.pos_enc = pos_enc
        self.rel_pos_clipping = rel_pos_clipping

        self.ff_bias = ff_bias

        self.specaug = specaug

        self.activation = activation

        self.block_final_norm = block_final_norm

        self.dropout = dropout
        self.att_dropout = att_dropout
        self.lstm_dropout = lstm_dropout

        self.dropout_in = dropout_in

        # key and value dimensions are the same
        self.enc_key_dim = enc_key_dim
        self.enc_value_dim = enc_key_dim
        self.att_num_heads = att_num_heads
        self.enc_key_per_head_dim = enc_key_dim // att_num_heads
        self.enc_val_per_head_dim = enc_key_dim // att_num_heads

        self.ff_dim = ff_dim
        if self.ff_dim is None:
            self.ff_dim = 2 * self.enc_key_dim

        self.l2 = l2
        self.self_att_l2 = self_att_l2
        self.rec_weight_dropout = rec_weight_dropout

        if batch_norm_opts is None:
            batch_norm_opts = {}

        if pytorch_bn_opts:
            batch_norm_opts["momentum"] = 0.1
            batch_norm_opts["epsilon"] = 1e-3
            batch_norm_opts["update_sample_only_in_training"] = True
            batch_norm_opts["delay_sample_update"] = True

        self.batch_norm_opts = batch_norm_opts

        self.start_conv_init = start_conv_init
        self.conv_module_init = conv_module_init
        self.mhsa_init = mhsa_init
        self.mhsa_out_init = mhsa_out_init
        self.ff_init = ff_init

        self.sandwich_conv = sandwich_conv

        # add maxpooling layers
        self.subsample = subsample
        self.subsample_list = [1] * num_blocks
        if subsample:
            for idx, s in enumerate(map(int, subsample.split("_")[:num_blocks])):
                self.subsample_list[idx] = s

        self.network = ReturnnNetwork()

        self.stoc_layers_prob = stoc_layers_prob
        if stoc_layers_prob:
            # this is only used to define the shape for the dropout mask (it needs source)
            self.mask_var = self.network.add_variable_layer("mask_var", shape=(1,), init=1)

        self.use_ln = use_ln

        self.pooling_str = pooling_str

        self.add_to_prefix_name = add_to_prefix_name
        self.output_layer_name = output_layer_name

        self.create_only_blocks = create_only_blocks

        self.no_mhsa_module = no_mhsa_module
        self.proj_input = proj_input

    def _get_stoc_layer_dropout(self, layer_index):
        """
        Returns the probability to drop a layer
          p_l = l / L * (1 - p)  where p is a hyperparameter

        :param int layer_index: index of layer
        :rtype float
        """
        return layer_index / self.num_blocks * (1 - self.stoc_layers_prob)

    def _add_stoc_res_layer(self, prefix_name, f_x, x, layer_index):
        """
        Add stochastic layer to the network. the layer will be scaled and masked
          M * F(x) * (1 / 1 - p_l)

        :param prefix_name: prefix name for layer
        :param f_x: module output. e.g self-attention or FF
        :param x: input
        :param int layer_index: index of layer
        :rtype list[str]
        """
        stoc_layer_drop = self._get_stoc_layer_dropout(layer_index)
        stoc_scale = 1 / 1 - stoc_layer_drop
        mask = self.network.add_dropout_layer("stoc_layer{}_mask".format(layer_index), self.mask_var, stoc_layer_drop)
        masked_and_scaled_out = self.network.add_eval_layer(
            "{}_scaled_mask_layer".format(prefix_name),
            [mask, f_x],
            eval="source(0) * source(1) * {}".format(stoc_scale),
        )
        return [masked_and_scaled_out, x]

    def _create_ff_module(self, prefix_name, i, source, layer_index):
        """
        Add Feed Forward Module:
          LN -> FFN -> Swish -> Dropout -> FFN -> Dropout

        :param str prefix_name: some prefix name
        :param int i: FF module index
        :param str source: name of source layer
        :param int layer_index: index of layer
        :return: last layer name of this module
        :rtype: str
        """
        prefix_name = prefix_name + "_ffmod_{}".format(i)

        ln = self.network.add_layer_norm_layer("{}_ln".format(prefix_name), source)

        ff1 = self.network.add_linear_layer(
            "{}_ff1".format(prefix_name),
            ln,
            n_out=self.ff_dim,
            l2=self.l2,
            forward_weights_init=self.ff_init,
            with_bias=self.ff_bias,
        )

        swish_act = self.network.add_activation_layer("{}_swish".format(prefix_name), ff1, activation=self.activation)

        drop1 = self.network.add_dropout_layer("{}_drop1".format(prefix_name), swish_act, dropout=self.dropout)

        ff2 = self.network.add_linear_layer(
            "{}_ff2".format(prefix_name),
            drop1,
            n_out=self.enc_key_dim,
            l2=self.l2,
            forward_weights_init=self.ff_init,
            with_bias=self.ff_bias,
        )

        drop2 = self.network.add_dropout_layer("{}_drop2".format(prefix_name), ff2, dropout=self.dropout)

        half_step_ff = self.network.add_eval_layer("{}_half_step".format(prefix_name), drop2, eval="0.5 * source(0)")

        res_inputs = [half_step_ff, source]

        if self.stoc_layers_prob:
            res_inputs = self._add_stoc_res_layer(prefix_name, f_x=half_step_ff, x=source, layer_index=layer_index)

        ff_module_res = self.network.add_combine_layer(
            "{}_res".format(prefix_name), kind="add", source=res_inputs, n_out=self.enc_key_dim
        )

        return ff_module_res

    def _create_mhsa_module(self, prefix_name, source, layer_index):
        """
        Add Multi-Headed Selft-Attention Module:
          LN + MHSA + Dropout

        :param str prefix_name: some prefix name
        :param str source: name of source layer
        :param int layer_index: index of layer
        :return: last layer name of this module
        :rtype: str
        """
        prefix_name = "{}_self_att".format(prefix_name)
        ln = self.network.add_layer_norm_layer("{}_ln".format(prefix_name), source)
        ln_rel_pos_enc = None

        if self.pos_enc == "rel":
            ln_rel_pos_enc = self.network.add_relative_pos_encoding_layer(
                "{}_ln_rel_pos_enc".format(prefix_name),
                ln,
                n_out=self.enc_key_per_head_dim,
                forward_weights_init=self.ff_init,
                clipping=self.rel_pos_clipping,
            )

        mhsa = self.network.add_self_att_layer(
            "{}".format(prefix_name),
            ln,
            n_out=self.enc_value_dim,
            num_heads=self.att_num_heads,
            total_key_dim=self.enc_key_dim,
            att_dropout=self.att_dropout,
            forward_weights_init=self.mhsa_init,
            key_shift=ln_rel_pos_enc if ln_rel_pos_enc is not None else None,
            l2=self.self_att_l2,
        )

        mhsa_linear = self.network.add_linear_layer(
            "{}_linear".format(prefix_name),
            mhsa,
            n_out=self.enc_key_dim,
            l2=self.l2,
            forward_weights_init=self.mhsa_out_init,
            with_bias=False,
        )

        drop = self.network.add_dropout_layer("{}_dropout".format(prefix_name), mhsa_linear, dropout=self.dropout)

        res_inputs = [drop, source]

        if self.stoc_layers_prob:
            res_inputs = self._add_stoc_res_layer(prefix_name, f_x=drop, x=source, layer_index=layer_index)

        mhsa_res = self.network.add_combine_layer(
            "{}_res".format(prefix_name), kind="add", source=res_inputs, n_out=self.enc_value_dim
        )
        return mhsa_res

    def _create_convolution_module(self, prefix_name, source, layer_index, half_step=False):
        """
        Add Convolution Module:
          LN + point-wise-conv + GLU + depth-wise-conv + BN + Swish + point-wise-conv + Dropout

        :param str prefix_name: some prefix name
        :param str source: name of source layer
        :param int layer_index: index of layer
        :return: last layer name of this module
        :rtype: str
        """
        prefix_name = "{}_conv_mod".format(prefix_name)

        ln = self.network.add_layer_norm_layer("{}_ln".format(prefix_name), source)

        pointwise_conv1 = self.network.add_linear_layer(
            "{}_pointwise_conv1".format(prefix_name),
            ln,
            n_out=2 * self.enc_key_dim,
            activation=None,
            l2=self.l2,
            with_bias=self.ff_bias,
            forward_weights_init=self.conv_module_init,
        )

        glu_act = self.network.add_gating_layer("{}_glu".format(prefix_name), pointwise_conv1)

        depthwise_conv = self.network.add_conv_layer(
            "{}_depthwise_conv2".format(prefix_name),
            glu_act,
            n_out=self.enc_key_dim,
            filter_size=(self.conv_kernel_size,),
            groups=self.enc_key_dim,
            l2=self.l2,
            forward_weights_init=self.conv_module_init,
        )

        if self.use_ln:
            bn = self.network.add_layer_norm_layer("{}_layer_norm".format(prefix_name), depthwise_conv)
        else:
            bn = self.network.add_batch_norm_layer(
                "{}_bn".format(prefix_name), depthwise_conv, opts=self.batch_norm_opts
            )

        swish_act = self.network.add_activation_layer("{}_swish".format(prefix_name), bn, activation="swish")

        pointwise_conv2 = self.network.add_linear_layer(
            "{}_pointwise_conv2".format(prefix_name),
            swish_act,
            n_out=self.enc_key_dim,
            activation=None,
            l2=self.l2,
            with_bias=self.ff_bias,
            forward_weights_init=self.conv_module_init,
        )

        drop = self.network.add_dropout_layer("{}_drop".format(prefix_name), pointwise_conv2, dropout=self.dropout)

        if half_step:
            drop = self.network.add_eval_layer("{}_half_step".format(prefix_name), drop, eval="0.5 * source(0)")

        res_inputs = [drop, source]

        if self.stoc_layers_prob:
            res_inputs = self._add_stoc_res_layer(prefix_name, f_x=drop, x=source, layer_index=layer_index)

        res = self.network.add_combine_layer(
            "{}_res".format(prefix_name), kind="add", source=res_inputs, n_out=self.enc_key_dim
        )
        return res

    def _create_conformer_block(self, i, source):
        """
        Add the whole Conformer block:
          x1 = x0 + 1/2 * FFN(x0)             (FFN module 1)
          x2 = x1 + MHSA(x1)                  (MHSA)
          x3 = x2 + Conv(x2)                  (Conv module)
          x4 = LayerNorm(x3 + 1/2 * FFN(x3))  (FFN module 2)

        :param int i: layer index
        :param str source: name of source layer
        :return: last layer name of this module
        :rtype: str
        """
        if self.add_to_prefix_name:
            prefix_name = "conformer_block_%s_%02i" % (self.add_to_prefix_name, i)
        else:
            prefix_name = "conformer_block_%02i" % i
        ff_module1 = self._create_ff_module(prefix_name, 1, source, i)

        if self.no_mhsa_module:
            mhsa = ff_module1  # use FF1 module output as input to conv module
        else:
            mhsa_input = ff_module1
            if self.sandwich_conv:
                conv_module1 = self._create_convolution_module(prefix_name + "_sandwich", ff_module1, i, half_step=True)
                mhsa_input = conv_module1
            mhsa = self._create_mhsa_module(prefix_name, mhsa_input, i)

        conv_module = self._create_convolution_module(prefix_name, mhsa, i, half_step=self.sandwich_conv)

        ff_module2 = self._create_ff_module(prefix_name, 2, conv_module, i)
        res = ff_module2
        if self.block_final_norm:
            res = self.network.add_layer_norm_layer("{}_ln".format(prefix_name), res)
        if self.subsample:
            assert 0 <= i - 1 < len(self.subsample)
            subsample_factor = self.subsample_list[i - 1]
            if subsample_factor > 1:
                res = self.network.add_pool_layer(res + "_pool{}".format(i), res, pool_size=(subsample_factor,))
        res = self.network.add_copy_layer(prefix_name, res)
        return res

    def _create_all_network_parts(self):
        """
        ConvSubsampling/LSTM -> Linear -> Dropout -> [Conformer Blocks] x N
        """
        data = self.input
        if self.specaug:
            data = self.network.add_eval_layer("source", data, eval=transform)

        subsampled_input = None
        if self.input_layer is None:
            subsampled_input = data
        elif "lstm" in self.input_layer:
            sample_factor = int(self.input_layer.split("-")[1])
            pool_sizes = None
            if sample_factor == 2:
                pool_sizes = [2, 1]
            elif sample_factor == 4:
                pool_sizes = [2, 2]
            elif sample_factor == 6:
                pool_sizes = [3, 2]
            # add 2 LSTM layers with max pooling to subsample and encode positional information
            subsampled_input = self.network.add_lstm_layers(
                data,
                num_layers=2,
                lstm_dim=self.enc_key_dim,
                dropout=self.lstm_dropout,
                bidirectional=True,
                rec_weight_dropout=self.rec_weight_dropout,
                l2=self.l2,
                pool_sizes=pool_sizes,
            )
        elif self.input_layer == "conv":
            # subsample by 4
            subsampled_input = self.network.add_conv_block(
                "conv_merged",
                data,
                hwpc_sizes=[((3, 3), (2, 2), 128), ((3, 3), (2, 2), 128)],
                l2=self.l2,
                activation="relu",
                init=self.start_conv_init,
            )
        elif self.input_layer == "conv-new":
            subsampled_input = self.network.add_conv_block(
                "conv_merged",
                data,
                hwpc_sizes=[((3, 3), (2, 2), 16), ((3, 3), (2, 2), 32)],
                l2=self.l2,
                activation="relu",
                init=self.start_conv_init,
            )
        elif self.input_layer == "vgg":
            subsampled_input = self.network.add_conv_block(
                "vgg_conv_merged",
                data,
                hwpc_sizes=[((3, 3), (2, 2), 32), ((3, 3), (2, 2), 64)],
                l2=self.l2,
                activation="relu",
                init=self.start_conv_init,
            )
        elif self.input_layer == "neural_sp_conv":
            subsampled_input = self.network.add_conv_block(
                "conv_merged",
                data,
                hwpc_sizes=([(3, 3), (1, 1), 32], [(3, 3), (2, 2), 32]),
                l2=self.l2,
                activation="relu",
                init=self.start_conv_init,
            )

        assert subsampled_input is not None

        source_linear = self.network.add_linear_layer(
            "source_linear",
            subsampled_input,
            n_out=self.enc_key_dim,
            l2=self.l2,
            forward_weights_init=self.ff_init,
            with_bias=False,
        )

        # add positional encoding
        if self.pos_enc == "abs":
            source_linear = self.network.add_pos_encoding_layer(
                "{}_abs_pos_enc".format(subsampled_input), source_linear
            )

        if self.dropout_in:
            source_linear = self.network.add_dropout_layer("source_dropout", source_linear, dropout=self.dropout_in)

        conformer_block_src = source_linear
        for i in range(1, self.num_blocks + 1):
            conformer_block_src = self._create_conformer_block(i, conformer_block_src)

        encoder = self.network.add_copy_layer(self.output_layer_name, conformer_block_src)

        return encoder

    def _create_conformer_blocks(self, input):
        if self.proj_input:
            conformer_block_src = self.network.add_linear_layer(
                "encoder_proj", input, n_out=self.enc_key_dim, activation=None, with_bias=False
            )
        else:
            conformer_block_src = input
        for i in range(1, self.num_blocks + 1):
            conformer_block_src = self._create_conformer_block(i, conformer_block_src)
        encoder = self.network.add_copy_layer(self.output_layer_name, conformer_block_src)
        return encoder

    def create_network(self):
        # create only conformer blocks without front-end, etc
        if self.create_only_blocks:
            return self._create_conformer_blocks(input=self.input)
        return self._create_all_network_parts()


class ReturnnNetwork:
    """
    Represents a generic RETURNN network
    see docs: https://returnn.readthedocs.io/en/latest/
    """

    def __init__(self):
        self._net = {}

    def get_net(self):
        return self._net

    def add_copy_layer(self, name, source, **kwargs):
        self._net[name] = {"class": "copy", "from": source}
        self._net[name].update(kwargs)
        return name

    def add_eval_layer(self, name, source, eval, **kwargs):
        self._net[name] = {"class": "eval", "eval": eval, "from": source}
        self._net[name].update(kwargs)
        return name

    def add_split_dim_layer(self, name, source, axis="F", dims=(-1, 1), **kwargs):
        self._net[name] = {"class": "split_dims", "axis": axis, "dims": dims, "from": source}
        self._net[name].update(kwargs)
        return name

    def add_conv_layer(
        self,
        name,
        source,
        filter_size,
        n_out,
        l2,
        padding="same",
        activation=None,
        with_bias=True,
        forward_weights_init=None,
        **kwargs,
    ):
        d = {
            "class": "conv",
            "from": source,
            "padding": padding,
            "filter_size": filter_size,
            "n_out": n_out,
            "activation": activation,
            "with_bias": with_bias,
        }
        if l2:
            d["L2"] = l2
        if forward_weights_init:
            d["forward_weights_init"] = forward_weights_init
        d.update(kwargs)
        self._net[name] = d
        return name

    def add_linear_layer(
        self,
        name,
        source,
        n_out,
        activation=None,
        with_bias=True,
        dropout=0.0,
        l2=0.0,
        forward_weights_init=None,
        **kwargs,
    ):
        d = {"class": "linear", "activation": activation, "with_bias": with_bias, "from": source, "n_out": n_out}
        if dropout:
            d["dropout"] = dropout
        if l2:
            d["L2"] = l2
        if forward_weights_init:
            d["forward_weights_init"] = forward_weights_init
        d.update(kwargs)
        self._net[name] = d
        return name

    def add_pool_layer(self, name, source, pool_size, mode="max", **kwargs):
        self._net[name] = {"class": "pool", "from": source, "pool_size": pool_size, "mode": mode, "trainable": False}
        self._net[name].update(kwargs)
        return name

    def add_merge_dims_layer(self, name, source, axes="static", **kwargs):
        self._net[name] = {"class": "merge_dims", "from": source, "axes": axes}
        self._net[name].update(kwargs)
        return name

    def add_rec_layer(
        self,
        name,
        source,
        n_out,
        l2,
        rec_weight_dropout=0.0,
        weights_init=None,
        direction=1,
        unit="nativelstm2",
        **kwargs,
    ):
        d = {"class": "rec", "unit": unit, "n_out": n_out, "direction": direction, "from": source}
        if l2:
            d["L2"] = l2
        if rec_weight_dropout:
            if "unit_opts" not in d:
                d["unit_opts"] = {}
            d["unit_opts"].update({"rec_weight_dropout": rec_weight_dropout})
        if weights_init:
            if "unit_opts" not in d:
                d["unit_opts"] = {}
            d["unit_opts"].update({"forward_weights_init": weights_init, "recurrent_weights_init": weights_init})
        d.update(kwargs)
        self._net[name] = d
        return name

    def add_choice_layer(self, name, source, target, beam_size=12, initial_output=0, input_type=None, **kwargs):
        self._net[name] = {
            "class": "choice",
            "target": target,
            "beam_size": beam_size,
            "from": source,
            "initial_output": initial_output,
        }
        if input_type:
            self._net[name]["input_type"] = input_type
        self._net[name].update(kwargs)
        return name

    def add_compare_layer(self, name, source, value, kind="equal", **kwargs):
        self._net[name] = {"class": "compare", "kind": kind, "from": source, "value": value}
        self._net[name].update(kwargs)
        return name

    def add_combine_layer(self, name, source, kind, n_out, **kwargs):
        self._net[name] = {"class": "combine", "kind": kind, "from": source, "n_out": n_out}
        self._net[name].update(kwargs)
        return name

    def add_activation_layer(self, name, source, activation, **kwargs):
        self._net[name] = {"class": "activation", "activation": activation, "from": source}
        self._net[name].update(kwargs)
        return name

    def add_softmax_over_spatial_layer(self, name, source, **kwargs):
        self._net[name] = {"class": "softmax_over_spatial", "from": source}
        self._net[name].update(kwargs)
        return name

    def add_generic_att_layer(self, name, weights, base, **kwargs):
        self._net[name] = {"class": "generic_attention", "weights": weights, "base": base}
        self._net[name].update(kwargs)
        return name

    def add_rnn_cell_layer(
        self, name, source, n_out, unit="LSTMBlock", l2=0.0, unit_opts=None, weights_init=None, **kwargs
    ):
        d = {"class": "rnn_cell", "unit": unit, "n_out": n_out, "from": source}
        if l2:
            d["L2"] = l2
        if unit_opts:
            d["unit_opts"] = unit_opts
        if weights_init:
            d["weights_init"] = weights_init
        d.update(kwargs)
        self._net[name] = d
        return name

    def add_softmax_layer(
        self,
        name,
        source,
        l2=None,
        loss=None,
        target=None,
        dropout=0.0,
        loss_opts=None,
        forward_weights_init=None,
        loss_scale=None,
        **kwargs,
    ):
        d = {"class": "softmax", "from": source}
        if dropout:
            d["dropout"] = dropout
        if target:
            d["target"] = target
        if loss:
            d["loss"] = loss
            if loss_opts:
                d["loss_opts"] = loss_opts
        if l2:
            d["L2"] = l2
        if forward_weights_init:
            d["forward_weights_init"] = forward_weights_init
        if loss_scale:
            d["loss_scale"] = loss_scale
        d.update(kwargs)
        self._net[name] = d
        return name

    def add_dropout_layer(self, name, source, dropout, dropout_noise_shape=None, **kwargs):
        self._net[name] = {"class": "dropout", "from": source, "dropout": dropout}
        if dropout_noise_shape:
            self._net[name]["dropout_noise_shape"] = dropout_noise_shape
        self._net[name].update(kwargs)
        return name

    def add_reduceout_layer(self, name, source, num_pieces=2, mode="max", **kwargs):
        self._net[name] = {"class": "reduce_out", "from": source, "num_pieces": num_pieces, "mode": mode}
        self._net[name].update(kwargs)
        return name

    def add_subnet_rec_layer(self, name, unit, target, source=None, **kwargs):
        if source is None:
            source = []
        self._net[name] = {
            "class": "rec",
            "from": source,
            "unit": unit,
            "target": target,
            "max_seq_len": "max_len_from('base:encoder')",
        }
        self._net[name].update(kwargs)
        return name

    def add_decide_layer(self, name, source, target, loss="edit_distance", **kwargs):
        self._net[name] = {"class": "decide", "from": source, "loss": loss, "target": target}
        self._net[name].update(kwargs)
        return name

    def add_slice_layer(self, name, source, axis, **kwargs):
        self._net[name] = {"class": "slice", "from": source, "axis": axis, **kwargs}
        return name

    def add_subnetwork(self, name, source, subnetwork_net, **kwargs):
        self._net[name] = {"class": "subnetwork", "from": source, "subnetwork": subnetwork_net, **kwargs}
        return name

    def add_layer_norm_layer(self, name, source, **kwargs):
        self._net[name] = {"class": "layer_norm", "from": source, **kwargs}
        return name

    def add_batch_norm_layer(self, name, source, opts=None, **kwargs):
        self._net[name] = {"class": "batch_norm", "from": source, **kwargs}
        if opts:
            assert isinstance(opts, dict)
            self._net[name].update(opts)
        return name

    def add_self_att_layer(
        self,
        name,
        source,
        n_out,
        num_heads,
        total_key_dim,
        att_dropout=0.0,
        key_shift=None,
        forward_weights_init=None,
        l2=0.0,
        **kwargs,
    ):
        d = {
            "class": "self_attention",
            "from": source,
            "n_out": n_out,
            "num_heads": num_heads,
            "total_key_dim": total_key_dim,
        }
        if att_dropout:
            d["attention_dropout"] = att_dropout
        if key_shift:
            d["key_shift"] = key_shift
        if forward_weights_init:
            d["forward_weights_init"] = forward_weights_init
        if l2:
            d["L2"] = l2
        d.update(kwargs)
        self._net[name] = d
        return name

    def add_pos_encoding_layer(self, name, source, add_to_input=True, **kwargs):
        self._net[name] = {"class": "positional_encoding", "from": source, "add_to_input": add_to_input}
        self._net[name].update(kwargs)
        return name

    def add_relative_pos_encoding_layer(self, name, source, n_out, forward_weights_init=None, **kwargs):
        self._net[name] = {"class": "relative_positional_encoding", "from": source, "n_out": n_out}
        if forward_weights_init:
            self._net[name]["forward_weights_init"] = forward_weights_init
        self._net[name].update(kwargs)
        return name

    def add_constant_layer(self, name, value, **kwargs):
        self._net[name] = {"class": "constant", "value": value}
        self._net[name].update(kwargs)
        return name

    def add_gating_layer(self, name, source, activation="identity", **kwargs):
        """
        out = activation(a) * gate_activation(b)  (gate_activation is sigmoid by default)
        In case of one source input, it will split by 2 over the feature dimension
        """
        self._net[name] = {"class": "gating", "from": source, "activation": activation}
        self._net[name].update(kwargs)
        return name

    def add_pad_layer(self, name, source, axes, padding, **kwargs):
        self._net[name] = {"class": "pad", "from": source, "axes": axes, "padding": padding}
        self._net[name].update(**kwargs)
        return name

    def add_reduce_layer(self, name, source, mode, axes, keep_dims=False, **kwargs):
        self._net[name] = {"class": "reduce", "from": source, "mode": mode, "axes": axes, "keep_dims": keep_dims}
        self._net[name].update(**kwargs)
        return name

    def add_variable_layer(self, name, shape, **kwargs):
        self._net[name] = {"class": "variable", "shape": shape}
        self._net[name].update(kwargs)
        return name

    def add_switch_layer(self, name, condition, true_from, false_from, **kwargs):
        self._net[name] = {"class": "switch", "condition": condition, "true_from": true_from, "false_from": false_from}
        self._net[name].update(kwargs)
        return name

    def add_kenlm_layer(self, name, lm_file, **kwargs):
        self._net[name] = {"class": "kenlm", "lm_file": lm_file, **kwargs}
        return name

    def add_length_layer(self, name, source, **kwargs):
        self._net[name] = {"class": "length", "from": source, **kwargs}
        return name

    def add_reinterpret_data_layer(self, name, source, **kwargs):
        self._net[name] = {"class": "reinterpret_data", "from": source, **kwargs}
        return name

    def add_conv_block(self, name, source, hwpc_sizes, l2, activation, dropout=0.0, init=None):
        src = self.add_split_dim_layer("source0", source)
        for idx, hwpc in enumerate(hwpc_sizes):
            filter_size, pool_size, n_out = hwpc
            src = self.add_conv_layer(
                "conv%i" % idx,
                src,
                filter_size=filter_size,
                n_out=n_out,
                l2=l2,
                activation=activation,
                forward_weights_init=init,
            )
            if pool_size:
                src = self.add_pool_layer("conv%ip" % idx, src, pool_size=pool_size, padding="same")
        if dropout:
            src = self.add_dropout_layer("conv_dropout", src, dropout=dropout)
        return self.add_merge_dims_layer(name, src)

    def add_lstm_layers(self, input, num_layers, lstm_dim, dropout, l2, rec_weight_dropout, pool_sizes, bidirectional):
        src = input
        pool_idx = 0
        for layer in range(num_layers):
            lstm_fw_name = self.add_rec_layer(
                name="lstm%i_fw" % layer,
                source=src,
                n_out=lstm_dim,
                direction=1,
                dropout=dropout,
                l2=l2,
                rec_weight_dropout=rec_weight_dropout,
            )
            if bidirectional:
                lstm_bw_name = self.add_rec_layer(
                    name="lstm%i_bw" % layer,
                    source=src,
                    n_out=lstm_dim,
                    direction=-1,
                    dropout=dropout,
                    l2=l2,
                    rec_weight_dropout=rec_weight_dropout,
                )
                src = [lstm_fw_name, lstm_bw_name]
            else:
                src = lstm_fw_name
            if pool_sizes and pool_idx < len(pool_sizes):
                lstm_pool_name = "lstm%i_pool" % layer
                src = self.add_pool_layer(
                    name=lstm_pool_name, source=src, pool_size=(pool_sizes[pool_idx],), padding="same"
                )
                pool_idx += 1
        return src

    def add_dot_layer(self, name, source, **kwargs):
        self._net[name] = {"class": "dot", "from": source}
        self._net[name].update(kwargs)
        return name

    def __setitem__(self, key, value):
        self._net[key] = value

    def __getitem__(self, item):
        return self._net[item]

    def update(self, d: dict):
        self._net.update(d)

    def __str__(self):
        """
        Only for debugging
        """
        res = "network = {\n"
        for k, v in self._net.items():
            res += "%s: %r\n" % (k, v)
        return res + "}"


def summary(name, x):
    """
    :param str name:
    :param tf.Tensor x: (batch,time,feature)
    """
    from returnn.tf.compat import v1 as tf

    # tf.summary.image wants [batch_size, height,  width, channels],
    # we have (batch, time, feature).
    img = tf.expand_dims(x, axis=3)  # (batch,time,feature,1)
    img = tf.transpose(img, [0, 2, 1, 3])  # (batch,feature,time,1)
    tf.summary.image(name, img, max_outputs=10)
    tf.summary.scalar("%s_max_abs" % name, tf.reduce_max(tf.abs(x)))
    mean = tf.reduce_mean(x)
    tf.summary.scalar("%s_mean" % name, mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(x - mean)))
    tf.summary.scalar("%s_stddev" % name, stddev)
    tf.summary.histogram("%s_hist" % name, tf.reduce_max(tf.abs(x), axis=2))


def _mask(x, batch_axis, axis, pos, max_amount):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param tf.Tensor pos: (batch,)
    :param int|tf.Tensor max_amount: inclusive
    """
    from returnn.tf.compat import v1 as tf

    ndim = x.get_shape().ndims
    n_batch = tf.shape(x)[batch_axis]
    dim = tf.shape(x)[axis]
    amount = tf.random_uniform(shape=(n_batch,), minval=1, maxval=max_amount + 1, dtype=tf.int32)
    pos2 = tf.minimum(pos + amount, dim)
    idxs = tf.expand_dims(tf.range(0, dim), 0)  # (1,dim)
    pos_bc = tf.expand_dims(pos, 1)  # (batch,1)
    pos2_bc = tf.expand_dims(pos2, 1)  # (batch,1)
    cond = tf.logical_and(tf.greater_equal(idxs, pos_bc), tf.less(idxs, pos2_bc))  # (batch,dim)
    if batch_axis > axis:
        cond = tf.transpose(cond)  # (dim,batch)
    cond = tf.reshape(cond, [tf.shape(x)[i] if i in (batch_axis, axis) else 1 for i in range(ndim)])
    from returnn.tf.util.basic import where_bc

    x = where_bc(cond, 0.0, x)
    return x


def random_mask(x, batch_axis, axis, min_num, max_num, max_dims):
    """
    :param tf.Tensor x: (batch,time,feature)
    :param int batch_axis:
    :param int axis:
    :param int|tf.Tensor min_num:
    :param int|tf.Tensor max_num: inclusive
    :param int|tf.Tensor max_dims: inclusive
    """
    from returnn.tf.compat import v1 as tf

    n_batch = tf.shape(x)[batch_axis]
    if isinstance(min_num, int) and isinstance(max_num, int) and min_num == max_num:
        num = min_num
    else:
        num = tf.random_uniform(shape=(n_batch,), minval=min_num, maxval=max_num + 1, dtype=tf.int32)
    # https://github.com/tensorflow/tensorflow/issues/9260
    # https://timvieira.github.io/blog/post/2014/08/01/gumbel-max-trick-and-weighted-reservoir-sampling/
    z = -tf.log(-tf.log(tf.random_uniform((n_batch, tf.shape(x)[axis]), 0, 1)))
    _, indices = tf.nn.top_k(z, num if isinstance(num, int) else tf.reduce_max(num))
    # indices should be sorted, and of shape (batch,num), entries (int32) in [0,dim)
    # indices = tf.Print(indices, ["indices", indices, tf.shape(indices)])
    if isinstance(num, int):
        for i in range(num):
            x = _mask(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims)
    else:
        _, x = tf.while_loop(
            cond=lambda i, _: tf.less(i, tf.reduce_max(num)),
            body=lambda i, x: (
                i + 1,
                tf.where(
                    tf.less(i, num),
                    _mask(x, batch_axis=batch_axis, axis=axis, pos=indices[:, i], max_amount=max_dims),
                    x,
                ),
            ),
            loop_vars=(0, x),
        )
    return x


def transform(source, self, **_kwargs):
    """specaugment"""
    data = source(0, as_data=True)
    network = self.network
    time_factor = 1
    x = data.placeholder
    from returnn.tf.compat import v1 as tf

    # summary("features", x)
    step = network.global_train_step
    step1 = tf.where(tf.greater_equal(step, 1000), 1, 0)
    step2 = tf.where(tf.greater_equal(step, 2000), 1, 0)

    def get_masked():
        """masked"""
        x_masked = x
        x_masked = random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.time_dim_axis,
            min_num=step1 + step2,
            max_num=tf.maximum(tf.shape(x)[data.time_dim_axis] // 100, 2) * (1 + step1 + step2 * 2),
            max_dims=20 // time_factor,
        )
        x_masked = random_mask(
            x_masked,
            batch_axis=data.batch_dim_axis,
            axis=data.feature_dim_axis,
            min_num=step1 + step2,
            max_num=2 + step1 + step2 * 2,
            max_dims=data.dim // 5,
        )
        # summary("features_mask", x_masked)
        return x_masked

    x = network.cond_on_train(get_masked, lambda: x)
    return x
