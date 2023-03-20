"""
Starting point, 2022-10-12
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Sequence
import contextlib
import numpy
from returnn_common import nn
from returnn_common.nn.encoder.blstm import BlstmEncoder
from returnn_common.asr.specaugment import specaugment_v2

from i6_experiments.users.zeyer.datasets.switchboard_2020.task import get_switchboard_task_bpe1k
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
from i6_experiments.users.zeyer.recog import recog_training_exp
from ..train import train


# Finished by sis-setup-utils/cleanup-training.py.
# Best results: {'best_scores': {'hub5e_00': 19.9, 'hub5e_01': 16.7, 'rt03s': 20.0}, 'best_epoch': 240}
_exclude_me = True


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
        horovod_num_processes=4,
        multi_node_slots=4,
        num_epochs=300,
        model_def=from_scratch_model_def,
        train_def=from_scratch_training,
    )
    recog_training_exp(prefix_name, task, model, recog_def=model_recog)


config = dict(
    horovod_dataset_distribution="random_seed_offset",
    horovod_reduce_type="param",
    horovod_param_sync_time_diff=100.0,
    batching="random",
    batch_size=10000,
    max_seqs=200,
    max_seq_length_default_target=75,
    accum_grad_multiple_step=2,
    # gradient_clip=0,
    # gradient_clip_global_norm = 1.0
    optimizer={"class": "nadam", "epsilon": 1e-8},
    # gradient_noise=0.0,
    learning_rate=0.0008,
    learning_rates=list(float(lr) for lr in numpy.linspace(1e-07, 0.0008, 60)),
    min_learning_rate=0.0008 / 50,
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


class Model(nn.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: nn.Dim,
        *,
        num_enc_layers: int = 12,
        nb_target_dim: nn.Dim,
        wb_target_dim: nn.Dim,
        blank_idx: int,
        bos_idx: int,
        enc_model_dim: nn.Dim = nn.FeatureDim("enc", 512),
        enc_ff_dim: nn.Dim = nn.FeatureDim("enc-ff", 2048),
        enc_att_num_heads: int = 4,
        enc_key_total_dim: nn.Dim = nn.FeatureDim("enc_key_total_dim", 200),
        att_num_heads: nn.Dim = nn.SpatialDim("att_num_heads", 1),
        att_dropout: float = 0.1,
        enc_dropout: float = 0.1,
        enc_att_dropout: float = 0.1,
        l2: float = 0.0001,
    ):
        super(Model, self).__init__()
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
                num_layers=2,
                time_reduction=6,
                dropout=enc_dropout,
            ),
            num_layers=num_enc_layers,
            num_heads=enc_att_num_heads,
            dropout=enc_dropout,
            att_dropout=enc_att_dropout,
        )

        self.nb_target_dim = nb_target_dim
        self.wb_target_dim = wb_target_dim
        self.blank_idx = blank_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        self.att_num_heads = att_num_heads
        self.att_dropout = att_dropout

        self.enc_ctx = nn.Linear(self.encoder.out_dim, enc_key_total_dim)
        self.enc_ctx_dropout = 0.2
        self.enc_win_dim = nn.SpatialDim("enc_win_dim", 5)
        self.att_query = nn.Linear(self.encoder.out_dim, enc_key_total_dim, with_bias=False)
        self.lm = DecoderLabelSync(nb_target_dim, l2=l2)
        self.readout_in_am = nn.Linear(2 * self.encoder.out_dim, nn.FeatureDim("readout", 1000), with_bias=False)
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
        source = specaugment_v2(source, spatial_dim=in_spatial_dim, feature_dim=self.in_dim)
        enc, enc_spatial_dim = self.encoder(source, in_spatial_dim=in_spatial_dim)
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
    epoch  # noqa
    return Model(
        in_dim,
        num_enc_layers=12,
        enc_model_dim=nn.FeatureDim("enc", 384),
        enc_ff_dim=nn.FeatureDim("enc-ff", 1536),
        enc_att_num_heads=6,
        nb_target_dim=target_dim,
        wb_target_dim=target_dim + 1,
        blank_idx=target_dim.dimension,
        bos_idx=_get_bos_idx(target_dim),
        enc_dropout=0.1,
        enc_att_dropout=0.1,
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
