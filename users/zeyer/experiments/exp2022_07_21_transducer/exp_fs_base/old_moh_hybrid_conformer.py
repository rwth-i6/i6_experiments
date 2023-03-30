"""
Via:
https://github.com/rwth-i6/returnn-experiments/blob/master/2022-swb-conformer-hybrid-sat/table_1_and_2/reduced_dim.config

Originally for hybrid models.
"""

from __future__ import annotations
from typing import Optional, Tuple, Dict, Sequence
import contextlib
from returnn_common import nn
from i6_experiments.common.setups.returnn_common import serialization

from i6_experiments.users.zeyer.datasets.switchboard_2020.task import get_switchboard_task_bpe1k
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
from i6_experiments.users.zeyer.recog import recog_training_exp
from ..train import train


# Finished by sis-setup-utils/cleanup-training.py.
# Best results: {'best_scores': {'hub5e_00': 27.7, 'hub5e_01': 23.0, 'rt03s': 27.3}, 'best_epoch': 149}
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
        model_def=from_scratch_model_def,
        train_def=from_scratch_training,
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
    learning_rate=0.0008,
    learning_rates=[
        1e-07,
        1.365762711864407e-05,
        2.7215254237288136e-05,
        4.077288135593221e-05,
        5.4330508474576277e-05,
        6.788813559322035e-05,
        8.144576271186441e-05,
        9.500338983050848e-05,
        0.00010856101694915255,
        0.00012211864406779663,
        0.0001356762711864407,
        0.00014923389830508475,
        0.00016279152542372882,
        0.0001763491525423729,
        0.00018990677966101695,
        0.00020346440677966102,
        0.0002170220338983051,
        0.00023057966101694917,
        0.00024413728813559325,
        0.0002576949152542373,
        0.0002712525423728814,
        0.00028481016949152545,
        0.0002983677966101695,
        0.0003119254237288136,
        0.00032548305084745765,
        0.0003390406779661017,
        0.0003525983050847458,
        0.00036615593220338985,
        0.0003797135593220339,
        0.000393271186440678,
        0.00040682881355932204,
        0.00042038644067796615,
        0.0004339440677966102,
        0.00044750169491525424,
        0.00046105932203389835,
        0.0004746169491525424,
        0.0004881745762711865,
        0.0005017322033898304,
        0.0005152898305084745,
        0.0005288474576271186,
        0.0005424050847457627,
        0.0005559627118644067,
        0.0005695203389830508,
        0.0005830779661016949,
        0.0005966355932203389,
        0.000610193220338983,
        0.0006237508474576271,
        0.0006373084745762711,
        0.0006508661016949152,
        0.0006644237288135593,
        0.0006779813559322033,
        0.0006915389830508474,
        0.0007050966101694915,
        0.0007186542372881355,
        0.0007322118644067796,
        0.0007457694915254237,
        0.0007593271186440677,
        0.0007728847457627118,
        0.0007864423728813559,
        0.0008,
    ],
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


encoder_out_dim = nn.FeatureDim("encoder", 384)


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
    ):
        super(Model, self).__init__()
        self.in_dim = in_dim

        self.nb_target_dim = nb_target_dim
        self.wb_target_dim = wb_target_dim
        self.blank_idx = blank_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        self.att_num_heads = att_num_heads
        self.att_dropout = att_dropout

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
            {"class": "subnetwork", "subnetwork": conformer_net_dict, "from": source},
            name=nn.NameCtx.top().root.get_child("encoder"),
        )
        enc_spatial_dim = enc.data.get_time_dim_tag()
        enc = nn.make_layer(
            {"class": "reinterpret_data", "set_dim_tags": {"F": encoder_out_dim}, "from": enc}, name="enc_set_out_dim"
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


conformer_net_dict = {
    "output": {"class": "copy", "from": "encoder"},
    "source": {"class": "eval", "eval": transform, "from": "data"},
    "source0": {"axis": "F", "class": "split_dims", "dims": (-1, 1), "from": "source"},
    "c1": {
        "class": "conv",
        "filter_size": (3, 3),
        "from": "source0",
        "in_spatial_dims": ("stag:time", "stag:audio"),
        "n_out": 32,
        "padding": "same",
        "with_bias": True,
    },
    "y1": {"activation": "swish", "batch_norm": False, "class": "activation", "from": "c1"},
    "p1": {
        "class": "pool",
        "from": "y1",
        "mode": "max",
        "padding": "same",
        "pool_size": (1, 2),
        "in_spatial_dims": ("stag:time", "stag:audio"),
    },
    "c3": {
        "class": "conv",
        "filter_size": (3, 3),
        "from": "p1",
        "n_out": 64,
        "padding": "same",
        "in_spatial_dims": ("stag:time", "stag:conv:s1"),
        "with_bias": True,
    },
    "y3": {"activation": "swish", "batch_norm": False, "class": "activation", "from": "c3"},
    "c4": {
        "class": "conv",
        "filter_size": (3, 3),
        "from": "y3",
        "n_out": 64,
        "padding": "same",
        "in_spatial_dims": ("stag:time", "stag:conv:s1"),
        "strides": (2, 1),
        "with_bias": True,
    },
    "y4": {"activation": "swish", "batch_norm": False, "class": "activation", "from": "c4"},
    "c2": {
        "class": "conv",
        "filter_size": (3, 3),
        "from": "y4",
        "n_out": 32,
        "padding": "same",
        "in_spatial_dims": ("stag:conv:s0", "stag:conv:s1"),
        "strides": (3, 1),
        "with_bias": True,
    },
    "y2": {"activation": "swish", "batch_norm": False, "class": "activation", "from": "c2"},
    "vgg_conv_merged": {"class": "merge_dims", "axes": ("stag:conv:s1", "F"), "from": "y2"},
    "source_linear": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "vgg_conv_merged",
        "out_dim": encoder_out_dim,
        "with_bias": False,
    },
    "source_linear_ln": {"class": "layer_norm", "from": "source_linear"},
    "source_dropout": {"class": "dropout", "dropout": 0.1, "from": "source_linear_ln"},
    "conformer_block_01": {"class": "copy", "from": "conformer_block_01_ln"},
    "conformer_block_01_covmod_1_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_01_covmod_1_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_01_covmod_1_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_01_covmod_1_conv_mod_pointwise_conv2",
    },
    "conformer_block_01_covmod_1_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_01_covmod_1_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_01_covmod_1_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_01_covmod_1_conv_mod_drop",
    },
    "conformer_block_01_covmod_1_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_01_ffmod_1_res"},
    "conformer_block_01_covmod_1_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_01_covmod_1_conv_mod_depthwise_conv2",
    },
    "conformer_block_01_covmod_1_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_01_covmod_1_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_01_covmod_1_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_01_covmod_1_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_01_covmod_1_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_01_covmod_1_conv_mod_half_step", "conformer_block_01_ffmod_1_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_01_covmod_1_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_01_covmod_1_conv_mod_ln_replace_bn",
    },
    "conformer_block_01_covmod_2_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_01_covmod_2_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_01_covmod_2_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_01_covmod_2_conv_mod_pointwise_conv2",
    },
    "conformer_block_01_covmod_2_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_01_covmod_2_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_01_covmod_2_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_01_covmod_2_conv_mod_drop",
    },
    "conformer_block_01_covmod_2_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_01_self_att_res"},
    "conformer_block_01_covmod_2_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_01_covmod_2_conv_mod_depthwise_conv2",
    },
    "conformer_block_01_covmod_2_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_01_covmod_2_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_01_covmod_2_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_01_covmod_2_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_01_covmod_2_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_01_covmod_2_conv_mod_half_step", "conformer_block_01_self_att_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_01_covmod_2_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_01_covmod_2_conv_mod_ln_replace_bn",
    },
    "conformer_block_01_ffmod_1_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_01_ffmod_1_swish",
    },
    "conformer_block_01_ffmod_1_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_01_ffmod_1_ff2"},
    "conformer_block_01_ffmod_1_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_01_ffmod_1_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_01_ffmod_1_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_01_ffmod_1_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_01_ffmod_1_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_01_ffmod_1_drop2",
    },
    "conformer_block_01_ffmod_1_ln": {"class": "layer_norm", "from": "source_dropout"},
    "conformer_block_01_ffmod_1_res": {
        "class": "combine",
        "from": ["conformer_block_01_ffmod_1_half_step", "source_dropout"],
        "kind": "add",
    },
    "conformer_block_01_ffmod_1_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_01_ffmod_1_ff1",
    },
    "conformer_block_01_ffmod_2_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_01_ffmod_2_swish",
    },
    "conformer_block_01_ffmod_2_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_01_ffmod_2_ff2"},
    "conformer_block_01_ffmod_2_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_01_ffmod_2_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_01_ffmod_2_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_01_ffmod_2_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_01_ffmod_2_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_01_ffmod_2_drop2",
    },
    "conformer_block_01_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_block_01_covmod_2_conv_mod_res"},
    "conformer_block_01_ffmod_2_res": {
        "class": "combine",
        "from": ["conformer_block_01_ffmod_2_half_step", "conformer_block_01_covmod_2_conv_mod_res"],
        "kind": "add",
    },
    "conformer_block_01_ffmod_2_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_01_ffmod_2_ff1",
    },
    "conformer_block_01_ln": {"class": "layer_norm", "from": "conformer_block_01_ffmod_2_res"},
    "conformer_block_01_self_att": {
        "attention_dropout": 0.1,
        "class": "self_attention",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_01_self_att_ln",
        "key_shift": "conformer_block_01_self_att_ln_rel_pos_enc",
        "out_dim": encoder_out_dim,
        "num_heads": 6,
        "total_key_dim": 384,
    },
    "conformer_block_01_self_att_dropout": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_01_self_att_linear",
    },
    "conformer_block_01_self_att_linear": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_01_self_att",
        "out_dim": encoder_out_dim,
        "with_bias": False,
    },
    "conformer_block_01_self_att_ln": {"class": "layer_norm", "from": "conformer_block_01_covmod_1_conv_mod_res"},
    "conformer_block_01_self_att_ln_rel_pos_enc": {
        "class": "relative_positional_encoding",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_01_self_att_ln",
        "n_out": 64,
    },
    "conformer_block_01_self_att_res": {
        "class": "combine",
        "from": ["conformer_block_01_self_att_dropout", "conformer_block_01_covmod_1_conv_mod_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_02": {"class": "copy", "from": "conformer_block_02_ln"},
    "conformer_block_02_covmod_1_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_02_covmod_1_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_02_covmod_1_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_02_covmod_1_conv_mod_pointwise_conv2",
    },
    "conformer_block_02_covmod_1_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_02_covmod_1_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_02_covmod_1_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_02_covmod_1_conv_mod_drop",
    },
    "conformer_block_02_covmod_1_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_02_ffmod_1_res"},
    "conformer_block_02_covmod_1_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_02_covmod_1_conv_mod_depthwise_conv2",
    },
    "conformer_block_02_covmod_1_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_02_covmod_1_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_02_covmod_1_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_02_covmod_1_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_02_covmod_1_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_02_covmod_1_conv_mod_half_step", "conformer_block_02_ffmod_1_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_02_covmod_1_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_02_covmod_1_conv_mod_ln_replace_bn",
    },
    "conformer_block_02_covmod_2_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_02_covmod_2_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_02_covmod_2_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_02_covmod_2_conv_mod_pointwise_conv2",
    },
    "conformer_block_02_covmod_2_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_02_covmod_2_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_02_covmod_2_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_02_covmod_2_conv_mod_drop",
    },
    "conformer_block_02_covmod_2_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_02_self_att_res"},
    "conformer_block_02_covmod_2_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_02_covmod_2_conv_mod_depthwise_conv2",
    },
    "conformer_block_02_covmod_2_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_02_covmod_2_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_02_covmod_2_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_02_covmod_2_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_02_covmod_2_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_02_covmod_2_conv_mod_half_step", "conformer_block_02_self_att_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_02_covmod_2_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_02_covmod_2_conv_mod_ln_replace_bn",
    },
    "conformer_block_02_ffmod_1_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_02_ffmod_1_swish",
    },
    "conformer_block_02_ffmod_1_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_02_ffmod_1_ff2"},
    "conformer_block_02_ffmod_1_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_02_ffmod_1_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_02_ffmod_1_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_02_ffmod_1_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_02_ffmod_1_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_02_ffmod_1_drop2",
    },
    "conformer_block_02_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_block_01"},
    "conformer_block_02_ffmod_1_res": {
        "class": "combine",
        "from": ["conformer_block_02_ffmod_1_half_step", "conformer_block_01"],
        "kind": "add",
    },
    "conformer_block_02_ffmod_1_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_02_ffmod_1_ff1",
    },
    "conformer_block_02_ffmod_2_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_02_ffmod_2_swish",
    },
    "conformer_block_02_ffmod_2_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_02_ffmod_2_ff2"},
    "conformer_block_02_ffmod_2_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_02_ffmod_2_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_02_ffmod_2_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_02_ffmod_2_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_02_ffmod_2_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_02_ffmod_2_drop2",
    },
    "conformer_block_02_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_block_02_covmod_2_conv_mod_res"},
    "conformer_block_02_ffmod_2_res": {
        "class": "combine",
        "from": ["conformer_block_02_ffmod_2_half_step", "conformer_block_02_covmod_2_conv_mod_res"],
        "kind": "add",
    },
    "conformer_block_02_ffmod_2_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_02_ffmod_2_ff1",
    },
    "conformer_block_02_ln": {"class": "layer_norm", "from": "conformer_block_02_ffmod_2_res"},
    "conformer_block_02_self_att": {
        "attention_dropout": 0.1,
        "class": "self_attention",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_02_self_att_ln",
        "key_shift": "conformer_block_02_self_att_ln_rel_pos_enc",
        "out_dim": encoder_out_dim,
        "num_heads": 6,
        "total_key_dim": 384,
    },
    "conformer_block_02_self_att_dropout": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_02_self_att_linear",
    },
    "conformer_block_02_self_att_linear": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_02_self_att",
        "out_dim": encoder_out_dim,
        "with_bias": False,
    },
    "conformer_block_02_self_att_ln": {"class": "layer_norm", "from": "conformer_block_02_covmod_1_conv_mod_res"},
    "conformer_block_02_self_att_ln_rel_pos_enc": {
        "class": "relative_positional_encoding",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_02_self_att_ln",
        "n_out": 64,
    },
    "conformer_block_02_self_att_res": {
        "class": "combine",
        "from": ["conformer_block_02_self_att_dropout", "conformer_block_02_covmod_1_conv_mod_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_03": {"class": "copy", "from": "conformer_block_03_ln"},
    "conformer_block_03_covmod_1_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_03_covmod_1_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_03_covmod_1_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_03_covmod_1_conv_mod_pointwise_conv2",
    },
    "conformer_block_03_covmod_1_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_03_covmod_1_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_03_covmod_1_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_03_covmod_1_conv_mod_drop",
    },
    "conformer_block_03_covmod_1_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_03_ffmod_1_res"},
    "conformer_block_03_covmod_1_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_03_covmod_1_conv_mod_depthwise_conv2",
    },
    "conformer_block_03_covmod_1_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_03_covmod_1_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_03_covmod_1_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_03_covmod_1_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_03_covmod_1_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_03_covmod_1_conv_mod_half_step", "conformer_block_03_ffmod_1_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_03_covmod_1_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_03_covmod_1_conv_mod_ln_replace_bn",
    },
    "conformer_block_03_covmod_2_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_03_covmod_2_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_03_covmod_2_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_03_covmod_2_conv_mod_pointwise_conv2",
    },
    "conformer_block_03_covmod_2_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_03_covmod_2_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_03_covmod_2_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_03_covmod_2_conv_mod_drop",
    },
    "conformer_block_03_covmod_2_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_03_self_att_res"},
    "conformer_block_03_covmod_2_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_03_covmod_2_conv_mod_depthwise_conv2",
    },
    "conformer_block_03_covmod_2_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_03_covmod_2_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_03_covmod_2_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_03_covmod_2_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_03_covmod_2_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_03_covmod_2_conv_mod_half_step", "conformer_block_03_self_att_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_03_covmod_2_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_03_covmod_2_conv_mod_ln_replace_bn",
    },
    "conformer_block_03_ffmod_1_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_03_ffmod_1_swish",
    },
    "conformer_block_03_ffmod_1_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_03_ffmod_1_ff2"},
    "conformer_block_03_ffmod_1_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_03_ffmod_1_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_03_ffmod_1_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_03_ffmod_1_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_03_ffmod_1_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_03_ffmod_1_drop2",
    },
    "conformer_block_03_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_block_02"},
    "conformer_block_03_ffmod_1_res": {
        "class": "combine",
        "from": ["conformer_block_03_ffmod_1_half_step", "conformer_block_02"],
        "kind": "add",
    },
    "conformer_block_03_ffmod_1_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_03_ffmod_1_ff1",
    },
    "conformer_block_03_ffmod_2_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_03_ffmod_2_swish",
    },
    "conformer_block_03_ffmod_2_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_03_ffmod_2_ff2"},
    "conformer_block_03_ffmod_2_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_03_ffmod_2_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_03_ffmod_2_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_03_ffmod_2_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_03_ffmod_2_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_03_ffmod_2_drop2",
    },
    "conformer_block_03_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_block_03_covmod_2_conv_mod_res"},
    "conformer_block_03_ffmod_2_res": {
        "class": "combine",
        "from": ["conformer_block_03_ffmod_2_half_step", "conformer_block_03_covmod_2_conv_mod_res"],
        "kind": "add",
    },
    "conformer_block_03_ffmod_2_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_03_ffmod_2_ff1",
    },
    "conformer_block_03_ln": {"class": "layer_norm", "from": "conformer_block_03_ffmod_2_res"},
    "conformer_block_03_self_att": {
        "attention_dropout": 0.1,
        "class": "self_attention",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_03_self_att_ln",
        "key_shift": "conformer_block_03_self_att_ln_rel_pos_enc",
        "out_dim": encoder_out_dim,
        "num_heads": 6,
        "total_key_dim": 384,
    },
    "conformer_block_03_self_att_dropout": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_03_self_att_linear",
    },
    "conformer_block_03_self_att_linear": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_03_self_att",
        "out_dim": encoder_out_dim,
        "with_bias": False,
    },
    "conformer_block_03_self_att_ln": {"class": "layer_norm", "from": "conformer_block_03_covmod_1_conv_mod_res"},
    "conformer_block_03_self_att_ln_rel_pos_enc": {
        "class": "relative_positional_encoding",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_03_self_att_ln",
        "n_out": 64,
    },
    "conformer_block_03_self_att_res": {
        "class": "combine",
        "from": ["conformer_block_03_self_att_dropout", "conformer_block_03_covmod_1_conv_mod_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_04": {"class": "copy", "from": "conformer_block_04_ln"},
    "conformer_block_04_covmod_1_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_04_covmod_1_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_04_covmod_1_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_04_covmod_1_conv_mod_pointwise_conv2",
    },
    "conformer_block_04_covmod_1_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_04_covmod_1_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_04_covmod_1_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_04_covmod_1_conv_mod_drop",
    },
    "conformer_block_04_covmod_1_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_04_ffmod_1_res"},
    "conformer_block_04_covmod_1_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_04_covmod_1_conv_mod_depthwise_conv2",
    },
    "conformer_block_04_covmod_1_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_04_covmod_1_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_04_covmod_1_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_04_covmod_1_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_04_covmod_1_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_04_covmod_1_conv_mod_half_step", "conformer_block_04_ffmod_1_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_04_covmod_1_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_04_covmod_1_conv_mod_ln_replace_bn",
    },
    "conformer_block_04_covmod_2_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_04_covmod_2_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_04_covmod_2_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_04_covmod_2_conv_mod_pointwise_conv2",
    },
    "conformer_block_04_covmod_2_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_04_covmod_2_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_04_covmod_2_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_04_covmod_2_conv_mod_drop",
    },
    "conformer_block_04_covmod_2_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_04_self_att_res"},
    "conformer_block_04_covmod_2_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_04_covmod_2_conv_mod_depthwise_conv2",
    },
    "conformer_block_04_covmod_2_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_04_covmod_2_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_04_covmod_2_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_04_covmod_2_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_04_covmod_2_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_04_covmod_2_conv_mod_half_step", "conformer_block_04_self_att_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_04_covmod_2_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_04_covmod_2_conv_mod_ln_replace_bn",
    },
    "conformer_block_04_ffmod_1_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_04_ffmod_1_swish",
    },
    "conformer_block_04_ffmod_1_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_04_ffmod_1_ff2"},
    "conformer_block_04_ffmod_1_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_04_ffmod_1_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_04_ffmod_1_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_04_ffmod_1_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_04_ffmod_1_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_04_ffmod_1_drop2",
    },
    "conformer_block_04_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_block_03"},
    "conformer_block_04_ffmod_1_res": {
        "class": "combine",
        "from": ["conformer_block_04_ffmod_1_half_step", "conformer_block_03"],
        "kind": "add",
    },
    "conformer_block_04_ffmod_1_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_04_ffmod_1_ff1",
    },
    "conformer_block_04_ffmod_2_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_04_ffmod_2_swish",
    },
    "conformer_block_04_ffmod_2_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_04_ffmod_2_ff2"},
    "conformer_block_04_ffmod_2_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_04_ffmod_2_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_04_ffmod_2_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_04_ffmod_2_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_04_ffmod_2_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_04_ffmod_2_drop2",
    },
    "conformer_block_04_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_block_04_covmod_2_conv_mod_res"},
    "conformer_block_04_ffmod_2_res": {
        "class": "combine",
        "from": ["conformer_block_04_ffmod_2_half_step", "conformer_block_04_covmod_2_conv_mod_res"],
        "kind": "add",
    },
    "conformer_block_04_ffmod_2_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_04_ffmod_2_ff1",
    },
    "conformer_block_04_ln": {"class": "layer_norm", "from": "conformer_block_04_ffmod_2_res"},
    "conformer_block_04_self_att": {
        "attention_dropout": 0.1,
        "class": "self_attention",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_04_self_att_ln",
        "key_shift": "conformer_block_04_self_att_ln_rel_pos_enc",
        "out_dim": encoder_out_dim,
        "num_heads": 6,
        "total_key_dim": 384,
    },
    "conformer_block_04_self_att_dropout": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_04_self_att_linear",
    },
    "conformer_block_04_self_att_linear": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_04_self_att",
        "out_dim": encoder_out_dim,
        "with_bias": False,
    },
    "conformer_block_04_self_att_ln": {"class": "layer_norm", "from": "conformer_block_04_covmod_1_conv_mod_res"},
    "conformer_block_04_self_att_ln_rel_pos_enc": {
        "class": "relative_positional_encoding",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_04_self_att_ln",
        "n_out": 64,
    },
    "conformer_block_04_self_att_res": {
        "class": "combine",
        "from": ["conformer_block_04_self_att_dropout", "conformer_block_04_covmod_1_conv_mod_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_05": {"class": "copy", "from": "conformer_block_05_ln"},
    "conformer_block_05_covmod_1_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_05_covmod_1_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_05_covmod_1_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_05_covmod_1_conv_mod_pointwise_conv2",
    },
    "conformer_block_05_covmod_1_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_05_covmod_1_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_05_covmod_1_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_05_covmod_1_conv_mod_drop",
    },
    "conformer_block_05_covmod_1_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_05_ffmod_1_res"},
    "conformer_block_05_covmod_1_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_05_covmod_1_conv_mod_depthwise_conv2",
    },
    "conformer_block_05_covmod_1_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_05_covmod_1_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_05_covmod_1_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_05_covmod_1_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_05_covmod_1_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_05_covmod_1_conv_mod_half_step", "conformer_block_05_ffmod_1_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_05_covmod_1_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_05_covmod_1_conv_mod_ln_replace_bn",
    },
    "conformer_block_05_covmod_2_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_05_covmod_2_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_05_covmod_2_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_05_covmod_2_conv_mod_pointwise_conv2",
    },
    "conformer_block_05_covmod_2_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_05_covmod_2_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_05_covmod_2_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_05_covmod_2_conv_mod_drop",
    },
    "conformer_block_05_covmod_2_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_05_self_att_res"},
    "conformer_block_05_covmod_2_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_05_covmod_2_conv_mod_depthwise_conv2",
    },
    "conformer_block_05_covmod_2_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_05_covmod_2_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_05_covmod_2_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_05_covmod_2_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_05_covmod_2_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_05_covmod_2_conv_mod_half_step", "conformer_block_05_self_att_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_05_covmod_2_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_05_covmod_2_conv_mod_ln_replace_bn",
    },
    "conformer_block_05_ffmod_1_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_05_ffmod_1_swish",
    },
    "conformer_block_05_ffmod_1_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_05_ffmod_1_ff2"},
    "conformer_block_05_ffmod_1_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_05_ffmod_1_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_05_ffmod_1_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_05_ffmod_1_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_05_ffmod_1_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_05_ffmod_1_drop2",
    },
    "conformer_block_05_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_block_04"},
    "conformer_block_05_ffmod_1_res": {
        "class": "combine",
        "from": ["conformer_block_05_ffmod_1_half_step", "conformer_block_04"],
        "kind": "add",
    },
    "conformer_block_05_ffmod_1_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_05_ffmod_1_ff1",
    },
    "conformer_block_05_ffmod_2_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_05_ffmod_2_swish",
    },
    "conformer_block_05_ffmod_2_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_05_ffmod_2_ff2"},
    "conformer_block_05_ffmod_2_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_05_ffmod_2_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_05_ffmod_2_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_05_ffmod_2_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_05_ffmod_2_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_05_ffmod_2_drop2",
    },
    "conformer_block_05_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_block_05_covmod_2_conv_mod_res"},
    "conformer_block_05_ffmod_2_res": {
        "class": "combine",
        "from": ["conformer_block_05_ffmod_2_half_step", "conformer_block_05_covmod_2_conv_mod_res"],
        "kind": "add",
    },
    "conformer_block_05_ffmod_2_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_05_ffmod_2_ff1",
    },
    "conformer_block_05_ln": {"class": "layer_norm", "from": "conformer_block_05_ffmod_2_res"},
    "conformer_block_05_self_att": {
        "attention_dropout": 0.1,
        "class": "self_attention",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_05_self_att_ln",
        "key_shift": "conformer_block_05_self_att_ln_rel_pos_enc",
        "out_dim": encoder_out_dim,
        "num_heads": 6,
        "total_key_dim": 384,
    },
    "conformer_block_05_self_att_dropout": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_05_self_att_linear",
    },
    "conformer_block_05_self_att_linear": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_05_self_att",
        "out_dim": encoder_out_dim,
        "with_bias": False,
    },
    "conformer_block_05_self_att_ln": {"class": "layer_norm", "from": "conformer_block_05_covmod_1_conv_mod_res"},
    "conformer_block_05_self_att_ln_rel_pos_enc": {
        "class": "relative_positional_encoding",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_05_self_att_ln",
        "n_out": 64,
    },
    "conformer_block_05_self_att_res": {
        "class": "combine",
        "from": ["conformer_block_05_self_att_dropout", "conformer_block_05_covmod_1_conv_mod_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_06": {"class": "copy", "from": "conformer_block_06_ln"},
    "conformer_block_06_covmod_1_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_06_covmod_1_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_06_covmod_1_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_06_covmod_1_conv_mod_pointwise_conv2",
    },
    "conformer_block_06_covmod_1_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_06_covmod_1_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_06_covmod_1_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_06_covmod_1_conv_mod_drop",
    },
    "conformer_block_06_covmod_1_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_06_ffmod_1_res"},
    "conformer_block_06_covmod_1_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_06_covmod_1_conv_mod_depthwise_conv2",
    },
    "conformer_block_06_covmod_1_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_06_covmod_1_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_06_covmod_1_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_06_covmod_1_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_06_covmod_1_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_06_covmod_1_conv_mod_half_step", "conformer_block_06_ffmod_1_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_06_covmod_1_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_06_covmod_1_conv_mod_ln_replace_bn",
    },
    "conformer_block_06_covmod_2_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_06_covmod_2_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_06_covmod_2_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_06_covmod_2_conv_mod_pointwise_conv2",
    },
    "conformer_block_06_covmod_2_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_06_covmod_2_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_06_covmod_2_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_06_covmod_2_conv_mod_drop",
    },
    "conformer_block_06_covmod_2_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_06_self_att_res"},
    "conformer_block_06_covmod_2_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_06_covmod_2_conv_mod_depthwise_conv2",
    },
    "conformer_block_06_covmod_2_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_06_covmod_2_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_06_covmod_2_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_06_covmod_2_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_06_covmod_2_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_06_covmod_2_conv_mod_half_step", "conformer_block_06_self_att_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_06_covmod_2_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_06_covmod_2_conv_mod_ln_replace_bn",
    },
    "conformer_block_06_ffmod_1_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_06_ffmod_1_swish",
    },
    "conformer_block_06_ffmod_1_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_06_ffmod_1_ff2"},
    "conformer_block_06_ffmod_1_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_06_ffmod_1_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_06_ffmod_1_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_06_ffmod_1_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_06_ffmod_1_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_06_ffmod_1_drop2",
    },
    "conformer_block_06_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_block_05"},
    "conformer_block_06_ffmod_1_res": {
        "class": "combine",
        "from": ["conformer_block_06_ffmod_1_half_step", "conformer_block_05"],
        "kind": "add",
    },
    "conformer_block_06_ffmod_1_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_06_ffmod_1_ff1",
    },
    "conformer_block_06_ffmod_2_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_06_ffmod_2_swish",
    },
    "conformer_block_06_ffmod_2_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_06_ffmod_2_ff2"},
    "conformer_block_06_ffmod_2_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_06_ffmod_2_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_06_ffmod_2_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_06_ffmod_2_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_06_ffmod_2_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_06_ffmod_2_drop2",
    },
    "conformer_block_06_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_block_06_covmod_2_conv_mod_res"},
    "conformer_block_06_ffmod_2_res": {
        "class": "combine",
        "from": ["conformer_block_06_ffmod_2_half_step", "conformer_block_06_covmod_2_conv_mod_res"],
        "kind": "add",
    },
    "conformer_block_06_ffmod_2_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_06_ffmod_2_ff1",
    },
    "conformer_block_06_ln": {"class": "layer_norm", "from": "conformer_block_06_ffmod_2_res"},
    "conformer_block_06_self_att": {
        "attention_dropout": 0.1,
        "class": "self_attention",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_06_self_att_ln",
        "key_shift": "conformer_block_06_self_att_ln_rel_pos_enc",
        "out_dim": encoder_out_dim,
        "num_heads": 6,
        "total_key_dim": 384,
    },
    "conformer_block_06_self_att_dropout": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_06_self_att_linear",
    },
    "conformer_block_06_self_att_linear": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_06_self_att",
        "out_dim": encoder_out_dim,
        "with_bias": False,
    },
    "conformer_block_06_self_att_ln": {"class": "layer_norm", "from": "conformer_block_06_covmod_1_conv_mod_res"},
    "conformer_block_06_self_att_ln_rel_pos_enc": {
        "class": "relative_positional_encoding",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_06_self_att_ln",
        "n_out": 64,
    },
    "conformer_block_06_self_att_res": {
        "class": "combine",
        "from": ["conformer_block_06_self_att_dropout", "conformer_block_06_covmod_1_conv_mod_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_07": {"class": "copy", "from": "conformer_block_07_ln"},
    "conformer_block_07_covmod_1_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_07_covmod_1_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_07_covmod_1_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_07_covmod_1_conv_mod_pointwise_conv2",
    },
    "conformer_block_07_covmod_1_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_07_covmod_1_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_07_covmod_1_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_07_covmod_1_conv_mod_drop",
    },
    "conformer_block_07_covmod_1_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_07_ffmod_1_res"},
    "conformer_block_07_covmod_1_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_07_covmod_1_conv_mod_depthwise_conv2",
    },
    "conformer_block_07_covmod_1_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_07_covmod_1_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_07_covmod_1_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_07_covmod_1_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_07_covmod_1_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_07_covmod_1_conv_mod_half_step", "conformer_block_07_ffmod_1_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_07_covmod_1_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_07_covmod_1_conv_mod_ln_replace_bn",
    },
    "conformer_block_07_covmod_2_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_07_covmod_2_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_07_covmod_2_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_07_covmod_2_conv_mod_pointwise_conv2",
    },
    "conformer_block_07_covmod_2_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_07_covmod_2_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_07_covmod_2_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_07_covmod_2_conv_mod_drop",
    },
    "conformer_block_07_covmod_2_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_07_self_att_res"},
    "conformer_block_07_covmod_2_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_07_covmod_2_conv_mod_depthwise_conv2",
    },
    "conformer_block_07_covmod_2_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_07_covmod_2_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_07_covmod_2_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_07_covmod_2_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_07_covmod_2_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_07_covmod_2_conv_mod_half_step", "conformer_block_07_self_att_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_07_covmod_2_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_07_covmod_2_conv_mod_ln_replace_bn",
    },
    "conformer_block_07_ffmod_1_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_07_ffmod_1_swish",
    },
    "conformer_block_07_ffmod_1_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_07_ffmod_1_ff2"},
    "conformer_block_07_ffmod_1_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_07_ffmod_1_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_07_ffmod_1_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_07_ffmod_1_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_07_ffmod_1_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_07_ffmod_1_drop2",
    },
    "conformer_block_07_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_block_06"},
    "conformer_block_07_ffmod_1_res": {
        "class": "combine",
        "from": ["conformer_block_07_ffmod_1_half_step", "conformer_block_06"],
        "kind": "add",
    },
    "conformer_block_07_ffmod_1_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_07_ffmod_1_ff1",
    },
    "conformer_block_07_ffmod_2_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_07_ffmod_2_swish",
    },
    "conformer_block_07_ffmod_2_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_07_ffmod_2_ff2"},
    "conformer_block_07_ffmod_2_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_07_ffmod_2_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_07_ffmod_2_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_07_ffmod_2_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_07_ffmod_2_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_07_ffmod_2_drop2",
    },
    "conformer_block_07_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_block_07_covmod_2_conv_mod_res"},
    "conformer_block_07_ffmod_2_res": {
        "class": "combine",
        "from": ["conformer_block_07_ffmod_2_half_step", "conformer_block_07_covmod_2_conv_mod_res"],
        "kind": "add",
    },
    "conformer_block_07_ffmod_2_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_07_ffmod_2_ff1",
    },
    "conformer_block_07_ln": {"class": "layer_norm", "from": "conformer_block_07_ffmod_2_res"},
    "conformer_block_07_self_att": {
        "attention_dropout": 0.1,
        "class": "self_attention",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_07_self_att_ln",
        "key_shift": "conformer_block_07_self_att_ln_rel_pos_enc",
        "out_dim": encoder_out_dim,
        "num_heads": 6,
        "total_key_dim": 384,
    },
    "conformer_block_07_self_att_dropout": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_07_self_att_linear",
    },
    "conformer_block_07_self_att_linear": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_07_self_att",
        "out_dim": encoder_out_dim,
        "with_bias": False,
    },
    "conformer_block_07_self_att_ln": {"class": "layer_norm", "from": "conformer_block_07_covmod_1_conv_mod_res"},
    "conformer_block_07_self_att_ln_rel_pos_enc": {
        "class": "relative_positional_encoding",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_07_self_att_ln",
        "n_out": 64,
    },
    "conformer_block_07_self_att_res": {
        "class": "combine",
        "from": ["conformer_block_07_self_att_dropout", "conformer_block_07_covmod_1_conv_mod_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_08": {"class": "copy", "from": "conformer_block_08_ln"},
    "conformer_block_08_covmod_1_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_08_covmod_1_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_08_covmod_1_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_08_covmod_1_conv_mod_pointwise_conv2",
    },
    "conformer_block_08_covmod_1_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_08_covmod_1_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_08_covmod_1_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_08_covmod_1_conv_mod_drop",
    },
    "conformer_block_08_covmod_1_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_08_ffmod_1_res"},
    "conformer_block_08_covmod_1_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_08_covmod_1_conv_mod_depthwise_conv2",
    },
    "conformer_block_08_covmod_1_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_08_covmod_1_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_08_covmod_1_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_08_covmod_1_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_08_covmod_1_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_08_covmod_1_conv_mod_half_step", "conformer_block_08_ffmod_1_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_08_covmod_1_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_08_covmod_1_conv_mod_ln_replace_bn",
    },
    "conformer_block_08_covmod_2_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_08_covmod_2_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_08_covmod_2_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_08_covmod_2_conv_mod_pointwise_conv2",
    },
    "conformer_block_08_covmod_2_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_08_covmod_2_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_08_covmod_2_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_08_covmod_2_conv_mod_drop",
    },
    "conformer_block_08_covmod_2_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_08_self_att_res"},
    "conformer_block_08_covmod_2_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_08_covmod_2_conv_mod_depthwise_conv2",
    },
    "conformer_block_08_covmod_2_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_08_covmod_2_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_08_covmod_2_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_08_covmod_2_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_08_covmod_2_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_08_covmod_2_conv_mod_half_step", "conformer_block_08_self_att_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_08_covmod_2_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_08_covmod_2_conv_mod_ln_replace_bn",
    },
    "conformer_block_08_ffmod_1_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_08_ffmod_1_swish",
    },
    "conformer_block_08_ffmod_1_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_08_ffmod_1_ff2"},
    "conformer_block_08_ffmod_1_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_08_ffmod_1_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_08_ffmod_1_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_08_ffmod_1_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_08_ffmod_1_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_08_ffmod_1_drop2",
    },
    "conformer_block_08_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_block_07"},
    "conformer_block_08_ffmod_1_res": {
        "class": "combine",
        "from": ["conformer_block_08_ffmod_1_half_step", "conformer_block_07"],
        "kind": "add",
    },
    "conformer_block_08_ffmod_1_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_08_ffmod_1_ff1",
    },
    "conformer_block_08_ffmod_2_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_08_ffmod_2_swish",
    },
    "conformer_block_08_ffmod_2_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_08_ffmod_2_ff2"},
    "conformer_block_08_ffmod_2_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_08_ffmod_2_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_08_ffmod_2_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_08_ffmod_2_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_08_ffmod_2_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_08_ffmod_2_drop2",
    },
    "conformer_block_08_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_block_08_covmod_2_conv_mod_res"},
    "conformer_block_08_ffmod_2_res": {
        "class": "combine",
        "from": ["conformer_block_08_ffmod_2_half_step", "conformer_block_08_covmod_2_conv_mod_res"],
        "kind": "add",
    },
    "conformer_block_08_ffmod_2_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_08_ffmod_2_ff1",
    },
    "conformer_block_08_ln": {"class": "layer_norm", "from": "conformer_block_08_ffmod_2_res"},
    "conformer_block_08_self_att": {
        "attention_dropout": 0.1,
        "class": "self_attention",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_08_self_att_ln",
        "key_shift": "conformer_block_08_self_att_ln_rel_pos_enc",
        "out_dim": encoder_out_dim,
        "num_heads": 6,
        "total_key_dim": 384,
    },
    "conformer_block_08_self_att_dropout": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_08_self_att_linear",
    },
    "conformer_block_08_self_att_linear": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_08_self_att",
        "out_dim": encoder_out_dim,
        "with_bias": False,
    },
    "conformer_block_08_self_att_ln": {"class": "layer_norm", "from": "conformer_block_08_covmod_1_conv_mod_res"},
    "conformer_block_08_self_att_ln_rel_pos_enc": {
        "class": "relative_positional_encoding",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_08_self_att_ln",
        "n_out": 64,
    },
    "conformer_block_08_self_att_res": {
        "class": "combine",
        "from": ["conformer_block_08_self_att_dropout", "conformer_block_08_covmod_1_conv_mod_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_09": {"class": "copy", "from": "conformer_block_09_ln"},
    "conformer_block_09_covmod_1_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_09_covmod_1_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_09_covmod_1_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_09_covmod_1_conv_mod_pointwise_conv2",
    },
    "conformer_block_09_covmod_1_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_09_covmod_1_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_09_covmod_1_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_09_covmod_1_conv_mod_drop",
    },
    "conformer_block_09_covmod_1_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_09_ffmod_1_res"},
    "conformer_block_09_covmod_1_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_09_covmod_1_conv_mod_depthwise_conv2",
    },
    "conformer_block_09_covmod_1_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_09_covmod_1_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_09_covmod_1_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_09_covmod_1_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_09_covmod_1_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_09_covmod_1_conv_mod_half_step", "conformer_block_09_ffmod_1_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_09_covmod_1_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_09_covmod_1_conv_mod_ln_replace_bn",
    },
    "conformer_block_09_covmod_2_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_09_covmod_2_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_09_covmod_2_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_09_covmod_2_conv_mod_pointwise_conv2",
    },
    "conformer_block_09_covmod_2_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_09_covmod_2_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_09_covmod_2_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_09_covmod_2_conv_mod_drop",
    },
    "conformer_block_09_covmod_2_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_09_self_att_res"},
    "conformer_block_09_covmod_2_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_09_covmod_2_conv_mod_depthwise_conv2",
    },
    "conformer_block_09_covmod_2_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_09_covmod_2_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_09_covmod_2_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_09_covmod_2_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_09_covmod_2_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_09_covmod_2_conv_mod_half_step", "conformer_block_09_self_att_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_09_covmod_2_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_09_covmod_2_conv_mod_ln_replace_bn",
    },
    "conformer_block_09_ffmod_1_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_09_ffmod_1_swish",
    },
    "conformer_block_09_ffmod_1_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_09_ffmod_1_ff2"},
    "conformer_block_09_ffmod_1_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_09_ffmod_1_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_09_ffmod_1_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_09_ffmod_1_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_09_ffmod_1_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_09_ffmod_1_drop2",
    },
    "conformer_block_09_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_block_08"},
    "conformer_block_09_ffmod_1_res": {
        "class": "combine",
        "from": ["conformer_block_09_ffmod_1_half_step", "conformer_block_08"],
        "kind": "add",
    },
    "conformer_block_09_ffmod_1_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_09_ffmod_1_ff1",
    },
    "conformer_block_09_ffmod_2_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_09_ffmod_2_swish",
    },
    "conformer_block_09_ffmod_2_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_09_ffmod_2_ff2"},
    "conformer_block_09_ffmod_2_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_09_ffmod_2_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_09_ffmod_2_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_09_ffmod_2_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_09_ffmod_2_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_09_ffmod_2_drop2",
    },
    "conformer_block_09_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_block_09_covmod_2_conv_mod_res"},
    "conformer_block_09_ffmod_2_res": {
        "class": "combine",
        "from": ["conformer_block_09_ffmod_2_half_step", "conformer_block_09_covmod_2_conv_mod_res"],
        "kind": "add",
    },
    "conformer_block_09_ffmod_2_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_09_ffmod_2_ff1",
    },
    "conformer_block_09_ln": {"class": "layer_norm", "from": "conformer_block_09_ffmod_2_res"},
    "conformer_block_09_self_att": {
        "attention_dropout": 0.1,
        "class": "self_attention",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_09_self_att_ln",
        "key_shift": "conformer_block_09_self_att_ln_rel_pos_enc",
        "out_dim": encoder_out_dim,
        "num_heads": 6,
        "total_key_dim": 384,
    },
    "conformer_block_09_self_att_dropout": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_09_self_att_linear",
    },
    "conformer_block_09_self_att_linear": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_09_self_att",
        "out_dim": encoder_out_dim,
        "with_bias": False,
    },
    "conformer_block_09_self_att_ln": {"class": "layer_norm", "from": "conformer_block_09_covmod_1_conv_mod_res"},
    "conformer_block_09_self_att_ln_rel_pos_enc": {
        "class": "relative_positional_encoding",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_09_self_att_ln",
        "n_out": 64,
    },
    "conformer_block_09_self_att_res": {
        "class": "combine",
        "from": ["conformer_block_09_self_att_dropout", "conformer_block_09_covmod_1_conv_mod_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_10": {"class": "copy", "from": "conformer_block_10_ln"},
    "conformer_block_10_covmod_1_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_10_covmod_1_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_10_covmod_1_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_10_covmod_1_conv_mod_pointwise_conv2",
    },
    "conformer_block_10_covmod_1_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_10_covmod_1_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_10_covmod_1_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_10_covmod_1_conv_mod_drop",
    },
    "conformer_block_10_covmod_1_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_10_ffmod_1_res"},
    "conformer_block_10_covmod_1_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_10_covmod_1_conv_mod_depthwise_conv2",
    },
    "conformer_block_10_covmod_1_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_10_covmod_1_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_10_covmod_1_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_10_covmod_1_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_10_covmod_1_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_10_covmod_1_conv_mod_half_step", "conformer_block_10_ffmod_1_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_10_covmod_1_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_10_covmod_1_conv_mod_ln_replace_bn",
    },
    "conformer_block_10_covmod_2_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_10_covmod_2_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_10_covmod_2_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_10_covmod_2_conv_mod_pointwise_conv2",
    },
    "conformer_block_10_covmod_2_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_10_covmod_2_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_10_covmod_2_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_10_covmod_2_conv_mod_drop",
    },
    "conformer_block_10_covmod_2_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_10_self_att_res"},
    "conformer_block_10_covmod_2_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_10_covmod_2_conv_mod_depthwise_conv2",
    },
    "conformer_block_10_covmod_2_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_10_covmod_2_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_10_covmod_2_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_10_covmod_2_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_10_covmod_2_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_10_covmod_2_conv_mod_half_step", "conformer_block_10_self_att_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_10_covmod_2_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_10_covmod_2_conv_mod_ln_replace_bn",
    },
    "conformer_block_10_ffmod_1_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_10_ffmod_1_swish",
    },
    "conformer_block_10_ffmod_1_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_10_ffmod_1_ff2"},
    "conformer_block_10_ffmod_1_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_10_ffmod_1_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_10_ffmod_1_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_10_ffmod_1_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_10_ffmod_1_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_10_ffmod_1_drop2",
    },
    "conformer_block_10_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_block_09"},
    "conformer_block_10_ffmod_1_res": {
        "class": "combine",
        "from": ["conformer_block_10_ffmod_1_half_step", "conformer_block_09"],
        "kind": "add",
    },
    "conformer_block_10_ffmod_1_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_10_ffmod_1_ff1",
    },
    "conformer_block_10_ffmod_2_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_10_ffmod_2_swish",
    },
    "conformer_block_10_ffmod_2_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_10_ffmod_2_ff2"},
    "conformer_block_10_ffmod_2_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_10_ffmod_2_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_10_ffmod_2_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_10_ffmod_2_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_10_ffmod_2_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_10_ffmod_2_drop2",
    },
    "conformer_block_10_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_block_10_covmod_2_conv_mod_res"},
    "conformer_block_10_ffmod_2_res": {
        "class": "combine",
        "from": ["conformer_block_10_ffmod_2_half_step", "conformer_block_10_covmod_2_conv_mod_res"],
        "kind": "add",
    },
    "conformer_block_10_ffmod_2_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_10_ffmod_2_ff1",
    },
    "conformer_block_10_ln": {"class": "layer_norm", "from": "conformer_block_10_ffmod_2_res"},
    "conformer_block_10_self_att": {
        "attention_dropout": 0.1,
        "class": "self_attention",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_10_self_att_ln",
        "key_shift": "conformer_block_10_self_att_ln_rel_pos_enc",
        "out_dim": encoder_out_dim,
        "num_heads": 6,
        "total_key_dim": 384,
    },
    "conformer_block_10_self_att_dropout": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_10_self_att_linear",
    },
    "conformer_block_10_self_att_linear": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_10_self_att",
        "out_dim": encoder_out_dim,
        "with_bias": False,
    },
    "conformer_block_10_self_att_ln": {"class": "layer_norm", "from": "conformer_block_10_covmod_1_conv_mod_res"},
    "conformer_block_10_self_att_ln_rel_pos_enc": {
        "class": "relative_positional_encoding",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_10_self_att_ln",
        "n_out": 64,
    },
    "conformer_block_10_self_att_res": {
        "class": "combine",
        "from": ["conformer_block_10_self_att_dropout", "conformer_block_10_covmod_1_conv_mod_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_11": {"class": "copy", "from": "conformer_block_11_ln"},
    "conformer_block_11_covmod_1_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_11_covmod_1_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_11_covmod_1_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_11_covmod_1_conv_mod_pointwise_conv2",
    },
    "conformer_block_11_covmod_1_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_11_covmod_1_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_11_covmod_1_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_11_covmod_1_conv_mod_drop",
    },
    "conformer_block_11_covmod_1_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_11_ffmod_1_res"},
    "conformer_block_11_covmod_1_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_11_covmod_1_conv_mod_depthwise_conv2",
    },
    "conformer_block_11_covmod_1_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_11_covmod_1_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_11_covmod_1_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_11_covmod_1_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_11_covmod_1_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_11_covmod_1_conv_mod_half_step", "conformer_block_11_ffmod_1_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_11_covmod_1_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_11_covmod_1_conv_mod_ln_replace_bn",
    },
    "conformer_block_11_covmod_2_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_11_covmod_2_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_11_covmod_2_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_11_covmod_2_conv_mod_pointwise_conv2",
    },
    "conformer_block_11_covmod_2_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_11_covmod_2_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_11_covmod_2_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_11_covmod_2_conv_mod_drop",
    },
    "conformer_block_11_covmod_2_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_11_self_att_res"},
    "conformer_block_11_covmod_2_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_11_covmod_2_conv_mod_depthwise_conv2",
    },
    "conformer_block_11_covmod_2_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_11_covmod_2_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_11_covmod_2_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_11_covmod_2_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_11_covmod_2_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_11_covmod_2_conv_mod_half_step", "conformer_block_11_self_att_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_11_covmod_2_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_11_covmod_2_conv_mod_ln_replace_bn",
    },
    "conformer_block_11_ffmod_1_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_11_ffmod_1_swish",
    },
    "conformer_block_11_ffmod_1_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_11_ffmod_1_ff2"},
    "conformer_block_11_ffmod_1_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_11_ffmod_1_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_11_ffmod_1_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_11_ffmod_1_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_11_ffmod_1_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_11_ffmod_1_drop2",
    },
    "conformer_block_11_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_block_10"},
    "conformer_block_11_ffmod_1_res": {
        "class": "combine",
        "from": ["conformer_block_11_ffmod_1_half_step", "conformer_block_10"],
        "kind": "add",
    },
    "conformer_block_11_ffmod_1_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_11_ffmod_1_ff1",
    },
    "conformer_block_11_ffmod_2_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_11_ffmod_2_swish",
    },
    "conformer_block_11_ffmod_2_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_11_ffmod_2_ff2"},
    "conformer_block_11_ffmod_2_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_11_ffmod_2_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_11_ffmod_2_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_11_ffmod_2_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_11_ffmod_2_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_11_ffmod_2_drop2",
    },
    "conformer_block_11_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_block_11_covmod_2_conv_mod_res"},
    "conformer_block_11_ffmod_2_res": {
        "class": "combine",
        "from": ["conformer_block_11_ffmod_2_half_step", "conformer_block_11_covmod_2_conv_mod_res"],
        "kind": "add",
    },
    "conformer_block_11_ffmod_2_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_11_ffmod_2_ff1",
    },
    "conformer_block_11_ln": {"class": "layer_norm", "from": "conformer_block_11_ffmod_2_res"},
    "conformer_block_11_self_att": {
        "attention_dropout": 0.1,
        "class": "self_attention",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_11_self_att_ln",
        "key_shift": "conformer_block_11_self_att_ln_rel_pos_enc",
        "out_dim": encoder_out_dim,
        "num_heads": 6,
        "total_key_dim": 384,
    },
    "conformer_block_11_self_att_dropout": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_11_self_att_linear",
    },
    "conformer_block_11_self_att_linear": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_11_self_att",
        "out_dim": encoder_out_dim,
        "with_bias": False,
    },
    "conformer_block_11_self_att_ln": {"class": "layer_norm", "from": "conformer_block_11_covmod_1_conv_mod_res"},
    "conformer_block_11_self_att_ln_rel_pos_enc": {
        "class": "relative_positional_encoding",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_11_self_att_ln",
        "n_out": 64,
    },
    "conformer_block_11_self_att_res": {
        "class": "combine",
        "from": ["conformer_block_11_self_att_dropout", "conformer_block_11_covmod_1_conv_mod_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_12": {"class": "copy", "from": "conformer_block_12_ln"},
    "conformer_block_12_covmod_1_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_12_covmod_1_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_12_covmod_1_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_12_covmod_1_conv_mod_pointwise_conv2",
    },
    "conformer_block_12_covmod_1_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_12_covmod_1_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_12_covmod_1_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_12_covmod_1_conv_mod_drop",
    },
    "conformer_block_12_covmod_1_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_12_ffmod_1_res"},
    "conformer_block_12_covmod_1_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_12_covmod_1_conv_mod_depthwise_conv2",
    },
    "conformer_block_12_covmod_1_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_12_covmod_1_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_12_covmod_1_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_12_covmod_1_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_12_covmod_1_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_12_covmod_1_conv_mod_half_step", "conformer_block_12_ffmod_1_res"],
        "kind": "add",
        "out_dim": encoder_out_dim,
    },
    "conformer_block_12_covmod_1_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_12_covmod_1_conv_mod_ln_replace_bn",
    },
    "conformer_block_12_covmod_2_conv_mod_depthwise_conv2": {
        "activation": None,
        "class": "conv",
        "filter_size": (8,),
        "from": "conformer_block_12_covmod_2_conv_mod_glu",
        "groups": 384,
        "out_dim": encoder_out_dim,
        "padding": "same",
        "with_bias": True,
    },
    "conformer_block_12_covmod_2_conv_mod_drop": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_12_covmod_2_conv_mod_pointwise_conv2",
    },
    "conformer_block_12_covmod_2_conv_mod_glu": {
        "activation": "identity",
        "class": "gating",
        "from": "conformer_block_12_covmod_2_conv_mod_pointwise_conv1",
        "gate_activation": "sigmoid",
    },
    "conformer_block_12_covmod_2_conv_mod_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_12_covmod_2_conv_mod_drop",
    },
    "conformer_block_12_covmod_2_conv_mod_ln": {"class": "layer_norm", "from": "conformer_block_12_self_att_res"},
    "conformer_block_12_covmod_2_conv_mod_ln_replace_bn": {
        "class": "layer_norm",
        "from": "conformer_block_12_covmod_2_conv_mod_depthwise_conv2",
    },
    "conformer_block_12_covmod_2_conv_mod_pointwise_conv1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_12_covmod_2_conv_mod_ln",
        "n_out": 768,
        "with_bias": True,
    },
    "conformer_block_12_covmod_2_conv_mod_pointwise_conv2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_12_covmod_2_conv_mod_swish",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_12_covmod_2_conv_mod_res": {
        "class": "combine",
        "from": ["conformer_block_12_covmod_2_conv_mod_half_step", "conformer_block_12_self_att_res"],
        "kind": "add",
    },
    "conformer_block_12_covmod_2_conv_mod_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_12_covmod_2_conv_mod_ln_replace_bn",
    },
    "conformer_block_12_ffmod_1_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_12_ffmod_1_swish",
    },
    "conformer_block_12_ffmod_1_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_12_ffmod_1_ff2"},
    "conformer_block_12_ffmod_1_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_12_ffmod_1_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_12_ffmod_1_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_12_ffmod_1_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_12_ffmod_1_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_12_ffmod_1_drop2",
    },
    "conformer_block_12_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_block_11"},
    "conformer_block_12_ffmod_1_res": {
        "class": "combine",
        "from": ["conformer_block_12_ffmod_1_half_step", "conformer_block_11"],
        "kind": "add",
    },
    "conformer_block_12_ffmod_1_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_12_ffmod_1_ff1",
    },
    "conformer_block_12_ffmod_2_drop1": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_12_ffmod_2_swish",
    },
    "conformer_block_12_ffmod_2_drop2": {"class": "dropout", "dropout": 0.1, "from": "conformer_block_12_ffmod_2_ff2"},
    "conformer_block_12_ffmod_2_ff1": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_12_ffmod_2_ln",
        "n_out": 1536,
        "with_bias": True,
    },
    "conformer_block_12_ffmod_2_ff2": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_12_ffmod_2_drop1",
        "out_dim": encoder_out_dim,
        "with_bias": True,
    },
    "conformer_block_12_ffmod_2_half_step": {
        "class": "eval",
        "eval": "0.5 * source(0)",
        "from": "conformer_block_12_ffmod_2_drop2",
    },
    "conformer_block_12_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_block_12_covmod_2_conv_mod_res"},
    "conformer_block_12_ffmod_2_res": {
        "class": "combine",
        "from": ["conformer_block_12_ffmod_2_half_step", "conformer_block_12_covmod_2_conv_mod_res"],
        "kind": "add",
    },
    "conformer_block_12_ffmod_2_swish": {
        "activation": "swish",
        "class": "activation",
        "from": "conformer_block_12_ffmod_2_ff1",
    },
    "conformer_block_12_ln": {"class": "layer_norm", "from": "conformer_block_12_ffmod_2_res"},
    "conformer_block_12_self_att": {
        "attention_dropout": 0.1,
        "class": "self_attention",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_12_self_att_ln",
        "key_shift": "conformer_block_12_self_att_ln_rel_pos_enc",
        "out_dim": encoder_out_dim,
        "num_heads": 6,
        "total_key_dim": 384,
    },
    "conformer_block_12_self_att_dropout": {
        "class": "dropout",
        "dropout": 0.1,
        "from": "conformer_block_12_self_att_linear",
    },
    "conformer_block_12_self_att_linear": {
        "activation": None,
        "class": "linear",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_12_self_att",
        "out_dim": encoder_out_dim,
        "with_bias": False,
    },
    "conformer_block_12_self_att_ln": {"class": "layer_norm", "from": "conformer_block_12_covmod_1_conv_mod_res"},
    "conformer_block_12_self_att_ln_rel_pos_enc": {
        "class": "relative_positional_encoding",
        "forward_weights_init": {
            "class": "VarianceScaling",
            "distribution": "uniform",
            "mode": "fan_in",
            "scale": 0.78,
        },
        "from": "conformer_block_12_self_att_ln",
        "n_out": 64,
    },
    "conformer_block_12_self_att_res": {
        "class": "combine",
        "from": ["conformer_block_12_self_att_dropout", "conformer_block_12_covmod_1_conv_mod_res"],
        "kind": "add",
    },
    "encoder": {"class": "copy", "from": "conformer_block_12"},
}
