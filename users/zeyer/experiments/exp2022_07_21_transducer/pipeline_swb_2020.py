"""
Replicating the pipeline of my 2020 transducer work:
https://github.com/rwth-i6/returnn-experiments/tree/master/2020-rnn-transducer

Note: this file is loaded in different contexts:

- as a sisyphus config file. In this case, the function `py` is called.
- via the generated RETURNN configs. In this case, all the Sisyphus stuff is ignored,
  and only selected functions will run.

The reason for this is that we want to have all the relevant code for the experiment
to be in one place, such that we can make a copy of the file as a base for another separate experiment.

Note on the hash of the model definition:
This is explicit, via the version object below,
and via the module name (__name__; this includes the package name),
and via the model def function name.

Note on the motivation for the interface:
- should be flexible to different tasks (datasets)
- should be simple (obviously!)
- for training and recognition (alignment)
- need to use dynamic get_network because we don't want to run the net code in the root config

"""

from __future__ import annotations
from typing import Optional, Dict, Sequence, Tuple
import contextlib
from sisyphus import tk
from returnn_common import nn
from returnn_common.nn.encoder.blstm_cnn_specaug import BlstmCnnSpecAugEncoder

from .task import Task, get_switchboard_task
from .train import train
from .recog import recog, beam_search, IDecoder
from .align import align


# version is used for the hash of the model definition,
# together with the model def function name together with the module name (__name__).
assert __name__.startswith("i6_experiments.")  # just a sanity check
version = 1
extra_hash = (version,)


def sis_config_main():
    """sis config function"""
    task = get_switchboard_task()
    pipeline(task)


py = sis_config_main  # `py` is the default sis config function name


def pipeline(task: Task):
    """run the pipeline for the given task, register outputs"""
    step1_model = train(
        task=task, model_def=from_scratch_model_def, train_def=from_scratch_training, extra_hash=extra_hash)
    step2_alignment = align(task=task, model=step1_model)
    # use step1 model params; different to the paper
    step3_model = train(
        task=task, model_def=extended_model_def, train_def=extended_model_training, extra_hash=extra_hash,
        alignment=step2_alignment, init_params=step1_model.checkpoint)
    step4_model = train(
        task=task, model_def=extended_model_def, train_def=extended_model_training, extra_hash=extra_hash,
        alignment=step2_alignment, init_params=step3_model.checkpoint)

    tk.register_output('step1', recog(task, step1_model, recog_def=model_recog).main_measure_value)
    tk.register_output('step3', recog(task, step3_model, recog_def=model_recog).main_measure_value)
    tk.register_output('step4', recog(task, step4_model, recog_def=model_recog).main_measure_value)


class Model(nn.Module):
    """Model definition"""

    def __init__(self, *,
                 num_enc_layers=6,
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
        self.encoder = BlstmCnnSpecAugEncoder(num_layers=num_enc_layers, l2=l2)

        self.nb_target_dim = nb_target_dim
        self.wb_target_dim = wb_target_dim
        self.blank_idx = blank_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        self.att_num_heads = att_num_heads
        self.att_dropout = att_dropout

        self.enc_ctx = nn.Linear(enc_key_total_dim)
        self.enc_ctx_dropout = 0.2
        self.enc_win_dim = nn.SpatialDim("enc_win_dim", 5)
        self.att_query = nn.Linear(enc_key_total_dim, with_bias=False)
        self.lm = DecoderLabelSync(l2=l2)
        self.readout_in_am = nn.Linear(nn.FeatureDim("readout", 1000), with_bias=False)
        self.readout_in_am_dropout = 0.1
        self.readout_in_lm = nn.Linear(self.readout_in_am.out_dim, with_bias=False)
        self.readout_in_lm_dropout = 0.1
        self.readout_in_bias = nn.Parameter([self.readout_in_am.out_dim])
        self.out_nb_label_logits = nn.Linear(nb_target_dim)
        self.label_log_prob_dropout = 0.3
        self.out_emit_logit = nn.Linear(nn.FeatureDim("emit", 1))

        for p in self.enc_ctx.parameters():
            p.weight_decay = l2

    def encode(self, source: nn.Tensor, *, in_spatial_dim: nn.Dim) -> (Dict[str, nn.Tensor], nn.Dim):
        """encode, and extend the encoder output for things we need in the decoder"""
        enc, enc_spatial_dim = self.encoder(source, spatial_dim=in_spatial_dim)
        enc_ctx = self.enc_ctx(nn.dropout(enc, self.enc_ctx_dropout, axis=enc.feature_dim))
        enc_ctx_win, _ = nn.window(enc_ctx, axis=enc_spatial_dim, window_dim=self.enc_win_dim)
        enc_val_win, _ = nn.window(enc, axis=enc_spatial_dim, window_dim=self.enc_win_dim)
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

    def decode(self, *,
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
            batch_dims = enc.batch_dims_ordered(
                remove=(enc.feature_dim, enc_spatial_dim)
                if enc_spatial_dim != nn.single_step_dim
                else (enc.feature_dim,))
            state = self.decoder_default_initial_state(batch_dims=batch_dims)
        state_ = nn.LayerState()

        att_query = self.att_query(enc)
        att_energy = nn.dot(enc_ctx_win, att_query, reduce=att_query.feature_dim)
        att_energy = att_energy * (att_energy.feature_dim.dimension ** -0.5)
        att_weights = nn.softmax(att_energy, axis=self.enc_win_dim)
        att_weights = nn.dropout(att_weights, dropout=self.att_dropout, axis=att_weights.shape_ordered)
        att = nn.dot(att_weights, enc_val_win, reduce=self.enc_win_dim)

        if all_combinations_out:
            assert prev_nb_target is not None and prev_nb_target_spatial_dim is not None
            assert prev_nb_target_spatial_dim in prev_nb_target.shape
            assert enc_spatial_dim != nn.single_step_dim
            lm_scope = contextlib.nullcontext()
            lm_input = prev_nb_target
            lm_axis = prev_nb_target_spatial_dim
        else:
            assert prev_wb_target is not None and wb_target_spatial_dim is not None
            assert wb_target_spatial_dim in {enc_spatial_dim, nn.single_step_dim}
            prev_out_emit = prev_wb_target != self.blank_idx
            lm_scope = nn.MaskedComputation(mask=prev_out_emit)
            lm_input = nn.reinterpret_set_sparse_dim(prev_wb_target, out_dim=self.nb_target_dim)
            lm_axis = wb_target_spatial_dim

        with lm_scope:
            lm, state_.lm = self.lm(lm_input, axis=lm_axis, state=state.lm)

            # We could have simpler code by directly concatenating the readout inputs.
            # However, for better efficiency, keep am/lm path separate initially.
            readout_in_lm_in = nn.dropout(lm, self.readout_in_lm_dropout, axis=lm.feature_dim)
            readout_in_lm = self.readout_in_lm(readout_in_lm_in)

        readout_in_am_in = nn.concat_features(enc, att)
        readout_in_am_in = nn.dropout(readout_in_am_in, self.readout_in_am_dropout, axis=readout_in_am_in.feature_dim)
        readout_in_am = self.readout_in_am(readout_in_am_in)
        readout_in = nn.combine_bc(readout_in_am, "+", readout_in_lm)
        readout_in += self.readout_in_bias
        readout = nn.reduce_out(readout_in, mode="max", num_pieces=2)

        return ProbsFromReadout(model=self, readout=readout), state_


class DecoderLabelSync(nn.Module):
    """
    Often called the (I)LM part.
    Runs label-sync, i.e. only on non-blank labels.
    """
    def __init__(self, *,
                 embed_dim: nn.Dim = nn.FeatureDim("embed", 256),
                 lstm_dim: nn.Dim = nn.FeatureDim("lstm", 1024),
                 dropout: float = 0.2,
                 l2: float = 0.0001,
                 ):
        super(DecoderLabelSync, self).__init__()
        self.embed = nn.Linear(embed_dim)
        self.dropout = dropout
        self.lstm = nn.LSTM(lstm_dim)
        for p in self.parameters():
            p.weight_decay = l2

    def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> Optional[nn.LayerState]:
        """init"""
        return self.lstm.default_initial_state(batch_dims=batch_dims)

    def __call__(self, source: nn.Tensor, *, axis: nn.Dim, state: nn.LayerState) -> (nn.Tensor, nn.LayerState):
        embed = self.embed(source)
        embed = nn.dropout(embed, self.dropout, axis=embed.feature_dim)
        lstm, state = self.lstm(embed, axis=axis, state=state)
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


def from_scratch_model_def(*, epoch: int, target_dim: nn.Dim) -> Model:
    """Function is run within RETURNN."""
    return Model(
        num_enc_layers=min((epoch - 1) // 2 + 2, 6) if epoch <= 10 else 6,
        nb_target_dim=target_dim,
        wb_target_dim=target_dim + 1,
        blank_idx=target_dim.dimension,
        bos_idx=_get_bos_idx(target_dim),
    )


def from_scratch_training(*,
                          model: Model,
                          data: nn.Tensor, data_spatial_dim: nn.Dim,
                          targets: nn.Tensor, targets_spatial_dim: nn.Dim
                          ):
    """Function is run within RETURNN."""
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    prev_targets, prev_targets_spatial_dim = nn.prev_target_seq(
        targets, spatial_dim=targets_spatial_dim, bos_idx=model.bos_idx, out_one_longer=True)
    probs, _ = model.decode(
        **enc_args,
        enc_spatial_dim=enc_spatial_dim,
        all_combinations_out=True,
        prev_nb_target=prev_targets,
        prev_nb_target_spatial_dim=prev_targets_spatial_dim)
    out_log_prob = probs.get_wb_label_log_probs()
    loss = nn.transducer_time_sync_full_sum_neg_log_prob(
        log_probs=out_log_prob,
        labels=targets,
        input_spatial_dim=enc_spatial_dim,
        labels_spatial_dim=targets_spatial_dim,
        blank_index=model.blank_idx)
    loss.mark_as_loss("full_sum")


def extended_model_def(*, epoch: int, target_dim: nn.Dim) -> Model:
    """Function is run within RETURNN."""
    assert target_dim.vocab
    assert target_dim.vocab.bos_label_id is not None
    # TODO extended model...
    return Model(
        num_enc_layers=6,
        nb_target_dim=target_dim,
        wb_target_dim=target_dim + 1,
        blank_idx=target_dim.dimension,
        bos_idx=target_dim.vocab.bos_label_id,
    )


def extended_model_training(*,
                            model: Model,
                            data: nn.Tensor, data_spatial_dim: nn.Dim,
                            align_targets: nn.Tensor, align_targets_spatial_dim: nn.Dim
                            ):
    """Function is run within RETURNN."""
    pass  # TODO


def model_recog(*,
                model: Model,
                data: nn.Tensor, data_spatial_dim: nn.Dim,
                targets_dim: nn.Dim,  # noqa
                ) -> nn.Tensor:
    """
    Function is run within RETURNN.

    :return: recog results including beam
    """
    batch_dims = data.batch_dims_ordered((data_spatial_dim, data.feature_dim))
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)

    class _Decoder(IDecoder):
        target_spatial_dim = enc_spatial_dim  # time-sync transducer
        include_eos = True

        def max_seq_len(self) -> nn.Tensor:
            """max seq len"""
            return nn.dim_value(enc_spatial_dim) * 2

        def initial_state(self) -> nn.LayerState:
            """initial state"""
            return model.decoder_default_initial_state(batch_dims=batch_dims)

        def bos_label(self) -> nn.Tensor:
            """BOS"""
            return nn.constant(model.blank_idx, shape=batch_dims, sparse_dim=model.wb_target_dim)

        def __call__(self, prev_target: nn.Tensor, *, state: nn.LayerState) -> Tuple[nn.Tensor, nn.LayerState]:
            enc = model.encoder_unstack(enc_args)
            probs, state = model.decode(
                **enc,
                enc_spatial_dim=nn.single_step_dim,
                wb_target_spatial_dim=nn.single_step_dim,
                prev_wb_target=prev_target,
                state=state)
            return probs.get_wb_label_log_probs(), state

    res = beam_search(_Decoder())
    assert model.blank_idx == targets_dim.dimension  # added at the end
    res.feature_dim.vocab = nn.Vocabulary.create_vocab_from_labels(
        targets_dim.vocab.labels + ["<blank>"], user_defined_symbols={"<blank>": model.blank_idx})
    return res


# RecogDef API
model_recog.output_with_beam = True
model_recog.output_blank_label = "<blank>"
