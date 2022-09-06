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
from typing import Optional, Dict, Sequence
import contextlib
from sisyphus import tk
from .task import get_switchboard_task
from .train import train
from .recog import recog
from .align import align
from returnn_common import nn
from returnn_common.nn.encoder.blstm_cnn_specaug import BlstmCnnSpecAugEncoder


# version is used for the hash of the model definition,
# together with the model def function name together with the module name (__name__).
assert __name__.startswith("i6_experiments.")  # just a sanity check
version = 1
extra_hash = (version,)


def sis_config_main():
    """sis config function"""
    task = get_switchboard_task()

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

    tk.register_output('step1', recog(task, step1_model).main_measure_value)
    tk.register_output('step3', recog(task, step3_model).main_measure_value)
    tk.register_output('step4', recog(task, step4_model).main_measure_value)


py = sis_config_main  # `py` is the default sis config function name


class Model(nn.Module):
    """Model definition"""

    # TODO implement generic interface https://github.com/rwth-i6/returnn_common/issues/49 ?

    def __init__(self, *, num_enc_layers=6):
        super(Model, self).__init__()
        self.encoder = BlstmCnnSpecAugEncoder(num_layers=num_enc_layers)
        # TODO decoder...


class Decoder(nn.Module):
    """Decoder"""

    def __init__(self, *,
                 nb_target_dim: nn.Dim,
                 wb_target_dim: nn.Dim,
                 blank_idx: int,
                 enc_key_total_dim: nn.Dim = nn.FeatureDim("enc_key_total_dim", 200),
                 att_num_heads: nn.Dim = nn.SpatialDim("att_num_heads", 1),
                 att_dropout: float,
                 ):
        super(Decoder, self).__init__()

        self.nb_target_dim = nb_target_dim
        self.wb_target_dim = wb_target_dim
        self.blank_idx = blank_idx

        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        self.att_num_heads = att_num_heads
        self.att_dropout = att_dropout

        self.att_query = nn.Linear(enc_key_total_dim, with_bias=False)
        self.lm = DecoderLabelSync()
        self.readout_in_am = nn.Linear(nn.FeatureDim("readout", 1000), with_bias=False)
        self.readout_in_am_dropout = 0.1
        self.readout_in_lm = nn.Linear(self.readout_in_am.out_dim, with_bias=False)
        self.readout_in_lm_dropout = 0.1
        self.readout_in_bias = nn.Parameter([self.readout_in_am.out_dim])
        self.out_nb_label_logits = nn.Linear(nb_target_dim)
        self.label_log_prob_dropout = 0.3
        self.out_emit_logit = nn.Linear(nn.FeatureDim("emit", 1))

    def encoder_ext(self, enc_out: nn.Tensor, enc_spatial_dim: nn.Dim) -> Dict[str, nn.Tensor]:
        """Extend the encoder output"""
        # TODO
        return enc_out

    def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> Optional[nn.LayerState]:
        """Default initial state"""
        return nn.LayerState(lm=self.lm.default_initial_state(batch_dims=batch_dims))

    def __call__(self, *,
                 enc: nn.Tensor,  # single frame if axis is single step, or sequence otherwise ("am" before)
                 enc_spatial_dim: nn.Dim,  # single step or time axis,
                 enc_ctx_win: nn.Tensor,  # like enc
                 enc_val_win: nn.Tensor,  # like enc
                 enc_win_axis: nn.Dim,  # for enc_..._win
                 all_combinations_out: bool = False,  # [...,prev_nb_target_spatial_dim,axis] out
                 prev_nb_target: Optional[nn.Tensor] = None,  # non-blank
                 nb_target_spatial_dim: Optional[nn.Dim] = None,
                 prev_wb_target: Optional[nn.Tensor] = None,  # with blank
                 wb_target_spatial_dim: Optional[nn.Dim] = None,  # single step or align-label spatial axis
                 state: Optional[nn.LayerState] = None,
                 ) -> (nn.Tensor, nn.LayerState):
        if state is None:
            batch_dims = enc.batch_dims_ordered(remove=(enc.feature_dim, enc_spatial_dim))
            state = self.default_initial_state(batch_dims=batch_dims)
        state_ = nn.LayerState()

        att_query = self.att_query(enc)
        att_energy = nn.dot(enc_ctx_win, att_query, reduce=att_query.feature_dim)
        att_weights = nn.softmax(att_energy, axis=enc_win_axis)
        att_weights = nn.dropout(att_weights, dropout=self.att_dropout, axis=enc_win_axis)
        att = nn.dot(att_weights, enc_val_win, reduce=enc_win_axis)

        if all_combinations_out:
            assert prev_nb_target is not None and nb_target_spatial_dim is not None
            lm_scope = contextlib.nullcontext()
            lm_input = prev_nb_target
            lm_axis = nb_target_spatial_dim
        else:
            assert prev_wb_target is not None and wb_target_spatial_dim is not None
            prev_out_emit = prev_wb_target != self.blank_idx
            lm_scope = nn.MaskedComputation(mask=prev_out_emit)
            lm_input = nn.reinterpret_set_sparse_dim(prev_wb_target, out_dim=self.nb_target_dim)
            lm_axis = wb_target_spatial_dim

        with lm_scope:
            lm, state_.lm = self.lm(lm_input, axis=lm_axis, state=state.lm)

        # We could have simpler code by directly concatenating them.
        # However, for better efficiency, keep am/lm path separate initially.
        readout_in_am_in = nn.concat_features(enc, att)
        readout_in_am_in = nn.dropout(readout_in_am_in, self.readout_in_am_dropout, axis=readout_in_am_in.feature_dim)
        readout_in_am = self.readout_in_am(readout_in_am_in)
        readout_in_lm_in = nn.dropout(lm, self.readout_in_lm_dropout, axis=lm.feature_dim)
        readout_in_lm = self.readout_in_lm(readout_in_lm_in)
        readout_in = nn.combine_bc(readout_in_am, "+", readout_in_lm)
        readout_in += self.readout_in_bias
        readout = nn.reduce_out(readout_in, mode="max", num_pieces=2)

        out_nb_label_embed_in = nn.dropout(readout, self.label_log_prob_dropout, axis=readout.feature_dim)

        # TODO maybe make this an interface - the label logits are not needed for framewise loss in the blank frames
        label_logits = self.out_nb_label_logits(out_nb_label_embed_in)
        label_log_prob = nn.log_softmax(label_logits, axis=label_logits.feature_dim)

        emit_logit = self.out_emit_logit(readout)
        emit_log_prob = nn.log_sigmoid(emit_logit)
        blank_log_prob = nn.log_sigmoid(-emit_logit)
        label_emit_log_prob = label_log_prob + nn.squeeze(emit_log_prob, axis=emit_log_prob.feature_dim)
        assert self.blank_idx == label_log_prob.feature_dim.dimension  # not implemented otherwise
        output_log_prob = nn.concat_features(label_emit_log_prob, blank_log_prob)

        return output_log_prob, state_


class DecoderLabelSync(nn.Module):
    """
    Often called the (I)LM part.
    Runs label-sync, i.e. only on non-blank labels.
    """
    def __init__(self, *,
                 embed_dim: nn.Dim = nn.FeatureDim("embed", 256),
                 dropout: float = 0.2,
                 lstm_dim: nn.Dim = nn.FeatureDim("lstm", 1024),
                 ):
        super(DecoderLabelSync, self).__init__()
        self.embed = nn.Linear(embed_dim)
        self.dropout = dropout
        self.lstm = nn.LSTM(lstm_dim)

    def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> Optional[nn.LayerState]:
        """init"""
        return self.lstm.default_initial_state(batch_dims=batch_dims)

    def __call__(self, source: nn.Tensor, *, axis: nn.Dim, state: nn.LayerState) -> (nn.Tensor, nn.LayerState):
        embed = self.embed(source)
        embed = nn.dropout(embed, self.dropout, axis=embed.feature_dim)
        lstm, state = self.lstm(embed, axis=axis, state=state)
        return lstm, state


def from_scratch_model_def(*, epoch: int, target_dim: nn.Dim) -> Model:
    """Function is run within RETURNN."""
    pass  # TODO


def from_scratch_training(*,
                          model: Model,
                          data: nn.Data, data_spatial_dim: nn.Dim,
                          targets: nn.Data, targets_spatial_dim: nn.Dim
                          ):
    """Function is run within RETURNN."""
    # TODO feed through model, define full sum loss, mark_as_loss


def extended_model_def(*, epoch: int, target_dim: nn.Dim) -> Model:
    """Function is run within RETURNN."""
    pass  # TODO


def extended_model_training(*,
                            model: Model,
                            data: nn.Data, data_spatial_dim: nn.Dim,
                            align_targets: nn.Data, align_targets_spatial_dim: nn.Dim
                            ):
    """Function is run within RETURNN."""
    pass  # TODO


def model_recog(*,
                model: Model,
                data: nn.Data, data_spatial_dim: nn.Dim,
                target_vocab: nn.Dim,
                ):
    """Function is run within RETURNN."""
    pass  # TODO
    # TODO probably this should be moved over to the recog module, as this will probably be the same always
