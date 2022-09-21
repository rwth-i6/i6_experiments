#!rnn.py

"""
https://github.com/rwth-i6/returnn/issues/1127
"""

from __future__ import annotations
from typing import Tuple, Optional, Sequence, Dict
import contextlib
import os
import sys


task = "initialize_model"
target = "classes"
use_tensorflow = True
behavior_version = 12
default_input = "data"
batching = "sorted"
batch_size = 20000
max_seqs = 200
# log = ["./test-recog.returnn.log"]
log_verbosity = 3
log_batch_size = True
tf_log_memory_usage = True
tf_session_opts = {"gpu_options": {"allow_growth": True}}
model = "dummy-model-init"


sys.path.insert(0, "/Users/az/i6/setups/2022-03-19--sis-i6-exp/recipe")
sys.path.insert(1, "/Users/az/i6/setups/2022-03-19--sis-i6-exp/ext/sisyphus")

from returnn_common import nn

from returnn.tf.util.data import batch_dim, SpatialDim, FeatureDim

time_dim = SpatialDim("time")
audio_dim = FeatureDim("audio", 80)
out_spatial_dim = SpatialDim("out-spatial")
phones_dim = FeatureDim("phones", 61)

extern_data = {
    "data": {"dim_tags": [batch_dim, time_dim, audio_dim]},
    "classes": {
        "dim_tags": [batch_dim, out_spatial_dim],
        "sparse_dim": phones_dim,
        "vocab": {
            "class": "Vocabulary",
            "labels": [
                "pau", "aa", "ae", "ah", "ao", "aw", "ax", "ax-h", "axr", "ay", "b", "bcl", "ch", "d", "dcl",
                "dh", "dx", "eh", "el", "em", "en", "eng", "epi", "er", "ey", "f", "g", "gcl", "h#", "hh", "hv",
                "ih", "ix", "iy", "jh", "k", "kcl", "l", "m", "n", "ng", "nx", "ow", "oy", "p", "pcl", "q", "r",
                "s", "sh", "t", "tcl", "th", "uh", "uw", "ux", "v", "w", "y", "z", "zh"],
            "vocab_file": None,
            "unknown_label": None,
            "user_defined_symbols": {"<sil>": 0},
        },
    },
}


from i6_experiments.users.zeyer.experiments.exp2022_07_21_transducer.recog import IDecoder, beam_search


class Model(nn.Module):
    """Model definition"""

    def __init__(self, *,
                 nb_target_dim: nn.Dim,
                 wb_target_dim: nn.Dim,
                 blank_idx: int,
                 bos_idx: int,
                 enc_key_total_dim: nn.Dim = nn.FeatureDim("enc_key_total_dim", 200),
                 att_num_heads: nn.Dim = nn.SpatialDim("att_num_heads", 1),
                 att_dropout: float = 0.1,
                 ):
        super(Model, self).__init__()
        self.encoder = nn.Linear(nn.FeatureDim("enc", 100))

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
        self.lm = DecoderLabelSync()
        self.readout_in_am = nn.Linear(nn.FeatureDim("readout", 1000), with_bias=False)
        self.readout_in_am_dropout = 0.1
        self.readout_in_lm = nn.Linear(self.readout_in_am.out_dim, with_bias=False)
        self.readout_in_lm_dropout = 0.1
        self.readout_in_bias = nn.Parameter([self.readout_in_am.out_dim])
        self.out_nb_label_logits = nn.Linear(nb_target_dim)
        self.label_log_prob_dropout = 0.3
        self.out_emit_logit = nn.Linear(nn.FeatureDim("emit", 1))

    def encode(self, source: nn.Tensor, *, in_spatial_dim: nn.Dim) -> (Dict[str, nn.Tensor], nn.Dim):
        """encode, and extend the encoder output for things we need in the decoder"""
        enc = self.encoder(source)
        enc_ctx = self.enc_ctx(nn.dropout(enc, self.enc_ctx_dropout, axis=enc.feature_dim))
        enc_ctx_win, _ = nn.window(enc_ctx, axis=in_spatial_dim, window_dim=self.enc_win_dim)
        enc_val_win, _ = nn.window(enc, axis=in_spatial_dim, window_dim=self.enc_win_dim)
        return dict(enc=enc, enc_ctx_win=enc_ctx_win, enc_val_win=enc_val_win), in_spatial_dim

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
        att_weights = nn.softmax(att_energy, axis=self.enc_win_dim)
        att_weights = nn.dropout(att_weights, dropout=self.att_dropout, axis=self.enc_win_dim)
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


def _model_def(*, epoch: int, target_dim: nn.Dim) -> Model:
    """Function is run within RETURNN."""
    return Model(
        nb_target_dim=target_dim,
        wb_target_dim=target_dim + 1,
        blank_idx=target_dim.dimension,
        bos_idx=0,
    )


def _recog_def(*,
               model: Model,
               data: nn.Tensor, data_spatial_dim: nn.Dim,
               targets_dim: nn.Dim,  # noqa
               ) -> nn.Tensor:
    """
    Function is run within RETURNN.

    :return: recog results including beam
    """
    batch_dims = data.batch_dims_ordered(data_spatial_dim)
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

    return beam_search(_Decoder())


from i6_experiments.users.zeyer.experiments.exp2022_07_21_transducer.recog import (
    _returnn_get_network as get_network,
)

# https://github.com/rwth-i6/returnn/issues/957
# https://stackoverflow.com/a/16248113/133374
import resource
import sys

try:
    resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
except Exception as exc:
    print(f"resource.setrlimit {type(exc).__name__}: {exc}")
sys.setrecursionlimit(10**6)
_cf_cache = {}


def cf(filename):
    "Cache manager"
    from subprocess import check_output, CalledProcessError

    if filename in _cf_cache:
        return _cf_cache[filename]
    if int(os.environ.get("RETURNN_DEBUG", "0")):
        print("use local file: %s" % filename)
        return filename  # for debugging
    try:
        cached_fn = check_output(["cf", filename]).strip().decode("utf8")
    except CalledProcessError:
        print("Cache manager: Error occured, using local file")
        return filename
    assert os.path.exists(cached_fn)
    _cf_cache[filename] = cached_fn
    return cached_fn
