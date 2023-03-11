#!rnn.py

"""
https://github.com/rwth-i6/returnn/issues/1127
"""

from __future__ import annotations
from typing import Tuple, Optional, Sequence, Dict, Any
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
                "pau",
                "aa",
                "ae",
                "ah",
                "ao",
                "aw",
                "ax",
                "ax-h",
                "axr",
                "ay",
                "b",
                "bcl",
                "ch",
                "d",
                "dcl",
                "dh",
                "dx",
                "eh",
                "el",
                "em",
                "en",
                "eng",
                "epi",
                "er",
                "ey",
                "f",
                "g",
                "gcl",
                "h#",
                "hh",
                "hv",
                "ih",
                "ix",
                "iy",
                "jh",
                "k",
                "kcl",
                "l",
                "m",
                "n",
                "ng",
                "nx",
                "ow",
                "oy",
                "p",
                "pcl",
                "q",
                "r",
                "s",
                "sh",
                "t",
                "tcl",
                "th",
                "uh",
                "uw",
                "ux",
                "v",
                "w",
                "y",
                "z",
                "zh",
            ],
            "vocab_file": None,
            "unknown_label": None,
            "user_defined_symbols": {"<sil>": 0},
        },
    },
}

from i6_experiments.users.zeyer.beam_search import beam_search, IDecoder


class Model(nn.Module):
    """Model definition"""

    def __init__(
        self,
        *,
        nb_target_dim: nn.Dim,
        wb_target_dim: nn.Dim,
        blank_idx: int,
        bos_idx: int,
    ):
        super(Model, self).__init__()

        self.nb_target_dim = nb_target_dim
        self.wb_target_dim = wb_target_dim
        self.blank_idx = blank_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        self.lm = DecoderLabelSync()
        self.readout_in_am = nn.Linear(nn.FeatureDim("readout", 20), with_bias=False)
        self.readout_in_lm = nn.Linear(self.readout_in_am.out_dim, with_bias=False)
        self.out_wb_label_logits = nn.Linear(wb_target_dim)

    def encode(self, source: nn.Tensor, *, in_spatial_dim: nn.Dim) -> (Dict[str, nn.Tensor], nn.Dim):
        """encode, and extend the encoder output for things we need in the decoder"""
        return dict(enc=source), in_spatial_dim

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
        prev_wb_target: Optional[nn.Tensor] = None,  # with blank
        wb_target_spatial_dim: Optional[nn.Dim] = None,  # single step or align-label spatial axis
        state: Optional[nn.LayerState] = None,
    ) -> Tuple[nn.Tensor, nn.LayerState]:
        """decoder step, or operating on full seq"""
        assert state is not None
        state_ = nn.LayerState()

        assert prev_wb_target is not None and wb_target_spatial_dim is not None
        assert wb_target_spatial_dim in {enc_spatial_dim, nn.single_step_dim}

        lm, state_.lm = self.lm(prev_wb_target, axis=wb_target_spatial_dim, state=state.lm)
        readout_in_lm = self.readout_in_lm(lm)

        readout_in_am = self.readout_in_am(enc)
        readout_in = nn.combine_bc(readout_in_am, "+", readout_in_lm)
        readout = nn.reduce_out(readout_in, mode="max", num_pieces=2)

        return self.out_wb_label_logits(readout), state_


class DecoderLabelSync(nn.Module):
    """
    Often called the (I)LM part.
    Runs label-sync, i.e. only on non-blank labels.
    """

    def __init__(
        self,
        *,
        embed_dim: nn.Dim = nn.FeatureDim("embed", 20),
        lstm_dim: nn.Dim = nn.FeatureDim("lstm", 20),
    ):
        super(DecoderLabelSync, self).__init__()
        self.embed = nn.Linear(embed_dim)
        self.lstm = nn.LSTM(lstm_dim)

    def default_initial_state(self, *, batch_dims: Sequence[nn.Dim]) -> Optional[nn.LayerState]:
        """init"""
        return self.lstm.default_initial_state(batch_dims=batch_dims)

    def __call__(self, source: nn.Tensor, *, axis: nn.Dim, state: nn.LayerState) -> (nn.Tensor, nn.LayerState):
        embed = self.embed(source)
        lstm, state = self.lstm(embed, axis=axis, state=state)
        return lstm, state


def _model_def(*, epoch: int, target_dim: nn.Dim) -> Model:
    """Function is run within RETURNN."""
    return Model(
        nb_target_dim=target_dim,
        wb_target_dim=target_dim + 1,
        blank_idx=target_dim.dimension,
        bos_idx=0,
    )


def _recog_def(
    *,
    model: Model,
    data: nn.Tensor,
    data_spatial_dim: nn.Dim,
    targets_dim: nn.Dim,  # noqa
) -> nn.Tensor:
    """
    Function is run within RETURNN.

    :return: recog results including beam
    """
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
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
            logits, state = model.decode(
                **enc,
                enc_spatial_dim=nn.single_step_dim,
                wb_target_spatial_dim=nn.single_step_dim,
                prev_wb_target=prev_target,
                state=state,
            )
            return nn.log_softmax(logits, axis=logits.feature_dim), state

    return beam_search(_Decoder())


def get_network(*, epoch: int, **_kwargs_unused) -> Dict[str, Any]:
    nn.reset_default_root_name_ctx()
    default_input_key = default_input
    default_target_key = target
    extern_data_dict = extern_data
    data = nn.Data(name=default_input_key, **extern_data_dict[default_input_key])
    targets = nn.Data(name=default_target_key, **extern_data_dict[default_target_key])
    data_spatial_dim = data.get_time_dim_tag()
    data = nn.get_extern_data(data)
    targets = nn.get_extern_data(targets)
    model = _model_def(epoch=epoch, target_dim=targets.feature_dim)
    recog_out = _recog_def(model=model, data=data, data_spatial_dim=data_spatial_dim, targets_dim=targets.feature_dim)
    assert isinstance(recog_out, nn.Tensor)
    recog_out.mark_as_default_output()
    net_dict = nn.get_returnn_config().get_net_dict_raw_dict(root_module=model)
    with open("gen_config.py", "w") as f:
        f.write(nn.get_returnn_config().get_complete_py_code_str(root_module=model))
    return net_dict


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
