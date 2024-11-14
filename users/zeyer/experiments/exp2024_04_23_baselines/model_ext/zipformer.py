"""
ZipFormer

copied from: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/zipformer.py
slightly adopted (also copied funcs/classes from other files)
6f1abd8, 2024-11-13
"""

#!/usr/bin/env python3
# Copyright    2022-2023  Xiaomi Corp.        (authors: Daniel Povey,
#                                                       Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import copy
import logging
import math
import random
import warnings
from typing import Optional, Union, Any, List, Tuple, Dict
import pathlib
import json
import argparse

import torch
from torch import Tensor, nn
from torch.cuda.amp import custom_bwd, custom_fwd

# K2 only required for swoosh activation...
# Check https://k2-fsa.github.io/k2/installation/from_wheels.html.
# Note, I made the import local in the functions that use it.
# TODO we could also remove this and replace it by pure PT implementations.
# import k2

# To provide a RF compatible encoder interface.
import returnn.frontend as rf
from returnn.frontend.encoder.base import ISeqDownsamplingEncoder
from returnn.tensor import Dim  # access Tensor via rf.Tensor, to not confuse with torch.Tensor
from returnn.torch.frontend.bridge import pt_module_to_rf_module


class RFZipFormerEncoder(ISeqDownsamplingEncoder):
    """
    The original icefall :class:`Zipformer2` wrapped in a RF module.

    This does not contain the original convolutional frontend.
    Instead, we currently follow the same configurable frontend
    as in :class:`returnn.frontend.encoder.conformer.ConformerEncoder`.

    In general, this API here is very similar (mostly compatible)
    to :class:`returnn.frontend.encoder.conformer.ConformerEncoder`.

    The main configuration of :class:`Zipformer2` happens via the ``params`` dict
    and :func:`get_encoder_model`.
    """

    def __init__(
        self,
        in_dim: Dim,
        *,
        input_layer: Optional[Union[ISeqDownsamplingEncoder, rf.Module, Any]],
        input_embedding_scale: float = 1.0,
        input_dropout: float = 0.1,
        params: Union[AttributeDict, Dict[str, Any]],
    ):
        """
        :param in_dim: input dim (for what comes into this module, then usually into the frontend)
        :param input_layer: usually the convolutional frontend
        :param input_embedding_scale: for the input projection
        :param input_dropout: for the input projection
        :param params: all the params for the :class:`Zipformer2`, via :func:`get_encoder_model`.
            We first get the default params via :func:`get_params` and then update it with this.
        """
        super().__init__()

        self.in_dim = in_dim

        params_ = get_params()
        params_.update(params)
        self._encoder_pt = get_encoder_model(params_)
        unused_opts = set(params.keys()) - set(params_.got_items)
        assert not unused_opts, f"options not used: {unused_opts}. used options: {params_.got_items}"
        self.encoder = pt_module_to_rf_module(self._encoder_pt)

        self.enc_in_dim = Dim(self._encoder_pt.encoder_dim[0], name="zip_in")
        self.out_dim = Dim(max(self._encoder_pt.encoder_dim), name="zip_out")

        if callable(input_layer) or input_layer is None:
            pass  # leave it as is
        elif isinstance(input_layer, dict):
            input_layer = rf.build_from_dict(input_layer, in_dim)
            input_layer: ISeqDownsamplingEncoder  # for attrib access
        else:
            raise TypeError(f"unexpected input_layer {input_layer!r}")
        self.input_layer = input_layer
        self.input_projection = (
            rf.Linear(self.input_layer.out_dim if self.input_layer else self.in_dim, self.enc_in_dim, with_bias=False)
            if input_layer
            else None
        )
        self.input_embedding_scale = input_embedding_scale
        self.input_dropout = input_dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()

    def __call__(
        self, source: rf.Tensor, *, in_spatial_dim: Dim, collected_outputs: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[rf.Tensor, Dim]:
        assert collected_outputs is None  # currently not supported

        # see train.py
        if rf.get_run_ctx().step % 10 == 0:
            set_batch_count(self._encoder_pt, rf.get_run_ctx().step)

        # see model.py forward_encoder
        if self.input_layer:
            x_subsample, out_spatial_dim = self.input_layer(source, in_spatial_dim=in_spatial_dim)
        else:
            x_subsample, out_spatial_dim = source, in_spatial_dim
        x = self.input_projection(x_subsample) if self.input_projection else x_subsample
        if self.input_embedding_scale != 1.0:
            x = x * self.input_embedding_scale
        x = rf.dropout(x, self.input_dropout, axis=self.dropout_broadcast and self.enc_in_dim)

        batch_dims = x.remaining_dims((out_spatial_dim, self.enc_in_dim))
        assert len(batch_dims) == 1  # just not implemented otherwise
        batch_dim = batch_dims[0]
        assert out_spatial_dim.dyn_size_ext.dims == (batch_dim,)
        x_lens = out_spatial_dim.dyn_size  # (N,)
        x_ = x.copy_compatible_to_dims_raw((out_spatial_dim, batch_dim, self.enc_in_dim))  # (T, N, C)

        src_key_padding_mask = make_pad_mask(x_lens)

        encoder_out, encoder_out_lens = self._encoder_pt(x_, x_lens, src_key_padding_mask)
        # encoder_out: (T, N, C)
        assert encoder_out_lens.shape == x_lens.shape
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)
        out_spatial_dim = Dim(rf.convert_to_tensor(encoder_out_lens, dims=[batch_dim]), name="zip_out_spatial")
        encoder_out_ = rf.convert_to_tensor(encoder_out, dims=[out_spatial_dim, batch_dim, self.out_dim])
        return encoder_out_, out_spatial_dim


# From train.py
def get_encoder_model(params: AttributeDict) -> Zipformer2:
    encoder = Zipformer2(
        output_downsampling_factor=2,
        downsampling_factor=_to_int_tuple(params.downsampling_factor),
        num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
        encoder_dim=_to_int_tuple(params.encoder_dim),
        encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
        query_head_dim=_to_int_tuple(params.query_head_dim),
        pos_head_dim=_to_int_tuple(params.pos_head_dim),
        value_head_dim=_to_int_tuple(params.value_head_dim),
        pos_dim=params.pos_dim,
        num_heads=_to_int_tuple(params.num_heads),
        feedforward_dim=_to_int_tuple(params.feedforward_dim),
        cnn_module_kernel=_to_int_tuple(params.cnn_module_kernel),
        dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        warmup_batches=4000.0,
        causal=params.causal,
        chunk_size=_to_int_tuple(params.chunk_size),
        left_context_frames=_to_int_tuple(params.left_context_frames),
    )
    return encoder


def set_batch_count(model: nn.Module, batch_count: float) -> None:
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name


def get_adjusted_batch_count(params: AttributeDict) -> float:
    # returns the number of batches we would have used so far if we had used the reference
    # duration.  This is for purposes of set_batch_count().
    # ref_duration = 600  # default
    return params.batch_idx_train * (params.max_duration * params.world_size) / params.ref_duration


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)


def _to_int_tuple(s: Union[str, Tuple[int, ...]]) -> Tuple[int, ...]:
    if isinstance(s, str):
        return tuple(map(int, s.split(",")))
    else:
        assert isinstance(s, tuple)
        assert all(isinstance(i, int) for i in s)
        return s


# From icefall/utils.py
class AttributeDict(dict):
    __slots__ = ("got_items",)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.got_items = set()

    def __getitem__(self, item):
        res = super().__getitem__(item)
        self.got_items.add(item)
        return res

    def get(self, item, default=None):
        """
        :param str item:
        :param T default:
        :rtype: T|typing.Any|None
        """
        try:
            return self[item]
        except KeyError:
            return default

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
            return
        raise AttributeError(f"No such attribute '{key}'")

    def __str__(self, indent: int = 2):
        tmp = {}
        for k, v in self.items():
            # PosixPath is ont JSON serializable
            if isinstance(v, pathlib.Path) or isinstance(v, torch.device):
                v = str(v)
            tmp[k] = v
        return json.dumps(tmp, indent=indent, sort_keys=True)


# From train.py, stripped down, extended by parser defaults.
def get_params() -> AttributeDict:
    """Return a dict containing training parameters.

    All training related parameters that are not passed from the commandline
    are saved in the variable `params`.

    We extend this by all the defaults from the parser (:func:`get_parser`).

    Explanation of options saved in `params`:

        - feature_dim: The model input dim. It has to match the one used
                       in computing features.

        - subsampling_factor:  The subsampling factor for the model.

    """
    params = AttributeDict(
        {
            # parameters for zipformer
            "feature_dim": 80,
            "subsampling_factor": 4,  # not passed in, this is fixed.
        }
    )

    parser = get_parser()
    args = parser.parse_args([])
    params.update(vars(args))

    return params


def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    add_model_arguments(parser)

    return parser


def add_model_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-encoder-layers",
        type=str,
        default="2,2,3,4,3,2",
        help="Number of zipformer encoder layers per stack, comma separated.",
    )

    parser.add_argument(
        "--downsampling-factor",
        type=str,
        default="1,2,4,8,4,2",
        help="Downsampling factor for each stack of encoder layers.",
    )

    parser.add_argument(
        "--feedforward-dim",
        type=str,
        default="512,768,1024,1536,1024,768",
        help="Feedforward dimension of the zipformer encoder layers, per stack, comma separated.",
    )

    parser.add_argument(
        "--num-heads",
        type=str,
        default="4,4,4,8,4,4",
        help="Number of attention heads in the zipformer encoder layers: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--encoder-dim",
        type=str,
        default="192,256,384,512,384,256",
        help="Embedding dimension in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--query-head-dim",
        type=str,
        default="32",
        help="Query/key dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--value-head-dim",
        type=str,
        default="12",
        help="Value dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-head-dim",
        type=str,
        default="4",
        help="Positional-encoding dimension per head in encoder stacks: a single int or comma-separated list.",
    )

    parser.add_argument(
        "--pos-dim",
        type=int,
        default="48",
        help="Positional-encoding embedding dimension",
    )

    parser.add_argument(
        "--encoder-unmasked-dim",
        type=str,
        default="192,192,256,256,256,192",
        help="Unmasked dimensions in the encoders, relates to augmentation during training.  "
        "A single int or comma-separated list.  Must be <= each corresponding encoder_dim.",
    )

    parser.add_argument(
        "--cnn-module-kernel",
        type=str,
        default="31,31,15,15,15,31",
        help="Sizes of convolutional kernels in convolution modules in each encoder stack: "
        "a single int or comma-separated list.",
    )

    parser.add_argument(
        "--decoder-dim",
        type=int,
        default=512,
        help="Embedding dimension in the decoder model.",
    )

    parser.add_argument(
        "--joiner-dim",
        type=int,
        default=512,
        help="""Dimension used in the joiner model.
        Outputs from the encoder and decoder model are projected
        to this dimension before adding.
        """,
    )

    parser.add_argument(
        "--attention-decoder-dim",
        type=int,
        default=512,
        help="""Dimension used in the attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-num-layers",
        type=int,
        default=6,
        help="""Number of transformer layers used in attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-attention-dim",
        type=int,
        default=512,
        help="""Attention dimension used in attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-num-heads",
        type=int,
        default=8,
        help="""Number of attention heads used in attention decoder""",
    )

    parser.add_argument(
        "--attention-decoder-feedforward-dim",
        type=int,
        default=2048,
        help="""Feedforward dimension used in attention decoder""",
    )

    parser.add_argument(
        "--causal",
        type=str2bool,
        default=False,
        help="If True, use causal version of model.",
    )

    parser.add_argument(
        "--chunk-size",
        type=str,
        default="16,32,64,-1",
        help="Chunk sizes (at 50Hz frame rate) will be chosen randomly from this list during training. "
        " Must be just -1 if --causal=False",
    )

    parser.add_argument(
        "--left-context-frames",
        type=str,
        default="64,128,256,-1",
        help="Maximum left-contexts for causal training, measured in frames which will "
        "be converted to a number of chunks.  If splitting into chunks, "
        "chunk left-context frames will be chosen randomly from this list; else not relevant.",
    )

    parser.add_argument(
        "--use-transducer",
        type=str2bool,
        default=True,
        help="If True, use Transducer head.",
    )

    parser.add_argument(
        "--use-ctc",
        type=str2bool,
        default=False,
        help="If True, use CTC head.",
    )

    parser.add_argument(
        "--use-attention-decoder",
        type=str2bool,
        default=False,
        help="If True, use attention-decoder head.",
    )

    parser.add_argument(
        "--use-cr-ctc",
        type=str2bool,
        default=False,
        help="If True, use consistency-regularized CTC.",
    )


def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class EncoderInterface(nn.Module):
    def forward(self, x: torch.Tensor, x_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x:
            A tensor of shape (batch_size, input_seq_len, num_features)
            containing the input features.
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames
            in `x` before padding.
        Returns:
          Return a tuple containing two tensors:
            - encoder_out, a tensor of (batch_size, out_seq_len, output_dim)
              containing unnormalized probabilities, i.e., the output of a
              linear layer.
            - encoder_out_lens, a tensor of shape (batch_size,) containing
              the number of frames in `encoder_out` before padding.
        """
        raise NotImplementedError("Please implement it in a subclass")


def logaddexp_onnx(x: Tensor, y: Tensor) -> Tensor:
    max_value = torch.max(x, y)
    diff = torch.abs(x - y)
    return max_value + torch.log1p(torch.exp(-diff))


# RuntimeError: Exporting the operator logaddexp to ONNX opset version
# 14 is not supported. Please feel free to request support or submit
# a pull request on PyTorch GitHub.
#
# The following function is to solve the above error when exporting
# models to ONNX via torch.jit.trace()
def logaddexp(x: Tensor, y: Tensor) -> Tensor:
    # Caution(fangjun): Put torch.jit.is_scripting() before
    # torch.onnx.is_in_onnx_export();
    # otherwise, it will cause errors for torch.jit.script().
    #
    # torch.logaddexp() works for both torch.jit.script() and
    # torch.jit.trace() but it causes errors for ONNX export.
    #
    if torch.jit.is_scripting():
        # Note: We cannot use torch.jit.is_tracing() here as it also
        # matches torch.onnx.export().
        return torch.logaddexp(x, y)
    elif torch.onnx.is_in_onnx_export():
        return logaddexp_onnx(x, y)
    else:
        # for torch.jit.trace()
        return torch.logaddexp(x, y)


class PiecewiseLinear(object):
    """
    Piecewise linear function, from float to float, specified as nonempty list of (x,y) pairs with
    the x values in order.  x values <[initial x] or >[final x] are map to [initial y], [final y]
    respectively.
    """

    def __init__(self, *args):
        assert len(args) >= 1, len(args)
        if len(args) == 1 and isinstance(args[0], PiecewiseLinear):
            self.pairs = list(args[0].pairs)
        else:
            self.pairs = [(float(x), float(y)) for x, y in args]
        for x, y in self.pairs:
            assert isinstance(x, (float, int)), type(x)
            assert isinstance(y, (float, int)), type(y)

        for i in range(len(self.pairs) - 1):
            assert self.pairs[i + 1][0] > self.pairs[i][0], (
                i,
                self.pairs[i],
                self.pairs[i + 1],
            )

    def __str__(self):
        # e.g. 'PiecewiseLinear((0., 10.), (100., 0.))'
        return f"PiecewiseLinear({str(self.pairs)[1:-1]})"

    def __call__(self, x):
        if x <= self.pairs[0][0]:
            return self.pairs[0][1]
        elif x >= self.pairs[-1][0]:
            return self.pairs[-1][1]
        else:
            cur_x, cur_y = self.pairs[0]
            for i in range(1, len(self.pairs)):
                next_x, next_y = self.pairs[i]
                if x >= cur_x and x <= next_x:
                    return cur_y + (next_y - cur_y) * (x - cur_x) / (next_x - cur_x)
                cur_x, cur_y = next_x, next_y
            assert False

    def __mul__(self, alpha):
        return PiecewiseLinear(*[(x, y * alpha) for x, y in self.pairs])

    def __add__(self, x):
        if isinstance(x, (float, int)):
            return PiecewiseLinear(*[(p[0], p[1] + x) for p in self.pairs])
        s, x = self.get_common_basis(x)
        return PiecewiseLinear(*[(sp[0], sp[1] + xp[1]) for sp, xp in zip(s.pairs, x.pairs)])

    def max(self, x):
        if isinstance(x, (float, int)):
            x = PiecewiseLinear((0, x))
        s, x = self.get_common_basis(x, include_crossings=True)
        return PiecewiseLinear(*[(sp[0], max(sp[1], xp[1])) for sp, xp in zip(s.pairs, x.pairs)])

    def min(self, x):
        if isinstance(x, float) or isinstance(x, int):
            x = PiecewiseLinear((0, x))
        s, x = self.get_common_basis(x, include_crossings=True)
        return PiecewiseLinear(*[(sp[0], min(sp[1], xp[1])) for sp, xp in zip(s.pairs, x.pairs)])

    def __eq__(self, other):
        return self.pairs == other.pairs

    def get_common_basis(self, p: "PiecewiseLinear", include_crossings: bool = False):
        """
        Returns (self_mod, p_mod) which are equivalent piecewise linear
        functions to self and p, but with the same x values.

          p: the other piecewise linear function
          include_crossings: if true, include in the x values positions
              where the functions indicate by this and p cross.
        """
        assert isinstance(p, PiecewiseLinear), type(p)

        # get sorted x-values without repetition.
        x_vals = sorted(set([x for x, _ in self.pairs] + [x for x, _ in p.pairs]))
        y_vals1 = [self(x) for x in x_vals]
        y_vals2 = [p(x) for x in x_vals]

        if include_crossings:
            extra_x_vals = []
            for i in range(len(x_vals) - 1):
                if (y_vals1[i] > y_vals2[i]) != (y_vals1[i + 1] > y_vals2[i + 1]):
                    # if the two lines in this subsegment potentially cross each other..
                    diff_cur = abs(y_vals1[i] - y_vals2[i])
                    diff_next = abs(y_vals1[i + 1] - y_vals2[i + 1])
                    # `pos`, between 0 and 1, gives the relative x position,
                    # with 0 being x_vals[i] and 1 being x_vals[i+1].
                    pos = diff_cur / (diff_cur + diff_next)
                    extra_x_val = x_vals[i] + pos * (x_vals[i + 1] - x_vals[i])
                    extra_x_vals.append(extra_x_val)
            if len(extra_x_vals) > 0:
                x_vals = sorted(set(x_vals + extra_x_vals))
        y_vals1 = [self(x) for x in x_vals]
        y_vals2 = [p(x) for x in x_vals]
        return (
            PiecewiseLinear(*zip(x_vals, y_vals1)),
            PiecewiseLinear(*zip(x_vals, y_vals2)),
        )


class ScheduledFloat(torch.nn.Module):
    """
    This object is a torch.nn.Module only because we want it to show up in [top_level module].modules();
    it does not have a working forward() function.  You are supposed to cast it to float, as
    in, float(parent_module.whatever), and use it as something like a dropout prob.

    It is a floating point value whose value changes depending on the batch count of the
    training loop.  It is a piecewise linear function where you specify the (x,y) pairs
    in sorted order on x; x corresponds to the batch index.  For batch-index values before the
    first x or after the last x, we just use the first or last y value.

    Example:
       self.dropout = ScheduledFloat((0.0, 0.2), (4000.0, 0.0), default=0.0)

    `default` is used when self.batch_count is not set or not in training mode or in
     torch.jit scripting mode.
    """

    def __init__(self, *args, default: float = 0.0):
        super().__init__()
        # self.batch_count and self.name will be written to in the training loop.
        self.batch_count = None
        self.name = None
        self.default = default
        self.schedule = PiecewiseLinear(*args)

    def extra_repr(self) -> str:
        return f"batch_count={self.batch_count}, schedule={str(self.schedule.pairs[1:-1])}"

    def __float__(self):
        batch_count = self.batch_count
        if batch_count is None or not self.training or torch.jit.is_scripting() or torch.jit.is_tracing():
            return float(self.default)
        else:
            ans = self.schedule(self.batch_count)
            if random.random() < 0.0002:
                logging.info(f"ScheduledFloat: name={self.name}, batch_count={self.batch_count}, ans={ans}")
            return ans

    def __add__(self, x):
        if isinstance(x, float) or isinstance(x, int):
            return ScheduledFloat(self.schedule + x, default=self.default)
        else:
            return ScheduledFloat(self.schedule + x.schedule, default=self.default + x.default)

    def max(self, x):
        if isinstance(x, float) or isinstance(x, int):
            return ScheduledFloat(self.schedule.max(x), default=self.default)
        else:
            return ScheduledFloat(self.schedule.max(x.schedule), default=max(self.default, x.default))


FloatLike = Union[float, ScheduledFloat]


def random_cast_to_half(x: Tensor, min_abs: float = 5.0e-06) -> Tensor:
    """
    A randomized way of casting a floating point value to half precision.
    """
    if x.dtype == torch.float16:
        return x
    x_abs = x.abs()
    is_too_small = x_abs < min_abs
    # for elements where is_too_small is true, random_val will contain +-min_abs with
    # probability (x.abs() / min_abs), and 0.0 otherwise.  [so this preserves expectations,
    # for those elements].
    random_val = min_abs * x.sign() * (torch.rand_like(x) * min_abs < x_abs)
    return torch.where(is_too_small, random_val, x).to(torch.float16)


class CutoffEstimator:
    """
    Estimates cutoffs of an arbitrary numerical quantity such that a specified
    proportion of items will be above the cutoff on average.

      p is the proportion of items that should be above the cutoff.
    """

    def __init__(self, p: float):
        self.p = p
        # total count of items
        self.count = 0
        # total count of items that were above the cutoff
        self.count_above = 0
        # initial cutoff value
        self.cutoff = 0

    def __call__(self, x: float) -> bool:
        """
        Returns true if x is above the cutoff.
        """
        ans = x > self.cutoff
        self.count += 1
        if ans:
            self.count_above += 1
        cur_p = self.count_above / self.count
        delta_p = cur_p - self.p
        if (delta_p > 0) == ans:
            q = abs(delta_p)
            self.cutoff = x * q + self.cutoff * (1 - q)
        return ans


class SoftmaxFunction(torch.autograd.Function):
    """
    Tries to handle half-precision derivatives in a randomized way that should
    be more accurate for training than the default behavior.
    """

    @staticmethod
    def forward(ctx, x: Tensor, dim: int):
        ans = x.softmax(dim=dim)
        # if x dtype is float16, x.softmax() returns a float32 because
        # (presumably) that op does not support float16, and autocast
        # is enabled.
        if torch.is_autocast_enabled():
            ans = ans.to(torch.get_autocast_gpu_dtype())
        ctx.save_for_backward(ans)
        ctx.x_dtype = x.dtype
        ctx.dim = dim
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor):
        (ans,) = ctx.saved_tensors
        with torch.cuda.amp.autocast(enabled=False):
            ans_grad = ans_grad.to(torch.float32)
            ans = ans.to(torch.float32)
            x_grad = ans_grad * ans
            x_grad = x_grad - ans * x_grad.sum(dim=ctx.dim, keepdim=True)
            return x_grad, None


def softmax(x: Tensor, dim: int):
    if not x.requires_grad or torch.jit.is_scripting() or torch.jit.is_tracing():
        return x.softmax(dim=dim)

    return SoftmaxFunction.apply(x, dim)


class MaxEigLimiterFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        coeffs: Tensor,
        direction: Tensor,
        channel_dim: int,
        grad_scale: float,
    ) -> Tensor:
        ctx.channel_dim = channel_dim
        ctx.grad_scale = grad_scale
        ctx.save_for_backward(x.detach(), coeffs.detach(), direction.detach())
        return x

    @staticmethod
    def backward(ctx, x_grad, *args):
        with torch.enable_grad():
            (x_orig, coeffs, new_direction) = ctx.saved_tensors
            x_orig.requires_grad = True
            num_channels = x_orig.shape[ctx.channel_dim]
            x = x_orig.transpose(ctx.channel_dim, -1).reshape(-1, num_channels)
            new_direction.requires_grad = False
            x = x - x.mean(dim=0)
            x_var = (x**2).mean()
            x_residual = x - coeffs * new_direction
            x_residual_var = (x_residual**2).mean()
            # `variance_proportion` is the proportion of the variance accounted for
            # by the top eigen-direction.  This is to be minimized.
            variance_proportion = (x_var - x_residual_var) / (x_var + 1.0e-20)
            variance_proportion.backward()
        x_orig_grad = x_orig.grad
        x_extra_grad = x_orig.grad * ctx.grad_scale * x_grad.norm() / (x_orig_grad.norm() + 1.0e-20)
        return x_grad + x_extra_grad.detach(), None, None, None, None


class BiasNormFunction(torch.autograd.Function):
    # This computes:
    #   scales = (torch.mean((x - bias) ** 2, keepdim=True)) ** -0.5 * log_scale.exp()
    #   return x * scales
    # (after unsqueezing the bias), but it does it in a memory-efficient way so that
    # it can just store the returned value (chances are, this will also be needed for
    # some other reason, related to the next operation, so we can save memory).
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        bias: Tensor,
        log_scale: Tensor,
        channel_dim: int,
        store_output_for_backprop: bool,
    ) -> Tensor:
        assert bias.ndim == 1
        if channel_dim < 0:
            channel_dim = channel_dim + x.ndim
        ctx.store_output_for_backprop = store_output_for_backprop
        ctx.channel_dim = channel_dim
        for _ in range(channel_dim + 1, x.ndim):
            bias = bias.unsqueeze(-1)
        scales = (torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5) * log_scale.exp()
        ans = x * scales
        ctx.save_for_backward(
            ans.detach() if store_output_for_backprop else x,
            scales.detach(),
            bias.detach(),
            log_scale.detach(),
        )
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tensor:
        ans_or_x, scales, bias, log_scale = ctx.saved_tensors
        if ctx.store_output_for_backprop:
            x = ans_or_x / scales
        else:
            x = ans_or_x
        x = x.detach()
        x.requires_grad = True
        bias.requires_grad = True
        log_scale.requires_grad = True
        with torch.enable_grad():
            # recompute scales from x, bias and log_scale.
            scales = (torch.mean((x - bias) ** 2, dim=ctx.channel_dim, keepdim=True) ** -0.5) * log_scale.exp()
            ans = x * scales
            ans.backward(gradient=ans_grad)
        return x.grad, bias.grad.flatten(), log_scale.grad, None, None


class BiasNorm(torch.nn.Module):
    """
    This is intended to be a simpler, and hopefully cheaper, replacement for
    LayerNorm.  The observation this is based on, is that Transformer-type
    networks, especially with pre-norm, sometimes seem to set one of the
    feature dimensions to a large constant value (e.g. 50), which "defeats"
    the LayerNorm because the output magnitude is then not strongly dependent
    on the other (useful) features.  Presumably the weight and bias of the
    LayerNorm are required to allow it to do this.

    Instead, we give the BiasNorm a trainable bias that it can use when
    computing the scale for normalization.  We also give it a (scalar)
    trainable scale on the output.


    Args:
       num_channels: the number of channels, e.g. 512.
       channel_dim: the axis/dimension corresponding to the channel,
         interpreted as an offset from the input's ndim if negative.
         This is NOT the num_channels; it should typically be one of
         {-2, -1, 0, 1, 2, 3}.
      log_scale: the initial log-scale that we multiply the output by; this
         is learnable.
      log_scale_min: FloatLike, minimum allowed value of log_scale
      log_scale_max: FloatLike, maximum allowed value of log_scale
      store_output_for_backprop: only possibly affects memory use; recommend
         to set to True if you think the output of this module is more likely
         than the input of this module to be required to be stored for the
         backprop.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,  # CAUTION: see documentation.
        log_scale: float = 1.0,
        log_scale_min: float = -1.5,
        log_scale_max: float = 1.5,
        store_output_for_backprop: bool = False,
    ) -> None:
        super(BiasNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.log_scale = nn.Parameter(torch.tensor(log_scale))
        self.bias = nn.Parameter(torch.empty(num_channels).normal_(mean=0, std=1e-4))

        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max

        self.store_output_for_backprop = store_output_for_backprop

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.channel_dim] == self.num_channels

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            channel_dim = self.channel_dim
            if channel_dim < 0:
                channel_dim += x.ndim
            bias = self.bias
            for _ in range(channel_dim + 1, x.ndim):
                bias = bias.unsqueeze(-1)
            scales = (torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5) * self.log_scale.exp()
            return x * scales

        log_scale = limit_param_value(
            self.log_scale,
            min=float(self.log_scale_min),
            max=float(self.log_scale_max),
            training=self.training,
        )

        return BiasNormFunction.apply(x, self.bias, log_scale, self.channel_dim, self.store_output_for_backprop)


def ScaledLinear(*args, initial_scale: float = 1.0, **kwargs) -> nn.Linear:
    """
    Behaves like a constructor of a modified version of nn.Linear
    that gives an easy way to set the default initial parameter scale.

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
    """
    ans = nn.Linear(*args, **kwargs)
    with torch.no_grad():
        ans.weight[:] *= initial_scale
        if ans.bias is not None:
            torch.nn.init.uniform_(ans.bias, -0.1 * initial_scale, 0.1 * initial_scale)
    return ans


def ScaledConv1d(*args, initial_scale: float = 1.0, **kwargs) -> nn.Conv1d:
    """
    Behaves like a constructor of a modified version of nn.Conv1d
    that gives an easy way to set the default initial parameter scale.

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
    """
    ans = nn.Conv1d(*args, **kwargs)
    with torch.no_grad():
        ans.weight[:] *= initial_scale
        if ans.bias is not None:
            torch.nn.init.uniform_(ans.bias, -0.1 * initial_scale, 0.1 * initial_scale)
    return ans


def ScaledConv2d(*args, initial_scale: float = 1.0, **kwargs) -> nn.Conv2d:
    """
    Behaves like a constructor of a modified version of nn.Conv2d
    that gives an easy way to set the default initial parameter scale.

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False, but:
    NO PADDING-RELATED ARGS.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
    """
    ans = nn.Conv2d(*args, **kwargs)
    with torch.no_grad():
        ans.weight[:] *= initial_scale
        if ans.bias is not None:
            torch.nn.init.uniform_(ans.bias, -0.1 * initial_scale, 0.1 * initial_scale)
    return ans


class ChunkCausalDepthwiseConv1d(torch.nn.Module):
    """
    Behaves like a depthwise 1d convolution, except that it is causal in
    a chunkwise way, as if we had a block-triangular attention mask.
    The chunk size is provided at test time (it should probably be
    kept in sync with the attention mask).

    This has a little more than twice the parameters of a conventional
    depthwise conv1d module: we implement it by having one
    depthwise convolution, of half the width, that is causal (via
    right-padding); and one depthwise convolution that is applied only
    within chunks, that we multiply by a scaling factor which depends
    on the position within the chunk.

    Args:
        Accepts the standard args and kwargs that nn.Linear accepts
        e.g. in_features, out_features, bias=False.

        initial_scale: you can override this if you want to increase
           or decrease the initial magnitude of the module's output
           (affects the initialization of weight_scale and bias_scale).
           Another option, if you want to do something like this, is
           to re-initialize the parameters.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        initial_scale: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        assert kernel_size % 2 == 1

        half_kernel_size = (kernel_size + 1) // 2
        # will pad manually, on one side.
        self.causal_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=half_kernel_size,
            padding=0,
            bias=True,
        )

        self.chunkwise_conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            groups=channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=bias,
        )

        # first row is correction factors added to the scale near the left edge of the chunk,
        # second row is correction factors added to the scale near the right edge of the chunk,
        # both of these are added to a default scale of 1.0.
        self.chunkwise_conv_scale = nn.Parameter(torch.zeros(2, channels, kernel_size))
        self.kernel_size = kernel_size

        with torch.no_grad():
            self.causal_conv.weight[:] *= initial_scale
            self.chunkwise_conv.weight[:] *= initial_scale
            if bias:
                torch.nn.init.uniform_(self.causal_conv.bias, -0.1 * initial_scale, 0.1 * initial_scale)

    def forward(self, x: Tensor, chunk_size: int = -1) -> Tensor:
        """Forward function.

        Args:
               x: a Tensor of shape (batch_size, channels, seq_len)
        chunk_size: the chunk size, in frames; does not have to divide seq_len exactly.
        """
        (batch_size, num_channels, seq_len) = x.shape

        # half_kernel_size = self.kernel_size + 1 // 2
        # left_pad is half_kernel_size - 1 where half_kernel_size is the size used
        # in the causal conv.  It's the amount by which we must pad on the left,
        # to make the convolution causal.
        left_pad = self.kernel_size // 2

        if chunk_size < 0 or chunk_size > seq_len:
            chunk_size = seq_len
        right_pad = -seq_len % chunk_size

        x = torch.nn.functional.pad(x, (left_pad, right_pad))

        x_causal = self.causal_conv(x[..., : left_pad + seq_len])
        assert x_causal.shape == (batch_size, num_channels, seq_len)

        x_chunk = x[..., left_pad:]
        num_chunks = x_chunk.shape[2] // chunk_size
        x_chunk = x_chunk.reshape(batch_size, num_channels, num_chunks, chunk_size)
        x_chunk = x_chunk.permute(0, 2, 1, 3).reshape(batch_size * num_chunks, num_channels, chunk_size)
        x_chunk = self.chunkwise_conv(x_chunk)  # does not change shape

        chunk_scale = self._get_chunk_scale(chunk_size)

        x_chunk = x_chunk * chunk_scale
        x_chunk = x_chunk.reshape(batch_size, num_chunks, num_channels, chunk_size).permute(0, 2, 1, 3)
        x_chunk = x_chunk.reshape(batch_size, num_channels, num_chunks * chunk_size)[..., :seq_len]

        return x_chunk + x_causal

    def _get_chunk_scale(self, chunk_size: int):
        """Returns tensor of shape (num_channels, chunk_size) that will be used to
        scale the output of self.chunkwise_conv."""
        left_edge = self.chunkwise_conv_scale[0]
        right_edge = self.chunkwise_conv_scale[1]
        if chunk_size < self.kernel_size:
            left_edge = left_edge[:, :chunk_size]
            right_edge = right_edge[:, -chunk_size:]
        else:
            t = chunk_size - self.kernel_size
            channels = left_edge.shape[0]
            pad = torch.zeros(channels, t, device=left_edge.device, dtype=left_edge.dtype)
            left_edge = torch.cat((left_edge, pad), dim=-1)
            right_edge = torch.cat((pad, right_edge), dim=-1)
        return 1.0 + (left_edge + right_edge)

    def streaming_forward(
        self,
        x: Tensor,
        cache: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Streaming Forward function.

        Args:
            x: a Tensor of shape (batch_size, channels, seq_len)
            cache: cached left context of shape (batch_size, channels, left_pad)
        """
        (batch_size, num_channels, seq_len) = x.shape

        # left_pad is half_kernel_size - 1 where half_kernel_size is the size used
        # in the causal conv.  It's the amount by which we must pad on the left,
        # to make the convolution causal.
        left_pad = self.kernel_size // 2

        # Pad cache
        assert cache.shape[-1] == left_pad, (cache.shape[-1], left_pad)
        x = torch.cat([cache, x], dim=2)
        # Update cache
        cache = x[..., -left_pad:]

        x_causal = self.causal_conv(x)
        assert x_causal.shape == (batch_size, num_channels, seq_len)

        x_chunk = x[..., left_pad:]
        x_chunk = self.chunkwise_conv(x_chunk)  # does not change shape

        chunk_scale = self._get_chunk_scale(chunk_size=seq_len)
        x_chunk = x_chunk * chunk_scale

        return x_chunk + x_causal, cache


class BalancerFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        min_mean: float,
        max_mean: float,
        min_rms: float,
        max_rms: float,
        grad_scale: float,
        channel_dim: int,
    ) -> Tensor:
        if channel_dim < 0:
            channel_dim += x.ndim
        ctx.channel_dim = channel_dim
        ctx.save_for_backward(x)
        ctx.config = (min_mean, max_mean, min_rms, max_rms, grad_scale, channel_dim)
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor) -> Tuple[Tensor, None, None, None, None, None]:
        (x,) = ctx.saved_tensors
        (min_mean, max_mean, min_rms, max_rms, grad_scale, channel_dim) = ctx.config

        try:
            with torch.enable_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x = x.to(torch.float32)
                    x = x.detach()
                    x.requires_grad = True
                    mean_dims = [i for i in range(x.ndim) if i != channel_dim]
                    uncentered_var = (x**2).mean(dim=mean_dims, keepdim=True)
                    mean = x.mean(dim=mean_dims, keepdim=True)
                    stddev = (uncentered_var - (mean * mean)).clamp(min=1.0e-20).sqrt()
                    rms = uncentered_var.clamp(min=1.0e-20).sqrt()

                    m = mean / stddev
                    # part of loss that relates to mean / stddev
                    m_loss = (m - m.clamp(min=min_mean, max=max_mean)).abs()

                    # put a much larger scale on the RMS-max-limit loss, so that if both it and the
                    # m_loss are violated we fix the RMS loss first.
                    rms_clamped = rms.clamp(min=min_rms, max=max_rms)
                    r_loss = (rms_clamped / rms).log().abs()

                    loss = m_loss + r_loss

                    loss.backward(gradient=torch.ones_like(loss))
                    loss_grad = x.grad
                    loss_grad_rms = (loss_grad**2).mean(dim=mean_dims, keepdim=True).sqrt().clamp(min=1.0e-20)

                    loss_grad = loss_grad * (grad_scale / loss_grad_rms)

                    x_grad_float = x_grad.to(torch.float32)
                    # scale each element of loss_grad by the absolute value of the corresponding
                    # element of x_grad, which we view as a noisy estimate of its magnitude for that
                    # (frame and dimension).  later we can consider factored versions.
                    x_grad_mod = x_grad_float + (x_grad_float.abs() * loss_grad)
                    x_grad = x_grad_mod.to(x_grad.dtype)
        except Exception as e:
            logging.info(f"Caught exception in Balancer backward: {e}, size={list(x_grad.shape)}, will continue.")

        return x_grad, None, None, None, None, None, None


class Balancer(torch.nn.Module):
    """
    Modifies the backpropped derivatives of a function to try to encourage, for
    each channel, that it is positive at least a proportion `threshold` of the
    time.  It does this by multiplying negative derivative values by up to
    (1+max_factor), and positive derivative values by up to (1-max_factor),
    interpolated from 1 at the threshold to those extremal values when none
    of the inputs are positive.

    Args:
           num_channels: the number of channels
           channel_dim: the dimension/axis corresponding to the channel, e.g.
               -1, 0, 1, 2; will be interpreted as an offset from x.ndim if negative.
           min_positive: the minimum, per channel, of the proportion of the time
               that (x > 0), below which we start to modify the derivatives.
           max_positive: the maximum, per channel, of the proportion of the time
               that (x > 0), above which we start to modify the derivatives.
           scale_gain_factor: determines the 'gain' with which we increase the
              change in gradient once the constraints on min_abs and max_abs
              are violated.
           min_abs:  the minimum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
           max_abs:  the maximum average-absolute-value difference from the mean
               value per channel, which we allow, before we start to modify
               the derivatives to prevent this.
         prob: determines the minimum probability with which we modify the
             gradients for the {min,max}_positive and {min,max}_abs constraints,
             on each forward().  This is done randomly to prevent all layers
             from doing it at the same time.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int,
        min_positive: FloatLike = 0.05,
        max_positive: FloatLike = 0.95,
        min_abs: FloatLike = 0.2,
        max_abs: FloatLike = 100.0,
        grad_scale: FloatLike = 0.04,
        prob: Optional[FloatLike] = None,
    ):
        super().__init__()

        if prob is None:
            prob = ScheduledFloat((0.0, 0.5), (8000.0, 0.125), default=0.4)
        self.prob = prob
        # 5% of the time we will return and do nothing because memory usage is
        # too high.
        self.mem_cutoff = CutoffEstimator(0.05)

        # actually self.num_channels is no longer needed except for an assertion.
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.min_positive = min_positive
        self.max_positive = max_positive
        self.min_abs = min_abs
        self.max_abs = max_abs
        self.grad_scale = grad_scale

    def forward(self, x: Tensor) -> Tensor:
        if (
            torch.jit.is_scripting()
            or not x.requires_grad
            or (x.is_cuda and self.mem_cutoff(torch.cuda.memory_allocated()))
        ):
            return _no_op(x)

        prob = float(self.prob)
        if random.random() < prob:
            # The following inner-functions convert from the way we historically specified
            # these limitations, as limits on the absolute value and the proportion of positive
            # values, to limits on the RMS value and the (mean / stddev).
            def _abs_to_rms(x):
                # for normally distributed data, if the expected absolute value is x, the
                # expected rms value will be sqrt(pi/2) * x.
                return 1.25331413732 * x

            def _proportion_positive_to_mean(x):
                def _atanh(x):
                    eps = 1.0e-10
                    # eps is to prevent crashes if x is exactly 0 or 1.
                    # we'll just end up returning a fairly large value.
                    return (math.log(1 + x + eps) - math.log(1 - x + eps)) / 2.0

                def _approx_inverse_erf(x):
                    # 1 / (sqrt(pi) * ln(2)),
                    # see https://math.stackexchange.com/questions/321569/approximating-the-error-function-erf-by-analytical-functions
                    # this approximation is extremely crude and gets progressively worse for
                    # x very close to -1 or +1, but we mostly care about the "middle" region
                    # e.g. _approx_inverse_erf(0.05) = 0.0407316414078772,
                    # and math.erf(0.0407316414078772) = 0.045935330944660666,
                    # which is pretty close to 0.05.
                    return 0.8139535143 * _atanh(x)

                # first convert x from the range 0..1 to the range -1..1 which the error
                # function returns
                x = -1 + (2 * x)
                return _approx_inverse_erf(x)

            min_mean = _proportion_positive_to_mean(float(self.min_positive))
            max_mean = _proportion_positive_to_mean(float(self.max_positive))
            min_rms = _abs_to_rms(float(self.min_abs))
            max_rms = _abs_to_rms(float(self.max_abs))
            grad_scale = float(self.grad_scale)

            assert x.shape[self.channel_dim] == self.num_channels

            return BalancerFunction.apply(x, min_mean, max_mean, min_rms, max_rms, grad_scale, self.channel_dim)
        else:
            return _no_op(x)


def penalize_abs_values_gt(x: Tensor, limit: float, penalty: float, name: str = None) -> Tensor:
    """
    Returns x unmodified, but in backprop will put a penalty for the excess of
    the absolute values of elements of x over the limit "limit".  E.g. if
    limit == 10.0, then if x has any values over 10 it will get a penalty.

    Caution: the value of this penalty will be affected by grad scaling used
    in automatic mixed precision training.  For this reasons we use this,
    it shouldn't really matter, or may even be helpful; we just use this
    to disallow really implausible values of scores to be given to softmax.

    The name is for randomly printed debug info.
    """
    x_sign = x.sign()
    over_limit = (x.abs() - limit) > 0
    # The following is a memory efficient way to penalize the absolute values of
    # x that's over the limit.  (The memory efficiency comes when you think
    # about which items torch needs to cache for the autograd, and which ones it
    # can throw away).  The numerical value of aux_loss as computed here will
    # actually be larger than it should be, by limit * over_limit.sum(), but it
    # has the same derivative as the real aux_loss which is penalty * (x.abs() -
    # limit).relu().
    aux_loss = penalty * ((x_sign * over_limit).to(torch.int8) * x)
    # note: we don't do sum() here on aux)_loss, but it's as if we had done
    # sum() due to how with_loss() works.
    x = with_loss(x, aux_loss, name)
    # you must use x for something, or this will be ineffective.
    return x


def _diag(x: Tensor):  # like .diag(), but works for tensors with 3 dims.
    if x.ndim == 2:
        return x.diag()
    else:
        (batch, dim, dim) = x.shape
        x = x.reshape(batch, dim * dim)
        x = x[:, :: dim + 1]
        assert x.shape == (batch, dim)
        return x


def _whitening_metric(x: Tensor, num_groups: int):
    """
    Computes the "whitening metric", a value which will be 1.0 if all the eigenvalues of
    of the centered feature covariance are the same within each group's covariance matrix
    and also between groups.
    Args:
        x: a Tensor of shape (*, num_channels)
     num_groups:  the number of groups of channels, a number >=1 that divides num_channels
    Returns:
        Returns a scalar Tensor that will be 1.0 if the data is "perfectly white" and
    greater than 1.0 otherwise.
    """
    assert x.dtype != torch.float16
    x = x.reshape(-1, x.shape[-1])
    (num_frames, num_channels) = x.shape
    assert num_channels % num_groups == 0
    channels_per_group = num_channels // num_groups
    x = x.reshape(num_frames, num_groups, channels_per_group).transpose(0, 1)
    # x now has shape (num_groups, num_frames, channels_per_group)
    # subtract the mean so we use the centered, not uncentered, covariance.
    # My experience has been that when we "mess with the gradients" like this,
    # it's better not do anything that tries to move the mean around, because
    # that can easily cause instability.
    x = x - x.mean(dim=1, keepdim=True)
    # x_covar: (num_groups, channels_per_group, channels_per_group)
    x_covar = torch.matmul(x.transpose(1, 2), x)
    x_covar_mean_diag = _diag(x_covar).mean()
    # the following expression is what we'd get if we took the matrix product
    # of each covariance and measured the mean of its trace, i.e.
    # the same as _diag(torch.matmul(x_covar, x_covar)).mean().
    x_covarsq_mean_diag = (x_covar**2).sum() / (num_groups * channels_per_group)
    # this metric will be >= 1.0; the larger it is, the less 'white' the data was.
    metric = x_covarsq_mean_diag / (x_covar_mean_diag**2 + 1.0e-20)
    return metric


class WhiteningPenaltyFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, module: nn.Module) -> Tensor:
        ctx.save_for_backward(x)
        ctx.module = module
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor):
        (x_orig,) = ctx.saved_tensors
        w = ctx.module

        try:
            with torch.enable_grad():
                with torch.cuda.amp.autocast(enabled=False):
                    x_detached = x_orig.to(torch.float32).detach()
                    x_detached.requires_grad = True

                    metric = _whitening_metric(x_detached, w.num_groups)

                    if random.random() < 0.005 or __name__ == "__main__":
                        logging.info(
                            f"Whitening: name={w.name}, num_groups={w.num_groups}, num_channels={x_orig.shape[-1]}, "
                            f"metric={metric.item():.2f} vs. limit={float(w.whitening_limit)}"
                        )

                    if metric < float(w.whitening_limit):
                        w.prob = w.min_prob
                        return x_grad, None
                    else:
                        w.prob = w.max_prob
                        metric.backward()
                        penalty_grad = x_detached.grad
                        scale = float(w.grad_scale) * (
                            x_grad.to(torch.float32).norm() / (penalty_grad.norm() + 1.0e-20)
                        )
                        penalty_grad = penalty_grad * scale
                        return x_grad + penalty_grad.to(x_grad.dtype), None
        except Exception as e:
            logging.info(f"Caught exception in Whiten backward: {e}, size={list(x_grad.shape)}, will continue.")
        return x_grad, None


class Whiten(nn.Module):
    def __init__(
        self,
        num_groups: int,
        whitening_limit: FloatLike,
        prob: Union[float, Tuple[float, float]],
        grad_scale: FloatLike,
    ):
        """
        Args:
          num_groups: the number of groups to divide the channel dim into before
            whitening.  We will attempt to make the feature covariance
            within each group, after mean subtraction, as "white" as possible,
            while having the same trace across all groups.
         whitening_limit: a value greater than 1.0, that dictates how much
           freedom we have to violate the constraints.  1.0 would mean perfectly
           white, with exactly the same trace across groups; larger values
           give more freedom.  E.g. 2.0.
         prob: the probability with which we apply the gradient modification
           (also affects the grad scale).  May be supplied as a float,
           or as a pair (min_prob, max_prob)

          grad_scale: determines the scale on the gradient term from this object,
            relative to the rest of the gradient on the attention weights.
            E.g. 0.02 (you may want to use smaller values than this if prob is large)
        """
        super(Whiten, self).__init__()
        assert num_groups >= 1
        assert float(whitening_limit) >= 1
        assert float(grad_scale) >= 0
        self.num_groups = num_groups
        self.whitening_limit = whitening_limit
        self.grad_scale = grad_scale

        if isinstance(prob, float):
            prob = (prob, prob)
        (self.min_prob, self.max_prob) = prob
        assert 0 < self.min_prob <= self.max_prob <= 1
        self.prob = self.max_prob
        self.name = None  # will be set in training loop

    def forward(self, x: Tensor) -> Tensor:
        """
        In the forward pass, this function just returns the input unmodified.
        In the backward pass, it will modify the gradients to ensure that the
        distribution in each group has close to (lambda times I) as the covariance
        after mean subtraction, with the same lambda across groups.
        For whitening_limit > 1, there will be more freedom to violate this
        constraint.

        Args:
           x: the input of shape (*, num_channels)

        Returns:
            x, unmodified.   You should make sure
        you use the returned value, or the graph will be freed
        and nothing will happen in backprop.
        """
        grad_scale = float(self.grad_scale)
        if not x.requires_grad or random.random() > self.prob or grad_scale == 0:
            return _no_op(x)
        else:
            return WhiteningPenaltyFunction.apply(x, self)


class WithLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor, name: str):
        ctx.y_shape = y.shape
        if random.random() < 0.002 and name is not None:
            loss_sum = y.sum().item()
            logging.info(f"WithLoss: name={name}, loss-sum={loss_sum:.3e}")
        return x

    @staticmethod
    def backward(ctx, ans_grad: Tensor):
        return (
            ans_grad,
            torch.ones(ctx.y_shape, dtype=ans_grad.dtype, device=ans_grad.device),
            None,
        )


def with_loss(x, y, name):
    # returns x but adds y.sum() to the loss function.
    return WithLoss.apply(x, y, name)


class ScaleGradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad: Tensor):
        return grad * ctx.alpha, None


def scale_grad(x: Tensor, alpha: float):
    return ScaleGradFunction.apply(x, alpha)


class ScaleGrad(nn.Module):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        if torch.jit.is_scripting() or torch.jit.is_tracing() or not self.training:
            return x
        return scale_grad(x, self.alpha)


class LimitParamValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, min: float, max: float):
        ctx.save_for_backward(x)
        assert max >= min
        ctx.min = min
        ctx.max = max
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor):
        (x,) = ctx.saved_tensors
        # where x < ctx.min, ensure all grads are negative (this will tend to make
        # x more positive).
        x_grad = x_grad * torch.where(torch.logical_and(x_grad > 0, x < ctx.min), -1.0, 1.0)
        # where x > ctx.max, ensure all grads are positive (this will tend to make
        # x more negative).
        x_grad *= torch.where(torch.logical_and(x_grad < 0, x > ctx.max), -1.0, 1.0)
        return x_grad, None, None


def limit_param_value(x: Tensor, min: float, max: float, prob: float = 0.6, training: bool = True):
    # You apply this to (typically) an nn.Parameter during training to ensure that its
    # (elements mostly) stays within a supplied range.  This is done by modifying the
    # gradients in backprop.
    # It's not necessary to do this on every batch: do it only some of the time,
    # to save a little time.
    if training and random.random() < prob:
        return LimitParamValue.apply(x, min, max)
    else:
        return x


def _no_op(x: Tensor) -> Tensor:
    if torch.jit.is_scripting() or torch.jit.is_tracing():
        return x
    else:
        # a no-op function that will have a node in the autograd graph,
        # to avoid certain bugs relating to backward hooks
        return x.chunk(1, dim=-1)[0]


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return _no_op(x)


class DoubleSwishFunction(torch.autograd.Function):
    """
      double_swish(x) = x * torch.sigmoid(x-1)

    This is a definition, originally motivated by its close numerical
    similarity to swish(swish(x)), where swish(x) =  x * sigmoid(x).

    Memory-efficient derivative computation:
     double_swish(x) = x * s, where s(x) = torch.sigmoid(x-1)
     double_swish'(x) = d/dx double_swish(x) =  x * s'(x) + x' * s(x) = x * s'(x) + s(x).
     Now, s'(x) = s(x) * (1-s(x)).
     double_swish'(x) =  x * s'(x) + s(x).
                      =  x * s(x) * (1-s(x)) + s(x).
                     = double_swish(x) * (1-s(x)) + s(x)
     ... so we just need to remember s(x) but not x itself.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        requires_grad = x.requires_grad
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.to(torch.float32)

        s = torch.sigmoid(x - 1.0)
        y = x * s

        if requires_grad:
            deriv = y * (1 - s) + s

            # notes on derivative of x * sigmoid(x - 1):
            # https://www.wolframalpha.com/input?i=d%2Fdx+%28x+*+sigmoid%28x-1%29%29
            # min \simeq -0.043638.  Take floor as -0.044 so it's a lower bund
            # max \simeq 1.1990.   Take ceil to be 1.2 so it's an upper bound.
            # the combination of "+ torch.rand_like(deriv)" and casting to torch.uint8 (which
            # floors), should be expectation-preserving.
            floor = -0.044
            ceil = 1.2
            d_scaled = (deriv - floor) * (255.0 / (ceil - floor)) + torch.rand_like(deriv)
            if __name__ == "__main__":
                # for self-testing only.
                assert d_scaled.min() >= 0.0
                assert d_scaled.max() < 256.0
            d_int = d_scaled.to(torch.uint8)
            ctx.save_for_backward(d_int)
        if x.dtype == torch.float16 or torch.is_autocast_enabled():
            y = y.to(torch.float16)
        return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        (d,) = ctx.saved_tensors
        # the same constants as used in forward pass.
        floor = -0.043637
        ceil = 1.2

        d = d * ((ceil - floor) / 255.0) + floor
        return y_grad * d


class DoubleSwish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """Return double-swish activation function which is an approximation to Swish(Swish(x)),
        that we approximate closely with x * sigmoid(x-1).
        """
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            return x * torch.sigmoid(x - 1.0)
        return DoubleSwishFunction.apply(x)


# Dropout2 is just like normal dropout, except it supports schedules on the dropout rates.
class Dropout2(nn.Module):
    def __init__(self, p: FloatLike):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.dropout(x, p=float(self.p), training=self.training)


class MulForDropout3(torch.autograd.Function):
    # returns (x * y * alpha) where alpha is a float and y doesn't require
    # grad and is zero-or-one.
    @staticmethod
    @custom_fwd
    def forward(ctx, x, y, alpha):
        assert not y.requires_grad
        ans = x * y * alpha
        ctx.save_for_backward(ans)
        ctx.alpha = alpha
        return ans

    @staticmethod
    @custom_bwd
    def backward(ctx, ans_grad):
        (ans,) = ctx.saved_tensors
        x_grad = ctx.alpha * ans_grad * (ans != 0)
        return x_grad, None, None


# Dropout3 is just like normal dropout, except it supports schedules on the dropout rates,
# and it lets you choose one dimension to share the dropout mask over
class Dropout3(nn.Module):
    def __init__(self, p: FloatLike, shared_dim: int):
        super().__init__()
        self.p = p
        self.shared_dim = shared_dim

    def forward(self, x: Tensor) -> Tensor:
        p = float(self.p)
        if not self.training or p == 0:
            return _no_op(x)
        scale = 1.0 / (1 - p)
        rand_shape = list(x.shape)
        rand_shape[self.shared_dim] = 1
        mask = torch.rand(*rand_shape, device=x.device) > p
        ans = MulForDropout3.apply(x, mask, scale)
        return ans


class SwooshLFunction(torch.autograd.Function):
    """
    swoosh_l(x) =  log(1 + exp(x-4)) - 0.08*x - 0.035
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        requires_grad = x.requires_grad
        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.to(torch.float32)

        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)

        coeff = -0.08

        with torch.cuda.amp.autocast(enabled=False):
            with torch.enable_grad():
                x = x.detach()
                x.requires_grad = True
                y = torch.logaddexp(zero, x - 4.0) + coeff * x - 0.035

                if not requires_grad:
                    return y

                y.backward(gradient=torch.ones_like(y))

                grad = x.grad
                floor = coeff
                ceil = 1.0 + coeff + 0.005

                d_scaled = (grad - floor) * (255.0 / (ceil - floor)) + torch.rand_like(grad)
                if __name__ == "__main__":
                    # for self-testing only.
                    assert d_scaled.min() >= 0.0
                    assert d_scaled.max() < 256.0

                d_int = d_scaled.to(torch.uint8)
                ctx.save_for_backward(d_int)
                if x.dtype == torch.float16 or torch.is_autocast_enabled():
                    y = y.to(torch.get_autocast_gpu_dtype())
                return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        (d,) = ctx.saved_tensors
        # the same constants as used in forward pass.

        coeff = -0.08
        floor = coeff
        ceil = 1.0 + coeff + 0.005
        d = d * ((ceil - floor) / 255.0) + floor
        return y_grad * d


class SwooshL(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-L activation."""
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
            return logaddexp(zero, x - 4.0) - 0.08 * x - 0.035
        import k2

        if not x.requires_grad:
            return k2.swoosh_l_forward(x)
        else:
            return k2.swoosh_l(x)
        # return SwooshLFunction.apply(x)


class SwooshLOnnx(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-L activation."""
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return logaddexp_onnx(zero, x - 4.0) - 0.08 * x - 0.035


class SwooshRFunction(torch.autograd.Function):
    """
     swoosh_r(x) =  log(1 + exp(x-1)) - 0.08*x - 0.313261687

    derivatives are between -0.08 and 0.92.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        requires_grad = x.requires_grad

        if x.dtype == torch.float16 or x.dtype == torch.bfloat16:
            x = x.to(torch.float32)

        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)

        with torch.cuda.amp.autocast(enabled=False):
            with torch.enable_grad():
                x = x.detach()
                x.requires_grad = True
                y = torch.logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687

                if not requires_grad:
                    return y
                y.backward(gradient=torch.ones_like(y))

                grad = x.grad
                floor = -0.08
                ceil = 0.925

                d_scaled = (grad - floor) * (255.0 / (ceil - floor)) + torch.rand_like(grad)
                if __name__ == "__main__":
                    # for self-testing only.
                    assert d_scaled.min() >= 0.0
                    assert d_scaled.max() < 256.0

                d_int = d_scaled.to(torch.uint8)
                ctx.save_for_backward(d_int)
                if x.dtype == torch.float16 or torch.is_autocast_enabled():
                    y = y.to(torch.get_autocast_gpu_dtype())
                return y

    @staticmethod
    def backward(ctx, y_grad: Tensor) -> Tensor:
        (d,) = ctx.saved_tensors
        # the same constants as used in forward pass.
        floor = -0.08
        ceil = 0.925
        d = d * ((ceil - floor) / 255.0) + floor
        return y_grad * d


class SwooshR(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-R activation."""
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
            return logaddexp(zero, x - 1.0) - 0.08 * x - 0.313261687
        import k2

        if not x.requires_grad:
            return k2.swoosh_r_forward(x)
        else:
            return k2.swoosh_r(x)
        # return SwooshRFunction.apply(x)


class SwooshROnnx(torch.nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """Return Swoosh-R activation."""
        zero = torch.tensor(0.0, dtype=x.dtype, device=x.device)
        return logaddexp_onnx(zero, x - 1.0) - 0.08 * x - 0.313261687


# simple version of SwooshL that does not redefine the backprop, used in
# ActivationDropoutAndLinearFunction.
def SwooshLForward(x: Tensor):
    x_offset = x - 4.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)
    return log_sum - 0.08 * x - 0.035


# simple version of SwooshR that does not redefine the backprop, used in
# ActivationDropoutAndLinearFunction.
def SwooshRForward(x: Tensor):
    x_offset = x - 1.0
    log_sum = (1.0 + x_offset.exp()).log().to(x.dtype)
    log_sum = torch.where(log_sum == float("inf"), x_offset, log_sum)
    return log_sum - 0.08 * x - 0.313261687


class ActivationDropoutAndLinearFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
        activation: str,
        dropout_p: float,
        dropout_shared_dim: Optional[int],
    ):
        if dropout_p != 0.0:
            dropout_shape = list(x.shape)
            if dropout_shared_dim is not None:
                dropout_shape[dropout_shared_dim] = 1
            # else it won't be very memory efficient.
            dropout_mask = (1.0 / (1.0 - dropout_p)) * (
                torch.rand(*dropout_shape, device=x.device, dtype=x.dtype) > dropout_p
            )
        else:
            dropout_mask = None

        ctx.save_for_backward(x, weight, bias, dropout_mask)

        ctx.activation = activation

        import k2

        forward_activation_dict = {
            "SwooshL": k2.swoosh_l_forward,
            "SwooshR": k2.swoosh_r_forward,
        }
        # it will raise a KeyError if this fails.  This will be an error.  We let it
        # propagate to the user.
        activation_func = forward_activation_dict[activation]
        x = activation_func(x)
        if dropout_mask is not None:
            x = x * dropout_mask
        x = torch.nn.functional.linear(x, weight, bias)
        return x

    @staticmethod
    @custom_bwd
    def backward(ctx, ans_grad: Tensor):
        saved = ctx.saved_tensors
        (x, weight, bias, dropout_mask) = saved

        import k2

        forward_and_deriv_activation_dict = {
            "SwooshL": k2.swoosh_l_forward_and_deriv,
            "SwooshR": k2.swoosh_r_forward_and_deriv,
        }
        # the following lines a KeyError if the activation is unrecognized.
        # This will be an error.  We let it propagate to the user.
        func = forward_and_deriv_activation_dict[ctx.activation]

        y, func_deriv = func(x)
        if dropout_mask is not None:
            y = y * dropout_mask
        # now compute derivative of y w.r.t. weight and bias..
        # y: (..., in_channels), ans_grad: (..., out_channels),
        (out_channels, in_channels) = weight.shape

        in_channels = y.shape[-1]
        g = ans_grad.reshape(-1, out_channels)
        weight_deriv = torch.matmul(g.t(), y.reshape(-1, in_channels))
        y_deriv = torch.matmul(ans_grad, weight)
        bias_deriv = None if bias is None else g.sum(dim=0)
        x_deriv = y_deriv * func_deriv
        if dropout_mask is not None:
            # order versus func_deriv does not matter
            x_deriv = x_deriv * dropout_mask

        return x_deriv, weight_deriv, bias_deriv, None, None, None


class ActivationDropoutAndLinear(torch.nn.Module):
    """
     This merges an activation function followed by dropout and then a nn.Linear module;
     it does so in a memory efficient way so that it only stores the input to the whole
     module.  If activation == SwooshL and dropout_shared_dim != None, this will be
     equivalent to:
       nn.Sequential(SwooshL(),
                     Dropout3(dropout_p, shared_dim=dropout_shared_dim),
                     ScaledLinear(in_channels, out_channels, bias=bias,
                                  initial_scale=initial_scale))
    If dropout_shared_dim is None, the dropout would be equivalent to
    Dropout2(dropout_p).  Note: Dropout3 will be more memory efficient as the dropout
    mask is smaller.

     Args:
        in_channels: number of input channels, e.g. 256
        out_channels: number of output channels, e.g. 256
        bias: if true, have a bias
        activation: the activation function, for now just support SwooshL.
        dropout_p: the dropout probability or schedule (happens after nonlinearity).
        dropout_shared_dim: the dimension, if any, across which the dropout mask is
             shared (e.g. the time dimension).  If None, this may be less memory
             efficient if there are modules before this one that cache the input
             for their backprop (e.g. Balancer or Whiten).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        activation: str = "SwooshL",
        dropout_p: FloatLike = 0.0,
        dropout_shared_dim: Optional[int] = -1,
        initial_scale: float = 1.0,
    ):
        super().__init__()
        # create a temporary module of nn.Linear that we'll steal the
        # weights and bias from
        l = ScaledLinear(in_channels, out_channels, bias=bias, initial_scale=initial_scale)

        self.weight = l.weight
        # register_parameter properly handles making it a parameter when l.bias
        # is None. I think there is some reason for doing it this way rather
        # than just setting it to None but I don't know what it is, maybe
        # something to do with exporting the module..
        self.register_parameter("bias", l.bias)

        self.activation = activation
        self.dropout_p = dropout_p
        self.dropout_shared_dim = dropout_shared_dim

    def forward(self, x: Tensor):
        if not self.training or torch.jit.is_scripting() or torch.jit.is_tracing():
            if self.activation == "SwooshL":
                x = SwooshLForward(x)
            elif self.activation == "SwooshR":
                x = SwooshRForward(x)
            else:
                assert False, self.activation
            return torch.nn.functional.linear(x, self.weight, self.bias)

        return ActivationDropoutAndLinearFunction.apply(
            x,
            self.weight,
            self.bias,
            self.activation,
            float(self.dropout_p),
            self.dropout_shared_dim,
        )


def convert_num_channels(x: Tensor, num_channels: int) -> Tensor:
    if num_channels <= x.shape[-1]:
        return x[..., :num_channels]
    else:
        shape = list(x.shape)
        shape[-1] = num_channels - shape[-1]
        zeros = torch.zeros(shape, dtype=x.dtype, device=x.device)
        return torch.cat((x, zeros), dim=-1)


def _test_whiten():
    for proportion in [0.1, 0.5, 10.0]:
        logging.info(f"_test_whiten(): proportion = {proportion}")
        x = torch.randn(100, 128)
        direction = torch.randn(128)
        coeffs = torch.randn(100, 1)
        x += proportion * direction * coeffs

        x.requires_grad = True

        m = Whiten(1, 5.0, prob=1.0, grad_scale=0.1)  # num_groups  # whitening_limit,  # grad_scale

        for _ in range(4):
            y = m(x)

        y_grad = torch.randn_like(x)
        y.backward(gradient=y_grad)

        if proportion < 0.2:
            assert torch.allclose(x.grad, y_grad)
        elif proportion > 1.0:
            assert not torch.allclose(x.grad, y_grad)


def _test_balancer_sign():
    probs = torch.arange(0, 1, 0.01)
    N = 1000
    x = 1.0 * ((2.0 * (torch.rand(probs.numel(), N) < probs.unsqueeze(-1))) - 1.0)
    x = x.detach()
    x.requires_grad = True
    m = Balancer(
        probs.numel(),
        channel_dim=0,
        min_positive=0.05,
        max_positive=0.95,
        min_abs=0.0,
        prob=1.0,
    )

    y_grad = torch.sign(torch.randn(probs.numel(), N))

    y = m(x)
    y.backward(gradient=y_grad)
    print("_test_balancer_sign: x = ", x)
    print("_test_balancer_sign: y grad = ", y_grad)
    print("_test_balancer_sign: x grad = ", x.grad)


def _test_balancer_magnitude():
    magnitudes = torch.arange(0, 1, 0.01)
    N = 1000
    x = torch.sign(torch.randn(magnitudes.numel(), N)) * magnitudes.unsqueeze(-1)
    x = x.detach()
    x.requires_grad = True
    m = Balancer(
        magnitudes.numel(),
        channel_dim=0,
        min_positive=0.0,
        max_positive=1.0,
        min_abs=0.2,
        max_abs=0.7,
        prob=1.0,
    )

    y_grad = torch.sign(torch.randn(magnitudes.numel(), N))

    y = m(x)
    y.backward(gradient=y_grad)
    print("_test_balancer_magnitude: x = ", x)
    print("_test_balancer_magnitude: y grad = ", y_grad)
    print("_test_balancer_magnitude: x grad = ", x.grad)


def _test_double_swish_deriv():
    x = torch.randn(10, 12, dtype=torch.double) * 3.0
    x.requires_grad = True
    m = DoubleSwish()

    tol = (1.2 - (-0.043637)) / 255.0
    torch.autograd.gradcheck(m, x, atol=tol)

    # for self-test.
    x = torch.randn(1000, 1000, dtype=torch.double) * 3.0
    x.requires_grad = True
    y = m(x)


def _test_swooshl_deriv():
    x = torch.randn(10, 12, dtype=torch.double) * 3.0
    x.requires_grad = True
    m = SwooshL()

    tol = 1.0 / 255.0
    torch.autograd.gradcheck(m, x, atol=tol, eps=0.01)

    # for self-test.
    x = torch.randn(1000, 1000, dtype=torch.double) * 3.0
    x.requires_grad = True
    y = m(x)


def _test_swooshr_deriv():
    x = torch.randn(10, 12, dtype=torch.double) * 3.0
    x.requires_grad = True
    m = SwooshR()

    tol = 1.0 / 255.0
    torch.autograd.gradcheck(m, x, atol=tol, eps=0.01)

    # for self-test.
    x = torch.randn(1000, 1000, dtype=torch.double) * 3.0
    x.requires_grad = True
    y = m(x)


def _test_softmax():
    a = torch.randn(2, 10, dtype=torch.float64)
    b = a.clone()
    a.requires_grad = True
    b.requires_grad = True
    a.softmax(dim=1)[:, 0].sum().backward()
    print("a grad = ", a.grad)
    softmax(b, dim=1)[:, 0].sum().backward()
    print("b grad = ", b.grad)
    assert torch.allclose(a.grad, b.grad)


def _test_piecewise_linear():
    p = PiecewiseLinear((0, 10.0))
    for x in [-100, 0, 100]:
        assert p(x) == 10.0
    p = PiecewiseLinear((0, 10.0), (1, 0.0))
    for x, y in [(-100, 10.0), (0, 10.0), (0.5, 5.0), (1, 0.0), (2, 0.0)]:
        print("x, y = ", x, y)
        assert p(x) == y, (x, p(x), y)

    q = PiecewiseLinear((0.5, 15.0), (0.6, 1.0))
    x_vals = [-1.0, 0.0, 0.1, 0.2, 0.5, 0.6, 0.7, 0.9, 1.0, 2.0]
    pq = p.max(q)
    for x in x_vals:
        y1 = max(p(x), q(x))
        y2 = pq(x)
        assert abs(y1 - y2) < 0.001
    pq = p.min(q)
    for x in x_vals:
        y1 = min(p(x), q(x))
        y2 = pq(x)
        assert abs(y1 - y2) < 0.001
    pq = p + q
    for x in x_vals:
        y1 = p(x) + q(x)
        y2 = pq(x)
        assert abs(y1 - y2) < 0.001


def _test_activation_dropout_and_linear():
    in_channels = 20
    out_channels = 30

    for bias in [True, False]:
        # actually we don't test for dropout_p != 0.0 because forward functions will give
        # different answers.  This is because we are using the k2 implementation of
        # swoosh_l an swoosh_r inside SwooshL() and SwooshR(), and they call randn()
        # internally, messing up the random state.
        for dropout_p in [0.0]:
            for activation in ["SwooshL", "SwooshR"]:
                m1 = nn.Sequential(
                    SwooshL() if activation == "SwooshL" else SwooshR(),
                    Dropout3(p=dropout_p, shared_dim=-1),
                    ScaledLinear(in_channels, out_channels, bias=bias, initial_scale=0.5),
                )
                m2 = ActivationDropoutAndLinear(
                    in_channels,
                    out_channels,
                    bias=bias,
                    initial_scale=0.5,
                    activation=activation,
                    dropout_p=dropout_p,
                )
                with torch.no_grad():
                    m2.weight[:] = m1[2].weight
                    if bias:
                        m2.bias[:] = m1[2].bias
                # make sure forward gives same result.
                x1 = torch.randn(10, in_channels)
                x1.requires_grad = True

                # TEMP.
                assert torch.allclose(SwooshRFunction.apply(x1), SwooshRForward(x1), atol=1.0e-03)

                x2 = x1.clone().detach()
                x2.requires_grad = True
                seed = 10
                torch.manual_seed(seed)
                y1 = m1(x1)
                y_grad = torch.randn_like(y1)
                y1.backward(gradient=y_grad)
                torch.manual_seed(seed)
                y2 = m2(x2)
                y2.backward(gradient=y_grad)

                print(f"bias = {bias}, dropout_p = {dropout_p}, activation = {activation}")
                print("y1 = ", y1)
                print("y2 = ", y2)
                assert torch.allclose(y1, y2, atol=0.02)
                assert torch.allclose(m1[2].weight.grad, m2.weight.grad, atol=1.0e-05)
                if bias:
                    assert torch.allclose(m1[2].bias.grad, m2.bias.grad, atol=1.0e-05)
                print("x1.grad = ", x1.grad)
                print("x2.grad = ", x2.grad)

                def isclose(a, b):
                    # return true if cosine similarity is > 0.9.
                    return (a * b).sum() > 0.9 * ((a**2).sum() * (b**2).sum()).sqrt()

                # the SwooshL() implementation has a noisy gradient due to 1-byte
                # storage of it.
                assert isclose(x1.grad, x2.grad)


class Zipformer2(EncoderInterface):
    """
    Args:

    Note: all "int or Tuple[int, ...]" arguments below will be treated as lists of the same length
    as downsampling_factor if they are single ints or one-element tuples.  The length of
    downsampling_factor defines the number of stacks.

        output_downsampling_factor (int): how much to downsample at the output.  Note:
            we also downsample by a factor of 2 in the Conv2dSubsampling encoder.
            You should probably leave this at 2.
        downsampling_factor (Tuple[int, ...]): downsampling factor for each encoder stack.
           Note: this is in addition to the downsampling factor of 2 that is applied in
           the frontend (self.encoder_embed).
        encoder_dim (Tuple[int, ...]): embedding dimension of each of the encoder stacks, one per
           encoder stack.
        num_encoder_layers (int or Tuple[int, ...])): number of encoder layers for each stack
        encoder_unmasked_dim (int or Tuple[int, ...]): unmasked dimension in each of
            the encoder stacks for purposes of per-frame dropout (recommend 256 for
            now).
        query_head_dim (int or Tuple[int, ...]): dimension of query and key per attention
           head: per stack, if a tuple..
        pos_head_dim (int or Tuple[int, ...]): dimension of positional-encoding projection per
           attention head
        value_head_dim (int or Tuple[int, ...]): dimension of value in each attention head
        num_heads: (int or Tuple[int, ...]): number of heads in the self-attention mechanism.
              Must be at least 4.
        feedforward_dim (int or Tuple[int, ...]): hidden dimension in feedforward modules
        cnn_module_kernel (int or Tuple[int, ...])): Kernel size of convolution module

        pos_dim (int): the dimension of each positional-encoding vector prior to projection,
            e.g. 128.

        dropout (float): dropout rate
        warmup_batches (float): number of batches to warm up over; this controls
          dropout of encoder layers.
        causal (bool): if True, support chunkwise causal convolution.  This should
          not hurt WER as no modeling power is lost, but the convolution modules will be
          slightly slower and use more memory.  Enables use of the chunk_size and
          left_context_chunks options in forward(), which simulates streaming
          decoding.
        chunk_size: (list of int): only set this to other than [-1] if causal;
           the chunk size will be randomly chosen from this list.  -1 means no chunking.
        left_context_frames: (list of int): determines the number of left-
           context chunks for causal training; will be rounded to a number of
           chunks.  Must not be less than cnn_module_kernel (after factoring in
           rounding and downsampling); an error will be thrown if this is violated.
    """

    def __init__(
        self,
        output_downsampling_factor: int = 2,
        downsampling_factor: Tuple[int, ...] = (2, 4),
        encoder_dim: Union[int, Tuple[int, ...]] = 384,
        num_encoder_layers: Union[int, Tuple[int, ...]] = 4,
        encoder_unmasked_dim: Union[int, Tuple[int, ...]] = 256,
        query_head_dim: Union[int, Tuple[int, ...]] = 24,
        pos_head_dim: Union[int, Tuple[int, ...]] = 4,
        value_head_dim: Union[int, Tuple[int, ...]] = 12,
        num_heads: Union[int, Tuple[int, ...]] = 8,
        feedforward_dim: Union[int, Tuple[int, ...]] = 1536,
        cnn_module_kernel: Union[int, Tuple[int, ...]] = 31,
        pos_dim: int = 192,
        dropout: FloatLike = None,  # see code below for default
        warmup_batches: float = 4000.0,
        causal: bool = False,
        chunk_size: Tuple[int, ...] = (-1,),
        left_context_frames: Tuple[int, ...] = (-1,),
    ) -> None:
        super(Zipformer2, self).__init__()

        if dropout is None:
            dropout = ScheduledFloat((0.0, 0.3), (20000.0, 0.1))

        def _to_tuple(x):
            """Converts a single int or a 1-tuple of an int to a tuple with the same length
            as downsampling_factor"""
            if isinstance(x, int):
                x = (x,)
            if len(x) == 1:
                x = x * len(downsampling_factor)
            else:
                assert len(x) == len(downsampling_factor) and isinstance(x[0], int)
            return x

        self.output_downsampling_factor = output_downsampling_factor  # int
        self.downsampling_factor = downsampling_factor  # tuple
        self.encoder_dim = encoder_dim = _to_tuple(encoder_dim)  # tuple
        self.encoder_unmasked_dim = encoder_unmasked_dim = _to_tuple(encoder_unmasked_dim)  # tuple
        num_encoder_layers = _to_tuple(num_encoder_layers)
        self.num_encoder_layers = num_encoder_layers
        self.query_head_dim = query_head_dim = _to_tuple(query_head_dim)
        self.value_head_dim = value_head_dim = _to_tuple(value_head_dim)
        pos_head_dim = _to_tuple(pos_head_dim)
        self.num_heads = num_heads = _to_tuple(num_heads)
        feedforward_dim = _to_tuple(feedforward_dim)
        self.cnn_module_kernel = cnn_module_kernel = _to_tuple(cnn_module_kernel)

        self.causal = causal
        self.chunk_size = chunk_size
        self.left_context_frames = left_context_frames

        for u, d in zip(encoder_unmasked_dim, encoder_dim):
            assert u <= d

        # each one will be Zipformer2Encoder or DownsampledZipformer2Encoder
        encoders = []

        num_encoders = len(downsampling_factor)
        for i in range(num_encoders):
            encoder_layer = Zipformer2EncoderLayer(
                embed_dim=encoder_dim[i],
                pos_dim=pos_dim,
                num_heads=num_heads[i],
                query_head_dim=query_head_dim[i],
                pos_head_dim=pos_head_dim[i],
                value_head_dim=value_head_dim[i],
                feedforward_dim=feedforward_dim[i],
                dropout=dropout,
                cnn_module_kernel=cnn_module_kernel[i],
                causal=causal,
            )

            # For the segment of the warmup period, we let the Conv2dSubsampling
            # layer learn something.  Then we start to warm up the other encoders.
            encoder = Zipformer2Encoder(
                encoder_layer,
                num_encoder_layers[i],
                pos_dim=pos_dim,
                dropout=dropout,
                warmup_begin=warmup_batches * (i + 1) / (num_encoders + 1),
                warmup_end=warmup_batches * (i + 2) / (num_encoders + 1),
                final_layerdrop_rate=0.035 * (downsampling_factor[i] ** 0.5),
            )

            if downsampling_factor[i] != 1:
                encoder = DownsampledZipformer2Encoder(
                    encoder,
                    dim=encoder_dim[i],
                    downsample=downsampling_factor[i],
                    dropout=dropout,
                    causal=causal,
                )

            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)

        self.downsample_output = SimpleDownsample(
            max(encoder_dim),
            downsample=output_downsampling_factor,
            dropout=dropout,
            causal=causal,
        )

    def get_feature_masks(self, x: Tensor) -> Union[List[float], List[Tensor]]:
        """
        In eval mode, returns [1.0] * num_encoders; in training mode, returns a number of
        randomized feature masks, one per encoder.
        On e.g. 15% of frames, these masks will zero out all encoder dims larger than
        some supplied number, e.g. >256, so in effect on those frames we are using
        a smaller encoder dim.

        We generate the random masks at this level because we want the 2 masks to 'agree'
        all the way up the encoder stack. This will mean that the 1st mask will have
        mask values repeated self.zipformer_subsampling_factor times.

        Args:
           x: the embeddings (needed for the shape and dtype and device), of shape
             (1, batch_size, encoder_dims0)
        """
        num_encoders = len(self.encoder_dim)
        if not self.training:
            return [1.0] * num_encoders

        (num_frames0, batch_size, _encoder_dims0) = x.shape

        assert self.encoder_dim[0] == _encoder_dims0, (
            self.encoder_dim[0],
            _encoder_dims0,
        )

        feature_mask_dropout_prob = 0.125

        # mask1 shape: (1, batch_size, 1)
        mask1 = (torch.rand(1, batch_size, 1, device=x.device) > feature_mask_dropout_prob).to(x.dtype)

        # mask2 has additional sequences masked, about twice the number.
        mask2 = torch.logical_and(
            mask1,
            (torch.rand(1, batch_size, 1, device=x.device) > feature_mask_dropout_prob).to(x.dtype),
        )

        # dim: (1, batch_size, 2)
        mask = torch.cat((mask1, mask2), dim=-1)

        feature_masks = []
        for i in range(num_encoders):
            channels = self.encoder_dim[i]
            feature_mask = torch.ones(1, batch_size, channels, dtype=x.dtype, device=x.device)
            u1 = self.encoder_unmasked_dim[i]
            u2 = u1 + (channels - u1) // 2

            feature_mask[:, :, u1:u2] *= mask[..., 0:1]
            feature_mask[:, :, u2:] *= mask[..., 1:2]

            feature_masks.append(feature_mask)

        return feature_masks

    def get_chunk_info(self) -> Tuple[int, int]:
        """
        Returns chunk_size and left_context_chunks.
        """
        if not self.causal:
            return -1, -1

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            assert len(self.chunk_size) == 1, self.chunk_size
            chunk_size = self.chunk_size[0]
        else:
            chunk_size = random.choice(self.chunk_size)

        if chunk_size == -1:
            left_context_chunks = -1
        else:
            if torch.jit.is_scripting() or torch.jit.is_tracing():
                assert len(self.left_context_frames) == 1, self.left_context_frames
                left_context_frames = self.left_context_frames[0]
            else:
                left_context_frames = random.choice(self.left_context_frames)
            # Note: in Python, -1 // n == -1 for n > 0
            left_context_chunks = left_context_frames // chunk_size
            if left_context_chunks == 0:
                left_context_chunks = 1

        return chunk_size, left_context_chunks

    def forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          x:
            The input tensor. Its shape is (seq_len, batch_size, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          src_key_padding_mask:
            The mask for padding, of shape (batch_size, seq_len); True means
            masked position. May be None.
        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (output_seq_len, batch_size, max(encoder_dim))
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
        """
        outputs = []
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            feature_masks = [1.0] * len(self.encoder_dim)
        else:
            feature_masks = self.get_feature_masks(x)

        chunk_size, left_context_chunks = self.get_chunk_info()

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            # Not support exporting a model for simulating streaming decoding
            attn_mask = None
        else:
            attn_mask = self._get_attn_mask(x, chunk_size, left_context_chunks)

        for i, module in enumerate(self.encoders):
            ds = self.downsampling_factor[i]
            x = convert_num_channels(x, self.encoder_dim[i])

            x = module(
                x,
                chunk_size=chunk_size,
                feature_mask=feature_masks[i],
                src_key_padding_mask=(None if src_key_padding_mask is None else src_key_padding_mask[..., ::ds]),
                attn_mask=attn_mask,
            )
            outputs.append(x)

        # if the last output has the largest dimension, x will be unchanged,
        # it will be the same as outputs[-1].  Otherwise it will be concatenated
        # from different pieces of 'outputs', taking each dimension from the
        # most recent output that has it present.
        x = self._get_full_dim_output(outputs)
        x = self.downsample_output(x)
        # class Downsample has this rounding behavior..
        assert self.output_downsampling_factor == 2, self.output_downsampling_factor
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            lengths = (x_lens + 1) // 2
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lengths = (x_lens + 1) // 2

        return x, lengths

    def _get_attn_mask(self, x: Tensor, chunk_size: int, left_context_chunks: int) -> Optional[Tensor]:
        """
        Return None if chunk_size == -1, else return attention mask of shape
          (seq_len, seq_len), interpreted as (tgt_seq_len, src_seq_len).  True
           means a masked position.
        Args:
           x: embeddings after self.encoder_embed(), of shape (seq_len, batch_size, embed_dim).
          chunk_size: chunk size, must divide
        """
        if chunk_size <= 0:
            return None
        assert all(chunk_size % d == 0 for d in self.downsampling_factor)
        if left_context_chunks >= 0:
            num_encoders = len(self.encoder_dim)
            assert all(
                chunk_size * left_context_chunks >= (self.cnn_module_kernel[i] // 2) * self.downsampling_factor[i]
                for i in range(num_encoders)
            )
        else:
            left_context_chunks = 1000000

        seq_len = x.shape[0]

        # t is frame index, shape (seq_len,)
        t = torch.arange(seq_len, dtype=torch.int32, device=x.device)
        # c is chunk index for each frame, shape (seq_len,)
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            c = t // chunk_size
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                c = t // chunk_size
        src_c = c
        tgt_c = c.unsqueeze(-1)

        attn_mask = torch.logical_or(src_c > tgt_c, src_c < tgt_c - left_context_chunks)
        if __name__ == "__main__":
            logging.info(f"attn_mask = {attn_mask}")
        return attn_mask

    def _get_full_dim_output(self, outputs: List[Tensor]):
        num_encoders = len(self.encoder_dim)
        assert len(outputs) == num_encoders
        output_dim = max(self.encoder_dim)
        output_pieces = [outputs[-1]]
        cur_dim = self.encoder_dim[-1]
        for i in range(num_encoders - 2, -1, -1):
            d = self.encoder_dim[i]
            if d > cur_dim:
                this_output = outputs[i]
                output_pieces.append(this_output[..., cur_dim:d])
                cur_dim = d
        assert cur_dim == output_dim
        return torch.cat(output_pieces, dim=-1)

    def streaming_forward(
        self,
        x: Tensor,
        x_lens: Tensor,
        states: List[Tensor],
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """
        Args:
          x:
            The input tensor. Its shape is (seq_len, batch_size, feature_dim).
          x_lens:
            A tensor of shape (batch_size,) containing the number of frames in
            `x` before padding.
          states: list of cached tensors of all encoder layers. For layer-i,
            states[i*6:(i+1)*6] is (cached_key, cached_nonlin_attn, cached_val1, cached_val2,
            cached_conv1, cached_conv2).
          src_key_padding_mask:
            The mask for padding, of shape (batch_size, seq_len); True means
            masked position. May be None.
        Returns:
          Return a tuple containing 2 tensors:
            - embeddings: its shape is (output_seq_len, batch_size, max(encoder_dim))
            - lengths, a tensor of shape (batch_size,) containing the number
              of frames in `embeddings` before padding.
            - updated states
        """
        outputs = []
        new_states = []
        layer_offset = 0

        for i, module in enumerate(self.encoders):
            num_layers = module.num_layers
            ds = self.downsampling_factor[i]
            x = convert_num_channels(x, self.encoder_dim[i])

            x, new_layer_states = module.streaming_forward(
                x,
                states=states[layer_offset * 6 : (layer_offset + num_layers) * 6],
                left_context_len=self.left_context_frames[0] // ds,
                src_key_padding_mask=src_key_padding_mask[..., ::ds],
            )
            layer_offset += num_layers
            outputs.append(x)
            new_states += new_layer_states

        # if the last output has the largest dimension, x will be unchanged,
        # it will be the same as outputs[-1].  Otherwise it will be concatenated
        # from different pieces of 'outputs', taking each dimension from the
        # most recent output that has it present.
        x = self._get_full_dim_output(outputs)
        x = self.downsample_output(x)
        # class Downsample has this rounding behavior..
        assert self.output_downsampling_factor == 2
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            lengths = (x_lens + 1) // 2
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lengths = (x_lens + 1) // 2

        return x, lengths, new_states

    @torch.jit.export
    def get_init_states(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> List[Tensor]:
        """Get initial states.

        A list of cached tensors of all encoder layers. For layer-i, states[i*6:(i+1)*6]
        is (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2).
        """
        states = []
        for i, module in enumerate(self.encoders):
            num_layers = module.num_layers
            embed_dim = self.encoder_dim[i]
            ds = self.downsampling_factor[i]
            num_heads = self.num_heads[i]
            key_dim = self.query_head_dim[i] * num_heads
            value_dim = self.value_head_dim[i] * num_heads
            downsample_left = self.left_context_frames[0] // ds
            nonlin_attn_head_dim = 3 * embed_dim // 4
            conv_left_pad = self.cnn_module_kernel[i] // 2
            for layer in range(num_layers):
                cached_key = torch.zeros(downsample_left, batch_size, key_dim).to(device)
                cached_nonlin_attn = torch.zeros(1, batch_size, downsample_left, nonlin_attn_head_dim).to(device)
                cached_val1 = torch.zeros(downsample_left, batch_size, value_dim).to(device)
                cached_val2 = torch.zeros(downsample_left, batch_size, value_dim).to(device)
                cached_conv1 = torch.zeros(batch_size, embed_dim, conv_left_pad).to(device)
                cached_conv2 = torch.zeros(batch_size, embed_dim, conv_left_pad).to(device)
                states += [
                    cached_key,
                    cached_nonlin_attn,
                    cached_val1,
                    cached_val2,
                    cached_conv1,
                    cached_conv2,
                ]

        return states


def _whitening_schedule(x: float, ratio: float = 2.0) -> ScheduledFloat:
    return ScheduledFloat((0.0, x), (20000.0, ratio * x), default=x)


def _balancer_schedule(min_prob: float):
    return ScheduledFloat((0.0, 0.4), (8000.0, min_prob))


class Zipformer2EncoderLayer(nn.Module):
    """
    Args:
        embed_dim: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        feedforward_dim: the dimension of the feedforward network model (required).
        dropout: the dropout value (default=0.1).
        cnn_module_kernel (int): Kernel size of convolution module (default=31).

    Examples::
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> pos_emb = torch.rand(32, 19, 512)
        >>> out = encoder_layer(src, pos_emb)
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        value_head_dim: int,
        feedforward_dim: int,
        dropout: FloatLike = 0.1,
        cnn_module_kernel: int = 31,
        causal: bool = False,
        attention_skip_rate: FloatLike = ScheduledFloat((0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),
        conv_skip_rate: FloatLike = ScheduledFloat((0.0, 0.2), (4000.0, 0.05), (16000, 0.0), default=0),
        const_attention_rate: FloatLike = ScheduledFloat((0.0, 0.25), (4000.0, 0.025), default=0),
        ff2_skip_rate: FloatLike = ScheduledFloat((0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)),
        ff3_skip_rate: FloatLike = ScheduledFloat((0.0, 0.1), (4000.0, 0.01), (50000.0, 0.0)),
        bypass_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5), (4000.0, 0.02), default=0),
    ) -> None:
        super(Zipformer2EncoderLayer, self).__init__()
        self.embed_dim = embed_dim

        # self.bypass implements layer skipping as well as bypass; see its default values.
        self.bypass = BypassModule(embed_dim, skip_rate=bypass_skip_rate, straight_through_rate=0)
        # bypass_mid is bypass used in the middle of the layer.
        self.bypass_mid = BypassModule(embed_dim, straight_through_rate=0)

        # skip probability for dynamic modules (meaning: anything but feedforward).
        self.attention_skip_rate = copy.deepcopy(attention_skip_rate)
        # an additional skip probability that applies to ConvModule to stop it from
        # contributing too much early on.
        self.conv_skip_rate = copy.deepcopy(conv_skip_rate)

        # ff2_skip_rate is to prevent the ff2 module from having output that's too big
        # compared to its residual.
        self.ff2_skip_rate = copy.deepcopy(ff2_skip_rate)
        self.ff3_skip_rate = copy.deepcopy(ff3_skip_rate)

        self.const_attention_rate = copy.deepcopy(const_attention_rate)

        self.self_attn_weights = RelPositionMultiheadAttentionWeights(
            embed_dim,
            pos_dim=pos_dim,
            num_heads=num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            dropout=0.0,
        )

        self.self_attn1 = SelfAttention(embed_dim, num_heads, value_head_dim)

        self.self_attn2 = SelfAttention(embed_dim, num_heads, value_head_dim)

        self.feed_forward1 = FeedforwardModule(embed_dim, (feedforward_dim * 3) // 4, dropout)

        self.feed_forward2 = FeedforwardModule(embed_dim, feedforward_dim, dropout)

        self.feed_forward3 = FeedforwardModule(embed_dim, (feedforward_dim * 5) // 4, dropout)

        self.nonlin_attention = NonlinAttention(embed_dim, hidden_channels=3 * embed_dim // 4)

        self.conv_module1 = ConvolutionModule(embed_dim, cnn_module_kernel, causal=causal)

        self.conv_module2 = ConvolutionModule(embed_dim, cnn_module_kernel, causal=causal)

        # TODO: remove it
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))

        self.norm = BiasNorm(embed_dim)

        self.balancer1 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            min_abs=0.2,
            max_abs=4.0,
        )

        # balancer for output of NonlinAttentionModule
        self.balancer_na = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.004), (4000.0, 0.02)),
            prob=0.05,  # out of concern for memory usage
        )

        # balancer for output of feedforward2, prevent it from staying too
        # small.  give this a very small probability, even at the start of
        # training, it's to fix a rare problem and it's OK to fix it slowly.
        self.balancer_ff2 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.0), (4000.0, 0.1), default=0.0),
            max_abs=2.0,
            prob=0.05,
        )

        self.balancer_ff3 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=0.7,
            min_abs=ScheduledFloat((0.0, 0.0), (4000.0, 0.2), default=0.0),
            max_abs=4.0,
            prob=0.05,
        )

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(4.0, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

        self.balancer2 = Balancer(
            embed_dim,
            channel_dim=-1,
            min_positive=0.45,
            max_positive=0.55,
            min_abs=0.1,
            max_abs=4.0,
        )

    def get_sequence_dropout_mask(self, x: Tensor, dropout_rate: float) -> Optional[Tensor]:
        if dropout_rate == 0.0 or not self.training or torch.jit.is_scripting() or torch.jit.is_tracing():
            return None
        batch_size = x.shape[1]
        mask = (torch.rand(batch_size, 1, device=x.device) > dropout_rate).to(x.dtype)
        return mask

    def sequence_dropout(self, x: Tensor, dropout_rate: float) -> Tensor:
        """
        Apply sequence-level dropout to x.
        x shape: (seq_len, batch_size, embed_dim)
        """
        dropout_mask = self.get_sequence_dropout_mask(x, dropout_rate)
        if dropout_mask is None:
            return x
        else:
            return x * dropout_mask

    def forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        chunk_size: int = -1,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
            Pass the input through the encoder layer.
            Args:
                src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
             pos_emb: (1, 2*seq_len-1, pos_emb_dim) or (batch_size, 2*seq_len-1, pos_emb_dim)
             chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking.
           feature_mask: something that broadcasts with src, that we'll multiply `src`
                  by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
             attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                    interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                   True means masked position. May be None.
        src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.

            Returns:
               A tensor which has the same shape as src
        """
        src_orig = src

        # dropout rate for non-feedforward submodules
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            attention_skip_rate = 0.0
        else:
            attention_skip_rate = float(self.attention_skip_rate) if self.training else 0.0

        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        attn_weights = self.self_attn_weights(
            src,
            pos_emb=pos_emb,
            attn_mask=attn_mask,
            key_padding_mask=src_key_padding_mask,
        )

        src = src + self.feed_forward1(src)

        self_attn_dropout_mask = self.get_sequence_dropout_mask(src, attention_skip_rate)

        selected_attn_weights = attn_weights[0:1]
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif self.training and random.random() < float(self.const_attention_rate):
            # Make attention weights constant.  The intention is to
            # encourage these modules to do something similar to an
            # averaging-over-time operation.
            # only need the mask, can just use the 1st one and expand later
            selected_attn_weights = selected_attn_weights[0:1]
            selected_attn_weights = (selected_attn_weights > 0.0).to(selected_attn_weights.dtype)
            selected_attn_weights = selected_attn_weights * (1.0 / selected_attn_weights.sum(dim=-1, keepdim=True))

        na = self.balancer_na(self.nonlin_attention(src, selected_attn_weights))

        src = src + (na if self_attn_dropout_mask is None else na * self_attn_dropout_mask)

        self_attn = self.self_attn1(src, attn_weights)

        src = src + (self_attn if self_attn_dropout_mask is None else self_attn * self_attn_dropout_mask)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            conv_skip_rate = 0.0
        else:
            conv_skip_rate = float(self.conv_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(
            self.conv_module1(src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask),
            conv_skip_rate,
        )

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            ff2_skip_rate = 0.0
        else:
            ff2_skip_rate = float(self.ff2_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(self.balancer_ff2(self.feed_forward2(src)), ff2_skip_rate)

        # bypass in the middle of the layer.
        src = self.bypass_mid(src_orig, src)

        self_attn = self.self_attn2(src, attn_weights)

        src = src + (self_attn if self_attn_dropout_mask is None else self_attn * self_attn_dropout_mask)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            conv_skip_rate = 0.0
        else:
            conv_skip_rate = float(self.conv_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(
            self.conv_module2(src, chunk_size=chunk_size, src_key_padding_mask=src_key_padding_mask),
            conv_skip_rate,
        )

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            ff3_skip_rate = 0.0
        else:
            ff3_skip_rate = float(self.ff3_skip_rate) if self.training else 0.0
        src = src + self.sequence_dropout(self.balancer_ff3(self.feed_forward3(src)), ff3_skip_rate)

        src = self.balancer1(src)
        src = self.norm(src)

        src = self.bypass(src_orig, src)

        src = self.balancer2(src)
        src = self.whiten(src)

        return src

    def streaming_forward(
        self,
        src: Tensor,
        pos_emb: Tensor,
        cached_key: Tensor,
        cached_nonlin_attn: Tensor,
        cached_val1: Tensor,
        cached_val2: Tensor,
        cached_conv1: Tensor,
        cached_conv2: Tensor,
        left_context_len: int,
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Pass the input through the encoder layer in streaming forward mode.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            pos_emb: (1, left_context_len+2*seq_len-1, pos_emb_dim) or
              (batch_size, left_context_len+2*seq_len-1, pos_emb_dim)
            cached_key: cached attention key tensor of left context,
              of shape (left_context_len, batch_size, key_dim)
            cached_nonlin_attn: left context for nonlin_attention module, a Tensor of shape
              (num_heads, batch_size, left_context_len, head_dim)
            cached_val1: cached left context for the first attention module,
              of shape (left_context_len, batch_size, value_dim)
            cached_val2: cached left context for the second attention module,
              of shape (left_context_len, batch_size, value_dim)
            cached_conv1: cached left context for the first convolution module,
              of shape (batch_size, channels, left_pad)
            cached_conv2: cached left context for the second convolution module,
              of shape (batch_size, channels, left_pad)
            left_context_len: number of left context frames.
            src_key_padding_mask:  the mask for padding, of shape
              (batch_size, left_context_len + seq_len); True means masked position.
              May be None.

        Returns:
            - x, with the same shape as src
            - updated cached_key
            - updated cached_nonlin_attn
            - updated cached_val1
            - updated cached_val2
            - updated cached_conv1
            - updated cached_conv2
        """
        src_orig = src

        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        attn_weights, cached_key = self.self_attn_weights.streaming_forward(
            src,
            pos_emb=pos_emb,
            cached_key=cached_key,
            left_context_len=left_context_len,
            key_padding_mask=src_key_padding_mask,
        )

        src = src + self.feed_forward1(src)

        na, cached_nonlin_attn = self.nonlin_attention.streaming_forward(
            src,
            attn_weights[0:1],
            cached_x=cached_nonlin_attn,
            left_context_len=left_context_len,
        )
        src = src + na

        self_attn, cached_val1 = self.self_attn1.streaming_forward(
            src,
            attn_weights=attn_weights,
            cached_val=cached_val1,
            left_context_len=left_context_len,
        )
        src = src + self_attn

        src_conv, cached_conv1 = self.conv_module1.streaming_forward(
            src,
            cache=cached_conv1,
            src_key_padding_mask=src_key_padding_mask[:, left_context_len:],
        )
        src = src + src_conv

        src = src + self.feed_forward2(src)

        # bypass in the middle of the layer.
        src = self.bypass_mid(src_orig, src)

        self_attn, cached_val2 = self.self_attn2.streaming_forward(
            src,
            attn_weights=attn_weights,
            cached_val=cached_val2,
            left_context_len=left_context_len,
        )
        src = src + self_attn

        src_conv, cached_conv2 = self.conv_module2.streaming_forward(
            src,
            cache=cached_conv2,
            src_key_padding_mask=src_key_padding_mask[:, left_context_len:],
        )
        src = src + src_conv

        src = src + self.feed_forward3(src)

        src = self.norm(src)

        src = self.bypass(src_orig, src)

        return (
            src,
            cached_key,
            cached_nonlin_attn,
            cached_val1,
            cached_val2,
            cached_conv1,
            cached_conv2,
        )


class Zipformer2Encoder(nn.Module):
    r"""Zipformer2Encoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the Zipformer2EncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
       pos_dim: the dimension for the relative positional encoding

    Examples::
        >>> encoder_layer = Zipformer2EncoderLayer(embed_dim=512, nhead=8)
        >>> zipformer_encoder = Zipformer2Encoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = zipformer_encoder(src)
    """

    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        pos_dim: int,
        dropout: float,
        warmup_begin: float,
        warmup_end: float,
        initial_layerdrop_rate: float = 0.5,
        final_layerdrop_rate: float = 0.05,
    ) -> None:
        super().__init__()
        self.encoder_pos = CompactRelPositionalEncoding(pos_dim, dropout_rate=0.15, length_factor=1.0)

        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for i in range(num_layers)])
        self.num_layers = num_layers

        assert 0 <= warmup_begin <= warmup_end, (warmup_begin, warmup_end)

        delta = (1.0 / num_layers) * (warmup_end - warmup_begin)
        cur_begin = warmup_begin  # interpreted as a training batch index
        for i in range(num_layers):
            cur_end = cur_begin + delta
            self.layers[i].bypass.skip_rate = ScheduledFloat(
                (cur_begin, initial_layerdrop_rate),
                (cur_end, final_layerdrop_rate),
                default=0.0,
            )
            cur_begin = cur_end

    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        feature_mask: Union[Tensor, float] = 1.0,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            chunk_size: the number of frames per chunk, of >= 0; if -1, no chunking.
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
            attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                 interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                 True means masked position. May be None.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.

        Returns: a Tensor with the same shape as src.
        """
        pos_emb = self.encoder_pos(src)
        output = src

        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            output = output * feature_mask

        for i, mod in enumerate(self.layers):
            output = mod(
                output,
                pos_emb,
                chunk_size=chunk_size,
                attn_mask=attn_mask,
                src_key_padding_mask=src_key_padding_mask,
            )

            if not torch.jit.is_scripting() and not torch.jit.is_tracing():
                output = output * feature_mask

        return output

    def streaming_forward(
        self,
        src: Tensor,
        states: List[Tensor],
        left_context_len: int,
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            states: list of cached tensors of N encoder layers. For layer-i, states[i*6:(i+1)*6] is
              (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2).
            left_context_len: Number of left context frames.
            src_key_padding_mask:  the mask for padding, of shape
              (batch_size, left_context_len + seq_len); True means masked position.
              May be None.

        Returns:
          - output, a Tensor with the same shape as src.
          - updated states
        """
        pos_emb = self.encoder_pos(src, left_context_len)
        output = src

        new_states = []
        for i, mod in enumerate(self.layers):
            (
                cached_key,
                cached_nonlin_attn,
                cached_val1,
                cached_val2,
                cached_conv1,
                cached_conv2,
            ) = states[i * 6 : (i + 1) * 6]
            (
                output,
                new_cached_key,
                new_cached_nonlin_attn,
                new_cached_val1,
                new_cached_val2,
                new_cached_conv1,
                new_cached_conv2,
            ) = mod.streaming_forward(
                output,
                pos_emb,
                cached_key=cached_key,
                cached_nonlin_attn=cached_nonlin_attn,
                cached_val1=cached_val1,
                cached_val2=cached_val2,
                cached_conv1=cached_conv1,
                cached_conv2=cached_conv2,
                left_context_len=left_context_len,
                src_key_padding_mask=src_key_padding_mask,
            )
            new_states += [
                new_cached_key,
                new_cached_nonlin_attn,
                new_cached_val1,
                new_cached_val2,
                new_cached_conv1,
                new_cached_conv2,
            ]

        return output, new_states


class BypassModule(nn.Module):
    """
    An nn.Module that implements a learnable bypass scale, and also randomized per-sequence
    layer-skipping.  The bypass is limited during early stages of training to be close to
    "straight-through", i.e. to not do the bypass operation much initially, in order to
    force all the modules to learn something.
    """

    def __init__(
        self,
        embed_dim: int,
        skip_rate: FloatLike = 0.0,
        straight_through_rate: FloatLike = 0.0,
        scale_min: FloatLike = ScheduledFloat((0.0, 0.9), (20000.0, 0.2), default=0),
        scale_max: FloatLike = 1.0,
    ):
        super().__init__()
        self.bypass_scale = nn.Parameter(torch.full((embed_dim,), 0.5))
        self.skip_rate = copy.deepcopy(skip_rate)
        self.straight_through_rate = copy.deepcopy(straight_through_rate)
        self.scale_min = copy.deepcopy(scale_min)
        self.scale_max = copy.deepcopy(scale_max)

    def _get_bypass_scale(self, batch_size: int):
        # returns bypass-scale of shape (num_channels,),
        # or (batch_size, num_channels,).  This is actually the
        # scale on the non-residual term, so 0 corresponds to bypassing
        # this module.
        if torch.jit.is_scripting() or torch.jit.is_tracing() or not self.training:
            return self.bypass_scale
        else:
            ans = limit_param_value(self.bypass_scale, min=float(self.scale_min), max=float(self.scale_max))
            skip_rate = float(self.skip_rate)
            if skip_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) > skip_rate
                ans = ans * mask
                # now ans is of shape (batch_size, num_channels), and is zero for sequences
                # on which we have randomly chosen to do layer-skipping.
            straight_through_rate = float(self.straight_through_rate)
            if straight_through_rate != 0.0:
                mask = torch.rand((batch_size, 1), device=ans.device) < straight_through_rate
                ans = torch.maximum(ans, mask.to(ans.dtype))
            return ans

    def forward(self, src_orig: Tensor, src: Tensor):
        """
        Args: src_orig and src are both of shape (seq_len, batch_size, num_channels)
        Returns: something with the same shape as src and src_orig
        """
        bypass_scale = self._get_bypass_scale(src.shape[1])
        return src_orig + (src - src_orig) * bypass_scale


class DownsampledZipformer2Encoder(nn.Module):
    r"""
    DownsampledZipformer2Encoder is a zipformer encoder evaluated at a reduced frame rate,
    after convolutional downsampling, and then upsampled again at the output, and combined
    with the origin input, so that the output has the same shape as the input.
    """

    def __init__(
        self,
        encoder: nn.Module,
        dim: int,
        downsample: int,
        dropout: FloatLike,
        causal: bool,
    ):
        super(DownsampledZipformer2Encoder, self).__init__()
        self.downsample_factor = downsample
        self.downsample = SimpleDownsample(dim, downsample, dropout, causal)
        self.num_layers = encoder.num_layers
        self.encoder = encoder
        self.upsample = SimpleUpsample(dim, downsample)
        self.out_combiner = BypassModule(dim, straight_through_rate=0)

    def forward(
        self,
        src: Tensor,
        chunk_size: int = -1,
        feature_mask: Union[Tensor, float] = 1.0,
        attn_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Downsample, go through encoder, upsample.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            feature_mask: something that broadcasts with src, that we'll multiply `src`
               by at every layer: if a Tensor, likely of shape (seq_len, batch_size, embedding_dim)
            attn_mask: the attention mask, of shape (batch_size, seq_len, seq_len) or (seq_len, seq_len),
                 interpreted as (batch_size, tgt_seq_len, src_seq_len) or (tgt_seq_len, src_seq_len).
                 True means masked position. May be None.
            src_key_padding_mask:  the mask for padding, of shape (batch_size, seq_len); True means
                 masked position.  May be None.

        Returns: a Tensor with the same shape as src.
        """
        src_orig = src
        src = self.downsample(src)
        ds = self.downsample_factor
        if attn_mask is not None:
            attn_mask = attn_mask[::ds, ::ds]

        src = self.encoder(
            src,
            chunk_size=chunk_size // ds,
            feature_mask=feature_mask,
            attn_mask=attn_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        src = self.upsample(src)
        # remove any extra frames that are not a multiple of downsample_factor
        src = src[: src_orig.shape[0]]

        return self.out_combiner(src_orig, src)

    def streaming_forward(
        self,
        src: Tensor,
        states: List[Tensor],
        left_context_len: int,
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, List[Tensor]]:
        r"""Downsample, go through encoder, upsample, in streaming forward mode.

        Args:
            src: the sequence to the encoder (required): shape (seq_len, batch_size, embedding_dim).
            states: list of cached tensors of N encoder layers. For layer-i, states[i*6:(i+1)*6] is
              (cached_key, cached_nonlin_attn, cached_val1, cached_val2, cached_conv1, cached_conv2).
            left_context_len: Number of left context frames.
            src_key_padding_mask: the mask for padding, of shape (batch_size, left_context_len+seq_len);
              True means masked position. May be None.

        Returns:
            - output, a Tensor with the same shape as src.
            - updated states
        """
        src_orig = src
        src = self.downsample(src)

        src, new_states = self.encoder.streaming_forward(
            src,
            states=states,
            left_context_len=left_context_len,
            src_key_padding_mask=src_key_padding_mask,
        )
        src = self.upsample(src)
        # remove any extra frames that are not a multiple of downsample_factor
        src = src[: src_orig.shape[0]]

        return self.out_combiner(src_orig, src), new_states


class SimpleDownsample(torch.nn.Module):
    """
    Does downsampling with attention, by weighted sum, and a projection..
    """

    def __init__(self, channels: int, downsample: int, dropout: FloatLike, causal: bool):
        super(SimpleDownsample, self).__init__()

        self.causal = causal
        self.bias = nn.Parameter(torch.zeros(downsample))

        self.name = None  # will be set from training code
        self.dropout = copy.deepcopy(dropout)

        self.downsample = downsample

    def forward(self, src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, in_channels)
        Returns a tensor of shape
           ( (seq_len+downsample-1)//downsample, batch_size, channels)
        """
        (seq_len, batch_size, in_channels) = src.shape
        ds = self.downsample
        d_seq_len = (seq_len + ds - 1) // ds

        # Pad to an exact multiple of self.downsample
        # right-pad src, repeating the last element.
        pad = d_seq_len * ds - seq_len

        if self.causal and torch.jit.is_tracing():
            assert pad == 0, f"pad should be zero for exporting streaming models. Given {pad}"

        # If we are exporting a streaming model, then we skip the if statement
        if not self.causal or not torch.jit.is_tracing():
            src_extra = src[src.shape[0] - 1 :].expand(pad, src.shape[1], src.shape[2])
            src = torch.cat((src, src_extra), dim=0)

        assert src.shape[0] == d_seq_len * ds, (src.shape, d_seq_len, ds)

        src = src.reshape(d_seq_len, ds, batch_size, in_channels)

        weights = self.bias.softmax(dim=0)
        # weights: (downsample, 1, 1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)

        # ans1 is the first `in_channels` channels of the output
        ans = (src * weights).sum(dim=1)

        return ans


class SimpleUpsample(torch.nn.Module):
    """
    A very simple form of upsampling that mostly just repeats the input, but
    also adds a position-specific bias.
    """

    def __init__(self, num_channels: int, upsample: int):
        super(SimpleUpsample, self).__init__()
        self.upsample = upsample

    def forward(self, src: Tensor) -> Tensor:
        """
        x: (seq_len, batch_size, num_channels)
        Returns a tensor of shape
           ( (seq_len*upsample), batch_size, num_channels)
        """
        upsample = self.upsample
        (seq_len, batch_size, num_channels) = src.shape
        src = src.unsqueeze(1).expand(seq_len, upsample, batch_size, num_channels)
        src = src.reshape(seq_len * upsample, batch_size, num_channels)
        return src


class CompactRelPositionalEncoding(torch.nn.Module):
    """
    Relative positional encoding module.  This version is "compact" meaning it is able to encode
    the important information about the relative position in a relatively small number of dimensions.
    The goal is to make it so that small differences between large relative offsets (e.g. 1000 vs. 1001)
    make very little difference to the embedding.   Such differences were potentially important
    when encoding absolute position, but not important when encoding relative position because there
    is now no need to compare two large offsets with each other.

    Our embedding works by projecting the interval [-infinity,infinity] to a finite interval
    using the atan() function, before doing the Fourier transform of that fixed interval.  The
    atan() function would compress the "long tails" too small,
    making it hard to distinguish between different magnitudes of large offsets, so we use a logarithmic
    function to compress large offsets to a smaller range before applying atan().
    Scalings are chosen in such a way that the embedding can clearly distinguish individual offsets as long
    as they are quite close to the origin, e.g. abs(offset) <= about sqrt(embedding_dim)


    Args:
        embed_dim: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length: just a heuristic for initialization.
        length_factor: a heuristic scale (should be >= 1.0) which, if larger, gives
           less weight to small differences of offset near the origin.
    """

    def __init__(
        self,
        embed_dim: int,
        dropout_rate: FloatLike,
        max_len: int = 1000,
        length_factor: float = 1.0,
    ) -> None:
        """Construct a CompactRelPositionalEncoding object."""
        super(CompactRelPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        assert embed_dim % 2 == 0, embed_dim
        self.dropout = Dropout2(dropout_rate)
        self.pe = None
        assert length_factor >= 1.0, length_factor
        self.length_factor = length_factor
        self.extend_pe(torch.tensor(0.0).expand(max_len))

    def extend_pe(self, x: Tensor, left_context_len: int = 0) -> None:
        """Reset the positional encodings."""
        T = x.size(0) + left_context_len

        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(0) >= T * 2 - 1:
                self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return

        # if T == 4, x would contain [ -3, -2, 1, 0, 1, 2, 3 ]
        x = torch.arange(-(T - 1), T, device=x.device).to(torch.float32).unsqueeze(1)

        freqs = 1 + torch.arange(self.embed_dim // 2, device=x.device)

        # `compression_length` this is arbitrary/heuristic, if it is larger we have more resolution
        # for small time offsets but less resolution for large time offsets.
        compression_length = self.embed_dim**0.5
        # x_compressed, like X, goes from -infinity to infinity as T goes from -infinity to infinity;
        # but it does so more slowly than T for large absolute values of T.
        # The formula is chosen so that d(x_compressed )/dx is 1 around x == 0, which
        # is important.
        x_compressed = (
            compression_length * x.sign() * ((x.abs() + compression_length).log() - math.log(compression_length))
        )

        # if self.length_factor == 1.0, then length_scale is chosen so that the
        # FFT can exactly separate points close to the origin (T == 0).  So this
        # part of the formulation is not really heuristic.
        # But empirically, for ASR at least, length_factor > 1.0 seems to work better.
        length_scale = self.length_factor * self.embed_dim / (2.0 * math.pi)

        # note for machine implementations: if atan is not available, we can use:
        #   x.sign() * ((1 / (x.abs() + 1)) - 1)  * (-math.pi/2)
        #  check on wolframalpha.com: plot(sign(x) *  (1 / ( abs(x) + 1) - 1 ) * -pi/2 , atan(x))
        x_atan = (x_compressed / length_scale).atan()  # results between -pi and pi

        cosines = (x_atan * freqs).cos()
        sines = (x_atan * freqs).sin()

        pe = torch.zeros(x.shape[0], self.embed_dim, device=x.device)
        pe[:, 0::2] = cosines
        pe[:, 1::2] = sines
        pe[:, -1] = 1.0  # for bias.

        self.pe = pe.to(dtype=x.dtype)

    def forward(self, x: Tensor, left_context_len: int = 0) -> Tensor:
        """Create positional encoding.

        Args:
            x (Tensor): Input tensor (time, batch, `*`).
            left_context_len: (int): Length of cached left context.

        Returns:
            positional embedding, of shape (batch, left_context_len + 2*time-1, `*`).
        """
        self.extend_pe(x, left_context_len)
        x_size_left = x.size(0) + left_context_len
        # length of positive side: x.size(0) + left_context_len
        # length of negative side: x.size(0)
        pos_emb = self.pe[
            self.pe.size(0) // 2 - x_size_left + 1 : self.pe.size(0) // 2 + x.size(0),  # noqa E203
            :,
        ]
        pos_emb = pos_emb.unsqueeze(0)
        return self.dropout(pos_emb)


class RelPositionMultiheadAttentionWeights(nn.Module):
    r"""Module that computes multi-head attention weights with relative position encoding.
    Various other modules consume the resulting attention weights: see, for example, the
    SimpleAttention module which allows you to compute conventional attention.

    This is a quite heavily modified from: "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context",
    we have to write up the differences.


    Args:
           embed_dim: number of channels at the input to this module, e.g. 256
             pos_dim: dimension of the positional encoding vectors, e.g. 128.
           num_heads:  number of heads to compute weights for, e.g. 8
     query_head_dim: dimension of the query (and key), per head.  e.g. 24.
       pos_head_dim: dimension of the projected positional encoding per head, e.g. 4.
            dropout: dropout probability for attn_output_weights. Default: 0.0.
       pos_emb_skip_rate: probability for skipping the pos_emb part of the scores on
                     any given call to forward(), in training time.
    """

    def __init__(
        self,
        embed_dim: int,
        pos_dim: int,
        num_heads: int,
        query_head_dim: int,
        pos_head_dim: int,
        dropout: float = 0.0,
        pos_emb_skip_rate: FloatLike = ScheduledFloat((0.0, 0.5), (4000.0, 0.0)),
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.query_head_dim = query_head_dim
        self.pos_head_dim = pos_head_dim
        self.dropout = dropout
        self.pos_emb_skip_rate = copy.deepcopy(pos_emb_skip_rate)
        self.name = None  # will be overwritten in training code; for diagnostics.

        key_head_dim = query_head_dim
        in_proj_dim = (query_head_dim + key_head_dim + pos_head_dim) * num_heads

        # the initial_scale is supposed to take over the "scaling" factor of
        # head_dim ** -0.5 that has been used in previous forms of attention,
        # dividing it between the query and key.   Note: this module is intended
        # to be used with the ScaledAdam optimizer; with most other optimizers,
        # it would be necessary to apply the scaling factor in the forward function.
        self.in_proj = ScaledLinear(embed_dim, in_proj_dim, bias=True, initial_scale=query_head_dim**-0.25)

        self.whiten_keys = Whiten(
            num_groups=num_heads,
            whitening_limit=_whitening_schedule(3.0),
            prob=(0.025, 0.25),
            grad_scale=0.025,
        )

        # add a balancer for the keys that runs with very small probability, and
        # tries to enforce that all dimensions have mean around zero.  The
        # weights produced by this module are invariant to adding a constant to
        # the keys, so the derivative of the bias is mathematically zero; but
        # due to how Adam/ScaledAdam work, it can learn a fairly large nonzero
        # bias because the small numerical roundoff tends to have a non-random
        # sign.  This module is intended to prevent that.  Use a very small
        # probability; that should be sufficient to fix the problem.
        self.balance_keys = Balancer(
            key_head_dim * num_heads,
            channel_dim=-1,
            min_positive=0.4,
            max_positive=0.6,
            min_abs=0.0,
            max_abs=100.0,
            prob=0.025,
        )

        # linear transformation for positional encoding.
        self.linear_pos = ScaledLinear(pos_dim, num_heads * pos_head_dim, bias=False, initial_scale=0.05)

        # the following are for diagnostics only, see --print-diagnostics option
        self.copy_pos_query = Identity()
        self.copy_query = Identity()

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Args:
            x: input of shape (seq_len, batch_size, embed_dim)
            pos_emb: Positional embedding tensor, of shape (1, 2*seq_len - 1, pos_dim)
            key_padding_mask: a bool tensor of shape (batch_size, seq_len).  Positions that
               are True in this mask will be ignored as sources in the attention weighting.
            attn_mask: mask of shape (seq_len, seq_len) or (batch_size, seq_len, seq_len),
               interpreted as ([batch_size,] tgt_seq_len, src_seq_len)
               saying which positions are allowed to attend to which other positions.
        Returns:
           a tensor of attention weights, of shape (hum_heads, batch_size, seq_len, seq_len)
           interpreted as (hum_heads, batch_size, tgt_seq_len, src_seq_len).
        """
        x = self.in_proj(x)
        query_head_dim = self.query_head_dim
        pos_head_dim = self.pos_head_dim
        num_heads = self.num_heads

        seq_len, batch_size, _ = x.shape

        query_dim = query_head_dim * num_heads

        # self-attention
        q = x[..., 0:query_dim]
        k = x[..., query_dim : 2 * query_dim]
        # p is the position-encoding query
        p = x[..., 2 * query_dim :]
        assert p.shape[-1] == num_heads * pos_head_dim, (
            p.shape[-1],
            num_heads,
            pos_head_dim,
        )

        q = self.copy_query(q)  # for diagnostics only, does nothing.
        k = self.whiten_keys(self.balance_keys(k))  # does nothing in the forward pass.
        p = self.copy_pos_query(p)  # for diagnostics only, does nothing.

        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(seq_len, batch_size, num_heads, pos_head_dim)
        k = k.reshape(seq_len, batch_size, num_heads, query_head_dim)

        # time1 refers to target, time2 refers to source.
        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, query_head_dim)
        p = p.permute(2, 1, 0, 3)  # (head, batch, time1, pos_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, time2)

        attn_scores = torch.matmul(q, k)

        use_pos_scores = False
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            # We can't put random.random() in the same line
            use_pos_scores = True
        elif not self.training or random.random() >= float(self.pos_emb_skip_rate):
            use_pos_scores = True

        if use_pos_scores:
            pos_emb = self.linear_pos(pos_emb)
            seq_len2 = 2 * seq_len - 1
            pos_emb = pos_emb.reshape(-1, seq_len2, num_heads, pos_head_dim).permute(2, 0, 3, 1)
            # pos shape now: (head, {1 or batch_size}, pos_dim, seq_len2)

            # (head, batch, time1, pos_dim) x (head, 1, pos_dim, seq_len2) -> (head, batch, time1, seq_len2)
            #  [where seq_len2 represents relative position.]
            pos_scores = torch.matmul(p, pos_emb)
            # the following .as_strided() expression converts the last axis of pos_scores from relative
            # to absolute position.  I don't know whether I might have got the time-offsets backwards or
            # not, but let this code define which way round it is supposed to be.
            if torch.jit.is_tracing():
                (num_heads, batch_size, time1, n) = pos_scores.shape
                rows = torch.arange(start=time1 - 1, end=-1, step=-1)
                cols = torch.arange(seq_len)
                rows = rows.repeat(batch_size * num_heads).unsqueeze(-1)
                indexes = rows + cols
                pos_scores = pos_scores.reshape(-1, n)
                pos_scores = torch.gather(pos_scores, dim=1, index=indexes)
                pos_scores = pos_scores.reshape(num_heads, batch_size, time1, seq_len)
            else:
                pos_scores = pos_scores.as_strided(
                    (num_heads, batch_size, seq_len, seq_len),
                    (
                        pos_scores.stride(0),
                        pos_scores.stride(1),
                        pos_scores.stride(2) - pos_scores.stride(3),
                        pos_scores.stride(3),
                    ),
                    storage_offset=pos_scores.stride(3) * (seq_len - 1),
                )

            attn_scores = attn_scores + pos_scores

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif self.training and random.random() < 0.1:
            # This is a harder way of limiting the attention scores to not be
            # too large.  It incurs a penalty if any of them has an absolute
            # value greater than 50.0.  this should be outside the normal range
            # of the attention scores.  We use this mechanism instead of, say,
            # something added to the loss function involving the entropy,
            # because once the entropy gets very small gradients through the
            # softmax can become very small, and we'd get zero derivatives.  The
            # choices of 1.0e-04 as the scale on the penalty makes this
            # mechanism vulnerable to the absolute scale of the loss function,
            # but we view this as a failsafe to avoid "implausible" parameter
            # values rather than a regularization method that should be active
            # under normal circumstances.
            attn_scores = penalize_abs_values_gt(attn_scores, limit=25.0, penalty=1.0e-04, name=self.name)

        assert attn_scores.shape == (num_heads, batch_size, seq_len, seq_len)

        if attn_mask is not None:
            assert attn_mask.dtype == torch.bool
            # use -1000 to avoid nan's where attn_mask and key_padding_mask make
            # all scores zero.  It's important that this be large enough that exp(-1000)
            # is exactly zero, for reasons related to const_attention_rate, it
            # compares the final weights with zero.
            attn_scores = attn_scores.masked_fill(attn_mask, -1000)

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (
                batch_size,
                seq_len,
            ), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1000,
            )

        # We use our own version of softmax, defined in scaling.py, which should
        # save a little of the memory used in backprop by, if we are in
        # automatic mixed precision mode (amp / autocast), by only storing the
        # half-precision output for backprop purposes.
        attn_weights = softmax(attn_scores, dim=-1)

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            pass
        elif random.random() < 0.001 and not self.training:
            self._print_attn_entropy(attn_weights)

        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        return attn_weights

    def streaming_forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        cached_key: Tensor,
        left_context_len: int,
        key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Args:
            x: input of shape (seq_len, batch_size, embed_dim)
            pos_emb: Positional embedding tensor, of shape (1, left_context_len+2*seq_len-1, pos_dim)
            cached_key: cached attention key tensor of left context,
              of shape (left_context_len, batch_size, key_dim)
            left_context_len: number of left context frames.
            key_padding_mask: a bool tensor of shape (batch_size, seq_len).  Positions that
              are True in this mask will be ignored as sources in the attention weighting.

        Returns:
           - attention weights, of shape (hum_heads, batch_size, seq_len, seq_len2),
             interpreted as (hum_heads, batch_size, tgt_seq_len, src_seq_len).
           - updated cached attention key tensor of left context.
        """
        x = self.in_proj(x)
        query_head_dim = self.query_head_dim
        pos_head_dim = self.pos_head_dim
        num_heads = self.num_heads

        seq_len, batch_size, _ = x.shape

        query_dim = query_head_dim * num_heads

        # self-attention
        q = x[..., 0:query_dim]
        k = x[..., query_dim : 2 * query_dim]
        # p is the position-encoding query
        p = x[..., 2 * query_dim :]
        assert p.shape[-1] == num_heads * pos_head_dim

        # Pad cached left contexts
        assert cached_key.shape[0] == left_context_len, (
            cached_key.shape[0],
            left_context_len,
        )
        k = torch.cat([cached_key, k], dim=0)
        # Update cached left contexts
        cached_key = k[-left_context_len:, ...]

        # The length of key
        k_len = k.shape[0]

        q = q.reshape(seq_len, batch_size, num_heads, query_head_dim)
        p = p.reshape(seq_len, batch_size, num_heads, pos_head_dim)
        k = k.reshape(k_len, batch_size, num_heads, query_head_dim)

        # time1 refers to target, time2 refers to source.
        q = q.permute(2, 1, 0, 3)  # (head, batch, time1, query_head_dim)
        p = p.permute(2, 1, 0, 3)  # (head, batch, time1, pos_head_dim)
        k = k.permute(2, 1, 3, 0)  # (head, batch, d_k, time2)

        attn_scores = torch.matmul(q, k)

        pos_emb = self.linear_pos(pos_emb)
        seq_len2 = 2 * seq_len - 1 + left_context_len
        pos_emb = pos_emb.reshape(-1, seq_len2, num_heads, pos_head_dim).permute(2, 0, 3, 1)
        # pos shape now: (head, {1 or batch_size}, pos_dim, seq_len2)

        # (head, batch, time1, pos_dim) x (head, 1, pos_dim, seq_len2) -> (head, batch, time1, seq_len2)
        #  [where seq_len2 represents relative position.]
        pos_scores = torch.matmul(p, pos_emb)

        if torch.jit.is_tracing():
            (num_heads, batch_size, time1, n) = pos_scores.shape
            rows = torch.arange(start=time1 - 1, end=-1, step=-1)
            cols = torch.arange(k_len)
            rows = rows.repeat(batch_size * num_heads).unsqueeze(-1)
            indexes = rows + cols
            pos_scores = pos_scores.reshape(-1, n)
            pos_scores = torch.gather(pos_scores, dim=1, index=indexes)
            pos_scores = pos_scores.reshape(num_heads, batch_size, time1, k_len)
        # the following .as_strided() expression converts the last axis of pos_scores from relative
        # to absolute position.  I don't know whether I might have got the time-offsets backwards or
        # not, but let this code define which way round it is supposed to be.
        else:
            pos_scores = pos_scores.as_strided(
                (num_heads, batch_size, seq_len, k_len),
                (
                    pos_scores.stride(0),
                    pos_scores.stride(1),
                    pos_scores.stride(2) - pos_scores.stride(3),
                    pos_scores.stride(3),
                ),
                storage_offset=pos_scores.stride(3) * (seq_len - 1),
            )

        attn_scores = attn_scores + pos_scores

        assert attn_scores.shape == (
            num_heads,
            batch_size,
            seq_len,
            k_len,
        ), attn_scores.shape

        if key_padding_mask is not None:
            assert key_padding_mask.shape == (batch_size, k_len), key_padding_mask.shape
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1),
                -1000,
            )

        attn_weights = attn_scores.softmax(dim=-1)

        return attn_weights, cached_key

    def _print_attn_entropy(self, attn_weights: Tensor):
        # attn_weights: (num_heads, batch_size, seq_len, seq_len)
        (num_heads, batch_size, seq_len, seq_len) = attn_weights.shape

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                attn_weights = attn_weights.to(torch.float32)
                attn_weights_entropy = -((attn_weights + 1.0e-20).log() * attn_weights).sum(dim=-1).mean(dim=(1, 2))
                logging.info(f"name={self.name}, attn_weights_entropy = {attn_weights_entropy}")


class SelfAttention(nn.Module):
    """
    The simplest possible attention module.  This one works with already-computed attention
    weights, e.g. as computed by RelPositionMultiheadAttentionWeights.

    Args:
          embed_dim: the input and output embedding dimension
          num_heads: the number of attention heads
          value_head_dim: the value dimension per head
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        value_head_dim: int,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(embed_dim, num_heads * value_head_dim, bias=True)

        self.out_proj = ScaledLinear(num_heads * value_head_dim, embed_dim, bias=True, initial_scale=0.05)

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """
        Args:
          x: input tensor, of shape (seq_len, batch_size, embed_dim)
         attn_weights: a tensor of shape (num_heads, batch_size, seq_len, seq_len),
          with seq_len being interpreted as (tgt_seq_len, src_seq_len).  Expect
          attn_weights.sum(dim=-1) == 1.
        Returns:
           a tensor with the same shape as x.
        """
        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

        x = self.in_proj(x)  # (seq_len, batch_size, num_heads * value_head_dim)
        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, value_head_dim)
        value_head_dim = x.shape[-1]

        # todo: see whether there is benefit in overriding matmul
        x = torch.matmul(attn_weights, x)
        # v: (num_heads, batch_size, seq_len, value_head_dim)

        x = x.permute(2, 1, 0, 3).contiguous().view(seq_len, batch_size, num_heads * value_head_dim)

        # returned value is of shape (seq_len, batch_size, embed_dim), like the input.
        x = self.out_proj(x)
        x = self.whiten(x)

        return x

    def streaming_forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
        cached_val: Tensor,
        left_context_len: int,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: input tensor, of shape (seq_len, batch_size, embed_dim)
            attn_weights: a tensor of shape (num_heads, batch_size, seq_len, seq_len),
              with seq_len being interpreted as (tgt_seq_len, src_seq_len).  Expect
              attn_weights.sum(dim=-1) == 1.
            cached_val: cached attention value tensor of left context,
              of shape (left_context_len, batch_size, value_dim)
            left_context_len: number of left context frames.

        Returns:
           - attention weighted output, a tensor with the same shape as x.
           - updated cached attention value tensor of left context.
        """
        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        seq_len2 = seq_len + left_context_len
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len2)

        x = self.in_proj(x)  # (seq_len, batch_size, num_heads * value_head_dim)

        # Pad cached left contexts
        assert cached_val.shape[0] == left_context_len, (
            cached_val.shape[0],
            left_context_len,
        )
        x = torch.cat([cached_val, x], dim=0)
        # Update cached left contexts
        cached_val = x[-left_context_len:, ...]

        x = x.reshape(seq_len2, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, value_head_dim)
        value_head_dim = x.shape[-1]

        # todo: see whether there is benefit in overriding matmul
        x = torch.matmul(attn_weights, x)
        # v: (num_heads, batch_size, seq_len, value_head_dim)

        x = x.permute(2, 1, 0, 3).contiguous().view(seq_len, batch_size, num_heads * value_head_dim)

        # returned value is of shape (seq_len, batch_size, embed_dim), like the input.
        x = self.out_proj(x)

        return x, cached_val


class FeedforwardModule(nn.Module):
    """Feedforward module in Zipformer2 model."""

    def __init__(self, embed_dim: int, feedforward_dim: int, dropout: FloatLike):
        super(FeedforwardModule, self).__init__()
        self.in_proj = nn.Linear(embed_dim, feedforward_dim)

        self.hidden_balancer = Balancer(
            feedforward_dim,
            channel_dim=-1,
            min_positive=0.3,
            max_positive=1.0,
            min_abs=0.75,
            max_abs=5.0,
        )

        # shared_dim=0 means we share the dropout mask along the time axis
        self.out_proj = ActivationDropoutAndLinear(
            feedforward_dim,
            embed_dim,
            activation="SwooshL",
            dropout_p=dropout,
            dropout_shared_dim=0,
            bias=True,
            initial_scale=0.1,
        )

        self.out_whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(self, x: Tensor):
        x = self.in_proj(x)
        x = self.hidden_balancer(x)
        # out_proj contains SwooshL activation, then dropout, then linear.
        x = self.out_proj(x)
        x = self.out_whiten(x)
        return x


class NonlinAttention(nn.Module):
    """This is like the ConvolutionModule, but refactored so that we use multiplication by attention weights (borrowed
       from the attention module) in place of actual convolution.  We also took out the second nonlinearity, the
       one after the attention mechanism.

    Args:
        channels (int): The number of channels of conv layers.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int,
    ) -> None:
        super().__init__()

        self.hidden_channels = hidden_channels

        self.in_proj = nn.Linear(channels, hidden_channels * 3, bias=True)

        # balancer that goes before the sigmoid.  Have quite a large min_abs value, at 2.0,
        # because we noticed that well-trained instances of this module have abs-value before the sigmoid
        # starting from about 3, and poorly-trained instances of the module have smaller abs values
        # before the sigmoid.
        self.balancer = Balancer(
            hidden_channels,
            channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.25), (20000.0, 0.05)),
            max_positive=ScheduledFloat((0.0, 0.75), (20000.0, 0.95)),
            min_abs=0.5,
            max_abs=5.0,
        )
        self.tanh = nn.Tanh()

        self.identity1 = Identity()  # for diagnostics.
        self.identity2 = Identity()  # for diagnostics.
        self.identity3 = Identity()  # for diagnostics.

        self.out_proj = ScaledLinear(hidden_channels, channels, bias=True, initial_scale=0.05)

        self.whiten1 = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(5.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

        self.whiten2 = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(5.0, ratio=3.0),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

    def forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """.
                Args:
                   x: a Tensor of shape (seq_len, batch_size, num_channels)
        attn_weights: a Tensor of shape (num_heads, batch_size, seq_len, seq_len)
                Returns:
                   a Tensor with the same shape as x
        """
        x = self.in_proj(x)

        (seq_len, batch_size, _) = x.shape
        hidden_channels = self.hidden_channels

        s, x, y = x.chunk(3, dim=2)

        # s will go through tanh.

        s = self.balancer(s)
        s = self.tanh(s)

        s = s.unsqueeze(-1).reshape(seq_len, batch_size, hidden_channels)
        x = self.whiten1(x)
        x = x * s
        x = self.identity1(x)  # diagnostics only, it's the identity.

        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (num_heads, batch_size, seq_len, seq_len)

        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = torch.matmul(attn_weights, x)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = x.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)

        y = self.identity2(y)
        x = x * y
        x = self.identity3(x)

        x = self.out_proj(x)
        x = self.whiten2(x)
        return x

    def streaming_forward(
        self,
        x: Tensor,
        attn_weights: Tensor,
        cached_x: Tensor,
        left_context_len: int,
    ) -> Tuple[Tensor, Tensor]:
        """.
        Args:
            x: a Tensor of shape (seq_len, batch_size, num_channels)
            attn_weights: a Tensor of shape (num_heads, batch_size, seq_len, seq_len)
            cached_x: left context, a Tensor of shape
              (num_heads, batch_size, left_context_len, head_dim)
            left_context_len: number of left context frames.
        Returns:
            - a Tensor with the same shape as x
            - updated left context with same shape as cached_x
        """
        x = self.in_proj(x)

        (seq_len, batch_size, _) = x.shape
        hidden_channels = self.hidden_channels

        s, x, y = x.chunk(3, dim=2)

        # s will go through tanh.
        s = self.tanh(s)

        s = s.unsqueeze(-1).reshape(seq_len, batch_size, hidden_channels)
        x = x * s

        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = attn_weights.shape[0]
        assert attn_weights.shape == (
            num_heads,
            batch_size,
            seq_len,
            left_context_len + seq_len,
        )

        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        # now x: (num_heads, batch_size, seq_len, head_dim)

        # Pad cached tensor
        assert cached_x.shape[2] == left_context_len, (
            cached_x.shape[2],
            left_context_len,
        )
        x_pad = torch.cat([cached_x, x], dim=2)
        # Update cached tensor
        cached_x = x_pad[:, :, -left_context_len:, :]

        x = torch.matmul(attn_weights, x_pad)
        # now x: (num_heads, batch_size, seq_len, head_dim)
        x = x.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)

        x = x * y

        x = self.out_proj(x)
        return x, cached_x


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Zipformer2 model.
    Modified from https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/zipformer/convolution.py

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.
        bias (bool): Whether to use bias in conv layers (default=True).

    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        causal: bool,
    ) -> None:
        """Construct a ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernerl_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        bottleneck_dim = channels
        self.causal = causal

        self.in_proj = nn.Linear(
            channels,
            2 * bottleneck_dim,
        )
        # the gradients on in_proj are a little noisy, likely to do with the
        # sigmoid in glu.

        # after in_proj we put x through a gated linear unit (nn.functional.glu).
        # For most layers the normal rms value of channels of x seems to be in the range 1 to 4,
        # but sometimes, for some reason, for layer 0 the rms ends up being very large,
        # between 50 and 100 for different channels.  This will cause very peaky and
        # sparse derivatives for the sigmoid gating function, which will tend to make
        # the loss function not learn effectively.  (for most layers the average absolute values
        # are in the range 0.5..9.0, and the average p(x>0), i.e. positive proportion,
        # at the output of pointwise_conv1.output is around 0.35 to 0.45 for different
        # layers, which likely breaks down as 0.5 for the "linear" half and
        # 0.2 to 0.3 for the part that goes into the sigmoid.  The idea is that if we
        # constrain the rms values to a reasonable range via a constraint of max_abs=10.0,
        # it will be in a better position to start learning something, i.e. to latch onto
        # the correct range.
        self.balancer1 = Balancer(
            bottleneck_dim,
            channel_dim=-1,
            min_positive=ScheduledFloat((0.0, 0.05), (8000.0, 0.025)),
            max_positive=1.0,
            min_abs=1.5,
            max_abs=ScheduledFloat((0.0, 5.0), (8000.0, 10.0), default=1.0),
        )

        self.activation1 = Identity()  # for diagnostics

        self.sigmoid = nn.Sigmoid()

        self.activation2 = Identity()  # for diagnostics

        assert kernel_size % 2 == 1

        self.depthwise_conv = (
            ChunkCausalDepthwiseConv1d(channels=bottleneck_dim, kernel_size=kernel_size)
            if causal
            else nn.Conv1d(
                in_channels=bottleneck_dim,
                out_channels=bottleneck_dim,
                groups=bottleneck_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )

        self.balancer2 = Balancer(
            bottleneck_dim,
            channel_dim=1,
            min_positive=ScheduledFloat((0.0, 0.1), (8000.0, 0.05)),
            max_positive=1.0,
            min_abs=ScheduledFloat((0.0, 0.2), (20000.0, 0.5)),
            max_abs=10.0,
        )

        self.whiten = Whiten(
            num_groups=1,
            whitening_limit=_whitening_schedule(7.5),
            prob=(0.025, 0.25),
            grad_scale=0.01,
        )

        self.out_proj = ActivationDropoutAndLinear(
            bottleneck_dim,
            channels,
            activation="SwooshR",
            dropout_p=0.0,
            initial_scale=0.05,
        )

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
        chunk_size: int = -1,
    ) -> Tensor:
        """Compute convolution module.

        Args:
            x: Input tensor (#time, batch, channels).
           src_key_padding_mask: the mask for the src keys per batch (optional):
               (batch, #time), contains True in masked positions.

        Returns:
            Tensor: Output tensor (#time, batch, channels).

        """

        x = self.in_proj(x)  # (time, batch, 2*channels)

        x, s = x.chunk(2, dim=2)
        s = self.balancer1(s)
        s = self.sigmoid(s)
        x = self.activation1(x)  # identity.
        x = x * s
        x = self.activation2(x)  # identity

        # (time, batch, channels)

        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)

        if not torch.jit.is_scripting() and not torch.jit.is_tracing() and chunk_size >= 0:
            # Not support exporting a model for simulated streaming decoding
            assert self.causal, "Must initialize model with causal=True if you use chunk_size"
            x = self.depthwise_conv(x, chunk_size=chunk_size)
        else:
            x = self.depthwise_conv(x)

        x = self.balancer2(x)
        x = x.permute(2, 0, 1)  # (time, batch, channels)

        x = self.whiten(x)  # (time, batch, channels)
        x = self.out_proj(x)  # (time, batch, channels)

        return x

    def streaming_forward(
        self,
        x: Tensor,
        cache: Tensor,
        src_key_padding_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute convolution module in streaming forward mode.

        Args:
            x: Input tensor (#time, batch, channels).
            cache: cached left context for depthwise_conv of shape
              (#batch, channels, left_pad)
            src_key_padding_mask: the mask for the src keys per batch (optional):
              (batch, #time), contains True in masked positions.

        Returns:
            - Output tensor (#time, batch, channels).
            - Updated cache (#batch, channels, left_pad)
        """

        x = self.in_proj(x)  # (time, batch, 2*channels)

        x, s = x.chunk(2, dim=2)
        s = self.sigmoid(s)
        x = x * s
        # (time, batch, channels)

        # exchange the temporal dimension and the feature dimension
        x = x.permute(1, 2, 0)  # (#batch, channels, time).

        if src_key_padding_mask is not None:
            x = x.masked_fill(src_key_padding_mask.unsqueeze(1).expand_as(x), 0.0)

        x, cache = self.depthwise_conv.streaming_forward(x, cache=cache)

        x = x.permute(2, 0, 1)  # (time, batch, channels)

        x = self.out_proj(x)  # (time, batch, channels)

        return x, cache


class ScalarMultiply(nn.Module):
    def __init__(self, scale: float):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale


def _test_zipformer_main(causal: bool = False):
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.

    c = Zipformer2(
        encoder_dim=(64, 96),
        encoder_unmasked_dim=(48, 64),
        num_heads=(4, 4),
        causal=causal,
        chunk_size=(4,) if causal else (-1,),
        left_context_frames=(64,),
    )
    batch_size = 5
    seq_len = 20
    # Just make sure the forward pass runs.
    f = c(
        torch.randn(seq_len, batch_size, 64),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f[0].sum().backward()
    c.eval()
    f = c(
        torch.randn(seq_len, batch_size, 64),
        torch.full((batch_size,), seq_len, dtype=torch.int64),
    )
    f  # to remove flake8 warnings


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _test_zipformer_main(False)
    _test_zipformer_main(True)
