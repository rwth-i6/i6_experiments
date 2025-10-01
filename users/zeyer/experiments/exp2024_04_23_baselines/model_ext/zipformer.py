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

from typing import TYPE_CHECKING, Optional, Union, Any, Tuple, Dict

# To provide a RF compatible encoder interface.
import returnn.frontend as rf
from returnn.frontend.encoder.base import ISeqDownsamplingEncoder
from returnn.tensor import Tensor, Dim

if TYPE_CHECKING:
    from .zipformer_common import AttributeDict


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
        from returnn.torch.frontend.bridge import pt_module_to_rf_module
        from .zipformer_impl import get_params, get_encoder_model

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
        self, source: Tensor, *, in_spatial_dim: Dim, collected_outputs: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Dim]:
        import torch
        from .zipformer_impl import set_batch_count, make_pad_mask

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
        out_size = out_spatial_dim.dyn_size_ext
        if len(batch_dims) > 1:
            x, batch_dim = rf.merge_dims(x, dims=batch_dims)
            out_size = out_size.copy_compatible_to(
                rf.Tensor("size_ex", dims=batch_dims, dtype="int32"),
                unbroadcast=True,
                check_sparse=False,
                check_dtype=False,
            )
            out_size, _ = rf.merge_dims(out_size, dims=batch_dims, out_dim=batch_dim)
        else:
            assert len(batch_dims) == 1  # just not implemented otherwise
            batch_dim = batch_dims[0]
        assert out_size.dims == (batch_dim,), f"x {x}"
        x_lens = out_size.raw_tensor  # (N,)
        x_ = x.copy_compatible_to_dims_raw((out_spatial_dim, batch_dim, self.enc_in_dim))  # (T, N, C)

        src_key_padding_mask = make_pad_mask(x_lens)
        src_key_padding_mask = src_key_padding_mask.to(x_.device)

        encoder_out, encoder_out_lens = self._encoder_pt(x_, x_lens, src_key_padding_mask)
        # encoder_out: (T, N, C)
        assert encoder_out_lens.shape == x_lens.shape
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)
        out_size = rf.convert_to_tensor(encoder_out_lens, dims=[batch_dim])
        if len(batch_dims) > 1:
            out_size = rf.split_dims(out_size, axis=batch_dim, dims=batch_dims)
        out_spatial_dim = Dim(out_size, name="zip_out_spatial")
        encoder_out_ = rf.convert_to_tensor(encoder_out, dims=[out_spatial_dim, batch_dim, self.out_dim])
        if len(batch_dims) > 1:
            encoder_out_ = rf.split_dims(encoder_out_, axis=batch_dim, dims=batch_dims)
        return encoder_out_, out_spatial_dim
