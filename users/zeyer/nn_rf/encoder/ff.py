from __future__ import annotations
from typing import Union, Optional, Tuple, Callable, Dict, Any
import copy as _copy
from returnn.util.basic import NotSpecified
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.encoder.base import ISeqDownsamplingEncoder
from returnn.frontend.encoder.conformer import (
    ConformerEncoderLayer,
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
    make_ff,
    make_norm,
)


class FeedForwardEncoder(rf.Module):
    """
    Represents a simple feed-forward encoder.
    """

    def __init__(
        self,
        in_dim: Dim,
        out_dim: Union[int, Dim] = Dim(512, name="conformer-enc-default-out-dim"),
        *,
        input_layer: Optional[
            Union[Frontend, ConformerConvSubsample, ISeqDownsamplingEncoder, rf.Module, Dict[str, Any], Any]
        ] = NotSpecified,
        input_dropout: float = 0.1,
        num_layers: int,
        encoder_layer: Optional[
            Union[FeedForwardEncoderLayer, ConformerEncoderLayer, rf.Module, type, Dict[str, Any], Any]
        ] = None,
    ):
        """
        :param in_dim: the input feature dimension (e.g. logmel)
        :param out_dim: model dim
        :param input_layer: the input layer module or its config dict / frontend, with downsampling, adding ctx
        :param input_dropout: dropout probability after the input layer
        :param num_layers: the number of encoder layers
        :param encoder_layer: the encoder layer module or its config dict
        """
        super().__init__()

        if isinstance(out_dim, int):
            out_dim = Dim(out_dim, name="enc_dim")

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout_broadcast = rf.dropout_broadcast_default()

        if input_layer is NotSpecified or input_layer is None:
            input_layer = Frontend(in_dim, out_dim)
        elif isinstance(input_layer, ISeqDownsamplingEncoder) or callable(input_layer):
            pass
        elif isinstance(input_layer, dict):
            input_layer = rf.build_from_dict(input_layer, in_dim, out_dim)
        else:
            raise ValueError(f"Invalid input_layer specification: {input_layer} (type {type(input_layer)})")
        input_layer: ISeqDownsamplingEncoder
        self.input_layer = input_layer
        self.input_dropout = input_dropout

        in_dim_ = self.input_layer.out_dim if self.input_layer else self.in_dim
        self.input_projection = (
            rf.Linear(in_dim_, self.out_dim, with_bias=False) if in_dim_ != self.out_dim else rf.identity
        )

        if not encoder_layer or isinstance(encoder_layer, dict):
            encoder_layer_opts_ = dict(out_dim=out_dim)
            encoder_layer_opts_ = {k: v for (k, v) in encoder_layer_opts_.items() if v is not NotSpecified}
            if not encoder_layer:
                encoder_layer = FeedForwardEncoderLayer(**encoder_layer_opts_)
            elif isinstance(encoder_layer, dict):
                encoder_layer_opts_ = {k: v for (k, v) in encoder_layer_opts_.items() if k not in encoder_layer}
                encoder_layer = rf.build_from_dict(encoder_layer, **encoder_layer_opts_)
            else:
                raise TypeError(f"unexpected encoder_layer {encoder_layer!r}")
        else:
            if not callable(encoder_layer):
                raise TypeError(f"{self}: invalid non-callable encoder_layer {encoder_layer!r}")

        self.layers = rf.Sequential(_copy.deepcopy(encoder_layer) for _ in range(num_layers))

    def __call__(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Dim]:
        x, out_spatial_dim = self.input_layer(source, in_spatial_dim=in_spatial_dim)
        x = self.input_projection(x)
        x = rf.dropout(x, self.input_dropout, axis=self.dropout_broadcast and self.input_projection.out_dim)
        x = self.layers(x, collected_outputs=collected_outputs)
        return x, out_spatial_dim


class FeedForwardEncoderLayer(rf.Module):
    def __init__(
        self,
        out_dim: Dim,
        *,
        ff: Union[type, ConformerPositionwiseFeedForward, Dict[str, Any], rf.Module] = NotSpecified,
        ff_dim: Dim = NotSpecified,
        ff_activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module] = NotSpecified,
        dropout: float = 0.1,
        norm: Union[type, Dict[str, Any], rf.Module, Callable] = rf.LayerNorm,
    ):
        super().__init__()

        self.out_dim = out_dim

        self.dropout = dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()

        self.ffn = make_ff(ff=ff, out_dim=out_dim, ff_dim=ff_dim, dropout=dropout, ff_activation=ff_activation)
        self.ffn_layer_norm = make_norm(norm, out_dim)
        self.final_layer_norm = make_norm(norm, out_dim)

    def __call__(self, inp: Tensor, *, spatial_dim: Dim, chunked_time_dim: Dim) -> Tensor:
        """forward"""
        # FFN
        x_ffn_ln = self.ffn_layer_norm(inp)
        x_ffn = self.ffn(x_ffn_ln)
        x_ffn_out = rf.dropout(x_ffn, self.dropout, axis=self.dropout_broadcast and self.out_dim) + inp

        # last LN layer
        return self.final_layer_norm(x_ffn_out)


class Frontend(ISeqDownsamplingEncoder):
    def __init__(
        self,
        in_dim: Dim,
        out_dim: Dim,
        *,
        window_size: int = 17,
        stride: int = 6,
        activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module] = NotSpecified,
    ):
        """
        :param in_dim: input feature dimension
        :param out_dim: output feature dimension
        :param window_size:
        :param stride:
        :param activation: by default identity here
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.window_size = window_size
        self.stride = stride

        self.conv = rf.Conv1d(in_dim, out_dim, filter_size=window_size, padding="same", strides=stride)
        self.activation = _make_activation(activation, default=rf.identity)

    def __call__(self, source: Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
        x, out_spatial_dim = self.conv(source, in_spatial_dim=in_spatial_dim)
        x = self.activation(x)
        return x, out_spatial_dim


def _make_activation(
    activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module, None], *, default=rf.swish
) -> Callable[[Tensor], Tensor]:
    if activation is NotSpecified or activation is None:
        return default
    if (not isinstance(activation, type) and callable(activation)) or isinstance(activation, rf.Module):
        return activation
    if isinstance(activation, dict):
        return rf.build_from_dict(activation)
    raise ValueError(f"Invalid activation specification {activation} (type {type(activation)})")
