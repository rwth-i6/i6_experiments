"""
Transformer decoder with custom readout (maxout).
"""

from __future__ import annotations
from typing import Optional, Any, Union, Tuple, Dict, Callable, Sequence
from types import FunctionType
import functools
import logging
import copy as _copy
from returnn.util.basic import NotSpecified, BehaviorVersion
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, single_step_dim
from returnn.frontend.decoder.transformer import TransformerDecoderLayer, make_norm


class ReadoutMaxout(rf.Module):
    def __init__(
        self,
        in_dim: Dim,
        vocab_dim: Dim,
        *,
        readout_dim: Union[int, Dim] = 1024,
        readout_dropout: float = 0.3,
        output_prob_with_bias: bool = False,
    ):
        super().__init__()

        if isinstance(readout_dim, int):
            readout_dim = Dim(readout_dim, name="readout")
        assert readout_dim.dimension % 2 == 0

        self.readout_in = rf.Linear(in_dim, readout_dim)
        self.readout_dropout = readout_dropout
        self.output_prob = rf.Linear(self.readout_in.out_dim // 2, vocab_dim, with_bias=output_prob_with_bias)

        self.dropout_broadcast = rf.dropout_broadcast_default()

    def __call__(self, source: Tensor) -> Tensor:
        # decode logits
        readout_in = self.readout_in(source)
        readout = rf.reduce_out(readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim)
        readout = rf.dropout(
            readout, drop_prob=self.readout_dropout, axis=self.dropout_broadcast and readout.feature_dim
        )
        logits = self.output_prob(readout)
        return logits


class TransformerDecoder(rf.Module):
    """
    Represents the Transformer decoder architecture
    """

    def __init__(
        self,
        encoder_dim: Optional[Dim],
        vocab_dim: Dim,
        model_dim: Union[Dim, int] = Dim(512, name="transformer-dec-default-model-dim"),
        *,
        num_layers: int,
        ff: Union[type, Dict[str, Any], rf.Module] = NotSpecified,
        ff_dim: Union[Dim, int] = NotSpecified,
        ff_activation: Union[Callable[[Tensor], Tensor], Dict[str, Any], rf.Module] = NotSpecified,
        pos_enc: Union[None, Callable, Dict[str, Any], rf.Module] = rf.sinusoidal_positional_encoding,
        dropout: float = 0.1,
        num_heads: int = 8,
        att_dropout: float = 0.1,
        norm: Union[type, Dict[str, Any], rf.Module, Callable] = rf.LayerNorm,
        layer: Optional[Union[TransformerDecoderLayer, rf.Module, type, Dict[str, Any], Any]] = None,
        layer_opts: Optional[Dict[str, Any]] = None,
        embed_dim: Optional[Dim] = None,
        share_embedding: bool = None,
        input_embedding: bool = True,
        input_embedding_scale: float = None,
        input_dropout: float = None,
        sequential=rf.Sequential,
        readout: Dict[str, Any],
    ):
        """
        :param encoder_dim: for cross-attention. None if no cross-attention.
        :param vocab_dim:
        :param model_dim: the output feature dimension
        :param num_layers: the number of encoder layers
        :param ff: feed-forward / MLP block. Default is :class:`FeedForward`
        :param ff_dim: the dimension of feed-forward layers. 2048 originally, or 4 times out_dim
        :param ff_activation: activation function for feed-forward network
        :param pos_enc: positional encoding. Default is sinusoidal positional encoding.
        :param dropout: the dropout value for the FF block
        :param num_heads: the number of attention heads
        :param att_dropout: attention dropout value
        :param norm: pre-normalization for FF and attention blocks
        :param layer: an instance of :class:`TransformerDecoderLayer` or similar
        :param layer_opts: options for the decoder layer
        :param embed_dim: if given, will first have an embedding [vocab,embed] and then a linear [embed,model].
        :param share_embedding:
        :param input_embedding: whether to use input embedding. If False, you must provide input of dimension model_dim.
        :param input_embedding_scale:
        :param input_dropout:
        :param logits_with_bias:
        :param sequential:
        """
        super().__init__()

        if not isinstance(vocab_dim, Dim):
            raise TypeError(f"TransformerDecoder: unexpected vocab_dim {vocab_dim!r} type {type(vocab_dim)}")
        if isinstance(model_dim, int):
            model_dim = Dim(model_dim, name="transformer-dec-model-dim")
        if not isinstance(model_dim, Dim):
            raise TypeError(f"TransformerDecoder: unexpected model_dim {model_dim!r} type {type(model_dim)}")

        self.encoder_dim = encoder_dim
        self.vocab_dim = vocab_dim
        self.model_dim = model_dim
        self.embed_dim = embed_dim

        # We could make this optional or configurable if we ever need to.
        # Or maybe you would just have another separate implementation of this module then...
        self.input_embedding = rf.Embedding(vocab_dim, embed_dim or model_dim) if input_embedding else None

        self.input_embedding_proj = None
        if embed_dim:
            self.input_embedding_proj = rf.Linear(embed_dim, model_dim, with_bias=False)

        if pos_enc is None:
            pass
        elif isinstance(pos_enc, dict):
            pos_enc = rf.build_from_dict(pos_enc, feat_dim=embed_dim or model_dim)
        elif isinstance(pos_enc, rf.Module):
            pass
        elif isinstance(pos_enc, FunctionType):
            pos_enc = functools.partial(pos_enc, feat_dim=embed_dim or model_dim)
        else:
            raise TypeError(f"unexpected pos_enc type {pos_enc!r}")
        self.pos_enc = pos_enc
        if share_embedding is None:
            share_embedding = False
        if input_embedding_scale is None:
            if input_embedding:
                if BehaviorVersion.get() < 20:
                    logging.getLogger("returnn.frontend").warning(
                        "TransformerDecoder input_embedding_scale default is suboptimal"
                        f" with your behavior version {BehaviorVersion.get()}."
                        " Explicitly set input_embedding_scale or switch to a new behavior version >= 20."
                    )
                input_embedding_scale = model_dim.dimension**0.5 if BehaviorVersion.get() >= 20 else 1.0
            elif pos_enc:
                input_embedding_scale = model_dim.dimension**0.5
            else:
                input_embedding_scale = 1.0
        self.input_embedding_scale = input_embedding_scale
        if input_dropout is None:
            if dropout > 0 and BehaviorVersion.get() < 20:
                logging.getLogger("returnn.frontend").warning(
                    "TransformerDecoder input_dropout default is suboptimal"
                    f" with your behavior version {BehaviorVersion.get()}."
                    " Explicitly set input_dropout or switch to a new behavior version >= 20."
                )
            input_dropout = dropout if BehaviorVersion.get() >= 20 else 0.0
        self.input_dropout = input_dropout

        if not layer or isinstance(layer, (dict, type)):
            layer_opts_ = dict(
                encoder_dim=encoder_dim,
                out_dim=model_dim,
                ff=ff,
                ff_dim=ff_dim,
                ff_activation=ff_activation,
                dropout=dropout,
                num_heads=num_heads,
                att_dropout=att_dropout,
                norm=norm,
            )
            layer_opts_ = {k: v for (k, v) in layer_opts_.items() if v is not NotSpecified}
            if layer_opts:
                layer_opts_.update(layer_opts)
            if not layer:
                layer = TransformerDecoderLayer(**layer_opts_)
            elif isinstance(layer, type):
                layer = layer(**layer_opts_)
            elif isinstance(layer, dict):
                layer_opts_ = {k: v for (k, v) in layer_opts_.items() if k not in layer}
                layer = rf.build_from_dict(layer, **layer_opts_)
            else:
                raise TypeError(f"unexpected layer {layer!r}")

        self.layers = sequential(_copy.deepcopy(layer) for _ in range(num_layers))

        self.final_layer_norm = make_norm(norm, model_dim)

        self.readout: ReadoutMaxout = rf.build_from_dict(readout, in_dim=model_dim, vocab_dim=vocab_dim)

        if share_embedding:
            self.readout.output_prob.weight = self.input_embedding.weight

    def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
        """default initial state"""
        state = rf.State({k: v.default_initial_state(batch_dims=batch_dims) for k, v in self.layers.items()})
        state.pos = rf.zeros((), dtype="int32", device="cpu")
        return state

    def transform_encoder(self, encoder: Tensor, *, axis: Dim) -> rf.State:
        """
        Transform encoder output.
        Note that the Transformer decoder usually expects that layer-norm was applied already on the encoder output.
        """
        return rf.State({k: v.transform_encoder(encoder, axis=axis) for k, v in self.layers.items()})

    def __call__(
        self,
        source: Tensor,
        *,
        spatial_dim: Dim,
        state: rf.State,
        encoder: Optional[rf.State] = None,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
        output_only_last_frame: bool = False,
    ) -> Tuple[Tensor, rf.State]:
        """
        forward, single step or whole sequence.

        :param source: labels
        :param spatial_dim: single_step_dim or spatial dim of source
        :param state: e.g. via :func:`default_initial_state`
        :param encoder: via :func:`transform_encoder`
        :param collected_outputs:
        :param output_only_last_frame: if True, and spatial_dim is not single_step_dim,
            the returned logits will only be for the last frame
        :return: logits, new state
        """
        new_state = rf.State()

        if self.input_embedding is not None:
            decoded = self.input_embedding(source)
        else:
            decoded = source
        if self.input_embedding_scale != 1:
            decoded = decoded * self.input_embedding_scale
        if self.pos_enc is not None:
            decoded = decoded + self.pos_enc(spatial_dim=spatial_dim, offset=state.pos)
        decoded = rf.dropout(decoded, self.input_dropout)
        if self.input_embedding_proj is not None:
            decoded = self.input_embedding_proj(decoded)

        new_state.pos = state.pos + (1 if spatial_dim == single_step_dim else spatial_dim.get_size_tensor())

        for layer_name, layer in self.layers.items():
            layer: TransformerDecoderLayer  # or similar
            decoded, new_state[layer_name] = layer(
                decoded,
                spatial_dim=spatial_dim,
                state=state[layer_name],
                encoder=encoder[layer_name] if encoder else None,
            )
            if collected_outputs is not None:
                collected_outputs[layer_name] = decoded

        if output_only_last_frame and spatial_dim != single_step_dim:
            decoded = rf.gather(decoded, axis=spatial_dim, indices=rf.last_frame_position_of_dim(spatial_dim))

        decoded = self.final_layer_norm(decoded)

        logits = self.readout(decoded)

        return logits, new_state
