import math
from typing import Callable, Optional, Tuple, cast

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoderLayer
from returnn.tensor import Dim as ReturnnDim
from returnn.tensor import Tensor as ReturnnTensor

# base of the geometric frequencies of the delay embedding
_DELAY_EMBEDDING_THETA = 10_000.0


def delay_embedding(delay_frames: ReturnnTensor, *, feat_dim: ReturnnDim) -> ReturnnTensor:
    """
    Embeds a delay in frames by the cosines and then the sines of geometrically spaced frequencies.

    :param delay_frames: the delay of every sequence [batch], or one for all []
    :param feat_dim: the width of the embedding, even
    :return: the embedding [..., feat_dim] in float32
    """
    width = feat_dim.dimension
    assert width is not None and width % 2 == 0, feat_dim
    half_dim = ReturnnDim(width // 2, name="delay_embedding_half")

    frequencies = rf.exp(
        rf.range_over_dim(half_dim, dtype="float32", device=delay_frames.device)
        * (-math.log(_DELAY_EMBEDDING_THETA) / (width // 2))
    )
    angles = rf.cast(delay_frames, "float32") * frequencies
    embedding, _ = rf.concat((rf.cos(angles), half_dim), (rf.sin(angles), half_dim), out_dim=feat_dim)
    return embedding


class AdaRmsNormDecoderLayer(TransformerDecoderLayer):
    """Decoder layer scaling its feed-forward input by an MLP of the delay, the Ada RMSNorm of Voxtral Realtime."""

    def __init__(
        self,
        encoder_dim: Optional[ReturnnDim],
        out_dim: ReturnnDim,
        *,
        ada_rms_norm_dim: int,
        delay_dim: Optional[ReturnnDim] = None,
        **kwargs,
    ):
        """
        :param encoder_dim: None, the layer has no cross-attention
        :param out_dim: the model dim
        :param ada_rms_norm_dim: width of the hidden layer of the delay MLP
        :param delay_dim: width of the delay embedding the MLP reads, None for the model dim
        :param kwargs: the options of :class:`TransformerDecoderLayer`
        """
        assert encoder_dim is None, "the delay takes the place of the cross-attention input"
        super().__init__(encoder_dim, out_dim, **kwargs)

        hidden_dim = ReturnnDim(ada_rms_norm_dim, name="ada_rms_norm_hidden")
        self.ada_rms_norm_in = rf.Linear(out_dim if delay_dim is None else delay_dim, hidden_dim, with_bias=False)
        self.ada_rms_norm_out = rf.Linear(hidden_dim, out_dim, with_bias=False)
        # zero init, the scale starts at one
        self.ada_rms_norm_out.weight.initial = 0.0

    def __call__(
        self, x: ReturnnTensor, *, spatial_dim: ReturnnDim, state: rf.State, encoder: Optional[rf.State] = None
    ) -> Tuple[ReturnnTensor, rf.State]:
        """
        Runs the plain self-attention and the feed-forward on its normed input times one plus the delay MLP.

        :param x: the layer input
        :param spatial_dim: the spatial dim, ``single_step_dim`` while decoding
        :param state: the self-attention state
        :param encoder: the input of the layer besides x, ``delay`` the embedding of the delay
        :return: (the layer output, the new state)
        """
        assert encoder is not None, "the layer takes the delay embedding as its encoder input"
        new_state = rf.State()

        # the modules of the plain layer with their call types
        self_att = cast(rf.CausalSelfAttention, self.self_att)
        self_att_norm = cast(Callable[[ReturnnTensor], ReturnnTensor], self.self_att_layer_norm)
        ff_norm = cast(Callable[[ReturnnTensor], ReturnnTensor], self.ff_layer_norm)
        ff = cast(Callable[[ReturnnTensor], ReturnnTensor], self.ff)

        # self-attention
        x_sa, new_state.self_att = self_att(self_att_norm(x), axis=spatial_dim, state=state.self_att)
        x_sa = rf.dropout(x_sa, self.dropout, axis=self.dropout_broadcast and self.out_dim)
        x = x_sa + x

        # feed-forward on the scaled norm
        scale = 1.0 + self.ada_rms_norm_out(rf.gelu(self.ada_rms_norm_in(encoder.delay)))
        x_ff = ff(ff_norm(x) * scale)
        x_ff = rf.dropout(x_ff, self.dropout, axis=self.dropout_broadcast and self.out_dim)
        x = x_ff + x

        return x, new_state
