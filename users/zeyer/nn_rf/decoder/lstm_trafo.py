"""
LSTM + Trafo hybrid decoder
"""

from __future__ import annotations
from typing import Optional, Union, Any, Sequence, Tuple, Dict, List
import functools
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder


class LstmTransformerDecoderV2(rf.Module):
    """
    LSTM Transformer hybrid decoder with cross attention to some encoder output.
    Only the LSTM has cross attention (MLP attention), the Transformer is decoder-only.

    The API should be compatible to :class:`TransformerDecoder`.

    V2: Simplify readout -> only one input projection before the Transformer.
    (V1 was unstable.)
    """

    def __init__(
        self,
        encoder_dim: Dim,
        vocab_dim: Dim,
        *,
        enc_key_total_dim: Union[Dim, int] = Dim(name="enc_key_total_dim", dimension=1024),
        att_num_heads: Union[Dim, int] = Dim(name="att_num_heads", dimension=1),
        lstm_dim: Union[Dim, int],  # can be small
        target_embed_dim: Union[Dim, int] = Dim(name="target_embed", dimension=640),
        transformer: Dict[str, Any],
    ):
        super().__init__()

        assert isinstance(encoder_dim, Dim)
        assert isinstance(vocab_dim, Dim)
        if isinstance(enc_key_total_dim, int):
            enc_key_total_dim = Dim(name="enc_key_total_dim", dimension=enc_key_total_dim)
        if isinstance(att_num_heads, int):
            att_num_heads = Dim(name="att_num_heads", dimension=att_num_heads)
        if isinstance(lstm_dim, int):
            lstm_dim = Dim(name="lstm", dimension=lstm_dim)
        if isinstance(target_embed_dim, int):
            target_embed_dim = Dim(name="target_embed", dimension=target_embed_dim)

        self.encoder_dim = encoder_dim
        self.vocab_dim = vocab_dim
        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        self.att_num_heads = att_num_heads
        self.dropout_broadcast = rf.dropout_broadcast_default()

        self.enc_ctx = rf.Linear(self.encoder_dim, enc_key_total_dim)
        self.inv_fertility = rf.Linear(self.encoder_dim, att_num_heads, with_bias=False)
        self.target_embed = rf.Embedding(vocab_dim, target_embed_dim)

        self.s = rf.ZoneoutLSTM(
            self.target_embed.out_dim + att_num_heads * self.encoder_dim,
            lstm_dim,
            zoneout_factor_cell=0.15,
            zoneout_factor_output=0.05,
            use_zoneout_output=False,  # like RETURNN/TF ZoneoutLSTM old default
            # parts_order="icfo",  # like RETURNN/TF ZoneoutLSTM
            # parts_order="ifco",
            parts_order="jifo",  # NativeLSTM (the code above converts it...)
            forget_bias=0.0,  # the code above already adds it during conversion
        )

        self.weight_feedback = rf.Linear(att_num_heads, enc_key_total_dim, with_bias=False)
        self.s_transformed = rf.Linear(self.s.out_dim, enc_key_total_dim, with_bias=False)
        self.energy = rf.Linear(enc_key_total_dim, att_num_heads, with_bias=False)

        self.transformer: TransformerDecoder = rf.build_from_dict(transformer, None, vocab_dim, input_embedding=None)
        self.input_proj = rf.Linear(
            self.s.out_dim + self.target_embed.out_dim + att_num_heads * self.encoder_dim, self.transformer.model_dim
        )

    def transform_encoder(self, encoder: Tensor, *, axis: Dim) -> rf.State:
        """encode, and extend the encoder output for things we need in the decoder (only for the LSTM part)"""
        enc_ctx = self.enc_ctx(encoder)
        inv_fertility = rf.sigmoid(self.inv_fertility(encoder))
        return rf.State(enc=encoder, enc_spatial_dim=axis, enc_ctx=enc_ctx, inv_fertility=inv_fertility)

    def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
        """Default initial state"""
        state = rf.State(
            rec=rf.State(
                s=self.s.default_initial_state(batch_dims=batch_dims),
                att=rf.zeros(list(batch_dims) + [self.att_num_heads * self.encoder_dim]),
                # Note: enc_spatial_dim missing! (Unfortunately the API of default_initial_state does not provide it.)
                accum_att_weights=rf.zeros(list(batch_dims) + [self.att_num_heads], feature_dim=self.att_num_heads),
            ),
            transformer=self.transformer.default_initial_state(batch_dims=batch_dims),
        )
        state.rec.att.feature_dim_axis = len(state.rec.att.dims) - 1
        return state

    def __call__(
        self,
        source: Tensor,
        *,
        spatial_dim: Dim,
        state: rf.State,
        encoder: rf.State,
        collected_outputs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, rf.State]:
        """
        forward, single step or whole sequence.

        :param source: labels
        :param spatial_dim: single_step_dim or spatial dim of source
        :param state: e.g. via :func:`default_initial_state`
        :param encoder: via :func:`transform_encoder`
        :return: logits, new state
        """
        enc_spatial_dim: Dim = encoder.enc_spatial_dim

        # accum_att_weights from default_initial_state could miss enc_spatial_dim
        if enc_spatial_dim not in state.rec.accum_att_weights.dims:
            state.rec.accum_att_weights = rf.expand_dim(state.rec.accum_att_weights, enc_spatial_dim)

        input_embed = self.target_embed(source)

        state_ = rf.State()

        if spatial_dim == single_step_dim:
            (s, att), state_.rec = self._loop_rec_step(input_embed=input_embed, state=state.rec, encoder=encoder)

        else:  # operating on whole sequence
            (s, att), state_.rec, _ = rf.scan(
                spatial_dim=spatial_dim,
                xs=input_embed,
                ys=self._loop_rec_step_output_templates(batch_dims=source.remaining_dims(spatial_dim)),
                initial=state.rec,
                body=functools.partial(self._loop_rec_step, encoder=encoder),
            )

        readout = self.input_proj(rf.concat_features(s, input_embed, att))

        if collected_outputs is not None:
            collected_outputs["-1"] = readout

        # Transformer. No cross-attention, only self-attention.
        logits, state_.transformer = self.transformer(
            source=readout, spatial_dim=spatial_dim, state=state.transformer, collected_outputs=collected_outputs
        )

        return logits, state_

    def _loop_rec_step(
        self, input_embed: Tensor, state: rf.State, *, encoder: rf.State
    ) -> Tuple[Tuple[rf.Tensor, rf.Tensor], rf.State]:
        enc_spatial_dim: Dim = encoder.enc_spatial_dim
        enc: Tensor = encoder.enc
        enc_ctx: Tensor = encoder.enc_ctx
        inv_fertility: Tensor = encoder.inv_fertility

        state_ = rf.State()

        prev_att = state.att

        s, state_.s = self.s(rf.concat_features(input_embed, prev_att), state=state.s, spatial_dim=single_step_dim)

        weight_feedback = self.weight_feedback(state.accum_att_weights)
        s_transformed = self.s_transformed(s)
        energy_in = enc_ctx + weight_feedback + s_transformed
        energy = self.energy(rf.tanh(energy_in))
        att_weights = rf.softmax(energy, axis=enc_spatial_dim)
        state_.accum_att_weights = state.accum_att_weights + att_weights * inv_fertility * 0.5
        att0 = rf.dot(att_weights, enc, reduce=enc_spatial_dim, use_mask=False)
        att0.feature_dim = self.encoder_dim
        att, _ = rf.merge_dims(att0, dims=(self.att_num_heads, self.encoder_dim))
        state_.att = att

        return (s, att), state_

    def _loop_rec_step_output_templates(self, batch_dims: List[Dim]) -> Tuple[Tensor, Tensor]:
        """loop step out"""
        return (
            Tensor("s", dims=batch_dims + [self.s.out_dim], dtype=rf.get_default_float_dtype(), feature_dim_axis=-1),
            Tensor(
                "att",
                dims=batch_dims + [self.att_num_heads * self.encoder_dim],
                dtype=rf.get_default_float_dtype(),
                feature_dim_axis=-1,
            ),
        )

    @property
    def final_layer_norm(self):
        """compat"""
        return self.transformer.final_layer_norm

    @property
    def logits(self):
        """compat"""
        return self.transformer.logits
