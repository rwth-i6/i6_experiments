"""
LSTM decoder, using cross-att to some encoder, following our old AED setup,
see :class:`i6_experiments.users.zeyer.experiments.exp2023_04_25_rf.conformer_import_moh_att_2023_06_30.Model`.
"""

from __future__ import annotations
from typing import Optional, Any, Sequence, Tuple, Dict, List
import functools
from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf


class LstmDecoder(rf.Module):
    """
    LSTM decoder with cross attention to some encoder output.

    The API should be compatible to :class:`TransformerDecoder`.
    """

    def __init__(
        self,
        in_dim: Dim,
        *,
        target_dim: Dim,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        enc_out_dim: Dim,
        enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
        att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
        att_dropout: float = 0.1,
    ):
        super().__init__()

        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        self.in_dim = in_dim
        self.target_dim = target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        self.enc_out_dim = enc_out_dim
        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        self.att_num_heads = att_num_heads
        self.att_dropout = att_dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()

        # https://github.com/rwth-i6/returnn-experiments/blob/master/2020-rnn-transducer/configs/base2.conv2l.specaug4a.ctc.devtrain.config

        self.enc_ctx = rf.Linear(self.enc_out_dim, enc_key_total_dim)
        self.enc_ctx_dropout = 0.2  # TODO use... # TODO make configurable

        self.inv_fertility = rf.Linear(self.enc_out_dim, att_num_heads, with_bias=False)

        self.target_embed = rf.Embedding(target_dim, Dim(name="target_embed", dimension=640))
        if config.float("embed_init_stddev", None):
            self.target_embed.weight.initial = rf.init.Normal(stddev=config.float("embed_init_stddev", 0.0))

        self.s = rf.ZoneoutLSTM(
            self.target_embed.out_dim + att_num_heads * self.enc_out_dim,
            Dim(name="lstm", dimension=1024),
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

        # TODO make this optional?
        self.readout_in = rf.Linear(
            self.s.out_dim + self.target_embed.out_dim + att_num_heads * self.enc_out_dim,
            Dim(name="readout", dimension=1024),
        )
        self.output_prob = rf.Linear(self.readout_in.out_dim // 2, target_dim)

    def transform_encoder(self, encoder: Tensor, *, axis: Dim) -> rf.State:
        """encode, and extend the encoder output for things we need in the decoder"""
        enc_ctx = self.enc_ctx(encoder)
        inv_fertility = rf.sigmoid(self.inv_fertility(encoder))
        return rf.State(enc=encoder, enc_spatial_dim=axis, enc_ctx=enc_ctx, inv_fertility=inv_fertility)

    def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
        """Default initial state"""
        state = rf.State(
            s=self.s.default_initial_state(batch_dims=batch_dims),
            att=rf.zeros(list(batch_dims) + [self.att_num_heads * self.enc_out_dim]),
            # Note: enc_spatial_dim missing! (Unfortunately the API of default_initial_state does not provide it.)
            accum_att_weights=rf.zeros(list(batch_dims) + [self.att_num_heads], feature_dim=self.att_num_heads),
        )
        state.att.feature_dim_axis = len(state.att.dims) - 1
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
        if enc_spatial_dim not in state.accum_att_weights.dims:
            state.accum_att_weights = rf.expand_dim(state.accum_att_weights, enc_spatial_dim)

        input_embed = self.target_embed(source)

        if spatial_dim == single_step_dim:
            (s, att), state_ = self._loop_step(input_embed=input_embed, state=state, encoder=encoder)

        else:  # operating on whole sequence
            (s, att), state_, _ = rf.scan(
                spatial_dim=spatial_dim,
                xs=source,
                ys=self._loop_step_output_templates(batch_dims=source.remaining_dims(spatial_dim)),
                initial=state,
                body=functools.partial(self._loop_step, encoder=encoder),
            )

        # decode logits
        readout_in = self.readout_in(rf.concat_features(s, input_embed, att))
        readout = rf.reduce_out(readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim)
        readout = rf.dropout(readout, drop_prob=0.3, axis=self.dropout_broadcast and readout.feature_dim)
        logits = self.output_prob(readout)

        return logits, state_

    def _loop_step(
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
        att0.feature_dim = self.enc_out_dim
        att, _ = rf.merge_dims(att0, dims=(self.att_num_heads, self.enc_out_dim))
        state_.att = att

        return (s, att), state_

    def _loop_step_output_templates(self, batch_dims: List[Dim]) -> Tuple[Tensor, Tensor]:
        """loop step out"""
        return (
            Tensor("s", dims=batch_dims + [self.s.out_dim], dtype=rf.get_default_float_dtype(), feature_dim_axis=-1),
            Tensor(
                "att",
                dims=batch_dims + [self.att_num_heads * self.enc_out_dim],
                dtype=rf.get_default_float_dtype(),
                feature_dim_axis=-1,
            ),
        )
