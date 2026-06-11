"""
Two-tower slow+fast streaming decoder (slow-fast-rna project, variant B).

Same fast-slow umbrella as :mod:`ext_transducer`, but the coupling is **cross-attention**
rather than slow-state injection:

- **text stack** (label-rate): causal Transformer over the emitted labels; produces per-label
  text states ``t_u`` (a label-context representation).
- **speech stack** (frame-rate): causal Transformer over encoder frames, with an additional
  cross-attention to the text stack. Frame ``t`` attends only to text states whose label was
  emitted before ``t`` (mask: ``text-position <= n_t``). A BOS text state is prepended so
  frames with ``n_t = 0`` always have at least one key (no all-masked softmax).

Both stacks have label context; the speech stack outputs the per-frame RNA logits. Coupling
is the cross-attn mask (key label idx <= query ``n_t``); v1 is **single-direction**
(speech ← text). Adding text ← speech (bidirectional) is a clean follow-up.

Training is one teacher-forced pass; recog uses a fixed-size emit buffer + re-run the text
stack over the prefix on each emission, mirroring the ext_transducer recog pattern.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, single_step_dim
from returnn.frontend.decoder.transformer import FeedForwardGated

from .framewise import FramewiseDecoderLayer
from .cross_attn import ChunkMaskedCrossAttention
from .base import label_smoothed_log_probs, rna_targets_on_enc_spatial

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import RecogDef


class SpeechTextCrossLayer(rf.Module):
    """Speech-stack layer: causal self-attn over frames + cross-attn to text states + FFN.

    Transformer++ components as in :class:`FramewiseDecoderLayer` (RoPE self-att, RMSNorm, gated FF).
    """

    def __init__(self, model_dim: Dim, ff_dim: Optional[Dim], *, num_heads: int, dropout: float, att_dropout: float):
        super().__init__()
        self.self_att_ln = rf.RMSNorm(model_dim)
        self.self_att = rf.RotaryPosCausalSelfAttention(
            model_dim,
            proj_dim=model_dim,
            key_dim_total=model_dim,
            value_dim_total=model_dim,
            num_heads=num_heads,
            with_bias=False,
            att_dropout=att_dropout,
        )
        self.cross_att_ln = rf.RMSNorm(model_dim)
        # encoder_dim=model_dim: keys/values come from text states (also model_dim), not the audio encoder.
        self.cross_att = ChunkMaskedCrossAttention(
            model_dim,
            model_dim,
            key_dim_total=model_dim,
            value_dim_total=model_dim,
            num_heads=num_heads,
            att_dropout=att_dropout,
        )
        self.ff_ln = rf.RMSNorm(model_dim)
        self.ff = FeedForwardGated(model_dim, ff_dim=ff_dim, dropout=dropout)
        self.dropout = dropout

    def __call__(
        self,
        x: Tensor,
        *,
        spatial_dim: Dim,
        self_att_state: rf.State,
        keys: Tensor,
        values: Tensor,
        text_ext_spatial_dim: Dim,
        query_n_t: Tensor,
        key_label_idx: Tensor,
    ) -> Tuple[Tensor, rf.State]:
        h, new_state = self.self_att(self.self_att_ln(x), spatial_dim, state=self_att_state)
        x = x + rf.dropout(h, self.dropout, axis=h.feature_dim)
        h = self.cross_att(
            self.cross_att_ln(x),
            keys=keys,
            values=values,
            enc_spatial_dim=text_ext_spatial_dim,
            query_chunk_idx=query_n_t,
            key_chunk_idx=key_label_idx,
        )
        x = x + rf.dropout(h, self.dropout, axis=h.feature_dim)
        x = x + self.ff(self.ff_ln(x))
        return x, new_state


class TwoTowerDecoder(rf.Module):
    """Two-tower fast-slow decoder: text + speech stacks, speech cross-attends to text."""

    def __init__(
        self,
        *,
        encoder_dim: Dim,
        vocab_dim: Dim,
        chunk_size: int,
        eoc_idx: int,
        model_dim: int = 512,
        ff_dim: Optional[int] = None,  # None -> FeedForwardGated default (Llama-style ~8/3 * model_dim)
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        att_dropout: float = 0.1,
        text_num_layers: int = None,
        version: int = 1,
    ):
        super().__init__()
        # v1 = the pre-Transformer++ decoder (LayerNorm + abs sin pos-enc + non-gated FF); that code is gone.
        # v2 = RMSNorm + RoPE causal self-att + gated FF.
        # rf.build_dict hashes the dict, not the module source,
        # so this explicit version is what forces a new sis hash for the rewrite.
        assert version >= 2, "TwoTowerDecoder v1 (pre-Transformer++) is removed; build with version=2"
        if isinstance(model_dim, int):
            model_dim = Dim(model_dim, name="dec_model")
        if isinstance(ff_dim, int):
            ff_dim = Dim(ff_dim, name="dec_ff")
        self.model_dim = model_dim
        self.vocab_dim = vocab_dim
        self.encoder_dim = encoder_dim
        self.blank_idx = eoc_idx
        self.input_embedding_scale = model_dim.dimension**0.5
        self.dropout = dropout

        # Text stack (label-rate, plain causal self-attn).
        self.text_embedding = rf.Embedding(vocab_dim, model_dim)
        self.text_layers = rf.Sequential(
            FramewiseDecoderLayer(model_dim, ff_dim, num_heads=num_heads, dropout=dropout, att_dropout=att_dropout)
            for _ in range(text_num_layers or num_layers)
        )
        self.text_final_ln = rf.RMSNorm(model_dim)

        # Speech stack (frame-rate, self-attn + cross-attn to text + FFN).
        self.speech_embedding = rf.Embedding(vocab_dim, model_dim)
        self.speech_enc_proj = rf.Linear(encoder_dim, model_dim, with_bias=False)
        self.speech_layers = rf.Sequential(
            SpeechTextCrossLayer(model_dim, ff_dim, num_heads=num_heads, dropout=dropout, att_dropout=att_dropout)
            for _ in range(num_layers)
        )
        self.speech_final_ln = rf.RMSNorm(model_dim)
        self.logits = rf.Linear(model_dim, vocab_dim)

    def _initial_state(self, layers: rf.Sequential, *, batch_dims: Sequence[Dim]) -> rf.State:
        return rf.State({k: v.self_att.default_initial_state(batch_dims=batch_dims) for k, v in layers.items()})

    def text_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
        return self._initial_state(self.text_layers, batch_dims=batch_dims)

    def speech_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
        return self._initial_state(self.speech_layers, batch_dims=batch_dims)

    def text_forward(self, prev_label: Tensor, *, spatial_dim: Dim, state: rf.State) -> Tuple[Tensor, rf.State]:
        """Text stack: causal self-attn over labels -> text states."""
        new_state = rf.State()
        x = self.text_embedding(prev_label) * self.input_embedding_scale
        x = rf.dropout(x, self.dropout, axis=x.feature_dim)
        for name, layer in self.text_layers.items():
            x, new_state[name] = layer(x, spatial_dim=spatial_dim, self_att_state=state[name])
        return self.text_final_ln(x), new_state

    def transform_text(self, text_states: Tensor, *, axis: Dim) -> rf.State:
        """Precompute cross-attention keys/values from text states (one pair per speech layer)."""
        return rf.State(
            {k: layer.cross_att.transform_encoder(text_states, axis=axis) for k, layer in self.speech_layers.items()}
        )

    def speech_forward(
        self,
        prev_symbol: Tensor,
        enc_frame: Tensor,
        *,
        spatial_dim: Dim,
        state: rf.State,
        text_kv: rf.State,
        text_ext_spatial_dim: Dim,
        query_n_t: Tensor,
        key_label_idx: Tensor,
    ) -> Tuple[Tensor, rf.State]:
        """Speech stack: causal self-attn + cross-attn to text + FFN -> logits."""
        new_state = rf.State()
        x = self.speech_embedding(prev_symbol) * self.input_embedding_scale
        x = x + self.speech_enc_proj(enc_frame)
        x = rf.dropout(x, self.dropout, axis=x.feature_dim)
        for name, layer in self.speech_layers.items():
            keys, values = text_kv[name]
            x, new_state[name] = layer(
                x,
                spatial_dim=spatial_dim,
                self_att_state=state[name],
                keys=keys,
                values=values,
                text_ext_spatial_dim=text_ext_spatial_dim,
                query_n_t=query_n_t,
                key_label_idx=key_label_idx,
            )
        x = self.speech_final_ln(x)
        return self.logits(x), new_state


def two_tower_train_forward(
    model,
    *,
    data: Tensor,
    data_spatial_dim: Dim,
    rna_targets: Tensor,
    rna_targets_spatial_dim: Dim,
) -> Dict[str, Tuple[Tensor, Dim]]:
    """
    Teacher-forced two-tower training over the per-frame RNA target.

    Derive n_t (= #labels before frame t, cumsum over valid frames) and the emitted labels y.
    Run the text stack over y -> text states; prepend a BOS state and use that as the cross-attn
    keys/values for the speech stack, which cross-attn-masks to ``key_label_idx <= n_t``.
    """
    collected_outputs = {} if model.enc_aux_logits else None
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    # Re-align the RNA target onto the encoder length (pad blank / cut blank padding),
    # so the encoder chunking is free to differ from the dataset's fixed pad-to-chunk-multiple.
    rna = rna_targets_on_enc_spatial(
        rna_targets, in_spatial_dim=rna_targets_spatial_dim, enc_spatial_dim=enc_spatial_dim, blank_idx=model.blank_idx
    )
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim) if data.feature_dim else data_spatial_dim)
    dec = model.decoder
    blank = model.blank_idx

    # Mask batch-padding before cumsum (RETURNN pads sparse targets with 0 = valid label id;
    # rf.cumsum doesn't respect the dynamic-dim mask, so unmasked is_label would overcount).
    valid_frame = rf.range_over_dim(enc_spatial_dim) < rf.copy_to_device(enc_spatial_dim.get_size_tensor())
    is_label = rf.logical_and(rna != blank, valid_frame)
    is_label_i = rf.cast(is_label, "int32")
    n_t = rf.cumsum(is_label_i, spatial_dim=enc_spatial_dim) - is_label_i  # [B, enc_spatial]: #labels before t

    # Emitted labels (text stack input).
    y, label_spatial_dim = rf.masked_select(rna, mask=is_label, dims=[enc_spatial_dim])
    y.sparse_dim = model.target_dim_ext

    # Text stack over the emitted labels.
    text_prev = rf.shift_right(y, axis=label_spatial_dim, pad_value=model.bos_idx)
    text_states, _ = dec.text_forward(
        text_prev, spatial_dim=label_spatial_dim, state=dec.text_initial_state(batch_dims=batch_dims)
    )

    # Prepend a (zero) BOS text state: frames with n_t=0 then always have >=1 key in cross-attn,
    # and text_ext positions map [0->BOS, k>=1 -> text_{k-1}]; the cross-attn mask
    # ``key_label_idx <= n_t`` admits BOS + the first n_t emitted labels for each frame.
    text_ext, (text_ext_spatial_dim,) = rf.pad(text_states, axes=[label_spatial_dim], padding=[(1, 0)], value=0.0)
    text_kv = dec.transform_text(text_ext, axis=text_ext_spatial_dim)
    key_label_idx = rf.range_over_dim(text_ext_spatial_dim)  # [text_ext]: 0..U positions

    # Speech stack over frames; cross-attn mask: key_label_idx <= n_t (per frame).
    speech_prev = rf.shift_right(rna, axis=enc_spatial_dim, pad_value=model.bos_idx)
    logits, _ = dec.speech_forward(
        speech_prev,
        enc,
        spatial_dim=enc_spatial_dim,
        state=dec.speech_initial_state(batch_dims=batch_dims),
        text_kv=text_kv,
        text_ext_spatial_dim=text_ext_spatial_dim,
        query_n_t=n_t,
        key_label_idx=key_label_idx,
    )

    log_probs = rf.log_softmax(logits, axis=model.target_dim_ext)
    log_probs = label_smoothed_log_probs(log_probs, axis=model.target_dim_ext)  # config-gated, default off
    ce = rf.cross_entropy(target=rna, estimated=log_probs, estimated_type="log-probs", axis=model.target_dim_ext)
    losses: Dict[str, Tuple[Tensor, Dim]] = {"ce": (ce, enc_spatial_dim)}

    if model.enc_aux_logits:
        raw_targets, raw_spatial_dim = rf.masked_select(rna, mask=rna != blank, dims=[enc_spatial_dim])
        raw_targets.sparse_dim = model.target_dim
        losses.update(
            model.aux_ctc_losses(
                collected_outputs=collected_outputs,
                raw_targets=raw_targets,
                raw_spatial_dim=raw_spatial_dim,
                enc_spatial_dim=enc_spatial_dim,
            )
        )
    return losses


def two_tower_training(*, model, data: Tensor, data_spatial_dim: Dim, targets: Tensor, targets_spatial_dim: Dim):
    """TrainDef: ``targets`` is the per-frame RNA alignment (the default target)."""
    losses = two_tower_train_forward(
        model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        rna_targets=targets,
        rna_targets_spatial_dim=targets_spatial_dim,
    )
    for name, (loss, norm_dim) in losses.items():
        loss.mark_as_loss(name, custom_inv_norm_factor=norm_dim.get_size_tensor(), use_normalized_loss=True)


two_tower_training.learning_rate_control_error_measure = "ce"


def model_recog(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Frame-synchronous greedy recog for two_tower (beam size 1).

    Mirrors ext_transducer's loop, but instead of gathering a slow state, the speech stack
    cross-attends to text states. On each emission we re-run the text stack over the emitted-
    label prefix to refresh the text states + cross-attn keys/values; the cross-attn mask
    (``key_label_idx <= n_emitted`` per element) admits BOS + the emitted labels so far.
    """
    from returnn.config import get_global_config
    from returnn.frontend.tensor_array import TensorArray

    config = get_global_config(return_empty_if_none=True)
    max_labels = config.int("max_labels", 0) or 200

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim) if data.feature_dim else data_spatial_dim)
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    enc_lens = rf.copy_to_device(enc_spatial_dim.get_size_tensor())
    T_max = int(rf.reduce_max(enc_lens, axis=enc_lens.dims).raw_tensor)
    dec = model.decoder
    blank, bos = model.blank_idx, model.bos_idx
    model_dim = dec.model_dim

    beam_dim = Dim(1, name="beam")
    bd = [beam_dim] + batch_dims
    label_dim = Dim(max_labels, name="emit_labels")
    text_ext_dim = label_dim + 1  # BOS + emitted labels buffer
    label_range = rf.range_over_dim(label_dim)
    key_label_idx = rf.range_over_dim(text_ext_dim)  # 0..max_labels positions in text_ext

    speech_state = dec.speech_initial_state(batch_dims=bd)
    prev_symbol = rf.constant(bos, dims=bd, sparse_dim=model.target_dim_ext, dtype="int32")
    blank_t = rf.constant(blank, dims=bd, sparse_dim=model.target_dim_ext, dtype="int32")
    n_emitted = rf.constant(0, dims=bd, dtype="int32")
    emitted_labels = rf.constant(blank, dims=bd + [label_dim], sparse_dim=model.target_dim_ext, dtype="int32")
    seq_log_prob = rf.constant(0.0, dims=bd)

    # Initial text_ext = all-zero (BOS row + zero pads); initial text_kv from this.
    text_ext = rf.constant(0.0, dims=bd + [text_ext_dim, model_dim])
    text_ext.feature_dim = model_dim
    text_kv = dec.transform_text(text_ext, axis=text_ext_dim)

    seq = TensorArray(prev_symbol)
    for t in range(T_max):
        t_t = rf.constant(t, dims=batch_dims, dtype="int32")
        valid = t_t < enc_lens
        idx = rf.where(valid, t_t, enc_lens - 1)
        enc_t = rf.gather(enc, indices=idx, axis=enc_spatial_dim)  # [batch, enc_dim]

        # query_n_t per element (per frame, scalar over bd): the current frame's n_t = n_emitted.
        # We need it broadcast to a single-step query for the cross-attn (no spatial axis).
        query_n_t = n_emitted  # [bd] int

        logits, speech_state = dec.speech_forward(
            prev_symbol,
            enc_t,
            spatial_dim=single_step_dim,
            state=speech_state,
            text_kv=text_kv,
            text_ext_spatial_dim=text_ext_dim,
            query_n_t=query_n_t,
            key_label_idx=key_label_idx,
        )
        log_probs = rf.log_softmax(logits, axis=model.target_dim_ext)
        sym = rf.cast(rf.reduce_argmax(log_probs, axis=model.target_dim_ext), "int32")
        sym.sparse_dim = model.target_dim_ext
        sym = rf.where(valid, sym, blank_t)
        is_label = sym != blank
        seq_log_prob = seq_log_prob + rf.where(valid, rf.gather(log_probs, indices=sym, axis=model.target_dim_ext), 0.0)
        seq = seq.push_back(sym)
        prev_symbol = sym

        # Append emitted label at position n_emitted (masked write).
        write = rf.logical_and(is_label, label_range == n_emitted)
        emitted_labels = rf.where(write, sym, emitted_labels)
        n_emitted = n_emitted + rf.cast(is_label, "int32")

        # If anything emitted this frame, re-run the text stack over the buffer + refresh text_kv.
        if bool(rf.reduce_all(rf.logical_not(is_label), axis=is_label.dims).raw_tensor):
            continue
        text_prev = rf.shift_right(emitted_labels, axis=label_dim, pad_value=bos)
        text_states_buf, _ = dec.text_forward(
            text_prev, spatial_dim=label_dim, state=dec.text_initial_state(batch_dims=bd)
        )
        # text_ext = BOS row + text_states_buf (over text_ext_dim of size label_dim+1).
        text_ext, _ = rf.pad(text_states_buf, axes=[label_dim], padding=[(1, 0)], value=0.0, out_dims=[text_ext_dim])
        text_kv = dec.transform_text(text_ext, axis=text_ext_dim)

    out_spatial_dim = Dim(T_max, name="out-spatial")
    aligned = seq.stack(axis=out_spatial_dim)
    seq_targets_out, seq_targets_spatial_dim = rf.masked_select(aligned, mask=aligned != blank, dims=[out_spatial_dim])
    seq_targets_out.sparse_dim = model.target_dim_ext
    return seq_targets_out, seq_log_prob, seq_targets_spatial_dim, beam_dim


model_recog: RecogDef
model_recog.output_with_beam = True
model_recog.output_blank_label = None
model_recog.batch_size_dependent = False
