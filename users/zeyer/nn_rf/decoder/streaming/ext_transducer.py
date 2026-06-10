"""
Extended-transducer slow+fast streaming decoder (slow-fast-rna project, variant A).

Port of the Zeyer et al. 2020 extended transducer (slow + fast RNNs) to Transformers,
RNA topology, on the chunked streaming encoder. Two stacks:

- **slow** (label-rate): a causal Transformer over the *emitted labels*; position ``u``
  consumes the previous label + the encoder frame at label ``u``'s emission time. Produces
  a slow state ``s_slow[u]``.
- **fast** (frame-rate): the framewise RNA fast stack (causal self-attn over frames + the
  current encoder frame ``h_t`` + previous-frame symbol), additionally injected with the
  *current* slow state ``s_slow[n_t]`` (``n_t`` = #labels emitted before frame ``t``).

Coupling is a gather: per frame ``t``, pick the slow state by ``n_t``. Training is one
teacher-forced pass (both stacks run in parallel, joined by the gather); everything is
derived in-graph from the per-frame RNA target (see ``segmentation.rna_frame_targets``).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, single_step_dim

from .framewise import FramewiseDecoderLayer
from .base import label_smoothed_log_probs

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import RecogDef


class ExtTransducerDecoder(rf.Module):
    """Slow (label-rate) + fast (frame-rate) stacks; slow state injected into the fast stack."""

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
        slow_num_layers: int = None,
        version: int = 1,
    ):
        super().__init__()
        # v1 = the pre-Transformer++ decoder (LayerNorm + abs sin pos-enc + non-gated FF); that code is gone.
        # v2 = RMSNorm + RoPE causal self-att + gated FF.
        # rf.build_dict hashes the dict, not the module source,
        # so this explicit version is what forces a new sis hash for the rewrite.
        assert version >= 2, "ExtTransducerDecoder v1 (pre-Transformer++) is removed; build with version=2"
        if isinstance(model_dim, int):
            model_dim = Dim(model_dim, name="dec_model")
        if isinstance(ff_dim, int):
            ff_dim = Dim(ff_dim, name="dec_ff")
        self.model_dim = model_dim
        self.vocab_dim = vocab_dim
        self.encoder_dim = encoder_dim
        self.blank_idx = eoc_idx  # the extra (last) vocab symbol is the RNA blank
        self.input_embedding_scale = model_dim.dimension**0.5
        self.dropout = dropout

        def _layers(n):
            return rf.Sequential(
                FramewiseDecoderLayer(model_dim, ff_dim, num_heads=num_heads, dropout=dropout, att_dropout=att_dropout)
                for _ in range(n)
            )

        # Slow (label-rate) stack.
        self.slow_embedding = rf.Embedding(vocab_dim, model_dim)
        self.slow_enc_proj = rf.Linear(encoder_dim, model_dim, with_bias=False)  # h at emission frame
        self.slow_layers = _layers(slow_num_layers or num_layers)
        self.slow_final_ln = rf.RMSNorm(model_dim)

        # Fast (frame-rate) stack.
        self.fast_embedding = rf.Embedding(vocab_dim, model_dim)
        self.fast_enc_proj = rf.Linear(encoder_dim, model_dim, with_bias=False)  # current frame h_t
        self.slow_to_fast = rf.Linear(model_dim, model_dim, with_bias=False)  # inject slow state
        self.fast_layers = _layers(num_layers)
        self.fast_final_ln = rf.RMSNorm(model_dim)
        self.logits = rf.Linear(model_dim, vocab_dim)

    def _initial_state(self, layers: rf.Sequential, *, batch_dims: Sequence[Dim]) -> rf.State:
        return rf.State({k: v.self_att.default_initial_state(batch_dims=batch_dims) for k, v in layers.items()})

    def slow_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
        return self._initial_state(self.slow_layers, batch_dims=batch_dims)

    def fast_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
        return self._initial_state(self.fast_layers, batch_dims=batch_dims)

    def slow_forward(
        self, prev_label: Tensor, h_emit: Tensor, *, spatial_dim: Dim, state: rf.State
    ) -> Tuple[Tensor, rf.State]:
        """label-rate step(s): prev label + encoder frame at emission -> slow state."""
        new_state = rf.State()
        x = self.slow_embedding(prev_label) * self.input_embedding_scale
        x = x + self.slow_enc_proj(h_emit)
        x = rf.dropout(x, self.dropout, axis=x.feature_dim)
        for name, layer in self.slow_layers.items():
            x, new_state[name] = layer(x, spatial_dim=spatial_dim, self_att_state=state[name])
        return self.slow_final_ln(x), new_state

    def fast_forward(
        self, prev_symbol: Tensor, enc_frame: Tensor, slow_state: Tensor, *, spatial_dim: Dim, state: rf.State
    ) -> Tuple[Tensor, rf.State]:
        """frame-rate step(s): prev symbol + current frame + current slow state -> logits."""
        new_state = rf.State()
        x = self.fast_embedding(prev_symbol) * self.input_embedding_scale
        x = x + self.fast_enc_proj(enc_frame)
        x = x + self.slow_to_fast(slow_state)
        x = rf.dropout(x, self.dropout, axis=x.feature_dim)
        for name, layer in self.fast_layers.items():
            x, new_state[name] = layer(x, spatial_dim=spatial_dim, self_att_state=state[name])
        x = self.fast_final_ln(x)
        return self.logits(x), new_state


def ext_transducer_train_forward(
    model,
    *,
    data: Tensor,
    data_spatial_dim: Dim,
    rna_targets: Tensor,
    rna_targets_spatial_dim: Dim,
) -> Dict[str, Tuple[Tensor, Dim]]:
    """
    Teacher-forced slow+fast training over the per-frame RNA target.

    From ``rna_targets`` (padded to the encoder length): n_t = #labels emitted before frame t
    (exclusive cumsum of non-blank); the emitted labels y and their encoder frames h_emit
    (masked_select). The slow stack runs over y -> s_slow; gathering s_slow by n_t gives each
    frame its current slow state, which the fast stack consumes alongside h_t.
    """
    collected_outputs = {} if model.enc_aux_logits else None
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    rna, _ = rf.replace_dim(rna_targets, in_dim=rna_targets_spatial_dim, out_dim=enc_spatial_dim)
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim) if data.feature_dim else data_spatial_dim)
    dec = model.decoder
    blank = model.blank_idx

    # Mask batch-padding frames: rf.cumsum does NOT respect the dynamic-dim mask, and the batched
    # rna pads shorter seqs with 0 (a valid label id), which would inflate n_t past the slow buffer
    # -> gather out-of-bounds. (masked_select below already masks padding, so y/h_emit are fine;
    # only the cumsum needs this.)
    valid_frame = rf.range_over_dim(enc_spatial_dim) < rf.copy_to_device(enc_spatial_dim.get_size_tensor())
    is_label = rf.logical_and(rna != blank, valid_frame)
    is_label_i = rf.cast(is_label, "int32")
    n_t = rf.cumsum(is_label_i, spatial_dim=enc_spatial_dim) - is_label_i  # [B, enc_spatial]: #labels before t

    # Emitted labels and their encoder frames (shared label_spatial_dim).
    y, label_spatial_dim = rf.masked_select(rna, mask=is_label, dims=[enc_spatial_dim])
    y.sparse_dim = model.target_dim_ext
    h_emit, _ = rf.masked_select(enc, mask=is_label, dims=[enc_spatial_dim], out_dim=label_spatial_dim)

    # Slow stack over the labels.
    slow_prev = rf.shift_right(y, axis=label_spatial_dim, pad_value=model.bos_idx)
    s_slow, _ = dec.slow_forward(
        slow_prev, h_emit, spatial_dim=label_spatial_dim, state=dec.slow_initial_state(batch_dims=batch_dims)
    )
    # Prepend a (zero) BOS slow state so gather index n_t in [0,U] maps 0->bos, k->s_slow[k-1].
    s_slow_ext, (label_ext_dim,) = rf.pad(s_slow, axes=[label_spatial_dim], padding=[(1, 0)], value=0.0)
    slow_per_frame = rf.gather(s_slow_ext, indices=n_t, axis=label_ext_dim)  # [B, enc_spatial, model_dim]

    # Fast stack over the frames, injected with the current slow state.
    fast_prev = rf.shift_right(rna, axis=enc_spatial_dim, pad_value=model.bos_idx)
    logits, _ = dec.fast_forward(
        fast_prev, enc, slow_per_frame, spatial_dim=enc_spatial_dim, state=dec.fast_initial_state(batch_dims=batch_dims)
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


def ext_transducer_training(*, model, data: Tensor, data_spatial_dim: Dim, targets: Tensor, targets_spatial_dim: Dim):
    """TrainDef: ``targets`` is the per-frame RNA alignment (the default target)."""
    losses = ext_transducer_train_forward(
        model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        rna_targets=targets,
        rna_targets_spatial_dim=targets_spatial_dim,
    )
    for name, (loss, norm_dim) in losses.items():
        loss.mark_as_loss(name, custom_inv_norm_factor=norm_dim.get_size_tensor(), use_normalized_loss=True)


ext_transducer_training.learning_rate_control_error_measure = "ce"


def model_recog(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Frame-synchronous greedy recog for the slow+fast extended transducer (beam size 1).

    The fast stack steps every encoder frame; the slow stack steps only on a (non-blank)
    emission. We accumulate emitted labels + their encoder frames in a fixed-size buffer
    (masked writes at position ``n_emitted``) and, whenever any element emits, re-run the
    slow stack over the buffer to refresh each element's current slow state ``s_slow[n_t-1]``
    (gathered at ``n_emitted-1``). Simple + correct; not the most efficient (re-runs the slow
    stack non-incrementally) -- fine for a first WER, optimize with an incremental slow cache
    later. Blanks are stripped from the output.
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
    model_dim, encoder_dim = dec.model_dim, dec.encoder_dim

    beam_dim = Dim(1, name="beam")
    bd = [beam_dim] + batch_dims
    label_dim = Dim(max_labels, name="emit_labels")  # fixed emitted-label buffer
    label_range = rf.range_over_dim(label_dim)

    fast_state = dec.fast_initial_state(batch_dims=bd)
    prev_symbol = rf.constant(bos, dims=bd, sparse_dim=model.target_dim_ext, dtype="int32")
    blank_t = rf.constant(blank, dims=bd, sparse_dim=model.target_dim_ext, dtype="int32")
    n_emitted = rf.constant(0, dims=bd, dtype="int32")
    current_slow = rf.constant(0.0, dims=bd + [model_dim])  # BOS slow state
    current_slow.feature_dim = model_dim
    emitted_labels = rf.constant(blank, dims=bd + [label_dim], sparse_dim=model.target_dim_ext, dtype="int32")
    emitted_h = rf.constant(0.0, dims=bd + [label_dim, encoder_dim])
    emitted_h.feature_dim = encoder_dim
    seq_log_prob = rf.constant(0.0, dims=bd)

    seq = TensorArray(prev_symbol)
    for t in range(T_max):
        t_t = rf.constant(t, dims=batch_dims, dtype="int32")
        valid = t_t < enc_lens
        idx = rf.where(valid, t_t, enc_lens - 1)
        enc_t = rf.gather(enc, indices=idx, axis=enc_spatial_dim)  # [batch, enc_dim]
        logits, fast_state = dec.fast_forward(
            prev_symbol, enc_t, current_slow, spatial_dim=single_step_dim, state=fast_state
        )
        log_probs = rf.log_softmax(logits, axis=model.target_dim_ext)
        sym = rf.cast(rf.reduce_argmax(log_probs, axis=model.target_dim_ext), "int32")
        sym.sparse_dim = model.target_dim_ext
        sym = rf.where(valid, sym, blank_t)
        is_label = sym != blank
        seq_log_prob = seq_log_prob + rf.where(valid, rf.gather(log_probs, indices=sym, axis=model.target_dim_ext), 0.0)
        seq = seq.push_back(sym)
        prev_symbol = sym

        # Append the emitted label + its encoder frame at position n_emitted (masked write).
        write = rf.logical_and(is_label, label_range == n_emitted)
        emitted_labels = rf.where(write, sym, emitted_labels)
        emitted_h = rf.where(write, enc_t, emitted_h)
        n_emitted = n_emitted + rf.cast(is_label, "int32")

        # If anything emitted, re-run the slow stack over the buffer and refresh current_slow.
        if bool(rf.reduce_all(rf.logical_not(is_label), axis=is_label.dims).raw_tensor):
            continue
        slow_prev = rf.shift_right(emitted_labels, axis=label_dim, pad_value=bos)
        s_slow, _ = dec.slow_forward(
            slow_prev, emitted_h, spatial_dim=label_dim, state=dec.slow_initial_state(batch_dims=bd)
        )
        last = rf.where(n_emitted > 0, n_emitted - 1, rf.zeros(bd, dtype="int32"))
        gathered = rf.gather(s_slow, indices=last, axis=label_dim)  # [.., model_dim]
        current_slow = rf.where(n_emitted > 0, gathered, current_slow)

    out_spatial_dim = Dim(T_max, name="out-spatial")
    aligned = seq.stack(axis=out_spatial_dim)
    seq_targets_out, seq_targets_spatial_dim = rf.masked_select(aligned, mask=aligned != blank, dims=[out_spatial_dim])
    seq_targets_out.sparse_dim = model.target_dim_ext
    return seq_targets_out, seq_log_prob, seq_targets_spatial_dim, beam_dim


model_recog: RecogDef
model_recog.output_with_beam = True
model_recog.output_blank_label = None
model_recog.batch_size_dependent = False
