"""
Standard monotonic RNN-T streaming decoder (slow-fast-rna project) -- RNA topology, label-only prediction net.

The classic RNN-T decomposition on the chunked streaming encoder:
- **prediction network** (label-rate): a causal Transformer over the *emitted labels*
  (prev label -> prediction state ``p[u]``; label history only, no encoder conditioning);
- **joiner** (per frame): additive combination of the current encoder frame ``h_t`` and the
  current prediction state ``p[n_t]`` (``n_t`` = #labels emitted before frame ``t``),
  ReLU, then the output linear.

Unlike ``framewise`` (which mixes the previous label into the frame stack) and ``ext_transducer``
(which adds a full frame-rate *fast* Transformer stack), the label context here enters *only*
through the prediction network + joiner -- no frame-rate self-attention, no previous-symbol feedback.
So this isolates "what the fast stack adds over a plain joiner".

The prediction net depends on the label history only (standard monotonic RNN-T),
so the SAME model trains with either framewise-CE on our fixed RNA alignment (``rnnt_training`` here)
or the marginalized full-sum loss (``rnnt_fullsum.rnnt_fullsum_training``),
switching only the objective.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim

from .framewise import FramewiseDecoderLayer
from .base import label_smoothed_log_probs, mark_frame_error, rna_targets_on_enc_spatial

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import RecogDef


class RnntDecoder(rf.Module):
    """Prediction network (label-rate, label-history only) + additive joiner; monotonic RNA topology."""

    def __init__(
        self,
        *,
        encoder_dim: Dim,
        vocab_dim: Dim,
        chunk_size: int,
        eoc_idx: int,
        model_dim: int = 512,
        ff_dim: Optional[int] = None,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        att_dropout: float = 0.1,
        version: int = 1,
    ):
        super().__init__()
        assert version == 4, "RnntDecoder: standard monotonic RNN-T (label-only pred net); build with version==4"
        self.version = version
        chunk_size  # noqa  (frame-rate joiner -- no chunk masking; accepted for the StreamingModel interface)
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

        # Prediction network (label-rate) -- causal Transformer over the emitted labels (label history only).
        self.pred_embedding = rf.Embedding(vocab_dim, model_dim)
        self.pred_layers = rf.Sequential(
            FramewiseDecoderLayer(model_dim, ff_dim, num_heads=num_heads, dropout=dropout, att_dropout=att_dropout)
            for _ in range(num_layers)
        )
        self.pred_final_ln = rf.RMSNorm(model_dim)

        # Joiner: additive (encoder frame + prediction state), ReLU, output linear.
        self.joiner_enc_proj = rf.Linear(encoder_dim, model_dim, with_bias=False)
        self.logits = rf.Linear(model_dim, vocab_dim)

    def pred_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
        return rf.State(
            {k: v.self_att.default_initial_state(batch_dims=batch_dims) for k, v in self.pred_layers.items()}
        )

    def pred_forward(self, prev_label: Tensor, *, spatial_dim: Dim, state: rf.State) -> Tuple[Tensor, rf.State]:
        """label-rate step(s): prev label -> prediction state (label history only)."""
        new_state = rf.State()
        x = self.pred_embedding(prev_label) * self.input_embedding_scale
        x = rf.dropout(x, self.dropout, axis=x.feature_dim)
        for name, layer in self.pred_layers.items():
            x, new_state[name] = layer(x, spatial_dim=spatial_dim, self_att_state=state[name])
        return self.pred_final_ln(x), new_state

    def joiner(self, enc_frame: Tensor, pred_state: Tensor) -> Tensor:
        """Additive joiner: ReLU(enc_proj(enc_frame) + pred_state) -> logits over vocab+blank."""
        joint = rf.relu(self.joiner_enc_proj(enc_frame) + pred_state)
        return self.logits(joint)


def rnnt_train_forward(
    model,
    *,
    data: Tensor,
    data_spatial_dim: Dim,
    rna_targets: Tensor,
    rna_targets_spatial_dim: Dim,
) -> Dict[str, Tuple[Tensor, Dim]]:
    """Teacher-forced training over the per-frame RNA target (framewise CE).

    ``n_t`` = #labels before frame t; the prediction net runs over the emitted labels y; gathering by
    ``n_t`` gives each frame its current prediction state, which the joiner combines with the current
    encoder frame h_t.
    """
    collected_outputs = {} if model.enc_aux_logits else None
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    rna = rna_targets_on_enc_spatial(
        rna_targets, in_spatial_dim=rna_targets_spatial_dim, enc_spatial_dim=enc_spatial_dim, blank_idx=model.blank_idx
    )
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim) if data.feature_dim else data_spatial_dim)
    dec = model.decoder
    blank = model.blank_idx

    valid_frame = rf.range_over_dim(enc_spatial_dim) < rf.copy_to_device(enc_spatial_dim.get_size_tensor())
    is_label = rf.logical_and(rna != blank, valid_frame)
    is_label_i = rf.cast(is_label, "int32")
    n_t = rf.cumsum(is_label_i, spatial_dim=enc_spatial_dim) - is_label_i  # [B, enc_spatial]: #labels before t

    y, label_spatial_dim = rf.masked_select(rna, mask=is_label, dims=[enc_spatial_dim])
    y.sparse_dim = model.target_dim_ext

    # Prediction net over the emitted labels: g_u = f(BOS, y_1..y_u), the state after u emitted labels.
    pred_in, (label_ext_dim,) = rf.pad(y, axes=[label_spatial_dim], padding=[(1, 0)], value=model.bos_idx)
    pred_in.sparse_dim = model.target_dim_ext
    pred, _ = dec.pred_forward(
        pred_in, spatial_dim=label_ext_dim, state=dec.pred_initial_state(batch_dims=batch_dims)
    )
    # Frame t has n_t emitted labels, so it uses g_{n_t}, conditioned on all n_t labels including the most recent.
    pred_per_frame = rf.gather(pred, indices=n_t, axis=label_ext_dim)  # [B, enc_spatial, model_dim]

    logits = dec.joiner(enc, pred_per_frame)
    log_probs = rf.log_softmax(logits, axis=model.target_dim_ext)
    log_probs = label_smoothed_log_probs(log_probs, axis=model.target_dim_ext)  # config-gated, default off
    ce = rf.cross_entropy(target=rna, estimated=log_probs, estimated_type="log-probs", axis=model.target_dim_ext)
    mark_frame_error(log_probs, targets=rna, axis=model.target_dim_ext)
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


def rnnt_training(*, model, data: Tensor, data_spatial_dim: Dim, targets: Tensor, targets_spatial_dim: Dim):
    """TrainDef: ``targets`` is the per-frame RNA alignment (the default target)."""
    losses = rnnt_train_forward(
        model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        rna_targets=targets,
        rna_targets_spatial_dim=targets_spatial_dim,
    )
    for name, (loss, norm_dim) in losses.items():
        loss.mark_as_loss(name, custom_inv_norm_factor=norm_dim.get_size_tensor(), use_normalized_loss=True)


rnnt_training.learning_rate_control_error_measure = "ce"


def model_recog(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """Frame-synchronous greedy recog (beam size 1), monotonic (one label-or-blank per frame).

    Per frame: joiner(current encoder frame, current prediction state) -> argmax. On a non-blank
    emission, append to the label buffer + re-run the prediction net over the buffer to refresh the
    current prediction state (gathered at n_emitted-1). Blanks stripped from output.
    """
    from returnn.config import get_global_config
    from returnn.frontend.tensor_array import TensorArray

    config = get_global_config(return_empty_if_none=True)
    max_labels = config.int("max_labels", 0) or 200

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim) if data.feature_dim else data_spatial_dim)
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    enc_lens = rf.copy_to_device(enc_spatial_dim.get_size_tensor())
    t_max = int(rf.reduce_max(enc_lens, axis=enc_lens.dims).raw_tensor)
    dec = model.decoder
    blank, bos = model.blank_idx, model.bos_idx
    model_dim = dec.model_dim

    beam_dim = Dim(1, name="beam")
    bd = [beam_dim] + batch_dims
    label_dim = Dim(max_labels, name="emit_labels")  # fixed emitted-label buffer
    label_range = rf.range_over_dim(label_dim)

    blank_t = rf.constant(blank, dims=bd, sparse_dim=model.target_dim_ext, dtype="int32")
    n_emitted = rf.constant(0, dims=bd, dtype="int32")
    # g_0 = f(BOS): the prediction state after zero emitted labels.
    bos1_dim = Dim(1, name="bos1")
    bos1 = rf.constant(bos, dims=bd + [bos1_dim], sparse_dim=model.target_dim_ext, dtype="int32")
    pred0, _ = dec.pred_forward(bos1, spatial_dim=bos1_dim, state=dec.pred_initial_state(batch_dims=bd))
    current_pred = rf.gather(pred0, indices=rf.constant(0, dims=bd, dtype="int32"), axis=bos1_dim)
    current_pred.feature_dim = model_dim
    emitted_labels = rf.constant(blank, dims=bd + [label_dim], sparse_dim=model.target_dim_ext, dtype="int32")
    seq_log_prob = rf.constant(0.0, dims=bd)

    seq = TensorArray(rf.constant(bos, dims=bd, sparse_dim=model.target_dim_ext, dtype="int32"))
    for t in range(t_max):
        t_t = rf.constant(t, dims=batch_dims, dtype="int32")
        valid = t_t < enc_lens
        idx = rf.where(valid, t_t, enc_lens - 1)
        enc_t = rf.gather(enc, indices=idx, axis=enc_spatial_dim)  # [batch, enc_dim]
        logits = dec.joiner(enc_t, current_pred)
        log_probs = rf.log_softmax(logits, axis=model.target_dim_ext)
        sym = rf.cast(rf.reduce_argmax(log_probs, axis=model.target_dim_ext), "int32")
        sym.sparse_dim = model.target_dim_ext
        sym = rf.where(valid, sym, blank_t)
        is_label = sym != blank
        seq_log_prob = seq_log_prob + rf.where(valid, rf.gather(log_probs, indices=sym, axis=model.target_dim_ext), 0.0)
        seq = seq.push_back(sym)

        # Append the emitted label at position n_emitted (masked write).
        write = rf.logical_and(is_label, label_range == n_emitted)
        emitted_labels = rf.where(write, sym, emitted_labels)
        n_emitted = n_emitted + rf.cast(is_label, "int32")

        # If anything emitted, re-run the prediction net over the buffer and refresh current_pred.
        if bool(rf.reduce_all(rf.logical_not(is_label), axis=is_label.dims).raw_tensor):
            continue
        # g_{n_emitted} = f(BOS, y_1..y_{n_emitted}): predict over [BOS, emitted...], gather at n_emitted.
        pred_in, (ext_dim,) = rf.pad(emitted_labels, axes=[label_dim], padding=[(1, 0)], value=bos)
        pred_in.sparse_dim = model.target_dim_ext
        pred, _ = dec.pred_forward(pred_in, spatial_dim=ext_dim, state=dec.pred_initial_state(batch_dims=bd))
        current_pred = rf.gather(pred, indices=n_emitted, axis=ext_dim)  # [.., model_dim]

    out_spatial_dim = Dim(t_max, name="out-spatial")
    aligned = seq.stack(axis=out_spatial_dim)
    seq_targets_out, seq_targets_spatial_dim = rf.masked_select(aligned, mask=aligned != blank, dims=[out_spatial_dim])
    seq_targets_out.sparse_dim = model.target_dim_ext
    return seq_targets_out, seq_log_prob, seq_targets_spatial_dim, beam_dim


model_recog: RecogDef
model_recog.output_with_beam = True
model_recog.output_blank_label = None
model_recog.batch_size_dependent = False
