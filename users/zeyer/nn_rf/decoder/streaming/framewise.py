"""
Frame-synchronous (RNA) fast-only streaming decoder (slow-fast-rna project).

A single Transformer stack runs at encoder-frame rate: per encoder frame it emits one
label or blank (RNA topology -- the non-blank labels, no repeat collapse, are the
transcription). Each position consumes the current encoder frame ``h_t`` (added in) plus
the previous frame's symbol (teacher-forced in training, fed back at recog); causal
self-attention over frames carries the label/acoustic history. No cross-attention: the
chunked encoder already supplies streaming-causal acoustic context in ``h_t``.

This is the fast-only baseline and the *fast stack* that ``ext_transducer`` extends with a
slow label-rate stack. Target: the per-frame RNA alignment (``segmentation.rna_frame_targets``)
padded to the encoder's chunk-multiple length, so it lines up with the encoder output
frame-for-frame.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, TYPE_CHECKING

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, single_step_dim
from returnn.frontend.decoder.transformer import FeedForwardGated

from .base import label_smoothed_log_probs, mark_frame_error, rna_targets_on_enc_spatial

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import RecogDef


class FramewiseDecoderLayer(rf.Module):
    """Transformer++ (Llama-style) causal block, as in the AED/LM baseline decoders:
    RoPE causal self-attention (no bias), RMSNorm, gated FF; positions via RoPE (no abs pos enc).
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
        self.ff_ln = rf.RMSNorm(model_dim)
        self.ff = FeedForwardGated(model_dim, ff_dim=ff_dim, dropout=dropout)
        self.dropout = dropout

    def __call__(self, x: Tensor, *, spatial_dim: Dim, self_att_state: rf.State) -> Tuple[Tensor, rf.State]:
        h, new_state = self.self_att(self.self_att_ln(x), spatial_dim, state=self_att_state)
        x = x + rf.dropout(h, self.dropout, axis=h.feature_dim)
        x = x + self.ff(self.ff_ln(x))
        return x, new_state


class FramewiseDecoder(rf.Module):
    """Frame-synchronous RNA decoder: causal self-attn over frames + per-frame encoder input."""

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
        delay_frames: int = 0,  # DSM audio->text delay in encoder frames (0 = label sits at its acoustic frame)
        factorized_blank: bool = False,  # separate sigmoid blank head instead of blank as a vocab entry
        version: int = 1,
    ):
        super().__init__()
        # v1 = the pre-Transformer++ decoder (LayerNorm + abs sin pos-enc + non-gated FF); that code is gone.
        # v2 = RMSNorm + RoPE causal self-att + gated FF.
        # rf.build_dict hashes the dict, not the module source,
        # so this explicit version is what forces a new sis hash for the rewrite.
        assert version >= 2, "FramewiseDecoder v1 (pre-Transformer++) is removed; build with version=2"
        if isinstance(model_dim, int):
            model_dim = Dim(model_dim, name="dec_model")
        if isinstance(ff_dim, int):
            ff_dim = Dim(ff_dim, name="dec_ff")
        self.model_dim = model_dim
        self.vocab_dim = vocab_dim
        self.encoder_dim = encoder_dim
        self.chunk_size = chunk_size
        self.blank_idx = eoc_idx  # the extra (last) vocab symbol is the RNA blank here
        self.delay_frames = delay_frames

        self.input_embedding = rf.Embedding(vocab_dim, model_dim)
        self.enc_proj = rf.Linear(encoder_dim, model_dim, with_bias=False)
        self.input_embedding_scale = model_dim.dimension**0.5
        self.dropout = dropout

        self.layers = rf.Sequential(
            FramewiseDecoderLayer(model_dim, ff_dim, num_heads=num_heads, dropout=dropout, att_dropout=att_dropout)
            for _ in range(num_layers)
        )
        self.final_ln = rf.RMSNorm(model_dim)
        self.logits = rf.Linear(model_dim, vocab_dim)
        # Separated blank, as in ...exp2024_04_23_baselines.aed.log_probs_with_eos_separated:
        # the blank logit is the existing entry at blank_idx, so there are no extra parameters.
        self.factorized_blank = factorized_blank
        self.returns_log_probs = factorized_blank  # then __call__ returns log-probs, not logits

    def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
        return rf.State({k: v.self_att.default_initial_state(batch_dims=batch_dims) for k, v in self.layers.items()})

    def __call__(
        self,
        source: Tensor,
        enc_frame: Tensor,
        *,
        spatial_dim: Dim,
        state: rf.State,
    ) -> Tuple[Tensor, rf.State]:
        """
        :param source: previous-frame symbol(s), sparse over ``vocab_dim``, on ``spatial_dim``.
        :param enc_frame: encoder output aligned to ``source`` (same ``spatial_dim``).
        :param spatial_dim: the frame axis (``enc_spatial_dim`` in training, ``single_step_dim`` at recog).
        :param state: self-attn state (carries the RoPE positions).
        """
        new_state = rf.State()
        x = self.input_embedding(source) * self.input_embedding_scale
        x = x + self.enc_proj(enc_frame)
        x = rf.dropout(x, self.dropout, axis=x.feature_dim)

        for name, layer in self.layers.items():
            x, new_state[name] = layer(x, spatial_dim=spatial_dim, self_att_state=state[name])
        x = self.final_ln(x)
        logits = self.logits(x)
        if not self.factorized_blank:
            return logits, new_state
        return log_probs_with_blank_separated(logits, vocab_dim=self.vocab_dim, blank_idx=self.blank_idx), new_state


def log_probs_with_blank_separated(logits: Tensor, *, vocab_dim: Dim, blank_idx: int) -> Tensor:
    """
    Log-probs with the blank probability separated from the label distribution.

    log P(blank) = log_sigmoid(logit_blank),
    log P(y) = log_sigmoid(-logit_blank) + log_softmax over the labels.
    The blank logit is the existing entry at ``blank_idx``, so this adds no parameters.
    Mirrors ...exp2024_04_23_baselines.aed.log_probs_with_eos_separated,
    which splits off index 0; here blank is the last entry.
    """
    assert blank_idx == vocab_dim.dimension - 1, f"blank {blank_idx} is not the last of {vocab_dim}"
    labels_dim = Dim(vocab_dim.dimension - 1, name="labels_wo_blank")
    blank_feat_dim = Dim(1, name="blank_feat")
    logits_wo_blank, logits_blank = rf.split(logits, axis=vocab_dim, out_dims=[labels_dim, blank_feat_dim])
    log_probs_wo_blank = rf.log_softmax(logits_wo_blank, axis=labels_dim)
    log_probs_blank = rf.log_sigmoid(logits_blank)
    log_probs_not_blank = rf.squeeze(rf.log_sigmoid(-logits_blank), axis=blank_feat_dim)
    log_probs, _ = rf.concat(
        (log_probs_wo_blank + log_probs_not_blank, labels_dim),
        (log_probs_blank, blank_feat_dim),
        out_dim=vocab_dim,
    )
    log_probs.feature_dim = vocab_dim
    return log_probs


def _decoder_log_probs(model, logits: Tensor) -> Tensor:
    """
    Log-probs from the decoder output.

    With a factorized blank head the decoder already returns log-probs
    (blank sigmoid times label softmax), so a second log_softmax would be wrong.
    """
    if model.decoder.returns_log_probs:
        return logits
    return rf.log_softmax(logits, axis=model.target_dim_ext)


def framewise_train_forward(
    model,
    *,
    data: Tensor,
    data_spatial_dim: Dim,
    rna_targets: Optional[Tensor] = None,
    rna_targets_spatial_dim: Optional[Dim] = None,
    labels: Optional[Tensor] = None,
    labels_spatial_dim: Optional[Dim] = None,
) -> Dict[str, Tuple[Tensor, Dim]]:
    """
    Teacher-forced frame-synchronous RNA training.

    ``rna_targets`` is the per-frame RNA alignment (label or blank per frame), padded by
    the dataset to the encoder's chunk-multiple length, so it matches ``enc_spatial_dim``
    frame-for-frame (we just re-tag the dim). Loss = framewise CE; optional aux CTC on the
    blank-removed labels (the transcription).

    :return: dict ``name -> (loss, inv_norm_spatial_dim)``.
    """
    collected_outputs = {} if (model.enc_aux_logits or rna_targets is None) else None
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    if rna_targets is not None:
        # Re-align the RNA target onto the encoder length (pad blank / cut blank padding),
        # so the encoder chunking is free to differ from the dataset's fixed pad-to-chunk-multiple
        # (dynamic chunk pools, offline, ...).
        rna = rna_targets_on_enc_spatial(
            rna_targets,
            in_spatial_dim=rna_targets_spatial_dim,
            enc_spatial_dim=enc_spatial_dim,
            blank_idx=model.blank_idx,
        )
    else:
        # On-the-fly alignment: Viterbi path of the model's OWN top aux CTC head,
        # label_loop=False == RNA topology, so blank removal (no collapse) gives back the labels.
        # Self-aligned, so it tracks the chunked encoder's own timing
        # instead of the offline base's fixed alignment.
        aux_logits = model.aux_logits_from_collected_outputs(model.enc_aux_logits[-1], collected_outputs)
        rna = rf.ctc_best_path(
            logits=aux_logits,
            targets=labels,
            input_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=labels_spatial_dim,
            blank_index=model.blank_idx,
            label_loop=False,
        )
        rna.sparse_dim = model.target_dim_ext

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim) if data.feature_dim else data_spatial_dim)

    # DSM delay: the text stream lags the audio by ``delay_frames`` encoder frames. Extend the encoder
    # by that many silence (zero) frames on the right and shift the RNA target right by the same amount
    # (prepend blanks) onto the extended axis, so every label must be emitted ``delay_frames`` frames
    # after its acoustic evidence (and the tail flushes into the trailing silence). delay == 0 -> plain
    # framewise (the else branch is numerically identical to the pre-delay code).
    delay = getattr(model.decoder, "delay_frames", 0)
    if delay > 0:
        enc_dec, (dec_spatial_dim,) = rf.pad(enc, axes=[enc_spatial_dim], padding=[(0, delay)], value=0.0)
        rna_dec, _ = rf.pad(
            rna, axes=[enc_spatial_dim], padding=[(delay, 0)], value=model.blank_idx, out_dims=[dec_spatial_dim]
        )
    else:
        enc_dec, dec_spatial_dim, rna_dec = enc, enc_spatial_dim, rna

    # Teacher forcing: decoder input at frame t is the previous frame's symbol (BOS at t=0).
    input_labels = rf.shift_right(rna_dec, axis=dec_spatial_dim, pad_value=model.bos_idx)

    state = model.decoder.default_initial_state(batch_dims=batch_dims)
    logits, _ = model.decoder(input_labels, enc_dec, spatial_dim=dec_spatial_dim, state=state)
    log_probs = _decoder_log_probs(model, logits)
    log_probs = label_smoothed_log_probs(log_probs, axis=model.target_dim_ext)  # config-gated, default off
    ce = rf.cross_entropy(target=rna_dec, estimated=log_probs, estimated_type="log-probs", axis=model.target_dim_ext)
    mark_frame_error(log_probs, targets=rna_dec, axis=model.target_dim_ext)
    losses: Dict[str, Tuple[Tensor, Dim]] = {"ce": (ce, dec_spatial_dim)}

    # Aux CTC on the blank-removed labels (== transcription), over the final encoder output.
    if model.enc_aux_logits:
        raw_targets, raw_spatial_dim = rf.masked_select(rna, mask=rna != model.blank_idx, dims=[enc_spatial_dim])
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


def framewise_training(*, model, data: Tensor, data_spatial_dim: Dim, targets: Tensor, targets_spatial_dim: Dim):
    """TrainDef: ``targets`` is the per-frame RNA alignment (the default target)."""
    losses = framewise_train_forward(
        model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        rna_targets=targets,
        rna_targets_spatial_dim=targets_spatial_dim,
    )
    for name, (loss, norm_dim) in losses.items():
        loss.mark_as_loss(
            name, custom_inv_norm_factor=norm_dim.get_size_tensor(device=loss.device), use_normalized_loss=True
        )


framewise_training.learning_rate_control_error_measure = "ce"


def framewise_ctc_align_training(
    *, model, data: Tensor, data_spatial_dim: Dim, targets: Tensor, targets_spatial_dim: Dim
):
    """
    TrainDef: ``targets`` is the plain transcript (target_mode="labels");
    the RNA alignment is derived on-the-fly from the model's own aux CTC head each step.
    """
    losses = framewise_train_forward(
        model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        labels=targets,
        labels_spatial_dim=targets_spatial_dim,
    )
    for name, (loss, norm_dim) in losses.items():
        loss.mark_as_loss(
            name, custom_inv_norm_factor=norm_dim.get_size_tensor(device=loss.device), use_normalized_loss=True
        )


framewise_ctc_align_training.learning_rate_control_error_measure = "ce"


def framewise_scaled_training(*, model, data: Tensor, data_spatial_dim: Dim, targets: Tensor, targets_spatial_dim: Dim):
    """TrainDef: like :func:`framewise_training`, but the main framewise CE is weighted by
    ``framewise_main_loss_scale`` from the config (aux CTC unscaled), as in ...streaming.rnnt_scaled.
    """
    from returnn.config import get_global_config

    scale = get_global_config().float("framewise_main_loss_scale", 1.0)
    losses = framewise_train_forward(
        model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        rna_targets=targets,
        rna_targets_spatial_dim=targets_spatial_dim,
    )
    for name, (loss, norm_dim) in losses.items():
        loss.mark_as_loss(
            name,
            scale=scale if name == "ce" else 1.0,
            custom_inv_norm_factor=norm_dim.get_size_tensor(device=loss.device),
            use_normalized_loss=True,
        )


framewise_scaled_training.learning_rate_control_error_measure = "ce"


def model_recog(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Frame-synchronous greedy recognition (beam size 1).

    Runs exactly ``T_enc`` steps (one per encoder frame); each step feeds the previous
    emitted symbol + the current encoder frame, argmaxes over labels+blank, and feeds it
    back. Blanks are removed from the output, giving the plain spm transcription (rendered
    via ``target_dim_ext``'s vocab; the blank == the last/extra symbol).

    :return: (seq_targets {batch,beam,out_spatial} sparse over target_dim_ext,
              seq_log_prob {batch,beam}, out_spatial_dim, beam_dim)
    """
    from returnn.frontend.tensor_array import TensorArray

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim) if data.feature_dim else data_spatial_dim)
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    enc_lens = rf.copy_to_device(enc_spatial_dim.get_size_tensor())  # [batch]
    T_max = int(rf.reduce_max(enc_lens, axis=enc_lens.dims).raw_tensor)

    delay = getattr(model.decoder, "delay_frames", 0)
    T_total = T_max + delay  # `delay` extra flush steps for the delayed-tail labels

    beam_dim = Dim(1, name="beam")
    batch_dims_ = [beam_dim] + batch_dims
    state = model.decoder.default_initial_state(batch_dims=batch_dims_)
    prev = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim_ext, dtype="int32")
    blank = rf.constant(model.blank_idx, dims=batch_dims_, sparse_dim=model.target_dim_ext, dtype="int32")
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    seq = TensorArray(prev)
    for t in range(T_total):
        t_t = rf.constant(t, dims=batch_dims, dtype="int32")  # [batch]
        audio_valid = t_t < enc_lens  # a real encoder frame is available for this seq
        emit_valid = t_t < enc_lens + delay  # emit window incl. the `delay`-frame flush tail
        idx = rf.where(audio_valid, t_t, enc_lens - 1)  # clip out-of-range frames
        enc_t = rf.gather(enc, indices=idx, axis=enc_spatial_dim)  # [batch, enc_dim]
        enc_t = rf.where(audio_valid, enc_t, 0.0)  # silence (zero) frames during flush + padding
        logits, state = model.decoder(prev, enc_t, spatial_dim=single_step_dim, state=state)
        log_probs = _decoder_log_probs(model, logits)
        sym = rf.cast(rf.reduce_argmax(log_probs, axis=model.target_dim_ext), "int32")
        sym.sparse_dim = model.target_dim_ext
        sym = rf.where(emit_valid, sym, blank)  # outside the emit window -> blank (dropped later)
        lp_sym = rf.gather(log_probs, indices=sym, axis=model.target_dim_ext)
        seq_log_prob = seq_log_prob + rf.where(emit_valid, lp_sym, 0.0)
        prev = sym
        seq = seq.push_back(sym)

    out_spatial_dim = Dim(T_total, name="out-spatial")
    aligned = seq.stack(axis=out_spatial_dim)  # [beam, batch, T_max] over target_dim_ext

    # Strip blanks -> plain spm label sequence (variable length per seq).
    seq_targets_out, seq_targets_spatial_dim = rf.masked_select(
        aligned, mask=aligned != model.blank_idx, dims=[out_spatial_dim]
    )
    seq_targets_out.sparse_dim = model.target_dim_ext
    return seq_targets_out, seq_log_prob, seq_targets_spatial_dim, beam_dim


model_recog: RecogDef
model_recog.output_with_beam = True
model_recog.output_blank_label = None
model_recog.batch_size_dependent = False


def model_recog_beam(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Frame-synchronous beam search (cf. :func:`model_recog`); beam_size from config (default 12),
    beam_size=1 == greedy. Blanks stripped.
    With ctc_scale > 0 the encoder's aux CTC head is added per frame (joint decoding).
    """
    from returnn.config import get_global_config
    from .beam_search import frame_sync_beam_search

    config = get_global_config(return_empty_if_none=True)
    beam_size = config.int("beam_size", 12)
    # The aux CTC head is frame-synchronous over the same encoder frames as this decoder,
    # so its log-probs can be added per frame. 0 keeps the decoder-only search.
    ctc_scale = config.float("ctc_scale", 0.0)

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim) if data.feature_dim else data_spatial_dim)
    collected_outputs = {} if ctc_scale else None
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    frame_scores = None
    if ctc_scale:
        aux_logits = model.aux_logits_from_collected_outputs(model.enc_aux_logits[-1], collected_outputs)
        ctc_log_probs = rf.log_softmax(aux_logits, axis=model.wb_target_dim)
        # both label spaces are the spm labels plus blank last, but they are distinct Dims
        frame_scores = rf.replace_dim_v2(ctc_log_probs, in_dim=model.wb_target_dim, out_dim=model.target_dim_ext)
        frame_scores = frame_scores * ctc_scale
    delay = model.decoder.delay_frames

    def _step(prev, enc_t, state):
        logits, new_state = model.decoder(prev, enc_t, spatial_dim=single_step_dim, state=state)
        log_probs = _decoder_log_probs(model, logits)
        return log_probs, new_state

    return frame_sync_beam_search(
        batch_dims=batch_dims,
        target_dim_ext=model.target_dim_ext,
        bos_idx=model.bos_idx,
        blank_idx=model.blank_idx,
        enc=enc,
        enc_spatial_dim=enc_spatial_dim,
        beam_size=beam_size,
        init_state=lambda bd: model.decoder.default_initial_state(batch_dims=bd),
        step=_step,
        num_flush_frames=delay,
        recomb=config.typed_value("recog_recomb", "max"),
        frame_scores=frame_scores,
    )


model_recog_beam: RecogDef
model_recog_beam.output_with_beam = True
model_recog_beam.output_blank_label = None
model_recog_beam.batch_size_dependent = False


def model_recog_beam_rescore_check(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Search-score sanity check: run the beam search, then teacher-force the returned alignment back
    through the decoder and re-sum its per-frame log-probs. Returns the alignment (blanks kept) with
    seq_log_prob = search_score - rescore, which must be ~0 for every hyp if the search score
    accumulation + backtracking are correct. Not a WER recog (blanks kept on purpose); read the raw
    scores from the forward output.
    """
    from returnn.config import get_global_config
    from .beam_search import frame_sync_beam_search

    config = get_global_config(return_empty_if_none=True)
    beam_size = config.int("beam_size", 12)

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim) if data.feature_dim else data_spatial_dim)
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    delay = getattr(model.decoder, "delay_frames", 0)

    def _step(prev, enc_t, state):
        logits, new_state = model.decoder(prev, enc_t, spatial_dim=single_step_dim, state=state)
        log_probs = _decoder_log_probs(model, logits)
        return log_probs, new_state

    align, search_score, out_spatial_dim, beam_dim = frame_sync_beam_search(
        batch_dims=batch_dims,
        target_dim_ext=model.target_dim_ext,
        bos_idx=model.bos_idx,
        blank_idx=model.blank_idx,
        enc=enc,
        enc_spatial_dim=enc_spatial_dim,
        beam_size=beam_size,
        init_state=lambda bd: model.decoder.default_initial_state(batch_dims=bd),
        step=_step,
        num_flush_frames=delay,
        recomb=config.typed_value("recog_recomb", "max"),
        return_alignment=True,
    )

    # Re-feed the returned alignment (same per-frame masking as the search) and re-sum its log-probs.
    bd = [beam_dim] + batch_dims
    enc_lens = rf.copy_to_device(enc_spatial_dim.get_size_tensor())
    t_total = int(rf.reduce_max(enc_lens, axis=enc_lens.dims).raw_tensor) + delay
    state = model.decoder.default_initial_state(batch_dims=bd)
    prev = rf.constant(model.bos_idx, dims=bd, sparse_dim=model.target_dim_ext, dtype="int32")
    rescore = rf.constant(0.0, dims=bd)
    for t in range(t_total):
        t_t = rf.constant(t, dims=batch_dims, dtype="int32")
        audio_valid = t_t < enc_lens
        emit_valid = t_t < enc_lens + delay
        idx = rf.where(audio_valid, t_t, enc_lens - 1)
        enc_t = rf.where(audio_valid, rf.gather(enc, indices=idx, axis=enc_spatial_dim), 0.0)
        a_t = rf.gather(align, indices=rf.constant(t, dims=batch_dims, dtype="int32"), axis=out_spatial_dim)
        a_t.sparse_dim = model.target_dim_ext
        logits, state = model.decoder(prev, enc_t, spatial_dim=single_step_dim, state=state)
        lp = rf.log_softmax(logits, axis=model.target_dim_ext)
        rescore = rescore + rf.where(emit_valid, rf.gather(lp, indices=a_t, axis=model.target_dim_ext), 0.0)
        prev = a_t

    # Return the alignment (blanks kept); blanks are removed downstream as a post-proc (rna_collapse).
    return align, search_score - rescore, out_spatial_dim, beam_dim


model_recog_beam_rescore_check: RecogDef
model_recog_beam_rescore_check.output_with_beam = True
model_recog_beam_rescore_check.output_blank_label = None
model_recog_beam_rescore_check.batch_size_dependent = False
