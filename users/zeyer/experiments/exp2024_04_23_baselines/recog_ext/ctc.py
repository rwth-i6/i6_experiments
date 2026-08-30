"""
CTC decoding with neural LM
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.zeyer.model_interfaces import RecogDef

from ..ctc import Model


def model_recog(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Time-synchronous beam search decoding with CTC + neural LM.
    No recombination here.

    Function is run within RETURNN.

    Note, for debugging, see :func:`model_recog_debug`.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    import returnn
    from returnn.config import get_global_config

    config = get_global_config()
    beam_size = config.int("beam_size", 12)
    version = config.int("recog_version", 1)
    assert version == 9

    # Optional LM softmax temperature T: score the external LM as log_softmax(lm_logits / T).
    # Default 1.0 is a no-op reproducing the untempered LM exactly
    # (hash-stable for existing recogs that don't set the key).
    lm_softmax_temperature = config.typed_value("lm_softmax_temperature", 1.0)

    def _lm_log_softmax(logits: Tensor) -> Tensor:
        if lm_softmax_temperature != 1.0:
            logits = logits / lm_softmax_temperature
        return rf.log_softmax(logits, axis=model.target_dim)

    # RETURNN version is like "1.20250115.110555"
    # There was an important fix in 2025-01-17 affecting masked_scatter.
    # And another important fix in 2025-01-24 affecting masked_scatter for old PyTorch versions.
    assert tuple(int(n) for n in returnn.__version__.split(".")) >= (1, 20250125, 0), returnn.__version__

    label_log_prob, _, enc_spatial_dim = model.encode_and_get_ctc_log_probs(data, in_spatial_dim=data_spatial_dim)
    batch_dims = label_log_prob.remaining_dims((enc_spatial_dim, label_log_prob.feature_dim))

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    # The label log probs include the AM and the (scaled) prior.
    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob_ta = TensorArray.unstack(label_log_prob, axis=enc_spatial_dim)  # t -> Batch, VocabWB

    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)  # Batch, InBeam -> Vocab
    target_wb = rf.constant(
        model.blank_idx, dims=batch_dims_, sparse_dim=model.wb_target_dim
    )  # Batch, InBeam -> VocabWB

    if getattr(model, "lm", None) is None:
        lm: Optional[TransformerDecoder] = None
        lm_scale: Optional[float] = None
        lm_log_probs = None
        lm_state = None
        labelwise_prior = None

    else:
        # We usually have TransformerDecoder, but any other type would also be ok when it has the same API.
        # noinspection PyUnresolvedReferences
        lm: TransformerDecoder = model.lm
        # noinspection PyUnresolvedReferences
        lm_scale: float = model.lm_scale

        # noinspection PyUnresolvedReferences
        labelwise_prior: Optional[rf.Parameter] = model.labelwise_prior

        lm_state = lm.default_initial_state(batch_dims=batch_dims_)  # Batch, InBeam, ...
        lm_logits, lm_state = lm(
            target,
            spatial_dim=single_step_dim,
            state=lm_state,
        )  # Batch, InBeam, Vocab / ...
        lm_log_probs = _lm_log_softmax(lm_logits)  # Batch, InBeam, Vocab
        lm_log_probs *= lm_scale
        if labelwise_prior is not None:
            lm_log_probs -= labelwise_prior  # prior scale already applied

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets_wb = []
    seq_backrefs = []
    for t in range(max_seq_len):
        prev_target = target
        prev_target_wb = target_wb

        # Explicit broadcast (each source misses a dim the other has: beam vs VocabWB)
        seq_log_prob = rf.combine_bc(seq_log_prob, "+", label_log_prob_ta[t])  # Batch, InBeam, VocabWB

        if lm is not None:
            # Now add LM score. If prev align label (target_wb) is blank or != cur, add LM score, otherwise 0.
            seq_log_prob += rf.where(
                (prev_target_wb == model.blank_idx)
                | rf.compare_bc(prev_target_wb, "!=", rf.range_over_dim(model.wb_target_dim)),
                _target_dense_extend_blank(
                    lm_log_probs,
                    target_dim=model.target_dim,
                    wb_target_dim=model.wb_target_dim,
                    blank_idx=model.blank_idx,
                    value=0.0,
                ),
                0.0,
            )  # Batch, InBeam, VocabWB

        seq_log_prob, (backrefs, target_wb), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, model.wb_target_dim]
        )
        # seq_log_prob, backrefs, target_wb: Batch, Beam
        # backrefs -> InBeam.
        # target_wb -> VocabWB.
        seq_targets_wb.append(target_wb)
        seq_backrefs.append(backrefs)

        if lm is not None:
            lm_log_probs = rf.gather(lm_log_probs, indices=backrefs)  # Batch, Beam, Vocab
            lm_state = rf.nested.gather_nested(lm_state, indices=backrefs)
        prev_target = rf.gather(prev_target, indices=backrefs)  # Batch, Beam -> Vocab
        prev_target_wb = rf.gather(prev_target_wb, indices=backrefs)  # Batch, Beam -> VocabWB
        got_new_label = (target_wb != model.blank_idx) & (target_wb != prev_target_wb)  # Batch, Beam -> 0|1
        target = rf.where(
            got_new_label,
            _target_remove_blank(
                target_wb, target_dim=model.target_dim, wb_target_dim=model.wb_target_dim, blank_idx=model.blank_idx
            ),
            prev_target,
        )  # Batch, Beam -> Vocab

        if lm is not None:
            got_new_label_cpu = rf.copy_to_device(got_new_label, "cpu")
            if got_new_label_cpu.raw_tensor.sum().item() > 0:
                (target_, lm_state_), packed_new_label_dim, packed_new_label_dim_map = rf.nested.masked_select_nested(
                    (target, lm_state),
                    mask=got_new_label,
                    mask_cpu=got_new_label_cpu,
                    dims=batch_dims + [beam_dim],
                )
                # packed_new_label_dim_map: old dim -> new dim. see _masked_select_prepare_dims
                assert packed_new_label_dim.get_dim_value() > 0

                lm_logits_, lm_state_ = lm(
                    target_,
                    spatial_dim=single_step_dim,
                    state=lm_state_,
                )  # Flat_Batch_Beam, Vocab / ...
                lm_log_probs_ = _lm_log_softmax(lm_logits_)  # Flat_Batch_Beam, Vocab
                lm_log_probs_ *= lm_scale
                if labelwise_prior is not None:
                    lm_log_probs_ -= labelwise_prior  # prior scale already applied

                lm_log_probs, lm_state = rf.nested.masked_scatter_nested(
                    (lm_log_probs_, lm_state_),
                    (lm_log_probs, lm_state),
                    mask=got_new_label,
                    mask_cpu=got_new_label_cpu,
                    dims=batch_dims + [beam_dim],
                    in_dim=packed_new_label_dim,
                    masked_select_dim_map=packed_new_label_dim_map,
                )  # Batch, Beam, Vocab / ...

    if lm is not None:
        # seq_log_prob, lm_log_probs: Batch, Beam
        # Add LM EOS score at the end.
        lm_eos_score = rf.gather(lm_log_probs, indices=model.eos_idx, axis=model.target_dim)
        seq_log_prob += lm_eos_score  # Batch, Beam -> VocabWB

    # Backtrack via backrefs, resolve beams.
    seq_targets_wb_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target_wb in zip(seq_backrefs[::-1], seq_targets_wb[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_wb_.insert(0, rf.gather(target_wb, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets_wb__ = TensorArray(seq_targets_wb_[0])
    for target_wb in seq_targets_wb_:
        seq_targets_wb__ = seq_targets_wb__.push_back(target_wb)
    out_spatial_dim = enc_spatial_dim
    seq_targets_wb = seq_targets_wb__.stack(axis=out_spatial_dim)

    return seq_targets_wb, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = True  # our models currently just are batch-size-dependent...


def model_recog_with_recomb(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Time-synchronous beam search decoding with CTC + neural LM.
    With recombination of paths with the same label seq (sum or max).

    Function is run within RETURNN.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    import returnn
    from returnn.config import get_global_config
    from i6_experiments.users.zeyer.nn_rf.soft_collapse_repeated import soft_collapse_repeated

    config = get_global_config()
    beam_size = config.int("beam_size", 12)
    version = config.int("recog_version", 1)
    assert version == 10
    recomb = config.typed_value("recog_recomb", None)  # None, "max", "sum"
    ctc_soft_collapse_threshold = config.typed_value("ctc_soft_collapse_threshold", None)
    ctc_soft_collapse_reduce_type = config.typed_value("ctc_soft_collapse_reduce_type", "logmeanexp")

    # Optional LM softmax temperature T: score the external LM as log_softmax(lm_logits / T).
    # Default 1.0 is a no-op reproducing the untempered LM exactly
    # (hash-stable for existing first-pass recogs: they don't set the key,
    # and this recog def is hashed by reference, not by source body).
    lm_softmax_temperature = config.typed_value("lm_softmax_temperature", 1.0)

    def _lm_log_softmax(logits: Tensor) -> Tensor:
        if lm_softmax_temperature != 1.0:
            logits = logits / lm_softmax_temperature
        return rf.log_softmax(logits, axis=model.target_dim)

    # RETURNN version is like "1.20250115.110555"
    # There was an important fix in 2025-01-17 affecting masked_scatter.
    # And another important fix in 2025-01-24 affecting masked_scatter for old PyTorch versions.
    assert tuple(int(n) for n in returnn.__version__.split(".")) >= (1, 20250125, 0), returnn.__version__

    # The label log probs include the AM and the (scaled) prior.
    label_log_prob, _, enc_spatial_dim = model.encode_and_get_ctc_log_probs(data, in_spatial_dim=data_spatial_dim)
    batch_dims = label_log_prob.remaining_dims((enc_spatial_dim, label_log_prob.feature_dim))

    if ctc_soft_collapse_threshold is not None:
        label_log_prob, enc_spatial_dim = soft_collapse_repeated(
            label_log_prob,
            spatial_dim=enc_spatial_dim,
            classes_dim=model.wb_target_dim,
            threshold=ctc_soft_collapse_threshold,
            reduce_type=ctc_soft_collapse_reduce_type,
        )

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    neg_inf = float("-inf")
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=neg_inf),
    )
    label_log_prob_ta = TensorArray.unstack(label_log_prob, axis=enc_spatial_dim)  # t -> Batch, VocabWB

    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)  # Batch, InBeam -> Vocab
    target_wb = rf.constant(
        model.blank_idx, dims=batch_dims_, sparse_dim=model.wb_target_dim
    )  # Batch, InBeam -> VocabWB

    seq_label = _seq_label_history_init_state(vocab_dim=model.target_dim, batch_dims=batch_dims_)

    if getattr(model, "lm", None) is None:
        lm: Optional[TransformerDecoder] = None
        lm_scale: Optional[float] = None
        lm_log_probs = None
        lm_state = None
        labelwise_prior = None

    else:
        # We usually have TransformerDecoder, but any other type would also be ok when it has the same API.
        # noinspection PyUnresolvedReferences
        lm: TransformerDecoder = model.lm
        # noinspection PyUnresolvedReferences
        lm_scale: float = model.lm_scale

        # noinspection PyUnresolvedReferences
        labelwise_prior: Optional[rf.Parameter] = model.labelwise_prior

        lm_state = lm.default_initial_state(batch_dims=batch_dims_)  # Batch, InBeam, ...
        lm_logits, lm_state = lm(
            target,
            spatial_dim=single_step_dim,
            state=lm_state,
        )  # Batch, InBeam, Vocab / ...
        lm_log_probs = _lm_log_softmax(lm_logits)  # Batch, InBeam, Vocab
        lm_log_probs *= lm_scale
        if labelwise_prior is not None:
            lm_log_probs -= labelwise_prior  # prior scale already applied

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets_wb = []
    seq_backrefs = []
    for t in range(max_seq_len):
        prev_target = target
        prev_target_wb = target_wb

        # Explicit broadcast (each source misses a dim the other has: beam vs VocabWB)
        seq_log_prob = rf.combine_bc(seq_log_prob, "+", label_log_prob_ta[t])  # Batch, InBeam, VocabWB

        if lm is not None:
            # Now add LM score. If prev align label (target_wb) is blank or != cur, add LM score, otherwise 0.
            seq_log_prob += rf.where(
                (prev_target_wb == model.blank_idx)
                | rf.compare_bc(prev_target_wb, "!=", rf.range_over_dim(model.wb_target_dim)),
                _target_dense_extend_blank(
                    lm_log_probs,
                    target_dim=model.target_dim,
                    wb_target_dim=model.wb_target_dim,
                    blank_idx=model.blank_idx,
                    value=0.0,
                ),
                0.0,
            )  # Batch, InBeam, VocabWB

        seq_log_prob, (backrefs, target_wb), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, model.wb_target_dim]
        )
        # seq_log_prob, backrefs, target_wb: Batch, Beam
        # backrefs -> InBeam.
        # target_wb -> VocabWB.
        seq_targets_wb.append(target_wb)
        seq_backrefs.append(backrefs)

        if lm is not None:
            lm_log_probs = rf.gather(lm_log_probs, indices=backrefs)  # Batch, Beam, Vocab
            lm_state = rf.nested.gather_nested(lm_state, indices=backrefs)
        seq_label = rf.nested.gather_nested(seq_label, indices=backrefs)

        prev_target = rf.gather(prev_target, indices=backrefs)  # Batch, Beam -> Vocab
        prev_target_wb = rf.gather(prev_target_wb, indices=backrefs)  # Batch, Beam -> VocabWB

        got_new_label: Tensor = (target_wb != model.blank_idx) & (target_wb != prev_target_wb)  # Batch, Beam -> 0|1
        target = rf.where(
            got_new_label,
            _target_remove_blank(
                target_wb, target_dim=model.target_dim, wb_target_dim=model.wb_target_dim, blank_idx=model.blank_idx
            ),
            prev_target,
        )  # Batch, Beam -> Vocab
        got_new_label_cpu = rf.copy_to_device(got_new_label, "cpu")
        if not rf.is_executing_eagerly() or got_new_label_cpu.raw_tensor.sum().item() > 0:
            seq_label = rf.nested.mask_nested(
                _seq_label_append(seq_label, target),
                mask=got_new_label,
                mask_cpu=got_new_label_cpu,
                mask_value=seq_label,
            )

            # Recombine paths with the same label seq.
            if not recomb:
                pass
            elif recomb in ("max", "sum"):
                # Set seq_log_prob for batch entries to neg_inf if they have the same label seq.
                same_seq_labels, beam_dual_dim = _same_seq_labels(
                    seq_label.history, spatial_dim=seq_label.hist_dim, beam_dim=beam_dim
                )
                seq_log_prob_ext = rf.where(
                    same_seq_labels, rf.replace_dim_v2(seq_log_prob, in_dim=beam_dim, out_dim=beam_dual_dim), neg_inf
                )  # Batch, Beam, BeamDual
                if recomb == "sum":
                    seq_log_prob = rf.reduce_logsumexp(seq_log_prob_ext, axis=beam_dual_dim)  # Batch, Beam
                argmax_seq_log_prob = rf.reduce_argmax(seq_log_prob_ext, axis=beam_dual_dim)  # Batch, Beam -> BeamDual
                mask = argmax_seq_log_prob == rf.range_over_dim(beam_dim)  # Batch, Beam -> 0|1
                seq_log_prob = rf.where(mask, seq_log_prob, neg_inf)
                got_new_label = got_new_label & mask  # don't re-eval the LM when masked out
                got_new_label_cpu = rf.copy_to_device(got_new_label, "cpu")
            else:
                raise ValueError(f"invalid recog_recomb {recomb!r}")

        if lm is not None:
            if not rf.is_executing_eagerly() or got_new_label_cpu.raw_tensor.sum().item() > 0:
                (target_, lm_state_), packed_new_label_dim, packed_new_label_dim_map = rf.nested.masked_select_nested(
                    (target, lm_state),
                    mask=got_new_label,
                    mask_cpu=got_new_label_cpu,
                    dims=batch_dims + [beam_dim],
                )
                # packed_new_label_dim_map: old dim -> new dim. see _masked_select_prepare_dims
                if not rf.is_static_traceable():
                    assert packed_new_label_dim.get_dim_value() > 0

                lm_logits_, lm_state_ = lm(
                    target_,
                    spatial_dim=single_step_dim,
                    state=lm_state_,
                )  # Flat_Batch_Beam, Vocab / ...
                lm_log_probs_ = _lm_log_softmax(lm_logits_)  # Flat_Batch_Beam, Vocab
                lm_log_probs_ *= lm_scale
                if labelwise_prior is not None:
                    lm_log_probs_ -= labelwise_prior  # prior scale already applied

                lm_log_probs, lm_state = rf.nested.masked_scatter_nested(
                    (lm_log_probs_, lm_state_),
                    (lm_log_probs, lm_state),
                    mask=got_new_label,
                    mask_cpu=got_new_label_cpu,
                    dims=batch_dims + [beam_dim],
                    in_dim=packed_new_label_dim,
                    masked_select_dim_map=packed_new_label_dim_map,
                )  # Batch, Beam, Vocab / ...

    if lm is not None:
        # seq_log_prob, lm_log_probs: Batch, Beam
        # Add LM EOS score at the end.
        lm_eos_score = rf.gather(lm_log_probs, indices=model.eos_idx, axis=model.target_dim)
        seq_log_prob += lm_eos_score  # Batch, Beam -> VocabWB

    # Backtrack via backrefs, resolve beams.
    seq_targets_wb_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target_wb in zip(seq_backrefs[::-1], seq_targets_wb[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_wb_.insert(0, rf.gather(target_wb, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets_wb__ = TensorArray(seq_targets_wb_[0])
    for target_wb in seq_targets_wb_:
        seq_targets_wb__ = seq_targets_wb__.push_back(target_wb)
    out_spatial_dim = enc_spatial_dim
    seq_targets_wb = seq_targets_wb__.stack(axis=out_spatial_dim)

    # Select valid.
    mask = rf.is_finite(seq_log_prob)  # Batch, Beam
    mask_cpu = rf.copy_to_device(mask, "cpu")
    (seq_targets_wb, seq_log_prob, out_spatial_dim), beam_dim, _ = rf.nested.masked_select_nested(
        (seq_targets_wb, seq_log_prob, out_spatial_dim), mask=mask, mask_cpu=mask_cpu, dims=[beam_dim]
    )

    return seq_targets_wb, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog_with_recomb: RecogDef[Model]
model_recog_with_recomb.output_with_beam = True
model_recog_with_recomb.output_blank_label = "<blank>"
model_recog_with_recomb.batch_size_dependent = True  # our models currently just are batch-size-dependent...


def model_recog_with_recomb_while_loop(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Like :func:`model_recog_with_recomb`, but the frame loop is :func:`rf.while_loop`, i.e. in the graph.
    Same structure as :func:`recog_ext.aed_ctc.model_recog_with_recomb_while_loop`,
    with the external LM in place of the AED decoder.

    The eager version cannot be built on a graph backend: its trip count is
    ``int(enc_spatial_dim.get_dim_value())``, a host-side value.
    Two more things there are not expressible with fixed shapes, and are replaced here:

    - the LM runs for every beam, not only for those with a new label
      (:func:`rf.nested.masked_select_nested` packs a data-dependent number of beams);
      the result is selected by the same mask afterwards, so the scores are the same, only the compute is more.
    - the label history is a fixed-capacity buffer (at most one label per frame) with an explicit length,
      instead of a growing history; recombination compares buffer and length.
      The slots beyond the length are never written, so equal prefixes give equal buffers.

    The final invalid-beam removal is also left out (data-dependent),
    so recombined-away hypotheses stay in the beam with score -inf.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    from returnn.config import get_global_config
    from i6_experiments.users.zeyer.nn_rf.soft_collapse_repeated import soft_collapse_repeated

    config = get_global_config()
    beam_size = config.int("beam_size", 12)
    recomb = config.typed_value("recog_recomb", None)  # None, "max", "sum"
    ctc_soft_collapse_threshold = config.typed_value("ctc_soft_collapse_threshold", None)
    ctc_soft_collapse_reduce_type = config.typed_value("ctc_soft_collapse_reduce_type", "logmeanexp")
    lm_softmax_temperature = config.typed_value("lm_softmax_temperature", 1.0)

    label_log_prob, _, enc_spatial_dim = model.encode_and_get_ctc_log_probs(data, in_spatial_dim=data_spatial_dim)
    batch_dims = label_log_prob.remaining_dims((enc_spatial_dim, label_log_prob.feature_dim))

    if ctc_soft_collapse_threshold is not None:
        label_log_prob, enc_spatial_dim = soft_collapse_repeated(
            label_log_prob,
            spatial_dim=enc_spatial_dim,
            classes_dim=model.wb_target_dim,
            threshold=ctc_soft_collapse_threshold,
            reduce_type=ctc_soft_collapse_reduce_type,
        )

    neg_inf = float("-inf")
    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=neg_inf),
    )
    label_log_prob_ta = TensorArray.unstack(label_log_prob, axis=enc_spatial_dim)  # t -> Batch, VocabWB

    beam_dim = Dim(beam_size, name="beam")
    step_beam_dim = Dim(beam_size, name="beam-step")  # top_k out dim, mapped back onto beam_dim
    batch_dims_ = [beam_dim] + batch_dims
    # at most one label per frame, so the frames bound both the LM KV cache and the label history
    max_seq_len_cpu = rf.copy_to_device(enc_spatial_dim.get_dim_value_tensor(), "cpu")
    # the accumulators run one entry per frame.
    # A graph loop that cannot grow its carry (JAX) needs that count up front.
    max_seq_len_int = int(enc_spatial_dim.get_dim_value())
    # The loop control lives where the loop runs.
    # TF drives it from the host, so the bound and the counter stay on CPU;
    # JAX runs the loop on the device, and a CPU-committed operand would split it across devices.
    _loop_on_device = rf.get_selected_backend() == "jax"
    loop_bound = max_seq_len_int if _loop_on_device else max_seq_len_cpu
    loop_counter_device = None if _loop_on_device else "cpu"
    backtrack_start = (
        rf.constant(max_seq_len_int - 1, dims=(), dtype="int32", device=None)
        if _loop_on_device
        else max_seq_len_cpu - 1
    )
    label_cap_dim = Dim(enc_spatial_dim.get_dim_value_tensor(), name="label-hist-capacity")

    if getattr(model, "lm", None) is None:
        lm: Optional[TransformerDecoder] = None
        lm_scale: Optional[float] = None
        labelwise_prior: Optional[rf.Parameter] = None
    else:
        # noinspection PyUnresolvedReferences
        lm: TransformerDecoder = model.lm
        # noinspection PyUnresolvedReferences
        lm_scale: float = model.lm_scale
        # noinspection PyUnresolvedReferences
        labelwise_prior: Optional[rf.Parameter] = model.labelwise_prior

    def _lm_log_probs(target: Tensor, lm_state: rf.State) -> Tuple[Tensor, rf.State]:
        lm_logits, lm_state = lm(target, spatial_dim=single_step_dim, state=lm_state)  # Batch, Beam, Vocab / ...
        if lm_softmax_temperature != 1.0:
            lm_logits = lm_logits / lm_softmax_temperature
        log_probs = rf.log_softmax(lm_logits, axis=model.target_dim) * lm_scale
        if labelwise_prior is not None:
            log_probs -= labelwise_prior  # prior scale already applied
        return log_probs, lm_state

    if lm is not None:
        lm_state0 = lm.default_initial_state(batch_dims=batch_dims_)
        # Only a new label re-runs the LM here, so the hypotheses diverge:
        # both the position and the KV history length are per hypothesis, not shared.
        lm_state0.pos = lm_state0.pos + rf.zeros(batch_dims_, dtype="int32")
        for _layer_key, _layer_state in lm_state0.items():
            if _layer_key == "pos":
                continue
            att_state = _layer_state.self_att
            hist_dim = Dim(rf.zeros(batch_dims_, dtype="int32"), name="self_att_hist")  # empty, as the initial one
            att_state.k_accum, _ = rf.replace_dim(att_state.k_accum, in_dim=att_state.accum_axis, out_dim=hist_dim)
            att_state.v_accum, _ = rf.replace_dim(att_state.v_accum, in_dim=att_state.accum_axis, out_dim=hist_dim)
            att_state.accum_axis = hist_dim
        lm_log_probs0, lm_state0 = _lm_log_probs(
            rf.constant(model.bos_idx, dims=batch_dims_, dtype="int32", sparse_dim=model.target_dim), lm_state0
        )
    else:
        lm_state0 = None
        lm_log_probs0 = None

    def _body(state):
        (
            t,
            seq_log_prob_,
            target_,
            target_wb_,
            label_hist_,
            label_hist_len_,
            lm_log_probs_,
            lm_state_,
            targets_wb_ta_,
            backrefs_ta_,
        ) = state
        prev_target = target_
        prev_target_wb = target_wb_

        # both sides broadcast, which is implicit only up to some behavior versions
        seq_log_prob_ = rf.combine(
            seq_log_prob_, "+", label_log_prob_ta[t], allow_broadcast_all_sources=True
        )  # Batch, InBeam, VocabWB
        if lm_state_ is not None:
            # add the LM score where the align label starts a new label, else 0
            seq_log_prob_ += rf.where(
                (prev_target_wb == model.blank_idx)
                | rf.compare(
                    prev_target_wb, "!=", rf.range_over_dim(model.wb_target_dim), allow_broadcast_all_sources=True
                ),
                _target_dense_extend_blank(
                    lm_log_probs_,
                    target_dim=model.target_dim,
                    wb_target_dim=model.wb_target_dim,
                    blank_idx=model.blank_idx,
                    value=0.0,
                ),
                0.0,
            )  # Batch, InBeam, VocabWB

        seq_log_prob_, (backrefs, target_wb_), _ = rf.top_k(
            seq_log_prob_, k_dim=step_beam_dim, axis=[beam_dim, model.wb_target_dim]
        )
        # replace_dim, not v2: same static size, and v2 is eager-only
        seq_log_prob_, _ = rf.replace_dim(seq_log_prob_, in_dim=step_beam_dim, out_dim=beam_dim)
        backrefs, _ = rf.replace_dim(backrefs, in_dim=step_beam_dim, out_dim=beam_dim)
        backrefs = rf.cast(backrefs, "int32")  # top_k index dtype is backend specific, loop var dtype is not
        backrefs.sparse_dim = beam_dim
        target_wb_, _ = rf.replace_dim(target_wb_, in_dim=step_beam_dim, out_dim=beam_dim)
        target_wb_ = rf.cast(target_wb_, "int32")
        target_wb_.sparse_dim = model.wb_target_dim

        if lm_state_ is not None:
            lm_log_probs_ = rf.gather(lm_log_probs_, indices=backrefs)  # Batch, Beam, Vocab
            lm_state_ = rf.nested.gather_nested(lm_state_, indices=backrefs)
        label_hist_ = rf.gather(label_hist_, indices=backrefs)
        label_hist_len_ = rf.gather(label_hist_len_, indices=backrefs)
        prev_target = rf.gather(prev_target, indices=backrefs)  # Batch, Beam -> Vocab
        prev_target_wb = rf.gather(prev_target_wb, indices=backrefs)  # Batch, Beam -> VocabWB

        got_new_label = (target_wb_ != model.blank_idx) & (target_wb_ != prev_target_wb)  # Batch, Beam -> 0|1
        target_ = rf.where(
            got_new_label,
            _target_remove_blank(
                target_wb_, target_dim=model.target_dim, wb_target_dim=model.wb_target_dim, blank_idx=model.blank_idx
            ),
            prev_target,
        )  # Batch, Beam -> Vocab
        label_hist_ = rf.where(
            got_new_label
            & rf.compare(rf.range_over_dim(label_cap_dim), "==", label_hist_len_, allow_broadcast_all_sources=True),
            target_,
            label_hist_,
        )
        label_hist_len_ = label_hist_len_ + rf.where(got_new_label, 1, 0)

        if recomb:
            # Recombine paths with the same label seq. The eager version does this only in steps where some
            # beam got a new label, so gate on the same (global) condition -- without it, paths that became
            # equal earlier would be recombined one step sooner and a hypothesis would drop out.
            any_new_label = rf.reduce_any(got_new_label, axis=got_new_label.dims)
            label_hist_dual, beam_dual_dim = rf.replace_dim(label_hist_, in_dim=beam_dim)
            label_hist_len_dual, _ = rf.replace_dim(label_hist_len_, in_dim=beam_dim, out_dim=beam_dual_dim)
            same_seq_labels = rf.logical_and(
                rf.reduce_all(rf.compare_bc(label_hist_, "==", label_hist_dual), axis=label_cap_dim),
                rf.compare_bc(label_hist_len_, "==", label_hist_len_dual),
            )  # Batch, Beam, BeamDual
            seq_log_prob_ext, _ = rf.replace_dim(seq_log_prob_, in_dim=beam_dim, out_dim=beam_dual_dim)
            seq_log_prob_ext = rf.where(same_seq_labels, seq_log_prob_ext, neg_inf)  # Batch, Beam, BeamDual
            if recomb == "sum":
                seq_log_prob_recomb = rf.reduce_logsumexp(seq_log_prob_ext, axis=beam_dual_dim)  # Batch, Beam
            elif recomb == "max":
                seq_log_prob_recomb = seq_log_prob_
            else:
                raise ValueError(f"invalid recog_recomb {recomb!r}")
            argmax_seq_log_prob = rf.reduce_argmax(seq_log_prob_ext, axis=beam_dual_dim)  # Batch, Beam -> BeamDual
            mask = argmax_seq_log_prob == rf.range_over_dim(beam_dim)  # Batch, Beam -> 0|1
            seq_log_prob_recomb = rf.where(mask, seq_log_prob_recomb, neg_inf)
            seq_log_prob_ = rf.where(any_new_label, seq_log_prob_recomb, seq_log_prob_)
            # don't re-eval the LM when masked out
            got_new_label = rf.where(any_new_label, got_new_label & mask, got_new_label)

        if lm_state_ is not None:
            # unlike the eager version, this runs for all beams, and the mask selects afterwards
            lm_log_probs_new, lm_state_new = _lm_log_probs(target_, lm_state_)
            lm_log_probs_, lm_state_ = rf.nested.mask_nested(
                (lm_log_probs_new, lm_state_new),
                mask=got_new_label,
                mask_value=(lm_log_probs_, lm_state_),
            )

        return (
            t + 1,
            seq_log_prob_,
            target_,
            target_wb_,
            label_hist_,
            label_hist_len_,
            lm_log_probs_,
            lm_state_,
            targets_wb_ta_.push_back(target_wb_),
            backrefs_ta_.push_back(backrefs),
        )

    target_wb_template = Tensor("target_wb", dims=batch_dims_, dtype="int32", sparse_dim=model.wb_target_dim)
    backrefs_template = Tensor("backrefs", dims=batch_dims_, dtype="int32", sparse_dim=beam_dim)
    _, seq_log_prob, _, _, _, _, lm_log_probs, _, seq_targets_wb_ta, seq_backrefs_ta = rf.while_loop(
        cond=lambda state: state[0] < loop_bound,
        body=_body,
        initial=(
            rf.constant(0, dims=(), dtype="int32", device=loop_counter_device),
            rf.where(rf.range_over_dim(beam_dim) == 0, rf.constant(0.0, dims=batch_dims_), neg_inf),
            rf.constant(model.bos_idx, dims=batch_dims_, dtype="int32", sparse_dim=model.target_dim),
            rf.constant(model.blank_idx, dims=batch_dims_, dtype="int32", sparse_dim=model.wb_target_dim),
            rf.zeros(batch_dims_ + [label_cap_dim], dtype="int32", sparse_dim=model.target_dim),
            rf.constant(0, dims=batch_dims_, dtype="int32"),
            lm_log_probs0,
            lm_state0,
            TensorArray(target_wb_template, capacity=max_seq_len_int),
            TensorArray(backrefs_template, capacity=max_seq_len_int),
        ),
    )
    if lm_log_probs is not None:
        # LM EOS score at the end, as in the eager version
        seq_log_prob += rf.gather(lm_log_probs, indices=model.eos_idx, axis=model.target_dim)

    # Backtrack via backrefs, resolve beams. Backwards, so the result is flipped below.
    def _backtrack_body(state):
        t, indices, out_ta_ = state
        out_ta_ = out_ta_.push_back(rf.gather(seq_targets_wb_ta[t], indices=indices))  # FinalBeam -> VocabWB
        indices = rf.gather(seq_backrefs_ta[t], indices=indices)  # FinalBeam -> PrevBeam
        return t - 1, indices, out_ta_

    # already with the batch dims: a loop var keeps its dims, and the gather in the body has them
    indices0 = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for dim in batch_dims:
        indices0 = rf.expand_dim(indices0, dim=dim)
    indices0.sparse_dim = beam_dim
    _, _, seq_targets_rev_ta = rf.while_loop(
        cond=lambda state: state[0] >= 0,
        body=_backtrack_body,
        initial=(backtrack_start, indices0, TensorArray(target_wb_template, capacity=max_seq_len_int)),
    )
    out_spatial_dim = enc_spatial_dim
    seq_targets_wb = seq_targets_rev_ta.stack(axis=out_spatial_dim)
    # Flip over the PADDED extent: entry i is frame max_seq_len-1-i for every sequence,
    # so clipping to the per-seq length would fold a short sequence onto its last frame.
    rev_indices = max_seq_len_cpu - 1 - rf.range_over_dim(out_spatial_dim, device="cpu")
    seq_targets_wb = rf.gather(
        seq_targets_wb, indices=rf.copy_to_device(rev_indices, seq_targets_wb.device), axis=out_spatial_dim
    )

    return seq_targets_wb, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog_with_recomb_while_loop: RecogDef[Model]
model_recog_with_recomb_while_loop.output_with_beam = True
model_recog_with_recomb_while_loop.output_blank_label = "<blank>"
model_recog_with_recomb_while_loop.batch_size_dependent = True  # as model_recog_with_recomb


def _target_remove_blank(target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int) -> Tensor:
    assert target.sparse_dim == wb_target_dim
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    return rf.set_sparse_dim(target, target_dim)


def _target_dense_extend_blank(
    target: Tensor, *, target_dim: Dim, wb_target_dim: Dim, blank_idx: int, value: float
) -> Tensor:
    assert target_dim in target.dims
    assert blank_idx == target_dim.dimension  # currently just not implemented otherwise
    res, _ = rf.pad(target, axes=[target_dim], padding=[(0, 1)], out_dims=[wb_target_dim], value=value)
    return res


def _seq_label_history_init_state(*, vocab_dim: Dim, batch_dims: Sequence[Dim]) -> rf.State:
    hist_dim = Dim(0, name="hist0")
    history = rf.zeros(list(batch_dims) + [hist_dim], dtype="int64", sparse_dim=vocab_dim)
    return rf.State(hist_dim=hist_dim, history=history)


def _seq_label_append(state: rf.State, new_label: Tensor) -> rf.State:
    hist_dim: Dim = state.hist_dim
    if new_label.dtype != state.history.dtype:
        new_label = rf.cast(new_label, state.history.dtype)
    new_history, new_hist_dim = rf.cum_concat_step(new_label, prev_accum=state.history, axis=hist_dim)
    return rf.State(hist_dim=new_hist_dim, history=new_history)


def _same_seq_labels(seq: Tensor, *, spatial_dim: Dim, beam_dim: Dim) -> Tuple[Tensor, Dim]:
    seq_label_dual, beam_dual_dim = rf.replace_dim(seq, in_dim=beam_dim)
    same_seq_labels = rf.compare_bc(seq, "==", seq_label_dual)  # Batch, Beam, BeamDual, Spatial
    same_seq_labels = rf.reduce_all(same_seq_labels, axis=spatial_dim)  # Batch, Beam, BeamDual
    if beam_dim in spatial_dim.get_size_tensor().dims:
        seq_labels_lens = spatial_dim.get_size_tensor(device=same_seq_labels.device)
        seq_labels_dual_lens = rf.replace_dim_v2(
            seq_labels_lens, in_dim=beam_dim, out_dim=beam_dual_dim
        )  # Batch, BeamDual
        same_seq_labels_lens = rf.compare_bc(seq_labels_lens, "==", seq_labels_dual_lens)  # Batch, Beam, BeamDual
        same_seq_labels = rf.logical_and(same_seq_labels, same_seq_labels_lens)
    return same_seq_labels, beam_dual_dim
