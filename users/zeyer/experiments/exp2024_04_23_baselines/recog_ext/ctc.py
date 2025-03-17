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
    Function is run within RETURNN.

    Note, for debugging, see :func:`model_recog_debug`.

    Note, some potential further improvements:
    There are many align label seqs which correspond to the same label seq,
    but the LM score is calculated for each of them.
    We could make this somehow unique depending on the label seq.
    (But unclear how exactly to do this in a GPU friendly, batched way.)

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

    # RETURNN version is like "1.20250115.110555"
    # There was an important fix in 2025-01-17 affecting masked_scatter.
    # And another important fix in 2025-01-24 affecting masked_scatter for old PyTorch versions.
    assert tuple(int(n) for n in returnn.__version__.split(".")) >= (1, 20250125, 0), returnn.__version__

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    # The label log probs include the AM and the (scaled) prior.
    label_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, VocabWB
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
        lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Batch, InBeam, Vocab
        lm_log_probs *= lm_scale
        if labelwise_prior is not None:
            lm_log_probs -= labelwise_prior  # prior scale already applied

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets_wb = []
    seq_backrefs = []
    for t in range(max_seq_len):
        prev_target = target
        prev_target_wb = target_wb

        seq_log_prob = seq_log_prob + label_log_prob_ta[t]  # Batch, InBeam, VocabWB

        if lm is not None:
            # Now add LM score. If prev align label (target_wb) is blank or != cur, add LM score, otherwise 0.
            seq_log_prob += rf.where(
                (prev_target_wb == model.blank_idx) | (prev_target_wb != rf.range_over_dim(model.wb_target_dim)),
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
                lm_log_probs_ = rf.log_softmax(lm_logits_, axis=model.target_dim)  # Flat_Batch_Beam, Vocab
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
    Function is run within RETURNN.

    Note, some potential further improvements:
    There are many align label seqs which correspond to the same label seq,
    but the LM score is calculated for each of them.
    We could make this somehow unique depending on the label seq.
    (But unclear how exactly to do this in a GPU friendly, batched way.)

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
    assert version == 10
    recomb = config.typed_value("recog_recomb", None)  # None, "max", "sum"

    # RETURNN version is like "1.20250115.110555"
    # There was an important fix in 2025-01-17 affecting masked_scatter.
    # And another important fix in 2025-01-24 affecting masked_scatter for old PyTorch versions.
    assert tuple(int(n) for n in returnn.__version__.split(".")) >= (1, 20250125, 0), returnn.__version__

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    neg_inf = float("-inf")
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    # The label log probs include the AM and the (scaled) prior.
    label_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, VocabWB
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
        lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Batch, InBeam, Vocab
        lm_log_probs *= lm_scale
        if labelwise_prior is not None:
            lm_log_probs -= labelwise_prior  # prior scale already applied

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets_wb = []
    seq_backrefs = []
    for t in range(max_seq_len):
        prev_target = target
        prev_target_wb = target_wb

        seq_log_prob = seq_log_prob + label_log_prob_ta[t]  # Batch, InBeam, VocabWB

        if lm is not None:
            # Now add LM score. If prev align label (target_wb) is blank or != cur, add LM score, otherwise 0.
            seq_log_prob += rf.where(
                (prev_target_wb == model.blank_idx) | (prev_target_wb != rf.range_over_dim(model.wb_target_dim)),
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
        if got_new_label_cpu.raw_tensor.sum().item() > 0:
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
                lm_log_probs_ = rf.log_softmax(lm_logits_, axis=model.target_dim)  # Flat_Batch_Beam, Vocab
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
