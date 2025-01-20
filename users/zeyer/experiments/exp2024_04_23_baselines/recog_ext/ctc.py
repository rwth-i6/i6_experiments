"""
CTC decoding with neural LM
"""

from typing import TypeVar, Sequence, Tuple, Dict
import functools

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

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    import tree
    import returnn
    from returnn.config import get_global_config

    config = get_global_config()
    beam_size = config.int("beam_size", 12)
    version = config.int("recog_version", 1)
    assert version == 7

    # RETURNN version is like "1.20250115.110555"
    # There was an important fix in 2025-01-17 affecting masked_scatter.
    assert tuple(int(n) for n in returnn.__version__.split(".")) >= (1, 20250119, 0), returnn.__version__

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

    # We usually have TransformerDecoder, but any other type would also be ok when it has the same API.
    # noinspection PyUnresolvedReferences
    lm: TransformerDecoder = model.lm
    # noinspection PyUnresolvedReferences
    lm_scale: float = model.lm_scale

    print(f"* beam size {beam_size}, lm scale {lm_scale}, prior scale {model.ctc_prior_scale}")

    lm_state = lm.default_initial_state(batch_dims=batch_dims_)  # Batch, InBeam, ...
    lm_logits, lm_state = lm(
        target,
        spatial_dim=single_step_dim,
        state=lm_state,
    )  # Batch, InBeam, Vocab / ...
    lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Batch, InBeam, Vocab
    lm_log_probs *= lm_scale

    lm_log_probs = lm_log_probs.copy_compatible_to_dims(batch_dims_ + [model.target_dim])
    print("* lm_log_probs initial:", lm_log_probs)
    print(
        f"* argmax LM begin: {model.target_dim.vocab.id_to_label(lm_log_probs.raw_tensor[0, 0].argmax().cpu().item())}"
    )

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets_wb = []
    seq_backrefs = []
    for t in range(max_seq_len):
        prev_target = target
        prev_target_wb = target_wb

        # print(f"* prev_target_wb {model.wb_target_dim.vocab.id_to_label(prev_target_wb.raw_tensor.cpu()[0, 0].item())}")

        seq_log_prob = seq_log_prob + label_log_prob_ta[t]  # Batch, InBeam, VocabWB

        seq_log_prob = seq_log_prob.copy_compatible_to_dims(batch_dims + [beam_dim, model.wb_target_dim])
        print(
            f"* argmax seq_log_prob (before LM) t={t}:"
            f" {model.wb_target_dim.vocab.id_to_label(seq_log_prob.raw_tensor[0, 0].argmax().cpu().item())}"
        )

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
        )  # Batch, InBeam -> VocabWB

        seq_log_prob = seq_log_prob.copy_compatible_to_dims(batch_dims + [beam_dim, model.wb_target_dim])
        print(
            f"* argmax seq_log_prob (with LM) t={t}:"
            f" {model.wb_target_dim.vocab.id_to_label(seq_log_prob.raw_tensor[0, 0].argmax().cpu().item())}"
        )

        seq_log_prob, (backrefs, target_wb), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, model.wb_target_dim]
        )
        # seq_log_prob, backrefs, target_wb: Batch, Beam
        # backrefs -> InBeam.
        # target_wb -> VocabWB.
        seq_targets_wb.append(target_wb)
        seq_backrefs.append(backrefs)

        target_wb = target_wb.copy_compatible_to_dims(batch_dims + [beam_dim])  # Batch, Beam -> VocabWB
        print(
            f"* target_wb t={t} beam:"
            f" {[model.wb_target_dim.vocab.id_to_label(l.item()) for l in target_wb.raw_tensor[0, :3].cpu()]}"
        )

        lm_log_probs = rf.gather(lm_log_probs, indices=backrefs)  # Batch, Beam, Vocab
        lm_state = tree.map_structure(functools.partial(_gather_backrefs, backrefs=backrefs), lm_state)
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

        target = target.copy_compatible_to_dims(batch_dims + [beam_dim])  # Batch, Beam -> Vocab
        print(
            f"* target t={t} beam:"
            f" {[model.target_dim.vocab.id_to_label(l.item()) for l in target.raw_tensor[0, :3].cpu()]}"
        )

        (target_, lm_state_), packed_new_label_dim, packed_new_label_dim_map = _masked_select_tree(
            (target, lm_state),
            mask=got_new_label,
            dims=batch_dims + [beam_dim],
        )

        if packed_new_label_dim.get_dim_value() > 0:
            print(
                f"* feed target"
                f" {[model.target_dim.vocab.id_to_label(l.item()) for l in target_.raw_tensor[:3].cpu()]}"
            )

            lm_logits_, lm_state_ = lm(
                target_,
                spatial_dim=single_step_dim,
                state=lm_state_,
            )  # Flat_Batch_Beam, Vocab / ...
            lm_log_probs_ = rf.log_softmax(lm_logits_, axis=model.target_dim)  # Flat_Batch_Beam, Vocab
            lm_log_probs_ *= lm_scale

            lm_log_probs_ = lm_log_probs_.copy_compatible_to_dims([packed_new_label_dim, model.target_dim])
            print(
                f"* argmax LM after feed:"
                f" {model.target_dim.vocab.id_to_label(lm_log_probs_.raw_tensor[0].argmax().cpu().item())}"
            )

            lm_log_probs, lm_state = _masked_scatter_tree(
                (lm_log_probs_, lm_state_),
                (lm_log_probs, lm_state),
                mask=got_new_label,
                dims=batch_dims + [beam_dim],
                in_dim=packed_new_label_dim,
                dim_map=packed_new_label_dim_map,
            )  # Batch, Beam, Vocab / ...

    # seq_log_prob, lm_log_probs: Batch, Beam
    # Add LM EOS score at the end.
    seq_log_prob += rf.gather(lm_log_probs, indices=model.eos_idx, axis=model.target_dim)  # Batch, Beam -> VocabWB

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


T = TypeVar("T")


def _gather_backrefs(s: T, *, backrefs: Tensor) -> T:
    if isinstance(s, Tensor):
        if backrefs.sparse_dim in s.dims:
            return rf.gather(s, indices=backrefs)  # really the default case
        return s  # e.g. scalar or so, independent from beam
    if isinstance(s, Dim):
        if s.dimension is not None:  # static
            return s
        assert backrefs.sparse_dim not in s.dyn_size_ext.dims  # currently not supported, also not expected
        return s
    raise TypeError(f"_gather_backrefs: unexpected type ({type(s)})")


def _masked_select_tree(s: T, *, mask: Tensor, dims: Sequence[Dim]) -> Tuple[T, Dim, Dict[Dim, Dim]]:
    import tree

    packed_new_label_dim = Dim(None, name="packed_new_label")  # Flat_Batch_InBeam
    packed_new_label_dim_map = {}
    tree.map_structure(
        functools.partial(
            _masked_select_prepare_dims,
            mask=mask,
            dims=dims,
            out_dim=packed_new_label_dim,
            dim_map=packed_new_label_dim_map,
        ),
        s,
    )
    s = tree.map_structure(
        functools.partial(
            _masked_select,
            mask=mask,
            dims=dims,
            out_dim=packed_new_label_dim,
            dim_map=packed_new_label_dim_map,
        ),
        s,
    )
    return s, packed_new_label_dim, packed_new_label_dim_map


def _masked_select_prepare_dims(s, *, mask: Tensor, dims: Sequence[Dim], out_dim: Dim, dim_map: Dict[Dim, Dim]):
    if isinstance(s, Tensor):
        return s  # ignored at this stage
    if isinstance(s, Dim):
        if s.dimension is not None:  # static
            return s
        if not any(d in s.dyn_size_ext.dims for d in dims):
            return s
        if s in dim_map:
            return dim_map[s]
        new_dyn_size = _masked_select(s.dyn_size_ext, mask=mask, dims=dims, out_dim=out_dim, dim_map=dim_map)
        new_dim = Dim(new_dyn_size, name=s.name + "_")
        dim_map[s] = new_dim
        return new_dim
    raise TypeError(f"_masked_select_prepare_dims: unexpected type ({type(s)})")


def _masked_select(s: T, *, mask: Tensor, dims: Sequence[Dim], out_dim: Dim, dim_map: Dict[Dim, Dim]) -> T:
    if isinstance(s, Tensor):
        if not any(d in s.dims for d in dims):
            return s  # e.g. scalar or so, independent from dims
        # For the masked_select, we need that all masked dims are present, so add them if not.
        # (E.g., when we mask [batch,beam], but we only have [batch], we need to add the beam dim.)
        if any(d not in s.dims for d in dims):
            s = rf.expand_dims(s, dims=[d for d in dims if d not in s.dims])
        # The packing itself (masked_select).
        s, _ = rf.masked_select(s, mask=mask, dims=dims, out_dim=out_dim)
        # In the resulting tensor, potentially replace dims.
        # In addition to the dim replacement, we also might need to slice, as the size might be smaller.
        if any(d in dim_map for d in s.dims):
            for d in s.dims:
                if d in dim_map:
                    s, _ = rf.slice(s, axis=d, size=dim_map[d])
        return s
    if isinstance(s, Dim):
        if s.dimension is not None:  # static
            return s
        if not any(d in s.dyn_size_ext.dims for d in dims):
            return s
        assert s in dim_map
        return dim_map[s]
    raise TypeError(f"_gather_backrefs: unexpected type ({type(s)})")


def _masked_scatter_tree(
    s: T, backup: T, *, mask: Tensor, dims: Sequence[Dim], in_dim: Dim, dim_map: Dict[Dim, Dim]
) -> T:
    import tree

    reverse_dim_map = {v: k for k, v in dim_map.items()}
    merged_dim_map = {}

    tree.map_structure(
        functools.partial(
            _masked_scatter_merge_dims,
            mask=mask,
            dims=dims,
            in_dim=in_dim,
            reverse_dim_map=reverse_dim_map,
            merged_dim_map=merged_dim_map,
        ),
        s,
        backup,
    )
    s = tree.map_structure(
        functools.partial(
            _masked_scatter,
            mask=mask,
            dims=dims,
            in_dim=in_dim,
            reverse_dim_map=reverse_dim_map,
            merged_dim_map=merged_dim_map,
        ),
        s,
        backup,
    )
    return s


def _masked_scatter_merge_dims(
    s: T,
    backup: T,
    *,
    mask: Tensor,
    dims: Sequence[Dim],
    in_dim: Dim,
    reverse_dim_map: Dict[Dim, Dim],
    merged_dim_map: Dict[Dim, Dim],
) -> T:
    if isinstance(s, Tensor):
        return s
    if isinstance(s, Dim):
        # This is slightly more complex than in the _masked_select case:
        # We need to merge the s and backup depending on the mask.
        if s in reverse_dim_map:
            s = reverse_dim_map[s]
        if s == backup:
            return s
        if s in merged_dim_map:
            return merged_dim_map[s]
        # Note: s/backup might even be static dims.
        new_size = _masked_scatter(
            s.get_size_tensor(),
            backup.get_size_tensor(),
            mask=mask,
            dims=dims,
            in_dim=in_dim,
            reverse_dim_map=reverse_dim_map,
            merged_dim_map=merged_dim_map,
        )
        new_dim = Dim(new_size, name=s.name + "_")
        merged_dim_map[s] = new_dim
        merged_dim_map[backup] = new_dim
        return new_dim
    raise TypeError(f"_masked_scatter_merge_dims: unexpected type ({type(s)})")


def _masked_scatter(
    s: T,
    backup: T,
    *,
    mask: Tensor,
    dims: Sequence[Dim],
    in_dim: Dim,
    reverse_dim_map: Dict[Dim, Dim],
    merged_dim_map: Dict[Dim, Dim],
) -> T:
    if isinstance(s, Tensor):
        if in_dim not in s.dims:
            return s  # e.g. scalar or so, independent from masking
        assert isinstance(backup, Tensor)
        # Do the reverse of _masked_select above.
        # First replace the dims back.
        if any(d in reverse_dim_map for d in s.dims):
            for d in s.dims:
                if d in reverse_dim_map:
                    s = _expand_slice(s, axis=d, expanded_size=reverse_dim_map[d])
        # We also might need to replace newly merged dims, both in s and backup.
        for d in s.dims:
            if d in merged_dim_map:
                s, _ = rf.slice(s, axis=d, size=merged_dim_map[d])
        # There is currently the implicit assumption that the backup might need extra padding,
        # while the s needs slicing...
        # (We think of the hist_dim, where s should only have more frames than backup, or the same.)
        for d in backup.dims:
            if d in merged_dim_map:
                backup = _expand_slice(backup, axis=d, expanded_size=merged_dim_map[d])
        # The unpacking itself (reversing the masked_select, i.e. masked_scatter).
        s = rf.masked_scatter(s, backup, mask=mask, dims=dims, in_dim=in_dim)
        # Now remove potential added dims.
        for d in s.dims:
            if d not in backup.dims:
                s = rf.gather(s, axis=d, indices=0)
        return s
    if isinstance(s, Dim):
        # This is slightly more complex than in the _masked_select case:
        # We need to merge the s and backup depending on the mask.
        if s in reverse_dim_map:
            s = reverse_dim_map[s]
        if s in merged_dim_map:
            return merged_dim_map[s]
        return s
    raise TypeError(f"_masked_scatter: unexpected type ({type(s)})")


def _expand_slice(source: Tensor, axis: Dim, expanded_size: Dim) -> Tensor:
    res, _ = rf.pad(
        source,
        axes=[axis],
        padding=[(0, expanded_size.get_dim_value_tensor() - axis.get_dim_value_tensor())],
        out_dims=[expanded_size],
    )
    return res


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
