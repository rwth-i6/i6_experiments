"""
CTC decoding with neural LM
"""

from __future__ import annotations
from typing import TypeVar, Optional, Sequence, Tuple, Dict, Generator
import re
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
    import returnn
    from returnn.config import get_global_config

    config = get_global_config()
    beam_size = config.int("beam_size", 12)
    version = config.int("recog_version", 1)
    assert version == 8

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

    # print(f"* beam size {beam_size}, lm scale {lm_scale}, prior scale {model.ctc_prior_scale}")

    lm_state = lm.default_initial_state(batch_dims=batch_dims_)  # Batch, InBeam, ...
    lm_logits, lm_state = lm(
        target,
        spatial_dim=single_step_dim,
        state=lm_state,
    )  # Batch, InBeam, Vocab / ...
    lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Batch, InBeam, Vocab
    lm_log_probs *= lm_scale

    lm_log_probs = lm_log_probs.copy_compatible_to_dims(batch_dims_ + [model.target_dim])
    # print("* lm_log_probs initial:", lm_log_probs)
    # print(
    #     f"* argmax LM begin: {model.target_dim.vocab.id_to_label(lm_log_probs.raw_tensor[0, 0].argmax().cpu().item())}"
    # )

    # For debugging, accumulate (non-blank) label history.
    # Note: When you use this, uncomment the seq_label usages below,
    # and also add this to the masked_select_tree and masked_scatter_tree.
    # seq_label = _seq_label_history_init_state(vocab_dim=model.target_dim, batch_dims=batch_dims_)

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets_wb = []
    seq_backrefs = []
    for t in range(max_seq_len):
        prev_target = target
        prev_target_wb = target_wb

        # print(f"* prev_target_wb {model.wb_target_dim.vocab.id_to_label(prev_target_wb.raw_tensor.cpu()[0, 0].item())}")

        seq_log_prob = seq_log_prob + label_log_prob_ta[t]  # Batch, InBeam, VocabWB

        # seq_log_prob = seq_log_prob.copy_compatible_to_dims(batch_dims + [beam_dim, model.wb_target_dim])
        # print(
        #     f"* argmax seq_log_prob (before LM) t={t}:"
        #     f" {model.wb_target_dim.vocab.id_to_label(seq_log_prob.raw_tensor[0, 0].argmax().cpu().item())}"
        # )

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

        # seq_log_prob = seq_log_prob.copy_compatible_to_dims(batch_dims + [beam_dim, model.wb_target_dim])
        # print(
        #     f"* argmax seq_log_prob (with LM) t={t}:"
        #     f" {model.wb_target_dim.vocab.id_to_label(seq_log_prob.raw_tensor[0, 0].argmax().cpu().item())}"
        # )

        seq_log_prob, (backrefs, target_wb), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, model.wb_target_dim]
        )
        # seq_log_prob, backrefs, target_wb: Batch, Beam
        # backrefs -> InBeam.
        # target_wb -> VocabWB.
        seq_targets_wb.append(target_wb)
        seq_backrefs.append(backrefs)

        # target_wb = target_wb.copy_compatible_to_dims(batch_dims + [beam_dim])  # Batch, Beam -> VocabWB
        # print(
        #     f"* target_wb t={t} beam:"
        #     f" {[model.wb_target_dim.vocab.id_to_label(l.item()) for l in target_wb.raw_tensor[0, :3].cpu()]}"
        # )

        backrefs_dim_map = {}  # old dim -> new dim
        lm_log_probs = rf.gather(lm_log_probs, indices=backrefs)  # Batch, Beam, Vocab
        lm_state = _gather_backrefs_tree(lm_state, backrefs=backrefs, dim_map=backrefs_dim_map)
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

        # target = target.copy_compatible_to_dims(batch_dims + [beam_dim])  # Batch, Beam -> Vocab
        # print(
        #     f"* target t={t} beam:"
        #     f" {[model.target_dim.vocab.id_to_label(l.item()) for l in target.raw_tensor[0, :3].cpu()]}"
        # )

        # seq_label = _gather_backrefs_tree(seq_label, backrefs=backrefs, dim_map=backrefs_dim_map)
        # _seq_label_print("gather backrefs", seq_label)

        got_new_label_cpu = rf.copy_to_device(got_new_label, "cpu")
        if got_new_label_cpu.raw_tensor.sum().item() > 0:
            (target_, lm_state_), packed_new_label_dim, packed_new_label_dim_map = _masked_select_tree(
                (target, lm_state),
                mask=got_new_label,
                mask_cpu=got_new_label_cpu,
                dims=batch_dims + [beam_dim],
            )
            # packed_new_label_dim_map: old dim -> new dim. see _masked_select_prepare_dims
            assert packed_new_label_dim.get_dim_value() > 0

            # print(
            #     f"* feed target"
            #     f" {[model.target_dim.vocab.id_to_label(l.item()) for l in target_.raw_tensor[:3].cpu()]}"
            # )

            lm_logits_, lm_state_ = lm(
                target_,
                spatial_dim=single_step_dim,
                state=lm_state_,
            )  # Flat_Batch_Beam, Vocab / ...
            lm_log_probs_ = rf.log_softmax(lm_logits_, axis=model.target_dim)  # Flat_Batch_Beam, Vocab
            lm_log_probs_ *= lm_scale

            # lm_log_probs_ = lm_log_probs_.copy_compatible_to_dims([packed_new_label_dim, model.target_dim])
            # print(
            #     f"* argmax LM after feed:"
            #     f" {model.target_dim.vocab.id_to_label(lm_log_probs_.raw_tensor[0].argmax().cpu().item())}"
            # )

            # seq_label_ = _seq_label_append(seq_label_, target_)
            # _seq_label_print("packed append", seq_label_)

            lm_log_probs, lm_state = _masked_scatter_tree(
                (lm_log_probs_, lm_state_),
                (lm_log_probs, lm_state),
                mask=got_new_label,
                mask_cpu=got_new_label_cpu,
                dims=batch_dims + [beam_dim],
                in_dim=packed_new_label_dim,
                dim_map=packed_new_label_dim_map,
            )  # Batch, Beam, Vocab / ...

            # _seq_label_print("masked scatter", seq_label)

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


def _gather_backrefs_tree(s: T, *, backrefs: Tensor, dim_map: Dict[Dim, Dim]) -> T:
    import tree

    tree.map_structure(functools.partial(_gather_backrefs_prepare_dims, backrefs=backrefs, dim_map=dim_map), s)
    s = tree.map_structure(functools.partial(_gather_backrefs, backrefs=backrefs, dim_map=dim_map), s)
    return s


def _gather_backrefs_prepare_dims(s: T, *, backrefs: Tensor, dim_map: Dict[Dim, Dim]) -> T:
    if isinstance(s, Tensor):
        return s  # ignored at this stage
    if isinstance(s, Dim):
        if s.dimension is not None:  # static
            return s
        if s in dim_map:
            return dim_map[s]
        if backrefs.sparse_dim in s.dyn_size_ext.dims:
            new_dyn_size = _gather_backrefs(s.dyn_size_ext, backrefs=backrefs, dim_map=dim_map)
            new_dim = Dim(new_dyn_size, name=_extend_dim_name(s.name))
            dim_map[s] = new_dim
            return new_dim
        return s
    raise TypeError(f"_gather_backrefs_prepare_dims: unexpected type ({type(s)})")


def _gather_backrefs(s: T, *, backrefs: Tensor, dim_map: Optional[Dict[Dim, Dim]] = None) -> T:
    if isinstance(s, Tensor):
        if dim_map and any(d in dim_map for d in s.dims):
            for d in s.dims:
                if d in dim_map:
                    s = _expand_slice(s, old_dim=d, new_dim=dim_map[d])
        if backrefs.sparse_dim in s.dims:
            # really the default case, otherwise e.g. scalar or so, independent from beam
            s = rf.gather(s, indices=backrefs)
        return s
    if isinstance(s, Dim):
        if s.dimension is not None:  # static
            return s
        if dim_map and s in dim_map:
            return dim_map[s]
        assert backrefs.sparse_dim not in s.dyn_size_ext.dims  # not expected, should be in dim_map
        return s
    raise TypeError(f"_gather_backrefs: unexpected type ({type(s)})")


def _masked_select_tree(s: T, *, mask: Tensor, mask_cpu: Tensor, dims: Sequence[Dim]) -> Tuple[T, Dim, Dict[Dim, Dim]]:
    import tree

    packed_new_label_dim = Dim(None, name="packed_new_label")  # Flat_Batch_InBeam
    packed_new_label_dim_map = {}
    tree.map_structure(
        functools.partial(
            _masked_select_prepare_dims,
            mask=mask_cpu,
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
            mask_cpu=mask_cpu,
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
        new_dim = Dim(new_dyn_size, name=_extend_dim_name(s.name))
        dim_map[s] = new_dim
        return new_dim
    raise TypeError(f"_masked_select_prepare_dims: unexpected type ({type(s)})")


def _masked_select(
    s: T, *, mask: Tensor, mask_cpu: Optional[Tensor] = None, dims: Sequence[Dim], out_dim: Dim, dim_map: Dict[Dim, Dim]
) -> T:
    if isinstance(s, Tensor):
        if not any(d in s.dims for d in dims):
            return s  # e.g. scalar or so, independent from dims
        if s.device == "cpu" and mask_cpu is not None:
            mask = mask_cpu
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
    s: T, backup: T, *, mask: Tensor, mask_cpu: Tensor, dims: Sequence[Dim], in_dim: Dim, dim_map: Dict[Dim, Dim]
) -> T:
    import tree

    reverse_dim_map = {v: k for k, v in dim_map.items()}
    merged_dim_map = {}

    tree.map_structure(
        functools.partial(
            _masked_scatter_merge_dims,
            mask=mask_cpu,
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
            mask_cpu=mask_cpu,
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
        assert new_size.dims_set == (
            (s.get_size_tensor().dims_set | backup.get_size_tensor().dims_set) - {in_dim}
        ) | set(dims)
        new_dim = Dim(new_size, name=backup.name)
        merged_dim_map[s] = new_dim
        merged_dim_map[backup] = new_dim
        return new_dim
    raise TypeError(f"_masked_scatter_merge_dims: unexpected type ({type(s)})")


def _masked_scatter(
    s: T,
    backup: T,
    *,
    mask: Tensor,
    mask_cpu: Optional[Tensor] = None,
    dims: Sequence[Dim],
    in_dim: Dim,
    reverse_dim_map: Dict[Dim, Dim],
    merged_dim_map: Dict[Dim, Dim],
) -> T:
    if isinstance(s, Tensor):
        assert isinstance(backup, Tensor)
        if s.device == "cpu" and mask_cpu is not None:
            mask = mask_cpu
        if in_dim not in s.dims:
            s = rf.expand_dim(s, in_dim)
        # Do the reverse of _masked_select above.
        # First replace the dims back.
        if any(d in reverse_dim_map for d in s.dims):
            for d in s.dims:
                if d in reverse_dim_map:
                    s = _expand_slice(s, old_dim=d, new_dim=reverse_dim_map[d], expect_expand=True)
        # We also might need to replace newly merged dims, both in s and backup.
        for d in s.dims:
            if d in merged_dim_map:
                s = _expand_slice(s, old_dim=d, new_dim=merged_dim_map[d])
        for d in backup.dims:
            if d in merged_dim_map:
                backup = _expand_slice(backup, old_dim=d, new_dim=merged_dim_map[d])
        # The unpacking itself (reversing the masked_select, i.e. masked_scatter).
        s = rf.masked_scatter(s, backup, mask=mask, dims=dims, in_dim=in_dim)
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


def _expand_slice(source: Tensor, old_dim: Dim, new_dim: Dim, *, expect_expand: Optional[bool] = None) -> Tensor:
    assert old_dim in source.dims
    old_size = old_dim.get_dim_value()
    new_size = new_dim.get_dim_value()
    if old_size == new_size:
        res, _ = rf.replace_dim(source, in_dim=old_dim, out_dim=new_dim)
    elif old_size < new_size:
        res, _ = rf.pad(
            source,
            axes=[old_dim],
            padding=[(0, new_dim.get_dim_value_tensor() - old_dim.get_dim_value_tensor())],
            out_dims=[new_dim],
            value=0,
        )
    else:
        if expect_expand is True:
            raise ValueError(
                f"expected expand, but got reduce (slice): {old_size} -> {new_size},"
                f" for {old_dim=} {new_dim=}, in {source=}"
            )
        res, _ = rf.slice(source, axis=old_dim, size=new_dim)
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


def _extend_dim_name(name: str) -> str:
    # check ends with _<num>
    m = re.match(r"^(.*)_(\d+)$", name)
    if m:
        return f"{m.group(1)}_{int(m.group(2)) + 1}"
    return name + "_1"


# for debugging:


def _seq_label_history_init_state(*, vocab_dim: Dim, batch_dims: Sequence[Dim]) -> rf.State:
    hist_dim = Dim(0, name="hist0")
    history = rf.zeros(list(batch_dims) + [hist_dim], dtype="int64", sparse_dim=vocab_dim)
    return rf.State(hist_dim=hist_dim, history=history)


def _seq_label_append(state: rf.State, new_label: Tensor) -> rf.State:
    hist_dim: Dim = state.hist_dim
    new_history, new_hist_dim = rf.cum_concat_step(new_label, prev_accum=state.history, axis=hist_dim)
    return rf.State(hist_dim=new_hist_dim, history=new_history)


def _seq_label_print(prefix: str, state: rf.State):
    hist_dim: Dim = state.hist_dim
    hist: Tensor = state.history
    hist = rf.copy_to_device(hist, "cpu")
    batch_dims = hist.remaining_dims(hist_dim)
    print(f"* seq_label history {prefix}: {hist}:")
    for indices in _iter_dims_indices(batch_dims):
        print(" ", end="")
        hist_seq_len_ = hist_dim.get_size_tensor()
        hist_ = hist
        for dim, i in zip(batch_dims, indices):
            hist_ = rf.gather(hist_, axis=dim, indices=i)
            if dim in hist_seq_len_.dims:
                hist_seq_len_ = rf.gather(hist_seq_len_, axis=dim, indices=i)
            print(f" {dim}={i}", end="")
        hist_, _ = rf.slice(hist_, axis=hist_dim, size=hist_seq_len_)
        print(
            f": len={hist_seq_len_.raw_tensor}"
            f" {[hist.sparse_dim.vocab.id_to_label(l.item()) for l in hist_.raw_tensor]}"
        )


def _iter_dims_indices(dims: Sequence[Dim]) -> Generator[Tuple[int, ...]]:
    if not dims:
        yield ()
        return
    dim, rest = dims[0], dims[1:]
    for i in range(dim.get_dim_value()):
        for rest_indices in _iter_dims_indices(rest):
            yield (i,) + rest_indices
