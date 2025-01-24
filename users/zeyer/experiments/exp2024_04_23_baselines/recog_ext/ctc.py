"""
CTC decoding with neural LM
"""

from __future__ import annotations
from typing import TypeVar, Optional, Any, Sequence, Tuple, Dict, Generator
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
    # lm_scores = rf.constant(0.0, dims=batch_dims_)  # Batch, InBeam

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

        # lm_scores = rf.gather(lm_scores, indices=backrefs)  # Batch, Beam
        # seq_label = _gather_backrefs_tree(seq_label, backrefs=backrefs, dim_map=backrefs_dim_map)
        # _seq_label_print(f"{t=} gather backrefs", seq_label)

        # prev = (lm_log_probs, lm_state, lm_scores, seq_label)
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

            # lm_log_probs_ = rf.gather(lm_log_probs_, axis=model.target_dim, indices=target_)  # Flat_Batch_Beam
            # assert lm_scores_.dims == lm_log_probs_.dims == (packed_new_label_dim,)
            # lm_scores_ += lm_log_probs_  # Flat_Batch_Beam
            # print(
            #     f"* {t=} new label"
            #     f" {[model.target_dim.vocab.id_to_label(l.item()) for l in target_.raw_tensor[:3].cpu()]}"
            #     f" new log probs {[l.item() for l in lm_log_probs_.raw_tensor[:3].cpu()]}"
            #     f" new seq scores {[l.item() for l in lm_scores_.raw_tensor[:3].cpu()]}"
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
            # _seq_label_print(f"{t=} packed append", seq_label_)

            lm_log_probs, lm_state = rf.nested.masked_scatter_nested(
                (lm_log_probs_, lm_state_),
                (lm_log_probs, lm_state),
                mask=got_new_label,
                mask_cpu=got_new_label_cpu,
                dims=batch_dims + [beam_dim],
                in_dim=packed_new_label_dim,
                masked_select_dim_map=packed_new_label_dim_map,
            )  # Batch, Beam, Vocab / ...

            # _seq_label_print("masked scatter", seq_label)

        # new_state = _state_update(prev, target=target, beam_dim=beam_dim, model=model, lm=lm, lm_scale=lm_scale)
        # _where_deep_check(
        #     new_state,
        #     prev,
        #     (lm_log_probs, lm_state, lm_scores, seq_label),
        #     mask=got_new_label,
        #     mask_cpu=got_new_label_cpu,
        # )

    # seq_log_prob, lm_log_probs: Batch, Beam
    # Add LM EOS score at the end.
    lm_eos_score = rf.gather(lm_log_probs, indices=model.eos_idx, axis=model.target_dim)
    seq_log_prob += lm_eos_score  # Batch, Beam -> VocabWB
    # lm_scores += lm_eos_score  # Batch, Beam

    # _seq_label_print("final", seq_label)
    # print("** final LM scores:")
    # _generic_print(lm_scores)
    # _seq_label_lm_score("final", seq_label, lm)

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

    # print(f"** Result {seq_targets_wb}:")
    # _generic_seq_label_print(seq_targets_wb, out_spatial_dim)
    # am_scores = rf.reduce_sum(
    #     rf.gather(label_log_prob, axis=model.wb_target_dim, indices=seq_targets_wb), axis=out_spatial_dim
    # )
    # print("** Final result AM scores:")
    # _generic_print(am_scores)
    # print("** Final result (combined) scores:")
    # _generic_print(seq_log_prob)

    # import sys
    # sys.exit(1)

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
                    s = rf.replace_dim_v2(s, in_dim=d, out_dim=dim_map[d])
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


def _state_update(
    prev: Any,
    *,
    target: Tensor,
    beam_dim: Dim,
    model: Model,
    lm: TransformerDecoder,
    lm_scale: float,
) -> Tuple[Tensor, Any, Tensor, Any]:
    from returnn.tensor import batch_dim

    lm_log_probs, lm_state, lm_scores, seq_label = prev

    lm_log_probs = rf.gather(lm_log_probs, axis=model.target_dim, indices=target)  # Batch, Beam
    assert lm_scores.dims_set == lm_log_probs.dims_set == target.dims_set == {batch_dim, beam_dim}
    lm_scores = lm_scores + lm_log_probs  # Batch, Beam

    lm_logits, lm_state = lm(
        target,
        spatial_dim=single_step_dim,
        state=lm_state,
    )  # Batch, Beam, Vocab / ...
    lm_log_probs = rf.log_softmax(lm_logits, axis=model.target_dim)  # Batch, Beam, Vocab
    lm_log_probs *= lm_scale

    seq_label = _seq_label_append(seq_label, target)

    return lm_log_probs, lm_state, lm_scores, seq_label


def _where_deep_check(a: Any, b: Any, ref_result: Any, *, mask: Tensor, mask_cpu: Tensor):
    import tree

    tree.assert_same_structure(a, b)
    tree.assert_same_structure(a, ref_result)

    dim_map = {}
    tree.map_structure(functools.partial(_where_prepare_dims, mask=mask_cpu, dim_map=dim_map), a, b)
    res = tree.map_structure(functools.partial(_where, mask=mask, mask_cpu=mask_cpu, dim_map=dim_map), a, b)

    check_dim_map = {}
    tree.map_structure(functools.partial(_where_res_check_equal_prepare_dims, dim_map=check_dim_map), res, ref_result)
    equal = tree.map_structure_with_path(
        functools.partial(_where_res_check_equal, dim_map=check_dim_map), res, ref_result
    )
    if all(tree.flatten(equal)):
        return res
    print("** Error, some elements are not equal:", equal)
    raise SystemExit(1)


def _where_prepare_dims(a: Any, b: Any, *, mask: Tensor, dim_map: Dict[Dim, Dim]):
    if isinstance(a, Dim):
        assert isinstance(b, Dim)
        if a == b:
            return a
        if a in dim_map:
            return dim_map[a]
        assert b not in dim_map
        a_size = a.get_size_tensor()
        b_size = b.get_size_tensor()
        res_size = rf.where(mask, a_size, b_size, allow_broadcast_all_sources=True)
        res_dim = Dim(res_size, name=_extend_dim_name(b.name))
        dim_map[a] = res_dim
        dim_map[b] = res_dim
        return res_dim
    if isinstance(a, Tensor):
        assert isinstance(b, Tensor)
        return a  # ignored at this stage
    raise TypeError(f"_where_prepare_dims: unexpected type ({type(a)}, {type(b)})")


def _where(a: Any, b: Any, *, mask: Tensor, mask_cpu: Tensor, dim_map: Dict[Dim, Dim]):
    if isinstance(a, Dim):
        assert isinstance(b, Dim)
        if a == b:
            return a
        assert a in dim_map
        assert b in dim_map
        return dim_map[a]
    if isinstance(a, Tensor):
        assert isinstance(b, Tensor)
        for d in a.dims:
            if d in dim_map:
                a = rf.replace_dim_v2(a, in_dim=d, out_dim=dim_map[d])
        for d in b.dims:
            if d in dim_map:
                b = rf.replace_dim_v2(b, in_dim=d, out_dim=dim_map[d])
        assert a.dims_set == b.dims_set
        if a.device == "cpu":
            mask = mask_cpu
        return rf.where(mask, a, b, allow_broadcast_all_sources=True)
    raise TypeError(f"_where: unexpected type ({type(a)}, {type(b)})")


def _where_res_check_equal_prepare_dims(a: Any, b: Any, *, dim_map: Dict[Dim, Dim]):
    if isinstance(a, Dim):
        assert isinstance(b, Dim)
        if a != b:
            dim_map[a] = b


def _where_res_check_equal(path: Tuple[Any, ...], a: Any, b: Any, *, dim_map: Dict[Dim, Dim]):
    import torch

    if isinstance(a, Dim):
        assert isinstance(b, Dim)
        if a == b:
            return True
        assert dim_map[a] == b
        return _where_res_check_equal(path + ("size",), a.get_size_tensor(), b.get_size_tensor(), dim_map=dim_map)
    if isinstance(a, Tensor):
        assert isinstance(b, Tensor)
        for d in a.dims:
            if d in dim_map:
                d_ = dim_map[d]
                _where_res_check_equal(path + (d,), d, d_, dim_map=dim_map)
                a, _ = rf.replace_dim(a, in_dim=d, out_dim=d_)
        assert a.dims_set == b.dims_set
        a = a.copy_transpose(b.dims)
        a = a.copy_masked(0)
        b = b.copy_masked(0)
        a = rf.copy_to_device(a, "cpu")
        b = rf.copy_to_device(b, "cpu")
        try:
            torch.testing.assert_close(a.raw_tensor, b.raw_tensor, rtol=5e-4, atol=1e-5)
        except AssertionError as exc:
            print(f"** Error in {path}:")
            print(exc)
            print(f"a {a}: {a.raw_tensor}")
            print(f"b {b}: {b.raw_tensor}")
            return False
        else:
            return True
    raise TypeError(f"_where_res_check_equal: unexpected type ({type(a)}, {type(b)})")


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
    print(f"* seq_label history {prefix}: {hist}:")
    _generic_seq_label_print(hist, hist_dim)


def _seq_label_lm_score(prefix: str, state: rf.State, lm: TransformerDecoder):
    from returnn.tensor import batch_dim

    print(f"*** scoring seq_label {prefix}...")

    targets_spatial_dim: Dim = state.hist_dim
    targets: Tensor = state.history
    vocab = lm.vocab_dim.vocab
    batch_dims = targets.remaining_dims(targets_spatial_dim)
    assert batch_dim in batch_dims
    # Reorder, but batch_dim first. This is just for nicer visualization, easier debug hacks.
    batch_dims = [batch_dim] + [d for d in batch_dims if d != batch_dim]

    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=vocab.bos_label_id
    )
    targets_w_eos, _ = rf.pad(
        targets,
        axes=[targets_spatial_dim],
        padding=[(0, 1)],
        value=vocab.eos_label_id,
        out_dims=[targets_w_eos_spatial_dim],
    )
    print(f"  seq lens {targets_w_eos_spatial_dim}:")
    _generic_print(targets_w_eos_spatial_dim.dyn_size_ext)

    logits, _ = lm(
        input_labels,
        spatial_dim=targets_w_eos_spatial_dim,
        encoder=None,
        state=lm.default_initial_state(batch_dims=batch_dims),
    )
    log_probs = rf.log_softmax(logits, axis=lm.vocab_dim)  # Batch, InBeam, Spatial, Vocab
    log_probs = rf.gather(log_probs, axis=lm.vocab_dim, indices=targets_w_eos)  # Batch, InBeam, Spatial
    log_probs = rf.reduce_sum(log_probs, axis=targets_w_eos_spatial_dim)  # Batch, InBeam
    print(f"  scores {log_probs}:")
    _generic_print(log_probs)


def _generic_seq_label_print(labels: Tensor, spatial_dim: Dim):
    labels = rf.copy_to_device(labels, "cpu")
    batch_dims = labels.remaining_dims(spatial_dim)
    for indices in _iter_dims_indices(batch_dims):
        print(" ", end="")
        hist_seq_len_ = spatial_dim.get_size_tensor()
        hist_ = labels
        for dim, i in zip(batch_dims, indices):
            hist_ = rf.gather(hist_, axis=dim, indices=i)
            if dim in hist_seq_len_.dims:
                hist_seq_len_ = rf.gather(hist_seq_len_, axis=dim, indices=i)
            print(f" {dim}={i}", end="")
        hist_, _ = rf.slice(hist_, axis=spatial_dim, size=hist_seq_len_)
        print(
            f": len={hist_seq_len_.raw_tensor}"
            f" {[labels.sparse_dim.vocab.id_to_label(l.item()) for l in hist_.raw_tensor]}"
        )


def _generic_print(tensor: Tensor):
    tensor = rf.copy_to_device(tensor, "cpu")
    for indices in _iter_dims_indices(tensor.dims):
        print(" ", end="")
        tensor_ = tensor
        for dim, i in zip(tensor.dims, indices):
            tensor_ = rf.gather(tensor_, axis=dim, indices=i)
            print(f" {dim}={i}", end="")
        print(f": {tensor_.raw_tensor.item()}")


def _iter_dims_indices(dims: Sequence[Dim]) -> Generator[Tuple[int, ...]]:
    if not dims:
        yield ()
        return
    dim, rest = dims[0], dims[1:]
    for i in range(dim.get_dim_value()):
        for rest_indices in _iter_dims_indices(rest):
            yield (i,) + rest_indices
