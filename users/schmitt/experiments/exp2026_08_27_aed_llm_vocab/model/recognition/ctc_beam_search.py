"""
CTC time-sync beam search on the top aux CTC head (no decoder involved).

Ported from :func:`i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc.model_recog`,
behaviour-identical at the time of the port (2026-08-27).

Why it lives here: it only needs ``model.encode_and_get_ctc_log_probs`` / ``blank_idx`` /
``wb_target_dim``, all of which the AED model provides, and it is far cheaper than the AED beam
search. ``exp2026_05_23_returnn.loq_train`` uses exactly this as its per-epoch recog for the
packed runs. Use it via ``recog_def_ctc_only=True`` in the recipe when the per-epoch AED search is
too expensive; it needs ``aux_loss_layers=[<top layer>]`` in the search config.

The same behaviour-version caveat as in :mod:`aed_beam_search` applies to packed-trained models.
"""

from __future__ import annotations

from typing import Tuple

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray

from i6_experiments.users.zeyer.model_interfaces import RecogDef

from ..definitions.aed import Model

__all__ = ["recog_def"]


def recog_def(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    from returnn.config import get_global_config

    config = get_global_config()

    label_log_prob, _, enc_spatial_dim = model.encode_and_get_ctc_log_probs(data, in_spatial_dim=data_spatial_dim)
    batch_dims = label_log_prob.remaining_dims((enc_spatial_dim, label_log_prob.feature_dim))
    beam_size = config.int("beam_size", 12)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob_pre_filter, (backrefs_pre_filter,), pre_filter_beam_dim = rf.top_k(
        label_log_prob,
        k_dim=Dim(min(beam_size, model.wb_target_dim.dimension), name="pre-filter-beam"),
        axis=[model.wb_target_dim],
    )  # seq_log_prob, backrefs_global: Batch, Spatial, PreFilterBeam. backrefs_pre_filter -> Vocab
    label_log_prob_pre_filter_ta = TensorArray.unstack(
        label_log_prob_pre_filter, axis=enc_spatial_dim
    )  # t -> Batch, PreFilterBeam
    backrefs_pre_filter_ta = TensorArray.unstack(backrefs_pre_filter, axis=enc_spatial_dim)  # t -> Batch, PreFilterBeam

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets = []
    seq_backrefs = []
    for t in range(max_seq_len):
        # Filter out finished beams
        # combine_bc: the frame posterior has no beam dim,
        # so neither source has all dims, and the outer product is what the top_k below consumes.
        # torch broadcasts this implicitly, TF wants it explicit.
        seq_log_prob = rf.combine_bc(seq_log_prob, "add", label_log_prob_pre_filter_ta[t])  # B, InBeam, PreFilter
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob,
            k_dim=Dim(min(beam_size, beam_dim.dimension * pre_filter_beam_dim.dimension), name=f"dec-step{t}-beam"),
            axis=[beam_dim, pre_filter_beam_dim],
        )  # seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> PreFilterBeam.
        target = rf.gather(backrefs_pre_filter_ta[t], indices=target)  # Batch, Beam -> Vocab
        seq_targets.append(target)
        seq_backrefs.append(backrefs)

    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_.insert(0, rf.gather(target, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets__ = TensorArray(seq_targets_[0])
    for target in seq_targets_:
        seq_targets__ = seq_targets__.push_back(target)
    out_spatial_dim = enc_spatial_dim
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
recog_def: RecogDef[Model]
recog_def.output_with_beam = True
recog_def.output_blank_label = "<blank>"
recog_def.batch_size_dependent = False  # not totally correct, but we treat it as such...
