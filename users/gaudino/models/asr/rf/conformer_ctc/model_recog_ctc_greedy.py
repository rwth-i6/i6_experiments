from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List, Collection
import tree
import math
import numpy as np
import torch
import hashlib
import contextlib
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.gaudino.model_interfaces.supports_label_scorer_torch import (
    RFModelWithMakeLabelScorer,
)

from i6_experiments.users.gaudino.models.asr.rf.conformer_rnnt.model_conformer_rnnt import Model

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.search_functions import remove_blank_and_eos

import numpy

_batch_size_factor = 160

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.trafo_lm import trafo_lm_kazuki_import

def model_recog(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
    max_seq_len: Optional[int] = None,
    search_args: Optional[Dict[str, Any]] = {},
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
    if hasattr(model, "search_args"):
        search_args = model.search_args

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)

    final_ctc_layer = getattr(model, model.final_ctc_name, None)
    enc_ctc = final_ctc_layer(enc_args["enc"])

    enc_ctc = rf.log_softmax(enc_ctc, axis=model.target_dim_w_blank)

    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
    print("** max seq len:", max_seq_len.raw_tensor)

    beam_dim = Dim(1, name="initial-beam")

    assert len(batch_dims) == 1
    batch_size_dim = batch_dims[0]

    ctc_out = enc_ctc.copy_transpose(
        (batch_size_dim, enc_spatial_dim, model.target_dim_w_blank)
    )  # [B,T,V+1]

    blank_index = model.target_dim.get_dim_value()

    if search_args.get("prior_scale", 0.0) > 0.0:
        ctc_out_raw = ctc_out.raw_tensor
        ctc_prior = numpy.loadtxt(
            search_args.get("ctc_prior_file", None), dtype="float32"
        )
        ctc_log_prior = torch.tensor(ctc_prior)
        if not search_args.get("ctc_log_prior", False):
            ctc_log_prior = torch.log(ctc_log_prior)

        ctc_out_raw = ctc_out_raw - (
            ctc_log_prior
            .repeat(ctc_out_raw.shape[0], ctc_out_raw.shape[1], 1)
            .to("cuda")
            * search_args.get("prior_scale", 0.3)
        )
        if search_args.get("prior_corr_renorm", False):
            ctc_out_raw = ctc_out_raw - torch.logsumexp(ctc_out_raw, dim=2, keepdim=True)
        ctc_out.raw_tensor = ctc_out_raw

    # ctc greedy
    hyps = rf.reduce_argmax(ctc_out, axis=ctc_out.feature_dim).raw_tensor
    scores = rf.reduce_max(ctc_out, axis=ctc_out.feature_dim).raw_tensor
    scores.sum = torch.sum(scores, 1).unsqueeze(1)
    seq_log_prob = rf.Tensor(
        name="seq_log_prob",
        dims=[batch_size_dim, beam_dim],
        dtype="float32",
        raw_tensor=scores.sum,
    )

    max_out_len = max_seq_len.raw_tensor[0]

    seq_targets, out_spatial_dim = remove_blank_and_eos(hyps.unsqueeze(1), max_out_len, batch_dims, beam_dim, model.target_dim, blank_idx=blank_index, eos_idx=0)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim



# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
# output_blank_label=blank is actually wrong for AED, but now we don't change it anymore
# because it would change all recog hashes.
# Also, it does not matter too much -- it will just cause an extra SearchRemoveLabelJob,
# which will not have any effect here.
model_recog.output_blank_label = None
model_recog.batch_size_dependent = False

def model_recog_ctc_zeyer(
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
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    final_ctc_layer = getattr(model, model.final_ctc_name, None)
    logits = final_ctc_layer(enc_args["enc"])
    # logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    beam_size = 12

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    label_log_prob = rf.log_softmax(logits, axis=model.target_dim_w_blank)  # Batch, Spatial, Vocab
    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.target_dim_w_blank, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob_pre_filter, (backrefs_pre_filter,), pre_filter_beam_dim = rf.top_k(
        label_log_prob, k_dim=Dim(beam_size, name=f"pre-filter-beam"), axis=[model.target_dim_w_blank]
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
        seq_log_prob = seq_log_prob + label_log_prob_pre_filter_ta[t]  # Batch, InBeam, PreFilterBeam
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, pre_filter_beam_dim]
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
model_recog_ctc_zeyer: RecogDef[Model]
model_recog_ctc_zeyer.output_with_beam = True
model_recog_ctc_zeyer.output_blank_label = "<blank>"
model_recog_ctc_zeyer.batch_size_dependent = False  # not totally correct, but we treat it as such...