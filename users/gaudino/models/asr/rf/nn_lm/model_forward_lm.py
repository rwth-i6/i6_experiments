from __future__ import annotations

from typing import Optional, Any, Tuple, Dict, Sequence, List
import tree


from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray

from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef

import torch
import numpy


def model_forward_lm(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
    # max_seq_len: Optional[int] = None,
    ground_truth: Tensor,
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
    batch_dims = [ground_truth.dims[0]]
    # beam_size = 1

    # length_normalization_exponent = model.search_args.get("length_normalization_exponent", 1.0)
    seq_len = ground_truth.dims[1].get_size_tensor()
    print("** seq len:", seq_len.raw_tensor)

    # Eager-mode implementation of beam search.
    # Initial state.
    # beam_dim = Dim(1, name="initial-beam")
    # batch_dims_ = batch_dims + [beam_dim]
    # decoder_state = model.decoder_default_initial_state(
    #     batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim
    # )
    # out_seq_len = rf.constant(0, dims=batch_dims_)
    ended = rf.constant(False, dims=batch_dims)
    target = rf.constant(0, dims=batch_dims, sparse_dim=model.in_dim)
    seq_log_prob = rf.constant(0.0, dims=batch_dims)

    trafo_lm_state = model.default_initial_state(batch_dims=batch_dims)

    assert len(batch_dims) == 1

    ground_truth_pad = rf.pad(ground_truth, axes=ground_truth.dims, padding=[(0, 0), (0, 2)], out_dims=ground_truth.dims, value=0)[0]

    for i in range(torch.max(seq_len.raw_tensor)+1):
        lm_out = model(target, state=trafo_lm_state, spatial_dim=single_step_dim)
        label_log_prob = rf.log_softmax(lm_out["output"], axis=model.target_dim)
        trafo_lm_state = lm_out["state"]

        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(
                0, # eos index
                axis=model.target_dim,
                label_value=0.0,
                other_value=-1.0e30,
            ),
            label_log_prob,
        )

        seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab

        gt_slice, slice_out_dim = rf.slice(ground_truth_pad, axis=ground_truth_pad.dims[1], start=i, end=i+1)
        gt_slice = rf.copy_to_device(rf.squeeze(gt_slice, slice_out_dim))
        seq_log_prob = rf.gather(seq_log_prob, indices=gt_slice, axis=seq_log_prob.dims[1])
        target = gt_slice

        # ended = rf.logical_or(ended, target == model.eos_idx)
        ended = rf.logical_or(ended, rf.copy_to_device(i >= seq_len))
        if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break

    return ground_truth, seq_log_prob


# RecogDef API
model_forward_lm: RecogDef[Model]
model_forward_lm.output_with_beam = True
model_forward_lm.output_blank_label = "<blank>"
model_forward_lm.batch_size_dependent = False
