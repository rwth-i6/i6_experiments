from __future__ import annotations

from typing import Optional, Any, Tuple, Dict, Sequence, List
import tree
from functools import partial


from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray


# from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.conformer_import_moh_att_2023_06_30 import Model
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.search_functions import (
    remove_blank_and_eos,
)

import torch
import numpy

# _ctc_prior_filename = "/u/luca.gaudino/debug/ctc/prior.txt"
# _ctc_prior_filename = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZdcvhAOyWl95/output/prior.txt"


def model_forward_ctc_sum(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
    max_seq_len: Optional[int] = None,
    search_args: Optional[Dict[str, Any]] = None,
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
    if hasattr(model, "search_args"):
        search_args = model.search_args

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    beam_size = 1
    length_normalization_exponent = search_args.get("length_normalization_exponent", 1.0)
    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")

    if search_args.get("remove_eos_from_gt", False):
        ground_truth.raw_tensor = ground_truth.raw_tensor[:, :-1]
        ground_truth.dims[1].dyn_size_ext = ground_truth.dims[1].get_size_tensor() - 1

    print("** max seq len:", max_seq_len.raw_tensor)
    # print("** ground truth:", ground_truth.raw_tensor)
    # print("** ground truth lens:", ground_truth.dims[1].get_size_tensor().raw_tensor)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = batch_dims + [beam_dim]
    decoder_state = model.decoder_default_initial_state(
        batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim
    )

    if search_args.get("lm_scale", 0.0) > 0:
        lm_state = model.language_model.default_initial_state(batch_dims=batch_dims_)

    if search_args.get("ilm_scale", 0.0) > 0:
        ilm_state = model.ilm.default_initial_state(batch_dims=batch_dims_)

    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
    ended = rf.constant(False, dims=batch_dims_)
    out_seq_len = rf.constant(0, dims=batch_dims_)
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    assert len(batch_dims) == 1
    batch_size_dim = batch_dims[0]
    batch_size = batch_dims[0].get_dim_value()
    target_ctc = [model.bos_idx for _ in range(batch_size * beam_size)]

    blank_index = model.target_dim.get_dim_value()

    ctc_out_raw = (
        enc_args["ctc"]
        .copy_transpose((batch_size_dim, enc_spatial_dim, model.target_dim_w_b))
        .raw_tensor
    )  # [B,T,V+1]

    if search_args.get("mask_eos", True):
        ctc_eos = ctc_out_raw[:, :, model.eos_idx].unsqueeze(2)
        ctc_blank = ctc_out_raw[:, :, model.blank_idx].unsqueeze(2)
        ctc_out_raw[:, :, model.blank_idx] = torch.logsumexp(
            torch.cat([ctc_eos, ctc_blank], dim=2), dim=2
        )
        ctc_out_raw[:, :, model.eos_idx] = -1e30

    ctc_out = rf.Tensor(
        name="ctc_out",
        dims=(batch_size_dim, enc_spatial_dim, model.target_dim_w_b),
        dtype="float32",
        raw_tensor=ctc_out_raw,
    )

    enc_args.pop("ctc")

    # compute ctc sum scores
    #
    from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.two_pass import ctc_forward_algorithm

    ctc_scores = torch.zeros((batch_size), dtype=torch.float32)
    for i in range(batch_size):
        seq = ground_truth.raw_tensor[i]
        seq = seq[seq != 0]
        ctc_scores[i] = ctc_forward_algorithm(ctc_out_raw[i].to('cpu'), seq.to('cpu'), blank_index)

    ctc_scores = rf.Tensor('ctc_scores', [batch_size_dim], dtype="float32", raw_tensor=ctc_scores)
    ctc_scores = rf.copy_to_device(ctc_scores, "cuda")

    # compute att scores
    #
    ground_truth_pad = rf.pad(ground_truth, axes=ground_truth.dims, padding=[(0, 0), (0, 1)], out_dims=ground_truth.dims, value=0)[0]

    i = 0
    while True:
        # fixed: before it was computed at step 0
        if i == 0:
            input_embed = rf.zeros(batch_dims_ + [model.target_embed.out_dim], feature_dim=model.target_embed.out_dim)
        else:
            input_embed = model.target_embed(target)

        step_out, decoder_state = model.loop_step(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=input_embed,
            state=decoder_state,
        )
        step_out.pop("att_weights", None)
        logits = model.decode_logits(input_embed=input_embed, **step_out)
        label_log_prob = rf.log_softmax(
            logits, axis=model.target_dim
        )  # (Dim{'initial-beam'(1)}, Dim{B}, Dim{F'target'(10025)})

        label_log_prob = label_log_prob * search_args.get("att_scale", 1.0)

        if search_args.get("lm_scale", 0.0) > 0:
            lm_out = model.language_model(target, state=lm_state, spatial_dim=single_step_dim)
            lm_state = lm_out["state"]
            lm_log_prob = rf.log_softmax(lm_out["output"], axis=model.target_dim)

            label_log_prob = (
                label_log_prob + search_args["lm_scale"] * lm_log_prob
            )

        if search_args.get("ilm_scale", 0.0) > 0:
            ilm_out = model.ilm(input_embed, state=ilm_state, spatial_dim=single_step_dim)
            ilm_state = ilm_out["state"]
            ilm_log_prob = rf.log_softmax(ilm_out["output"], axis=model.target_dim)

            label_log_prob = (
                label_log_prob - search_args["ilm_scale"] * ilm_log_prob
            )

        # Filter out finished beams
        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(
                model.eos_idx,
                axis=model.target_dim,
                label_value=0.0,
                other_value=-1.0e30,
            ),
            label_log_prob,
        )
        seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab

        gt_slice, slice_out_dim = rf.slice(ground_truth_pad, axis=ground_truth_pad.dims[1], start=i, end=i+1)
        gt_slice = rf.copy_to_device(rf.squeeze(gt_slice, slice_out_dim))
        seq_log_prob = rf.gather(seq_log_prob, indices=gt_slice, axis=seq_log_prob.dims[2])
        target = rf.reshape(gt_slice, gt_slice.dims, target.dims)

        i += 1

        ended = rf.logical_or(ended, target == model.eos_idx)
        ended = rf.logical_or(ended, rf.copy_to_device(i >= max_seq_len))
        if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break
        out_seq_len = out_seq_len + rf.where(ended, 0, 1)

        if i > 1 and length_normalization_exponent != 0:
            # Length-normalized scores, so we evaluate score_t/len
            # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
            # Because we count with EOS symbol, shifted by one.
            seq_log_prob *= rf.where(
                ended,
                (i / (i - 1)) ** length_normalization_exponent,
                1.0,
            )

    if i > 0 and length_normalization_exponent != 0:
        seq_log_prob *= (1 / i) ** length_normalization_exponent

    seq_log_prob = seq_log_prob + ctc_scores * search_args.get("ctc_scale", 0.0)

    out_spatial_dim = ground_truth.dims[1]
    seq_targets = rf.reshape(ground_truth, ground_truth.dims, batch_dims_ + [out_spatial_dim])

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim
    # return seq_targets, seq_log_prob, viterbi_paths, out_spatial_dim, beam_dim


# RecogDef API
model_forward_ctc_sum: RecogDef[Model]
model_forward_ctc_sum.output_with_beam = True
model_forward_ctc_sum.output_blank_label = "<blank>"
model_forward_ctc_sum.batch_size_dependent = False
