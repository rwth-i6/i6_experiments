"""
Copied from here https://github.com/rwth-i6/i6_experiments/blob/zijian_dev/users/yang/torch/decoding/ctc_label_sync.py
Modified to work with RF LM and ILM
"""

from __future__ import annotations

from typing import Optional, Any, Tuple, Dict, Sequence, List
import tree


from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray


# from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.conformer_import_moh_att_2023_06_30 import Model
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
import torch
import numpy

# _ctc_prior_filename = "/u/luca.gaudino/debug/ctc/prior.txt"
# _ctc_prior_filename = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZdcvhAOyWl95/output/prior.txt"


def model_recog_label_sync(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
    max_seq_len: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    from returnn.config import get_global_config

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
    # get ctc output
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    assert model.enc_aux_logits_12, "Expected final ctc logits in enc_aux_logits_12"
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    enc_ctc = model.enc_aux_logits_12(enc_args["enc"])
    


    config = get_global_config()
    search_args = config.typed_value("search_args", {})


    beam_size = search_args.get("beam_size", 12)
    lm_scale = search_args.get("lm_scale", 0.0)

    length_normalization_exponent = search_args.get("length_norm_scale", 1.0)
    lm_linear_combine = search_args.get("lm_linear_combine", False)


    # by default no length norm
    # length_normalization_exponent = model.search_args.get("length_normalization_exponent", 0.0)

    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
    print("** max seq len:", max_seq_len.raw_tensor)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims

    # Init LM and ILM state
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


    ctc_out = (
        enc_ctc
        .copy_transpose((batch_size_dim, enc_spatial_dim, model.target_dim_w_blank))
        .raw_tensor
    )  # [B,T,V+1]
    log_ctc_prob = torch.nn.functional.log_softmax(ctc_out, dim=-1)
    ctc_out = log_ctc_prob # to make sure ctc_out is always (B, T, V+1)

    if search_args.get("mask_eos", True):
        ctc_eos = ctc_out[:, :, model.eos_idx].unsqueeze(2)
        ctc_blank = ctc_out[:, :, model.blank_idx].unsqueeze(2)
        ctc_out[:, :, model.blank_idx] = torch.logsumexp(
            torch.cat([ctc_eos, ctc_blank], dim=2), dim=2
        )
        ctc_out[:, :, model.eos_idx] = -1e30
    from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.espnet_ctc.ctc_prefix_score_espnet import (
        CTCPrefixScoreTH,
    )
    hlens = max_seq_len.raw_tensor
    # prior correction
    if search_args.get("prior_scale", 0.0) > 0.0:
        ctc_prior = numpy.loadtxt(search_args.get("prior_file", search_args.get("ctc_prior_file", "")),
                                      dtype="float32")
        if search_args.get("ctc_log_prior"):
            ctc_log_prior = ctc_prior
        else:
            ctc_log_prior = np.log(ctc_prior)
        ctc_out = ctc_out - (
                torch.tensor(ctc_log_prior)
                .repeat(ctc_out.shape[0], ctc_out.shape[1], 1)
                .to(ctc_out.device)
                * search_args["prior_scale"]
        )
        ctc_out = ctc_out - torch.logsumexp(ctc_out, dim=2, keepdim=True)
    ctc_prefix_scorer = CTCPrefixScoreTH(
        ctc_out,
        hlens,
        blank_index,
        0,
        search_args.get("window_margin", 0),
        search_args.get("mask_eos", True),
    )
    ctc_state = None

    enc_args.pop("ctc", None)

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        ctc_prefix_scores, ctc_state = ctc_prefix_scorer(
            output_length=i,
            last_ids=target_ctc,
            state=ctc_state,
            att_w= None,
        )

        if i == 0:
            ctc_prefix_scores = ctc_prefix_scores.view(batch_size, beam_size, -1)[:, 0, :].unsqueeze(1)
        else:
            ctc_prefix_scores = ctc_prefix_scores.view(batch_size, beam_size, -1)

        ctc_prefix_scores = rf.Tensor(
            name="ctc_prefix_scores",
            # dims=batch_dims_ + [model.target_dim],
            dims=[batch_size_dim, beam_dim, model.target_dim],
            dtype="float32",
            raw_tensor=ctc_prefix_scores[:, :, :blank_index],
        )
        # what is the beam_dim here
        label_log_prob = ctc_prefix_scores

        if search_args.get("lm_scale", 0.0) > 0:
            lm_out = model.language_model(target, state=lm_state, spatial_dim=single_step_dim)
            lm_state = lm_out["state"]
            lm_log_prob = rf.log_softmax(lm_out["output"], axis=model.target_dim)

            label_log_prob = (
                label_log_prob + search_args["lm_scale"] * lm_log_prob
            )


        if search_args.get("ilm_scale", 0.0) > 0:
            ilm_out = model.ilm(target, state=ilm_state, spatial_dim=single_step_dim)
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
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob,
            k_dim=Dim(beam_size, name=f"dec-step{i}-beam"),
            axis=[beam_dim, model.target_dim],
        )  # seq_log_prob, backrefs, target: Batch, Beam
        seq_targets.append(target)
        seq_backrefs.append(backrefs)

        # update LM and ILM states
        if search_args.get("lm_scale", 0.0) > 0:
            lm_state = model.language_model.select_state(lm_state, backrefs)
        
        if search_args.get("ilm_scale", 0.0) > 0:
            ilm_state = model.ilm.select_state(ilm_state, backrefs)


        

        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        i += 1

        best_ids = target
        if search_args.get("ctc_state_fix", True):
            # if i >= 1:
            #     best_ids = target + model.target_dim.get_dim_value()
            best_ids = target + backrefs * (model.target_dim.get_dim_value() + 1)

        # ctc state selection
        ctc_state = ctc_prefix_scorer.index_select_state(
            ctc_state, best_ids.raw_tensor
        )
        target_ctc = torch.flatten(target.raw_tensor)

        ended = rf.logical_or(ended, target == model.eos_idx)
        ended = rf.logical_or(ended, rf.copy_to_device(i >= max_seq_len))
        if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break
        out_seq_len = out_seq_len + rf.where(ended, 0, 1)

        if i > 1 and length_normalization_exponent != 0:
            # Length-normalized scores, so we evaluate score_t/len.
            # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
            # Because we count with EOS symbol, shifted by one.
            seq_log_prob *= rf.where(
                ended,
                (i / (i - 1)) ** length_normalization_exponent,
                1.0,
            )

    if i > 0 and length_normalization_exponent != 0.0:
        seq_log_prob *= (1 / i) ** length_normalization_exponent

    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target in zip(
        seq_backrefs[::-1], seq_targets[::-1]
    ):  # [::-1] reverse
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_.insert(0, rf.gather(target, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets__ = TensorArray(seq_targets_[0])
    for target in seq_targets_:
        seq_targets__ = seq_targets__.push_back(target)
    out_spatial_dim = Dim(out_seq_len, name="out-spatial")
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog_label_sync: RecogDef[Model]
model_recog_label_sync.output_with_beam = True
model_recog_label_sync.output_blank_label = "<blank>"
model_recog_label_sync.batch_size_dependent = False