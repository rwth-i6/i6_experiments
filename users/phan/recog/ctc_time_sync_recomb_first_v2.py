"""
Modified to do recombination before pruning
https://github.com/rwth-i6/i6_experiments/blob/main/users/gaudino/models/asr/rf/conformer_ctc/model_recog_ctc_ts.py
"""
from __future__ import annotations

import copy
from typing import Optional, Any, Tuple, Dict, Sequence, List
import tree
from functools import partial


from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray


# from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.conformer_import_moh_att_2023_06_30 import Model
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
from i6_experiments.users.phan.recog.search_functions import (
    remove_blank_and_eos,
)

from i6_experiments.users.phan.recog.blank_collapse import (
    blank_collapse_batched,
)

from i6_experiments.users.phan.rf_models.trafo_lm_luca import Trafo_LM_Model
from i6_experiments.users.phan.rf_models.lstm_lm_luca import LSTM_LM_Model
from i6_experiments.users.phan.rf_models.lstm_lm_luca_hardcoded_layers import LSTM_LM_Model_Hardcoded_Layers

import torch
import numpy

import time

_ctc_prior_filename = "/u/luca.gaudino/debug/ctc/prior.txt"
# _ctc_prior_filename = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZdcvhAOyWl95/output/prior.txt"

# debug print settigns
torch.set_printoptions(precision=2, linewidth=200, threshold=10000)

def model_recog_time_sync_recomb_first_v2(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
    max_seq_len: Optional[int] = None,
    search_args: Optional[Dict[str, Any]] = None,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    Modified to do recombination before pruning
    This only compatible with the kazuki trafo LM
    (but can be further modified to work with the tedlium2 lstm)

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    # if hasattr(model, "search_args"):
    #     search_args = model.search_args

    from returnn.config import get_global_config
    config = get_global_config()
    search_args = config.typed_value("search_args", {})

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    enc_ctc = model.enc_aux_logits_12(enc_args["enc"])

    enc_ctc = rf.softmax(enc_ctc, axis=model.target_dim_w_blank)

    beam_size = search_args.get("beam_size", 32)
    if "length_normalization_exponent" in search_args:
        length_normalization_exponent = search_args.get("length_normalization_exponent")
    elif "length_norm_scale" in search_args:
        length_normalization_exponent = search_args.get("length_norm_scale")
    else:
        raise RuntimeError("search_args must contain either length_normalization_exponent or length_norm_scale")
    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
    print("** max seq len:", max_seq_len.raw_tensor)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = batch_dims + [beam_dim]

    # Init lm state
    if search_args.get("lm_scale", 0.0) > 0.0:
        if isinstance(model.language_model, Trafo_LM_Model):
            trafo_lm_state = _get_init_trafo_state(model.language_model, batch_dims_)
        elif isinstance(model.language_model, LSTM_LM_Model) or isinstance(model.language_model, LSTM_LM_Model_Hardcoded_Layers):
            lstm_lm_state = model.language_model.default_initial_state(batch_dims=batch_dims_)
            prev_lstm_lm_state = lstm_lm_state
        else:
            raise NotImplementedError(f"External LM type {model.language_model.__class__.__name__} is not supported")
    if search_args.get("ilm_scale", 0.0) > 0:
        ilm_state = model.ilm.default_initial_state(batch_dims=batch_dims_)
        prev_ilm_state = ilm_state

    initial_target = rf.constant(
        model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim_w_blank
    )
    target = initial_target
    ended = rf.constant(False, dims=batch_dims_)
    out_seq_len = rf.constant(0, dims=batch_dims_)
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)
    eos_log_prob = rf.constant(0.0, dims=batch_dims_)

    assert len(batch_dims) == 1
    batch_size_dim = batch_dims[0]
    batch_size = batch_dims[0].get_dim_value()
    target_ctc = [model.bos_idx for _ in range(batch_size * beam_size)]

    blank_index = model.target_dim.get_dim_value()

    ctc_out_raw = enc_ctc.copy_transpose(
        (batch_size_dim, enc_spatial_dim, model.target_dim_w_blank)
    ).raw_tensor  # [B,T,V+1]

    if search_args.get("mask_eos", True):
        ctc_eos = ctc_out_raw[:, :, model.eos_idx].unsqueeze(2)
        ctc_blank = ctc_out_raw[:, :, model.blank_idx].unsqueeze(2)
        ctc_out_raw[:, :, model.blank_idx] = torch.sum(
            torch.cat([ctc_eos, ctc_blank], dim=2), dim=2
        )
        ctc_out_raw[:, :, model.eos_idx] = 0.0

    orig_max_seq_len_raw = max_seq_len.raw_tensor

    ctc_spatial_dim = enc_spatial_dim
    if search_args.get("blank_collapse", False):
        col_probs, col_lens = blank_collapse_batched(
            ctc_out_raw.to("cpu"),
            orig_max_seq_len_raw,
            search_args.get("blank_threshold", 0.0),
            blank_index,
        )
        ctc_spatial_dim = enc_spatial_dim.copy(description="ctc-spatial")
        orig_max_seq_len_raw = col_lens.to(torch.int32)
        ctc_spatial_dim.dyn_size_ext.raw_tensor = orig_max_seq_len_raw
        ctc_out_raw = col_probs.to("cuda")

    # important to do this after blank collapse
    if search_args.get("prior_scale", 0.0) > 0.0:
        prior_type = search_args.get("prior_type", "precomputed_average")
        if prior_type == "precomputed_average":
            ctc_log_prior = numpy.loadtxt(
                search_args.get("prior_file", None), dtype="float32"
            )

            if not search_args.get("ctc_log_prior", False):
                ctc_log_prior = numpy.log(ctc_log_prior)

            ctc_log_prior = (
                torch.tensor(ctc_log_prior)
                .repeat(ctc_out_raw.shape[0], ctc_out_raw.shape[1], 1)
                .to(enc_ctc.raw_tensor.device)
            )
        elif prior_type == "zero_encoder":
            prior_args, prior_spatial_dim = model.encode(
                data,
                in_spatial_dim=data_spatial_dim,
                audio_features_mask=torch.zeros(ctc_out_raw.shape[:2], device=data.raw_tensor.device),
                )
            enc_prior = model.enc_aux_logits_12(prior_args["enc"])
            enc_prior = rf.log_softmax(enc_prior, axis=model.target_dim_w_blank)
            ctc_log_prior = (
                enc_prior
                .copy_transpose((batch_size_dim, prior_spatial_dim, model.target_dim_w_blank))
                .raw_tensor
            )  # [B,T,V+1]
            # print("Zero encoder log prior")
            # print(ctc_log_prior)
        else:
            raise NotImplementedError(f"Prior type {prior_type} not implemented")
        # better do this in log space
        ctc_out_log_raw = torch.log(ctc_out_raw)

        if search_args.get("renormalize_prior", False):
            ctc_log_prior[:, :, blank_index] = 0.0
            ctc_log_prior[:, :, :blank_index] = ctc_log_prior[:, :, :blank_index].log_softmax(-1)

        ctc_out_log_raw = ctc_out_log_raw - (
            ctc_log_prior * search_args.get("prior_scale", 0.1)
        )
        ctc_out_log_raw = ctc_out_log_raw - torch.logsumexp(
            ctc_out_log_raw, dim=2, keepdim=True
        )

        ctc_out_raw = torch.exp(ctc_out_log_raw)

    if search_args.get("blank_scale", 0.0) > 0.0:
        ctc_blank = ctc_out_raw[:, :, model.blank_idx]
        ctc_out_raw[:, :, model.blank_idx] = ctc_blank - search_args.get(
            "blank_scale", 0.0
        )

    # ctc_out = rf.Tensor(
    #     name="ctc_out",
    #     dims=(batch_size_dim, ctc_spatial_dim, model.target_dim_w_blank),
    #     dtype="float32",
    #     raw_tensor=ctc_out_raw,
    # )

    ctc_out_log_raw = torch.log(ctc_out_raw)

    # enc_args.pop("ctc")

    prev_target = initial_target
    prev_target_non_blank = initial_target

    eps = 1e-20
    i = 0
    seq_targets = []
    seq_backrefs = []
    # contains hyps without any blank, e.g.  (0, 2, 5, 9), (0, 3, 5)
    hyps_label_idxs = [[(model.bos_idx,)] * beam_size for _ in range(batch_size)]
    recomb_possible_pairs = [set() for _ in range(batch_size)]

    if search_args.get("add_eos_to_end", False):
        max_seq_len = max_seq_len + 1  # add one step to loop

    t = time.time()

    for i in range(torch.max(max_seq_len.raw_tensor)):
        # print("Loop step", i)
        is_last_step = i + 1 == torch.max(max_seq_len.raw_tensor)

        # gather prev non-blank targets and prev_decoder_state via backrefs
        if i == 1:
            prev_target = rf.gather(initial_target, indices=backrefs)
        elif i > 1:
            prev_target = rf.gather(seq_targets[i - 2], indices=backrefs)

        # handle states
        if i > 0:
            mask_combined_gather = rf.gather(mask_combined, indices=backrefs)
            prev_target_non_blank_gather = rf.gather(
                prev_target_non_blank, indices=backrefs
            )
            prev_target_non_blank = rf.where(
                mask_combined_gather, prev_target, prev_target_non_blank_gather
            )

            # decoder_state = tree.map_structure(
            #     lambda s: rf.gather(s, indices=backrefs), decoder_state
            # )
            # prev_decoder_state = tree.map_structure(
            #     lambda s: rf.gather(s, indices=backrefs),
            #     decoder_state_1,
            # )

            if search_args.get("lm_scale", 0.0) > 0:
                if isinstance(model.language_model, LSTM_LM_Model) or isinstance(model.language_model, LSTM_LM_Model_Hardcoded_Layers):
                    lstm_lm_state = model.language_model.select_state(lstm_lm_state, backrefs)
                    prev_lstm_lm_state = model.language_model.select_state(lstm_lm_state_1, backrefs)

            if search_args.get("ilm_scale", 0.0) > 0:  # TODO
                ilm_state = model.ilm.select_state(ilm_state, backrefs)
                prev_ilm_state = model.ilm.select_state(ilm_state_1, backrefs)

        mask_not_blank = rf.compare(target, "not_equal", model.blank_idx)
        mask_not_repeat = rf.compare(target, "not_equal", prev_target)
        mask_combined = rf.logical_and(mask_not_blank, mask_not_repeat)

        if not torch.any(mask_combined.raw_tensor) or i == 0:
            # decoder_state_1 = prev_decoder_state
            target_1 = prev_target_non_blank
        else:
            partial_mask_function = partial(rf.where, mask_combined)
            # decoder_state_1 = tree.map_structure(
            #     lambda s, prev_s: partial_mask_function(s, prev_s),
            #     decoder_state,
            #     prev_decoder_state,
            # )
            target_1 = rf.where(mask_combined, target, prev_target_non_blank)
        # print(f"target_1: {target_1}")

        if search_args.get("lm_scale", 0.0) > 0.0:
            if isinstance(model.language_model, Trafo_LM_Model):
                if i > 0:
                    trafo_lm_state_1 = _get_masked_trafo_state(
                        trafo_lm_state_1, trafo_lm_state, mask_combined, backrefs
                    )
                else:
                    trafo_lm_state_1 = trafo_lm_state
            elif isinstance(model.language_model, LSTM_LM_Model) or isinstance(model.language_model, LSTM_LM_Model_Hardcoded_Layers):
                if not torch.any(mask_combined.raw_tensor) or i == 0: # no recombination, analogous to ILM case
                    lstm_lm_state_1 = prev_lstm_lm_state
                else:
                    lstm_lm_state_1 = tree.map_structure(
                        lambda s, prev_s: partial_mask_function(s, prev_s),
                        lstm_lm_state,
                        prev_lstm_lm_state,
                    )
                    prev_lstm_lm_state = lstm_lm_state # Reset previous ilm state for next time step

        # ---------------------------------------------------------------
        # Set the state of the ILM
        # Only use new state if there are some recombinations
        # ilm_state_1 should only exist inside this time step
        # ilm_state is the previous output ILM state from last time step
        # ---------------------------------------------------------------
        if search_args.get("ilm_scale", 0.0) > 0.0:
            # if i > 0:
            #     print("backrefs", backrefs.raw_tensor)
            #     print("target", target.raw_tensor)
            if not torch.any(mask_combined.raw_tensor) or i == 0: # no recombination
                # select state happens to "synchronize" the beam dim (dec-step{n}-beam to dec-step{n+1}-beam)
                ilm_state_1 = prev_ilm_state
            else:
                ilm_state_1 = tree.map_structure(
                    lambda s, prev_s: partial_mask_function(s, prev_s),
                    ilm_state,
                    prev_ilm_state,
                )
                prev_ilm_state = ilm_state # Reset previous ilm state for next time step

        # remove blank from target
        target_1.sparse_dim = model.target_dim


        # continue in pure pytorch because slicing is easier

        # add beam dim to ctc_out_raw and get step i
        ctc_index = min(i, torch.max(orig_max_seq_len_raw) - 1)
        ctc_out_log_raw_step = ctc_out_log_raw.unsqueeze(1).repeat(
            [1, beam_dim.get_dim_value(), 1, 1]
        )[
            :, :, ctc_index
        ]  # [B, beam, T, V+1]
        ctc_non_blank = ctc_out_log_raw_step[:, :, :blank_index]

        # renormalize ctc_out_raw_step
        # ctc_non_blank = ctc_non_blank - torch.logsumexp(
        #     ctc_non_blank, dim=2, keepdim=True
        # )

        label_log_prob_non_blank = (
            ctc_non_blank * search_args.get("ctc_scale", 1.0)
            # + att_label_log_prob_raw
        )

        # print("target_1 line", target_1)
        if search_args.get("lm_scale", 0.0) > 0.0:
            if isinstance(model.language_model, Trafo_LM_Model):
                if (
                    not torch.any(mask_combined.raw_tensor)
                    and i > 0
                    and search_args.get("lm_skip", False)
                ):
                    trafo_log_prob = rf.gather(trafo_log_prob, indices=backrefs)
                    trafo_lm_state = _get_trafo_state_after(trafo_lm_state, backrefs)
                else:
                    trafo_lm_out = model.language_model(
                        target_1, state=trafo_lm_state_1, spatial_dim=single_step_dim
                    )
                    trafo_lm_state = trafo_lm_out["state"]

                    trafo_log_prob = rf.log_softmax(
                        trafo_lm_out["output"], axis=model.target_dim
                    )

                trafo_log_prob_raw = trafo_log_prob.raw_tensor
                if search_args.get("add_eos_to_end", False) and is_last_step:
                    eos_log_prob.raw_tensor = eos_log_prob.raw_tensor + trafo_log_prob_raw[
                        :, :, model.eos_idx
                    ] * search_args.get("lm_scale", 0.0)

                if search_args.get("remove_trafo_lm_eos", False):
                    # warning this basically set eos to 0 for the whole prob distribution
                    trafo_log_prob_raw[:, :, model.eos_idx] = -1e30
                    # trafo_log_prob_raw = trafo_log_prob_raw - torch.logsumexp(
                    #     trafo_log_prob_raw, dim=2, keepdim=True
                    # )

                label_log_prob_non_blank = (
                    label_log_prob_non_blank
                    + search_args.get("lm_scale", 0.0)
                    * trafo_log_prob_raw  # still raw tensor for omt slicing etc
                )
            elif isinstance(model.language_model, LSTM_LM_Model) or isinstance(model.language_model, LSTM_LM_Model_Hardcoded_Layers):
                lstm_lm_out = model.language_model(target_1, state=lstm_lm_state_1, spatial_dim=single_step_dim)
                lstm_lm_state = lstm_lm_out["state"]
                lstm_lm_log_prob = rf.log_softmax(lstm_lm_out["output"], axis=model.target_dim)
                if search_args.get("add_eos_to_end", False) and is_last_step:
                    eos_log_prob.raw_tensor = eos_log_prob.raw_tensor + lstm_lm_log_prob.raw_tensor[
                        :, :, model.eos_idx
                    ] * search_args.get("lm_scale", 0.0)

                label_log_prob_non_blank = (
                    label_log_prob_non_blank + search_args.get("lm_scale", 0.0) * lstm_lm_log_prob.raw_tensor
                )
        # print("label_log_prob_non_blank", label_log_prob_non_blank)

        # ----------------------------------------------------------
        # Calculate ILM score and subtract
        # ilm_state is the output ILM state for next time step
        # ----------------------------------------------------------
        if search_args.get("ilm_scale", 0.0) > 0:
            # target_1: (Batch, Beam). State should be (Batch, Beam, Hidden state dim)
            ilm_out = model.ilm(target_1, state=ilm_state_1, spatial_dim=single_step_dim)
            ilm_state = ilm_out["state"]
            # print("ilm state after forwarding", ilm_state.raw_tensor)
            ilm_log_prob = rf.log_softmax(ilm_out["output"], axis=model.target_dim)
            # print("ilm_state out", ilm_state)
            if search_args.get("add_eos_to_end", False) and is_last_step:
                eos_log_prob.raw_tensor = eos_log_prob.raw_tensor - ilm_log_prob.raw_tensor[
                    :, :, model.eos_idx
                ] * search_args.get("ilm_scale", 0.0)

            label_log_prob_non_blank = (
                label_log_prob_non_blank - search_args.get("ilm_scale", 0.0) * ilm_log_prob.raw_tensor
            )

        blank_log_prob = ctc_out_log_raw_step[:, :, blank_index]
        repeat_log_prob = torch.gather(
            ctc_out_log_raw_step, 2, target.raw_tensor.to(torch.int64).unsqueeze(2)
        ).squeeze(2)

        if search_args.get("use_one_minus_term", False):
            ctc_out_raw_step = ctc_out_raw.unsqueeze(1).repeat(
                [1, beam_dim.get_dim_value(), 1, 1]
            )[
                :, :, ctc_index
            ]  # [B, beam, T, V+1]
            blank_prob = ctc_out_raw_step[:, :, blank_index]
            repeat_prob = torch.gather(
                ctc_out_raw_step, 2, target.raw_tensor.to(torch.int64).unsqueeze(2)
            ).squeeze(2)
            one_minus_term = torch.ones_like(blank_prob) - blank_prob

            one_minus_term = torch.where(
                mask_not_blank.raw_tensor,
                one_minus_term - repeat_prob,
                one_minus_term,
            )
            one_minus_term_log = torch.log(
                torch.maximum(one_minus_term, torch.fill(one_minus_term, eps))
            ).unsqueeze(2)
            label_log_prob_non_blank = one_minus_term_log + label_log_prob_non_blank

        label_log_prob = torch.cat(
            [label_log_prob_non_blank, blank_log_prob.unsqueeze(2)], dim=2
        )

        label_log_prob = label_log_prob.scatter_(
            2,
            target.raw_tensor.unsqueeze(2).to(torch.int64),
            repeat_log_prob.unsqueeze(2),
        )

        label_log_prob = rf.Tensor(
            name="label_log_prob",
            dims=(batch_size_dim, beam_dim, model.target_dim_w_blank),
            dtype="float32",
            raw_tensor=label_log_prob,
        )

        # Filter out finished beams
        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(
                model.eos_idx,
                axis=model.target_dim_w_blank,
                label_value=0.0,
                other_value=-1.0e30,
            ),
            label_log_prob,
        )

        if search_args.get("add_eos_to_end", False) and is_last_step:
            seq_log_prob_w_eos = seq_log_prob + eos_log_prob

            seq_log_prob = rf.where(
                ended,
                seq_log_prob,
                seq_log_prob_w_eos,
            ) # dont add it twice
            break
        
        seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab (batch, beam, V)

        # --------------------------- recombination -------------------------------
        # recomb between possible pairs of hypotheses
        for batch in range(batch_size):
            for beam1, beam2 in recomb_possible_pairs[batch]:
                hyp1, hyp2 = hyps_label_idxs[batch][beam1], hyps_label_idxs[batch][beam2]
                if len(hyp1) + 1 == len(hyp2) and hyp1 == hyp2[:-1]:
                    # last output label in the collapsed sequence
                    last_label1, last_label2 = hyp1[-1], hyp2[-1]

                    # last symbol in the alignment
                    target_prev1, target_prev2 = target.raw_tensor[batch][[beam1, beam2]]

                    # no recomb possible here, since beam 1 can not emit new label
                    if last_label1 == last_label2 and target_prev1 != model.blank_idx:
                        continue

                    # consider candidates in V dim to recombine
                    beam1_cands = [last_label2]
                    beam2_cands = [model.blank_idx]
                    if target_prev2 != model.blank_idx:
                        beam2_cands.append(last_label2)

                    # create tuple to slice tensor
                    recom_beams = tuple([beam1] * len(beam1_cands) + [beam2] * len(beam2_cands))
                    recom_targets = tuple(beam1_cands + beam2_cands)
                    
                    # keep max, replace other with -inf
                    considered_scores = seq_log_prob.raw_tensor[batch][recom_beams, recom_targets]
                    arg_max_score = considered_scores.argmax()
                    seq_log_prob.raw_tensor[batch][recom_beams, recom_targets] = torch.where(
                        torch.arange(len(recom_beams)).to(seq_log_prob.device) == arg_max_score,
                        considered_scores,
                        -1e30,
                    )
                    # if i > 1000:
                    #     print("hyp1", hyp1, "target1", target_prev1)
                    #     print("hyp2", hyp2, "target2", target_prev2)
                    #     print("recom beams", recom_beams)
                    #     print("recom targets", recom_targets)
                    #     print("considered scores", considered_scores)
                    #     print("beam scores after recomb", seq_log_prob.raw_tensor[batch][recom_beams, recom_targets])
            
        # recomb within one hypothesis
        # e.g. beam (a,b,c) with target c -> must recombine c and blank in expansion
        seq_log_prob_raw = seq_log_prob.raw_tensor # (batch, beam, V)
        target_raw = target.raw_tensor # (batch, beam)
        # only recombine when hypothesis not end with blank
        target_not_blank_not_eos = (target_raw != model.bos_idx) & (target_raw != model.blank_idx)
        target_and_blank = torch.concat( 
            [target_raw.unsqueeze(-1), torch.full_like(target_raw.unsqueeze(-1), model.blank_idx)],
            dim=-1
        ) # (batch, beam, 2) target, blank
        # print("target and blank", target_and_blank)
        seq_log_prob_lastlabel_blank = seq_log_prob_raw.gather(-1, target_and_blank.long()) # score of target vs blank (batch, beam, 2)
        # print("score before recomb", seq_log_prob_lastlabel_blank)
        max_score, max_score_idx = torch.topk(seq_log_prob_lastlabel_blank, k=1, dim=-1) # take max
        recomb_score = torch.full_like(seq_log_prob_lastlabel_blank, -1e30)
        recomb_score = recomb_score.scatter_(-1, max_score_idx, max_score) # fill max in the right place
        recomb_score = torch.where( # recombine only if hypothesis not end with blank
            target_not_blank_not_eos.unsqueeze(-1).expand(-1, -1, 2),
            recomb_score,
            seq_log_prob_lastlabel_blank
            )
        # print("recomb score", recomb_score)
        seq_log_prob.raw_tensor = seq_log_prob_raw.scatter_(-1, target_and_blank.long(), recomb_score)
        # print("score after recomb", seq_log_prob.raw_tensor.gather(-1, target_and_blank.long()))
        # # -------------------------- end recombination ---------------------------

        # keep this to properly expand hypotheses            
        target_prev = target

        # pruning
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob,
            k_dim=Dim(beam_size, name=f"dec-step{i}-beam"),
            axis=[beam_dim, model.target_dim_w_blank],
        )  # seq_log_prob, backrefs, target: Batch, Beam
        # print("seq_log_prob after top_k", seq_log_prob.raw_tensor)
        # print("backrefs", backrefs)
        # print("target", target.raw_tensor)
        batch_dims_ = batch_dims + [beam_dim]
        seq_targets.append(target)
        seq_backrefs.append(backrefs)

        # Store each hypothesis, collapsed and blank removed
        hyps_label_idxs_prev = copy.deepcopy(hyps_label_idxs)
        for batch in range(batch_size):
            for beam in range(beam_size):
                beam_target = target.raw_tensor[batch, beam]
                backref = backrefs.raw_tensor[batch, beam]
                prev_beam_last_target = target_prev.raw_tensor[batch][backref]
                if (beam_target != model.blank_idx) and \
                    (beam_target != prev_beam_last_target):
                    # (prev_beam_last_target == model.blank_idx or \
                    # beam_target != prev_beam_last_target):
                    hyps_label_idxs[batch][beam] = hyps_label_idxs_prev[batch][backref] + (beam_target.item(),)
                else:
                    hyps_label_idxs[batch][beam] = hyps_label_idxs_prev[batch][backref]
        
        # Now, check where recombination can happen
        # It can only happen between 2 hyps h1, h2 where h1 == h2[:-1]
        # convention: If (h1, h2) is possible pair then h1 == h2[:-1]
        recomb_possible_pairs = [set() for _ in range(batch_size)]
        for batch in range(batch_size):
            for beam1 in range(beam_size):
                for beam2 in range(beam_size):
                    hyp1, hyp2 = hyps_label_idxs[batch][beam1], hyps_label_idxs[batch][beam2]
                    if len(hyp1) + 1 == len(hyp2) and hyp1 == hyp2[:-1]: # short circuit, faster?
                        recomb_possible_pairs[batch].add((beam1, beam2))

        # # ----- debug print, check if two hypotheses are the same --------
        # # there should not be any such pairs
        # # print("hyps label set", hyps_label_idxs)
        # # print("recomb posible pairs", recomb_possible_pairs)
        # for batch in range(batch_size):
        #     for b1 in range(len(hyps_label_idxs[batch])):
        #         for b2 in range(b1+1, len(hyps_label_idxs[batch])):
        #             if hyps_label_idxs[batch][b1] == hyps_label_idxs[batch][b2]:
        #                 print(f"Hyps {b1} and {b2} of batch {batch} are equal!")


        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)

        ended = rf.logical_or(ended, target == model.eos_idx)  # TODO: keep this or not?
        ended = rf.logical_or(ended, rf.copy_to_device(i + 1 >= max_seq_len))
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
        # print("End of loop step", i, "\n")

    print("Time forward loop:", time.time() - t)

    if i > 0 and length_normalization_exponent != 0:
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

    hyps_raw = seq_targets.copy_transpose(
        batch_dims + [beam_dim] + [out_spatial_dim]
    ).raw_tensor

    seq_targets, out_spatial_dim = remove_blank_and_eos(
        hyps_raw,
        orig_max_seq_len_raw[0],
        batch_dims,
        beam_dim,
        model.target_dim,
        model.blank_idx,
        model.eos_idx,
    )
    # print(seq_targets.raw_tensor)
    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


def _get_init_trafo_state(model, batch_dims: Sequence[Dim]):
    trafo_state = model.default_initial_state(batch_dims=batch_dims)
    for state in trafo_state:
        if state == "pos":
            trafo_state[state] = rf.zeros(batch_dims, dtype="int32")
        else:
            self_att_expand_dim = Dim(
                rf.zeros(batch_dims, dtype="int32"), name="self_att_expand_dim_init"
            )
            trafo_state[state].self_att.accum_axis = self_att_expand_dim

            k_accum = trafo_state[state].self_att.k_accum  # type: rf.Tensor
            k_accum_raw = k_accum.raw_tensor
            trafo_state[state].self_att.k_accum = k_accum.copy_template_replace_dim_tag(
                k_accum.get_axis_from_description("stag:self_att_expand_dim_init"),
                self_att_expand_dim,
            )
            trafo_state[state].self_att.k_accum.raw_tensor = k_accum_raw

            v_accum = trafo_state[state].self_att.v_accum  # type: rf.Tensor
            v_accum_raw = v_accum.raw_tensor
            trafo_state[state].self_att.v_accum = v_accum.copy_template_replace_dim_tag(
                v_accum.get_axis_from_description("stag:self_att_expand_dim_init"),
                self_att_expand_dim,
            )
            trafo_state[state].self_att.v_accum.raw_tensor = v_accum_raw

    return trafo_state


def _get_masked_trafo_state(
    trafo_state,
    trafo_state_updated,
    update_state_mask: Tensor,
    backrefs: Tensor,
):
    for state in trafo_state:
        if state == "pos":
            trafo_state[state] = rf.where(
                update_state_mask,
                rf.gather(trafo_state_updated[state], indices=backrefs),
                rf.gather(trafo_state[state], indices=backrefs),
            )
        else:
            updated_accum_axis = trafo_state_updated[state].self_att.accum_axis

            updated_self_att_expand_dim_dyn_size_ext = rf.gather(
                updated_accum_axis.dyn_size_ext, indices=backrefs
            )
            masked_self_att_expand_dim_dyn_size_ext = rf.where(
                update_state_mask,
                updated_self_att_expand_dim_dyn_size_ext,
                updated_self_att_expand_dim_dyn_size_ext - 1,
            )
            masked_self_att_expand_dim = Dim(
                masked_self_att_expand_dim_dyn_size_ext, name="self_att_expand_dim_init"
            )
            trafo_state[state].self_att.accum_axis = masked_self_att_expand_dim

            def _mask_lm_state(tensor: rf.Tensor):
                tensor = rf.gather(tensor, indices=backrefs)  # gather after top_k
                tensor = tensor.copy_transpose(
                    [updated_accum_axis] + tensor.remaining_dims(updated_accum_axis)
                )
                tensor_raw = tensor.raw_tensor
                tensor_raw = tensor_raw[
                    : rf.reduce_max(
                        masked_self_att_expand_dim_dyn_size_ext,
                        axis=masked_self_att_expand_dim_dyn_size_ext.dims,
                    ).raw_tensor.item()
                ]
                tensor = tensor.copy_template_replace_dim_tag(
                    tensor.get_axis_from_description(updated_accum_axis),
                    masked_self_att_expand_dim,
                )
                tensor.raw_tensor = tensor_raw
                return tensor

            trafo_state[state].self_att.k_accum = _mask_lm_state(
                trafo_state_updated[state].self_att.k_accum
            )
            trafo_state[state].self_att.v_accum = _mask_lm_state(
                trafo_state_updated[state].self_att.v_accum
            )

    return trafo_state


def _get_trafo_state_after(
    trafo_state,
    backrefs: Tensor,
):
    for state in trafo_state:
        if state == "pos":
            trafo_state[state] = rf.gather(trafo_state[state], indices=backrefs)
        else:
            updated_accum_axis = trafo_state[state].self_att.accum_axis

            updated_self_att_expand_dim_dyn_size_ext = rf.gather(
                updated_accum_axis.dyn_size_ext, indices=backrefs
            )
            masked_self_att_expand_dim = Dim(
                updated_self_att_expand_dim_dyn_size_ext,
                name="self_att_expand_dim_init",
            )
            trafo_state[state].self_att.accum_axis = masked_self_att_expand_dim

            def _mask_lm_state(tensor: rf.Tensor):
                tensor = rf.gather(tensor, indices=backrefs)  # gather after top_k
                tensor_raw = tensor.raw_tensor
                tensor = tensor.copy_template_replace_dim_tag(
                    tensor.get_axis_from_description(updated_accum_axis),
                    masked_self_att_expand_dim,
                )
                tensor.raw_tensor = tensor_raw
                return tensor

            trafo_state[state].self_att.k_accum = _mask_lm_state(
                trafo_state[state].self_att.k_accum
            )
            trafo_state[state].self_att.v_accum = _mask_lm_state(
                trafo_state[state].self_att.v_accum
            )

    return trafo_state


# RecogDef API
model_recog_time_sync_recomb_first_v2: RecogDef[Model]
model_recog_time_sync_recomb_first_v2.output_with_beam = True
model_recog_time_sync_recomb_first_v2.output_blank_label = None
model_recog_time_sync_recomb_first_v2.batch_size_dependent = False
