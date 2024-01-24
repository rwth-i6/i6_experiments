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

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.blank_collapse import (
    blank_collapse_batched,
)

import torch
import numpy

_ctc_prior_filename = "/u/luca.gaudino/debug/ctc/prior.txt"
# _ctc_prior_filename = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZdcvhAOyWl95/output/prior.txt"


def model_recog_time_sync(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
    max_seq_len: Optional[int] = None,
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
    beam_size = model.search_args.get("beam_size", 12)
    length_normalization_exponent = model.search_args.get(
        "length_normalization_exponent", 1.0
    )
    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
    print("** max seq len:", max_seq_len.raw_tensor)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = batch_dims + [beam_dim]
    decoder_state = model.decoder_default_initial_state(
        batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim
    )

    if model.search_args.get("add_trafo_lm", False):
        trafo_lm_state = model.trafo_lm.default_initial_state(batch_dims=batch_dims_)
        prev_trafo_lm_state = trafo_lm_state
        prev_trafo_lm_state_all = prev_trafo_lm_state

    initial_target = rf.constant(
        model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim_w_b
    )
    target = initial_target
    ended = rf.constant(False, dims=batch_dims_)
    out_seq_len = rf.constant(0, dims=batch_dims_)
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    assert len(batch_dims) == 1
    batch_size_dim = batch_dims[0]
    batch_size = batch_dims[0].get_dim_value()
    target_ctc = [model.bos_idx for _ in range(batch_size * beam_size)]

    blank_index = model.target_dim.get_dim_value()

    # already in log space
    ctc_out_raw = (
        enc_args["ctc"]
        .copy_transpose((batch_size_dim, enc_spatial_dim, model.target_dim_w_b))
        .raw_tensor
    )  # [B,T,V+1]

    if model.search_args.get("prior_corr", False):
        ctc_log_prior = numpy.loadtxt(
            model.search_args.get("ctc_prior_file", None), dtype="float32"
        )
        ctc_out_raw = ctc_out_raw - (
            torch.tensor(ctc_log_prior)
            .repeat(ctc_out_raw.shape[0], ctc_out_raw.shape[1], 1)
            .to("cuda")
            * model.search_args.get("prior_scale", 0.3)
        )
        ctc_out_raw = ctc_out_raw - torch.logsumexp(ctc_out_raw, dim=2, keepdim=True)

    if model.search_args.get("mask_eos", True):
        ctc_eos = ctc_out_raw[:, :, model.eos_idx].unsqueeze(2)
        ctc_blank = ctc_out_raw[:, :, model.blank_idx].unsqueeze(2)
        ctc_out_raw[:, :, model.blank_idx] = torch.logsumexp(
            torch.cat([ctc_eos, ctc_blank], dim=2), dim=2
        )
        ctc_out_raw[:, :, model.eos_idx] = -1e30

    hlens = max_seq_len.raw_tensor

    ctc_spatial_dim = enc_spatial_dim
    if model.search_args.get("blank_collapse", False):
        col_probs, col_lens = blank_collapse_batched(
            ctc_out_raw.to("cpu"),
            hlens,
            model.search_args.get("blank_threshold", 0.0),
            blank_index,
        )
        ctc_spatial_dim = enc_spatial_dim.copy(description="ctc-spatial")
        hlens = col_lens.to(torch.int32)
        ctc_spatial_dim.dyn_size_ext.raw_tensor = hlens
        ctc_out_raw = col_probs.to("cuda")

    ctc_out = rf.Tensor(
        name="ctc_out",
        dims=(batch_size_dim, ctc_spatial_dim, model.target_dim_w_b),
        dtype="float32",
        raw_tensor=ctc_out_raw,
    )

    # if model.search_args["use_ctc"]:
    #     # ctc prefix scorer espnet
    #     from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.espnet_ctc.ctc_prefix_score_espnet import (
    #         CTCPrefixScoreTH,
    #     )
    #
    #     # hlens = max_seq_len.raw_tensor.repeat(beam_size).view(beam_size, data.raw_tensor.shape[0]).transpose(0, 1)
    #
    #
    #
    #     ctc_prefix_scorer = CTCPrefixScoreTH(
    #         ctc_out,
    #         hlens,
    #         10025,
    #         0,
    #         model.search_args["window_margin"],
    #         model.search_args["mask_eos"],
    #     )
    #     ctc_state = None
    enc_args.pop("ctc")

    prev_decoder_state = decoder_state
    prev_target = initial_target
    prev_target_non_blank = initial_target

    eps = 1e-30
    i = 0
    seq_targets = []
    seq_backrefs = []

    def trafo_lm_state_func(backrefs, s):
        if type(s) == Dim:
            return s
        else:
            return rf.gather(s, indices=backrefs)

    for i in range(torch.max(hlens)):
        # gather prev non-blank targets and prev_decoder_state via backrefs
        if i == 1:
            prev_target = rf.gather(initial_target, indices=seq_backrefs[i - 1])
        elif i > 1:
            prev_target = rf.gather(seq_targets[i - 2], indices=seq_backrefs[i - 1])

        if i > 0:
            prev_decoder_state = tree.map_structure(
                lambda s: rf.gather(s, indices=seq_backrefs[i - 1]),
                prev_decoder_state_all,
            )
            mask_combined_gather = rf.gather(mask_combined, indices=seq_backrefs[i - 1])
            prev_target_non_blank_gather = rf.gather(
                prev_target_non_blank, indices=seq_backrefs[i - 1]
            )
            prev_target_non_blank = rf.where(
                mask_combined_gather, prev_target, prev_target_non_blank_gather
            )

        mask_not_blank = rf.compare(target, "not_equal", model.blank_idx)
        mask_not_repeat = rf.compare(target, "not_equal", prev_target)
        mask_combined = rf.logical_and(mask_not_blank, mask_not_repeat)
        partial_mask_function = partial(rf.where, mask_combined)
        decoder_state_1 = tree.map_structure(
            lambda s, prev_s: partial_mask_function(s, prev_s),
            decoder_state,
            prev_decoder_state,
        )
        target_1 = rf.where(mask_combined, target, prev_target_non_blank)

        # remove blank from target
        target_1.sparse_dim = model.target_dim

        # set for next iteration
        prev_decoder_state_all = decoder_state_1

        # handle trafo lm state
        if model.search_args.get("add_trafo_lm", False) and i > 0:
            prev_pos = prev_trafo_lm_state_all["pos"]
            prev_trafo_lm_state_all.pop("pos")

            if i > 1:
                prev_trafo_lm_state = tree.map_structure(
                    partial(trafo_lm_state_func, seq_backrefs[i - 1]),
                    prev_trafo_lm_state_all,
                )
            else:
                prev_trafo_lm_state = prev_trafo_lm_state_all

            # shift hist of prev_trafo_lm_state
            if torch.all(mask_combined.raw_tensor):
                # if all are not blank or repeat copy curr state
                pos = trafo_lm_state["pos"]
                trafo_lm_state.pop("pos")
                if pos.batch_ndim == 0:
                    pos = rf.full(
                        dims=mask_combined.dims,
                        fill_value=prev_pos,
                    )
                else:
                    pos_temp = pos.copy_template_replace_dim_tag(
                        1, mask_combined.dims[1]
                    )
                    pos_temp.raw_tensor = pos.raw_tensor
                    pos = pos_temp
                trafo_lm_state["pos"] = pos
                trafo_lm_state_1 = trafo_lm_state
            elif torch.any(mask_combined.raw_tensor):
                for lay in range(model.trafo_lm.num_layers):
                    lay = str(lay)
                    old_accum_axis = prev_trafo_lm_state[lay]["self_att"]["accum_axis"]
                    new_accum_axis = trafo_lm_state[lay]["self_att"]["accum_axis"]
                    k_accum = prev_trafo_lm_state[lay]["self_att"]["k_accum"]

                    fill_value = rf.full(
                        dims=list(k_accum.dims[:2])
                        # + [Dim(1, name="accum_step_dim")]
                        + list(k_accum.dims[-2:]),
                        fill_value=1e-30,
                    )

                    k_accum_shifted, hist_dim = rf.cum_concat_step(
                        fill_value, prev_accum=k_accum, axis=old_accum_axis
                    )
                    v_accum_shifted, _ = rf.cum_concat_step(
                        fill_value,
                        prev_accum=prev_trafo_lm_state[str(lay)]["self_att"]["v_accum"],
                        out_spatial_dim=hist_dim,
                        axis=old_accum_axis,
                    )

                    if i == 1:
                        k_accum_shifted_raw = k_accum_shifted.raw_tensor.repeat(
                            (1, beam_size, 1, 1, 1)
                        )
                        k_accum_shifted = rf.Tensor(
                            name="k_accum_shifted",
                            dims=(batch_size_dim, beam_dim) + k_accum_shifted.dims[2:],
                            dtype="float32",
                            raw_tensor=k_accum_shifted_raw,
                        )
                        v_accum_shifted_raw = v_accum_shifted.raw_tensor.repeat(
                            (1, beam_size, 1, 1, 1)
                        )
                        v_accum_shifted = rf.Tensor(
                            name="v_accum_shifted",
                            dims=(batch_size_dim, beam_dim) + v_accum_shifted.dims[2:],
                            dtype="float32",
                            raw_tensor=v_accum_shifted_raw,
                        )

                    prev_trafo_lm_state[lay]["self_att"]["k_accum"] = k_accum_shifted
                    prev_trafo_lm_state[lay]["self_att"]["v_accum"] = v_accum_shifted
                    prev_trafo_lm_state[lay]["self_att"]["accum_axis"] = new_accum_axis

                # mask state
                pos = trafo_lm_state["pos"]
                trafo_lm_state.pop("pos")

                def trafo_lm_state_mask_func(s, prev_s):
                    # if i > 0:
                    #     breakpoint()
                    if type(s) == Dim:
                        return s
                    return rf.where(mask_combined, s, prev_s)

                trafo_lm_state_1 = tree.map_structure(
                    trafo_lm_state_mask_func,
                    trafo_lm_state,
                    prev_trafo_lm_state,
                )
                if i == 1:
                    trafo_lm_state_1["pos"] = rf.where(
                        rf.copy_to_device(mask_combined, "cpu"),
                        rf.copy_to_device(pos, "cpu"),
                        rf.copy_to_device(prev_pos, "cpu"),
                    )
                elif i > 1:
                    pos_temp = pos.copy_template_replace_dim_tag(
                        1, mask_combined.dims[1]
                    )
                    pos_temp.raw_tensor = pos.raw_tensor
                    pos = pos_temp

                    prev_pos_temp = prev_pos.copy_template_replace_dim_tag(
                        1, mask_combined.dims[1]
                    )
                    prev_pos_temp.raw_tensor = prev_pos.raw_tensor
                    prev_pos = prev_pos_temp

                    trafo_lm_state_1["pos"] = rf.where(
                        rf.copy_to_device(mask_combined, "cpu"),
                        rf.copy_to_device(pos, "cpu"),
                        rf.copy_to_device(prev_pos, "cpu"),
                    )
                else:
                    trafo_lm_state_1["pos"] = pos
            else:
                # if all are blank or repeat copy prev state
                if prev_pos.batch_ndim == 0:
                    prev_pos = rf.full(
                        dims=mask_combined.dims,
                        fill_value=prev_pos,
                    )
                else:
                    prev_pos_temp = prev_pos.copy_template_replace_dim_tag(
                        1, prev_trafo_lm_state["0"]["self_att"]["k_accum"].dims[1]
                    )
                    prev_pos_temp.raw_tensor = prev_pos.raw_tensor
                    prev_pos = prev_pos_temp
                prev_trafo_lm_state["pos"] = prev_pos
                trafo_lm_state_1 = prev_trafo_lm_state
            prev_trafo_lm_state_all = trafo_lm_state_1
        elif model.search_args.get("add_trafo_lm", False) and i == 0:
            trafo_lm_state_1 = trafo_lm_state
            prev_trafo_lm_state_all = trafo_lm_state_1

        # fixed: before it was computed at step 0
        if i == 0:
            input_embed = rf.zeros(
                batch_dims_ + [model.target_embed.out_dim],
                feature_dim=model.target_embed.out_dim,
            )
        else:
            input_embed = model.target_embed(target_1)

        step_out, decoder_state = model.loop_step(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=input_embed,
            state=decoder_state_1,
        )
        step_out.pop("att_weights", None)
        logits = model.decode_logits(input_embed=input_embed, **step_out)
        att_label_log_prob = rf.log_softmax(logits, axis=model.target_dim)

        att_label_log_prob = att_label_log_prob * model.search_args.get(
            "att_scale", 1.0
        )

        # continue in pure pytorch because slicing is easier
        # rf.gather(ctc_out, indices=i, axis=enc_spatial_dim) does not work

        # add beam dim to ctc_out_raw and get step i
        ctc_out_raw_step = ctc_out_raw.unsqueeze(1).repeat(
            [1, beam_dim.get_dim_value(), 1, 1]
        )[
            :, :, i
        ]  # [B, beam, T, V+1]

        # renormalize ctc_out_raw_step
        ctc_non_blank = ctc_out_raw_step[:, :, :blank_index]
        ctc_non_blank = ctc_non_blank - torch.logsumexp(
            ctc_non_blank, dim=2, keepdim=True
        )

        label_log_prob_non_blank = (
            ctc_non_blank * model.search_args.get("ctc_scale", 0.0)
            + att_label_log_prob.raw_tensor
        )

        if model.search_args.get("add_trafo_lm", False):
            # breakpoint()
            trafo_lm_out = model.trafo_lm(
                target_1, state=trafo_lm_state_1, spatial_dim=single_step_dim
            )
            trafo_lm_state = trafo_lm_out["state"]

            trafo_log_prob = rf.log_softmax(
                trafo_lm_out["output"], axis=model.target_dim
            )
            if i > 0:
                trafo_log_prob_raw = trafo_log_prob.raw_tensor
                if model.search_args.get("remove_trafo_lm_eos", False):
                    # warning this basically set eos to 0 for the whole prob distribution
                    trafo_log_prob_raw[:, :, model.eos_idx] = -1e30
                    trafo_log_prob_raw = trafo_log_prob_raw - torch.logsumexp(
                        trafo_log_prob_raw, dim=2, keepdim=True
                    )
                label_log_prob_non_blank = (
                    label_log_prob_non_blank
                    + model.search_args["lm_scale"] * trafo_log_prob_raw
                )

        blank_log_prob = ctc_out_raw_step[:, :, blank_index]

        one_minus_term = torch.ones_like(blank_log_prob) - torch.exp(blank_log_prob)

        repeat_prob = torch.gather(
            ctc_out_raw_step, 2, target.raw_tensor.to(torch.int64).unsqueeze(2)
        ).squeeze(2)
        one_minus_term = torch.where(
            mask_not_blank.raw_tensor,
            one_minus_term - torch.exp(repeat_prob),
            one_minus_term,
        )

        label_log_prob_non_blank = (
            torch.log(
                torch.maximum(one_minus_term, torch.fill(one_minus_term, eps))
            ).unsqueeze(2)
            + label_log_prob_non_blank
        )

        label_log_prob = torch.cat(
            [label_log_prob_non_blank, blank_log_prob.unsqueeze(2)], dim=2
        )

        label_log_prob = label_log_prob.scatter_(
            2, target.raw_tensor.unsqueeze(2).to(torch.int64), repeat_prob.unsqueeze(2)
        )

        label_log_prob = rf.Tensor(
            name="label_log_prob",
            dims=(batch_size_dim, beam_dim, model.target_dim_w_b),
            dtype="float32",
            raw_tensor=label_log_prob,
        )

        # Filter out finished beams
        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(
                model.eos_idx,
                axis=model.target_dim_w_b,
                label_value=0.0,
                other_value=-1.0e30,
            ),
            label_log_prob,
        )
        seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab

        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob,
            k_dim=Dim(beam_size, name=f"dec-step{i}-beam"),
            axis=[beam_dim, model.target_dim_w_b],
        )  # seq_log_prob, backrefs, target: Batch, Beam
        curr_beam_size = beam_dim.get_size_tensor().raw_tensor
        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        decoder_state = tree.map_structure(
            lambda s: rf.gather(s, indices=backrefs), decoder_state
        )
        # if model.search_args.get("add_trafo_lm", False):
        #     trafo_lm_state_copy = trafo_lm_state
        #     for k in range(batch_num_seq):
        #         for j in range(curr_beam_size):
        #             trafo_lm_state[f"batch_{k}"][f"beam_{j}"] = trafo_lm_state_copy[f"batch_{k}"][f"beam_{backrefs.raw_tensor[k][j]}"]
        if model.search_args.get("add_trafo_lm", False):
            pos = trafo_lm_state["pos"]
            trafo_lm_state.pop("pos")
            trafo_lm_state = tree.map_structure(
                partial(trafo_lm_state_func, backrefs), trafo_lm_state
            )
            trafo_lm_state["pos"] = pos

        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)

        # ended = rf.logical_or(ended, target == model.eos_idx)
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
        max_seq_len.raw_tensor[0],
        batch_dims,
        beam_dim,
        model.target_dim,
        model.blank_idx,
        model.eos_idx,
    )

    if model.search_args.get("rescore_w_ctc", False):
        from .two_pass import rescore_w_ctc

        seq_targets, seq_log_prob = rescore_w_ctc(
            model,
            seq_targets,
            seq_log_prob,
            ctc_out,
            batch_size,
            beam_size,
            model.blank_idx,
        )

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog_time_sync: RecogDef[Model]
model_recog_time_sync.output_with_beam = True
model_recog_time_sync.output_blank_label = "<blank>"
model_recog_time_sync.batch_size_dependent = False
