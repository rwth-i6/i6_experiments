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
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.search_functions import (
    remove_blank_and_eos,
)

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.blank_collapse import (
    blank_collapse_batched,
)

import torch
import numpy

import time

_ctc_prior_filename = "/u/luca.gaudino/debug/ctc/prior.txt"
# _ctc_prior_filename = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZdcvhAOyWl95/output/prior.txt"


def model_recog_time_sync(
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
    if search_args.get("encoder_ctc", False):
        enc_args_ctc, enc_spatial_dim_ctc = model.encode_ctc(
            data, in_spatial_dim=data_spatial_dim
        )

    beam_size = search_args.get("beam_size", 12)
    length_normalization_exponent = search_args.get(
        "length_normalization_exponent", 0.0
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
    prev_decoder_state = decoder_state

    if search_args.get("add_trafo_lm", False):
        trafo_lm_state = _get_init_trafo_state(model.language_model, batch_dims_)

    if search_args.get("ilm_scale", 0.0) > 0:
        ilm_state = model.ilm.default_initial_state(batch_dims=batch_dims_)
        prev_ilm_state = ilm_state

    initial_target = rf.constant(
        model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim_w_b
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

    if search_args.get("encoder_ctc", False):
        enc_ctc = enc_args_ctc["ctc"]
    else:
        enc_ctc = enc_args["ctc"]

    ctc_out_raw = enc_ctc.copy_transpose(
        (batch_size_dim, enc_spatial_dim, model.target_dim_w_b)
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
        ctc_log_prior = numpy.loadtxt(
            search_args.get("ctc_prior_file", None), dtype="float32"
        )

        ctc_log_prior = (
            torch.tensor(ctc_log_prior)
            .repeat(ctc_out_raw.shape[0], ctc_out_raw.shape[1], 1)
            .to("cuda")
        )

        # better do this in log space
        ctc_out_log_raw = torch.log(ctc_out_raw)

        ctc_out_log_raw = ctc_out_log_raw - (
            ctc_log_prior * search_args.get("prior_scale", 0.3)
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
    #     dims=(batch_size_dim, ctc_spatial_dim, model.target_dim_w_b),
    #     dtype="float32",
    #     raw_tensor=ctc_out_raw,
    # )

    ctc_out_log_raw = torch.log(ctc_out_raw)

    enc_args.pop("ctc")

    prev_target = initial_target
    prev_target_non_blank = initial_target

    eps = 1e-20
    i = 0
    seq_targets = []
    seq_backrefs = []
    out_str = [[""] * beam_size for _ in range(batch_size)]

    if search_args.get("add_eos_to_end", False):
        max_seq_len = max_seq_len + 1  # add one step to loop

    t = time.time()

    for i in range(torch.max(max_seq_len.raw_tensor)):
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

            decoder_state = tree.map_structure(
                lambda s: rf.gather(s, indices=backrefs), decoder_state
            )
            prev_decoder_state = tree.map_structure(
                lambda s: rf.gather(s, indices=backrefs),
                decoder_state_1,
            )

            if search_args.get("ilm_scale", 0.0) > 0:  # TODO
                ilm_state = model.ilm.select_state(ilm_state, backrefs)
                prev_ilm_state = model.ilm.select_state(ilm_state_1, backrefs)

        mask_not_blank = rf.compare(target, "not_equal", model.blank_idx)
        mask_not_repeat = rf.compare(target, "not_equal", prev_target)
        mask_combined = rf.logical_and(mask_not_blank, mask_not_repeat)

        if not torch.any(mask_combined.raw_tensor) or i == 0:
            decoder_state_1 = prev_decoder_state
            target_1 = prev_target_non_blank
        else:
            partial_mask_function = partial(rf.where, mask_combined)
            decoder_state_1 = tree.map_structure(
                lambda s, prev_s: partial_mask_function(s, prev_s),
                decoder_state,
                prev_decoder_state,
            )
            target_1 = rf.where(mask_combined, target, prev_target_non_blank)

        if search_args.get("ilm_scale", 0.0) > 0.0:
            if not torch.any(mask_combined.raw_tensor) or i == 0:
                ilm_state_1 = prev_ilm_state
            else:
                ilm_state_1 = tree.map_structure(
                    lambda s, prev_s: partial_mask_function(s, prev_s),
                    ilm_state,
                    prev_ilm_state,
                )

        if search_args.get("add_trafo_lm", False):
            if i > 0:
                trafo_lm_state_1 = _get_masked_trafo_state(
                    trafo_lm_state_1, trafo_lm_state, mask_combined, backrefs
                )
            else:
                trafo_lm_state_1 = trafo_lm_state

        # remove blank from target
        target_1.sparse_dim = model.target_dim

        # step computations
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

        att_label_log_prob = att_label_log_prob * search_args.get("att_scale", 1.0)
        att_label_log_prob_raw = att_label_log_prob.raw_tensor

        if search_args.get("add_eos_to_end", False) and is_last_step:
            eos_log_prob = rf.Tensor(
                name="eos_log_prob",
                dtype="float32",
                dims=att_label_log_prob.dims[:-1],
                raw_tensor=att_label_log_prob_raw[:, :, model.eos_idx],
            )

        if search_args.get("remove_att_eos", False):
            # warning this basically set eos to 0 for the whole prob distribution
            att_label_log_prob_raw[:, :, model.eos_idx] = -1e30
            # att_label_log_prob_raw = att_label_log_prob_raw - torch.logsumexp(
            #     att_label_log_prob_raw, dim=2, keepdim=True
            # )

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
            ctc_non_blank * search_args.get("ctc_scale", 0.0)
            + att_label_log_prob_raw
        )

        if search_args.get("add_trafo_lm", False):
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

        if search_args.get("ilm_scale", 0.0) > 0:
            ilm_out = model.ilm(input_embed, state=ilm_state_1, spatial_dim=single_step_dim)
            ilm_state = ilm_out["state"]
            ilm_log_prob = rf.log_softmax(ilm_out["output"], axis=model.target_dim)

            if search_args.get("add_eos_to_end", False) and is_last_step:
                eos_log_prob.raw_tensor = eos_log_prob.raw_tensor - ilm_log_prob.raw_tensor[
                    :, :, model.eos_idx
                ] * search_args.get("lm_scale", 0.0)

            label_log_prob_non_blank = (
                label_log_prob_non_blank - search_args.get("lm_scale", 0.0) * ilm_log_prob.raw_tensor
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

        if search_args.get("add_eos_to_end", False) and is_last_step:
            seq_log_prob_w_eos = seq_log_prob + eos_log_prob

            seq_log_prob = rf.where(
                ended,
                seq_log_prob,
                seq_log_prob_w_eos,
            ) # dont add it twice
            break

        seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab

        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob,
            k_dim=Dim(beam_size, name=f"dec-step{i}-beam"),
            axis=[beam_dim, model.target_dim_w_b],
        )  # seq_log_prob, backrefs, target: Batch, Beam
        batch_dims_ = batch_dims + [beam_dim]
        seq_targets.append(target)
        seq_backrefs.append(backrefs)

        # recombination
        if i == 0:
            prev_target = rf.gather(initial_target, indices=backrefs)
        elif i > 0:
            prev_target = rf.gather(seq_targets[i - 1], indices=backrefs)

        mask_not_blank_rec = rf.compare(target, "not_equal", model.blank_idx)
        mask_not_repeat_rec = rf.compare(target, "not_equal", prev_target)
        mask_combined_rec = rf.logical_and(mask_not_blank_rec, mask_not_repeat_rec)
        out_str_prev = copy.deepcopy(out_str)
        for batch in range(batch_size):
            for beam in range(beam_size):
                backref = backrefs.raw_tensor[batch, beam]
                if mask_combined_rec.raw_tensor[batch, beam]:
                    out_str[batch][beam] = (
                        out_str_prev[batch][backref]
                        + ",,"
                        + model.target_dim.vocab.id_to_label(
                            int(target.raw_tensor[batch, beam])
                        )
                    )
                else:
                    out_str[batch][beam] = out_str_prev[batch][backref]

        for batch in range(batch_size):
            hyp_set = {}
            for beam in range(beam_size):
                if hyp_set.get(out_str[batch][beam], None) is not None:
                    hyp_set[out_str[batch][beam]].append(beam)
                else:
                    hyp_set[out_str[batch][beam]] = [beam]

            for hyp in hyp_set.keys():
                if len(hyp_set[hyp]) > 1:
                    max_idx = hyp_set[hyp][0]
                    max_prob = seq_log_prob.raw_tensor[batch, max_idx]
                    for idx in hyp_set[hyp][1:]:
                        if seq_log_prob.raw_tensor[batch, idx] > max_prob:
                            max_idx = idx
                            max_prob = seq_log_prob.raw_tensor[batch, idx]

                    for idx in hyp_set[hyp]:
                        if idx != max_idx:
                            seq_log_prob.raw_tensor[batch, idx] = -1e30

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

# ------------------- old code ---------------------------

# debug_out_seq = [[[0]] * beam_size for _ in range(batch_size)]

# debug_out_seq_prev = copy.deepcopy(debug_out_seq)
# for batch in range(batch_size):
#     for beam in range(beam_size):
#         backref = backrefs.raw_tensor[batch, beam]
#         debug_out_seq[batch][beam] = debug_out_seq_prev[batch][backref] + [
#             int(target.raw_tensor[batch, beam])
#         ]

# prev_trafo_lm_state = tree.map_structure(lambda v: v.copy(), trafo_lm_state_1)
# prev_trafo_lm_state_updated = tree.map_structure(
#     lambda v: v.copy(), trafo_lm_state
# )


def _old_handl_trafo_lm_state():
    # handle trafo lm state
    if search_args.get("add_trafo_lm", False) and i > 0:
        prev_pos_raw = prev_trafo_lm_state_all["pos"].raw_tensor
        prev_trafo_lm_state_all.pop("pos")
        pos_raw = trafo_lm_state["pos"].raw_tensor
        trafo_lm_state.pop("pos")
        if i == 1:
            # expand pos to beam size
            pos_raw = pos_raw.repeat((1, beam_size))
            prev_pos_raw = prev_pos_raw.repeat((1, beam_size))
        pos_change_dim = rf.Tensor(
            name="pos",
            dims=batch_dims_,
            dtype="int32",
            raw_tensor=pos_raw,
        )
        prev_pos_change_dim = rf.Tensor(
            name="pos",
            dims=batch_dims_,
            dtype="int32",
            raw_tensor=prev_pos_raw,
        )

        # prepare prev_trafo_lm_state
        if (
            prev_trafo_lm_state_all["0"]["self_att"]["accum_axis"].dimension != 0
        ):  # check for initial state
            prev_trafo_lm_state = tree.map_structure(
                partial(trafo_lm_state_func, seq_backrefs[i - 1]),
                prev_trafo_lm_state_all,
            )
        else:
            prev_trafo_lm_state = prev_trafo_lm_state_all
            # change beam dim of k_accum and v_accum
            for lay in range(model.language_model.num_layers):
                lay = str(lay)
                k_accum_temp = prev_trafo_lm_state[lay]["self_att"][
                    "k_accum"
                ].copy_template_replace_dim_tag(1, beam_dim)
                k_accum_temp.raw_tensor = prev_trafo_lm_state[lay]["self_att"][
                    "k_accum"
                ].raw_tensor.repeat((1, beam_size, 1, 1, 1))
                prev_trafo_lm_state[lay]["self_att"]["k_accum"] = k_accum_temp
                v_accum_temp = prev_trafo_lm_state[lay]["self_att"][
                    "v_accum"
                ].copy_template_replace_dim_tag(1, beam_dim)
                v_accum_temp.raw_tensor = prev_trafo_lm_state[lay]["self_att"][
                    "v_accum"
                ].raw_tensor.repeat((1, beam_size, 1, 1, 1))
                prev_trafo_lm_state[lay]["self_att"]["v_accum"] = v_accum_temp

        # shift hist of prev_trafo_lm_state if needed
        if torch.all(mask_combined.raw_tensor):
            # if all are not blank or repeat copy curr state
            trafo_lm_state["pos"] = pos_change_dim
            trafo_lm_state_1 = trafo_lm_state
        elif torch.any(mask_combined.raw_tensor):
            # shift
            for lay in range(model.language_model.num_layers):
                lay = str(lay)
                old_accum_axis = prev_trafo_lm_state[lay]["self_att"]["accum_axis"]
                new_accum_axis = trafo_lm_state[lay]["self_att"]["accum_axis"]
                k_accum = prev_trafo_lm_state[lay]["self_att"]["k_accum"]

                fill_value = rf.full(
                    dims=batch_dims_
                    # + [Dim(1, name="accum_step_dim")]
                    + list(k_accum.dims[-2:]),
                    fill_value=1e-30,
                )

                k_accum_shifted, hist_dim = rf.cum_concat_step(
                    fill_value, prev_accum=k_accum, axis=old_accum_axis
                )
                v_accum_shifted, _ = rf.cum_concat_step(
                    fill_value,
                    prev_accum=prev_trafo_lm_state[lay]["self_att"]["v_accum"],
                    out_spatial_dim=hist_dim,
                    axis=old_accum_axis,
                )
                #
                # if prev_trafo_lm_state_all["0"]["self_att"]["accum_axis"].dimension == 0:
                #     k_accum_shifted_raw = k_accum_shifted.raw_tensor.repeat(
                #         (1, beam_size, 1, 1, 1)
                #     )
                #     k_accum_shifted = rf.Tensor(
                #         name="k_accum_shifted",
                #         dims=(batch_size_dim, beam_dim) + k_accum_shifted.dims[2:],
                #         dtype="float32",
                #         raw_tensor=k_accum_shifted_raw,
                #     )
                #     v_accum_shifted_raw = v_accum_shifted.raw_tensor.repeat(
                #         (1, beam_size, 1, 1, 1)
                #     )
                #     v_accum_shifted = rf.Tensor(
                #         name="v_accum_shifted",
                #         dims=(batch_size_dim, beam_dim) + v_accum_shifted.dims[2:],
                #         dtype="float32",
                #         raw_tensor=v_accum_shifted_raw,
                #     )

                prev_trafo_lm_state[lay]["self_att"]["k_accum"] = k_accum_shifted
                prev_trafo_lm_state[lay]["self_att"]["v_accum"] = v_accum_shifted
                prev_trafo_lm_state[lay]["self_att"]["accum_axis"] = new_accum_axis

            # mask state
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

            trafo_lm_state_1["pos"] = rf.where(
                rf.copy_to_device(mask_combined, "cpu"),
                rf.copy_to_device(pos_change_dim, "cpu"),
                rf.copy_to_device(prev_pos_change_dim, "cpu"),
            )
        else:
            # if all are blank or repeat copy prev state
            prev_trafo_lm_state["pos"] = prev_pos_change_dim
            trafo_lm_state_1 = prev_trafo_lm_state
        prev_trafo_lm_state_all = trafo_lm_state_1
    elif search_args.get("add_trafo_lm", False) and i == 0:
        trafo_lm_state_1 = trafo_lm_state
        prev_trafo_lm_state_all = trafo_lm_state_1


# RecogDef API
model_recog_time_sync: RecogDef[Model]
model_recog_time_sync.output_with_beam = True
model_recog_time_sync.output_blank_label = "<blank>"
model_recog_time_sync.batch_size_dependent = False
