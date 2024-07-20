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


def model_recog(
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

    model_aed = model.model_aed
    model_ctc = model.model_ctc

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    enc_args, enc_spatial_dim = model_aed.encode(data, in_spatial_dim=data_spatial_dim)

    enc_args_ctc, enc_spatial_dim_ctc = model_ctc.encode(data, in_spatial_dim=data_spatial_dim)
    final_ctc_layer = getattr(model_ctc, model_ctc.final_ctc_name, None)
    enc_ctc = final_ctc_layer(enc_args_ctc["enc"])
    enc_ctc = rf.log_softmax(enc_ctc, axis=model_ctc.target_dim_w_blank)

    beam_size = model_aed.search_args.get("beam_size", 12)
    length_normalization_exponent = model_aed.search_args.get("length_normalization_exponent", 1.0)
    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
    print("** max seq len:", max_seq_len.raw_tensor)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    decoder_state = model_aed.decoder_default_initial_state(
        batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim
    )

    if model.search_args.get("lm_scale", 0.0) > 0:
        lm_state = model.language_model.default_initial_state(batch_dims=batch_dims_)

    if model.search_args.get("ilm_scale", 0.0) > 0:
        ilm_state = model_aed.ilm.default_initial_state(batch_dims=batch_dims_)

    # if model.search_args.get("add_lstm_lm", False):
    #     lm_state = model.lstm_lm.lm_default_initial_state(batch_dims=batch_dims_)
    # if model.search_args.get("add_trafo_lm", False):
    #     trafo_lm_state = model.trafo_lm.default_initial_state(batch_dims=batch_dims_)

    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
    ended = rf.constant(False, dims=batch_dims_)
    out_seq_len = rf.constant(0, dims=batch_dims_)
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    assert len(batch_dims) == 1
    batch_size_dim = batch_dims[0]
    batch_size = batch_dims[0].get_dim_value()
    target_ctc = [model.bos_idx for _ in range(batch_size * beam_size)]

    blank_index = model.target_dim.get_dim_value()

    if model.search_args.get("use_ctc", False) or model.search_args.get("rescore_with_ctc", False):

        ctc_out = (
            enc_ctc
            .copy_transpose((batch_size_dim, enc_spatial_dim, model_ctc.target_dim_w_blank))
            .raw_tensor
        )  # [B,T,V+1]

    if model.search_args.get("mask_eos", True) and (model.search_args.get("use_ctc", False) or model.search_args.get("rescore_with_ctc", False)):
        ctc_eos = ctc_out[:, :, model.eos_idx].unsqueeze(2)
        ctc_blank = ctc_out[:, :, model.blank_idx].unsqueeze(2)
        ctc_out[:, :, model.blank_idx] = torch.logsumexp(
            torch.cat([ctc_eos, ctc_blank], dim=2), dim=2
        )
        ctc_out[:, :, model.eos_idx] = -1e30

    if model.search_args.get("use_ctc", False):
        # ctc prefix scorer espnet
        from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.espnet_ctc.ctc_prefix_score_espnet import (
            CTCPrefixScoreTH,
        )

        # hlens = max_seq_len.raw_tensor.repeat(beam_size).view(beam_size, data.raw_tensor.shape[0]).transpose(0, 1)
        hlens = max_seq_len.raw_tensor

        if model.search_args.get("prior_corr", False):
            ctc_log_prior = numpy.loadtxt(model.search_args.get("prior_file", model.search_args.get("ctc_prior_file", "")), dtype="float32")
            ctc_out = ctc_out - (
                torch.tensor(ctc_log_prior)
                .repeat(ctc_out.shape[0], ctc_out.shape[1], 1)
                .to(ctc_out.device)
                * model.search_args["prior_scale"]
            )
            ctc_out = ctc_out - torch.logsumexp(ctc_out, dim=2, keepdim=True)

        ctc_prefix_scorer = CTCPrefixScoreTH(
            ctc_out,
            hlens,
            blank_index,
            0,
            model.search_args.get("window_margin", 0),
            model.search_args.get("mask_eos", True),
        )
        ctc_state = None
    enc_args.pop("ctc")

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        # fixed: before it was computed at step 0
        if i == 0:
            input_embed = rf.zeros(batch_dims_ + [model_aed.target_embed.out_dim], feature_dim=model_aed.target_embed.out_dim)
        else:
            input_embed = model_aed.target_embed(target)

        step_out, decoder_state = model_aed.loop_step(
            **enc_args,
            enc_spatial_dim=enc_spatial_dim,
            input_embed=input_embed,
            state=decoder_state,
        )
        att_weights = step_out.pop("att_weights", None).raw_tensor
        if i==0:
            att_weights = torch.flatten(att_weights.squeeze(3).view(batch_size, 1, -1), end_dim=1)
        else:
            att_weights = torch.flatten(att_weights.squeeze(3).view(batch_size, beam_size, -1), end_dim=1)
        logits = model_aed.decode_logits(input_embed=input_embed, **step_out)
        label_log_prob = rf.log_softmax(
            logits, axis=model.target_dim
        )  # (Dim{'initial-beam'(1)}, Dim{B}, Dim{F'target'(10025)})

        label_log_prob = label_log_prob * model.search_args.get("att_scale", 1.0)

        if model.search_args.get("lm_scale", 0.0) > 0:
            lm_out = model.language_model(target, state=lm_state, spatial_dim=single_step_dim)
            lm_state = lm_out["state"]
            lm_log_prob = rf.log_softmax(lm_out["output"], axis=model.target_dim)

            if model.search_args.get("use_lm_first_label", True) or i > 0:
                label_log_prob = (
                    label_log_prob + model.search_args["lm_scale"] * lm_log_prob
                )

        if model.search_args.get("ilm_scale", 0.0) > 0:
            ilm_out = model_aed.ilm(input_embed, state=ilm_state, spatial_dim=single_step_dim)
            ilm_state = ilm_out["state"]
            ilm_log_prob = rf.log_softmax(ilm_out["output"], axis=model.target_dim)

            label_log_prob = (
                label_log_prob - model.search_args["ilm_scale"] * ilm_log_prob
            )

        if model.search_args.get("use_ctc", False):
            # add ctc espnet
            ctc_prefix_scores, ctc_state = ctc_prefix_scorer(
                output_length=i,
                last_ids=target_ctc,
                state=ctc_state,
                att_w=att_weights if model.search_args.get("window_margin", 0) > 0 else None,
            )

            if i == 0:
                ctc_prefix_scores = ctc_prefix_scores.view(batch_size, beam_size, -1)[
                    :, 0, :
                ].unsqueeze(1)
            else:
                ctc_prefix_scores = ctc_prefix_scores.view(batch_size, beam_size, -1)

            ctc_prefix_scores = rf.Tensor(
                name="ctc_prefix_scores",
                # dims=batch_dims_ + [model.target_dim],
                dims=[batch_size_dim, beam_dim, model.target_dim],
                dtype="float32",
                raw_tensor=ctc_prefix_scores[:, :, :blank_index],
            )
            label_log_prob = (
                label_log_prob + model.search_args.get("ctc_scale") * ctc_prefix_scores
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
        decoder_state = tree.map_structure(
            lambda s: rf.gather(s, indices=backrefs), decoder_state
        )

        if model.search_args.get("lm_scale", 0.0) > 0:
            lm_state = model.language_model.select_state(lm_state, backrefs)

        if model.search_args.get("ilm_scale", 0.0) > 0:
            ilm_state = model_aed.ilm.select_state(ilm_state, backrefs)


        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        i += 1

        if model.search_args.get("use_ctc", False):
            best_ids = target
            if model.search_args.get("ctc_state_fix", True):
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

    # if model.search_args.get("rescore_w_ctc",False):
    #     from .two_pass import rescore_w_ctc
    #     seq_targets, seq_log_prob = rescore_w_ctc(model, seq_targets, seq_log_prob, ctc_out, batch_size, beam_size, model.blank_idx)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = False
