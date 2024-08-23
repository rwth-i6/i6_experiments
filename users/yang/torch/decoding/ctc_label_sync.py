from __future__ import annotations

from typing import Optional, Any, Tuple, Dict, Sequence, List
import tree


from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray


# from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.conformer_import_moh_att_2023_06_30 import Model
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
from i6_experiments.users.yang.torch.lm.network.lstm_lm import LSTMLM, LSTMLMConfig
from i6_experiments.users.yang.torch.luca_ctc.model_conformer_kd_ctc import Model
from i6_experiments.users.yang.torch.luca_ctc.model_conformer_kd_ctc import get_lstm_default_config
from i6_experiments.users.yang.torch.utils.masking import get_seq_mask_v2
import torch
import numpy

# _ctc_prior_filename = "/u/luca.gaudino/debug/ctc/prior.txt"
# _ctc_prior_filename = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZdcvhAOyWl95/output/prior.txt"


# many hard-coded functions/variables, just to test the label-sync search
# for now only ctc (and ELM) is considered

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
    log_ctc_prob = torch.nn.functional.log_softmax(enc_ctc.raw_tensor, dim=-1)

    #
    config = get_global_config()
    search_args = config.typed_value("search_args", {})


    beam_size = search_args.get("beam_size", 12)
    lm_scale = search_args.get("lm_scale", 0.0)
    length_normalization_exponent = search_args.get("length_norm_scale", 0.0)
    lm_linear_combine = search_args.get("lm_linear_combine", False)
    if lm_scale > 0:
        # for now hard coded pure torch based LSTM
        if hasattr(model, "train_extern_lm"):
            lm = model.train_extern_lm
        else:
            print("*********************init LM")
            lstm_cfg = get_lstm_default_config(log_prob_output=True)
            print('if lm log output***************************', lstm_cfg.log_prob_output)
            lm = LSTMLM(step=0, cfg=lstm_cfg)
            lstm_path = "/work/asr4/zyang/torch/librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.la2CPTQHhFyg/output/models/epoch.030.pt"
            lm.load_state_dict(torch.load(lstm_path)["model"])
            print('*************************train extern lm loaded***********************************')
        cur_device = log_ctc_prob.device
        if next(lm.parameters()).device != cur_device:
            print("move the LM to gpu")
            lm.to(cur_device)
        lm.dropout = None
    # by default no length norm
    #length_normalization_exponent = model.search_args.get("length_normalization_exponent", 0.0)
    #length_normalization_exponent = config.typed_value("length_norm_scale", 0.0)
    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
    print("** max seq len:", max_seq_len.raw_tensor)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    # pure ctc no decoder state
    # decoder_state = model.decoder_default_initial_state(
    #     batch_dims=batch_dims_, enc_spatial_dim=enc_spatial_dim
    # )



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

    batch_shift_base = torch.arange(0, batch_size, dtype=torch.int64, device=log_ctc_prob.device)
    batch_shift = batch_shift_base * beam_size

    # if model.search_args.get("use_ctc", False) or model.search_args.get("rescore_with_ctc", False):
    #     if model.search_args.get("encoder_ctc", False):
    #         enc_ctc = enc_args_ctc["ctc"]
    #     else:
    #         enc_ctc = enc_args["ctc"]
    #
    #     ctc_out = (
    #         enc_ctc
    #         .copy_transpose((batch_size_dim, enc_spatial_dim, model.target_dim_w_b))
    #         .raw_tensor
    #     )  # [B,T,V+1]
    ctc_out = log_ctc_prob

    if model.search_args.get("mask_eos", True):
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
    if model.search_args.get("prior_corr", False):
        ctc_log_prior = numpy.loadtxt(model.search_args.get("prior_file", model.search_args.get("ctc_prior_file", "")),
                                      dtype="float32")
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



    # if model.search_args.get("use_ctc", False):
    #     # ctc prefix scorer espnet
    #     from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.espnet_ctc.ctc_prefix_score_espnet import (
    #         CTCPrefixScoreTH,
    #     )
    #
    #     # hlens = max_seq_len.raw_tensor.repeat(beam_size).view(beam_size, data.raw_tensor.shape[0]).transpose(0, 1)
    #     hlens = max_seq_len.raw_tensor
    #
    #     if model.search_args.get("prior_corr", False):
    #         ctc_log_prior = numpy.loadtxt(model.search_args.get("prior_file", model.search_args.get("ctc_prior_file", "")), dtype="float32")
    #         ctc_out = ctc_out - (
    #             torch.tensor(ctc_log_prior)
    #             .repeat(ctc_out.shape[0], ctc_out.shape[1], 1)
    #             .to(ctc_out.device)
    #             * model.search_args["prior_scale"]
    #         )
    #         ctc_out = ctc_out - torch.logsumexp(ctc_out, dim=2, keepdim=True)
    #
    #     ctc_prefix_scorer = CTCPrefixScoreTH(
    #         ctc_out,
    #         hlens,
    #         blank_index,
    #         0,
    #         model.search_args.get("window_margin", 0),
    #         model.search_args.get("mask_eos", True),
    #     )
    #     ctc_state = None
    #enc_args.pop("ctc")


    # lm init states:
    if lm_scale > 0:
        (h0, c0) = lm.get_default_init_state(batch_size, device=ctc_out.device) # shape (N,B,d), init step b=1
        # the forwarding of lstm lm should be (B*d, 1, d), namely no T dim
        num_lstm_layers = h0.shape[0]
        lstm_dim = h0.shape[-1]

    seq_lm_score = torch.zeros([batch_size,beam_size], device=ctc_out.device)



    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        # fixed: before it was computed at step 0
        # if i == 0:
        #     input_embed = rf.zeros(batch_dims_ + [model.target_embed.out_dim], feature_dim=model.target_embed.out_dim)
        # else:
        #     input_embed = model.target_embed(target)
        # # seemed to be used for aed?
        # if model.search_args.get("use_aed", False):
        #     step_out, decoder_state = model.loop_step(
        #         **enc_args,
        #         enc_spatial_dim=enc_spatial_dim,
        #         input_embed=input_embed,
        #         state=decoder_state,
        #     )
        #     att_weights = step_out.pop("att_weights", None).raw_tensor
        #     if i==0:
        #         att_weights = torch.flatten(att_weights.squeeze(3).view(batch_size, 1, -1), end_dim=1)
        #     else:
        #         att_weights = torch.flatten(att_weights.squeeze(3).view(batch_size, beam_size, -1), end_dim=1)
        #     logits = model.decode_logits(input_embed=input_embed, **step_out)
        #     label_log_prob = rf.log_softmax(
        #         logits, axis=model.target_dim
        #     )  # (Dim{'initial-beam'(1)}, Dim{B}, Dim{F'target'(10025)})
        #
        #     label_log_prob = label_log_prob * model.search_args.get("att_scale", 1.0)
        # else:
        #     label_log_prob = 0
        #
        # if model.search_args.get("lm_scale", 0.0) > 0:
        #     lm_out = model.language_model(target, state=lm_state, spatial_dim=single_step_dim)
        #     lm_state = lm_out["state"]
        #     lm_log_prob = rf.log_softmax(lm_out["output"], axis=model.target_dim)
        #
        #     if model.search_args.get("use_lm_first_label", True) or i > 0:
        #         label_log_prob = (
        #             label_log_prob + model.search_args["lm_scale"] * lm_log_prob
        #         )
        #
        # if model.search_args.get("ilm_scale", 0.0) > 0:
        #     ilm_out = model.ilm(input_embed, state=ilm_state, spatial_dim=single_step_dim)
        #     ilm_state = ilm_out["state"]
        #     ilm_log_prob = rf.log_softmax(ilm_out["output"], axis=model.target_dim)
        #
        #     label_log_prob = (
        #         label_log_prob - model.search_args["ilm_scale"] * ilm_log_prob
        #     )

        if model.search_args.get("use_ctc", True):
            # add ctc espnet
            ctc_prefix_scores, ctc_state = ctc_prefix_scorer(
                output_length=i,
                last_ids=target_ctc,
                state=ctc_state,
                att_w= None,
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
            # what is the beam_dim here
            label_log_prob = ctc_prefix_scores
        # lm_scores:
        if lm_scale > 0:
            raw_target = target.raw_tensor
            # raw_target shape (B*b,1)
            raw_target = raw_target.view(-1,1)
            lm_log_prob, (h0, c0) = lm.incremental_step(raw_target, h0, c0) # (B*b,1,V) and (N, B*b, h)

            if i == 0:
                lm_log_prob = lm_log_prob.view(batch_size, 1, model.target_dim.get_dim_value())
            else:
                lm_log_prob = lm_log_prob.view(batch_size, beam_size, model.target_dim.get_dim_value())
                # states are anyway the same
            if lm_linear_combine:
                lm_log_prob_renorm = (lm_scale*lm_log_prob).log_softmax(axis=-1)
                lm_linear_comb_scale = search_args.get("lm_liner_combine_scale", 0.0)
                assert lm_linear_comb_scale < 1
                # add ctc/aed score and lm score linearly
                raw_label_log_prob = label_log_prob.raw_tensor
                lm_linear_comb_scale = torch.tensor(lm_linear_comb_scale)
                raw_combined_score = torch.logaddexp(raw_label_log_prob + torch.log(1-lm_linear_comb_scale), lm_log_prob_renorm + torch.log(lm_linear_comb_scale))
                label_log_prob.raw_tensor = raw_combined_score

            else:
                rf_lm_log_prob = rf.Tensor(
                    name="external_lm_scores",
                    dims=[batch_size_dim, beam_dim, model.target_dim],
                    dtype="float32",
                    raw_tensor=lm_log_prob
                )
                label_log_prob = label_log_prob + lm_scale * rf_lm_log_prob

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

        # sanity check lm output


        # update lm states
        if lm_scale > 0:
            raw_backrefs = backrefs.raw_tensor
            if i > 0:
                raw_backrefs = backrefs.raw_tensor + batch_shift.unsqueeze(-1)

            #

            # raw_backrefs_tmp = backrefs.raw_tensor + batch_shift.unsqueeze(-1)
            raw_backrefs = raw_backrefs.view(-1) # (B*b)
            # assert! bug here
            h0 = h0[:, raw_backrefs, :]
            c0 = c0[:, raw_backrefs, :]




            # if i==0:
            #     tmp_beam = 1
            # else:
            #     tmp_beam = beam_size
            # select_lm_log_prob = lm_log_prob.view(batch_size*tmp_beam,model.target_dim.get_dim_value())[raw_backrefs,:]
            # select_lm_log_prob = select_lm_log_prob.view(batch_size, beam_size, -1)
            # raw_target = target.raw_tensor
            # lm_beam_score = torch.gather(select_lm_log_prob, index=raw_target.unsqueeze(-1), dim=-1).squeeze(-1)
            # print("**step ",i)
            # print("lm beam score", lm_beam_score)
            # print('raw target', raw_target)




        seq_targets.append(target)
        seq_backrefs.append(backrefs)



        # decoder state not needed
        # decoder_state = tree.map_structure(
        #     lambda s: rf.gather(s, indices=backrefs), decoder_state
        # )

        # if model.search_args.get("lm_scale", 0.0) > 0:
        #     lm_state = model.language_model.select_state(lm_state, backrefs)
        #
        # if model.search_args.get("ilm_scale", 0.0) > 0:
        #     ilm_state = model.ilm.select_state(ilm_state, backrefs)


        ended = rf.gather(ended, indices=backrefs)


        # if lm_scale >0:
        #     raw_ended = ended.raw_tensor.transpose(0,1)
        #     # print("********raw ended", raw_ended)
        #     # print("***** raw ended shape", raw_ended.shape)
        #     seq_lm_score_reorder = seq_lm_score.reshape(batch_size*beam_size)
        #     seq_lm_score_reorder = seq_lm_score_reorder[raw_backrefs]
        #     seq_lm_score_reorder = seq_lm_score_reorder.view(batch_size, beam_size)
        #     seq_lm_score = torch.where(raw_ended, seq_lm_score_reorder, seq_lm_score_reorder + lm_beam_score)
            # print(' **** seq_lm_score', seq_lm_score)

        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        i += 1

        if model.search_args.get("use_ctc", True):
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
        #ended = rf.logical_or(ended, rf.copy_to_device(i >= rf.convert_to_tensor(50, dtype="int32")))
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



    # print("!!!!!!!!!!!!!!!************raw seq shape",  seq_targets.raw_tensor.shape)
    # raw_seq_targets = seq_targets.raw_tensor[:,0,0] # 1-D tensor (L, B, b)
    # lm_input = torch.cat([torch.zeros_like(raw_seq_targets)[:1], raw_seq_targets[:-1]], dim=0)
    # lm_input = lm_input.unsqueeze(0)
    # max_len = lm_input.shape[1]
    # lm_output = lm(lm_input)
    # lm_output_probs = lm_output.log_softmax(dim=-1)
    # seq_len = out_seq_len.raw_tensor[0,:1]# (5, 3)?
    # seq_len = seq_len +1
    # seq_len_mask = get_seq_mask_v2(seq_len, max_len, device=lm_output_probs.device) # (B=1,)
    # lm_target_probs = torch.gather(lm_output_probs, index=raw_seq_targets.unsqueeze(0).unsqueeze(-1), dim=-1).squeeze()
    # print("batch size", batch_size)
    #
    # print('seq len', seq_len)
    # print('max len', max_len)
    # print('output len', out_seq_len.raw_tensor+1)
    # print('lm_output shape', lm_output_probs.shape)
    # print('lm target probs', lm_target_probs)
    #
    #
    # h1, c1 = lm.get_default_init_state(1, device=ctc_out.device)
    #
    #
    # accu_score = 0
    # for i in range(max_len):
    #     step_output, (h1,c1) = lm.incremental_step(lm_input[:,i].unsqueeze(1), h1, c1)
    #     step_output = step_output.log_softmax(dim=-1)
    #     # print(step_output.shape)
    #     # print(raw_seq_targets[i].unsqueeze(0).unsqueeze(-1).shape)
    #     step_target_prob = torch.gather(step_output, index=raw_seq_targets[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1), dim=-1)
    #     # print('!!!step label',raw_seq_targets[i] )
    #     # print('!!!step lm score', step_target_prob)
    #     if i < seq_len.squeeze(0):
    #         accu_score = accu_score + step_target_prob.squeeze(0).squeeze(0)
    #
    # # print("*************again**********************")
    # # lm_output = lm(lm_input)
    # # lm_output_probs = lm_output.log_softmax(dim=-1)
    # # lm_target_probs = torch.gather(lm_output_probs, index=raw_seq_targets.unsqueeze(0).unsqueeze(-1), dim=-1).squeeze()
    #
    # # h1, c1 = lm.get_default_init_state(batch_size, device=ctc_out.device)
    # # for i in range(max_len):
    # #     step_output, (h1,c1) = lm.incremental_step(lm_input[:,i].unsqueeze(1), h1, c1)
    # #     step_output = step_output.log_softmax(dim=-1)
    # #     print(step_output.shape)
    # #     print(raw_seq_targets[i].unsqueeze(0).unsqueeze(-1).shape)
    # #     step_target_prob = torch.gather(step_output, index=raw_seq_targets[i].unsqueeze(0).unsqueeze(-1).unsqueeze(-1), dim=-1)
    # #     print('!!!step label',raw_seq_targets[i] )
    # #     print('!!!step lm score', step_target_prob)
    #
    # lm_final_score = torch.sum(torch.where(seq_len_mask, lm_target_probs, torch.zeros_like(lm_target_probs)))
    # print('best seq',raw_seq_targets)
    # print('best seq tensor shape', raw_seq_targets.shape)
    # print('!!!!!!!!!!!!!!*****************seq log prob', seq_log_prob.raw_tensor)
    # print('!!!!!!!!!!!!!!*********************************seq lm score', lm_final_score)
    # print('!!!!!!!!!!!!!!********************************* accu lm score', accu_score)
    # print('!!!!!!!!!!!!!!************************************accumulated lm score', seq_lm_score)

    # check the computation of lm score

    # if model.search_args.get("rescore_w_ctc",False):
    #     from .two_pass import rescore_w_ctc
    #     seq_targets, seq_log_prob = rescore_w_ctc(model, seq_targets, seq_log_prob, ctc_out, batch_size, beam_size, model.blank_idx)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog_label_sync: RecogDef[Model]
model_recog_label_sync.output_with_beam = True
model_recog_label_sync.output_blank_label = "<blank>"
model_recog_label_sync.batch_size_dependent = False
