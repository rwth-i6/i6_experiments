from __future__ import annotations

from typing import Optional, Any, Tuple, Dict, Sequence, List
import tree


from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray


# from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.conformer_import_moh_att_2023_06_30 import Model
from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.espnet_ctc.beam_search_timesync_espnet import BeamSearchTimeSync

from i6_experiments.users.gaudino.models.asr.decoder.att_decoder_rf import ATTDecoder
from i6_experiments.users.gaudino.models.asr.rf.scorers_rf_espnet.lm_ilm_scorer import LM_ILM_Scorer

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.search_functions import remove_eos_from_start_and_end

import torch
import numpy

# _ctc_prior_filename = "/u/luca.gaudino/debug/ctc/prior.txt"
# _ctc_prior_filename = "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-02-22--conformer-swb/work/i6_core/returnn/extract_prior/ReturnnComputePriorJobV2.ZdcvhAOyWl95/output/prior.txt"


def model_recog_ts_espnet(
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
    batch_size_dim = batch_dims[0]
    beam_size = search_args.get("beam_size", 12)

    beam_dim = Dim(beam_size, name="beam-dim")
    batch_dims_ = batch_dims + [beam_dim]
    assert batch_size_dim.get_dim_value() == 1, f"batch size {batch_size_dim.get_dim_value()} > 1 not supported yet"

    enc_args, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    if search_args.get("encoder_ctc", False):
        enc_args_ctc, enc_spatial_dim_ctc = model.encode_ctc(data, in_spatial_dim=data_spatial_dim)

    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
    print("** max seq len:", max_seq_len.raw_tensor)

    if search_args.get("encoder_ctc", False):
        enc_ctc = enc_args_ctc["ctc"]
    else:
        enc_ctc = enc_args["ctc"]
    enc_args.pop("ctc")

    ctc_out = (
        enc_ctc
        .copy_transpose((batch_size_dim, enc_spatial_dim, model.target_dim_w_b))
        .raw_tensor[0]
    )

    if search_args.get("mask_eos", True):
        ctc_eos = ctc_out[ :, model.eos_idx].unsqueeze(1)
        ctc_blank = ctc_out[ :, model.blank_idx].unsqueeze(1)
        ctc_out[ :, model.blank_idx] = torch.logsumexp(
            torch.cat([ctc_eos, ctc_blank], dim=1), dim=1
        )
        ctc_out[ :, model.eos_idx] = -1e30

    if search_args.get("prior_scale", 0.0) > 0.0:
        prior = numpy.loadtxt(search_args.get("prior_file", search_args.get("ctc_prior_file", "")),
                                      dtype="float32")
        prior = torch.tensor(prior).repeat(ctc_out.shape[0], 1).to(ctc_out.device)
        if not search_args.get("is_log_prior", True):
            prior = torch.log(prior)
        ctc_out = ctc_out - (
            prior
            * search_args["prior_scale"]
        )
        ctc_out = ctc_out - torch.logsumexp(ctc_out, dim=1, keepdim=True)

    att_scorer = ATTDecoder(model=model, batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim)

    scorers = {
        "ctc": model.ctc,
        "decoder": att_scorer
    }

    if search_args.get("lm_scale", 0.0) > 0.0 or search_args.get("ilm_scale", 0.0) > 0.0:
        scorers["lm"] = LM_ILM_Scorer(model=model, batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim, lm_scale=search_args.get("lm_scale", 0.0), ilm_scale=search_args.get("ilm_scale", 0.0))


    time_sync_beam_search = BeamSearchTimeSync(
        sos=model.bos_idx,
        beam_size=beam_size,
        scorers=scorers,
        weights={
            "ctc": search_args.get("ctc_scale", 1.0),
            "decoder": search_args.get("att_scale", 1.0),
            "lm": 1.0,
            "length_bonus": search_args.get("length_bonus", 0.0),
        },
        token_list=model.target_dim.vocab,
        blank=model.blank_idx,
    )

    res = time_sync_beam_search.forward(ctc_out, enc_args)

    hyps = [res[i].yseq for i in range(beam_size)]

    for i in range(beam_size):
        hyps[i] = remove_eos_from_start_and_end(hyps[i], model.eos_idx)

    lens = [len(hyps[i]) for i in range(beam_size)]

    out_lens = rf.Tensor("out_lens", dims=batch_dims_, dtype="int32", raw_tensor=torch.tensor(lens).unsqueeze(0).to(torch.int32))

    max_output_len = max(lens)
    out_spatial_dim = Dim(max_output_len, name="out_spatial")
    t_array = None
    scores = []

    for i in range(beam_size):

        hyp_raw = hyps[i]

        if len(hyps[i]) < max_output_len:
            hyp_raw_pad = torch.zeros(max_output_len - len(hyp_raw), dtype=torch.int64)
            hyp_raw = torch.cat([hyp_raw, hyp_raw_pad])

        hyp = rf.Tensor(
            f"hyp_{i}",
            dims=[out_spatial_dim],
            sparse_dim=model.target_dim,
            dtype="int64",
            raw_tensor=hyp_raw,
        )

        if t_array:
            t_array.push_back(hyp)
        else:
            t_array = TensorArray(hyp)
            t_array.push_back(hyp)

        scores.append(res[i].score)

    t_array = t_array.stack(axis=batch_size_dim)
    out_spatial_dim.dyn_size_ext = rf.reshape(out_lens, out_lens.dims, batch_dims_[::-1])
    t_array = rf.reshape(t_array,t_array.dims, batch_dims_ + [out_spatial_dim])
    seq_targets = t_array.copy_transpose([out_spatial_dim] + batch_dims_)

    seq_log_prob = rf.Tensor('seq_log_prob', batch_dims_, dtype="float32", raw_tensor=torch.tensor(scores).to(torch.float32).unsqueeze(0))
    # seq_log_prob = rf.copy_to_device(seq_log_prob, "cuda")


    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog_ts_espnet: RecogDef[Model]
model_recog_ts_espnet.output_with_beam = True
model_recog_ts_espnet.output_blank_label = "<blank>"
model_recog_ts_espnet.batch_size_dependent = False
