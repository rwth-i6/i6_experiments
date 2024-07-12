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
from i6_experiments.users.yang.torch.utils.tensor_ops import mask_eos_label

import numpy

_batch_size_factor = 160

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.trafo_lm import trafo_lm_kazuki_import

def model_recog(
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

    assert model.enc_aux_logits_12, "Expected final ctc logits in enc_aux_logits_12"

    enc_ctc = model.enc_aux_logits_12(enc_args["enc"])

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

    # always add up blank and eos in log space
    ctc_out = mask_eos_label(ctc_out, add_to_blank=True)


    blank_index = model.target_dim.get_dim_value()

    # how to pass the arg to rec model
    blank_scale = 1.0


    if False: # not supported yet
        ctc_out_raw = ctc_out.raw_tensor
        ctc_log_prior = numpy.loadtxt(
            model.search_args.get("ctc_prior_file", None), dtype="float32"
        )
        ctc_out_raw = ctc_out_raw - (
            torch.tensor(ctc_log_prior)
            .repeat(ctc_out_raw.shape[0], ctc_out_raw.shape[1], 1)
            .to("cuda")
            * model.search_args.get("prior_scale", 0.3)
        )
        if model.search_args.get("prior_corr_renorm", False):
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
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = False
