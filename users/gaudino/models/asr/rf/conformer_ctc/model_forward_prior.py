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

import numpy

_batch_size_factor = 160

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.trafo_lm import trafo_lm_kazuki_import

def model_forward_prior(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
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

    max_seq_len = enc_spatial_dim.get_size_tensor()

    print("** max seq len:", max_seq_len.raw_tensor)
    print("** max seq len shape:", max_seq_len.raw_tensor.shape)

    beam_dim = Dim(1, name="initial-beam")

    assert len(batch_dims) == 1
    batch_size_dim = batch_dims[0]

    ctc_out_logits = enc_ctc.copy_transpose(
        (batch_size_dim, enc_spatial_dim, model.target_dim_w_blank)
    ) # [B,T,V+1]

    ctc_out = rf.softmax(ctc_out_logits, axis=model.target_dim_w_blank)

    # batch_size = int(batch_size_dim.get_size_tensor().raw_tensor)
    #
    # sum_scores_raw = torch.zeros((batch_size, model.target_dim_w_blank.dimension))
    #
    # for i in range(batch_size):
    #     sum_scores_raw[i] = ctc_out[i, : max_seq_len.raw_tensor[i]].sum(0) / max_seq_len.raw_tensor[i]
    #
    # sum_scores = rf.Tensor(
    #     name="sum_scores",
    #     dims=[batch_size_dim, model.target_dim_w_blank],
    #     dtype="float32",
    #     raw_tensor=sum_scores_raw,
    # )

    return ctc_out, enc_spatial_dim



# RecogDef API
model_forward_prior: RecogDef[Model]
model_forward_prior.output_with_beam = True
# output_blank_label=blank is actually wrong for AED, but now we don't change it anymore
# because it would change all recog hashes.
# Also, it does not matter too much -- it will just cause an extra SearchRemoveLabelJob,
# which will not have any effect here.
model_forward_prior.output_blank_label = "<blank>"
model_forward_prior.batch_size_dependent = False
