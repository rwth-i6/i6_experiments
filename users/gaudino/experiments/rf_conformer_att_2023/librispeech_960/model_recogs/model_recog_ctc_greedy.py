from __future__ import annotations

from typing import Optional, Any, Tuple, Dict, Sequence, List

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.conformer_import_moh_att_2023_06_30 import (
    Model,
)
from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.search_functions import remove_blank_and_eos

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray

from i6_experiments.users.zeyer.model_interfaces import RecogDef

import torch

def model_recog_ctc(
    *,
    model: Model,
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

    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
    print("** max seq len:", max_seq_len.raw_tensor)

    beam_dim = Dim(1, name="initial-beam")

    assert len(batch_dims) == 1
    batch_size_dim = batch_dims[0]

    ctc_out = enc_args["ctc"].copy_transpose(
        (batch_size_dim, enc_spatial_dim, model.target_dim_w_b)
    )  # [B,T,V+1]

    enc_args.pop("ctc")

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

    seq_targets, out_spatial_dim = remove_blank_and_eos(hyps.unsqueeze(1), max_out_len, batch_dims, beam_dim, model.target_dim, blank_idx=10025, eos_idx=0)

    # # torchaudio ctc decoder
    # # only runs on cpu -> slow
    # # maybe dump ctc_out and load it in on cpu fast
    # from torchaudio.models.decoder import ctc_decoder
    # from returnn.datasets.util.vocabulary import Vocabulary
    # vocab_1 = Vocabulary("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab", eos_label=0)
    #
    # beam_search_decoder = ctc_decoder(
    #     lexicon=None,  # lexicon free decoding
    #     tokens=vocab_1.labels + ['<b>', '|'],  # files.tokens,
    #     lm=None,
    #     nbest=3,
    #     beam_size=12,
    #     word_score=0,
    #     blank_token='<b>',
    # )
    #
    # hypos = beam_search_decoder(ctc_out.raw_tensor.to('cpu'), hlens)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# CTCRecogDef API
model_recog_ctc: RecogDef[Model]
model_recog_ctc.output_with_beam = True
model_recog_ctc.output_blank_label = "<blank>"
model_recog_ctc.batch_size_dependent = False
