__all__ = ["ctc_forward_step_v1"]

from abc import abstractmethod
from typing import Generic, Optional, Union

import torch
from torch import Tensor
import torch.nn.functional as F

import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.networks.interfaces.base_encoder_decoder_model import \
    BaseEncoderDecoderModel
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.recognition.ctc.ctc_beam_search import beam_search_v1


def ctc_forward_step_v1(
        *,
        model: Union[BaseEncoderDecoderModel],
        extern_data: TensorDict,
        beam_size: int,
        use_dec_aux_log_probs: bool = False,
        ctc_soft_collapse_threshold: Optional[float] = None,
        ctc_soft_collapse_reduce_type: str = "logmeanexp",
        **kwargs,
):
    """
    Fix beam search, before all beam entries were identical
    (uses initial_beam = 1 in both model.forward_encoder and beam_search_v2).
    Now compatible with SpeechLmV2 and SpeechLmV3.

    If `greedy_ctc_output_layer` is given, also performs greedy CTC decoding on that output layer and registers
    the result with repetitions and blanks removed as additional output "ctc_tokens". To correctly write these
    results to file, use `RecognitionToTextDictCallbackV2`.

    Args:
        model:
        extern_data:
        beam_size:
        max_tokens_per_sec: Optional maximum tokens per second for decoding.
        sample_rate:
        greedy_ctc_output_layer: Optional index of the CTC output layer to use for additional CTC decoding.
    """

    from returnn.config import get_global_config

    config = get_global_config(return_empty_if_none=True)

    assert beam_size > 0

    data_key = config.value("default_data_key", "audio")
    data_ = extern_data[data_key]
    if data_.feature_dim and data_.feature_dim.dimension == 1:
        data_ = rf.squeeze(data_, axis=data_.feature_dim)
    data = data_.raw_tensor
    seq_len = extern_data[data_key].dims[1].dyn_size_ext.raw_tensor.to(device=data.device)

    seq_targets, seq_log_prob, out_seq_len = beam_search_v1(
        model=model,
        raw_audio=data,
        raw_audio_lens=seq_len,
        beam_size=beam_size,
        batch_size=data.shape[0],
        device=data.device,
        use_dec_aux_log_probs=use_dec_aux_log_probs,
        ctc_soft_collapse_threshold=ctc_soft_collapse_threshold,
        ctc_soft_collapse_reduce_type=ctc_soft_collapse_reduce_type,
    )

    beam_dim = Dim(beam_size, name="beam")
    vocab_dim = Dim(model.num_labels, name="vocab")
    lens_data = rf.convert_to_tensor(out_seq_len, dims=[batch_dim, beam_dim])
    lens_dim = Dim(lens_data, name="seq_len")

    ctx = rf.get_run_ctx()
    seq_targets_rf = rf.convert_to_tensor(seq_targets, dims=[batch_dim, beam_dim, lens_dim], sparse_dim=vocab_dim)
    ctx.mark_as_output(seq_targets_rf, "tokens", dims=[batch_dim, beam_dim, lens_dim])
    ctx.mark_as_output(seq_log_prob, "scores", dims=[batch_dim, beam_dim])