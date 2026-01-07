__all__ = ["forward_step", "forward_step_ctc_greedy_decoding"]

from typing import Optional

import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim
from returnn.tensor import Tensor as ReturnnTensor
from torch import Tensor

from .beam_search import beam_search_decode
from .ctc_label_sync_espnet import ctc_label_sync_search_v1
from ..networks.conformer_qwen_v1 import Qwen2DecoderState
from ..networks.interfaces.base_encoder_decoder_model import BaseEncoderDecoderModel


def forward_step(
    *,
    model: BaseEncoderDecoderModel,
    extern_data: TensorDict,

    beam_size: int,
    max_tokens_per_sec: Optional[int] = None,
    sample_rate: Optional[int] = None,

    **kwargs,
):
    """
    Runs full recognition on the given data.

    RETURNN ENTRYPOINT (for search/inference)!!
    """
    assert beam_size > 0

    data_: ReturnnTensor = extern_data["data"]
    data: Tensor = data_.raw_tensor
    seq_len: Tensor = data_.dims[1].dyn_size_ext.raw_tensor.to(device=data.device)
    max_seq_len = define_maximum_sequence_length(max_tokens_per_sec, sample_rate, seq_len)

    # ENCODER (FORWARD) STEP (for inference)
    decoder_state: Qwen2DecoderState
    decoder_state, _, _ = model.forward_encoder(data, seq_len, beam_size) # Initial beam size is beam_size

    # BEAM SEARCH DECODING (contains DECODER (FORWARD) STEPs)
    seq_targets, seq_log_prob, _, out_seq_len = beam_search_decode(
        model=model,
        decoder_state=decoder_state,
        beam_size=beam_size,
        batch_size=data.shape[0],
        device=data.device,
        max_seq_len=max_seq_len,
    )

    beam_dim = Dim(beam_size, name="beam")
    vocab_dim = Dim(model.num_labels, name="vocab")
    lens_data = rf.convert_to_tensor(out_seq_len, dims=[batch_dim, beam_dim])
    lens_dim = Dim(lens_data, name="seq_len")

    seq_targets_rf = rf.convert_to_tensor(seq_targets, dims=[batch_dim, beam_dim, lens_dim], sparse_dim=vocab_dim)

    ctx = rf.get_run_ctx()
    ctx.mark_as_output(seq_targets_rf, "tokens", dims=[batch_dim, beam_dim, lens_dim])
    ctx.mark_as_output(seq_log_prob, "scores", dims=[batch_dim, beam_dim])

def define_maximum_sequence_length(max_tokens_per_sec: int | None, sample_rate: int | None, seq_len: Tensor) -> Tensor:
    max_seq_len = seq_len
    if max_tokens_per_sec is not None and sample_rate is not None:
        assert max_tokens_per_sec > 0 and sample_rate > 0
        max_seq_len = max_tokens_per_sec * (seq_len / sample_rate)
    return max_seq_len



def forward_step_ctc_greedy_decoding(
        *,
        model: BaseEncoderDecoderModel,
        extern_data: TensorDict,

        beam_size: int,

        # OLD (not remove just in case)
        max_tokens_per_sec: Optional[int] = None,
        sample_rate: Optional[int] = None,

        # CTC Search params
        ctc_soft_collapse_threshold: Optional[float] = None,
        ctc_top_k_pruning: Optional[int] = None,
        ctc_top_k_pruning_reduce_func: str = "mean",
        ctc_scale: float = 1.0,
        prior_scale: float = 1.0,
        lm_scale: float = 1.0,

        **kwargs,
):
    """
    Runs full recognition on the given data. Using only CTC
    """
    assert beam_size > 0

    data_: ReturnnTensor = extern_data["data"]

    # From robins'
    #if data_.feature_dim and data_.feature_dim.dimension == 1:
    #    data_ = rf.squeeze(data_, axis=data_.feature_dim)

    data: Tensor = data_.raw_tensor
    seq_len: Tensor = data_.dims[1].dyn_size_ext.raw_tensor.to(device=data.device)

    seq_targets_rf, seq_log_prob_rf, lens_dim, beam_dim = ctc_label_sync_search_v1(
        model=model,
        data=data,
        data_seq_lens=seq_len,
        beam_size=beam_size,
        ctc_soft_collapse_threshold=ctc_soft_collapse_threshold,
        ctc_top_k_pruning=ctc_top_k_pruning,
        ctc_top_k_pruning_reduce_func=ctc_top_k_pruning_reduce_func,
        ctc_scale=ctc_scale,
        prior_scale=prior_scale,
        lm_scale=lm_scale,
    )

    ctx = rf.get_run_ctx()
    ctx.mark_as_output(seq_targets_rf, "tokens", dims=[batch_dim, beam_dim, lens_dim])
    ctx.mark_as_output(seq_log_prob_rf, "scores", dims=[batch_dim, beam_dim])

