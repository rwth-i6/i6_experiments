__all__ = ["forward_step"]

from typing import Optional

import returnn.frontend as rf
import torch
from returnn.tensor import Dim, TensorDict, batch_dim
from returnn.tensor import Dim, TensorDict, batch_dim
from returnn.tensor import Tensor as ReturnnTensor
from torch import Tensor

from .beam_search import beam_search_v1
from .ctc_label_sync_espnet import ctc_label_sync_search_v2
from .encoder_decoder_protocol import EncoderDecoderModel


def forward_step(
    *,
    model: EncoderDecoderModel,
    extern_data: TensorDict,
    beam_size: int,
    max_tokens_per_sec: Optional[int] = None,
    sample_rate: Optional[int] = None,

    length_norm_exponent: float = 1.0,

    **kwargs,
):
    """
    Runs full recognition on the given data.

    CALLED FROM RETURNN.CONFIG!!
    """

    assert beam_size > 0

    data = extern_data["data"].raw_tensor
    seq_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor.to(device=data.device)

    max_seq_len = seq_len
    if max_tokens_per_sec is not None and sample_rate is not None:
        assert max_tokens_per_sec > 0 and sample_rate > 0
        max_seq_len = max_tokens_per_sec * (seq_len / sample_rate)

    decoder_state = model.forward_encoder(data, seq_len)
    seq_targets, seq_log_prob, _label_log_probs, out_seq_len = beam_search_v1(
        model=model,
        beam_size=beam_size,
        batch_size=data.shape[0],
        decoder_state=decoder_state,
        device=data.device,
        max_seq_len=max_seq_len,
        length_norm_exponent=length_norm_exponent,
    )

    beam_dim = Dim(beam_size, name="beam")
    vocab_dim = Dim(model.num_labels, name="vocab")
    lens_data = rf.convert_to_tensor(out_seq_len, dims=[batch_dim, beam_dim])
    lens_dim = Dim(lens_data, name="seq_len")

    ctx = rf.get_run_ctx()
    seq_targets_rf = rf.convert_to_tensor(seq_targets, dims=[batch_dim, beam_dim, lens_dim], sparse_dim=vocab_dim)
    ctx.mark_as_output(seq_targets_rf, "tokens", dims=[batch_dim, beam_dim, lens_dim])
    ctx.mark_as_output(seq_log_prob, "scores", dims=[batch_dim, beam_dim])



def forward_step_ctc_decoding_v2(
        *,
        model: EncoderDecoderModel,
        extern_data: TensorDict,

        beam_size: int,
        ctc_scale: float = 1.0, # Default CTC
        prior_scale: float = 0.0,
        lm_scale: float = 0.0,
        sllm_scale: float = 0.0,

        sllm_as_llm: bool = False,

        ctc_soft_collapse_threshold: Optional[float] = None,
        ctc_top_k_pruning: Optional[int] = None,
        ctc_top_k_pruning_reduce_func: str = "mean",

        **kwargs,
):
    """
    Runs full recognition on the given data. Using only CTC
    """
    assert beam_size > 0

    data_: ReturnnTensor = extern_data["data"]
    data: Tensor = data_.raw_tensor
    seq_len: Tensor = data_.dims[1].dyn_size_ext.raw_tensor.to(device=data.device)

    seq_targets_rf, seq_log_prob_rf, lens_dim, beam_dim = ctc_label_sync_search_v2(
        model=model,
        data=data,
        data_seq_lens=seq_len,
        beam_size=beam_size,
        ctc_soft_collapse_threshold=ctc_soft_collapse_threshold,
        ctc_top_k_pruning=ctc_top_k_pruning,
        ctc_top_k_pruning_reduce_func=ctc_top_k_pruning_reduce_func,
        ctc_scale=ctc_scale,
        prior_scale=prior_scale,
        external_lm_scale=lm_scale,
        sllm_scale=sllm_scale,
        sllm_as_llm=sllm_as_llm,
    )

    ctx = rf.get_run_ctx()
    ctx.mark_as_output(seq_targets_rf, "tokens", dims=[batch_dim, beam_dim, lens_dim])
    ctx.mark_as_output(seq_log_prob_rf, "scores", dims=[batch_dim, beam_dim])

def forward_step_greedy_ctc(
        *,
        model: EncoderDecoderModel,
        extern_data: TensorDict,
        **kwargs,
):
    """
    Runs full recognition on the given data. Using only CTC
    """
    data_: ReturnnTensor = extern_data["data"]
    data: Tensor = data_.raw_tensor
    seq_len: Tensor = data_.dims[1].dyn_size_ext.raw_tensor.to(device=data.device)

    ## ENCODING FORWARD
    decoder_state, aux_logits, encoder_lens = model.forward_encoder_with_ctc(
        data,
        seq_len,
    )
    ctc_log_prob = torch.nn.functional.log_softmax(aux_logits, dim=-1)  # [B, T, V]
    greedy_ids = torch.argmax(ctc_log_prob, dim=-1)  # [B, T]
    greedy_log_prob = torch.gather(ctc_log_prob, dim=-1, index=greedy_ids.unsqueeze(-1)).squeeze(-1)  # [B, T]
    tokens_list, seq_log_prob = _ctc_greedy_collapse(greedy_ids, greedy_log_prob, encoder_lens, model.blank_idx)

    max_len = max(len(t) for t in tokens_list)
    seq_targets = torch.full(
        (len(tokens_list), max_len),
        fill_value=model.blank_idx,
        device=ctc_log_prob.device,
        dtype=torch.long,
    )  # [B, T']
    for b, t in enumerate(tokens_list):
        seq_targets[b, : len(t)] = t

    out_seq_len = torch.tensor([len(t) for t in tokens_list],
                               device=seq_targets.device,
                               dtype=torch.int32,
                               )

    # Add beam dimension (beam_size=1 for greedy)
    seq_targets = seq_targets.unsqueeze(1)  # [B, 1, T']
    out_seq_len = out_seq_len.unsqueeze(1)  # [B, 1]
    seq_log_prob = seq_log_prob.unsqueeze(1)  # [B, 1]

    beam_dim = Dim(1, name="beam")
    vocab_dim = Dim(model.num_labels, name="vocab")
    lens_data = rf.convert_to_tensor(out_seq_len, dims=[batch_dim, beam_dim])
    lens_dim = Dim(lens_data, name="seq_len")
    seq_targets_rf = rf.convert_to_tensor(seq_targets, dims=[batch_dim, beam_dim, lens_dim], sparse_dim=vocab_dim)

    ctx = rf.get_run_ctx()
    ctx.mark_as_output(seq_targets_rf, "tokens", dims=[batch_dim, beam_dim, lens_dim])
    ctx.mark_as_output(seq_log_prob, "scores", dims=[batch_dim, beam_dim])


def _ctc_greedy_collapse(ids, logp, lens, blank_id):
    """
    ids:   [B, T]
    logp:  [B, T]
    lens:  [B]
    """
    out_ids = []
    out_logp = []

    B, T = ids.shape
    for b in range(B):
        prev = blank_id
        seq = []
        score = 0.0

        for t in range(lens[b]):
            cur = ids[b, t].item()
            if cur != blank_id and cur != prev:
                seq.append(cur)
                score += logp[b, t].item()
            prev = cur

        out_ids.append(torch.tensor(seq, device=ids.device))
        out_logp.append(score)

    return out_ids, torch.tensor(out_logp, device=ids.device)