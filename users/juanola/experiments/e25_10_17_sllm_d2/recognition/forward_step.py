__all__ = ["forward_step", "forward_step_v2", "forward_step_ctc_decoding", "forward_step_greedy_ctc"]

from typing import Optional

import returnn.frontend as rf
import torch
from returnn.tensor import Dim, TensorDict, batch_dim
from returnn.tensor import Tensor as ReturnnTensor
from torch import Tensor

from .beam_search import beam_search_decode, beam_search_v2
from .ctc.ctc_label_sync_espnet import ctc_label_sync_search_v1, ctc_label_sync_search_v2
from ..networks.conformer_qwen_v1 import Qwen2DecoderState
from ..networks.interfaces.base_encoder_decoder_model import BaseEncoderDecoderModel
from ..networks.sllm_with_ext_modules import SllmV4


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
    if max_tokens_per_sec is not None:
        assert max_tokens_per_sec > 0, "max_tokens_per_sec needs to be > 0"
    if sample_rate is not None:
        assert sample_rate > 0, "sample_rate needs to be > 0"

    data_: ReturnnTensor = extern_data["data"]
    data: Tensor = data_.raw_tensor
    seq_len: Tensor = data_.dims[1].dyn_size_ext.raw_tensor.to(device=data.device)
    if max_tokens_per_sec is not None and sample_rate is not None:
        max_seq_len = max_tokens_per_sec * (seq_len / sample_rate)
    else:
        max_seq_len = seq_len

    # ENCODER (FORWARD) STEP (for inference)
    decoder_state: Qwen2DecoderState
    decoder_state, _, _ = model.forward_encoder(data, seq_len, beam_size)  # Initial beam size is beam_size

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


def forward_step_v2(  # TODO: Fix!
    *,
    model: BaseEncoderDecoderModel,
    extern_data: TensorDict,
    beam_size: int,
    max_tokens_per_sec: Optional[int] = None,
    sample_rate: Optional[int] = None,
    **kwargs,
):
    """
    Runs full recognition on the given data. Fixes beam search to the first one

    RETURNN ENTRYPOINT (for search/inference)!!
    """
    assert beam_size > 0
    if max_tokens_per_sec is not None:
        assert max_tokens_per_sec > 0, "max_tokens_per_sec needs to be > 0"
    if sample_rate is not None:
        assert sample_rate > 0, "sample_rate needs to be > 0"

    data_: ReturnnTensor = extern_data["data"]
    data: Tensor = data_.raw_tensor
    seq_len: Tensor = data_.dims[1].dyn_size_ext.raw_tensor.to(device=data.device)
    if max_tokens_per_sec is not None and sample_rate is not None:
        max_seq_len = max_tokens_per_sec * (seq_len / sample_rate)
    else:
        max_seq_len = seq_len

    # ENCODER (FORWARD) STEP (for inference)
    decoder_state: Qwen2DecoderState
    decoder_state, _, _ = model.forward_encoder(data, seq_len, 1)

    # BEAM SEARCH DECODING (contains DECODER (FORWARD) STEPs)
    seq_targets, seq_log_prob, _, out_seq_len = beam_search_v2(
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


def forward_step_ctc_decoding(
    *,
    model: BaseEncoderDecoderModel,
    extern_data: TensorDict,
    beam_size: int,
    ctc_scale: float = 1.0,
    prior_scale: float = 1.0,
    lm_scale: float = 1.0,
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

def forward_step_ctc_decoding_v2(
        *,
        model: SllmV4,
        extern_data: TensorDict,

        beam_size: int,
        ctc_scale: float = 1.0,
        prior_scale: float = 0.0,
        lm_scale: float = 0.0,
        sllm_scale: float = 0.0,

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
    )

    ctx = rf.get_run_ctx()
    ctx.mark_as_output(seq_targets_rf, "tokens", dims=[batch_dim, beam_dim, lens_dim])
    ctx.mark_as_output(seq_log_prob_rf, "scores", dims=[batch_dim, beam_dim])

def forward_step_greedy_ctc(
    *,
    model: BaseEncoderDecoderModel,
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
    decoder_state, aux_logits, encoder_lens = model.forward_encoder(
        data,
        seq_len,
        initial_beam_size=1,
    )
    ctc_log_prob = torch.nn.functional.log_softmax(aux_logits[-1], dim=-1)  # [B, T, V]
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

"""
PRIORS
"""

def prior_step_v1(
        *,
        model: BaseEncoderDecoderModel,
        extern_data: TensorDict,

        aux_layer_idx: int = -1,
        **kwargs,
):
    """
    From Robins code (SLLM repo)
    """
    data_: ReturnnTensor = extern_data["data"]
    data: Tensor = data_.raw_tensor
    seq_len: Tensor = data_.dims[1].dyn_size_ext.raw_tensor.to(device=data.device)

    _, aux_log_probs, encoder_lens = model.forward_encoder(
        data,
        seq_len,
        initial_beam_size=1, # Not really needed...
    )

    ctc_log_probs = aux_log_probs[aux_layer_idx]
    ctc_probs = torch.exp(ctc_log_probs)

    #vocab_dim = Dim(model.num_labels, name="vocab")
    extended_vocab_dim = Dim(model.num_labels+1, name="extended_vocab")
    lens_data = rf.convert_to_tensor(encoder_lens, dims=[batch_dim])
    time_dim = Dim(lens_data, name="seq_len")

    ctc_probs = rf.convert_to_tensor(ctc_probs, dims=[batch_dim, time_dim, extended_vocab_dim], feature_dim=extended_vocab_dim)

    ctx = rf.get_run_ctx()
    ctx.mark_as_output(ctc_probs, "output", dims=[batch_dim, time_dim, extended_vocab_dim])


"""
TESTS
"""

# def debug_ext_lm(
#         *,
#         model: SllmV4,
#         extern_data: TensorDict,
#
#         **kwargs,
# ):
#     """
#     Diagnostic: runs lm_step_decoder autoregressively over a hardcoded training sample.
#     Compares teacher-forced log-probs vs autoregressive log-probs to check if the
#     external LM produces sensible scores on text it was trained on.
#
#     Replace HARDCODED_SAMPLE with an actual training sentence.
#     """
#     import torch
#     import torch.nn.functional as F
#
#     # -------------------------------------------------------------------------
#     # 1. HARDCODED TRAINING SAMPLE — replace with a real one from your training set
#     # -------------------------------------------------------------------------
#     HARDCODED_SAMPLE = "the quick brown fox jumps over the lazy dog"
#
#     token_ids: list[int] = tokenizer.encode(HARDCODED_SAMPLE)
#     print(f"Token ids ({len(token_ids)} tokens): {token_ids}")
#     print(f"Decoded back: {tokenizer.decode(token_ids)}")
#
#     # Full input sequence: [BOS, TK1, TK2, ..., TKn]
#     input_ids = [model.bos_idx] + token_ids
#     # Targets:            [TK1, TK2, ..., TKn, EOS]
#     target_ids = token_ids + [model.eos_idx]
#
#     assert len(input_ids) == len(target_ids)
#     seq_len = len(input_ids)
#     print(f"Sequence length (with BOS): {seq_len}")
#
#     input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)   # [SeqLen]
#     target_tensor = torch.tensor(target_ids, dtype=torch.long, device=device) # [SeqLen]
#
#     # -------------------------------------------------------------------------
#     # 2. TEACHER-FORCED PASS (mirrors training exactly)
#     #    input_embeds = embed([BOS, TK1, ..., TKn]), predict all at once
#     # -------------------------------------------------------------------------
#     print("\n--- TEACHER-FORCED PASS (as in training) ---")
#     model.external_lm.eval()
#     with torch.no_grad():
#         # Add batch dim: [1, SeqLen]
#         input_batch = input_tensor.unsqueeze(0)  # [1, SeqLen]
#         input_embeds = model.external_lm.decoder_embed_func(input_batch)  # [1, SeqLen, F]
#
#         from transformers.modeling_outputs import CausalLMOutputWithPast
#         tf_output: CausalLMOutputWithPast = model.external_lm.decoder.forward(
#             inputs_embeds=input_embeds,
#             logits_to_keep=seq_len,
#         )
#         tf_logits = tf_output.logits.squeeze(0)  # [SeqLen, Vocab]
#         tf_log_probs = F.log_softmax(tf_logits, dim=-1)  # [SeqLen, Vocab]
#
#         tf_nll = F.nll_loss(tf_log_probs, target_tensor.long(), reduction="none")  # [SeqLen]
#         tf_ppl = torch.exp(tf_nll.mean())
#
#         print(f"  Teacher-forced NLL per token: {tf_nll.tolist()}")
#         print(f"  Teacher-forced PPL: {tf_ppl.item():.4f}")
#         print(f"  Teacher-forced mean NLL: {tf_nll.mean().item():.4f}")
#
#         # Top-1 accuracy
#         tf_top1 = (tf_logits.argmax(dim=-1) == target_tensor).float().mean()
#         print(f"  Teacher-forced top-1 accuracy: {tf_top1.item():.4f}")
#
#     # -------------------------------------------------------------------------
#     # 3. AUTOREGRESSIVE PASS (mirrors lm_step_decoder at inference)
#     #    Step-by-step, feeding one token at a time with KV cache
#     # -------------------------------------------------------------------------
#     print("\n--- AUTOREGRESSIVE PASS (as in inference via lm_step_decoder) ---")
#     with torch.no_grad():
#         # State init — same as in ctc_label_sync_search_v2
#         ext_lm_state = {
#             "input_embeds": None,
#             "past_key_values": None,
#         }
#
#         ar_nll_list = []
#         ar_predicted_tokens = []
#
#         for step_idx, (inp_tok, tgt_tok) in enumerate(zip(input_ids, target_ids)):
#             # labels shape expected by lm_step_decoder: [Batch, Beam, Time=1]
#             label_tensor = torch.tensor(
#                 [[[inp_tok]]],  # [B=1, beam=1, T=1]
#                 dtype=torch.long,
#                 device=device,
#             )
#
#             logits_raw, ext_lm_state = model.external_llm_step_decoder(label_tensor, ext_lm_state)
#             # logits_raw: [B=1, beam=1, T=1, Vocab]
#             logits_step = logits_raw[0, 0, 0, :]  # [Vocab]
#             log_probs_step = F.log_softmax(logits_step, dim=-1)  # [Vocab]
#
#             tgt = torch.tensor(tgt_tok, dtype=torch.long, device=device)
#             nll = F.nll_loss(log_probs_step.unsqueeze(0), tgt.unsqueeze(0).long()).item()
#             ar_nll_list.append(nll)
#             ar_predicted_tokens.append(logits_step.argmax().item())
#
#             if step_idx < 10 or step_idx == seq_len - 1:  # print first 10 + last
#                 top5 = log_probs_step.topk(5)
#                 top5_tokens = [tokenizer.decode([t.item()]) for t in top5.indices]
#                 print(
#                     f"  Step {step_idx:3d} | input={tokenizer.decode([inp_tok])!r:12s} "
#                     f"| target={tokenizer.decode([tgt_tok])!r:12s} "
#                     f"| NLL={nll:.4f} "
#                     f"| top5={top5_tokens}"
#                 )
#
#         ar_nll_tensor = torch.tensor(ar_nll_list)
#         ar_ppl = torch.exp(ar_nll_tensor.mean())
#         ar_top1 = sum(p == t for p, t in zip(ar_predicted_tokens, target_ids)) / seq_len
#
#         print(f"\n  Autoregressive NLL per token: {ar_nll_list}")
#         print(f"  Autoregressive PPL: {ar_ppl.item():.4f}")
#         print(f"  Autoregressive mean NLL: {ar_nll_tensor.mean().item():.4f}")
#         print(f"  Autoregressive top-1 accuracy: {ar_top1:.4f}")
#
#     # -------------------------------------------------------------------------
#     # 4. COMPARE: flag if there's a large discrepancy
#     # -------------------------------------------------------------------------
#     print("\n--- COMPARISON ---")
#     print(f"  Teacher-forced PPL : {tf_ppl.item():.4f}")
#     print(f"  Autoregressive PPL : {ar_ppl.item():.4f}")
#     ppl_ratio = ar_ppl.item() / tf_ppl.item()
#     print(f"  AR/TF PPL ratio    : {ppl_ratio:.4f}  (should be ~1.0 if consistent)")
#     if ppl_ratio > 1.5:
#         print("  !! LARGE DISCREPANCY: inference decoding diverges from training forward pass")
#         print("     Likely cause: embedding mismatch, positional encoding, or cache bug")
#     else:
#         print("  OK: teacher-forced and autoregressive passes are consistent")
#
#     # -------------------------------------------------------------------------
#     # 5. EMBEDDING SANITY CHECK
#     # -------------------------------------------------------------------------
#     print("\n--- EMBEDDING SANITY CHECK ---")
#     ext_embed = model.external_lm.decoder_embed_func
#     internal_embed = model.external_lm.decoder.model.embed_tokens
#     print(f"  decoder_embed_func norm      : {ext_embed.weight.norm().item():.4f}")
#     print(f"  decoder embed_tokens norm    : {internal_embed.weight.norm().item():.4f}")
#     print(f"  Same object?                 : {ext_embed.weight is internal_embed.weight}")
#     print(f"  Max weight diff              : {(ext_embed.weight - internal_embed.weight).abs().max().item():.6f}")
#     if not (ext_embed.weight is internal_embed.weight):
#         diff = (ext_embed.weight - internal_embed.weight).abs().max().item()
#         if diff > 1e-4:
#             print("  !! MISMATCH: decoder_embed_func and embed_tokens are different!")
#             print("     The LM input embeddings differ from the decoder's internal embeddings.")
#             print("     This is almost certainly your bug.")