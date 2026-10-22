"""
RETURNN Training Step for UNdreaMT (Artetxe et al., ICLR 2018):
Combines:
  1. Denoising Autoencoding (src -> src and trg -> trg) with UNdreaMT word-order swap noise
     (length // 2 random adjacent swaps on input sequences).
  2. On-the-fly Greedy Backtranslation (src_mono -> greedy trg_hat -> src and trg_mono -> greedy src_hat -> trg).
"""

from typing import Dict, Optional, Tuple
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from returnn.tensor import TensorDict
import returnn.frontend as rf

from ..definitions.undreamt_rnn_aed_v1 import (
    UNdreaMTModel,
    PAD_ID,
    OOV_ID,
    EOS_ID,
    SOS_ID,
)


def prepare_undreamt_tensors(
    raw_seqs: torch.Tensor,
    raw_lens: torch.Tensor,
    apply_swap_noise: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts a batch of RETURNN sequences (batch, max_len) and lengths (batch,) into UNdreaMT's
    time-major tensors:
      - enc_ids: (max_len + 1, batch) with EOS_ID appended at the end of each sequence (and optional
                 UNdreaMT word-order swap noise: length // 2 random swaps of adjacent tokens i and i+1).
      - enc_lens: (batch,) = raw_lens + 1 (including EOS_ID).
      - dec_input_ids: (max_len + 1, batch) starting with SOS_ID followed by clean tokens.
      - dec_target_ids: (max_len + 1, batch) clean tokens followed by EOS_ID (padded with PAD_ID=0).
    """
    device = raw_seqs.device
    batch_size, max_raw_len = raw_seqs.shape
    out_len = max_raw_len + 1

    enc_ids = torch.full((out_len, batch_size), PAD_ID, dtype=torch.long, device=device)
    dec_input_ids = torch.full((out_len, batch_size), PAD_ID, dtype=torch.long, device=device)
    dec_target_ids = torch.full((out_len, batch_size), PAD_ID, dtype=torch.long, device=device)
    enc_lens = torch.zeros(batch_size, dtype=torch.long, device=device)

    raw_seqs_cpu = raw_seqs.detach().cpu().tolist()
    raw_lens_cpu = raw_lens.detach().cpu().tolist()

    for b in range(batch_size):
        L = max(1, min(int(raw_lens_cpu[b]), max_raw_len))
        clean_toks = [int(t) for t in raw_seqs_cpu[b][:L] if int(t) not in (PAD_ID, EOS_ID, SOS_ID)]
        if not clean_toks:
            clean_toks = [OOV_ID]
        L_clean = len(clean_toks)
        seq_with_eos = clean_toks + [EOS_ID]
        total_L = L_clean + 1  # includes EOS, matching undreamt/translator.py

        # Decoder input: [SOS_ID] + clean_toks
        dec_in = [SOS_ID] + clean_toks
        # Decoder target: clean_toks + [EOS_ID]
        dec_trg = seq_with_eos

        # Encoder input: optionally apply UNdreaMT word-order swap corruption (length // 2 random adjacent swaps)
        enc_seq = list(seq_with_eos)
        if apply_swap_noise and total_L > 2:
            for _ in range(total_L // 2):
                j = random.randint(0, total_L - 2)
                enc_seq[j], enc_seq[j + 1] = enc_seq[j + 1], enc_seq[j]

        enc_ids[:total_L, b] = torch.tensor(enc_seq, dtype=torch.long, device=device)
        dec_input_ids[:total_L, b] = torch.tensor(dec_in, dtype=torch.long, device=device)
        dec_target_ids[:total_L, b] = torch.tensor(dec_trg, dtype=torch.long, device=device)
        enc_lens[b] = total_L

    # Trim to actual max length in batch
    actual_max = int(enc_lens.max().item())
    return (
        enc_ids[:actual_max],
        enc_lens,
        dec_input_ids[:actual_max],
        dec_target_ids[:actual_max],
    )


def apply_swap_noise_to_generated_batch(
    gen_ids: torch.Tensor,
    gen_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies UNdreaMT adjacent swap corruption (length // 2 swaps) to a batch of greedily generated
    sequences (gen_len, batch), matching undreamt where trg2src_translator and src2trg_translator
    apply corruption when train=True.
    """
    device = gen_ids.device
    max_len, batch_size = gen_ids.shape
    noisy_ids = gen_ids.clone()
    lens_list = gen_lens.detach().cpu().tolist()
    for b in range(batch_size):
        L = int(lens_list[b])
        if L > 2:
            seq = noisy_ids[:L, b].detach().cpu().tolist()
            for _ in range(L // 2):
                j = random.randint(0, L - 2)
                seq[j], seq[j + 1] = seq[j + 1], seq[j]
            noisy_ids[:L, b] = torch.tensor(seq, dtype=torch.long, device=device)
    return noisy_ids, gen_lens


def compute_undreamt_nll_loss(
    logprobs: torch.Tensor,
    target_ids: torch.Tensor,
) -> Tuple[torch.Tensor, int]:
    """
    Computes token-level Negative Log-Likelihood loss ignoring PAD_ID=0 and masking PAD_ID logprob
    (matching undreamt/translator.py).
    :param logprobs: (trg_len, batch, vocab_size + 4)
    :param target_ids: (trg_len, batch)
    :return: (mean_nll_loss_per_token, num_non_pad_tokens)
    """
    vocab_dim = logprobs.shape[-1]
    flat_logprobs = logprobs.reshape(-1, vocab_dim)
    flat_targets = target_ids.reshape(-1)
    non_pad = flat_targets != PAD_ID
    num_tokens = int(non_pad.sum().item())
    if num_tokens == 0:
        return flat_logprobs.sum() * 0.0, 1
    loss_sum = F.nll_loss(flat_logprobs, flat_targets, ignore_index=PAD_ID, reduction="sum")
    return loss_sum / num_tokens, num_tokens


def train_step(
    *,
    model: UNdreaMTModel,
    extern_data: TensorDict,
    default_data_key: str = "data",
    default_target_key: str = "text",
    enable_denoising: bool = True,
    enable_backtranslation: bool = True,
    denoise_backtranslation: bool = True,
    bt_max_ratio: float = 2.0,
    denoise_scale: float = 1.0,
    bt_scale: float = 1.0,
    **kwargs,
):
    """
    UNdreaMT training step executing:
      1. Denoising autoencoding on source monolingual batch (src -> src)
      2. Denoising autoencoding on target monolingual batch (trg -> trg)
      3. On-the-fly greedy backtranslation: src_mono -> greedy trg_hat -> train trg2src (trg_hat -> src)
      4. On-the-fly greedy backtranslation: trg_mono -> greedy src_hat -> train src2trg (src_hat -> trg)
    """
    ctx = rf.get_run_ctx()

    src_tensor = extern_data[default_data_key]
    trg_tensor = extern_data[default_target_key]

    src_raw = src_tensor.raw_tensor.long()  # (batch, src_time)
    src_lens = src_tensor.dims[1].dyn_size_ext.raw_tensor.long().to(src_raw.device)

    trg_raw = trg_tensor.raw_tensor.long()  # (batch, trg_time)
    trg_lens = trg_tensor.dims[1].dyn_size_ext.raw_tensor.long().to(trg_raw.device)

    # Prepare clean and noisy tensors for src and trg
    src_noisy_enc, src_enc_lens, src_dec_in, src_dec_trg = prepare_undreamt_tensors(
        src_raw, src_lens, apply_swap_noise=enable_denoising
    )
    trg_noisy_enc, trg_enc_lens, trg_dec_in, trg_dec_trg = prepare_undreamt_tensors(
        trg_raw, trg_lens, apply_swap_noise=enable_denoising
    )

    # 1. Denoising Autoencoding: src -> src
    if enable_denoising:
        logprobs_src2src = model.forward_seq2seq(
            src_ids=src_noisy_enc,
            src_lengths=src_enc_lens,
            trg_input_ids=src_dec_in,
            src_lang="src",
            trg_lang="src",
        )
        loss_denoise_src, _ = compute_undreamt_nll_loss(logprobs_src2src, src_dec_trg)
        ctx.mark_as_loss(
            loss_denoise_src,
            name="loss_denoise_src",
            scale=denoise_scale,
        )

        # 2. Denoising Autoencoding: trg -> trg
        logprobs_trg2trg = model.forward_seq2seq(
            src_ids=trg_noisy_enc,
            src_lengths=trg_enc_lens,
            trg_input_ids=trg_dec_in,
            src_lang="trg",
            trg_lang="trg",
        )
        loss_denoise_trg, _ = compute_undreamt_nll_loss(logprobs_trg2trg, trg_dec_trg)
        ctx.mark_as_loss(
            loss_denoise_trg,
            name="loss_denoise_trg",
            scale=denoise_scale,
        )

    # 3 & 4. On-the-fly Greedy Backtranslation
    if enable_backtranslation:
        # Prepare uncorrupted encoder inputs for greedy generation (matching train=False in undreamt)
        src_clean_enc, src_clean_lens, _, _ = prepare_undreamt_tensors(
            src_raw, src_lens, apply_swap_noise=False
        )
        trg_clean_enc, trg_clean_lens, _, _ = prepare_undreamt_tensors(
            trg_raw, trg_lens, apply_swap_noise=False
        )

        # 3a. Greedy translate src -> synthetic trg_hat (no grad, eval mode)
        with torch.no_grad():
            syn_trg_ids, syn_trg_lens = model.greedy_translate(
                src_ids=src_clean_enc,
                src_lengths=src_clean_lens,
                src_lang="src",
                trg_lang="trg",
                max_ratio=bt_max_ratio,
            )
            if denoise_backtranslation:
                syn_trg_ids, syn_trg_lens = apply_swap_noise_to_generated_batch(syn_trg_ids, syn_trg_lens)

        # 3b. Train trg -> src on (syn_trg_ids -> clean src)
        logprobs_bt_trg2src = model.forward_seq2seq(
            src_ids=syn_trg_ids,
            src_lengths=syn_trg_lens,
            trg_input_ids=src_dec_in,
            src_lang="trg",
            trg_lang="src",
        )
        loss_bt_trg2src, _ = compute_undreamt_nll_loss(logprobs_bt_trg2src, src_dec_trg)
        ctx.mark_as_loss(
            loss_bt_trg2src,
            name="loss_bt_trg2src",
            scale=bt_scale,
        )

        # 4a. Greedy translate trg -> synthetic src_hat (no grad, eval mode)
        with torch.no_grad():
            syn_src_ids, syn_src_lens = model.greedy_translate(
                src_ids=trg_clean_enc,
                src_lengths=trg_clean_lens,
                src_lang="trg",
                trg_lang="src",
                max_ratio=bt_max_ratio,
            )
            if denoise_backtranslation:
                syn_src_ids, syn_src_lens = apply_swap_noise_to_generated_batch(syn_src_ids, syn_src_lens)

        # 4b. Train src -> trg on (syn_src_ids -> clean trg)
        logprobs_bt_src2trg = model.forward_seq2seq(
            src_ids=syn_src_ids,
            src_lengths=syn_src_lens,
            trg_input_ids=trg_dec_in,
            src_lang="src",
            trg_lang="trg",
        )
        loss_bt_src2trg, _ = compute_undreamt_nll_loss(logprobs_bt_src2trg, trg_dec_trg)
        ctx.mark_as_loss(
            loss_bt_src2trg,
            name="loss_bt_src2trg",
            scale=bt_scale,
        )

