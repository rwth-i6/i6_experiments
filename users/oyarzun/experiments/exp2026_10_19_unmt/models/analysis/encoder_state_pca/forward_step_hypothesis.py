__all__ = ["forward_step"]

from typing import Optional
import torch

import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim

from ...train_steps.undreamt_backtranslation_train_step import prepare_undreamt_tensors
from ...definitions.undreamt_rnn_aed_v1 import (
    PAD_ID,
    OOV_ID,
    EOS_ID,
    SOS_ID,
)


def _lang_present(data) -> bool:
    """A language stream is present in the batch if its (padded) time dim is non-empty and at least one
    sequence has a non-zero length."""
    if data is None:
        return False
    seq_lens = data.dims[1].dyn_size_ext.raw_tensor
    return data.raw_tensor.shape[1] > 0 and bool(seq_lens.max() > 0)


def forward_step(
    *,
    model,
    extern_data: TensorDict,
    src_data_key: str = "data",
    audio_data_key: str = "data",
    beam_size: int = 1,
    max_ratio: float = 2.0,
    lp_alpha: float = 0.6,
    src_lang: str = "src",
    trg_lang: str = "trg",
    **kwargs,
):
    """
    Forward step for the cross-lingual shared-encoder PCA comparing sequence A vs. its
    hypothesized translation N(A).

    1. Encodes the source sequence A (via ``model.forward_src_encoder`` or ``model.forward_audio``)
       to obtain representations E_src(A) of shape [B, T_src, F].
    2. Translates sequence A into the hypothesized target sequence N(A) using the model's
       greedy decoding (or beam search if beam_size > 1).
    3. Feeds N(A) back into the target encoder (via ``model.forward_trg_encoder`` or
       ``model.forward_text``) to obtain representations E_trg(N(A)) of shape [B, T_hyp, F].
    4. Exposes E_src(A) as 'audio_states' and E_trg(N(A)) as 'text_states', alongside their
       respective token sequences 'src_indices' and 'trg_indices'.
    """
    ctx = rf.get_run_ctx()

    s_key = src_data_key if src_data_key in extern_data.data else audio_data_key
    src_data = extern_data.data.get(s_key)
    if not _lang_present(src_data):
        return

    src_time_dim = src_data.dims[1]
    src_indices = src_data.raw_tensor
    src_lens = src_time_dim.dyn_size_ext.raw_tensor.to(device=src_indices.device)

    # 1. Forward source encoder on sequence A
    if hasattr(model, "forward_src_encoder"):
        src_enc, _, _, _ = model.forward_src_encoder(src_indices.long(), src_lens)
    else:
        src_enc, _, _, _ = model.forward_audio(src_indices.long(), src_lens)
    feat_dim = Dim(int(src_enc.shape[-1]), name="enc_feat")
    src_enc_rf = rf.convert_to_tensor(src_enc, dims=[batch_dim, src_time_dim, feat_dim])
    ctx.mark_as_output(src_enc_rf, "audio_states", dims=[batch_dim, src_time_dim, feat_dim])
    ctx.mark_as_output(src_data, "src_indices", dims=[batch_dim, src_time_dim])

    # 2. Decode sequence A into hypothesized translation N(A)
    enc_ids, enc_lens, _, _ = prepare_undreamt_tensors(
        src_indices.long(), src_lens.long(), apply_swap_noise=False
    )
    with torch.no_grad():
        if beam_size <= 1:
            pred_ids_tb, pred_lens_b = model.greedy_translate(
                src_ids=enc_ids,
                src_lengths=enc_lens,
                src_lang=src_lang,
                trg_lang=trg_lang,
                max_ratio=max_ratio,
            )
            batch_size = src_indices.shape[0]
            token_seqs = []
            lengths = []
            for b in range(batch_size):
                seq = pred_ids_tb[: pred_lens_b[b], b].tolist()
                # strip trailing EOS if present
                if seq and seq[-1] == EOS_ID:
                    seq = seq[:-1]
                clean_seq = [t for t in seq if t not in (PAD_ID, SOS_ID)]
                if not clean_seq:
                    clean_seq = [OOV_ID]
                token_seqs.append(clean_seq)
                lengths.append(len(clean_seq))
        else:
            hyps = model.beam_search_translate(
                src_ids=enc_ids,
                src_lengths=enc_lens,
                src_lang=src_lang,
                trg_lang=trg_lang,
                beam_size=beam_size,
                max_ratio=max_ratio,
                lp_alpha=lp_alpha,
            )
            token_seqs = []
            lengths = []
            for tokens, _ in hyps:
                clean_seq = [t for t in tokens if t not in (PAD_ID, SOS_ID)]
                if not clean_seq:
                    clean_seq = [OOV_ID]
                token_seqs.append(clean_seq)
                lengths.append(len(clean_seq))

    # 3. Pack N(A) into batch tensor [B, max_len]
    batch_size = src_indices.shape[0]
    max_len = max(lengths)
    hyp_ids = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long, device=src_indices.device)
    for b, tokens in enumerate(token_seqs):
        hyp_ids[b, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=src_indices.device)
    hyp_lens = torch.tensor(lengths, dtype=torch.long, device=src_indices.device)

    # 4. Feed N(A) back into target encoder
    with torch.no_grad():
        if hasattr(model, "forward_trg_encoder"):
            trg_enc, _, _, _ = model.forward_trg_encoder(hyp_ids.long(), hyp_lens)
        else:
            trg_enc, _, _, _ = model.forward_text(hyp_ids.long(), hyp_lens)

    # 5. Mark outputs for RETURNN callback
    hyp_lens_rf = rf.convert_to_tensor(hyp_lens.to(torch.int32), dims=[batch_dim])
    hyp_time_dim = Dim(hyp_lens_rf, name="hyp_time")
    trg_enc_rf = rf.convert_to_tensor(trg_enc, dims=[batch_dim, hyp_time_dim, feat_dim])
    hyp_ids_rf = rf.convert_to_tensor(hyp_ids.to(torch.int32), dims=[batch_dim, hyp_time_dim])

    ctx.mark_as_output(trg_enc_rf, "text_states", dims=[batch_dim, hyp_time_dim, feat_dim])
    ctx.mark_as_output(hyp_ids_rf, "trg_indices", dims=[batch_dim, hyp_time_dim])
