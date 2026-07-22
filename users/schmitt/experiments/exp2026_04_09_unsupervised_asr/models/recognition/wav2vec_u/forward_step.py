__all__ = ["forward_step"]

from typing import Optional

import torch

import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim

from ....models.definitions.wav2vec_u import Model


def forward_step(
    *,
    model: Model,
    extern_data: TensorDict,
    input_data_key: str = "data",
    collapse_repetitions: bool = True,
    **_kwargs,
):
    """Recognition for the wav2vec-U GAN: audio features -> phoneme sequence.

    There is no beam search here (the GAN has no autoregressive decoder). The generator maps the
    (segmented) speech features to a per-position distribution over the phoneme vocabulary, the
    ``JOIN`` segmenter collapses consecutive-identical predictions into segments, and we take the
    per-segment argmax as the phoneme hypothesis. The dummy pad class (``model.pad``, appended after
    the real phonemes) is masked out so only real phonemes 0..V-1 can be emitted.

    Output is marked as ``tokens`` (``[batch, beam=1, time]``, sparse over the phoneme vocab) and
    ``scores`` (``[batch, beam=1]``), exactly like the AED recognition step, so the same
    ``RecognitionToTextDictCallback`` + sclite scoring apply. ``collapse_repetitions`` additionally
    merges any consecutive-identical phonemes left after the segmenter's pooling (fairseq's
    inference convention).

    Extra ``forward_init_args`` from the shared eval pipeline (e.g. ``beam_size``) are ignored.
    """
    features = extern_data[input_data_key].raw_tensor  # [B, T, F]
    seq_len = extern_data[input_data_key].dims[1].dyn_size_ext.raw_tensor.to(device=features.device)
    B, T = features.shape[0], features.shape[1]
    device = features.device

    # True at padding positions
    padding_mask = torch.arange(T, device=device)[None, :] >= seq_len[:, None]

    # dense_x_only -> return the (segmented) softmax over the phoneme vocab + its padding mask
    result = model(features, padding_mask, dense_x_only=True, segment=True)
    logits: torch.Tensor = result["logits"]  # [B, T', V]
    out_pad: torch.Tensor = result["padding_mask"]  # [B, T'] (True = pad)
    valid = ~out_pad

    # never emit the dummy pad class (robust to its index position)
    logits = logits.clone()
    logits[..., model.pad] = float("-inf")
    num_real = model.pad  # #real phoneme classes (pad is appended right after them)

    preds = logits.argmax(dim=-1)  # [B, T']
    log_max = torch.log(logits.gather(-1, preds.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12))  # [B, T']

    # per-seq: keep valid positions, optionally collapse consecutive repeats
    hyps = []
    scores_list = []
    for b in range(B):
        p = preds[b][valid[b]]
        s = log_max[b][valid[b]]
        if collapse_repetitions and p.numel() > 0:
            keep = torch.ones_like(p, dtype=torch.bool)
            keep[1:] = p[1:] != p[:-1]
            p = p[keep]
        hyps.append(p)
        scores_list.append(s.sum())

    lens = torch.tensor([h.numel() for h in hyps], device=device, dtype=torch.int32)  # [B]
    T_out = max(int(lens.max().item()) if B > 0 else 0, 1)
    tokens = torch.zeros(B, 1, T_out, dtype=torch.int64, device=device)
    for b, h in enumerate(hyps):
        if h.numel() > 0:
            tokens[b, 0, : h.numel()] = h
    scores = torch.stack(scores_list).unsqueeze(1) if B > 0 else torch.zeros(B, 1, device=device)  # [B, 1]

    ctx = rf.get_run_ctx()
    beam_dim = Dim(1, name="beam")
    vocab_dim = Dim(num_real, name="vocab")
    lens_data = rf.convert_to_tensor(lens[:, None], dims=[batch_dim, beam_dim])
    lens_dim = Dim(lens_data, name="out_seq_len")
    tokens_rf = rf.convert_to_tensor(tokens, dims=[batch_dim, beam_dim, lens_dim], sparse_dim=vocab_dim)
    ctx.mark_as_output(tokens_rf, "tokens", dims=[batch_dim, beam_dim, lens_dim])
    scores_rf = rf.convert_to_tensor(scores, dims=[batch_dim, beam_dim])
    ctx.mark_as_output(scores_rf, "scores", dims=[batch_dim, beam_dim])
