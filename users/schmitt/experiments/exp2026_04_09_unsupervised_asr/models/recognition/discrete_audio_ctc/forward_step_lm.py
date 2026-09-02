"""
Forward step for label-synchronous CTC decoding with phoneme-LM shallow fusion.

Counterpart of ``forward_step.py`` (greedy argmax + repeat collapsing), but running the
label-synchronous prefix beam search from :mod:`.label_sync_search`, so the autoregressive phoneme
LM can contribute once per emitted label. Expects the combined
``definitions.ctc_with_lm_v1.Model`` (ASR + LM in one module).

Marks the same outputs as the greedy step (``tokens [B, T]`` sparse, ``scores [B]``), so it reuses
``discrete_audio_aed.callback.RecognitionToTextDictCallback`` and the sclite scoring path unchanged.
"""

__all__ = ["forward_step"]

from typing import Dict, Optional

import torch

import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim

from ...train_steps.util import get_random_mask, mask_sequence
from .label_sync_search import ctc_lm_label_sync_search
from .time_sync_search import ctc_lm_time_sync_search


def forward_step(
    *,
    model,
    extern_data: TensorDict,
    beam_size: int = 12,
    lm_scale: float = 0.0,
    ctc_scale: float = 1.0,
    length_reward: float = 0.0,
    score_type: str = "sum",
    search_type: str = "label_sync",
    input_data_key: str = "data",
    input_modality: str = "audio",
    output_modality: str = "text",
    masking_opts: Optional[Dict] = None,
    **kwargs,
):
    """
    :param beam_size: label-synchronous beam width.
    :param lm_scale: weight of the phoneme LM log prob per label (0 -> pure CTC prefix search).
    :param ctc_scale: weight of the CTC prefix score.
    :param length_reward: per-label bonus, counteracting the length bias shallow fusion introduces.
    :param score_type: "sum" (CTC marginal) or "max" (Viterbi / best-path).
    :param search_type: "time_sync" (classic CTC prefix beam search) or "label_sync" (the
        ESPnet-style prefix scorer). **Use "time_sync".** "label_sync" is only valid when the CTC
        prefix score is an auxiliary term next to a dominant decoder: on its own it compares
        hypotheses that have consumed different numbers of frames, fills the beam with prefixes that
        crammed their labels into the first frames, and ends up ~13 nats below a verified optimum
        with no sensitivity to beam width. It is kept only so earlier results stay reproducible.
    """
    assert input_modality in ("audio", "text"), input_modality
    assert output_modality in ("audio", "text"), output_modality

    asr, lm = model.asr, model.lm
    data = extern_data[input_data_key]
    indices = data.raw_tensor
    seq_lens = data.dims[1].dyn_size_ext.raw_tensor.to(device=indices.device)

    if input_modality == "audio":
        forward_func, mask_idx = asr.forward_audio, asr.audio_mask_idx
    else:
        forward_func, mask_idx = asr.forward_text, asr.text_mask_idx
    if masking_opts is not None and masking_opts.get("mask_prob", 0.0) > 0.0:
        mask = get_random_mask(seq_lens, **masking_opts)
        indices, seq_lens = mask_sequence(indices, seq_lens, mask, mask_value=mask_idx)

    # `forward_audio` without aux_logit_modality returns the *text* head -- the cross-modal path
    # recognition uses (see the model's forward_* defaults).
    _, aux_logits, enc_lens, _ = forward_func(indices, seq_lens)
    ctc_log_probs = aux_logits[-1].float().log_softmax(dim=-1)  # [B, T, num_phon + 1]

    out_dim = asr.text_out_dim if output_modality == "text" else asr.audio_out_dim
    num_labels = ctc_log_probs.shape[-1] - 1
    blank_idx = asr.text_blank_idx if output_modality == "text" else asr.audio_blank_idx
    assert blank_idx == num_labels, (
        f"expected blank ({blank_idx}) as the last CTC output of {num_labels + 1}"
    )
    if lm_scale != 0.0:
        assert lm.out_dim - 3 == num_labels, (
            f"LM vocab ({lm.out_dim - 3} labels) does not match the CTC head ({num_labels});"
            " the LM must be trained on the same phoneme inventory"
        )

    assert search_type in ("time_sync", "label_sync"), search_type
    search = ctc_lm_time_sync_search if search_type == "time_sync" else ctc_lm_label_sync_search
    tokens, scores, lens = search(
        ctc_log_probs=ctc_log_probs,
        enc_lens=enc_lens.to(device=ctc_log_probs.device),
        lm=lm,
        beam_size=beam_size,
        blank_idx=blank_idx,
        num_labels=num_labels,
        lm_scale=lm_scale,
        ctc_scale=ctc_scale,
        length_reward=length_reward,
        score_type=score_type,
    )
    best_tokens, best_lens, best_scores = tokens[:, 0], lens[:, 0], scores[:, 0]

    ctx = rf.get_run_ctx()
    vocab_dim = Dim(out_dim, name="vocab")
    lens_data = rf.convert_to_tensor(best_lens.to(torch.int32), dims=[batch_dim])
    lens_dim = Dim(lens_data, name="seq_len")
    tokens_rf = rf.convert_to_tensor(
        best_tokens[:, : int(best_lens.max())].to(torch.int32),
        dims=[batch_dim, lens_dim],
        sparse_dim=vocab_dim,
    )
    ctx.mark_as_output(tokens_rf, "tokens", dims=[batch_dim, lens_dim])
    ctx.mark_as_output(rf.convert_to_tensor(best_scores, dims=[batch_dim]), "scores", dims=[batch_dim])
