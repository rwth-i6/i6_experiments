__all__ = ["forward_step"]

from typing import Dict, Optional

import torch
import torch.nn.functional as F

import returnn.frontend as rf
from returnn.tensor import TensorDict, batch_dim

from ....models.train_steps.util import get_random_mask, mask_sequence


def forward_step(
    *,
    model,
    extern_data: TensorDict,
    target_data_key: str = "data",
    input_modality: Optional[str] = None,
    input_data_key: str = "data",
    output_modality: str = "text",
    masking_opts: Optional[Dict] = None,
    **kwargs,
):
    """
    Compute the per-sequence cross entropy / perplexity of a label sequence (``target_data_key``).

    Two modes, so this works for both the phoneme LM and our AED models:

    - ``input_modality=None`` (default): decoder-only LM. The label sequence is scored
      autoregressively by ``model.score_text`` -- used for
      ``definitions.transformer_decoder_lm_v1.Model``.
    - ``input_modality in ("audio", "text")``: AED model. The encoder is run over the input modality
      (``input_data_key``, optionally masked exactly like in training via ``masking_opts``) and the
      label sequence is scored by the teacher-forced ``output_modality`` decoder *conditioned* on the
      encoder output -- i.e. the conditional PPL of an AED system (e.g. audio->phoneme).

    Teacher forcing: the decoder input is ``bos + labels`` and the scored target is ``labels + eos``.
    Marks two per-seq outputs, ``ce`` (summed token CE) and ``num_tokens`` (scored tokens incl. eos);
    the callback (:class:`...callback.PplScoresCallback`) turns these into per-seq / corpus PPL.
    """
    assert input_modality in (None, "audio", "text"), input_modality
    assert output_modality in ("audio", "text"), output_modality

    ctx = rf.get_run_ctx()

    target = extern_data[target_data_key]
    labels = target.raw_tensor
    labels_lens = target.dims[1].dyn_size_ext.raw_tensor.to(device=labels.device)

    # special (bos/eos) ids: from the LM for the decoder-only path, else from the AED output modality
    if input_modality is None:
        bos_idx, eos_idx = model.bos_idx, model.eos_idx
    elif output_modality == "text":
        bos_idx, eos_idx = model.text_bos_idx, model.text_eos_idx
    else:
        bos_idx, eos_idx = model.audio_bos_idx, model.audio_eos_idx

    # teacher forcing: decoder input = bos + labels, scored target = labels + eos
    input_labels = F.pad(labels, (1, 0), "constant", value=bos_idx)
    input_labels_lens = labels_lens + 1

    if input_modality is None:
        # decoder-only LM: no encoder
        logits = model.score_text(input_labels, input_labels_lens)
    else:
        # AED: encode the input modality, then score the labels conditioned on the encoder output
        data = extern_data[input_data_key].raw_tensor
        data_lens = extern_data[input_data_key].dims[1].dyn_size_ext.raw_tensor.to(device=data.device)
        if input_modality == "audio":
            forward_func = model.forward_audio
            mask_idx = model.audio_mask_idx
        else:
            forward_func = model.forward_text
            mask_idx = model.text_mask_idx
        enc_indices, enc_lens = data, data_lens
        if masking_opts is not None and masking_opts.get("mask_prob", 0.0) > 0.0:
            mask = get_random_mask(data_lens, **masking_opts)
            enc_indices, enc_lens = mask_sequence(data, data_lens, mask, mask_value=mask_idx)
        encoder_output, _, encoder_lens, _ = forward_func(enc_indices, enc_lens)
        if output_modality == "text":
            logits = model.decode_text_seq(input_labels, input_labels_lens, encoder_output, encoder_lens)
            # with a shared decoder, decode_text_seq returns logits over the full shared vocab
            # (text+audio); restrict to the text part so the softmax / perplexity is over the phoneme
            # vocab only -- matching inference (model.step_text_decoder) and the decoder-only LM, so
            # the AED conditional PPL is comparable to the phoneme LM PPL. No-op for a non-shared
            # decoder (already text_out_dim wide).
            if model.share_decoder and not model.fix_decode_text_seq_for_shared_dec:
                logits = logits[..., : model.text_out_dim]
        else:
            logits = model.decode_audio_seq(input_labels, input_labels_lens, encoder_output, encoder_lens)
            # decode_audio_seq already returns the audio-part logits for a shared decoder; slicing to
            # audio_out_dim is a no-op there but keeps the audio path robust for a non-shared decoder.
            logits = logits[..., : model.audio_out_dim]

    # build the scored target sequence: labels followed by a single eos, aligned to the logits' time.
    B, T = input_labels.shape  # T == max(labels_lens) + 1 == logits time dim
    targets = F.pad(labels, (0, 1), "constant", value=0).long()  # [B, T]
    targets[torch.arange(B, device=labels.device), labels_lens] = eos_idx  # eos right after last label
    log_probs = F.log_softmax(logits.float(), dim=-1)  # [B, T, V]
    tgt_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # [B, T]
    valid = torch.arange(T, device=labels.device)[None, :] < input_labels_lens[:, None]  # [B, T]
    ce_per_seq = -(tgt_log_probs * valid).sum(dim=-1)  # [B]
    num_tokens = input_labels_lens.to(ce_per_seq.dtype)  # [B]

    ce_rf = rf.convert_to_tensor(ce_per_seq, dims=[batch_dim])
    num_tokens_rf = rf.convert_to_tensor(num_tokens, dims=[batch_dim])
    ctx.mark_as_output(ce_rf, "ce", dims=[batch_dim])
    ctx.mark_as_output(num_tokens_rf, "num_tokens", dims=[batch_dim])
