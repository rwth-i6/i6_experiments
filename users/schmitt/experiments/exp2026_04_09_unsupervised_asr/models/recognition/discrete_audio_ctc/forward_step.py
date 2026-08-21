__all__ = ["forward_step"]

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ....models.definitions.conformer_ctc_discrete_shared_v1 import Model
from ....models.train_steps.util import get_random_mask, mask_sequence, expand_sequence
import returnn.frontend as rf
from returnn.tensor import Dim, TensorDict, batch_dim


def _get_collapsed_out_seqs(
    seq_targets: torch.Tensor,
    blank_idx: int,
):
    """

    :param seq_targets: (B, beam, Time)
    :return:
    """
    ctc_batch_indices = []
    ctc_output_lens = []
    max_ctc_output_len = 0
    B, beam_size = seq_targets.shape[:2]  # noqa
    for b in range(B):
        ctc_beam_indices = []
        ctc_beam_lens = []
        max_ctc_output_len_beam = 0
        for beam in range(beam_size):
            seq_wo_reps = torch.unique_consecutive(seq_targets[b, beam], dim=0)
            seq_wo_reps_wo_blank = seq_wo_reps[seq_wo_reps != blank_idx]
            ctc_beam_indices.append(seq_wo_reps_wo_blank)
            ctc_beam_lens.append(len(seq_wo_reps_wo_blank))
            max_ctc_output_len_beam = max(max_ctc_output_len_beam, ctc_beam_lens[-1])
        max_ctc_output_len = max(max_ctc_output_len, max_ctc_output_len_beam)

        ctc_beam_indices = [F.pad(bi, (0, max_ctc_output_len_beam - bi.size(0)), value=0) for bi in ctc_beam_indices]

        ctc_batch_indices.append(torch.stack(ctc_beam_indices, dim=0))
        ctc_output_lens.append(torch.LongTensor(ctc_beam_lens))
    ctc_batch_indices = [F.pad(bi, (0, max_ctc_output_len - bi.size(1)), value=0) for bi in ctc_batch_indices]

    ctc_batch_indices = torch.stack(ctc_batch_indices, dim=0)
    ctc_output_lens = torch.stack(ctc_output_lens, dim=0)

    return ctc_batch_indices, ctc_output_lens


def forward_step(
    *,
    model: Model,
    extern_data: TensorDict,
    beam_size: int,
    input_data_key: str = "data",
    input_modality: str = "audio",
    output_modality: str = "text",
    masking_opts: Optional[Dict] = None,
    expansion_opts: Optional[Dict] = None,
    **kwargs,
):

    assert beam_size > 0
    assert input_modality in ("audio", "text"), input_modality
    assert output_modality in ("audio", "text"), output_modality

    data = extern_data[input_data_key].raw_tensor
    seq_len = extern_data[input_data_key].dims[1].dyn_size_ext.raw_tensor.to(device=data.device)

    # input modality -> which encoder path / mask token
    if input_modality == "audio":
        forward_func = model.forward_audio
        mask_idx = model.audio_mask_idx
    else:
        forward_func = model.forward_text
        mask_idx = model.text_mask_idx

    # output modality -> which decoder / vocab / special symbols
    if output_modality == "text":
        out_dim = model.text_out_dim
    else:
        out_dim = model.audio_out_dim

    # optionally mask the encoder input the same way as during training
    enc_indices, enc_lens = data, seq_len
    if masking_opts is not None and masking_opts.get("mask_prob", 0.0) > 0.0:
        mask = get_random_mask(seq_len, **masking_opts)
        enc_indices, enc_lens = mask_sequence(data, seq_len, mask, mask_value=mask_idx)

    # optionally upsample the (masked) encoder input like in training (after masking), so the encoder
    # sees the same longer sequence; the decode/score length stays at the original (max_seq_len).
    if expansion_opts is not None:
        enc_indices, enc_lens = expand_sequence(enc_indices, enc_lens, **expansion_opts)

    _, out_aux_logits, _, _ = forward_func(enc_indices, enc_lens)

    ctc_log_probs = F.log_softmax(out_aux_logits[-1], dim=-1)
    max_ = torch.max(ctc_log_probs, dim=-1)
    seq_targets = max_.indices.squeeze(-1)
    ctc_scores = max_.values.squeeze(-1)
    ctc_scores = torch.where(
        torch.arange(enc_lens.max().item())[None].to(enc_lens.device) > enc_lens[:, None],
        0.0,
        ctc_scores,
    )
    seq_log_prob = ctc_scores.sum(dim=-1)
    seq_log_prob = rf.convert_to_tensor(seq_log_prob, dims=[batch_dim])

    # breakpoint()

    seq_targets, ctc_output_lens = _get_collapsed_out_seqs(
        seq_targets=seq_targets[:, None],  # (B, 1, T)
        blank_idx=model.blank_idx,
    )
    seq_targets = seq_targets.squeeze(1)  # (B, T)
    ctc_output_lens = ctc_output_lens.squeeze(1)  # (B,)

    vocab_dim = Dim(out_dim, name="vocab")
    lens_data = rf.convert_to_tensor(ctc_output_lens, dims=[batch_dim])
    lens_dim = Dim(lens_data, name="seq_len")

    ctx = rf.get_run_ctx()
    seq_targets_rf = rf.convert_to_tensor(seq_targets, dims=[batch_dim, lens_dim], sparse_dim=vocab_dim)
    ctx.mark_as_output(seq_targets_rf, "tokens", dims=[batch_dim, lens_dim])
    ctx.mark_as_output(seq_log_prob, "scores", dims=[batch_dim])
