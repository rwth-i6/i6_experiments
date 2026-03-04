__all__ = ["forward_step"]

import returnn.frontend as rf
import returnn.tensor
import torch
import torch.nn.functional as F
from returnn.tensor import Dim, TensorDict, batch_dim

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.networks.interfaces.base_encoder_decoder_model import \
    BaseEncoderDecoderModel


def forward_step_v1(
        *,
        model: BaseEncoderDecoderModel,
        extern_data: TensorDict,
        **kwargs,
):
    """from SLLM repo"""

    from returnn.config import get_global_config

    config = get_global_config(return_empty_if_none=True)

    data_key = config.value("default_data_key", "audio")
    data_ = extern_data[data_key]
    if data_.feature_dim and data_.feature_dim.dimension == 1:
        data_ = rf.squeeze(data_, axis=data_.feature_dim)
    data = data_.raw_tensor  # (B, T, F)
    data_lens = (
        extern_data[data_key].dims[1].dyn_size_ext.raw_tensor.to(device=data.device)
    )

    targets_flat = extern_data["hyps_flat"]
    targets_flat_time_dim = targets_flat.dims[1]
    targets_seq_lens = extern_data["hyps_seq_lens"]  # [B, beam]
    targets_beam_dim = targets_seq_lens.dims[1]
    targets_spatial_dim = Dim(
        rf.copy_to_device(targets_seq_lens, "cpu"), name="targets_spatial"
    )
    targets = rf.pad_packed(
        targets_flat,
        in_dim=targets_flat_time_dim,
        dims=[targets_beam_dim, targets_spatial_dim],
    )
    targets = targets.copy_transpose(
        [batch_dim, targets_beam_dim, targets_spatial_dim]
    )  # [B, beam, T]

    beam_size = targets_beam_dim.dimension
    batch_size = targets.raw_tensor.shape[0]
    target_lens = targets_seq_lens.raw_tensor.view(-1).to(data.device)  # [B * beam]
    targets = targets.raw_tensor.reshape(batch_size * beam_size, -1)  # [B * beam, T]

    adapter_output, aux_log_probs, adapter_output_lens, _, encoder_output_lens = (
        model.forward(data, data_lens)
    )
    adapter_output = adapter_output.repeat_interleave(
        beam_size, dim=0
    )  # [B * beam, T, D]
    adapter_output_lens = adapter_output_lens.repeat_interleave(
        beam_size, dim=0
    )  # [B * beam]

    input_labels = F.pad(targets, (1, 0), "constant", value=model.bos_idx)
    input_labels_len = target_lens + 1
    logits = model.decode_seq(
        x=input_labels,
        x_lens=input_labels_len.to(device=input_labels.device),
        audio_features=adapter_output,
        audio_features_lens=adapter_output_lens,
    )  # [B * beam, T+1, vocab]
    if model.dec_out_blank_logits is not None:
        # decode_seq returns a tuple of (logits, aux_log_probs)
        logits, _ = logits

    log_probs = F.log_softmax(logits, dim=-1)
    # this is wrong because of missing EOS -> see v2 below
    seq_log_prob = torch.gather(
        log_probs, dim=-1, index=targets.unsqueeze(-1).long()
    ).squeeze(
        -1
    )  # [B * beam, T]
    seq_log_prob = torch.where(
        torch.arange(seq_log_prob.size(1), device=seq_log_prob.device).unsqueeze(0)
        < target_lens.unsqueeze(1),
        seq_log_prob,
        torch.zeros_like(seq_log_prob),
        )
    seq_log_prob = torch.sum(seq_log_prob, dim=-1).view(
        batch_size, beam_size
    )  # [B, beam]

    beam_dim = Dim(beam_size, name="beam")
    vocab_dim = Dim(model.num_labels, name="vocab")
    lens_data = rf.convert_to_tensor(
        target_lens.view(batch_size, beam_size), dims=[batch_dim, beam_dim]
    )
    lens_dim = Dim(lens_data, name="seq_len")

    ctx = rf.get_run_ctx()
    targets = targets.view(batch_size, beam_size, -1)  # [B, beam, T]
    seq_targets_rf = rf.convert_to_tensor(
        targets, dims=[batch_dim, beam_dim, lens_dim], sparse_dim=vocab_dim
    )
    ctx.mark_as_output(seq_targets_rf, "tokens", dims=[batch_dim, beam_dim, lens_dim])
    ctx.mark_as_output(seq_log_prob, "scores", dims=[batch_dim, beam_dim])