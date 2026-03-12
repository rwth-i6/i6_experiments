__all__ = ["forward_step"]

from typing import Optional

import returnn.frontend as rf
import returnn.tensor
import torch
import torch.nn.functional as F
from returnn.tensor import Dim, TensorDict
from returnn.tensor import Dim as ReturnnDim
from returnn.tensor import Tensor as ReturnnTensor
from torch import Tensor

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.networks.interfaces.base_encoder_decoder_model import (
    BaseEncoderDecoderModel,
)
from i6_experiments.users.zeyer.nn_rf.soft_collapse_repeated import (
    soft_collapse_repeated,
)


def apply_ctc_compression_to_log_probs(
    wb_target_dim: int,
    encoder_output_lens: Tensor,
    log_probs: Tensor,
    ctc_soft_collapse_threshold: float,
    ctc_soft_collapse_reduce_type: str,
):
    batch_dim = ReturnnDim(encoder_output_lens.size(0), name="batch")
    wb_target_dim = ReturnnDim(wb_target_dim, name="wb_target")
    enc_spatial_dim = ReturnnDim(
        name="time",
        dimension=None,
        dyn_size_ext=ReturnnTensor(
            "lens",
            raw_tensor=encoder_output_lens.to("cpu"),
            dims=[batch_dim],
            dtype=rf.get_default_int_dtype(),
        ),
    )
    ctc_log_prob = ReturnnTensor(
        "log_probs",
        dims=[batch_dim, enc_spatial_dim, wb_target_dim],
        raw_tensor=log_probs,
        dtype="float32",
    )
    ctc_log_prob, enc_spatial_dim = soft_collapse_repeated(
        ctc_log_prob,
        spatial_dim=enc_spatial_dim,
        classes_dim=wb_target_dim,
        threshold=ctc_soft_collapse_threshold,
        reduce_type=ctc_soft_collapse_reduce_type,
    )
    ctc_log_prob = ctc_log_prob.raw_tensor
    encoder_output_lens = enc_spatial_dim.dyn_size_ext.raw_tensor.to(log_probs.device)

    return ctc_log_prob, encoder_output_lens


def forward_step_v1(
    *,
    model: BaseEncoderDecoderModel,
    extern_data: TensorDict,
    use_ext_ctc: bool = False,
    ctc_soft_collapse_threshold: Optional[float] = None,
    ctc_soft_collapse_reduce_type: str = "logmeanexp",
    ctc_top_k_pruning: Optional[int] = None,
    ctc_top_k_pruning_reduce_func: str = "mean",
    use_dec_aux_log_probs: bool = False,
    **kwargs,
):
    """Runs rescoring on the given data."""

    from returnn.config import get_global_config

    config = get_global_config(return_empty_if_none=True)
    data_key = config.value("default_data_key", "audio")
    data_ = extern_data[data_key]
    # if data_.feature_dim and data_.feature_dim.dimension == 1:
    #     data_ = rf.squeeze(data_, axis=data_.feature_dim)
    data = data_.raw_tensor  # (B, T, F)
    data_lens = extern_data[data_key].dims[1].dyn_size_ext.raw_tensor.to(device=data.device)

    targets_flat = extern_data["hyps_flat"]
    targets_flat_time_dim = targets_flat.dims[1]
    targets_seq_lens = extern_data["hyps_seq_lens"]  # [B, beam]
    targets_beam_dim = targets_seq_lens.dims[1]
    batch_dim = targets_seq_lens.dims[0]
    targets_spatial_dim = Dim(rf.copy_to_device(targets_seq_lens, "cpu"), name="targets_spatial")
    targets = rf.pad_packed(
        targets_flat,
        in_dim=targets_flat_time_dim,
        dims=[targets_beam_dim, targets_spatial_dim],
    )
    targets = targets.copy_transpose([batch_dim, targets_beam_dim, targets_spatial_dim])  # [B, beam, T]

    beam_size = targets_beam_dim.dimension # Changed
    batch_size = targets.raw_tensor.shape[0]
    beam_size_tensor = torch.full((batch_size,), beam_size, dtype=torch.long, device=data.device) # Changed
    target_lens = targets_seq_lens.raw_tensor.view(-1).to(data.device)  # [B * beam]
    targets = targets.raw_tensor.reshape(batch_size * beam_size, -1)  # [B * beam, T]

    if use_ext_ctc:
        (llm_audio_features_in, aux_log_probs, encoder_output_lens, _) = model.external_ctc.forward(data, data_lens)
    else:
        (llm_audio_features_in, aux_log_probs, encoder_output_lens, _) = model.forward(data, data_lens)

    if use_dec_aux_log_probs:
        assert False, "Not addapted!"
        _, dec_aux_log_probs = model.decode_seq(
            x=torch.empty((data.size(0), 0), dtype=torch.long, device=data.device),
            x_lens=torch.zeros((data.size(0),), dtype=torch.long, device=data.device),
            audio_features=llm_audio_features_in,
            audio_features_lens=adapter_output_lengths,
        )
        ctc_log_prob = dec_aux_log_probs[-1]  # Batch, Time, Vocab
        encoder_output_lens = adapter_output_lengths
    else:
        # hard code last aux logit for now. make adjustable later if needed.
        ctc_log_prob = aux_log_probs[-1]

    if ctc_top_k_pruning is not None:
        reduce_func = getattr(torch, ctc_top_k_pruning_reduce_func)
        # assumes that blank is the last index in the vocab
        reduced_log_probs = reduce_func(ctc_log_prob[:, :, :-1], dim=1)
        if ctc_top_k_pruning_reduce_func in ("max", "min"):
            reduced_log_probs = reduced_log_probs[0]
        # get top k log probs for non-blank labels over reduced time frames
        _, pruned_indices = torch.topk(reduced_log_probs, k=ctc_top_k_pruning, dim=-1)
        # add blank to pruned indices
        pruned_indices_wb = torch.cat(
            [
                pruned_indices,
                # EOS is needed for CTC prefix scoring
                torch.full(
                    (pruned_indices.size(0), 1),
                    model.blank_idx,
                    device=pruned_indices.device,
                ),
            ],
            dim=-1,
        )
        # gather selected log probs and re-normalize
        ctc_log_prob = torch.gather(
            ctc_log_prob,
            dim=-1,
            index=pruned_indices_wb.unsqueeze(1).expand(-1, ctc_log_prob.size(1), -1),
        )
        ctc_log_prob = torch.nn.functional.log_softmax(ctc_log_prob, dim=-1)
        pruned_wb_target_dim = Dim(pruned_indices_wb.size(1), name="pruned_wb_target_dim")
        enc_spatial_dim = Dim(
            rf.convert_to_tensor(encoder_output_lens, dims=[batch_dim]),
            name="enc_spatial_dim",
        )
        wb_target_dim = Dim(model.wb_target_dim, name="wb_target_dim")
        pruned_indices_wb_rf = rf.convert_to_tensor(
            pruned_indices_wb,
            dims=[batch_dim, pruned_wb_target_dim],
            sparse_dim=wb_target_dim,
        )
        ctc_log_prob = rf.convert_to_tensor(ctc_log_prob, dims=[batch_dim, enc_spatial_dim, pruned_wb_target_dim])
        # scatter pruned log probs back to original vocab size with -inf for non-selected
        ctc_log_prob = rf.scatter(
            ctc_log_prob,
            fill_value=float("-inf"),
            indices=pruned_indices_wb_rf,
            indices_dim=pruned_wb_target_dim,
            out_dim=wb_target_dim,
            mode="max",
        )
        ctc_log_prob = ctc_log_prob.copy_transpose((batch_dim, enc_spatial_dim, wb_target_dim)).raw_tensor

    if ctc_soft_collapse_threshold is not None:
        ctc_log_prob, encoder_output_lens = apply_ctc_compression_to_log_probs(
            wb_target_dim=model.wb_target_dim,
            encoder_output_lens=encoder_output_lens,
            log_probs=ctc_log_prob,
            ctc_soft_collapse_threshold=ctc_soft_collapse_threshold,
            ctc_soft_collapse_reduce_type=ctc_soft_collapse_reduce_type,
        )

    log_prob = ctc_log_prob.repeat_interleave(beam_size, dim=0)  # [B * beam, T, D]
    encoder_output_lens = encoder_output_lens.repeat_interleave(beam_size, dim=0)  # [B * beam]

    seq_log_prob = -F.ctc_loss(
        log_probs=log_prob.transpose(0, 1).to(torch.float32),
        targets=targets,
        input_lengths=encoder_output_lens,
        target_lengths=target_lens,
        blank=model.blank_idx,
        reduction="none",
        zero_infinity=True,
    ).view(
        batch_size, beam_size
    )  # [B, beam]

    seq_log_prob = torch.where(
        torch.arange(beam_size, device=beam_size_tensor.device).unsqueeze(0) < beam_size_tensor.unsqueeze(1),
        seq_log_prob,
        torch.full_like(seq_log_prob, float("-inf")),
    )

    beam_dim = Dim(beam_size, name="beam")
    vocab_dim = Dim(model.num_labels, name="vocab")
    lens_data = rf.convert_to_tensor(target_lens.view(batch_size, beam_size), dims=[batch_dim, beam_dim])
    lens_dim = Dim(lens_data, name="seq_len")

    ctx = rf.get_run_ctx()
    targets = targets.view(batch_size, beam_size, -1)  # [B, beam, T]
    seq_targets_rf = rf.convert_to_tensor(targets, dims=[batch_dim, beam_dim, lens_dim], sparse_dim=vocab_dim)
    ctx.mark_as_output(seq_targets_rf, "tokens", dims=[batch_dim, beam_dim, lens_dim])
    ctx.mark_as_output(seq_log_prob, "scores", dims=[batch_dim, beam_dim])
