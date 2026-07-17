__all__ = ["train_step"]

from typing import Protocol

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, unpad_sequence

import returnn.frontend as rf
from returnn.tensor import Tensor as ReturnnTensor
from returnn.tensor import TensorDict


class LmModel(Protocol):
    bos_idx: int
    eos_idx: int

    def score_text(self, labels: Tensor, labels_lens: Tensor) -> Tensor:
        """Teacher-forced logits ``[B, T, V]`` over the given (bos-prefixed) label sequence."""


def train_step(
    *,
    model: LmModel,
    extern_data: TensorDict,
    data_key: str = "data",
    ce_loss_scale: float = 1.0,
    label_smoothing: float = 0.0,
    label_smoothing_start_epoch: int = 0,
    **_kwargs,
):
    """
    Plain next-token cross-entropy language-model training on label sequences.

    Teacher forcing: the decoder input is ``bos + labels`` and the target is ``labels + eos``. The
    per-token CE (normalized by sequence length, incl. eos) is marked as the training loss, plus the
    frame error rate as an error metric.
    """
    ctx = rf.get_run_ctx()
    num_eos_symbols = 1

    label_indices_: ReturnnTensor = extern_data[data_key]
    assert label_indices_.sparse_dim is not None
    label_indices: Tensor = label_indices_.raw_tensor
    label_indices_lens: Tensor = label_indices_.dims[1].dyn_size_ext.raw_tensor

    # when using a CombinedDataset we can have zero-length sequences (harmless for a plain text set)
    if torch.all(label_indices_lens == 0).item():
        return
    label_indices = label_indices[label_indices_lens > 0]
    label_indices_lens = label_indices_lens[label_indices_lens > 0]

    input_labels = F.pad(label_indices, (1, 0), "constant", value=model.bos_idx)
    input_labels_lens = label_indices_lens + 1
    logits = model.score_text(input_labels, input_labels_lens.to(device=input_labels.device))
    logits_packed = pack_padded_sequence(logits, input_labels_lens, batch_first=True, enforce_sorted=False)

    single_seqs = unpad_sequence(label_indices, label_indices_lens, batch_first=True)
    eos_tensor = torch.tensor(num_eos_symbols * [model.eos_idx], device=label_indices.device, dtype=torch.int32)
    targets_w_eos_packed = pack_sequence(
        [torch.concat((seq, eos_tensor), dim=-1) for seq in single_seqs],
        enforce_sorted=False,
    ).data

    ce_loss = F.cross_entropy(
        logits_packed.data,
        targets_w_eos_packed.long(),
        label_smoothing=label_smoothing if ctx.epoch >= label_smoothing_start_epoch else 0.0,
        reduction="none",
    )
    error = torch.argmax(logits_packed.data, dim=-1).not_equal(targets_w_eos_packed)

    if ce_loss_scale > 0.0:
        ctx.mark_as_loss(
            ce_loss,
            "ce",
            scale=ce_loss_scale,
            custom_inv_norm_factor=label_indices_.dims[1].get_size_tensor() + num_eos_symbols,
            use_normalized_loss=True,
        )
    ctx.mark_as_loss(error, "fer", as_error=True)

    # pp = p(a_1^S)^{-1/S} = exp(-1/S * log(p(a_1^S)))
    log_probs_packed = torch.nn.functional.log_softmax(logits_packed.data, dim=-1)  # (B, T)
    target_log_probs = torch.gather(
        log_probs_packed,
        dim=-1,
        index=targets_w_eos_packed.unsqueeze(1),
    )  # (B * T)
    target_log_probs_avg = torch.sum(target_log_probs) / target_log_probs.shape[0]
    ppl = torch.exp(-target_log_probs_avg)
    ctx.mark_as_loss(ppl, "ppl", as_error=True)
