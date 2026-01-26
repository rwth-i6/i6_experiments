__all__ = ["perplexity_forward_step"]

import torch
from torch import Tensor
import torch.nn.functional as F

import returnn.frontend as rf
from returnn.tensor import TensorDict, batch_dim
from returnn.frontend import RunCtx


def perplexity_forward_step(*,

                            model: torch.nn.Module,
                            extern_data: TensorDict,

                            **_kwargs):

    ctx: RunCtx = rf.get_run_ctx()

    targets_ = extern_data["data"]  # Target / label / ground truth # TODO: extract const
    targets: Tensor = targets_.raw_tensor
    target_lens: Tensor = targets_.dims[1].dyn_size_ext.raw_tensor

    # DECODER (FORWARD) STEP
    input_labels = F.pad(targets, (1, 0), "constant", value=model.bos_idx)  # [B, MaxTextLen]
    input_labels_len = target_lens + 1  # [B]
    logits: Tensor = model.decode_seq_lm(input_labels,
                                         input_labels_len)  # [B, SeqLen, vocab_size] | ex. SeqLen [TK1, TK2, EOS]

    targets = F.pad(targets, (0, 1), "constant", value=model.eos_idx)
    target_lens = target_lens + 1

    log_flat = logits.flatten(0, 1)
    tar_flat = targets.flatten(0, 1)

    loss = F.cross_entropy(log_flat, tar_flat.long(), reduction="none")
    loss = loss.reshape(logits.shape[:-1]) # Only last layer

    r = torch.arange(loss.shape[1], device=loss.device)  # [T]

    seq_mask = torch.less(r[None, :], target_lens[:, None].to(loss.device))  # broadcast to [B,T]
    log_prob_targets = torch.where(seq_mask, loss, 0)
    log_prob_targets_seq = torch.sum(log_prob_targets, dim=-1)

    ctx.mark_as_output(log_prob_targets_seq, "scores_ce", dims=[batch_dim])
    ctx.mark_as_output(torch.tensor(target_lens), "len_ce")
