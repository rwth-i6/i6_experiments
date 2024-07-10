"""
Train a bidirectional masked LSTM LM with a procedure similar to BERT:
masked some tokens according to a masking rate and then compute loss
only on masked tokens. For dev score, PLL is used (not PPL).
"""

from returnn.tensor import batch_dim
from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf
import torch
from i6_experiments.users.phan.utils.masking import get_seq_mask, mask_target_sequences


def train_step(
    *,
    model: torch.nn.Module,
    extern_data: TensorDict,
    phase: str,
    mask_ratio: float,
    mask_idx: int,
    **kwargs
):
    """
    THIS ONLY WORKS FOR mask_idx = 0 !!!

    Note that the dataloader should not have EOS!

    :param phase: "train" or "eval". Needed for models having different train and eval procedures.
    :param mask_ratio: How many target labels to msak
    :param mask_idx: index used to represent <mask> in the masked LM
    :param mask_audio: if yes, also mask acoustic input of mask 
    """
    assert extern_data["data"].raw_tensor is not None
    targets = extern_data["data"].raw_tensor.long()
    targets_len_rf = extern_data["data"].dims[1].dyn_size_ext
    assert targets_len_rf is not None
    targets_len = targets_len_rf.raw_tensor
    assert targets_len is not None

    model.train()
    # This targets and targets_len have no EOS
    batch_size, max_seq_len = targets.shape
    device = targets.device

    model.train()
    # This targets and targets_len already have EOS
    batch_size, max_seq_len = targets.shape
    device = targets.device
    # Reason for target_shifted: the phonemes are 1, 2, ..., 78, while the model outputs 0, 1, ..., 77
    # targets_shifted should be used for eval against model outputs 
    targets_shifted = torch.where(targets > 0, targets-1, torch.tensor(0).to(device))
    
    if phase == "train":
        targets_masked, target_masking = mask_target_sequences(
            targets,
            mask_ratio=mask_ratio,
            eos_idx=0,
            mask_idx=mask_idx,
        )
        targets_masked_onehot = torch.nn.functional.one_hot(targets_masked, num_classes=model.cfg.vocab_dim).float() # should be (B, S+1, 80)
        log_lm_probs = model(targets_masked_onehot, targets_len) # pack for bidirectional LSTM, dont count eos
        ce = torch.nn.functional.cross_entropy(log_lm_probs.transpose(1, 2), targets_shifted, reduction='none')
        # Compute pseudo LL only on masked tokens
        loss = (ce * target_masking).sum() / (target_masking.sum())
        rf.get_run_ctx().mark_as_loss(
            name="log_pseudo_ppl", loss=loss,
        )
        loss_exp = torch.exp(loss)
        rf.get_run_ctx().mark_as_loss(
            name="pseudo_ppl", loss=loss_exp, as_error=True,
        )

    elif phase == "eval":
        # # This approach is fine but caused OOM
        # # MAYBE: decrease batch size?
        # # for each label position s, mask out and compute loss for that label position (pseudo PPL)
        # # Create a mask like this for sequence lengths [3, 2]:
        # # [1, 0, 0]
        # # [0, 1, 0]
        # # [0, 0, 1]
        # # [1, 0, 0]
        # # [0, 1, 0]
        # mask = torch.zeros((targets_len.sum(), max_seq_len)).to(device)
        # col_idx = torch.concat([torch.arange(s) for s in targets_len], dim=0).unsqueeze(-1).to(device)
        # mask = torch.scatter(mask, 1, col_idx, 1)
        # # Repeat the input sequence and mask the positions one-by-one. Example:
        # # [<mask>, 2, 3, ..., 10] first sequence
        # # [1, <mask>, 3, ..., 10]
        # # [1, 2, 3, ..., <mask>]
        # # [<mask>, 12, ...] second sequence
        # # [11, <mask>, ...]
        # # ... and so on
        # targets_repeat = torch.repeat_interleave(targets, targets_len.to(device), dim=0)
        # targets_repeat_masked = torch.where(mask.bool(), mask_idx, targets_repeat)
        # targets_repeat_masked_onehot = torch.nn.functional.one_hot(targets_repeat_masked, num_classes=model.cfg.vocab_dim).float()
        # targets_len_repeat = torch.repeat_interleave(targets_len, targets_len, dim=0)
        # log_lm_probs = model(targets_repeat_masked_onehot, targets_len_repeat)
        # ce = torch.nn.functional.cross_entropy(log_lm_probs.transpose(1, 2), targets_repeat, reduction='none')
        # loss = (ce * mask.float()).sum() / mask.float().sum()

        # Much slower but fine with memory
        seq_mask = get_seq_mask(targets_len, max_seq_len, device)
        acc_loss = 0
        for s in range(max_seq_len):
            targets_s = targets.clone()
            targets_s[:, s] = mask_idx
            targets_s_onehot = torch.nn.functional.one_hot(targets_s, num_classes=model.cfg.vocab_dim).float()
            log_lm_probs = model(targets_s_onehot, targets_len)
            ce = torch.nn.functional.cross_entropy(log_lm_probs.transpose(1, 2), targets_shifted, reduction='none')
            acc_loss += (ce[:, s] * seq_mask[:, s]).sum()
        loss = acc_loss/seq_mask.sum()

        rf.get_run_ctx().mark_as_loss(
            name="log_pseudo_ppl", loss=loss, as_error=True,
        )
        loss_exp = torch.exp(loss)
        rf.get_run_ctx().mark_as_loss(
            name="pseudo_ppl", loss=loss_exp, as_error=True,
        )
