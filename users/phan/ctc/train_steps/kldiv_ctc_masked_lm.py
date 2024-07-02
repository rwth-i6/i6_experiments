"""
Train a bidirectional masked LSTM LM with a procedure similar to BERT:
masked some tokens according to a masking rate and then compute loss
only on masked tokens. For dev score, PLL is used (not PPL).

Important: In this training, all sequences in the same batch share
the same masking for implementation reason.
"""

from returnn.tensor import batch_dim
from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf
import torch
from i6_experiments.users.phan.utils.masking import get_seq_mask, mask_audio_features_exact_label_pos
from i6_experiments.users.phan.utils.alignments import convert_alignments_to_target_sequences
from i6_experiments.users.phan.ctc.ctc_masked_score import ctc_masked_score


def train_step(
    *,
    model: torch.nn.Module,
    extern_data: TensorDict,
    phase: str,
    mask_ratio: float,
    mask_idx: int,
    mask_audio: bool,
    sil_index: int,
    **kwargs
):
    """
    :param phase: "train" or "eval". Needed for models having different train and eval procedures.
    :param mask_ratio: How many target labels to msak
    :param mask_idx: index used to represent <mask> in the masked LM
    :param mask_audio: if yes, also mask acoustic input of mask 
    :param sil_index: index of [SILENCE] in the alignments
    """
    audio_features = extern_data["data"].raw_tensor
    assert extern_data["data"].dims[1].dyn_size_ext is not None

    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    assert audio_features_len is not None

    assert extern_data["targets"].raw_tensor is not None
    targets = extern_data["targets"].raw_tensor.long()

    targets_len_rf = extern_data["targets"].dims[1].dyn_size_ext
    assert targets_len_rf is not None
    targets_len = targets_len_rf.raw_tensor.long()

    model.train()
    model.module_dict["teacher_ctc"].eval()
    device = audio_features.device
    batch_size, max_seq_len = targets.shape

    # Prepare forward kwargs
    # If mask audio, prepare the mask and replace the targets
    # provided by the dataloader by the targets represented by the alignment
    forward_kwargs = {
        "audio_features": audio_features,
        "audio_features_len": audio_features_len.to("cuda"),
    }
    # label_mask = (torch.rand((max_seq_len,)) < mask_ratio).long() # Generate the mask (S,)
    # while not (label_mask == 1).any():
    #     label_mask = (torch.rand((max_seq_len,)) < mask_ratio).long()
    # min_seq_len = targets_len.min()
    # label_mask = torch.cat([torch.zeros((min_seq_len,)), torch.ones((max_seq_len-min_seq_len,))], dim=-1).long()
    # print(min_seq_len, max_seq_len)
    # print(label_mask)
    label_mask = torch.zeros((max_seq_len,)).long()
    # label_mask[0] = 1
    # print(label_mask)
    if phase == "train" and mask_audio:
        assert extern_data["align"] is not None, "Alignments must be given to mask audio features"
        alignments = extern_data["align"].raw_tensor
        targets, targets_len = convert_alignments_to_target_sequences(alignments, sil_index, 0)
        max_seq_len = targets.shape[1]
        batch_label_mask = label_mask.unsqueeze(0).expand(batch_size, -1)
        audio_features_mask = mask_audio_features_exact_label_pos(alignments, label_mask, sil_index)
        forward_kwargs["audio_features_mask"] = audio_features_mask

    seq_mask = get_seq_mask(targets_len, max_seq_len, targets.device)
    lm_input_dim = model.cfg.module_config["student_lm"].vocab_dim
    lm_output_dim = model.cfg.module_config["student_lm"].output_dim

    # In training, mask ground truth, compute loss only
    # on masked tokens
    if phase == "train":
        log_probs, sequence_mask, _ = model(
            args=[],
            kwargs=forward_kwargs,
            module="teacher_ctc",
            inference=True,
        )
        input_lengths = sequence_mask.sum(-1).long()
        log_probs = torch.transpose(log_probs, 0, 1) # (T, B, F)
        target_masking = label_mask.unsqueeze(0).expand(batch_size, -1) # (B, S)
        masked_idx = torch.arange(max_seq_len)[label_mask.bool()]
        targets_masked = targets.clone() - 1 # because there is no EOS here
        targets_masked[targets_masked < 0] = 0 # doesn't matter anyway, these are paddings
        targets_masked[:, masked_idx] = mask_idx
        targets_masked_onehot = torch.nn.functional.one_hot(targets_masked, num_classes=lm_input_dim).float() # should be (B, S, 79)
        log_lm_probs = model( # (B, S, F-1)
            args=[targets_masked_onehot, targets_len],
            kwargs={},
            module="student_lm",
            inference=False,
        )
        log_lm_masked_score = log_lm_probs[:, masked_idx, :] # (B, M, F-1)
        log_ctc_masked_score, _, _, _, _ = ctc_masked_score( # (B, M, F-1)
            log_probs,
            targets,
            label_mask,
            input_lengths,
            targets_len,
        )
        mask_inside_seq = seq_mask[:, masked_idx] # (B, M), indicates if a mask in the actual seq or not
        mask_inside_seq = mask_inside_seq.unsqueeze(-1).expand(-1, -1, log_lm_probs.shape[-1])
        # take kldiv only on masked tokens
        kldiv = torch.nn.functional.kl_div(
            input=log_lm_masked_score,
            target=log_ctc_masked_score,
            log_target=True,
            reduction="none",
        )
        torch.set_printoptions(precision=2, threshold=100000, linewidth=150)
        # print(label_mask)
        # print(targets_len)
        # print(torch.stack([log_ctc_masked_score, log_lm_masked_score, kldiv, mask_inside_seq], dim=-1))

        loss = (kldiv * mask_inside_seq).sum() / (mask_inside_seq.sum())
        rf.get_run_ctx().mark_as_loss(
            name="kldiv_ctc_masked_lm", loss=loss,
        )


    if phase == "eval":
        # It would be infeasible to calculate log PPPL 
        # with each acoustic input masked out
        # We simply calculate the
        seq_mask = get_seq_mask(targets_len, max_seq_len, device)
        acc_loss = 0
        for s in range(max_seq_len):
            targets_s = targets.clone()
            targets_s[:, s] = mask_idx
            targets_s_onehot = torch.nn.functional.one_hot(targets_s, num_classes=lm_input_dim).float()
            log_lm_probs = model(targets_s_onehot, targets_len)
            ce = torch.nn.functional.cross_entropy(log_lm_probs.transpose(1, 2), targets, reduction='none')
            acc_loss += (ce[:, s] * seq_mask[:, s]).sum()
        loss = acc_loss/seq_mask.sum()

        rf.get_run_ctx().mark_as_loss(
            name="log_pseudo_ppl", loss=loss, as_error=True,
        )
        loss_exp = torch.exp(loss)
        rf.get_run_ctx().mark_as_loss(
            name="pseudo_ppl", loss=loss_exp, as_error=True,
        )
