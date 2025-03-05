"""
Forward functions to compute the kldiv of an ILM
and the encoder and the external LM (specificially trafo LM) 
"""

import torch
import json
import numpy as np

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, batch_dim, TensorDict
from returnn.forward_iface import ForwardCallbackIface
from returnn.config import get_global_config

from i6_experiments.users.phan.rf_models.model_conformer_with_ilm_v2 import Model
from i6_experiments.users.yang.torch.loss.ctc_pref_scores_loss import kldiv_ctc_lm_loss
from i6_experiments.users.yang.torch.utils.tensor_ops import mask_eos_label
from i6_experiments.users.phan.utils.masking import get_seq_mask

def forward_compute_kldiv(
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    non_blank_targets: Tensor,
    non_blank_targets_spatial_dim: Dim,
):
    data_raw = data.raw_tensor
    targets_raw = non_blank_targets.raw_tensor
    
    # First compute the KL Div between the ILM and the CTC

    # Compute the CTC log posteriors
    config = get_global_config()  # noqa
    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    collected_outputs = {}
    enc_args, enc_spatial_dim = model.encode(
        data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs
    )
    mask_eos = config.bool("mask_eos_output", True)
    add_eos_to_blank = config.bool("add_eos_to_blank", False)

    # This should be the final logits of the whole encoder
    aux_logits = model.enc_aux_logits_12(collected_outputs[str(11)])

    if mask_eos:
        mask_eos_label(aux_logits, add_to_blank=add_eos_to_blank)

    # This is the log probs
    aux_logits_raw = aux_logits.raw_tensor # (B, T, V + blank), good
    targets_raw = non_blank_targets.raw_tensor
    torch_target_lengths = non_blank_targets_spatial_dim.dyn_size_ext.raw_tensor # (B, S)
    torch_input_lengths = enc_spatial_dim.dyn_size_ext.raw_tensor # (B, T)
    ctc_log_posteriors = aux_logits_raw.transpose(0, 1).log_softmax(-1) # here it is

    # Compute the ILM log probs
    targets_w_bos, targets_spatial_dim_pad = rf.pad(
        non_blank_targets,
        padding=[(1, 0)],
        axes=[non_blank_targets_spatial_dim],
        value=model.eos_idx
    )
    ilm_out = model.ilm_forward(targets_w_bos, targets_spatial_dim_pad[0])
    ilm_out_raw = ilm_out["output"].raw_tensor # (S+1, B, V)
    ilm_log_probs = ilm_out_raw.transpose(0, 1).log_softmax(-1) # (B, S+1, V)

    ctc_kldiv = kldiv_ctc_lm_loss( # (B, S+1, V)
        ctc_log_posteriors,
        targets_raw.clone().long(),
        torch_input_lengths.long(),
        torch_target_lengths.long(),
        ilm_log_probs,
        blank_idx=model.blank_idx,
        eos_idx=model.eos_idx,
        return_unsummed_loss=True,
    )

    if config.bool("no_bos_eos", False):
        ctc_kldiv = ctc_kldiv[:, :-1, :]
    ctc_kldiv_per_pos = ctc_kldiv.sum(-1)
    rf.get_run_ctx().mark_as_output(
        ctc_kldiv_per_pos,
        "ctc_kldiv",
    )

    # Now compute the KL Div between the LM and the ILM
    # Because the LM is not always available, this is optional
    if config.bool("with_extern_lm", True):
        extern_lm_out = model.language_model(
            targets_w_bos,
            state=model.language_model.default_initial_state(batch_dims=(batch_dim,)),
            spatial_dim=targets_spatial_dim_pad[0],
            batch_dims=(batch_dim,),
        )
        extern_lm_out_rf = extern_lm_out["output"].copy_as_batch_spatial_major()
        extern_lm_out_raw = extern_lm_out_rf.raw_tensor # (B, S+1, V)
        extern_lm_log_probs = extern_lm_out_raw.log_softmax(-1) # (B, S+1, V)

        extern_lm_kldiv = torch.nn.functional.kl_div( # (B, S+1, V)
            input=ilm_log_probs,
            target=extern_lm_log_probs,
            log_target=True,
            reduction="none",
        )

        extern_lm_kldiv_per_pos = extern_lm_kldiv.sum(-1) # (B, S+1)
        rf.get_run_ctx().mark_as_output(
            extern_lm_kldiv_per_pos,
            "extern_lm_kldiv",
        )

        # compute PPL of extern LM
        batch_size, max_seq_len = targets_raw.shape
        targets_eos = torch.cat(
            [targets_raw, torch.full((batch_size, 1), fill_value=model.eos_idx, device=targets_raw.device)],
            dim=1,
        ).long()
        ce_elm = torch.nn.functional.cross_entropy(extern_lm_log_probs.transpose(1, 2), targets_eos, reduction='none')
        seq_mask = get_seq_mask(torch_target_lengths+1, max_seq_len+1, targets_raw.device)
        ce_elm = (ce_elm*seq_mask).sum(-1)
        print(ce_elm)
        rf.get_run_ctx().mark_as_output(
            ce_elm,
            "ce_extern_lm",
        )


    # TODO: Also returns seq len to properly sum in the callback iface
    if config.bool("no_bos_eos", False):
        target_lengths = torch_target_lengths
        max_seq_len_ = max_seq_len
    else:
        target_lengths = torch_target_lengths + 1
        max_seq_len_ = max_seq_len + 1
    rf.get_run_ctx().mark_as_output(
        target_lengths,
        "target_len_w_bos",
    )

    # TODO: Also compute dev-other PPL
    # Also report PPL of the ILM
    batch_size, max_seq_len = targets_raw.shape
    
    if config.bool("no_bos_eos", False):
        targets_eos = targets_raw.long()
        ilm_log_probs = ilm_log_probs[:, :-1, :]
    else:
        targets_eos = torch.cat(
            [targets_raw, torch.full((batch_size, 1), fill_value=model.eos_idx, device=targets_raw.device)],
            dim=1,
        ).long()

    # ilm_log_probs = ilm_log_probs.transpose(0,1)
    ce = torch.nn.functional.cross_entropy(ilm_log_probs.transpose(1, 2), targets_eos, reduction='none')
    seq_mask = get_seq_mask(target_lengths, max_seq_len_, targets_raw.device)
    # print("EOS CE", ce.gather(-1, (target_lengths.to(ce.device).long().unsqueeze(-1)-1)))
    ce = (ce*seq_mask).sum(-1)
    rf.get_run_ctx().mark_as_output(
        ce,
        "ce",
    )

default_out_file_name = "ilm_stats.json"
output_files = [default_out_file_name]

class Compute_ILM_KLDiv_Stats_Forward_Callback(ForwardCallbackIface):
    def __init__(self, output_file_name=default_out_file_name):
        self.output_file_name = output_file_name

    def init(self, model):
        self.avg_ctc_kldiv = 0.
        # PPL is the average of PPL of all seqs
        # This is different from KL Div which is summed and averages
        # over all target labels
        self.sum_log_ppl_ilm = 0.
        self.n_target_labels = 0 # number of total target labels
        self.n_seqs = 0
        config = get_global_config()
        self.with_extern_lm = config.bool("with_extern_lm", True)
        if self.with_extern_lm:
            self.avg_extern_lm_kldiv = 0.
            self.sum_log_ppl_extern_lm = 0.
            print("Model has extern LM")
        else:
            print("Model has no extern LM")

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
        ctc_kldiv_tensor = outputs["ctc_kldiv"].raw_tensor
        
        ce_tensor = outputs["ce"].raw_tensor
        seq_len_tensor = outputs["target_len_w_bos"].raw_tensor
        # print("seq tag", seq_tag)
        # print("ctc_kldiv", ctc_kldiv_tensor)
        # print("extern_lm_kldiv", extern_lm_kldiv_tensor)
        # print("ce", ce_tensor)
        # print("seq len", seq_len_tensor)
        for i in range(seq_len_tensor.item()):
            self.n_target_labels += 1
            self.avg_ctc_kldiv += (ctc_kldiv_tensor[i] - self.avg_ctc_kldiv) / self.n_target_labels
        self.sum_log_ppl_ilm += ce_tensor
            
        self.n_seqs += 1
        if self.with_extern_lm:
            extern_lm_kldiv_tensor = outputs["extern_lm_kldiv"].raw_tensor
            for i in range(seq_len_tensor.item()):
                self.avg_extern_lm_kldiv += (extern_lm_kldiv_tensor[i] - self.avg_extern_lm_kldiv) / self.n_target_labels
            ce_tensor_extern_lm = outputs["ce_extern_lm"].raw_tensor
            self.sum_log_ppl_extern_lm += ce_tensor_extern_lm
        # print("avg ctc kldiv", self.avg_ctc_kldiv)
        # print("avg extern lm kldiv", self.avg_extern_lm_kldiv)
        # print("avg ppl", self.avg_ppl)
        # print("n target labels", self.n_target_labels)
        # print("n seqs", self.n_seqs)

    def finish(self):
        res = {
            "ctc_kldiv": self.avg_ctc_kldiv,
            "ilm_ppl": np.exp(self.sum_log_ppl_ilm / self.n_target_labels),
            "n_target_labels": self.n_target_labels,
            "n_seqs": self.n_seqs,
        }
        if hasattr(self, "avg_extern_lm_kldiv"):
            res["extern_lm_kldiv"] = self.avg_extern_lm_kldiv
        if hasattr(self, "sum_log_ppl_extern_lm"):
            res["extern_lm_ppl"] = np.exp(self.sum_log_ppl_extern_lm / self.n_target_labels)
        with open(self.output_file_name, "w") as out_file:
            json.dump(res, out_file, indent=4)

def forward_callback_wrapper():
    return Compute_ILM_KLDiv_Stats_Forward_Callback()
