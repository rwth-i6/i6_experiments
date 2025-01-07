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

def forward_compute_ppl(
    model: Model,
    non_blank_targets: Tensor,
    non_blank_targets_spatial_dim: Dim,
):
    targets_raw = non_blank_targets.raw_tensor
    torch_target_lengths = non_blank_targets_spatial_dim.dyn_size_ext.raw_tensor # (B, S)

    # Compute the LM log probs
    targets_w_bos, targets_spatial_dim_pad = rf.pad(
        non_blank_targets,
        padding=[(1, 0)],
        axes=[non_blank_targets_spatial_dim],
        value=model.eos_idx
    )


    # Now compute the KL Div between the LM and the ILM
    # Because the LM is not always available, this is optional
    extern_lm_out = model.language_model(
        targets_w_bos,
        state=model.language_model.default_initial_state(batch_dims=(batch_dim,)),
        spatial_dim=targets_spatial_dim_pad[0],
        batch_dims=(batch_dim,),
    )
    extern_lm_out_rf = extern_lm_out["output"].copy_as_batch_spatial_major()
    extern_lm_out_raw = extern_lm_out_rf.raw_tensor # (B, S+1, V)
    extern_lm_log_probs = extern_lm_out_raw.log_softmax(-1) # (B, S+1, V)


    # compute PPL of extern LM
    batch_size, max_seq_len = targets_raw.shape
    targets_eos = torch.cat(
        [targets_raw, torch.full((batch_size, 1), fill_value=model.eos_idx, device=targets_raw.device)],
        dim=1,
    ).long()
    ce_elm = torch.nn.functional.cross_entropy(extern_lm_log_probs.transpose(1, 2), targets_eos, reduction='none')
    seq_mask = get_seq_mask(torch_target_lengths+1, max_seq_len+1, targets_raw.device)
    ce_elm = (ce_elm*seq_mask).sum(-1)

    rf.get_run_ctx().mark_as_output(
        ce_elm,
        "ce_extern_lm",
    )


    # TODO: Also returns seq len to properly sum in the callback iface
    rf.get_run_ctx().mark_as_output(
        torch_target_lengths+1,
        "target_len_w_bos",
    )



default_out_file_name = "lm_stats.json"
output_files = [default_out_file_name]

class Compute_PPL_Callback(ForwardCallbackIface):
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
        seq_len_tensor = outputs["target_len_w_bos"].raw_tensor
        # print("seq tag", seq_tag)
        # print("ctc_kldiv", ctc_kldiv_tensor)
        # print("extern_lm_kldiv", extern_lm_kldiv_tensor)
        # print("ce", ce_tensor)
        # print("seq len", seq_len_tensor)
        self.n_target_labels += seq_len_tensor.item()
            
        self.n_seqs += 1
        if self.with_extern_lm:
            ce_tensor_extern_lm = outputs["ce_extern_lm"].raw_tensor
            self.sum_log_ppl_extern_lm += ce_tensor_extern_lm
        # print("avg ctc kldiv", self.avg_ctc_kldiv)
        # print("avg extern lm kldiv", self.avg_extern_lm_kldiv)
        # print("avg ppl", self.avg_ppl)
        # print("n target labels", self.n_target_labels)
        # print("n seqs", self.n_seqs)

    def finish(self):
        res = {
            "n_target_labels": self.n_target_labels,
            "n_seqs": self.n_seqs,
        }
        if hasattr(self, "sum_log_ppl_extern_lm"):
            res["extern_lm_ppl"] = np.exp(self.sum_log_ppl_extern_lm / self.n_target_labels)
        with open(self.output_file_name, "w") as out_file:
            json.dump(res, out_file, indent=4)

def forward_callback_wrapper():
    return Compute_PPL_Callback()
