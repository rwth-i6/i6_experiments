"""
For Albert
"""

from __future__ import annotations
import numpy
import copy
from .chunkwise_att_base import get_ctc_rna_based_chunk_alignments, prefix_name, default_args, run_exp


def sis_config_main():
    """sis config function"""
    ctc_align_wo_speed_pert = get_ctc_rna_based_chunk_alignments()

    # train with ctc chunk-sync alignment
    total_epochs = 40
    chunk_size = 20
    chunk_step_factor = 0.9
    start_lr = 1e-4
    decay_pt_factor = 1 / 3

    train_args = copy.deepcopy(default_args)
    train_args["speed_pert"] = False  # no speed pert
    train_args["search_type"] = None  # fixed alignment

    train_args["max_seq_length"] = None  # no filtering!

    train_args["encoder_args"].with_ctc = False  # No CTC

    decay_pt = int(total_epochs * decay_pt_factor)

    train_args["chunk_size"] = chunk_size

    chunk_step = max(1, int(chunk_size * chunk_step_factor))
    train_args["chunk_step"] = chunk_step

    train_args["learning_rates_list"] = [start_lr] * decay_pt + list(
        numpy.linspace(start_lr, 1e-6, total_epochs - decay_pt)
    )

    run_exp(
        prefix_name=prefix_name,
        exp_name=f"base_chunkwise_att"
        f"_chunk-{chunk_size}_step-{chunk_step}"
        f"_linDecay{total_epochs}_{start_lr}_decayPt{decay_pt_factor}"
        f"_fixed_align",
        train_args=train_args,
        num_epochs=total_epochs,
        train_fixed_alignment=ctc_align_wo_speed_pert["train"][f"{chunk_size}_{chunk_step}"],
        cv_fixed_alignment=ctc_align_wo_speed_pert["dev"][f"{chunk_size}_{chunk_step}"],
        epoch_wise_filter=None,
        time_rqmt=72,
        selected_datasets=["dev-other"],
        key="dev_score",
        use_sclite=True,
    )


py = sis_config_main  # `py` is the default sis config function name
