"""
For Albert
"""

from __future__ import annotations
from typing import Optional
import numpy
import copy
from .chunkwise_att_base import get_ctc_rna_based_chunk_alignments, prefix_name, default_args, run_exp


def _run_exp_baseline_v1(
    *,
    enc_stream_type: Optional[str],
    total_epochs: int,
):
    start_lr = 1e-4
    decay_pt_factor = 1 / 3

    # train with ctc chunk-sync alignment
    train_args = copy.deepcopy(default_args)
    train_args["speed_pert"] = False  # no speed pert
    train_args["search_type"] = None  # fixed alignment

    train_args["max_seq_length"] = None  # no filtering!

    train_args["chunk_size"] = None  # no chunking in decoder

    # Strange, I got OOM?
    train_args["batch_size"] = int(train_args["batch_size"] * 0.75)
    train_args["accum_grad"] = int(train_args.get("accum_grad", 2) * 1.5)

    if enc_stream_type == "causal" or enc_stream_type.startswith("causal-"):
        train_args["encoder_args"].use_causal_layers = True
        if enc_stream_type == "causal-reset-conv":
            train_args["encoder_args"].conv_alternative_name = "depthwise_conv2_causal"
            train_args.setdefault("retrain_checkpoint_opts", {}).setdefault("ignore_params_prefixes", []).extend(
                [
                    "conformer_block_%02i_conv_mod_depthwise_conv2_causal/" % (i + 1)
                    for i in range(train_args["encoder_args"].num_blocks)
                ]
            )

    decay_pt = int(total_epochs * decay_pt_factor)

    if enc_stream_type == "chunked":
        train_args["chunk_level"] = "input-encoder-only"  # TODO...

        # It needs more memory because there are mini batches
        # where the chunk size is larger than the sequences,
        # thus increasing the overall memory consumption of the whole encoder.
        train_args["batch_size"] = int(train_args["batch_size"] * 0.75)
        train_args["accum_grad"] = int(train_args.get("accum_grad", 2) * 1.5)
        raise NotImplementedError  # TODO

    train_args["learning_rates_list"] = [start_lr] * decay_pt + list(
        numpy.linspace(start_lr, 1e-6, total_epochs - decay_pt)
    )

    run_exp(
        prefix_name=prefix_name,
        exp_name=f"baseline"
        f"_enc-{enc_stream_type}-conf"
        f"_linDecay{total_epochs}_{start_lr}_decayPt{decay_pt_factor}"
        f"_fixed_align",
        train_args=train_args,
        num_epochs=total_epochs,
        epoch_wise_filter=None,
        time_rqmt=72,
        selected_datasets=["dev-other"],
        key="dev_score",
        use_sclite=True,
    )


def _run_exp_chunked_v1(
    *,
    enc_stream_type: str,
    chunk_size: int,
    chunk_step_factor: float,
    total_epochs: int,
):
    start_lr = 1e-4
    decay_pt_factor = 1 / 3

    ctc_align_wo_speed_pert = get_ctc_rna_based_chunk_alignments(
        chunk_sizes=[chunk_size],
        chunk_step_factors=[chunk_step_factor],
    )

    # train with ctc chunk-sync alignment
    train_args = copy.deepcopy(default_args)
    train_args["speed_pert"] = False  # no speed pert
    train_args["search_type"] = None  # fixed alignment

    train_args["max_seq_length"] = None  # no filtering!

    train_args["encoder_args"].with_ctc = False  # No CTC

    if enc_stream_type == "causal" or enc_stream_type.startswith("causal-"):
        train_args["encoder_args"].use_causal_layers = True
        if enc_stream_type == "causal-reset-conv":
            train_args["encoder_args"].conv_alternative_name = "depthwise_conv2_causal"
            train_args.setdefault("retrain_checkpoint_opts", {}).setdefault("ignore_params_prefixes", []).extend(
                [
                    "conformer_block_%02i_conv_mod_depthwise_conv2_causal/" % (i + 1)
                    for i in range(train_args["encoder_args"].num_blocks)
                ]
            )

    decay_pt = int(total_epochs * decay_pt_factor)

    train_args["chunk_size"] = chunk_size

    chunk_step = max(1, int(chunk_size * chunk_step_factor))
    train_args["chunk_step"] = chunk_step

    chunk_level = "input" if enc_stream_type == "chunked" else "encoder"
    train_args["chunk_level"] = chunk_level

    if chunk_level == "input":
        # It needs more memory because there are mini batches
        # where the chunk size is larger than the sequences,
        # thus increasing the overall memory consumption of the whole encoder.
        train_args["batch_size"] = int(train_args["batch_size"] * 0.75)
        train_args["accum_grad"] = int(train_args.get("accum_grad", 2) * 1.5)

    train_args["learning_rates_list"] = [start_lr] * decay_pt + list(
        numpy.linspace(start_lr, 1e-6, total_epochs - decay_pt)
    )

    run_exp(
        prefix_name=prefix_name,
        exp_name=f"chunk_att"
        f"_chunk-{chunk_size}_step-{chunk_step}"
        f"_enc-{enc_stream_type}-conf"
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


def sis_config_main():
    """sis config function"""
    _run_exp_baseline_v1(enc_stream_type="global", total_epochs=40)
    _run_exp_baseline_v1(enc_stream_type="causal", total_epochs=40)
    _run_exp_baseline_v1(enc_stream_type="causal-reset-conv", total_epochs=40)
    _run_exp_chunked_v1(enc_stream_type="chunked", chunk_size=20, chunk_step_factor=0.9, total_epochs=40)
    _run_exp_chunked_v1(enc_stream_type="causal", chunk_size=20, chunk_step_factor=0.9, total_epochs=40)
    _run_exp_chunked_v1(enc_stream_type="causal-reset-conv", chunk_size=20, chunk_step_factor=0.9, total_epochs=40)
    _run_exp_chunked_v1(enc_stream_type="global", chunk_size=20, chunk_step_factor=0.9, total_epochs=40)
    _run_exp_chunked_v1(enc_stream_type="chunked", chunk_size=50, chunk_step_factor=0.9, total_epochs=40)
    _run_exp_chunked_v1(enc_stream_type="causal", chunk_size=50, chunk_step_factor=0.9, total_epochs=40)
    _run_exp_chunked_v1(enc_stream_type="global", chunk_size=50, chunk_step_factor=0.9, total_epochs=40)
    _run_exp_chunked_v1(enc_stream_type="chunked", chunk_size=20, chunk_step_factor=0.9, total_epochs=100)
    _run_exp_chunked_v1(enc_stream_type="causal", chunk_size=20, chunk_step_factor=0.9, total_epochs=100)
    _run_exp_chunked_v1(enc_stream_type="causal-reset-conv", chunk_size=20, chunk_step_factor=0.9, total_epochs=100)
    _run_exp_chunked_v1(enc_stream_type="global", chunk_size=20, chunk_step_factor=0.9, total_epochs=100)
    _run_exp_chunked_v1(enc_stream_type="global", chunk_size=20, chunk_step_factor=0.5, total_epochs=40)


py = sis_config_main  # `py` is the default sis config function name
