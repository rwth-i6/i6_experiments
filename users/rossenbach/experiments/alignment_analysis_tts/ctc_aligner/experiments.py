import os
import numpy as np
from sisyphus import tk

from .data import build_training_dataset
from .config import get_training_config, get_forward_config
from .pipeline import ctc_training, ctc_forward

from ..default_tools import RETURNN_EXE, RETURNN_RC_ROOT, RETURNN_COMMON, RETURNN_DATA_ROOT
from ..storage import add_duration

def get_baseline_ctc_alignment():
    """
    Baseline for the ctc aligner in returnn_common with serialization
    :return: durations_hdf
    """


    name = "experiments/alignment_analysis_tts/ctc_aligner/baseline"

    training_datasets = build_training_dataset()


    aligner_config = get_training_config(
        returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets
    )  # implicit reconstruction loss
    forward_config = get_forward_config(
        returnn_common_root=RETURNN_COMMON,
        forward_dataset=training_datasets.joint,
        datastreams=training_datasets.datastreams
    )
    train_job = ctc_training(
        config=aligner_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_RC_ROOT,
        prefix=name,
    )
    forward = ctc_forward(
        checkpoint=train_job.out_checkpoints[100],
        config=forward_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_RC_ROOT,
        prefix=name
    )


def get_baseline_ctc_alignment_v2(silence_preprocessed=True):
    """
    Baseline for the ctc aligner in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """


    name = "experiments/alignment_analysis_tts/ctc_aligner_v2/baseline"

    if silence_preprocessed:
        name += "_spp"
    else:
        name += "_nospp"

    training_datasets = build_training_dataset(silence_preprocessed=silence_preprocessed)

    aligner_config = get_training_config(
        returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets, use_v2=True,
    )  # implicit reconstruction loss
    forward_config = get_forward_config(
        returnn_common_root=RETURNN_COMMON,
        forward_dataset=training_datasets.joint,
        datastreams=training_datasets.datastreams,
        use_v2=True,
    )
    train_job = ctc_training(
        config=aligner_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_RC_ROOT,
        prefix=name,
    )
    duration_hdf = ctc_forward(
        checkpoint=train_job.out_checkpoints[100],
        config=forward_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_RC_ROOT,
        prefix=name
    )
    return duration_hdf


def get_ls460_ctc_alignment_v2():
    """
    Baseline for the ctc aligner in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """


    name_prefix = "experiments/alignment_analysis_tts/ctc_aligner_v2/"

    training_datasets = build_training_dataset(ls_corpus_key="train-clean-460")
    training_datasets_pe3 = build_training_dataset(ls_corpus_key="train-clean-460", partition_epoch=3)


    def run_exp(name, training_config, net_args):
        local_name = name_prefix + name
        forward_config = get_forward_config(
            returnn_common_root=RETURNN_COMMON,
            forward_dataset=training_datasets.joint,
            datastreams=training_datasets.datastreams,
            use_v2=True,
            net_args=net_args
        )
        train_job = ctc_training(
            config=training_config,
            returnn_exe=RETURNN_EXE,
            returnn_root=RETURNN_RC_ROOT,
            prefix=local_name,
        )
        duration_hdf = ctc_forward(
            checkpoint=train_job.out_checkpoints[100],
            config=forward_config,
            returnn_exe=RETURNN_EXE,
            returnn_root=RETURNN_RC_ROOT,
            prefix=local_name
        )

        add_duration(name, duration_hdf)


    net_args = {}
    aligner_config = get_training_config(
        returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets, use_v2=True,
    )  # implicit reconstruction loss

    run_exp("ls460", aligner_config, net_args)

    aligner_config_pe3 = get_training_config(
        returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets_pe3, use_v2=True,
    )  # implicit reconstruction loss
    run_exp("ls460_pe3", aligner_config_pe3, net_args)

    aligner_config_pe3_oclr = get_training_config(
        returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets_pe3, use_v2=True,
    )  # implicit reconstruction loss
    aligner_config_pe3_oclr.config["learning_rates"] = list(np.linspace(0.0002, 0.002, 40)) + list(np.linspace(0.002, 0.0002, 40))
    run_exp("ls460_pe3_oclrv1", aligner_config_pe3_oclr, net_args)
    

    # Went instable at 0.0028

    #aligner_config_pe3_oclr_v2 = get_training_config(
    #    returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets_pe3, use_v2=True,
    #)  # implicit reconstruction loss
    #aligner_config_pe3_oclr_v2.config["learning_rates"] = list(np.linspace(0.0003, 0.003, 40)) + list(np.linspace(0.003, 0.00003, 60))
    #net_args_norec = {
    #    "reconstruction_scale": 0,
    #}
    #run_exp("ls460_pe3_scale0_oclrv2", aligner_config_pe3_oclr_v2, net_args_norec)


def get_ls960_ctc_alignment_v2():
    """
    Baseline for the ctc aligner in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    name_prefix = "experiments/alignment_analysis_tts/ctc_aligner_v2/"

    training_datasets_pe6 = build_training_dataset(ls_corpus_key="train-other-960", partition_epoch=6)

    def run_exp(name, training_config, net_args):
        local_name = name_prefix + name
        forward_config = get_forward_config(
            returnn_common_root=RETURNN_COMMON,
            forward_dataset=training_datasets_pe6.joint,
            datastreams=training_datasets_pe6.datastreams,
            use_v2=True,
            net_args=net_args
        )
        train_job = ctc_training(
            config=training_config,
            returnn_exe=RETURNN_EXE,
            returnn_root=RETURNN_RC_ROOT,
            prefix=local_name,
        )
        duration_hdf = ctc_forward(
            checkpoint=train_job.out_checkpoints[100],
            config=forward_config,
            returnn_exe=RETURNN_EXE,
            returnn_root=RETURNN_RC_ROOT,
            prefix=local_name
        )

        add_duration(name, duration_hdf)

    net_args = {}
    #aligner_config = get_training_config(
    #    returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets, use_v2=True,
    #)  # implicit reconstruction loss

    #run_exp("ls460", aligner_config, net_args)

    #aligner_config_pe3 = get_training_config(
    #    returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets_pe3, use_v2=True,
    #)  # implicit reconstruction loss
    #run_exp("ls460_pe3", aligner_config_pe3, net_args)

    aligner_config_pe3_oclr = get_training_config(
        returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets_pe6, use_v2=True,
    )  # implicit reconstruction loss
    aligner_config_pe3_oclr.config["learning_rates"] = list(np.linspace(0.0002, 0.002, 40)) + list(
        np.linspace(0.002, 0.0002, 40))
    run_exp("ls960_pe6_oclrv1", aligner_config_pe3_oclr, net_args)


    #aligner_config_pe3_oclr_v2 = get_training_config(
    #    returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets_pe3, use_v2=True,
    #)  # implicit reconstruction loss
    #aligner_config_pe3_oclr_v2.config["learning_rates"] = list(np.linspace(0.0003, 0.003, 40)) + list(
    #    np.linspace(0.003, 0.00003, 60))
    #net_args_norec = {
    #    "reconstruction_scale": 0,
    #}
    #run_exp("ls460_pe3_scale0_oclrv2", aligner_config_pe3_oclr_v2, net_args_norec)


def get_baseline_ctc_alignment_v2_centered():
    """
    Baseline for the ctc aligner in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """


    name = "experiments/alignment_analysis_tts/ctc_aligner_v2/baseline_center"

    training_datasets = build_training_dataset(center=True)

    aligner_config = get_training_config(
        returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets, use_v2=True,
    )  # implicit reconstruction loss
    forward_config = get_forward_config(
        returnn_common_root=RETURNN_COMMON,
        forward_dataset=training_datasets.joint,
        datastreams=training_datasets.datastreams,
        use_v2=True,
    )
    train_job = ctc_training(
        config=aligner_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_RC_ROOT,
        prefix=name,
    )
    duration_hdf = ctc_forward(
        checkpoint=train_job.out_checkpoints[100],
        config=forward_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_RC_ROOT,
        prefix=name
    )
    return duration_hdf


def get_loss_scale_ctc_alignment():
    """
    Baseline for the ctc aligner in returnn_common with serialization
    :return: durations_hdf
    """
    base_name = "experiments/alignment_analysis_tts/ctc_aligner/baseline"

    duration_hdfs = {}

    for center in [True, False]:
        training_datasets = build_training_dataset(center=center)
        center_name = base_name + f"_{center}"
        for reconstruction_loss_scale in [0.0, 0.25, 0.5, 1.0]:
            name = center_name + f"_rloss{reconstruction_loss_scale:.2f}"
            aligner_config = get_training_config(
                returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets,
                net_args={"reconstruction_scale": reconstruction_loss_scale}, use_v2=True,
            )  # explicit reconstruction loss
            forward_config = get_forward_config(
                returnn_common_root=RETURNN_COMMON,
                forward_dataset=training_datasets.joint,
                datastreams=training_datasets.datastreams,
                use_v2=True,
            )
            train_job = ctc_training(
                config=aligner_config,
                returnn_exe=RETURNN_EXE,
                returnn_root=RETURNN_RC_ROOT,
                prefix=name,
            )
            duration_hdf = ctc_forward(
                checkpoint=train_job.out_checkpoints[100],
                config=forward_config,
                returnn_exe=RETURNN_EXE,
                returnn_root=RETURNN_RC_ROOT,
                prefix=name
            )
            duration_hdfs[f"center-{center}_loss-{reconstruction_loss_scale:.1f}"] = duration_hdf

    return duration_hdfs
