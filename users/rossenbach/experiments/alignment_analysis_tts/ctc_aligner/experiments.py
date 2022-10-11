import os
from sisyphus import tk

from .data import build_training_dataset
from .config import get_training_config, get_forward_config
from .pipeline import ctc_training, ctc_forward

from ..default_tools import RETURNN_EXE, RETURNN_RC_ROOT, RETURNN_COMMON, RETURNN_DATA_ROOT


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


def get_baseline_ctc_alignment_v2():
    """
    Baseline for the ctc aligner in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """


    name = "experiments/alignment_analysis_tts/ctc_aligner_v2/baseline"

    training_datasets = build_training_dataset()


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
    forward = ctc_forward(
        checkpoint=train_job.out_checkpoints[100],
        config=forward_config,
        returnn_exe=RETURNN_EXE,
        returnn_root=RETURNN_RC_ROOT,
        prefix=name
    )


def get_loss_scale_ctc_alignment():
    """
    Baseline for the ctc aligner in returnn_common with serialization
    :return: durations_hdf
    """
    base_name = "experiments/alignment_analysis_tts/ctc_aligner/baseline"

    for center in [True, False]:
        training_datasets = build_training_dataset(center=center)
        center_name = base_name + f"_{center}"
        for reconstruction_loss_scale in [0.0, 0.25, 0.5, 1.0]:
            name = center_name + f"_rloss{reconstruction_loss_scale:.2f}"
            aligner_config = get_training_config(
                returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets,
                reconstruction_scale=reconstruction_loss_scale,
            )  # explicit reconstruction loss
            train_job = ctc_training(
                config=aligner_config,
                returnn_exe=RETURNN_EXE,
                returnn_root=RETURNN_RC_ROOT,
                prefix=name,
            )
