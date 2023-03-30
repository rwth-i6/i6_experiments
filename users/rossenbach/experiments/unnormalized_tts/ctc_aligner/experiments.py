import numpy as np
import os
from sisyphus import tk

from .data import build_training_dataset
from .config import get_training_config, get_forward_config
from .pipeline import ctc_training, ctc_forward

from ..default_tools import RETURNN_EXE, RETURNN_ROOT, RETURNN_COMMON


from ..storage import add_duration


def get_baseline_ctc_alignment_v2():
    """
    Baseline for the ctc aligner in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """


    name = "experiments/unnormalized_tts/ctc_aligner_v2/baseline"
    base_name = "experiments/unnormalized_tts/ctc_aligner_v2/"

    training_datasets = build_training_dataset(center=True)

    aligner_config = get_training_config(
        returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets, network_kwargs={}, debug=True,
    )  # implicit reconstruction loss

    forward_config = get_forward_config(
        returnn_common_root=RETURNN_COMMON,
        forward_dataset=training_datasets.joint,
        datastreams=training_datasets.datastreams,
        network_kwargs={}
    )

    def run_baseline(name, train_config, forward_config):
        train_job = ctc_training(
            config=train_config,
            returnn_exe=RETURNN_EXE,
            returnn_root=RETURNN_ROOT,
            prefix=name,
        )
        duration_hdf = ctc_forward(
            checkpoint=train_job.out_checkpoints[100],
            config=forward_config,
            returnn_exe=RETURNN_EXE,
            returnn_root=RETURNN_ROOT,
            prefix=name
        )
        return duration_hdf

    run_baseline(name, aligner_config, forward_config)


    for reconstruction_scale in [0.0, 0.1, 0.2]:
        short_name = "oclr_v1_rec_%.2f" % reconstruction_scale
        network_kwargs = {
            "reconstruction_scale": reconstruction_scale
        }
        name = base_name + short_name
        aligner_config_oclr_v1 = get_training_config(
            returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets, network_kwargs=network_kwargs,
        )  # implicit reconstruction loss
        aligner_config_oclr_v1.config["learning_rates"] = list(np.linspace(1e-05, 2e-03, 50)) + list(np.linspace(2e-03, 1e-5, 50))
        run_baseline(name, aligner_config_oclr_v1, forward_config)


        short_name = "oclr_v2_rec_%.2f" % reconstruction_scale
        name = base_name + short_name
        aligner_config_oclr_v2 = get_training_config(
            returnn_common_root=RETURNN_COMMON, training_datasets=training_datasets, network_kwargs=network_kwargs,
        )  # implicit reconstruction loss
        aligner_config_oclr_v2.config["batch_size"] = 56000*160
        aligner_config_oclr_v2.config["gradient_clip"] = 2.0
        aligner_config_oclr_v2.config.pop("accum_grad_multiple_step")
        aligner_config_oclr_v2.config["learning_rates"] = list(np.linspace(1e-05, 2e-03, 50)) + list(np.linspace(2e-03, 1e-5, 50))
        duration_hdf = run_baseline(name, aligner_config_oclr_v2, forward_config)

        add_duration(short_name, duration_hdf)




