import copy
import os
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from .data import build_training_dataset
from .config import get_training_config, get_extract_durations_forward__config, get_forward_config
from .pipeline import glowTTS_training, glowTTS_forward

from ..default_tools import RETURNN_COMMON, RETURNN_PYTORCH_EXE, MINI_RETURNN_ROOT

from i6_experiments.users.rossenbach.experiments.alignment_analysis_tts.gl_vocoder.default_vocoder import get_default_vocoder


def get_pytorch_glowTTS():
    """
    Baseline for the glow TTS in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    config = {
        "optimizer": {"class": "adam", "epsilon": 1e-8},
        "learning_rate_control": "newbob_multi_epoch",
        "learning_rate_control_min_num_epochs_per_new_lr": 5,
        "learning_rate_control_relative_error_relative_lr": True,
        "learning_rates": [0.001],
        "gradient_clip_norm": 2.0,
        "use_learning_rate_control_always": True,
        "learning_rate_control_error_measure": "dev_loss_mle",
        ############
        "newbob_learning_rate_decay": 0.9,
        "newbob_multi_num_epochs": 5,
        "newbob_multi_update_interval": 1,
        "newbob_relative_error_threshold": 0,
        #############
        # "batch_size": 56000,
        "batch_size": 28000,
        "max_seq_length": {"audio_features": 1600},
        "max_seqs": 200,
    }

    prefix = "experiments/librispeech/tts_architecture/glow_tts/pytorch/"

    def run_exp(name, params, net_module, config, dataset, num_epochs=100, use_custom_engine=False, debug=False):
        tts_config = get_training_config(
            returnn_common_root=RETURNN_COMMON,
            training_datasets=dataset,
            network_module=net_module,
            net_args=params,
            config=config,
            debug=debug,
            use_custom_engine=use_custom_engine,
            pytorch_mode=True
        )  # implicit reconstruction loss
        forward_config = get_forward_config(
            returnn_common_root=RETURNN_COMMON,
            forward_dataset=dataset,
            network_module=net_module,
            net_args=params,
            debug=debug,
            pytorch_mode=True
        )
        train_job = glowTTS_training(
            config=tts_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name,
            num_epochs=num_epochs
        )
        
        tts_hdf = glowTTS_forward(
            checkpoint=train_job.out_checkpoints[num_epochs],
            config=forward_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name
        )

        # name = "vocoderTest"
        # vocoder = get_default_vocoder(name)
        # forward_vocoded, vocoder_forward_job = vocoder.vocode(
        #     tts_hdf, iterations=30, cleanup=True, name=name
        # )

        # tk.register_output(name + "/forward_dev_corpus.xml.gz", forward_vocoded)

        return tts_hdf

    net_module = "glowTTS"
    training_datasets = build_training_dataset(silence_preprocessed=True, center=True)
    params = {
        "n_vocab": training_datasets.datastreams["phonemes"].vocab_size,
        "hidden_channels": 192,
        "filter_channels": 192, 
        "filter_channels_dp": 256,
        "out_channels": 80,
        "n_speakers": 251,
        "gin_channels": 256,
        "p_dropout": 0.1,
        "p_dropout_decoder": 0.05,
        "dilation_rate": 1,
        "n_sqz": 2,
        "prenet": True,
        "window_size": 4
    }

    # tts_hdf = run_exp(net_module, params, net_module, config, dataset=training_datasets, debug=True)
   
    net_module = "glowTTS_v2"

    training_datasets = build_training_dataset(silence_preprocessed=True, durations_file="/work/asr4/rossenbach/sisyphus_work_folders/tts_asr_2021_work/i6_experiments/users/rossenbach/tts/duration_extraction/ViterbiAlignmentToDurationsJob.AyAO6JWXTnVc/output/durations.hdf", center=False)

    tts_hdf = run_exp(name=net_module + "injected_durations_AMP", params=params, net_module=net_module, config=config, dataset=training_datasets, debug=True)
    
    return tts_hdf