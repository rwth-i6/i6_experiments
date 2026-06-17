import copy
import os
import numpy as np
from sisyphus import tk
from dataclasses import asdict


from i6_core.tools.git import CloneGitRepositoryJob

from i6_experiments.users.rossenbach.experiments.jaist_project.data.aligner import build_training_dataset
from i6_experiments.users.rossenbach.experiments.jaist_project.config import get_training_config, get_prior_config, get_forward_config
from i6_experiments.users.rossenbach.experiments.jaist_project.pipeline import training, extract_durations, tts_eval
from i6_experiments.users.rossenbach.experiments.jaist_project.data.tts_phon import get_tts_log_mel_datastream

from i6_experiments.common.setups.returnn.datastreams.audio import DBMelFilterbankOptions

from i6_experiments.users.rossenbach.experiments.jaist_project.default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from i6_experiments.users.rossenbach.experiments.jaist_project.storage import add_duration

MINI_RETURNN_ROOT = CloneGitRepositoryJob("https://github.com/JackTemaki/MiniReturnn", commit="c1d208106c281a8586826b5309899b960036a6d4").out_repository
MINI_RETURNN_ROOT.hash_overwrite="LIBRISPEECH_DEFAULT_RETURNN_ROOT"



def train_vocoder():
    """
    Baseline for the ctc aligner in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    config = {
        "updater": {
            "disc": {
                "optimizer": {"class": "adam", "epsilon": 1e-12, "betas": (0.8, 0.99)},
                "submodel": "discriminator",
            },
            "gen": {
                "optimizer": {"class": "adam", "epsilon": 1e-12, "betas": (0.8, 0.99)},
                "submodel": "generator",
            },
        },
        "chunking_options": {
            "chunk_streams": {
                "raw_audio":{
                    "step": 16000,
                    "size": 16000,
                    "min_chunk_size": 16000,
                },
            },
            "random_chunk_start": True,
        },
        "learning_rates":  list(0.0002 * np.cumprod([0.995] * 200)),
        #############
        "batch_size": 300 * 16000,
        "max_seq_length": {"audio_features": 30 * 16000},
        "batch_drop_last": True,  # otherwise might cause issues in indexing after local sort in train_step
        "max_seqs": 200,

    }

    prefix = "experiments/jaist_project/vocoder/univnet/"
    training_datasets = build_training_dataset(ls_corpus_key="train-clean-100", partition_epoch=1)

    def run_exp(name, params, net_module, config, post_config, use_custom_engine=False, debug=False, num_epochs=200):
        train_config = get_training_config(
            training_datasets=training_datasets,
            network_module=net_module,
            net_args=params,
            config=config,
            post_config=post_config,
            debug=debug,
            use_custom_engine=use_custom_engine,
        )  # implicit reconstruction loss
        train_job = training(
            returnn_config=train_config,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix_name=prefix + name,
            num_epochs=num_epochs,
        )
        return train_job

    log_mel_datastream = get_tts_log_mel_datastream(ls_corpus_key="train-clean-100", silence_preprocessed=False)

    # verify that normalization exists
    assert "norm_mean" in log_mel_datastream.additional_options
    assert "norm_std_dev" in log_mel_datastream.additional_options

    norm = (log_mel_datastream.additional_options["norm_mean"], log_mel_datastream.additional_options["norm_std_dev"])

    from ...pytorch_networks.vocoder.univnet.timur_univnet import UnivNetConfig, UnivNetGeneratorConfig, DbMelFeatureExtractionConfig
    assert isinstance(log_mel_datastream.options.feature_options, DBMelFilterbankOptions)
    fe_config = DbMelFeatureExtractionConfig(
        sample_rate=log_mel_datastream.options.sample_rate,
        win_size=log_mel_datastream.options.window_len,
        hop_size=log_mel_datastream.options.step_len,
        f_min=log_mel_datastream.options.feature_options.fmin,
        f_max=log_mel_datastream.options.feature_options.fmax,
        min_amp=log_mel_datastream.options.feature_options.min_amp,
        num_filters=log_mel_datastream.options.num_feature_filters,
        center=log_mel_datastream.options.feature_options.center,
        norm=norm
    )

    net_module = "vocoder.univnet.timur_univnet"

    model_config = UnivNetConfig(
        generator_config=UnivNetGeneratorConfig(
            cond_in_channels=64,
            out_channels=1,
            cg_channels=32,
            num_ff=80,
            upsample_rates=[8,5,5],
            num_lvc_blocks=4,
            lvc_kernels=3,
            lvc_hidden_channels=64,
            lvc_conv_size=3,
            dropout=0.0,
        ),
        feature_extraction_config=fe_config,
        mel_loss_scale=2.5,
        start_discriminator_epoch=100,
    )
    model_post_config = {
        "cleanup_old_models": {
            "keep": [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        }
    }

    params = {
        "config": asdict(model_config)
    }
    local_config = copy.deepcopy(config)
    train = run_exp(net_module + "_v1", params, net_module, config=local_config, post_config=model_post_config, debug=True,
                    use_custom_engine="vocoder.univnet.gan_engine")
    train.hold()


    ##################################
    
    model_config = UnivNetConfig(
        generator_config=UnivNetGeneratorConfig(
            cond_in_channels=64,
            out_channels=1,
            cg_channels=32,
            num_ff=80,
            upsample_rates=[8,5,5],
            num_lvc_blocks=4,
            lvc_kernels=3,
            lvc_hidden_channels=64,
            lvc_conv_size=3,
            dropout=0.0,
        ),
        feature_extraction_config=fe_config,
        mel_loss_scale=2.5,
        start_discriminator_epoch=1,
    )
    model_post_config = {
        "cleanup_old_models": {
            "keep": [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        }
    }

    params = {
        "config": asdict(model_config)
    }
    local_config = copy.deepcopy(config)
    local_config["batch_size"] = 30 * 16000
    train = run_exp(net_module + "_v1_all_disc", params, net_module, config=local_config, post_config=model_post_config, debug=True,
                    use_custom_engine="vocoder.univnet.gan_engine")
    train.hold()