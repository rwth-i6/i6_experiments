import copy
import os
import numpy as np
from sisyphus import tk
from dataclasses import asdict


from i6_core.tools.git import CloneGitRepositoryJob

from .data import build_training_dataset, TrainingDatasetSettings
from .config import get_training_config
from .pipeline import training
from ..data import get_tts_log_mel_datastream

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.audio import DBMelFilterbankOptions

from i6_experiments.users.rilling.experiments.librispeech.librispeech_glowtts.default_tools import RETURNN_PYTORCH_EXE, MINI_RETURNN_ROOT, RETURNN_COMMON
# from i6_experiments.users.rossenbach.experiments.jaist_project.storage import add_vocoder, VocoderPackage

# MINI_RETURNN_ROOT = CloneGitRepositoryJob("https://github.com/JackTemaki/MiniReturnn", commit="bfe4c7fcf6a17db951e4a28274737d92ae60a69f").out_repository
# MINI_RETURNN_ROOT.hash_overwrite="LIBRISPEECH_DEFAULT_RETURNN_ROOT"

def train_gl_vocoder():
    """
    Baseline for the ctc aligner in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    config = {
        "optimizer": {"class": "adam", "epsilon": 1e-16},
        "learning_rates": list(np.linspace(1e-3, 1e-6, 50)),
        #############
        "batch_size": 300 * 16000,
        "accum_grad_multiple_step": 5, # to have 1500 * 16000 as total batch size
        "max_seq_length": {"audio_features": 30 * 16000},
        "batch_drop_last": True,  # otherwise might cause issues in indexing after local sort in train_step
        "max_seqs": 200,
    }

    prefix = "experiments/librispeech/vocoder/simple_gl/"
    train_settings = TrainingDatasetSettings(
        custom_processing_function=None, partition_epoch=1, epoch_wise_filters=[], seq_ordering="laplace:.1000"
    )
    training_datasets = build_training_dataset(librispeech_key="train-clean-100", settings=train_settings)

    def run_exp(name, params, net_module, config, use_custom_engine=False, debug=False, num_epochs=50):
        train_config = get_training_config(
            training_datasets=training_datasets,
            network_module=net_module,
            net_args=params,
            config=config,
            debug=debug,
            use_custom_engine=use_custom_engine,
            returnn_common_root=RETURNN_COMMON,
            pytorch_mode=True
        )  # implicit reconstruction loss
        train_job = training(
            config=train_config,
            returnn_exe=RETURNN_PYTORCH_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            prefix=prefix + name,
            num_epochs=num_epochs,
        )

        return train_job

    log_mel_datastream = get_tts_log_mel_datastream(silence_preprocessing=False)

    # verify that normalization exists
    assert "norm_mean" in log_mel_datastream.additional_options
    assert "norm_std_dev" in log_mel_datastream.additional_options

    norm = (log_mel_datastream.additional_options["norm_mean"], log_mel_datastream.additional_options["norm_std_dev"])

    from ..pytorch_networks.vocoder.simple_gl.blstm_gl_predictor import BlstmGLPredictorConfig
    from ..feature_config import DbMelFeatureExtractionConfig
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

    net_module = "vocoder.simple_gl.blstm_gl_predictor"

    model_config = BlstmGLPredictorConfig(
        feature_extraction_config=fe_config,
        hidden_size=512,
    )
    params = {
        "config": asdict(model_config)
    }
    local_config = copy.deepcopy(config)
    train = run_exp(net_module + "_v1", params, net_module, config=local_config, debug=True)

    return {"checkpoint": train.out_checkpoints[50], "config": params["config"]}
    # train.hold()

    # package = VocoderPackage(checkpoint=train.out_checkpoints[50], config=params["config"])
    # add_vocoder("blstm_gl_v1", vocoder=package)

    ##################################
