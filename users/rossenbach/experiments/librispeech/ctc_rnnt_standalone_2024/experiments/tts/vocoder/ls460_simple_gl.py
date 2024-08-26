import copy
import os
import numpy as np
from sisyphus import tk
from dataclasses import asdict


from i6_core.tools.git import CloneGitRepositoryJob

from i6_experiments.common.setups.returnn.datastreams.audio import DBMelFilterbankOptions

from ....data.tts.aligner import build_training_dataset
from ....config import get_training_config, get_prior_config, get_forward_config
from ....pipeline import training
from ....data.tts.tts_phon import get_tts_log_mel_datastream

from ....default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ....storage import add_vocoder, VocoderPackage


def train_gl_vocoder_ls460():
    """
    Baseline for the ctc aligner in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    config = {
        "optimizer": {"class": "adam", "epsilon": 1e-16},
        "learning_rates":   list(np.linspace(1e-3, 1e-6, 50)),
        #############
        "batch_size": 1000 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "batch_drop_last": True,  # otherwise might cause issues in indexing after local sort in train_step
        "max_seqs": 200,
    }

    prefix = "experiments/librispeech/ctc_rnnt_standalone_2024/vocoder/simple_gl/"
    training_datasets = build_training_dataset(ls_corpus_key="train-clean-100", partition_epoch=2)

    def run_exp(name, params, net_module, config, debug=False, num_epochs=50):
        train_args = {
            "network_module": net_module,
            "net_args": params,
            "config": config,
            "debug": debug,
        }
        train_job = training(
            training_name=prefix + name,
            datasets=training_datasets,
            train_args=train_args,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            num_epochs=num_epochs,
        )
        return train_job

    log_mel_datastream = get_tts_log_mel_datastream(ls_corpus_key="train-clean-460", silence_preprocessed=False)

    # verify that normalization exists
    assert "norm_mean" in log_mel_datastream.additional_options
    assert "norm_std_dev" in log_mel_datastream.additional_options

    norm = (log_mel_datastream.additional_options["norm_mean"], log_mel_datastream.additional_options["norm_std_dev"])

    from ....pytorch_networks.vocoder.simple_gl.blstm_gl_predictor import BlstmGLPredictorConfig, DbMelFeatureExtractionConfig
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

    package = VocoderPackage(checkpoint=train.out_checkpoints[50], config=params["config"])
    add_vocoder("blstm_gl_v1_ls460", vocoder=package)
    ##################################