import copy
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from i6_experiments.common.setups.returnn.datastreams.audio import DBMelFilterbankOptions

from ...data.tts_phon import build_durationtts_training_dataset
from ...data.tts_phon import get_vocab_datastream
from ...data.tts_phon import get_tts_log_mel_datastream

from ...config import get_training_config, get_forward_config
from ...pipeline import training, tts_eval_v2, generate_synthetic

from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...storage import duration_alignments, vocoders


def run_tacotron2_like_tts():
    """
    New setup using the shared encoder for fastspeech style NAR-TTS system
    """

    config = {
        "optimizer": {"class": "adam", "epsilon": 1e-9},
        "learning_rates": list(np.linspace(1e-4, 1e-3, 100)) + list(
            np.linspace(1e-3, 1e-6, 100)),
        # "gradient_clip": 1.0,
        "gradient_clip_norm": 2.0,
        "use_learning_rate_control_always": True,
        #############
        "batch_size": 300 * 16000,
        "max_seq_length": {"audio_features": 30 * 16000},
        "batch_drop_last": True,  # otherwise might cause issues in indexing after local sort in train_step
        "max_seqs": 200,
    }

    prefix = "experiments/jaist_project/standalone_2024/nar_tts/fastspeech_like/"

    def run_exp(name, params, net_module, config, duration_hdf, decoder_options, extra_decoder=None, use_custom_engine=False, debug=False):
        training_datasets = build_durationtts_training_dataset(duration_hdf=duration_hdf)
        training_config = get_training_config(
            training_datasets=training_datasets,
            network_module=net_module,
            net_args=params,
            config=config,
            debug=debug,
            use_custom_engine=use_custom_engine,
        )  # implicit reconstruction loss
        forward_config = get_forward_config(
            network_module=net_module,
            net_args=params,
            decoder=extra_decoder or net_module,
            decoder_args=decoder_options,
            config={
                "forward": training_datasets.cv.as_returnn_opts()
            },
            debug=debug,
        )
        train_job = training(
            prefix_name=prefix + name,
            returnn_config=training_config,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            num_epochs=200
        )
        forward_job = tts_eval_v2(
            prefix_name=prefix + name,
            returnn_config=forward_config,
            checkpoint=train_job.out_checkpoints[200],
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
        )
        tk.register_output(prefix + name + "/audio_files", forward_job.out_files["audio_files"])
        return train_job, forward_job

    log_mel_datastream = get_tts_log_mel_datastream(ls_corpus_key="train-clean-100")

    # verify that normalization exists
    assert "norm_mean" in log_mel_datastream.additional_options
    assert "norm_std_dev" in log_mel_datastream.additional_options

    norm = (log_mel_datastream.additional_options["norm_mean"], log_mel_datastream.additional_options["norm_std_dev"])

    net_module = "nar_tts.fastspeech_like.fastspeech_like_v1"

    from ...pytorch_networks.nar_tts.fastspeech_like.fastspeech_like_v1 import (
        DbMelFeatureExtractionConfig,
        FastSpeechDecoderConfig,
        TTSTransformerTextEncoderV1Config,
        GlowTTSMultiHeadAttentionV1Config,
        SimpleConvDurationPredictorV1Config,
        Config
    )
    from ...pytorch_networks.tts_shared.encoder.prenet import TTSEncoderPreNetV1Config

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