import copy
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from i6_experiments.common.setups.returnn.datastreams.audio import DBMelFilterbankOptions

from i6_experiments.users.rossenbach.experiments.jaist_project.data.tts_phon import build_durationtts_training_dataset
from i6_experiments.users.rossenbach.experiments.jaist_project.data.tts_phon import get_vocab_datastream
from i6_experiments.users.rossenbach.experiments.jaist_project.data.tts_phon import get_tts_log_mel_datastream

from i6_experiments.users.rossenbach.experiments.jaist_project.config import get_training_config, get_forward_config
from i6_experiments.users.rossenbach.experiments.jaist_project.pipeline import training, tts_eval_v2, generate_synthetic, tts_training

from i6_experiments.users.rossenbach.experiments.jaist_project.default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from i6_experiments.users.rossenbach.experiments.jaist_project.storage import duration_alignments, vocoders


def run_tacotron2_like_tts():
    """
    """

    config = {
        "optimizer": {"class": "adam", "epsilon": 1e-9},
        "learning_rates":  list(np.linspace(5e-5, 5e-4, 100)) + list(np.linspace(5e-4, 5e-7, 100)),
        # "gradient_clip": 1.0,
        "gradient_clip_norm": 2.0,
        "use_learning_rate_control_always": True,
        #############
        "batch_size": 600 * 16000,
        "max_seq_length": {"audio_features": 30 * 16000},
        "batch_drop_last": True,  # otherwise might cause issues in indexing after local sort in train_step
        "max_seqs": 200,
    }

    prefix = "experiments/jaist_project/tts/nar/tacotron2_like/"

    def run_exp(name, params, net_module, config, duration_hdf, decoder_options, extra_decoder=None, use_custom_engine=False, debug=False, num_epochs=200):
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
            num_epochs=num_epochs
        )
        forward_job = tts_eval_v2(
            prefix_name=prefix + name,
            returnn_config=forward_config,
            checkpoint=train_job.out_checkpoints[num_epochs],
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

    net_module = "nar_tts.tacotron2_like.tacotron2_like_vanilla_blstm"

    from ...pytorch_networks.nar_tts.tacotron2_like.tacotron2_like_vanilla_blstm import (
        NarTacotronDecoderConfig,
        Config
    )
    from ...pytorch_networks.tts_shared.tts_base_model.base_model_v1 import (
        DbMelFeatureExtractionConfig,
        TTSTransformerTextEncoderV1Config,
        SimpleConvDurationPredictorV1Config
    )
    from ...pytorch_networks.tts_shared.encoder.prenet import TTSEncoderPreNetV1Config
    from ...pytorch_networks.tts_shared.encoder.rel_mhsa import GlowTTSMultiHeadAttentionV1Config

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

    prenet_config = TTSEncoderPreNetV1Config(
        input_embedding_size=256,
        hidden_dimension=256,
        kernel_size=5,
        num_layers=3,
        dropout=0.5,
        output_dimension=256,
    )
    mhsa_config = GlowTTSMultiHeadAttentionV1Config(
        input_dim=256,
        num_att_heads=2,
        dropout=0.1,
        att_weights_dropout=0.1,
        window_size=4,  # one-sided, so technically 9
        heads_share=True,
    )
    encoder_config = TTSTransformerTextEncoderV1Config(
        num_layers=6,
        vocab_size=get_vocab_datastream(with_blank=True).vocab_size,
        basic_dim=256,
        conv_dim=1024,
        conv_kernel_size=3,
        dropout=0.1,
        mhsa_config=mhsa_config,
        prenet_config=prenet_config,
    )
    duration_predictor_config = SimpleConvDurationPredictorV1Config(
        num_convs=2,
        hidden_dim=384,
        kernel_size=3,
        dropout=0.1,
    )

    decoder_config = NarTacotronDecoderConfig(
        target_channels=log_mel_datastream.options.num_feature_filters,
        num_lstm_layers=2,
        basic_dim=768,
        lstm_dropout=0.1,
        post_conv_dim=512,
        post_conv_kernel_size=5,
        post_conv_num_layers=5,
        post_conv_dropout=0.5,
        reduction_factor=2
    )

    model_config = Config(
        speaker_embedding_size=256,
        num_speakers=251,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        duration_predictor_config=duration_predictor_config,
        feature_extraction_config=fe_config,
    )

    model_config_512 = copy.deepcopy(model_config)
    model_config_512.decoder_config.basic_dim = 512

    params = {
        "config": asdict(model_config)
    }
    params512 = {
        "config": asdict(model_config_512)
    }

    decoder_options = {
        "norm_mean": norm[0],
        "norm_std_dev": norm[1]
    }

    decoder_options = copy.deepcopy(decoder_options)
    vocoder = vocoders["blstm_gl_v1"]
    decoder_options["gl_net_checkpoint"] = vocoder.checkpoint
    decoder_options["gl_net_config"] = vocoder.config

    decoder_options_synthetic = copy.deepcopy(decoder_options)
    decoder_options_synthetic["gl_momentum"] = 0.0
    decoder_options_synthetic["gl_iter"] = 1
    decoder_options_synthetic["create_plots"] = False

    local_config = copy.deepcopy(config)
    duration_hdf = duration_alignments["glow_tts.glow_tts_v1_bs600_v2_base256"]
    train, forward = run_exp(net_module + "_glow256align_200eps_bs600_oclr", params, net_module,
                             local_config,
                             extra_decoder="nar_tts.tacotron2_like.simple_gl_decoder", decoder_options=decoder_options,
                             duration_hdf=duration_hdf, debug=True)
    
    generate_synthetic(prefix, net_module + "_glow256align_200eps_bs600_oclr_syn", "train-clean-100",
                       train.out_checkpoints[200], params, net_module,
                       extra_decoder="nar_tts.tacotron2_like.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic, debug=True)

    config_400eps = copy.deepcopy(config)
    config_400eps["learning_rates"] = list(np.linspace(5e-5, 5e-4, 100)) + list(np.linspace(5e-4, 5e-7, 300))

    train, forward = tts_training(prefix, net_module + "_size512_glow256align_400eps_bs600_oclr", params512, net_module,
                                  config_400eps,
                                  extra_decoder="nar_tts.tacotron2_like.simple_gl_decoder", decoder_options=decoder_options,
                                  duration_hdf=duration_hdf, debug=True, num_epochs=400, evaluate_swer="ls960eow_phon_ctc_50eps_fastsearch")

    decoder_options_final_gl32 = copy.deepcopy(decoder_options)
    decoder_options_final_gl32["gl_momentum"] = 0.99
    decoder_options_final_gl32["gl_iter"] = 32
    decoder_options_final_gl32["create_plots"] = False

    generate_synthetic(prefix, net_module + "_size512_glow256align_400eps_bs600_oclr_gl32_syn", "train-clean-100",
                       train.out_checkpoints[400], params512, net_module,
                       extra_decoder="nar_tts.tacotron2_like.simple_gl_decoder",
                       decoder_options=decoder_options_final_gl32, debug=True)

    generate_synthetic(prefix, net_module + "_size512_glow256align_400eps_bs600_oclr_gl32_syn_fixspk", "train-clean-100",
                       train.out_checkpoints[400], params512, net_module,
                       extra_decoder="nar_tts.tacotron2_like.simple_gl_decoder",
                       decoder_options=decoder_options_final_gl32, debug=True, randomize_speaker=False)

    generate_synthetic(prefix, net_module + "_size512_glow256align_400eps_bs600_oclr_gl32_syn", "train-clean-360",
                       train.out_checkpoints[400], params512, net_module,
                       extra_decoder="nar_tts.tacotron2_like.simple_gl_decoder",
                       decoder_options=decoder_options_final_gl32, debug=True)

    generate_synthetic(prefix, net_module + "_size512_glow256align_400eps_bs600_oclr_gl32_syn", "train-clean-360",
                       train.out_checkpoints[400], params512, net_module,
                       extra_decoder="nar_tts.tacotron2_like.simple_gl_decoder",
                       decoder_options=decoder_options_final_gl32, debug=True, use_subset=True)
    
    
    

    # variant with zoneout

    net_module = "nar_tts.tacotron2_like.tacotron2_like_zoneout_blstm"
    from ...pytorch_networks.nar_tts.tacotron2_like.tacotron2_like_zoneout_blstm import (
        NarTacotronDecoderConfig,
        Config
    )
    decoder_config = NarTacotronDecoderConfig(
        target_channels=log_mel_datastream.options.num_feature_filters,
        num_lstm_layers=2,
        basic_dim=512,
        post_conv_dim=512,
        post_conv_kernel_size=5,
        post_conv_num_layers=5,
        post_conv_dropout=0.5,
        reduction_factor=2,
        zoneout_rate=0.1,
    )

    model_config = Config(
        speaker_embedding_size=256,
        num_speakers=251,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        duration_predictor_config=duration_predictor_config,
        feature_extraction_config=fe_config,
    )
    params512_zoneout = {
        "config": asdict(model_config)
    }

    # train, forward = tts_training(prefix, net_module + "_size512_glow256align_400eps_bs600_oclr", params512_zoneout, net_module,
    #                               config_400eps,
    #                               extra_decoder="nar_tts.tacotron2_like.simple_gl_decoder", decoder_options=decoder_options,
    #                               duration_hdf=duration_hdf, debug=True, num_epochs=400, evaluate_swer="ls960eow_phon_ctc_50eps_fastsearch")