import copy
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from i6_experiments.common.setups.returnn.datastreams.audio import DBMelFilterbankOptions

from ...data.tts_phon import build_durationtts_training_dataset
from ...data.tts_phon import get_vocab_datastream
from ...data.tts_phon import get_tts_log_mel_datastream

from ...config import get_training_config, get_forward_config
from ...pipeline import training, tts_eval_v2, generate_synthetic, cross_validation_nisqa

from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...storage import duration_alignments, vocoders


def run_fastspeech_like_tts():
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

    mhsa_decoder_config = GlowTTSMultiHeadAttentionV1Config(
        input_dim=256,
        num_att_heads=2,
        dropout=0.1,
        att_weights_dropout=0.1,
        window_size=16,  # one-sided, so technically 9
        heads_share=True,
    )

    decoder_config = FastSpeechDecoderConfig(
        target_channels=log_mel_datastream.options.num_feature_filters,
        basic_dim=256,
        conv_dim=1024,
        conv_kernel_size=3,
        num_layers=6,
        dropout=0.1,
        mhsa_config=mhsa_decoder_config,
    )

    model_config = Config(
        speaker_embedding_size=256,
        num_speakers=251,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        duration_predictor_config=duration_predictor_config,
        feature_extraction_config=fe_config,
    )

    params = {
        "config": asdict(model_config)
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
    
    
    duration_hdf = duration_alignments["ctc.tts_aligner_1223.ctc_aligner_tts_fe_v8_tfstyle_v2_fullength"]
    train, forward = run_exp(net_module + "_fromctc_v1_halfbatch", params, net_module, config,
                             extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder", decoder_options=decoder_options,duration_hdf=duration_hdf, debug=True)

    generate_synthetic(prefix, net_module + "_fromctc_v1_halfbatch_syn", "train-clean-100",
                       train.out_checkpoints[200], params, net_module,
                       extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic, debug=True)
    
    config_halflr_fp16 = copy.deepcopy(config)
    config_halflr_fp16["learning_rates"] = list(np.linspace(5e-5, 5e-4, 100)) + list(np.linspace(5e-4, 5e-7, 100))
    config_halflr_fp16["torch_amp_options"] = {"dtype": "bfloat16"}
    train, forward = run_exp(net_module + "_fromctc_v1_halfbatch_fixlr_fp16", params, net_module, config_halflr_fp16,
                             extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder", decoder_options=decoder_options,duration_hdf=duration_hdf, debug=True)
    # train.hold()
    
    generate_synthetic(prefix, net_module + "_fromctc_v1_halfbatch_fixlr_fp16_syn", "train-clean-100",
                       train.out_checkpoints[200], params, net_module,
                       extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic, debug=True)
    
    generate_synthetic(prefix, net_module + "_fromctc_v1_halfbatch_fixlr_fp16_syn", "train-clean-360",
                       train.out_checkpoints[200], params, net_module,
                       extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic, debug=True)

    # Durations from "small GlowTTS"
    duration_hdf = duration_alignments["glow_tts.lukas_baseline_bs600_v2"]
    train, forward = run_exp(net_module + "_fromglow_v1_halfbatch", params, net_module, config,
                             extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder", decoder_options=decoder_options,duration_hdf=duration_hdf, debug=True)

    # config_run2 = copy.deepcopy(config)
    # config_run2["random_seed"] = 43
    # train, forward = run_exp(net_module + "_fromglow_v1_halfbatch_run2", params, net_module, config_run2,
    #                          extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder", decoder_options=decoder_options,duration_hdf=duration_hdf, debug=True)

    # config_fp16 = copy.deepcopy(config)
    # config_fp16["torch_amp_options"] = {"dtype": "bfloat16"}
    # train, forward = run_exp(net_module + "_fromglow_v1_halfbatch_fp16", params, net_module, config_fp16,
    #                          extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder", decoder_options=decoder_options,duration_hdf=duration_hdf, debug=True)


    train, forward = run_exp(net_module + "_fromglow_v1_halfbatch_fixlr_fp16", params, net_module, config_halflr_fp16,
                             extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder", decoder_options=decoder_options,duration_hdf=duration_hdf, debug=True)
    
    # nisqa synthetic
    cross_validation_nisqa(prefix, net_module + "_fromglow_v1_halfbatch_fixlr_fp16_noglnisqa", params, net_module, checkpoint=train.out_checkpoints[200],
                           decoder_options=decoder_options_synthetic, extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder")

    # train.hold()
    generate_synthetic(prefix, net_module + "_fromglow_v1_halfbatch_fixlr_fp16_syn", "train-clean-100",
                       train.out_checkpoints[200], params, net_module,
                       extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic, debug=True)
    
    generate_synthetic(prefix, net_module + "_fromglow_v1_halfbatch_fixlr_fp16_syn", "train-clean-360",
                       train.out_checkpoints[200], params, net_module,
                       extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic, debug=True)


    duration_hdf = duration_alignments["glow_tts.glow_tts_v1_bs600_v2_base256"]
    train, forward = run_exp(net_module + "_fromglowbase256_v1_halfbatch_fixlr_fp16", params, net_module, config_halflr_fp16,
                             extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder", decoder_options=decoder_options,duration_hdf=duration_hdf, debug=True)
    
    generate_synthetic(prefix, net_module + "_fromglowbase256_v1_halfbatch_fixlr_fp16_syn", "train-clean-100",
                       train.out_checkpoints[200], params, net_module,
                       extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic, debug=True)
    
    generate_synthetic(prefix, net_module + "_fromglowbase256_v1_halfbatch_fixlr_fp16_syn_fixspk", "train-clean-100",
                       train.out_checkpoints[200], params, net_module,
                       extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic, debug=True,
                       randomize_speaker=False)

    generate_synthetic(prefix, net_module + "_fromglowbase256_v1_halfbatch_fixlr_fp16_syn", "train-clean-360",
                       train.out_checkpoints[200], params, net_module,
                       extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic, debug=True, use_subset=True)

    generate_synthetic(prefix, net_module + "_fromglowbase256_v1_halfbatch_fixlr_fp16_syn", "train-clean-360",
                       train.out_checkpoints[200], params, net_module,
                       extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic, debug=True)
    
    from recipe.i6_experiments.users.rossenbach.common_setups.lr_scheduling import controlled_noam
    config_longer_noam = copy.deepcopy(config)
    config_longer_noam["optimizer"] = {"class": "adam", "epsilon": 1e-9, "betas": (0.9, 0.98)}
    config_longer_noam["learning_rates"] = controlled_noam(20, 380, 1e-3, 1e-4)
    config_longer_noam["torch_amp_options"] = {"dtype": "bfloat16"}
    train, forward = run_exp(net_module + "_fromglowbase256_v1_bs300_noam_400eps_fp16", params, net_module, config_longer_noam,
                             extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder", decoder_options=decoder_options,duration_hdf=duration_hdf, debug=True, num_epochs=400)

    config_longer_oclr = copy.deepcopy(config)
    config_longer_oclr["learning_rates"] = list(np.linspace(5e-5, 5e-4, 100)) + list(np.linspace(5e-4, 5e-7, 300))
    config_longer_oclr["torch_amp_options"] = {"dtype": "bfloat16"}
    train, forward = run_exp(net_module + "_fromglowbase256_v1_bs300_400eps_fp16", params, net_module, config_longer_oclr,
                             extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder", decoder_options=decoder_options,duration_hdf=duration_hdf, debug=True, num_epochs=400)
