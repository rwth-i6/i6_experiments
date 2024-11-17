import copy
import numpy as np
from sisyphus import tk
from dataclasses import asdict


from i6_experiments.common.setups.returnn.datastreams.audio import DBMelFilterbankOptions

from i6_experiments.users.rossenbach.experiments.jaist_project.data.aligner import build_training_dataset
from i6_experiments.users.rossenbach.experiments.jaist_project.config import get_training_config, get_forward_config
from i6_experiments.users.rossenbach.experiments.jaist_project.pipeline import training, extract_durations, tts_eval_v2, generate_synthetic, cross_validation_nisqa
from i6_experiments.users.rossenbach.experiments.jaist_project.data.tts_phon import get_tts_log_mel_datastream, build_durationtts_training_dataset

from i6_experiments.users.rossenbach.experiments.jaist_project.default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from i6_experiments.users.rossenbach.experiments.jaist_project.storage import add_duration, vocoders


def run_flow_tts():
    """
    Baseline for the ctc aligner in returnn_common with serialization

    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    config = {
        "optimizer": {"class": "adam", "epsilon": 1e-9},
        "learning_rates":  list(np.linspace(5e-5, 5e-4, 100)) + list(
                np.linspace(5e-4, 1e-6, 100)),
        # "gradient_clip": 1.0,
        "gradient_clip_norm": 2.0,
        "use_learning_rate_control_always": True,
        #############
        "batch_size": 300 * 16000,
        "max_seq_length": {"audio_features": 30 * 16000},
        "batch_drop_last": True,  # otherwise might cause issues in indexing after local sort in train_step
        "max_seqs": 200,
    }

    prefix = "experiments/jaist_project/tts/glow_tts/"
    training_datasets = build_training_dataset(ls_corpus_key="train-clean-100", partition_epoch=1)

    def run_exp(name, params, net_module, config, decoder_options, extra_decoder=None, use_custom_engine=False, target_durations=None, debug=False, num_epochs=100, evaluate_swer=None, hash_debug=False):
        if target_durations is not None:
            training_datasets_ = build_durationtts_training_dataset(duration_hdf=target_durations, ls_corpus_key="train-clean-100")
        else:
            training_datasets_ = training_datasets
        train_config = get_training_config(
            training_datasets=training_datasets_,
            network_module=net_module,
            net_args=params,
            config=config,
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
        forward_job = tts_eval_v2(
            prefix_name=prefix + name,
            returnn_config=forward_config,
            checkpoint=train_job.out_checkpoints[num_epochs],
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
        )
        tk.register_output(prefix + name + "/audio_files", forward_job.out_files["audio_files"])
        if evaluate_swer is not None:
            from ...storage import asr_recognizer_systems
            from ...pipeline import run_swer_evaluation
            from i6_experiments.users.rossenbach.corpus.transform import MergeCorporaWithPathResolveJob, MergeStrategy
            synthetic_bliss_absolute = MergeCorporaWithPathResolveJob(
                bliss_corpora=[forward_job.out_files["out_corpus.xml.gz"]],
                name="train-clean-100",  # important to keep the original sequence names for matching later
                merge_strategy=MergeStrategy.FLAT
            ).out_merged_corpus
            run_swer_evaluation(
                prefix_name= prefix + name + "/swer/" + evaluate_swer,
                synthetic_bliss=synthetic_bliss_absolute,
                system=asr_recognizer_systems[evaluate_swer]
            )
        return train_job


    def local_extract_durations(name, checkpoint, params, net_module, use_custom_engine=False, debug=False):
        forward_config = get_forward_config(
            network_module=net_module,
            net_args=params,
            decoder="glow_tts.duration_extraction_decoder",
            decoder_args={},
            config={
                "forward": training_datasets.joint.as_returnn_opts()
            },
            debug=debug,
        )
        durations_hdf = extract_durations(
            prefix_name=prefix + name,
            returnn_config=forward_config,
            checkpoint=checkpoint,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
        )
        tk.register_output(prefix + name + "/durations.hdf", durations_hdf)
        add_duration(name, durations_hdf)
        return durations_hdf


    log_mel_datastream = get_tts_log_mel_datastream(ls_corpus_key="train-clean-100", silence_preprocessed=False)

    # verify that normalization exists
    assert "norm_mean" in log_mel_datastream.additional_options
    assert "norm_std_dev" in log_mel_datastream.additional_options

    norm = (log_mel_datastream.additional_options["norm_mean"], log_mel_datastream.additional_options["norm_std_dev"])

    from ...pytorch_networks.tts_shared.encoder.transformer import (
        GlowTTSMultiHeadAttentionV1Config,
        TTSEncoderPreNetV1Config
    )
    from ...pytorch_networks.glow_tts.glow_tts_v1 import (
        DbMelFeatureExtractionConfig,
        TTSTransformerTextEncoderV1Config,
        SimpleConvDurationPredictorV1Config,
        FlowDecoderConfig,
        Config,
    )
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
        input_embedding_size=192,
        hidden_dimension=192,
        kernel_size=5,
        num_layers=3,
        dropout=0.5,
        output_dimension=192,
    )
    mhsa_config = GlowTTSMultiHeadAttentionV1Config(
        input_dim=192,
        num_att_heads=2,
        dropout=0.1,
        att_weights_dropout=0.1,
        window_size=4,  # one-sided, so technically 9
        heads_share=True,
    )
    encoder_config = TTSTransformerTextEncoderV1Config(
        num_layers=6,
        vocab_size=training_datasets.datastreams["phonemes"].vocab_size,
        basic_dim=192,
        conv_dim=768,
        conv_kernel_size=3,
        dropout=0.1,
        mhsa_config=mhsa_config,
        prenet_config=prenet_config,
    )
    duration_predictor_config = SimpleConvDurationPredictorV1Config(
        num_convs=2,
        hidden_dim=256,
        kernel_size=3,
        dropout=0.1,
    )
    decoder_config = FlowDecoderConfig(
        target_channels=fe_config.num_filters,
        hidden_channels=192,
        kernel_size=5,
        dilation_rate=1,
        num_blocks=12,
        num_layers_per_block=4,
        num_splits=4,
        num_squeeze=2,
        dropout=0.05,
        use_sigmoid_scale=False
    )
    model_config = Config(
        feature_extraction_config=fe_config,
        encoder_config=encoder_config,
        duration_predictor_config=duration_predictor_config,
        flow_decoder_config=decoder_config,
        num_speakers=251,
        speaker_embedding_size=256,
        mean_only=True,
    )

    net_module = "glow_tts.glow_tts_v1"

    params = {
        "config": asdict(model_config)
    }

    vocoder = vocoders["blstm_gl_v1"]
    decoder_options_base = {
        "norm_mean": norm[0],
        "norm_std_dev": norm[1],
        "gl_net_checkpoint": vocoder.checkpoint,
        "gl_net_config": vocoder.config,
    }
    
    local_config = copy.deepcopy(config)
    local_config["batch_size"] = 600 * 16000
    local_config["learning_rates"] = list(np.linspace(5e-5, 5e-4, 100)) + list(np.linspace(5e-4, 5e-7, 100))
    decoder_options = copy.deepcopy(decoder_options_base)
    decoder_options["glowtts_noise_scale"] = 0.3
    train = run_exp(net_module + "_bs600_newgl_noise0.3", params, net_module, local_config, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options,
                    debug=True, num_epochs=200)
    # train.hold()

    decoder_options_synthetic = copy.deepcopy(decoder_options)
    decoder_options_synthetic["glowtts_noise_scale"] = 0.7
    decoder_options_synthetic["gl_momentum"] = 0.0
    decoder_options_synthetic["gl_iter"] = 1
    decoder_options_synthetic["create_plots"] = False
    synthetic_corpus = generate_synthetic(prefix, net_module + "_bs600_newgl_noise0.7_syn", "train-clean-100",
                                          train.out_checkpoints[200], params, net_module,
                                          extra_decoder="glow_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_synthetic, debug=True)


    # continued training
    local_config_resume = copy.deepcopy(local_config)
    local_config_resume["learning_rates"] = list(np.linspace(5e-5, 5e-6, 100))
    local_config_resume["import_model_train_epoch1"] = train.out_checkpoints[200]
    local_config_resume["gradient_clip_norm"] = 1.0
    train = run_exp(net_module + "_bs600_v2_newgl_noise0.3_cont100", params, net_module, local_config_resume,
                    extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options,
                    debug=True, num_epochs=100)
    
    synthetic_corpus = generate_synthetic(prefix, net_module + "_bs600_newgl_noise0.7_cont100_syn", "train-clean-100",
                                          train.out_checkpoints[100], params, net_module,
                                          extra_decoder="glow_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_synthetic, debug=True)
    
    # bigger
    
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
        vocab_size=training_datasets.datastreams["phonemes"].vocab_size,
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
    decoder_config = FlowDecoderConfig(
        target_channels=fe_config.num_filters,
        hidden_channels=256,
        kernel_size=5,
        dilation_rate=1,
        num_blocks=12,
        num_layers_per_block=4,
        num_splits=4,
        num_squeeze=2,
        dropout=0.05,
        use_sigmoid_scale=False
    )
    model_config_base256 = Config(
        feature_extraction_config=fe_config,
        encoder_config=encoder_config,
        duration_predictor_config=duration_predictor_config,
        flow_decoder_config=decoder_config,
        num_speakers=251,
        speaker_embedding_size=256,
        mean_only=True,
    )

    params_base256 = {
        "config": asdict(model_config_base256)
    }

    local_config = copy.deepcopy(config)
    local_config["batch_size"] = 600 * 16000
    local_config["learning_rates"] = list(np.linspace(5e-5, 5e-4, 100)) + list(np.linspace(5e-4, 5e-7, 100))

    decoder_options = copy.deepcopy(decoder_options_base)
    decoder_options["glowtts_noise_scale"] = 0.7
    train = run_exp(net_module + "_bs600_v2_base256_newgl_noise0.7", params_base256, net_module, local_config, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options,
                    debug=True, num_epochs=200, hash_debug=True)
    # train.hold()
    durations = local_extract_durations(net_module + "_bs600_v2_base256", train.out_checkpoints[200], params_base256, net_module, debug=True)

    for noise_scale in [0.3, 0.7]:
        decoder_options_local = copy.deepcopy(decoder_options_synthetic)
        decoder_options_local["glowtts_noise_scale"] = 0.7

        synthetic_corpus = generate_synthetic(prefix, net_module + "_bs600_v2_base256_newgl_noise%.1f_syn" % noise_scale, "train-clean-100",
                                              train.out_checkpoints[200], params_base256, net_module,
                                              extra_decoder="glow_tts.simple_gl_decoder",
                                              decoder_options=decoder_options_local, debug=True)
    for noise_scale in [0.7]:
        decoder_options_local = copy.deepcopy(decoder_options_synthetic)
        decoder_options_local["glowtts_noise_scale"] = 0.7

        synthetic_corpus = generate_synthetic(prefix, net_module + "_bs600_v2_base256_newgl_noise%.1f_syn" % noise_scale, "train-clean-360",
                                              train.out_checkpoints[200], params_base256, net_module,
                                              extra_decoder="glow_tts.simple_gl_decoder",
                                              decoder_options=decoder_options_local, debug=True)


    net_module_extdur = "glow_tts.glow_tts_v1_ext_dur"
    train = run_exp(net_module + "_bs600_v2_base256_newgl_extdur_noise0.7", params_base256, net_module_extdur, local_config, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options,
                    target_durations=durations, debug=True, num_epochs=200, evaluate_swer="ls960eow_phon_ctc_50eps_fastsearch")

    # perform NISQA also for synthesis condition
    cross_validation_nisqa(prefix, net_module + "_bs600_v2_base256_newgl_extdur_noise0.7_noglnisqa", params_base256, net_module_extdur, checkpoint=train.out_checkpoints[200],
                           decoder_options=decoder_options_synthetic, extra_decoder="glow_tts.simple_gl_decoder")

    for noise in [0.0, 0.3, 0.5, 0.7, 1.0]:
        decoder_options_synthetic_noise = copy.deepcopy(decoder_options_synthetic)
        decoder_options_synthetic_noise["glowtts_noise_scale"] = noise
        synthetic_corpus = generate_synthetic(prefix, net_module + "_bs600_v2_base256_newgl_extdur_noise%.1f_syn" % noise, "train-clean-100",
                                              train.out_checkpoints[200], params_base256, net_module,
                                              extra_decoder="glow_tts.simple_gl_decoder",
                                              decoder_options=decoder_options_synthetic_noise, debug=True)

        # perform NISQA also for synthesis condition
        decoder_options_synthetic_noise_gl32 = copy.deepcopy(decoder_options_synthetic_noise)
        decoder_options_synthetic_noise_gl32["gl_momentum"] = 0.99
        decoder_options_synthetic_noise_gl32["gl_iter"] = 32

        cross_validation_nisqa(prefix, net_module + "_bs600_v2_base256_newgl_extdur_noise%.1f_gl32nisqa" % noise, params_base256,
                               net_module_extdur, checkpoint=train.out_checkpoints[200],
                               decoder_options=decoder_options_synthetic_noise_gl32, extra_decoder="glow_tts.simple_gl_decoder")

        if noise == 0.7:
            # fixed speaker
            synthetic_corpus = generate_synthetic(prefix, net_module + "_bs600_v2_base256_newgl_extdur_noise%.1f_syn_fixspk" % noise,
                                                  "train-clean-100",
                                                  train.out_checkpoints[200], params_base256, net_module,
                                                  extra_decoder="glow_tts.simple_gl_decoder",
                                                  decoder_options=decoder_options_synthetic_noise, debug=True,
                                                  randomize_speaker=False)
            # text from 360
            synthetic_corpus = generate_synthetic(prefix, net_module + "_bs600_v2_base256_newgl_extdur_noise%.1f_syn" % noise,
                                                  "train-clean-360",
                                                  train.out_checkpoints[200], params_base256, net_module,
                                                  extra_decoder="glow_tts.simple_gl_decoder",
                                                  decoder_options=decoder_options_synthetic_noise, debug=True,
                                                  use_subset=True)

    synthetic_corpus = generate_synthetic(prefix, net_module + "_bs600_v2_base256_newgl_extdur_noise0.7_syn", "train-clean-360",
                                          train.out_checkpoints[200], params_base256, net_module,
                                          extra_decoder="glow_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_synthetic, debug=True)


    # FP16 yields the same training scores, but now we did most of the stuff without it so keep it that way. Also speedup was for some reason not that much...
    local_config_fp16 = copy.deepcopy(local_config)
    local_config_fp16["torch_amp_options"] = {"dtype": "bfloat16"}
    train = run_exp(net_module + "_bs600_v2_base256_fp16_newgl_noise0.7", params_base256, net_module, local_config_fp16, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options,
                    debug=True, num_epochs=200)
    # train.hold()
    
    # 400 epochs stuff
    # This is a final model for the paper

    local_config_longer = copy.deepcopy(config)
    local_config_longer["batch_size"] = 600 * 16000
    local_config_longer["learning_rates"] = list(np.linspace(5e-5, 5e-4, 100)) + list(np.linspace(5e-4, 5e-7, 300))
    train = run_exp(net_module + "_bs600_v2_longer_base256_newgl_extdur_noise0.7", params_base256, net_module_extdur, local_config_longer, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options,
                    target_durations=durations, debug=True, num_epochs=400, evaluate_swer="ls960eow_phon_ctc_50eps_fastsearch")
    
    # perform NISQA also for synthesis condition
    #cross_validation_nisqa(prefix, net_module + "_bs600_v2_longer_base256_newgl_extdur_noise0.7_noglnisqa", params_base256, net_module_extdur, checkpoint=train.out_checkpoints[400],
    #                       decoder_options=decoder_options_synthetic, extra_decoder="glow_tts.simple_gl_decoder")

    for momentum in [0.0, 0.9, 0.99]:
        for iter in [1, 8, 16, 32, 64]:
            decoder_options_half_gl = copy.deepcopy(decoder_options_synthetic)
            decoder_options_half_gl["gl_momentum"] = momentum
            decoder_options_half_gl["gl_iter"] = iter
            cross_validation_nisqa(prefix, net_module + "_bs600_v2_longer_base256_newgl_extdur_noise0.7_nisqa_ablations/mom%.2f_gl%i" % (momentum, iter), params_base256, net_module_extdur, checkpoint=train.out_checkpoints[400],
                                   decoder_options=decoder_options_half_gl, extra_decoder="glow_tts.simple_gl_decoder")


    synthetic_corpus = generate_synthetic(prefix, net_module + "_bs600_v2_longer_base256_newgl_extdur_noise0.7_syn",
                                          "train-clean-100",
                                          train.out_checkpoints[400], params_base256, net_module,
                                          extra_decoder="glow_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_synthetic, debug=True)


    for iter in [16, 32, 64]:
        decoder_options_half_gl = copy.deepcopy(decoder_options_synthetic)
        decoder_options_half_gl["gl_momentum"] = 0.99
        decoder_options_half_gl["gl_iter"] = iter

        synthetic_corpus = generate_synthetic(prefix, net_module + "_glow256align_400eps_oclr_gl%i_noise0.7_syn" % iter,
                                              "train-clean-100",
                                              train.out_checkpoints[400], params_base256, net_module,
                                              extra_decoder="glow_tts.simple_gl_decoder",
                                              decoder_options=decoder_options_half_gl, debug=True)

    decoder_options_gl32 = copy.deepcopy(decoder_options_synthetic)
    decoder_options_gl32["gl_momentum"] = 0.99
    decoder_options_gl32["gl_iter"] = 32

    synthetic_corpus = generate_synthetic(prefix,
                                          net_module + "_glow256align_400eps_oclr_gl32_noise0.7_syn_fixspk",
                                          "train-clean-100",
                                          train.out_checkpoints[400], params_base256, net_module,
                                          extra_decoder="glow_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_gl32, debug=True, randomize_speaker=False)

    synthetic_corpus = generate_synthetic(prefix,
                                          net_module + "_glow256align_400eps_oclr_gl32_noise0.7_syn",
                                          "train-clean-360",
                                          train.out_checkpoints[400], params_base256, net_module,
                                          extra_decoder="glow_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_gl32, debug=True, use_subset=True)

    synthetic_corpus = generate_synthetic(prefix,
                                          net_module + "_glow256align_400eps_oclr_gl32_noise0.7_syn",
                                          "train-clean-360",
                                          train.out_checkpoints[400], params_base256, net_module,
                                          extra_decoder="glow_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_gl32, debug=True)

    # run this by accident, so keep it
    decoder_options_half_gl64 = copy.deepcopy(decoder_options_synthetic)
    decoder_options_half_gl64["gl_momentum"] = 0.99
    decoder_options_half_gl64["gl_iter"] = 64

    synthetic_corpus = generate_synthetic(prefix,
                                          net_module + "_bs600_v2_longer_base256_newgl64_extdur_noise0.7_syn_fixspk",
                                          "train-clean-100",
                                          train.out_checkpoints[400], params_base256, net_module,
                                          extra_decoder="glow_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_half_gl64, debug=True, randomize_speaker=False)

    synthetic_corpus = generate_synthetic(prefix,
                                          net_module + "_bs600_v2_longer_base256_newgl64_extdur_noise0.7_syn",
                                          "train-clean-360",
                                          train.out_checkpoints[400], params_base256, net_module,
                                          extra_decoder="glow_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_half_gl64, debug=True, use_subset=True)

    synthetic_corpus = generate_synthetic(prefix,
                                          net_module + "_bs600_v2_longer_base256_newgl64_extdur_noise0.7_syn",
                                          "train-clean-360",
                                          train.out_checkpoints[400], params_base256, net_module,
                                          extra_decoder="glow_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_half_gl64, debug=True)


    # also with std prediction once (is worse in MOS and MLE)
    model_config_base256_withsigma = copy.deepcopy(model_config_base256)
    model_config_base256_withsigma.mean_only = False
    params_base256_withsigma = {
        "config": asdict(model_config_base256_withsigma)
    }
    train = run_exp(net_module + "_bs600_v2_withsigma_longer_base256_newgl_extdur_noise0.7", params_base256_withsigma, net_module_extdur, local_config_longer, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options,
                    target_durations=durations, debug=True, num_epochs=400)
    
    
    # Spectrogram plot job
    decoder_options_gl32_plot = copy.deepcopy(decoder_options_gl32)
    decoder_options_gl32_plot["create_plots"] = True
    decoder_options_gl32_plot["store_log_mels"] = True
    cross_validation_nisqa(prefix,
                           net_module + "_bs600_v2_longer_base256_newgl_extdur_noise0.7/plot_export",
                           params_base256, net_module_extdur, checkpoint=train.out_checkpoints[400],
                           decoder_options=decoder_options_gl32_plot, extra_decoder="glow_tts.simple_gl_decoder")

    """
    Kind of trying to replicate what was exactly in the original paper, just based on epochs and with longer warmup
    
    Those two experiments show:
     - bs300 vs bs600 makes no difference in terms of NISQA MOS
     - Noam scheduling is inferior to OCLR in both NISQA and synthetic training
    """
    from recipe.i6_experiments.users.rossenbach.common_setups.lr_scheduling import controlled_noam
    local_config_longer_noam = copy.deepcopy(config)
    local_config_longer_noam["optimizer"] = {"class": "adam", "epsilon": 1e-9, "betas": (0.9, 0.98)}
    local_config_longer_noam["learning_rates"] = controlled_noam(20, 380, 1e-3, 1e-4)
    train = run_exp(net_module + "_bs300_v2_longer_noam_base256_newgl_extdur_noise0.7", params_base256, net_module_extdur, local_config_longer_noam, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options,
                    target_durations=durations, debug=True, num_epochs=400)

    local_config_longer_noam_bs600 = copy.deepcopy(local_config_longer_noam)
    local_config_longer_noam_bs600["batch_size"] = 600 * 16000
    train = run_exp(net_module + "_bs600_v2_longer_noam_base256_newgl_extdur_noise0.7", params_base256, net_module_extdur, local_config_longer_noam_bs600, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options,
                    target_durations=durations, debug=True, num_epochs=400)
    
    synthetic_corpus = generate_synthetic(prefix, net_module + "_bs600_v2_longer_noam_base256_newgl_extdur_noise0.7_syn",
                                          "train-clean-100",
                                          train.out_checkpoints[400], params_base256, net_module,
                                          extra_decoder="glow_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_synthetic, debug=True)


    # Checking if gradient clipping makes a problem here (result, it does not)

    local_config_longer_noam_bs600_gradclip5 = copy.deepcopy(local_config_longer_noam_bs600)
    local_config_longer_noam_bs600_gradclip5.pop("gradient_clip_norm")
    local_config_longer_noam_bs600_gradclip5["gradient_clip"] = 5

    train = run_exp(net_module + "_bs600_v2_longer_noam_base256_newgl_extdur_clip5_noise0.7", params_base256, net_module_extdur, local_config_longer_noam_bs600_gradclip5, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options,
                    target_durations=durations, debug=True, num_epochs=400)
    train.hold()

    # also do the same for the "normal" one
    local_config_longer_gradclip5 = copy.deepcopy(local_config_longer)
    local_config_longer_gradclip5.pop("gradient_clip_norm")
    local_config_longer_gradclip5["gradient_clip"] = 5
    train = run_exp(net_module + "_bs600_v2_longerbase256_newgl_extdur_clip5_noise0.7", params_base256, net_module_extdur, local_config_longer_gradclip5, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options,
                    target_durations=durations, debug=True, num_epochs=400)
    train.hold()

    # check for dropout
    model_config_base256_nodrop = copy.deepcopy(model_config_base256)
    model_config_base256_nodrop.flow_decoder_config.dropout = 0.0
    params_base256_nodrop = {
        "config": asdict(model_config_base256_nodrop)
    }
    train = run_exp(net_module + "_glow256align_400eps_oclr_nodrop_gl32_noise0.7", params_base256_nodrop, net_module_extdur, local_config_longer, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options,
                    target_durations=durations, debug=True, num_epochs=400, evaluate_swer="ls960eow_phon_ctc_50eps_fastsearch")
    # train.hold()
    
    synthetic_corpus = generate_synthetic(prefix,
                                          net_module + "_glow256align_400eps_oclr_nodrop_gl32_noise0.7_syn",
                                          "train-clean-100",
                                          train.out_checkpoints[400], params_base256, net_module,
                                          extra_decoder="glow_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_gl32, debug=True)

    synthetic_corpus = generate_synthetic(prefix,
                                          net_module + "_glow256align_400eps_oclr_nodrop_gl32_noise0.7_syn",
                                          "train-clean-360",
                                          train.out_checkpoints[400], params_base256, net_module,
                                          extra_decoder="glow_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_gl32, debug=True, use_subset=True)

    # larger
    model_config_base320 = copy.deepcopy(model_config_base256)
    model_config_base320.encoder_config.basic_dim = 320
    model_config_base320.encoder_config.conv_dim = 1280
    model_config_base320.encoder_config.mhsa_config.input_dim = 320
    model_config_base320.encoder_config.prenet_config.input_embedding_size = 320
    model_config_base320.encoder_config.prenet_config.hidden_dimension = 320
    model_config_base320.encoder_config.prenet_config.output_dimension = 320
    model_config_base320.flow_decoder_config.hidden_channels = 320
    model_config_base320.duration_predictor_config.hidden_dim = 512

    params_base320 = {
        "config": asdict(model_config_base320)
    }
    train = run_exp(net_module + "_base320_glow256align_400eps_oclr_nodrop_gl32_noise0.7", params_base320, net_module_extdur, local_config_longer, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options,
                    target_durations=durations, debug=True, num_epochs=400, evaluate_swer="ls960eow_phon_ctc_50eps_fastsearch")
    # train.hold()
    
    
    # even longer
    local_config_longer = copy.deepcopy(config)
    local_config_longer["batch_size"] = 600 * 16000
    local_config_longer["learning_rates"] = list(np.linspace(5e-5, 5e-4, 200)) + list(np.linspace(5e-4, 5e-7, 600))
    train = run_exp(net_module + "_bs600_v2_800eps_base256_newgl_extdur_noise0.7", params_base256, net_module_extdur, local_config_longer, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options,
                    target_durations=durations, debug=True, num_epochs=800, evaluate_swer="ls960eow_phon_ctc_50eps_fastsearch")
    
    synthetic_corpus = generate_synthetic(prefix, net_module + "_bs600_v2_800eps_base256_newgl_extdur_noise0.7_syn",
                                          "train-clean-100",
                                          train.out_checkpoints[800], params_base256, net_module,
                                          extra_decoder="glow_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_gl32, debug=True)

    synthetic_corpus = generate_synthetic(prefix, net_module + "_bs600_v2_800eps_base256_newgl_extdur_noise0.7_syn",
                                          "train-clean-360",
                                          train.out_checkpoints[800], params_base256, net_module,
                                          extra_decoder="glow_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_gl32, debug=True, use_subset=True)