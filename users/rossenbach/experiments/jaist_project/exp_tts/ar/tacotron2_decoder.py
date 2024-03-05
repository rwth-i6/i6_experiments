import copy
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from i6_experiments.common.setups.returnn.datastreams.audio import DBMelFilterbankOptions

from i6_experiments.users.rossenbach.experiments.jaist_project.data.tts_phon import build_durationtts_training_dataset
from i6_experiments.users.rossenbach.experiments.jaist_project.data.tts_phon import get_vocab_datastream
from i6_experiments.users.rossenbach.experiments.jaist_project.data.tts_phon import get_tts_log_mel_datastream

from i6_experiments.users.rossenbach.experiments.jaist_project.config import get_training_config, get_forward_config
from i6_experiments.users.rossenbach.experiments.jaist_project.pipeline import training, tts_eval_v2, generate_synthetic, cross_validation_nisqa, tts_training


from i6_experiments.users.rossenbach.experiments.jaist_project.default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from i6_experiments.users.rossenbach.experiments.jaist_project.storage import duration_alignments, vocoders



def run_tacotron2_decoder_tts():
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

    prefix = "experiments/jaist_project/tts/ar/tacotron2_decoder/"

    log_mel_datastream = get_tts_log_mel_datastream(ls_corpus_key="train-clean-100")

    # verify that normalization exists
    assert "norm_mean" in log_mel_datastream.additional_options
    assert "norm_std_dev" in log_mel_datastream.additional_options

    norm = (log_mel_datastream.additional_options["norm_mean"], log_mel_datastream.additional_options["norm_std_dev"])

    net_module = "ar_tts.tacotron2_decoding.tacotron2_decoding_v1"

    from ...pytorch_networks.ar_tts.tacotron2_decoding.tacotron2_decoding_v1 import (
        DbMelFeatureExtractionConfig,
        TTSTransformerTextEncoderV1Config,
        SimpleConvDurationPredictorV1Config,
        Tacotron2DecoderConfig,
        Config
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

    decoder_config = Tacotron2DecoderConfig(
        dlayers=2,
        dunits=1024,
        prenet_layers=2,
        prenet_units=256,
        postnet_layers=5,
        postnet_chans=512,
        postnet_filts=5,
        use_batch_norm=True,
        use_concate=True,
        dropout_rate=0.5,
        zoneout_rate=0.1,
        reduction_factor=2,
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


    duration_hdf = duration_alignments["glow_tts.lukas_baseline_bs600_v2"]

    #train, forward = run_exp(net_module + "_fromglow_v1", params, net_module, config,
    #                         extra_decoder="ar_tts.tacotron2_decoding.simple_gl_decoder", decoder_options=decoder_options,duration_hdf=duration_hdf, debug=True)
    # train.hold()

    net_module = "ar_tts.tacotron2_decoding.tacotron2_decoding_v2"
    train, forward = tts_training(prefix, net_module + "_fromglow_v1", params, net_module, config,
                             extra_decoder="ar_tts.tacotron2_decoding.simple_gl_decoder", decoder_options=decoder_options,duration_hdf=duration_hdf, debug=True)
    
    generate_synthetic(prefix, net_module + "_fromglow_v1_syn", "train-clean-100",
                       train.out_checkpoints[200], params, net_module,
                       extra_decoder="ar_tts.tacotron2_decoding.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic, debug=True)
    
    duration_hdf = duration_alignments["glow_tts.glow_tts_v1_bs600_v2_base256"]
    train, forward = tts_training(prefix, net_module + "_fromglowbase256_v1", params, net_module, config,
                             extra_decoder="ar_tts.tacotron2_decoding.simple_gl_decoder", decoder_options=decoder_options,duration_hdf=duration_hdf, debug=True)

    # nisqa synthetic
    cross_validation_nisqa(prefix, net_module + "_fromglowbase256_v1_noglnisqa", params, net_module, checkpoint=train.out_checkpoints[200],
                           decoder_options=decoder_options_synthetic, extra_decoder="ar_tts.tacotron2_decoding.simple_gl_decoder")
    
    generate_synthetic(prefix, net_module + "_fromglowbase256_v1_syn", "train-clean-100",
                       train.out_checkpoints[200], params, net_module,
                       extra_decoder="ar_tts.tacotron2_decoding.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic, debug=True)

    # with gl32
    decoder_options_synthetic_gl32 = copy.deepcopy(decoder_options_synthetic)
    decoder_options_synthetic_gl32["gl_momentum"] = 0.99
    decoder_options_synthetic_gl32["gl_iter"] = 32

    generate_synthetic(prefix, net_module + "_fromglowbase256_v1_gl32_syn", "train-clean-100",
                       train.out_checkpoints[200], params, net_module,
                       extra_decoder="ar_tts.tacotron2_decoding.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic_gl32, debug=True)

    generate_synthetic(prefix, net_module + "_fromglowbase256_v1_syn_fixspk", "train-clean-100",
                       train.out_checkpoints[200], params, net_module,
                       extra_decoder="ar_tts.tacotron2_decoding.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic, debug=True, randomize_speaker=False)

    generate_synthetic(prefix, net_module + "_fromglowbase256_v1_syn", "train-clean-360",
                       train.out_checkpoints[200], params, net_module,
                       extra_decoder="ar_tts.tacotron2_decoding.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic, debug=True, use_subset=True)

    generate_synthetic(prefix, net_module + "_fromglowbase256_v1_syn", "train-clean-360",
                       train.out_checkpoints[200], params, net_module,
                       extra_decoder="ar_tts.tacotron2_decoding.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic, debug=True)

    config_400eps = copy.deepcopy(config)
    config_400eps["learning_rates"] = list(np.linspace(5e-5, 5e-4, 100)) + list(np.linspace(5e-4, 5e-7, 300))
    #train, forward = run_exp(net_module + "_fromglowbase256_400eps_v1", params, net_module, config_400eps,
    #                         extra_decoder="ar_tts.tacotron2_decoding.simple_gl_decoder", decoder_options=decoder_options,duration_hdf=duration_hdf, debug=True, num_epochs=400)

    train, forward = tts_training(prefix, net_module + "_fromglowbase256_400eps_v1", params, net_module, config_400eps,
                                  extra_decoder="ar_tts.tacotron2_decoding.simple_gl_decoder", decoder_options=decoder_options,duration_hdf=duration_hdf, debug=True, num_epochs=400, evaluate_swer="ls960eow_phon_ctc_50eps_fastsearch")


    generate_synthetic(prefix, net_module + "_fromglowbase256_400eps_v1_syn", "train-clean-100",
                           train.out_checkpoints[400], params, net_module,
                           extra_decoder="ar_tts.tacotron2_decoding.simple_gl_decoder",
                           decoder_options=decoder_options_synthetic, debug=True)

    # with gl32
    generate_synthetic(prefix, net_module + "_fromglowbase256_400eps_gl32_syn", "train-clean-100",
                       train.out_checkpoints[400], params, net_module,
                       extra_decoder="ar_tts.tacotron2_decoding.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic_gl32, debug=True)

    generate_synthetic(prefix, net_module + "_fromglowbase256_400eps_gl32_syn_fixspk", "train-clean-100",
                       train.out_checkpoints[400], params, net_module,
                       extra_decoder="ar_tts.tacotron2_decoding.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic, debug=True, randomize_speaker=False)

    generate_synthetic(prefix, net_module + "_fromglowbase256_400eps_gl32_syn", "train-clean-360",
                       train.out_checkpoints[400], params, net_module,
                       extra_decoder="ar_tts.tacotron2_decoding.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic, debug=True, use_subset=True)

    generate_synthetic(prefix, net_module + "_fromglowbase256_400eps_gl32_syn", "train-clean-360",
                       train.out_checkpoints[400], params, net_module,
                       extra_decoder="ar_tts.tacotron2_decoding.simple_gl_decoder",
                       decoder_options=decoder_options_synthetic, debug=True)



    # large model
    model_config_large = copy.deepcopy(model_config)
    model_config_large.encoder_config.prenet_config.input_embedding_size = 320
    model_config_large.encoder_config.prenet_config.hidden_dimension = 320
    model_config_large.encoder_config.prenet_config.output_dimension = 320
    model_config_large.encoder_config.basic_dim = 320
    model_config_large.encoder_config.conv_dim = 1280
    model_config_large.encoder_config.mhsa_config.input_dim = 320
    model_config_large.duration_predictor_config.hidden_dim = 512
    model_config_large.decoder_config.dunits = 1280
    model_config_large.decoder_config.prenet_units = 320
    model_config_large.decoder_config.postnet_chans = 640
    params_large = {
        "config": asdict(model_config_large)
    }
    train, forward = tts_training(prefix, net_module + "_base320_fromglowbase256_400eps_v1", params_large, net_module, config_400eps,
                                  extra_decoder="ar_tts.tacotron2_decoding.simple_gl_decoder", decoder_options=decoder_options,duration_hdf=duration_hdf, debug=True, num_epochs=400, evaluate_swer="ls960eow_phon_ctc_50eps_fastsearch")

    
    # No Zoneout model
    decoder_config = Tacotron2DecoderConfig(
        dlayers=2,
        dunits=1024,
        prenet_layers=2,
        prenet_units=256,
        postnet_layers=5,
        postnet_chans=512,
        postnet_filts=5,
        use_batch_norm=True,
        use_concate=True,
        dropout_rate=0.5,
        zoneout_rate=0.1,
        reduction_factor=2,
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


    # large model