import copy
import os
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from i6_core.corpus.transform import MergeCorporaJob, MergeStrategy
from i6_core.corpus.convert import CorpusReplaceOrthFromReferenceCorpus
from i6_core.returnn.oggzip import BlissToOggZipJob

from i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict
from i6_experiments.common.setups.returnn.datastreams.audio import DBMelFilterbankOptions
from i6_experiments.users.rossenbach.corpus.transform import MergeCorporaWithPathResolveJob

from ..data.aligner import build_training_dataset
from ..config import get_training_config, get_prior_config, get_forward_config
from ..pipeline import training, extract_durations, tts_eval, tts_generation
from ..data.tts_phon import get_tts_log_mel_datastream, build_fixed_speakers_generating_dataset, get_tts_extended_bliss, build_durationtts_training_dataset



from ..default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ..storage import add_duration, vocoders, add_synthetic_data, duration_alignments



def run_diffusion_tts():
    """
    Uses updated RETURNN_COMMON

    :return: durations_hdf
    """

    config = {
        "optimizer": {"class": "adam", "epsilon": 1e-9},
        "learning_rates": [1e-4] * 200,
        # "gradient_clip": 1.0,
        "gradient_clip_norm": 2.0,
        "use_learning_rate_control_always": True,
        #############
        "batch_size": 300 * 16000,
        "max_seq_length": {"audio_features": 30 * 16000},
        "batch_drop_last": True,  # otherwise might cause issues in indexing after local sort in train_step
        "max_seqs": 200,
    }

    prefix = "experiments/jaist_project/standalone_2024/grad_tts/"
    training_datasets = build_training_dataset(ls_corpus_key="train-clean-100", partition_epoch=1)

    def run_exp(name, params, net_module, config, decoder_options, extra_decoder=None, use_custom_engine=False, target_durations=None, debug=False, num_epochs=100):
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
            trigger=False,
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
        forward_job = tts_eval(
            prefix_name=prefix + name,
            returnn_config=forward_config,
            checkpoint=train_job.out_checkpoints[num_epochs],
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
        )
        tk.register_output(prefix + name + "/audio_files", forward_job.out_files["audio_files"])
        return train_job


    def generate_synthetic(name, target_ls_corpus, checkpoint, params, net_module, decoder_options, extra_decoder=None, use_custom_engine=False, debug=False, splits=10):
        # we want to get ls360 but with the vocab settings from ls100
        asr_bliss = get_bliss_corpus_dict()[target_ls_corpus]
        tts_bliss = get_tts_extended_bliss(ls_corpus_key=target_ls_corpus, lexicon_ls_corpus_key=target_ls_corpus)
        generating_datasets = build_fixed_speakers_generating_dataset(
            text_bliss=tts_bliss,
            num_splits=splits,
            ls_corpus_key="train-clean-100",  # this is always ls100
        )
        split_out_bliss = []
        for i in range(splits):
            forward_config = get_forward_config(
                network_module=net_module,
                net_args=params,
                decoder=extra_decoder or net_module,
                decoder_args=decoder_options,
                config={
                    "forward": generating_datasets.split_datasets[i].as_returnn_opts()
                },
                debug=debug,
            )
            forward_config.config["max_seqs"] = 30
            forward_job = tts_generation(
                prefix_name=prefix + name + f"/{target_ls_corpus}_split{i}",
                returnn_config=forward_config,
                checkpoint=checkpoint,
                returnn_exe=RETURNN_EXE,
                returnn_root=MINI_RETURNN_ROOT,
            )
            split_out_bliss.append(forward_job.out_files["out_corpus.xml.gz"])

        merged_corpus = MergeCorporaWithPathResolveJob(
            bliss_corpora=split_out_bliss, name=target_ls_corpus, merge_strategy=MergeStrategy.FLAT
        ).out_merged_corpus
        merged_corpus_with_text = CorpusReplaceOrthFromReferenceCorpus(
            bliss_corpus=merged_corpus,
            reference_bliss_corpus=asr_bliss,
        ).out_corpus
        ogg_zip_job = BlissToOggZipJob(
            merged_corpus_with_text,
            no_conversion=True,
            returnn_python_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT
        )
        ogg_zip_job.add_alias(prefix + name + f"/{target_ls_corpus}/create_synthetic_zip")
        add_synthetic_data(name + "_" + target_ls_corpus, ogg_zip_job.out_ogg_zip, merged_corpus_with_text)
        return merged_corpus_with_text
    
    
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

    from ..pytorch_networks.tts_shared.encoder.transformer import (
        GlowTTSMultiHeadAttentionV1Config,
        TTSEncoderPreNetV1Config
    )
    from ..pytorch_networks.grad_tts.grad_tts_v1 import (
        DbMelFeatureExtractionConfig,
        TTSTransformerTextEncoderV1Config,
        SimpleConvDurationPredictorV1Config,
        DiffusionDecoderConfig,
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
    decoder_config = DiffusionDecoderConfig(
        n_feats=log_mel_datastream.options.num_feature_filters,
        dim=64,
        beta_min=0.05,
        beta_max=20.0,
        pe_scale=1000,
    )
    model_config = Config(
        feature_extraction_config=fe_config,
        encoder_config=encoder_config,
        duration_predictor_config=duration_predictor_config,
        diffusion_decoder_config=decoder_config,
        num_speakers=251,
        speaker_embedding_size=256,
        decoder_segment_num_frames=160,  # 2 seconds, just like in reference implementation
    )

    net_module = "grad_tts.grad_tts_v1"

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
    decoder_options = copy.deepcopy(decoder_options_base)
    decoder_options["gradtts_num_steps"] = 10
    train = run_exp(net_module + "_bs600_newgl", params, net_module, local_config, extra_decoder="grad_tts.simple_gl_decoder", decoder_options=decoder_options,
                    debug=True, num_epochs=200)
    train.hold()


    # net_module = "grad_tts.grad_tts_v1_ext_dur"
    # duration_hdf = duration_alignments["glow_tts.lukas_baseline_bs600_v2"]
    # print(duration_hdf)
    # train = run_exp(net_module + "_bs600_newgl_extdurtest", params, net_module, local_config, extra_decoder="grad_tts.simple_gl_decoder", decoder_options=decoder_options,
    #                 target_durations=duration_hdf, debug=True, num_epochs=200)
    # train.hold()


    
    
    from ..pytorch_networks.grad_tts.grad_tts_v2 import (
        MuNetConfig,
        Config,
    )
    
    
    munet_mhsa_config = GlowTTSMultiHeadAttentionV1Config(
        input_dim=256,
        num_att_heads=4,
        dropout=0.1,
        att_weights_dropout=0.1,
        window_size=4,  # one-sided, so technically 9
        heads_share=True,
    )

    mu_net_config = MuNetConfig(
        num_layers=2,
        input_dim=192,
        basic_dim=256,
        conv_dim=1024,
        conv_kernel_size=3,
        mhsa_config=munet_mhsa_config,
        dropout=0.1
    )

    model_config = Config(
        feature_extraction_config=fe_config,
        encoder_config=encoder_config,
        duration_predictor_config=duration_predictor_config,
        mu_net_config=mu_net_config,
        diffusion_decoder_config=decoder_config,
        num_speakers=251,
        speaker_embedding_size=256,
        decoder_segment_num_frames=160,  # 2 seconds, just like in reference implementation
    )

    net_module = "grad_tts.grad_tts_v2"

    params = {
        "config": asdict(model_config)
    }

    vocoder = vocoders["blstm_gl_v1"]
    decoder_options_base = {
        "norm_mean": norm[0],
        "norm_std_dev": norm[1],
        "gl_net_checkpoint": vocoder.checkpoint,
        "gl_net_config": vocoder.config,
        "gradtts_num_steps": 4,
        "gradtts_noise_scale": 0.5,

    }

    local_config = copy.deepcopy(config)
    decoder_options = copy.deepcopy(decoder_options_base)
    decoder_options["gradtts_num_steps"] = 10
    train = run_exp(net_module + "_bs600_newgl_mu2", params, net_module, local_config,
                    extra_decoder="grad_tts.simple_gl_decoder", decoder_options=decoder_options,
                    debug=True, num_epochs=200)
    train.hold()
    
    net_module = "grad_tts.grad_tts_v2_ext_dur"
    duration_hdf = duration_alignments["glow_tts.lukas_baseline_bs600_v2"]
    train = run_exp(net_module + "_bs300_newgl_extdurtest", params, net_module, local_config, extra_decoder="grad_tts.simple_gl_decoder", decoder_options=decoder_options,
                    target_durations=duration_hdf, debug=True, num_epochs=200)
    # train.hold()

    decoder_options_syn = copy.deepcopy(decoder_options_base)
    synthetic_corpus = generate_synthetic(net_module + "_bs300_newgl_extdurtest_syn", "train-clean-100",
                                          train.out_checkpoints[200], params, net_module,
                                          extra_decoder="grad_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_syn, debug=True)
    
    synthetic_corpus = generate_synthetic(net_module + "_bs300_newgl_extdurtest_syn", "train-clean-360",
                                          train.out_checkpoints[200], params, net_module,
                                          extra_decoder="grad_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_syn, debug=True)
    
    
    # correct size
    
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

    munet_mhsa_config = GlowTTSMultiHeadAttentionV1Config(
        input_dim=256,
        num_att_heads=2,
        dropout=0.1,
        att_weights_dropout=0.1,
        window_size=4,  # one-sided, so technically 9
        heads_share=True,
    )

    mu_net_config = MuNetConfig(
        num_layers=2,
        input_dim=256,
        basic_dim=256,
        conv_dim=1024,
        conv_kernel_size=3,
        mhsa_config=munet_mhsa_config,
        dropout=0.1
    )
    
    decoder_config = DiffusionDecoderConfig(
        n_feats=log_mel_datastream.options.num_feature_filters,
        dim=64,
        beta_min=0.05,
        beta_max=20.0,
        pe_scale=1000,
    )

    model_config = Config(
        feature_extraction_config=fe_config,
        encoder_config=encoder_config,
        duration_predictor_config=duration_predictor_config,
        mu_net_config=mu_net_config,
        diffusion_decoder_config=decoder_config,
        num_speakers=251,
        speaker_embedding_size=256,
        decoder_segment_num_frames=160,  # 2 seconds, just like in reference implementation
    )

    params_base256 = {
        "config": asdict(model_config)
    }

    net_module = "grad_tts.grad_tts_v2_ext_dur"
    duration_hdf = duration_alignments["glow_tts.glow_tts_v1_bs600_v2_base256"]
    train = run_exp(net_module + "_bs300_newgl_extdurglowbase256", params_base256, net_module, local_config,
                    extra_decoder="grad_tts.simple_gl_decoder", decoder_options=decoder_options,
                    target_durations=duration_hdf, debug=True, num_epochs=200)
    # train.hold()

    decoder_options_syn = copy.deepcopy(decoder_options_base)
    synthetic_corpus = generate_synthetic(net_module + "_bs300_newgl_extdurglowbase256_syn", "train-clean-100",
                                          train.out_checkpoints[200], params_base256, net_module,
                                          extra_decoder="grad_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_syn, debug=True)

    synthetic_corpus = generate_synthetic(net_module + "_bs300_newgl__extdurglowbase256_syn", "train-clean-360",
                                          train.out_checkpoints[200], params_base256, net_module,
                                          extra_decoder="grad_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_syn, debug=True)