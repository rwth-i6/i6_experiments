import copy
import numpy as np
from sisyphus import tk
from dataclasses import asdict

from i6_core.returnn.oggzip import BlissToOggZipJob
from i6_core.corpus.convert import CorpusReplaceOrthFromReferenceCorpus

from i6_experiments.common.setups.returnn.datastreams.audio import DBMelFilterbankOptions
from i6_experiments.users.rossenbach.tts.speaker_embedding import ResemblyzerEmbeddingHDFFromBliss

from ....config import get_forward_config
from ....pipeline import training, prepare_tts_model, TTSModel, tts_eval_v2, extract_durations
from ....data.tts.tts_phon import get_tts_log_mel_datastream, build_durationtts_training_dataset
from ....data.tts.tts_phon import build_dynamic_speakers_generating_dataset
from ....data.common import get_bliss_corpus_dict
from ....data.tts.tts_phon import get_tts_extended_bliss, get_tts_bliss_and_zip, GeneratingDataset

from ....default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ....storage import vocoders, add_duration, duration_alignments


def run_grad_tts():
    """
    """

    prefix = "experiments/loquacious/standalone_2025/tts/grad_tts/"

    log_mel_datastream = get_tts_log_mel_datastream(loq_corpus_key="train.small", silence_preprocessed=False)

    # verify that normalization exists
    assert "norm_mean" in log_mel_datastream.additional_options
    assert "norm_std_dev" in log_mel_datastream.additional_options

    norm = (log_mel_datastream.additional_options["norm_mean"], log_mel_datastream.additional_options["norm_std_dev"])

    loq_bliss = get_bliss_corpus_dict()["train.small"]
    speaker_embedding_hdf = ResemblyzerEmbeddingHDFFromBliss(loq_bliss).out_speaker_hdf
    training_datasets = build_durationtts_training_dataset(
        duration_hdf=duration_alignments["base"],
        loq_corpus_key="train.small",
        partition_epoch=1,
        dynamic_speaker_embeddings=speaker_embedding_hdf,
        dynamic_speaker_embedding_size=256,
    )
    
    def run_exp(name, train_args, num_epochs=100):
        train_job = training(
            training_name=prefix + name,
            datasets=training_datasets,
            train_args=train_args,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            num_epochs=num_epochs,
        )
        return train_job

    def synthesize_dataset(
            name,
            tts_model: TTSModel,
            decoder,
            decoder_options,
            corpus_name: str,
            dataset: GeneratingDataset,
            cpu_rqmt=10,
            use_gpu=True,
            local_prefix=None,
            seed=None,
            time_rqmt=24,
    ):
        if local_prefix is None:
            local_prefix = prefix
        forward_config = get_forward_config(
            network_module=tts_model.network_module,
            net_args=tts_model.net_args,
            decoder=decoder,
            decoder_args=decoder_options,
            config={
                "forward": dataset.split_datasets[0].as_returnn_opts()
            },
            debug=False,
        )
        # this is now characters!
        forward_config.config["batch_size"] = 10000
        forward_config.config["max_seqs"] = 8
        forward_config.config["torch_amp_options"] = {"dtype": "bfloat16"}
        if seed is not None:
            forward_config.config["random_seed"] = seed
        forward_job = tts_eval_v2(
            prefix_name=local_prefix + "/" + tts_model.prefix_name + "/" + name,
            returnn_config=forward_config,
            checkpoint=tts_model.checkpoint,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            mem_rqmt=12,
            cpu_rqmt=cpu_rqmt,
            use_gpu=use_gpu,
        )
        forward_job.rqmt["gpu_mem"] = 48
        forward_job.rqmt["time"] = time_rqmt
        forward_job.add_alias(local_prefix + "/" + tts_model.prefix_name + "/" + name + "/forward")
        tk.register_output(local_prefix + "/" + tts_model.prefix_name + "/" + name + "/audio_files",
                           forward_job.out_files["audio_files"])
        corpus = forward_job.out_files["out_corpus.xml.gz"]
        from i6_experiments.users.rossenbach.corpus.transform import MergeCorporaWithPathResolveJob, MergeStrategy
        realpath_corpus = MergeCorporaWithPathResolveJob(bliss_corpora=[corpus],
                                                         name=corpus_name,
                                                         # important to keep the original sequence names for matching later
                                                         merge_strategy=MergeStrategy.FLAT
                                                         )
        return realpath_corpus.out_merged_corpus


    from ....pytorch_networks.tts_shared.encoder.transformer import (
        GlowTTSMultiHeadAttentionV1Config,
        TTSEncoderPreNetV1Config
    )
    from ....pytorch_networks.grad_tts.grad_tts_v3_ext_dur import (
        DbMelFeatureExtractionConfig,
        TTSTransformerTextEncoderV2Config,
        SimpleConvDurationPredictorV1Config,
        DiffusionDecoderConfig,
        MuNetConfig,
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
    encoder_config = TTSTransformerTextEncoderV2Config(
        num_layers=6,
        vocab_size=training_datasets.datastreams["phonemes"].vocab_size,
        basic_dim=256,
        conv_dim=1024,
        conv_kernel_size=3,
        dropout=0.1,
        mhsa_config=mhsa_config,
        prenet_config=prenet_config,
        combine_speaker_embedding_at_layer=3,
    )
    duration_predictor_config = SimpleConvDurationPredictorV1Config(
        num_convs=2,
        hidden_dim=384,
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
        input_dim=256,
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

    net_module = "grad_tts.grad_tts_v3_ext_dur"

    vocoder = vocoders["blstm_gl_v1"]
    decoder_options_base = {
        "norm_mean": norm[0],
        "norm_std_dev": norm[1],
        "gl_net_checkpoint": vocoder.checkpoint,
        "gl_net_config": vocoder.config,
        "gradtts_num_steps": 10,
        "gradtts_noise_scale": 0.5,
    }

    decoder_options = copy.deepcopy(decoder_options_base)

    decoder_options_synthetic = copy.deepcopy(decoder_options)
    decoder_options_synthetic["gradtts_noise_scale"] = 0.7
    decoder_options_synthetic["gl_momentum"] = 0.99
    decoder_options_synthetic["gl_iter"] = 32
    decoder_options_synthetic["create_plots"] = False
    
    config = {
        "cleanup_old_models": {
            "keep_last_n": 4,
            "keep_best_n": 1,
            "keep": [100, 200, 300, 400]
        },  # overwrite default
        "optimizer": {"class": "adam", "epsilon": 1e-9},
        "learning_rates": [1e-4] * 400,
        "gradient_clip_norm": 2.0,
        #############
        "batch_size": 300 * 16000,
        "max_seq_length": {"audio_features": 30 * 16000},
        "max_seqs": 200,
        "torch_amp_options": {"dtype": "bfloat16"},
        "batch_drop_last": True,  # otherwise might cause issues in indexing after local sort in train_step
        # also, not doing this might cause shape issues in the official GradTTS implementation, because masks
        # and features might mis-align if there is one sequences shorter than 2 seconds remaining in the last batch
    }

    train_args = {
        "network_module": net_module,
        "net_args": {"config": asdict(model_config)},
        "config": config,
        "debug": True,
    }
    
    train_name = "train.small_" + net_module + "_glowttsv2align_400eps"

    train_job = run_exp(train_name, train_args=train_args, num_epochs=400)
    train_job.rqmt["gpu_mem"] = 48
    #train_job.hold()
    #train_job.move_to_hpc = True

    tts_model = prepare_tts_model(train_name, train_job, train_args, get_specific_checkpoint=400)

    train_small_tts_bliss = get_tts_extended_bliss(loq_corpus_key="train.small",
                                                   lexicon_loq_corpus_key="train.small")
    train_medium_wo_small_tts_bliss = get_tts_extended_bliss(loq_corpus_key="train.medium-wo-small",
                                                             lexicon_loq_corpus_key="train.medium")
    train_small_bliss = get_bliss_corpus_dict()["train.small"]
    train_medium_wo_small_bliss = get_bliss_corpus_dict()["train.medium-wo-small"]

    syn_name = "train.small_shuffled_syn"
    train_small_shuffled_dataset = build_dynamic_speakers_generating_dataset(
        text_bliss=train_small_tts_bliss,
        speaker_embedding_hdf=speaker_embedding_hdf,
        speaker_embedding_size=256,
        num_splits=1,
        distribute_speakers=True,
        loq_corpus_key="train.small",
    )
    train_small_static_dataset = build_dynamic_speakers_generating_dataset(
        text_bliss=train_small_tts_bliss,
        speaker_embedding_hdf=speaker_embedding_hdf,
        speaker_embedding_size=256,
        num_splits=1,
        distribute_speakers=False,
        loq_corpus_key="train.small",
    )
    train_medium_wo_small_shuffled_dataset = build_dynamic_speakers_generating_dataset(
        text_bliss=train_medium_wo_small_tts_bliss,
        speaker_embedding_hdf=speaker_embedding_hdf,
        speaker_embedding_size=256,
        num_splits=1,
        distribute_speakers=True,
        loq_corpus_key="train.medium",
    )

    for noise_scale in [0.5, 0.6, 0.7, 0.8, 0.9]:
        decoder_options_synthetic_ = copy.deepcopy(decoder_options_synthetic)
        decoder_options_synthetic_["gradtts_noise_scale"] = noise_scale
        result_corpus = synthesize_dataset(
            syn_name + "_noise_%.2f" % noise_scale,
            tts_model=tts_model,
            decoder="grad_tts.simple_gl_decoder",
            decoder_options=decoder_options_synthetic_,
            corpus_name="loquacious-train-small",
            dataset=train_small_shuffled_dataset,
        )

        merged_corpus_with_text = CorpusReplaceOrthFromReferenceCorpus(
            bliss_corpus=result_corpus,
            reference_bliss_corpus=train_small_bliss,
        ).out_corpus
        tk.register_output(prefix + "/" + train_name + "/" + "generated_synthetic/train-small-shuffled-speakers.xml.gz",
                           merged_corpus_with_text)
        from ....storage import add_synthetic_data
        ogg_zip_job = BlissToOggZipJob(
            bliss_corpus=merged_corpus_with_text,
            no_conversion=True,
            returnn_python_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
        )
        ogg_zip_job.rqmt = {"cpu": 1, "mem": 4, "time": 4}
        add_synthetic_data(
            train_name + f"_noise_{noise_scale:.2f}_train-small-shuffled",
            ogg_zip=ogg_zip_job.out_ogg_zip,
            bliss=merged_corpus_with_text,
        )

    # best is 0.8, so continue with that
    syn_name = "train.small_static_syn"
    noise_scale = 0.8
    decoder_options_synthetic_best = copy.deepcopy(decoder_options_synthetic)
    decoder_options_synthetic_best["gradtts_noise_scale"] = 0.8

    result_corpus = synthesize_dataset(
        syn_name + "_noise_%.2f" % noise_scale,
        tts_model=tts_model,
        decoder="grad_tts.simple_gl_decoder",
        decoder_options=decoder_options_synthetic_best,
        corpus_name="loquacious-train-small",
        dataset=train_small_static_dataset,
    )

    merged_corpus_with_text = CorpusReplaceOrthFromReferenceCorpus(
        bliss_corpus=result_corpus,
        reference_bliss_corpus=train_small_bliss,
    ).out_corpus
    tk.register_output(prefix + "/" + train_name + "/" + "generated_synthetic/train-small-static-speakers.xml.gz",
                       merged_corpus_with_text)
    from ....storage import add_synthetic_data
    ogg_zip_job = BlissToOggZipJob(
        bliss_corpus=merged_corpus_with_text,
        no_conversion=True,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
    )
    ogg_zip_job.rqmt = {"cpu": 1, "mem": 4, "time": 4}
    add_synthetic_data(
        train_name + f"_noise_{noise_scale:.2f}_train-small-static",
        ogg_zip=ogg_zip_job.out_ogg_zip,
        bliss=merged_corpus_with_text,
    )

    # medium without small
    syn_name = "train.medium-wo-small_shuffled_syn"
    result_corpus = synthesize_dataset(
        syn_name + "_noise_%.2f" % noise_scale,
        tts_model=tts_model,
        decoder="grad_tts.simple_gl_decoder",
        decoder_options=decoder_options_synthetic_best,
        corpus_name="loquacious-train-medium-wo-small",
        dataset=train_medium_wo_small_shuffled_dataset,
        time_rqmt=96,
    )

    merged_corpus_with_text = CorpusReplaceOrthFromReferenceCorpus(
        bliss_corpus=result_corpus,
        reference_bliss_corpus=train_medium_wo_small_bliss,
    ).out_corpus
    tk.register_output(prefix + "/" + train_name + "/" + "generated_synthetic/train-medium-wo-small-shuffled-speakers.xml.gz",
                       merged_corpus_with_text)
    from ....storage import add_synthetic_data
    ogg_zip_job = BlissToOggZipJob(
        bliss_corpus=merged_corpus_with_text,
        no_conversion=True,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
    )
    ogg_zip_job.rqmt = {"cpu": 1, "mem": 8, "time": 24}
    add_synthetic_data(
        train_name + f"_noise_{noise_scale:.2f}_train-medium-wo-small-shuffled",
        ogg_zip=ogg_zip_job.out_ogg_zip,
        bliss=merged_corpus_with_text,
    )
