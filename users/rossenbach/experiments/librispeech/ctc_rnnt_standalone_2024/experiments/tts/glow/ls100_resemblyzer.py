# Experiments to reproduce the jaist_project baseline

import copy
import numpy as np
from sisyphus import tk
from dataclasses import asdict


from i6_core.returnn.oggzip import BlissToOggZipJob


from i6_experiments.common.setups.returnn.datastreams.audio import DBMelFilterbankOptions
from i6_experiments.users.rossenbach.tts.speaker_embedding import ResemblyzerEmbeddingHDFFromBliss

from ....data.tts.aligner import build_training_dataset
from ....config import get_forward_config
from ....pipeline import training, prepare_tts_model, TTSModel, tts_eval_v2, extract_durations
from ....data.tts.tts_phon import get_tts_log_mel_datastream, build_durationtts_training_dataset
from ....data.tts.tts_phon import build_dynamic_speakers_generating_dataset
from ....data.common import get_bliss_corpus_dict
from ....data.tts.tts_phon import get_tts_extended_bliss, get_tts_bliss_and_zip

from ....default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ....storage import vocoders, add_duration


def run_resemblyzer_flow_tts():
    """
    :return: durations_hdf
    """

    prefix = "experiments/librispeech/ctc_rnnt_standalone_2024/tts/glow_tts_resemblyzer/"

    ls_bliss = get_bliss_corpus_dict(audio_format="ogg")["train-clean-100"]
    speaker_embedding_hdf = ResemblyzerEmbeddingHDFFromBliss(ls_bliss).out_speaker_hdf
    training_datasets = build_training_dataset(
        ls_corpus_key="train-clean-100",
        partition_epoch=1,
        dynamic_speaker_embeddings=speaker_embedding_hdf,
        dynamic_speaker_embedding_size=256,
    )

    def run_exp(name, train_args, target_durations=None, num_epochs=100):
        if target_durations is not None:
            training_datasets_ = build_durationtts_training_dataset(duration_hdf=target_durations,
                                                                    ls_corpus_key="train-clean-100")
        else:
            training_datasets_ = training_datasets

        train_job = training(
            training_name=prefix + name,
            datasets=training_datasets_,
            train_args=train_args,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            num_epochs=num_epochs,
        )
        return train_job

    def eval_exp(name, tts_model: TTSModel, decoder, decoder_options):
        forward_config = get_forward_config(
            network_module=tts_model.network_module,
            net_args=tts_model.net_args,
            decoder=decoder,
            decoder_args=decoder_options,
            config={
                "forward": training_datasets.cv.as_returnn_opts()
            },
            debug=False,
        )
        forward_job = tts_eval_v2(
            prefix_name=prefix + name,
            returnn_config=forward_config,
            checkpoint=tts_model.checkpoint,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
            mem_rqmt=12,
            use_gpu=True,
        )
        forward_job.add_alias(prefix + "/" + tts_model.prefix_name + "/" + name + "/forward")
        tk.register_output(prefix + "/" + tts_model.prefix_name + "/" + name + "/audio_files",
                           forward_job.out_files["audio_files"])
        corpus = forward_job.out_files["out_corpus.xml.gz"]
        from ....pipeline import evaluate_nisqa
        evaluate_nisqa(prefix_name=prefix + "/" + tts_model.prefix_name + "/" + name, bliss_corpus=corpus)
        from i6_experiments.users.rossenbach.experiments.jaist_project.evaluation.swer import run_evaluate_reference_swer
        from i6_experiments.users.rossenbach.corpus.transform import MergeCorporaWithPathResolveJob, MergeStrategy
        realpath_corpus = MergeCorporaWithPathResolveJob(bliss_corpora=[corpus],
                                                         name="train-clean-100",  # important to keep the original sequence names for matching later
                                                         merge_strategy=MergeStrategy.FLAT
                                                         )
        run_evaluate_reference_swer(prefix=prefix + "/" + tts_model.prefix_name, bliss=realpath_corpus.out_merged_corpus)
        
    def local_extract_durations(name, tts_model: TTSModel, debug=False):
        forward_config = get_forward_config(
            network_module=tts_model.network_module,
            net_args=tts_model.net_args,
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
            checkpoint=tts_model.checkpoint,
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
        )
        tk.register_output(prefix + name + "/durations.hdf", durations_hdf)
        add_duration(name, durations_hdf)
        return durations_hdf
    
    from ....data.tts.tts_phon import GeneratingDataset

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
        forward_config.config["max_seqs"] = 32
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
        # forward_job.add_alias(prefix + "/" + tts_model.prefix_name + "/" + name + "/forward")
        tk.register_output(local_prefix + "/" + tts_model.prefix_name + "/" + name + "/audio_files", forward_job.out_files["audio_files"])
        corpus = forward_job.out_files["out_corpus.xml.gz"]
        from i6_experiments.users.rossenbach.corpus.transform import MergeCorporaWithPathResolveJob, MergeStrategy
        realpath_corpus = MergeCorporaWithPathResolveJob(bliss_corpora=[corpus],
                                                         name=corpus_name,  # important to keep the original sequence names for matching later
                                                         merge_strategy=MergeStrategy.FLAT
                                                         )
        return realpath_corpus.out_merged_corpus


    log_mel_datastream = get_tts_log_mel_datastream(ls_corpus_key="train-clean-100", silence_preprocessed=False)

    # verify that normalization exists
    assert "norm_mean" in log_mel_datastream.additional_options
    assert "norm_std_dev" in log_mel_datastream.additional_options

    norm = (log_mel_datastream.additional_options["norm_mean"], log_mel_datastream.additional_options["norm_std_dev"])

    from ....pytorch_networks.tts_shared.encoder.transformer import (
        GlowTTSMultiHeadAttentionV1Config,
        TTSEncoderPreNetV1Config
    )
    from ....pytorch_networks.glow_tts.glow_tts_dynamic_speakers_v1 import (
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

    net_module = "glow_tts.glow_tts_dynamic_speakers_v1"

    vocoder = vocoders["blstm_gl_v1"]
    decoder_options_base = {
        "norm_mean": norm[0],
        "norm_std_dev": norm[1],
        "gl_net_checkpoint": vocoder.checkpoint,
        "gl_net_config": vocoder.config,
    }
    
    decoder_options = copy.deepcopy(decoder_options_base)
    decoder_options["glowtts_noise_scale"] = 0.3

    decoder_options_synthetic = copy.deepcopy(decoder_options)
    decoder_options_synthetic["glowtts_noise_scale"] = 0.7
    decoder_options_synthetic["gl_momentum"] = 0.0
    decoder_options_synthetic["gl_iter"] = 1
    decoder_options_synthetic["create_plots"] = False

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
        speaker_embedding_size=256,
        mean_only=True,
    )

    params_base256 = {
        "config": asdict(model_config_base256)
    }

    config = {
        "cleanup_old_models": {
            "keep_last_n": 4,
            "keep_best_n": 1,
            "keep": [100, 200, 300, 400]
        },  # overwrite default
        "optimizer": {"class": "adam", "epsilon": 1e-9},
        "learning_rates": list(np.linspace(5e-5, 5e-4, 100)) + list(np.linspace(5e-4, 5e-7, 300)),
        "gradient_clip_norm": 2.0,
        #############
        "batch_size": 600 * 16000,
        "max_seq_length": {"audio_features": 30 * 16000},
        "batch_drop_last": True,  # otherwise might cause issues in indexing after local sort in train_step
        "max_seqs": 200,
        "torch_amp_options": {"dtype": "bfloat16"},
    }
    
    train_args = {
        "network_module": net_module,
        "net_args": params_base256,
        "config": config,
        "debug": True,
    }

    train_name = "ls100_" + net_module + "_base256_400eps"

    train_job = run_exp(train_name, train_args=train_args, num_epochs=400)
    train_job.rqmt["gpu_mem"] = 24
    tts_model = prepare_tts_model(train_name, train_job, train_args, get_specific_checkpoint=400)
    eval_exp("base", tts_model=tts_model, decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options)

    train_clean_360_tts_bliss = get_tts_extended_bliss(ls_corpus_key="train-clean-360",
                                                       lexicon_ls_corpus_key="train-clean-460")
    train_clean_360_bliss = get_bliss_corpus_dict()["train-clean-360"]

    # Simple generation of ls-360 data
    syn_name = "train_clean_360_syn"
    dataset_part = build_dynamic_speakers_generating_dataset(
        text_bliss=train_clean_360_tts_bliss,
        speaker_embedding_hdf=speaker_embedding_hdf,
        speaker_embedding_size=256,
        num_splits=1,  # we already splitted before
        distribute_speakers=True,
        ls_corpus_key="train-clean-100",
    )
    result_corpus = synthesize_dataset(
        syn_name,
        tts_model=tts_model,
        decoder="glow_tts.simple_gl_decoder_dynamic_speakers",
        decoder_options=decoder_options_synthetic,
        corpus_name="train-clean-360",
        dataset=dataset_part,
    )

    from i6_core.corpus.convert import CorpusReplaceOrthFromReferenceCorpus
    merged_corpus_with_text = CorpusReplaceOrthFromReferenceCorpus(
        bliss_corpus=result_corpus,
        reference_bliss_corpus=train_clean_360_bliss,
    ).out_corpus
    tk.register_output(prefix + "/" + train_name + "/" + "generated_synthetic/train-clean-360.xml.gz", merged_corpus_with_text)
    
    from ....storage import add_synthetic_data
    ogg_zip_job = BlissToOggZipJob(
        bliss_corpus=merged_corpus_with_text,
        no_conversion=True,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=MINI_RETURNN_ROOT,
    )
    ogg_zip_job.rqmt = {"cpu": 1, "mem": 4, "time": 4}
    print(train_name + "_train-clean-360")
    add_synthetic_data(
        train_name + "_train-clean-360",
        ogg_zip=ogg_zip_job.out_ogg_zip,
        bliss=merged_corpus_with_text,
    )

