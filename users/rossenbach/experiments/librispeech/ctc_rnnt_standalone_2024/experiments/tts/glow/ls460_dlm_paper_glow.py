# Experiments to reproduce the jaist_project baseline

import copy
import numpy as np
from sisyphus import tk, Path
from dataclasses import asdict

from i6_core.lexicon.modification import WriteLexiconJob, MergeLexiconJob
from i6_core.lib.lexicon import Lexicon, Lemma
from i6_core.returnn.oggzip import BlissToOggZipJob

from i6_experiments.common.setups.returnn.datastreams.audio import DBMelFilterbankOptions
from i6_experiments.users.rossenbach.setups.tts.preprocessing import process_corpus_text_with_extended_lexicon

from ....config import get_forward_config
from ....data.tts.aligner import build_training_dataset
from ....data.tts.tts_phon import (
    get_tts_log_mel_datastream,
    build_durationtts_training_dataset,
    build_fixed_speakers_generating_dataset,
    GeneratingDataset
)
from ....default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ....pipeline import training, prepare_tts_model, TTSModel, tts_eval_v2
from ....storage import vocoders, add_synthetic_data_lexicon, add_synthetic_data


def run_flow_tts_460h_dlm_paper():
    """
    Calls the training of a LibriSpeech 460h GlowTTS model and
    creates all of the LibriSpeech LM data in chunks of about 100 hours each
    """
    prefix = "experiments/librispeech/ctc_rnnt_standalone_2024/tts/glow_tts_460h_dlm_paper/"
    training_datasets = build_training_dataset(ls_corpus_key="train-clean-460", partition_epoch=1)

    def run_exp(name, train_args, target_durations=None, num_epochs=100):
        if target_durations is not None:
            training_datasets_ = build_durationtts_training_dataset(duration_hdf=target_durations, ls_corpus_key="train-clean-460")
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
        tk.register_output(prefix + "/" + tts_model.prefix_name + "/" + name + "/audio_files", forward_job.out_files["audio_files"])
        corpus = forward_job.out_files["out_corpus.xml.gz"]
        from ....pipeline import evaluate_nisqa
        evaluate_nisqa(prefix_name=prefix + "/" + tts_model.prefix_name + "/" + name, bliss_corpus=corpus)
        from i6_experiments.users.rossenbach.experiments.jaist_project.evaluation.swer import run_evaluate_reference_swer
        from i6_experiments.users.rossenbach.corpus.transform import MergeCorporaWithPathResolveJob, MergeStrategy
        realpath_corpus = MergeCorporaWithPathResolveJob(bliss_corpora=[corpus],
                                                         name="train-clean-460",  # important to keep the original sequence names for matching later
                                                         merge_strategy=MergeStrategy.FLAT
                                                         )
        run_evaluate_reference_swer(prefix=prefix + "/" + tts_model.prefix_name + "/" + name, bliss=realpath_corpus.out_merged_corpus)


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

    log_mel_datastream = get_tts_log_mel_datastream(ls_corpus_key="train-clean-460", silence_preprocessed=False)

    # verify that normalization exists
    assert "norm_mean" in log_mel_datastream.additional_options
    assert "norm_std_dev" in log_mel_datastream.additional_options

    norm = (log_mel_datastream.additional_options["norm_mean"], log_mel_datastream.additional_options["norm_std_dev"])

    from ....pytorch_networks.tts_shared.encoder.transformer import (
        GlowTTSMultiHeadAttentionV1Config,
        TTSEncoderPreNetV1Config
    )
    from ....pytorch_networks.glow_tts.glow_tts_v1 import (
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

    net_module = "glow_tts.glow_tts_v1"

    vocoder = vocoders["blstm_gl_v1_ls460"]
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
    decoder_options_synthetic["gl_momentum"] = 0.99
    decoder_options_synthetic["gl_iter"] = 32
    decoder_options_synthetic["create_plots"] = False
    decoder_options_synthetic["num_pool_processes"] = 8

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
        num_speakers=training_datasets.datastreams["speaker_labels"].vocab_size,
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

    train_name = net_module + "_base256_400eps"
    train_job = run_exp(train_name, train_args=train_args, num_epochs=400)
    train_job.rqmt["gpu_mem"] = 24
    tts_model = prepare_tts_model(train_name, train_job, train_args, get_specific_checkpoint=400)
    eval_exp("base", tts_model=tts_model, decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options)

    for noise_scale in [0.3, 0.5, 0.7, 0.9]:
        decoder_options_noised = copy.deepcopy(decoder_options)
        decoder_options_noised["glowtts_noise_scale"] = noise_scale
        eval_exp("noise_%.1f" % noise_scale, tts_model=tts_model, decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options_noised)


    # test generation of all of librispeech data
    prefix = "experiments/librispeech/ctc_rnnt_standalone_2024/ls_lm_data"
    from ....data.tts.generation import create_data_lexicon, bliss_from_text
    from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data
    lm_data = get_librispeech_normalized_lm_data()

    # misuse shuffle and split segments
    from i6_core.corpus.segments import ShuffleAndSplitSegmentsJob
    shuffle_job = ShuffleAndSplitSegmentsJob(
        segment_file=lm_data,
        split={"part%i" % (i + 1): 1.0/750.0 for i in range(750)}
    )
    shuffle_job.add_alias(prefix + "/shuffle_job")


    # Full lexicon
    lm_data_bliss = bliss_from_text(prefix=prefix, name="librispeech-full", lm_text=lm_data)
    lm_data_lexicon = create_data_lexicon(prefix=prefix + "/librispeech-full_lexicon", lexicon_bliss=lm_data_bliss)
    l = Lexicon()
    l.add_lemma(
        Lemma(
            orth=["HHHH"],
            phon=["HH HH AH"]
        )
    )
    l.add_lemma(
        Lemma(
            orth=["HHH"],
            phon=["HH HH AH"]
        )
    )
    lexicon_edit_full = WriteLexiconJob(static_lexicon=l).out_bliss_lexicon
    lm_data_lexicon = MergeLexiconJob(bliss_lexica=[lm_data_lexicon, lexicon_edit_full]).out_bliss_lexicon
    tk.register_output(prefix + "librispeech-full_lexicon.xml.gz", lm_data_lexicon)

    add_synthetic_data_lexicon("ls_lm_data_lexicon", lm_data_lexicon)

    for i in range(750):
        index = i+1
        lm_data_part = shuffle_job.out_segments["part%i" % index]
        lm_data_part_bliss = bliss_from_text(prefix=prefix, name="librispeech-lm-part%i" % index, lm_text=lm_data_part)

        tk.register_output(prefix + "/lm_data_part%i.xml.gz" % index, lm_data_part_bliss)
        lm_data_part_lexicon = create_data_lexicon(prefix=prefix + "/lm_data_part%i_lexicon" % index, lexicon_bliss=lm_data_part_bliss)
        if index in [198, 312, 421]:
            l = Lexicon()
            l.add_lemma(
                Lemma(
                    orth=["HHHH"] if index == 198 else ["HHH"],
                    phon=["HH HH AH"]
                )
            )
            lexicon_edit = WriteLexiconJob(static_lexicon=l).out_bliss_lexicon
            lm_data_part_lexicon = MergeLexiconJob(bliss_lexica=[lm_data_part_lexicon, lexicon_edit]).out_bliss_lexicon

        tk.register_output(prefix + "/lm_data_part%i_lexicon.xml.gz" % index, lm_data_part_lexicon)

        lm_data_part_tts_bliss = process_corpus_text_with_extended_lexicon(
            bliss_corpus=lm_data_part_bliss,
            lexicon=lm_data_part_lexicon,
            prefix=prefix + "/processing_part%i" % index,
        )
        tk.register_output(prefix + "/lm_data_part%i_tts_input.xml.gz" % index, lm_data_part_tts_bliss)

        dataset_part = build_fixed_speakers_generating_dataset(
            text_bliss=lm_data_part_tts_bliss,
            num_splits=1,  # we already splitted before
            ls_corpus_key="train-clean-460",
            randomize_speaker=True
        )

        result_corpus = synthesize_dataset(
            "librispeech-lm-part%i" % index,
            tts_model=tts_model,
            decoder="glow_tts.simple_gl_decoder",
            decoder_options=decoder_options_synthetic,
            corpus_name="librispeech-lm-part%i" % index,
            dataset=dataset_part,
        )

        from i6_core.corpus.convert import CorpusReplaceOrthFromReferenceCorpus
        merged_corpus_with_text = CorpusReplaceOrthFromReferenceCorpus(
            bliss_corpus=result_corpus,
            reference_bliss_corpus=lm_data_part_bliss,
        ).out_corpus

        tk.register_output(prefix + "/lm_data_part%i_synthesized.xml.gz" % index, merged_corpus_with_text)

        ogg_zip_job = BlissToOggZipJob(
            bliss_corpus=merged_corpus_with_text,
            no_conversion=True,
            returnn_python_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
        )
        ogg_zip_job.rqmt = {"cpu": 1, "mem": 4, "time": 4}
        ogg_zip_job.add_alias(prefix + "/part%i_ogg_zip" % index)
        tk.register_output(prefix + "/lm_data_part%i_ogg.zip" % index, ogg_zip_job.out_ogg_zip)
        add_synthetic_data("glowtts460_lm_data_%i" % i, ogg_zip_job.out_ogg_zip, bliss=merged_corpus_with_text)
