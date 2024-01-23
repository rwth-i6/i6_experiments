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
from ..data.tts_phon import get_tts_log_mel_datastream, build_fixed_speakers_generating_dataset, get_tts_extended_bliss



from ..default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ..storage import add_duration, vocoders, add_synthetic_data



def get_flow_tts():
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

    prefix = "experiments/jaist_project/standalone_2024/glow_tts/"
    training_datasets = build_training_dataset(ls_corpus_key="train-clean-100", silence_preprocessed=False, partition_epoch=1)

    def run_exp(name, params, net_module, config, decoder_options, extra_decoder=None, use_custom_engine=False, debug=False, num_epochs=100):
        train_config = get_training_config(
            training_datasets=training_datasets,
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

    from ..pytorch_networks.glow_tts.lukas_baseline import DbMelFeatureExtractionConfig, Config
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

    net_module = "glow_tts.lukas_baseline"

    model_config = Config(
        n_vocab=training_datasets.datastreams["phonemes"].vocab_size,
        hidden_channels=192,
        filter_channels=768,
        filter_channels_dp=256,
        out_channels=80,
        n_speakers=251,
        gin_channels=256,
        p_dropout=0.1,
        p_dropout_dec=0.05,
        dilation_rate=1,
        n_sqz=2,
        prenet=True,
        window_size=4,
        fe_config=fe_config,
        mean_only=True,
    )

    decoder_options = {
        "norm_mean": norm[0],
        "norm_std_dev": norm[1]
    }

    params = {
        "config": asdict(model_config)
    }
    local_config = copy.deepcopy(config)
    train = run_exp(net_module + "_v1", params, net_module, local_config, decoder_options=decoder_options, debug=True)
    # train.hold()
    
    params = {
        "config": asdict(model_config)
    }
    local_config = copy.deepcopy(config)
    local_config["batch_size"] = 600 * 16000
    train = run_exp(net_module + "_bs600", params, net_module, local_config, decoder_options=decoder_options, debug=True)
    # train.hold()

    # With new GL decoder
    decoder_options = copy.deepcopy(decoder_options)
    vocoder = vocoders["blstm_gl_v1"]
    decoder_options["gl_net_checkpoint"] = vocoder.checkpoint
    decoder_options["gl_net_config"] = vocoder.config
    train = run_exp(net_module + "_bs600_newgl", params, net_module, local_config, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options,
                    debug=True)

    # lesser noise scale
    decoder_options_noise = copy.deepcopy(decoder_options)
    vocoder = vocoders["blstm_gl_v1"]
    decoder_options_noise["gl_net_checkpoint"] = vocoder.checkpoint
    decoder_options_noise["gl_net_config"] = vocoder.config
    decoder_options_noise["glowtts_noise_scale"] = 0.1
    train = run_exp(net_module + "_bs600_newgl_noise0.1", params, net_module, local_config, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options_noise,
                    debug=True, num_epochs=100)

    decoder_options_noise2 = copy.deepcopy(decoder_options_noise)
    decoder_options_noise2["glowtts_noise_scale"] = 0.3
    train = run_exp(net_module + "_bs600_newgl_noise0.3", params, net_module, local_config, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options_noise2,
                    debug=True, num_epochs=100)

    # With new GL decoder and corrected training scheme using 200 epochs
    local_config = copy.deepcopy(config)
    local_config["batch_size"] = 600 * 16000
    local_config["learning_rates"] = list(np.linspace(5e-5, 5e-4, 100)) + list(np.linspace(5e-4, 5e-7, 100))
    decoder_options = copy.deepcopy(decoder_options)
    vocoder = vocoders["blstm_gl_v1"]
    decoder_options["gl_net_checkpoint"] = vocoder.checkpoint
    decoder_options["gl_net_config"] = vocoder.config
    train = run_exp(net_module + "_bs600_v2_newgl", params, net_module, local_config, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options,
                    debug=True, num_epochs=200)
    # train.hold()

    decoder_options_noise2 = copy.deepcopy(decoder_options_noise)
    decoder_options_noise2["glowtts_noise_scale"] = 0.1
    train = run_exp(net_module + "_bs600_v2_newgl_noise0.1_ogg", params, net_module, local_config, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options_noise2,
                    debug=True, num_epochs=200)

    decoder_options_noise2 = copy.deepcopy(decoder_options_noise)
    decoder_options_noise2["glowtts_noise_scale"] = 0.3
    train = run_exp(net_module + "_bs600_v2_newgl_noise0.3", params, net_module, local_config, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options_noise2,
                    debug=True, num_epochs=200)

    decoder_options_noise2 = copy.deepcopy(decoder_options_noise)
    decoder_options_noise2["glowtts_noise_scale"] = 0.5
    train = run_exp(net_module + "_bs600_v2_newgl_noise0.5", params, net_module, local_config, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options_noise2,
                    debug=True, num_epochs=200)

    decoder_options_noise2 = copy.deepcopy(decoder_options_noise)
    decoder_options_noise2["glowtts_noise_scale"] = 0.7
    train = run_exp(net_module + "_bs600_v2_newgl_noise0.7", params, net_module, local_config, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options_noise2,
                    debug=True, num_epochs=200)
    
    
    decoder_options_speedtest = copy.deepcopy(decoder_options_noise)
    decoder_options_speedtest["glowtts_noise_scale"] = 0.3
    decoder_options_speedtest["gl_momentum"] = 0.0
    decoder_options_speedtest["gl_iter"] = 1
    decoder_options_speedtest["create_plots"] = False
    train = run_exp(net_module + "_bs600_v2_newgl_noise0.3_speedtest", params, net_module, local_config, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options_speedtest,
                    debug=True, num_epochs=200)
    
    synthetic_corpus = generate_synthetic(net_module + "_bs600_v2_newgl_noise0.3_syn", "train-clean-360", train.out_checkpoints[200], params, net_module, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options_speedtest, debug=True)
    tk.register_output(prefix + "/test_corpus.xml.gz", synthetic_corpus)

    synthetic_corpus = generate_synthetic(net_module + "_bs600_v2_newgl_noise0.3_syn", "train-clean-100", train.out_checkpoints[200], params, net_module, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options_speedtest, debug=True)

    decoder_options_speedtest_00 = copy.deepcopy(decoder_options_speedtest)
    decoder_options_speedtest_00["glowtts_noise_scale"] = 0.0
    synthetic_corpus = generate_synthetic(net_module + "_bs600_v2_newgl_noise0.0_syn", "train-clean-100", train.out_checkpoints[200], params, net_module, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options_speedtest_00, debug=True)

    decoder_options_speedtest_05 = copy.deepcopy(decoder_options_speedtest)
    decoder_options_speedtest_05["glowtts_noise_scale"] = 0.5
    synthetic_corpus = generate_synthetic(net_module + "_bs600_v2_newgl_noise0.5_syn", "train-clean-100", train.out_checkpoints[200], params, net_module, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options_speedtest_05, debug=True)

    decoder_options_speedtest_07 = copy.deepcopy(decoder_options_speedtest)
    decoder_options_speedtest_07["glowtts_noise_scale"] = 0.7
    synthetic_corpus = generate_synthetic(net_module + "_bs600_v2_newgl_noise0.7_syn", "train-clean-100",
                                          train.out_checkpoints[200], params, net_module,
                                          extra_decoder="glow_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_speedtest_07, debug=True)

    decoder_options_speedtest_10 = copy.deepcopy(decoder_options_speedtest)
    decoder_options_speedtest_10["glowtts_noise_scale"] = 1.0
    synthetic_corpus = generate_synthetic(net_module + "_bs600_v2_newgl_noise1.0_syn", "train-clean-100",
                                          train.out_checkpoints[200], params, net_module,
                                          extra_decoder="glow_tts.simple_gl_decoder",
                                          decoder_options=decoder_options_speedtest_10, debug=True)

    durations = local_extract_durations(net_module + "_bs600_v2", train.out_checkpoints[200], params, net_module, debug=True)
    ##################################

    # with std dev
    model_config_stddev = copy.deepcopy(model_config)
    model_config_stddev.mean_only = False
    params = {
        "config": asdict(model_config_stddev)
    }
    local_config = copy.deepcopy(config)
    local_config["batch_size"] = 600 * 16000
    local_config["learning_rates"] = list(np.linspace(5e-5, 5e-4, 100)) + list(np.linspace(5e-4, 5e-7, 100))
    decoder_options = copy.deepcopy(decoder_options)
    vocoder = vocoders["blstm_gl_v1"]
    decoder_options["gl_net_checkpoint"] = vocoder.checkpoint
    decoder_options["gl_net_config"] = vocoder.config
    train = run_exp(net_module + "_bs600_v2_stddev_newgl", params, net_module, local_config, extra_decoder="glow_tts.simple_gl_decoder", decoder_options=decoder_options,
                    debug=True, num_epochs=200)
    # train.hold()