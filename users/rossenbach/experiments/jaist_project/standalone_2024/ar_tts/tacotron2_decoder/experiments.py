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

from ...data.tts_phon import build_durationtts_training_dataset
from ...data.tts_phon import get_vocab_datastream
from ...data.tts_phon import get_tts_log_mel_datastream, build_fixed_speakers_generating_dataset, get_tts_extended_bliss

from ...config import get_training_config, get_forward_config
from ...pipeline import training, tts_eval, tts_generation


from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...storage import duration_alignments, vocoders, add_synthetic_data



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

    prefix = "experiments/jaist_project/standalone_2024/ar_tts/tacotron2_decoder/"

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
        forward_job = tts_eval(
            prefix_name=prefix + name,
            returnn_config=forward_config,
            checkpoint=train_job.out_checkpoints[200],
            returnn_exe=RETURNN_EXE,
            returnn_root=MINI_RETURNN_ROOT,
        )
        tk.register_output(prefix + name + "/audio_files", forward_job.out_files["audio_files"])
        return train_job, forward_job

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

    train, forward = run_exp(net_module + "_fromglow_v1", params, net_module, config,
                             extra_decoder="nar_tts.fastspeech_like.simple_gl_decoder", decoder_options=decoder_options,duration_hdf=duration_hdf, debug=True)
    # train.hold()
