from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset, build_short_dev_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon, get_eow_lexicon, synthetic_bliss_to_ogg_zip
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT, KENLM_BINARY_PATH
from ...pipeline import training, prepare_asr_model, search, ASRModel, evaluate_all
from ...tune_eval import tune_and_evaluate_helper
from ... import PACKAGE
from ...storage import get_synthetic_data


def eow_phon_small_trial_nostress():
    prefix_name = "experiments/loquacious/standalone_2025/ctc_eow_phon_small_trial_nostress"

    # export some stuff
    tk.register_output(prefix_name + "/recog_lex.xml", get_eow_lexicon(g2p_loquacious_key="train.small", with_g2p=False, variant=2))
    tk.register_output(prefix_name + "/train_lex.xml", get_eow_lexicon(g2p_loquacious_key="train.small", with_g2p=True, variant=2))

    tk.register_output(prefix_name + "/recog_lex_60%.xml", get_eow_lexicon(g2p_loquacious_key="train.small", with_g2p=False, variant=3))
    tk.register_output(prefix_name + "/recog_lex_80%.xml", get_eow_lexicon(g2p_loquacious_key="train.small", with_g2p=False, variant=4))

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.1000",
    )
    
    train_settings_laplace4 = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.4000",
    )


    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v2 import DecoderConfig


    from ...pytorch_networks.ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1_cfg import \
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig, LogMelFeatureExtractionV1Config, ConformerPosEmbConfig

    fe_config = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.025,
        hop_size=0.01,
        f_min=60,
        f_max=7600,
        min_amp=1e-10,
        num_filters=80,
        center=False,
    )
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,  # classic style
        num_repeat_feat=5,
    )
    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=512,
        activation=None,
    )

    # Try to do like returnn frontend
    posemb_config = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )

    train_config_24gbgpu_amp = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 240)) + list(
                np.linspace(5e-4, 5e-5, 240)) + list(np.linspace(5e-5, 1e-7, 20)),
        #############
        "batch_size": 240 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
        "torch_amp_options": {"dtype": "bfloat16"},
    }

    network_module = "ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1"
    global_train_args = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module,
        "use_speed_perturbation": True,
        "debug": True,
    }


    from i6_core.lm.kenlm import CreateBinaryLMJob
    arpa_4gram_binary_lm_job = CreateBinaryLMJob(
        arpa_lm=tk.Path("/work/asr4/rossenbach/corpora/loquacious/LoquaciousAdditionalResources/4gram-pruned-test2.arpa.gz"),
        kenlm_binary_folder=KENLM_BINARY_PATH
    )
    arpa_4gram_lm = arpa_4gram_binary_lm_job.out_lm

    train_data_eow_phon = build_eow_phon_training_datasets(
        prefix=prefix_name,
        loquacious_key="train.small",
        settings=train_settings,
        variant=2,
    )

    label_datastream_eow_phon = cast(LabelDatastream, train_data_eow_phon.datastreams["labels"])
    vocab_size_without_blank = label_datastream_eow_phon.vocab_size

    short_dev_dataset_tuples = {
        "dev.short": build_short_dev_dataset(train_settings)
    }

    dev_dataset_tuples = {}
    for testset in ["dev.commonvoice", "dev.librispeech", "dev.voxpopuli", "dev.yodas"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    test_dataset_tuples = {}
    for testset in ["test.commonvoice", "test.librispeech", "test.voxpopuli", "test.yodas"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )


    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        pos_emb_config=posemb_config,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        mhsa_with_bias=True,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=11,
        dropout_broadcast_axes=None, # No dropout broadcast yet to properly compare
        module_list=["ff", "conv", "mhsa", "ff"],
        module_scales=[0.5, 1.0, 1.0, 0.5],
        aux_ctc_loss_layers=None,
        aux_ctc_loss_scales=None,
    )


    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(variant=2),
        returnn_vocab=label_datastream_eow_phon.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    # default_decoder_config_bpe = DecoderConfig(
    #     lexicon=get_text_lexicon(prefix=prefix_name, loquacious_key="train.small", bpe_size=BPE_SIZE),
    #     returnn_vocab=label_datastream_bpe.vocab,
    #     beam_size=1024,
    #     beam_size_token=16,  # makes it much faster
    #     arpa_lm=arpa_4gram_lm,
    #     beam_threshold=14,
    # )

    # decoder_config_bpe_large_search = DecoderConfig(
    #     lexicon=get_text_lexicon(prefix=prefix_name, loquacious_key="train.small", bpe_size=BPE_SIZE),
    #     returnn_vocab=label_datastream_bpe.vocab,
    #     beam_size=2048,
    #     beam_size_token=32,  # makes it much faster
    #     arpa_lm=arpa_4gram_lm,
    #     beam_threshold=16,
    # )

    # decoder_config_bpe_ls4gram = DecoderConfig(
    #     lexicon=get_text_lexicon(prefix=prefix_name, loquacious_key="train.small", bpe_size=BPE_SIZE),
    #     returnn_vocab=label_datastream_bpe.vocab,
    #     beam_size=1024,
    #     beam_size_token=16,  # makes it much faster
    #     arpa_lm=arpa_4gram_ls_lm,
    #     beam_threshold=14,
    # )

    train_args = copy.deepcopy(global_train_args)
    train_args["net_args"] = {"model_config_dict": asdict(model_config)}

    training_name = prefix_name + "/" + network_module + f".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_smallbatch"
    train_job = training(training_name, train_data_eow_phon, train_args, num_epochs=500, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    base_asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data_eow_phon, get_specific_checkpoint=500
    )

    tune_and_evaluate_helper(
        training_name=training_name + "/first_shot_tuning",
        asr_model=base_asr_model,
        base_decoder_config=default_decoder_config,
        dev_dataset_tuples=short_dev_dataset_tuples,
        test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
        default_returnn=default_returnn,
        lm_scales=[0.5,1.0,1.5,2.0,2.5],
        prior_scales=[0.0, 0.1, 0.2],
    )
    
    tune_and_evaluate_helper(
        training_name=training_name + "/second_shot_tuning",
        asr_model=base_asr_model,
        base_decoder_config=default_decoder_config,
        dev_dataset_tuples=short_dev_dataset_tuples,
        test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
        default_returnn=default_returnn,
        lm_scales=[2.0, 2.25, 2.5, 2.75, 3.0],
        prior_scales=[0.2, 0.3, 0.4],
    )

    tune_and_evaluate_helper(
        training_name=training_name + "/lexicon_no_lm",
        asr_model=base_asr_model,
        base_decoder_config=default_decoder_config,
        dev_dataset_tuples=short_dev_dataset_tuples,
        test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
        default_returnn=default_returnn,
        lm_scales=[0.0],
        prior_scales=[0.0],
    )


    synthetic_data_keys = [
        "train.small_glow_tts.glow_tts_dynamic_speakers_v1_base256_400eps_train-small-shuffled",
        "train.small_glow_tts.glow_tts_dynamic_speakers_v2_base256_400eps_noise_0.50_train-small-shuffled",
        "train.small_glow_tts.glow_tts_dynamic_speakers_v2_base256_400eps_noise_0.60_train-small-shuffled",
        "train.small_glow_tts.glow_tts_dynamic_speakers_v2_base256_400eps_noise_0.70_train-small-shuffled",
        "train.small_glow_tts.glow_tts_dynamic_speakers_v2_base256_400eps_noise_0.80_train-small-shuffled",
        "train.small_glow_tts.glow_tts_dynamic_speakers_v2_base256_400eps_noise_0.90_train-small-shuffled",
        "train.small_glow_tts.glow_tts_dynamic_speakers_v2_base256_400eps_noise_0.70_train-small-static",
        "train.small_nar_tts.fastspeech_like.fastspeech_like_v2_glowttsv2align_400eps_train-small-shuffled",
        "train.small_nar_tts.fastspeech_like.fastspeech_like_v2_glowttsv2align_400eps_train-small-static",
        "train.small_ar_tts.tacotron2_decoding.tacotron2_decoding_v3_glowttsv2align_400eps_train-small-shuffled",
        "train.small_ar_tts.tacotron2_decoding.tacotron2_decoding_v3_glowttsv2align_400eps_train-small-static",
        "train.small_grad_tts.grad_tts_v3_ext_dur_glowttsv2align_400eps_noise_0.50_train-small-shuffled",
        "train.small_grad_tts.grad_tts_v3_ext_dur_glowttsv2align_400eps_noise_0.60_train-small-shuffled",
        "train.small_grad_tts.grad_tts_v3_ext_dur_glowttsv2align_400eps_noise_0.70_train-small-shuffled",
        "train.small_grad_tts.grad_tts_v3_ext_dur_glowttsv2align_400eps_noise_0.80_train-small-shuffled",
        "train.small_grad_tts.grad_tts_v3_ext_dur_glowttsv2align_400eps_noise_0.90_train-small-shuffled",
        "train.small_grad_tts.grad_tts_v3_ext_dur_glowttsv2align_400eps_noise_0.80_train-small-static",
    ]

    for syn_data_key in  synthetic_data_keys:
        bliss, _ = get_synthetic_data(syn_data_key)
        zip = synthetic_bliss_to_ogg_zip(
            prefix=prefix_name + "/synthetic_250/" + syn_data_key,
            bliss=bliss,
            lexicon_loq_key="train.small",
            variant=2,
            custom_lexicon=None,
        )
        train_data_syn_phon = build_eow_phon_training_datasets(
            prefix=prefix_name + "/synthetic_250/" + syn_data_key,
            loquacious_key="train.small",
            settings=train_settings,
            extra_train_ogg_zips=[zip],
            data_repetition_factors=[0, 1],
            variant=2,
        )
        train_args = copy.deepcopy(global_train_args)
        train_args["net_args"] = {"model_config_dict": asdict(model_config)}

        training_name = prefix_name + "/" + network_module + f".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_smallbatch" + f"/synthetic_250/{syn_data_key}"
        train_job = training(training_name, train_data_syn_phon, train_args, num_epochs=500, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        asr_model = prepare_asr_model(
            training_name, train_job, train_args, with_prior=True, datasets=train_data_syn_phon, get_specific_checkpoint=500
        )
        tune_and_evaluate_helper(
            training_name=training_name + "/first_shot_tuning",
            asr_model=asr_model,
            base_decoder_config=default_decoder_config,
            dev_dataset_tuples=short_dev_dataset_tuples,
            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            default_returnn=default_returnn,
            lm_scales=[0.5,1.0,1.5,2.0,2.5, 3.0, 3.5],
            prior_scales=[0.0, 0.1, 0.2],
        )
        
        tune_and_evaluate_helper(
            training_name=training_name + "/second_shot_tuning",
            asr_model=asr_model,
            base_decoder_config=default_decoder_config,
            dev_dataset_tuples=short_dev_dataset_tuples,
            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            default_returnn=default_returnn,
            lm_scales=[2.5, 3.0, 3.5, 4.0, 4.5],
            prior_scales=[0.2, 0.3, 0.4, 0.5],
        )

        tune_and_evaluate_helper(
            training_name=training_name + "/third_shot_tuning",
            asr_model=asr_model,
            base_decoder_config=default_decoder_config,
            dev_dataset_tuples=short_dev_dataset_tuples,
            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            default_returnn=default_returnn,
            lm_scales=[3.5, 4.0, 4.5, 5.0, 5.5],
            prior_scales=[0.0, 0.1, 0.2, 0.3],
        )
    
    
    from i6_experiments.users.rossenbach.datasets.loquacious import get_250_from_2250_segment_list
    loq250_from_2250_segments = get_250_from_2250_segment_list()
    
    synthetic_data_keys = [
        "train.small_glow_tts.glow_tts_dynamic_speakers_v2_base256_400eps_noise_0.70_train-medium-wo-small-shuffled",
        "train.small_grad_tts.grad_tts_v3_ext_dur_glowttsv2align_400eps_noise_0.80_train-medium-wo-small-shuffled",
        "train.small_ar_tts.tacotron2_decoding.tacotron2_decoding_v3_glowttsv2align_400eps_train-medium-wo-small-shuffled",
        "train.small_nar_tts.fastspeech_like.fastspeech_like_v2_glowttsv2align_400eps_train-medium-wo-small-shuffled",
    ]

    for syn_data_key in synthetic_data_keys:
        bliss, _ = get_synthetic_data(syn_data_key)
        zip = synthetic_bliss_to_ogg_zip(
            prefix=prefix_name + "/synthetic/" + syn_data_key,
            bliss=bliss,
            lexicon_loq_key="train.medium",
            variant=2,
            custom_lexicon=None,
        )
        train_data_syn_phon = build_eow_phon_training_datasets(
            prefix=prefix_name + "/synthetic/" + syn_data_key,
            loquacious_key="train.medium",
            settings=train_settings,
            extra_train_ogg_zips=[zip],
            data_repetition_factors=[0, 1],
            variant=2,
        )
        
        train_data_syn_phon.train.datasets["zip_dataset"]["segment_file"] = loq250_from_2250_segments
        train_args = copy.deepcopy(global_train_args)
        train_args["net_args"] = {"model_config_dict": asdict(model_config)}

        training_name = prefix_name + "/" + network_module + f".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_smallbatch" + f"/synthetic_250/{syn_data_key}"
        train_job = training(training_name, train_data_syn_phon, train_args, num_epochs=500, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        asr_model = prepare_asr_model(
            training_name, train_job, train_args, with_prior=True, datasets=train_data_syn_phon,
            get_specific_checkpoint=500
        )

        tune_and_evaluate_helper(
            training_name=training_name + "/third_shot_tuning",
            asr_model=asr_model,
            base_decoder_config=default_decoder_config,
            dev_dataset_tuples=short_dev_dataset_tuples,
            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            default_returnn=default_returnn,
            lm_scales=[3.5, 4.0, 4.5, 5.0, 5.5],
            prior_scales=[0.0, 0.1, 0.2, 0.3],
        )





    # Resume training for combined
    training_name = prefix_name + "/" + network_module + f".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_smallbatch" + "_resume"
    train_args_resume = copy.deepcopy(global_train_args)
    train_args_resume["net_args"] = {"model_config_dict": asdict(model_config)}
    train_args_resume["config"]["import_model_train_epoch1"] = base_asr_model.checkpoint  # only get checkpoint, rest should be identical

    train_job = training(training_name, train_data_eow_phon, train_args_resume, num_epochs=500, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_resume, with_prior=True, datasets=train_data_eow_phon,
        get_specific_checkpoint=500
    )

    tune_and_evaluate_helper(
        training_name=training_name + "/second_shot_tuning",
        asr_model=asr_model,
        base_decoder_config=default_decoder_config,
        dev_dataset_tuples=short_dev_dataset_tuples,
        test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
        default_returnn=default_returnn,
        lm_scales=[2.5, 3.0, 3.5, 4.0, 4.5],
        prior_scales=[0.2, 0.3, 0.4, 0.5],
    )

    tune_and_evaluate_helper(
        training_name=training_name + "/third_shot_tuning",
        asr_model=asr_model,
        base_decoder_config=default_decoder_config,
        dev_dataset_tuples=short_dev_dataset_tuples,
        test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
        default_returnn=default_returnn,
        lm_scales=[1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
        prior_scales=[0.2, 0.3, 0.4, 0.5],
    )

    large_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(variant=2),
        returnn_vocab=label_datastream_eow_phon.vocab,
        beam_size=1024,
        beam_size_token=24,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=16,
    )
    tune_and_evaluate_helper(
        training_name=training_name + "/third_shot_tuning_large_search",
        asr_model=asr_model,
        base_decoder_config=large_decoder_config,
        dev_dataset_tuples=short_dev_dataset_tuples,
        test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
        default_returnn=default_returnn,
        lm_scales=[1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
        prior_scales=[0.2, 0.3, 0.4, 0.5],
    )

    # combined training
    synthetic_data_keys_with_segments = [
        ("train.small_glow_tts.glow_tts_dynamic_speakers_v2_base256_400eps_noise_0.70_train-small-shuffled", None, 1),
        ("train.small_grad_tts.grad_tts_v3_ext_dur_glowttsv2align_400eps_noise_0.80_train-small-shuffled", None, 1),
        ("train.small_ar_tts.tacotron2_decoding.tacotron2_decoding_v3_glowttsv2align_400eps_train-small-shuffled", None, 1),
        ("train.small_nar_tts.fastspeech_like.fastspeech_like_v2_glowttsv2align_400eps_train-small-shuffled", None, 1),
        ("train.small_glow_tts.glow_tts_dynamic_speakers_v2_base256_400eps_noise_0.70_train-medium-wo-small-shuffled", loq250_from_2250_segments, 1),
        ("train.small_grad_tts.grad_tts_v3_ext_dur_glowttsv2align_400eps_noise_0.80_train-medium-wo-small-shuffled", loq250_from_2250_segments, 1),
        ("train.small_ar_tts.tacotron2_decoding.tacotron2_decoding_v3_glowttsv2align_400eps_train-medium-wo-small-shuffled", loq250_from_2250_segments, 1),
        ("train.small_nar_tts.fastspeech_like.fastspeech_like_v2_glowttsv2align_400eps_train-medium-wo-small-shuffled", loq250_from_2250_segments, 1),
        ("train.small_glow_tts.glow_tts_dynamic_speakers_v2_base256_400eps_noise_0.70_train-medium-wo-small-shuffled", None, 9),
        ("train.small_grad_tts.grad_tts_v3_ext_dur_glowttsv2align_400eps_noise_0.80_train-medium-wo-small-shuffled", None, 9),
        ("train.small_ar_tts.tacotron2_decoding.tacotron2_decoding_v3_glowttsv2align_400eps_train-medium-wo-small-shuffled", None, 9),
        ("train.small_nar_tts.fastspeech_like.fastspeech_like_v2_glowttsv2align_400eps_train-medium-wo-small-shuffled", None, 9),
    ]

    for syn_data_key, segments, factor in synthetic_data_keys_with_segments:
        bliss, _ = get_synthetic_data(syn_data_key)
        train_settings_ = copy.deepcopy(train_settings)
        train_settings_.train_partition_epoch = 5 * (factor*2)

        zip, syn_bliss = synthetic_bliss_to_ogg_zip(
            prefix=prefix_name + "/synthetic/" + syn_data_key,
            bliss=bliss,
            lexicon_loq_key="train.medium",
            variant=2,
            custom_lexicon=None,
            return_bliss=True,
        )
        train_data_syn_phon = build_eow_phon_training_datasets(
            prefix=prefix_name + "/synthetic/" + syn_data_key,
            loquacious_key="train.small",
            lexicon_loquacious_key="train.medium",
            settings=train_settings_,
            extra_train_ogg_zips=[zip],
            data_repetition_factors=[factor, 1],
            variant=2,
            explicit_devtrain=True if ("medium-wo-small" in syn_data_key and segments is None) else False,
            extra_train_bliss=[syn_bliss] if ("medium-wo-small" in syn_data_key and segments is None) else None,
        )

        sub_name = "full"
        if segments is not None:
            from ...data.phon import get_eow_bliss
            bliss = get_eow_bliss("train.small", "train.small", variant=2)
            from i6_core.corpus.segments import SegmentCorpusJob
            train_small_segments = SegmentCorpusJob(bliss, num_segments=1).out_single_segment_files[1]
            from i6_core.text.processing import ConcatenateJob
            concat_segments = ConcatenateJob([segments, train_small_segments]).out
            train_data_syn_phon.train.datasets["zip_dataset"]["segment_file"] = concat_segments
            sub_name = "250h-subset"

        train_args_ = copy.deepcopy(global_train_args)
        train_args_["net_args"] = {"model_config_dict": asdict(model_config)}
        train_args_["config"]["import_model_train_epoch1"] = base_asr_model.checkpoint  # only get checkpoint, rest should be identical

        training_name = prefix_name + "/" + network_module + f".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_smallbatch_resume" + f"/synthetic_combined/{syn_data_key}_{sub_name}"
        train_job = training(training_name, train_data_syn_phon, train_args_, num_epochs=500, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        train_job.rqmt["mem"] = 120
        #train_job.hold()
        #train_job.move_to_hpc = True

        asr_model = prepare_asr_model(
            training_name, train_job, train_args_, with_prior=True, datasets=train_data_syn_phon,
            get_specific_checkpoint=500
        )

        tune_and_evaluate_helper(
            training_name=training_name + "/second_shot_tuning",
            asr_model=asr_model,
            base_decoder_config=default_decoder_config,
            dev_dataset_tuples=short_dev_dataset_tuples,
            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            default_returnn=default_returnn,
            lm_scales=[2.5, 3.0, 3.5, 4.0, 4.5],
            prior_scales=[0.2, 0.3, 0.4, 0.5],
        )

        tune_and_evaluate_helper(
            training_name=training_name + "/third_shot_tuning",
            asr_model=asr_model,
            base_decoder_config=default_decoder_config,
            dev_dataset_tuples=short_dev_dataset_tuples,
            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            default_returnn=default_returnn,
            lm_scales=[1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
            prior_scales=[0.2, 0.3, 0.4, 0.5],
        )

    # finetuning test
    synthetic_data_keys_with_segments = [
        ("train.small_grad_tts.grad_tts_v3_ext_dur_glowttsv2align_400eps_noise_0.80_train-medium-wo-small-shuffled", loq250_from_2250_segments, 1),
    ]
    for syn_data_key, segments, factor in synthetic_data_keys_with_segments:
        bliss, _ = get_synthetic_data(syn_data_key)
        train_settings_ = copy.deepcopy(train_settings)
        train_settings_.train_partition_epoch = 5 * (factor*2)

        zip, syn_bliss = synthetic_bliss_to_ogg_zip(
            prefix=prefix_name + "/synthetic/" + syn_data_key,
            bliss=bliss,
            lexicon_loq_key="train.medium",
            variant=2,
            custom_lexicon=None,
            return_bliss=True,
        )
        train_data_syn_phon = build_eow_phon_training_datasets(
            prefix=prefix_name + "/synthetic/" + syn_data_key,
            loquacious_key="train.small",
            lexicon_loquacious_key="train.medium",
            settings=train_settings_,
            extra_train_ogg_zips=[zip],
            data_repetition_factors=[factor, 1],
            variant=2,
            explicit_devtrain=True if ("medium-wo-small" in syn_data_key and segments is None) else False,
            extra_train_bliss=[syn_bliss] if ("medium-wo-small" in syn_data_key and segments is None) else None,
        )

        sub_name = "full"
        if segments is not None:
            from ...data.phon import get_eow_bliss
            bliss = get_eow_bliss("train.small", "train.small", variant=2)
            from i6_core.corpus.segments import SegmentCorpusJob
            train_small_segments = SegmentCorpusJob(bliss, num_segments=1).out_single_segment_files[1]
            from i6_core.text.processing import ConcatenateJob
            concat_segments = ConcatenateJob([segments, train_small_segments]).out
            train_data_syn_phon.train.datasets["zip_dataset"]["segment_file"] = concat_segments
            sub_name = "250h-subset"

        train_args_ = copy.deepcopy(global_train_args)
        train_args_["net_args"] = {"model_config_dict": asdict(model_config)}
        train_args_["config"]["import_model_train_epoch1"] = base_asr_model.checkpoint  # only get checkpoint, rest should be identical
        train_args_["config"]["learning_rates"] = list(np.linspace(7e-6, 7e-6, 10))

        training_name = prefix_name + "/" + network_module + f".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_smallbatch_finetune" + f"/synthetic_combined/{syn_data_key}_{sub_name}"
        train_job = training(training_name, train_data_syn_phon, train_args_, num_epochs=10, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        train_job.rqmt["mem"] = 60

        asr_model = prepare_asr_model(
            training_name, train_job, train_args_, with_prior=True, datasets=train_data_syn_phon,
            get_specific_checkpoint=10
        )

        tune_and_evaluate_helper(
            training_name=training_name + "/third_shot_tuning",
            asr_model=asr_model,
            base_decoder_config=default_decoder_config,
            dev_dataset_tuples=short_dev_dataset_tuples,
            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            default_returnn=default_returnn,
            lm_scales=[1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0],
            prior_scales=[0.2, 0.3, 0.4, 0.5],
        )