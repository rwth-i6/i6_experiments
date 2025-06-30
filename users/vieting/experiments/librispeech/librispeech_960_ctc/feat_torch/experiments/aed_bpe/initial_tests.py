from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List, Optional

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel
from ...storage import get_ctc_model, get_lm_model, NeuralLM


def aed_bpe_ls960_1023_initial_tests():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_aed_bpe_5k"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data_bpe5000 = build_bpe_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        bpe_size=5000,
        settings=train_settings,
        use_postfix=True,  # RNN-T now, use postfix
    )
    label_datastream_bpe5000 = cast(LabelDatastream, train_data_bpe5000.datastreams["labels"])
    vocab_size_without_blank = label_datastream_bpe5000.vocab_size

    dev_dataset_tuples = {}
    for testset in ["dev-clean", "dev-other"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    test_dataset_tuples = {}
    for testset in ["test-clean", "test-other"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.aed.decoder.greedy_prototype import DecoderConfig as GreedyDecoderConfig
    from ...pytorch_networks.aed.decoder.beam_search_single_v1 import DecoderConfig as BeamSearchDecoderConfig, BeamSearchOpts

    def greedy_search_helper(
            training_name: str,
            asr_model: ASRModel,
            decoder_config: GreedyDecoderConfig,
            seed: Optional[int] = None,
            use_gpu: bool = False,
        ):
        # remove prior if exists
        asr_model = copy.deepcopy(asr_model)
        asr_model.prior_file = None

        search_name = training_name + "/search_greedy"
        search_jobs, wers = search(
            search_name,
            forward_config={} if seed is None else {"seed": seed},
            asr_model=asr_model,
            decoder_module="aed.decoder.greedy_prototype",
            decoder_args={"config": asdict(decoder_config)},
            use_gpu=use_gpu,
            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            **default_returnn,
        )

    def beam_search_prototype(
            training_name: str,
            asr_model: ASRModel,
            decoder_config: BeamSearchDecoderConfig,
            seed: Optional[int] = None,
            use_gpu: bool = False,
    ):
        # remove prior if exists
        asr_model = copy.deepcopy(asr_model)
        asr_model.prior_file = None

        search_name = training_name + "/search_bs"
        search_jobs, wers = search(
            search_name,
            forward_config={"max_seqs": 20} if seed is None else {"max_seqs": 20, "seed": seed},
            asr_model=asr_model,
            decoder_module="aed.decoder.beam_search_single_v1",
            decoder_args={"config": asdict(decoder_config)},
            use_gpu=use_gpu,
            debug=True,
            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            **default_returnn,
        )
        
    
    def beam_search_with_lm_prototype(
            training_name: str,
            asr_model: ASRModel,
            decoder_config: BeamSearchDecoderConfig,
            seed: Optional[int] = None,
            use_gpu: bool = False,
            dev_only=True,
    ):
        # remove prior if exists
        asr_model = copy.deepcopy(asr_model)
        asr_model.prior_file = None

        search_name = training_name + "/search_bs"
        search_jobs, wers = search(
            search_name,
            forward_config={"max_seqs": 20} if seed is None else {"max_seqs": 20, "seed": seed},
            asr_model=asr_model,
            decoder_module="aed.decoder.beam_search_single_v1_with_lm",
            decoder_args={"config": asdict(decoder_config)},
            use_gpu=use_gpu,
            debug=True,
            test_dataset_tuples={**dev_dataset_tuples} if dev_only else {**dev_dataset_tuples, **test_dataset_tuples},
            **default_returnn,
        )
    

    from ...pytorch_networks.aed.conformer_zoneout_prototype_0624.aed_prototype_v1_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
        LogMelFeatureExtractionV1Config,
        AttentionLSTMDecoderV1Config,
        AdditiveAttentionConfig
    )

    from ...pytorch_networks.aed.conformer_zoneout_prototype_0624.aed_prototype_v2_cfg import ModelConfig as ModelConfigV2

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
        max_dim_feat=8,  # Jingjing style
        num_repeat_feat=5,
    )
    specaug_config_full = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,  # Old style
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
        pool1_kernel_size=(3, 1),
        pool1_stride=(3, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=512,
        activation=None,
    )
    decoder_attention_small_cfg = AdditiveAttentionConfig(
        attention_dim=256, att_weights_dropout=0.1
    )
    decoder_small_cfg = AttentionLSTMDecoderV1Config(
        encoder_dim=512,
        vocab_size=vocab_size_without_blank,
        target_embed_dim=256,
        target_embed_dropout=0.1,
        lstm_hidden_size=1024,
        zoneout_drop_h=0.05,
        zoneout_drop_c=0.15,
        attention_cfg=decoder_attention_small_cfg,
        output_proj_dim=512,
        output_dropout=0.3,
    )
    model_prototype_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        decoder_config=decoder_small_cfg,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=1,
    )

    decoder_attention_zeineldeen_cfg = AdditiveAttentionConfig(
        attention_dim=1024, att_weights_dropout=0.0
    )
    decoder_zeineldeen_cfg = AttentionLSTMDecoderV1Config(
        encoder_dim=512,
        vocab_size=vocab_size_without_blank,
        target_embed_dim=640,
        target_embed_dropout=0.1,
        lstm_hidden_size=1024,
        zoneout_drop_h=0.05,
        zoneout_drop_c=0.15,
        attention_cfg=decoder_attention_zeineldeen_cfg,
        output_proj_dim=1024,
        output_dropout=0.3,
    )
    model_zeineldeen_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        decoder_config=decoder_zeineldeen_cfg,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=1,
    )
    
    model_zeineldeen_config_v2 = ModelConfigV2(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        decoder_config=decoder_zeineldeen_cfg,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=1,
        ctc_loss_scale=0.3,
    )


    greedy_decoder_config = GreedyDecoderConfig(
        returnn_vocab=label_datastream_bpe5000.vocab,
    )

    train_config_24gbgpu = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": list(np.linspace(5e-5, 5e-4, 120)) + list(
        np.linspace(5e-4, 5e-5, 120)) + list(np.linspace(5e-5, 1e-7, 10)),
        #############
        "batch_size": 200 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        #"accum_grad_multiple_step": 3,
        #"torch_amp_options": {"dtype": "bfloat16"},
    }

    network_module = "aed.conformer_zoneout_prototype_0624.aed_prototype_v1"
    train_args = {
        "config": train_config_24gbgpu,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_prototype_config)},
        "debug": True,
    }

    train_args_pretrained = copy.deepcopy(train_args)
    train_args_pretrained["config"]["preload_from_files"] = {
        "encoder": {
            "filename": get_ctc_model(
                "ls960_ctc_bpe_5k.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6.512dim_sub6_24gbgpu_50eps_ckpt500"
            ).checkpoint,
            "init_for_train": True,
            "ignore_missing": True,
        }
    }
    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_25eps_no_amp_continue_from_ctc50eps"
    train_job = training(training_name, train_data_bpe5000, train_args_pretrained, num_epochs=250, **default_returnn)
    #train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


    # With zeineldeen config and AMP
    train_args_amp_zeineldeen = copy.deepcopy(train_args_pretrained)
    train_args_amp_zeineldeen["config"]["torch_amp_options"] = {"dtype": "bfloat16"}
    train_args_amp_zeineldeen["net_args"] = {"model_config_dict": asdict(model_zeineldeen_config)}
    
    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_25eps_amp_zeineldeen_decoder_continue_from_ctc50eps"
    train_job = training(training_name, train_data_bpe5000, train_args_amp_zeineldeen, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    asr_model = prepare_asr_model(
        training_name, train_job, train_args_amp_zeineldeen, with_prior=False, datasets=train_data_bpe5000,
        get_specific_checkpoint=250
    )
    greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config, seed=3)  # first run wrong, re-run with new seed


    train_args_amp_zeineldeen_radam = copy.deepcopy(train_args_amp_zeineldeen)
    train_args_amp_zeineldeen_radam["config"]["optimizer"] = {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-3, "decoupled_weight_decay": True}
    train_args_amp_zeineldeen_radam["config"]["learning_rates"] = list(np.linspace(1e-4, 3e-4, 10)) + list(
        np.linspace(3e-4, 5e-5, 230)) + list(np.linspace(5e-5, 1e-7, 10))
    train_args_amp_zeineldeen_radam["config"]["gradient_clip_norm"] = 1.0

    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_25eps_amp_zeineldeen_decoder_radamv1_continue_from_ctc50eps"
    train_job = training(training_name, train_data_bpe5000, train_args_amp_zeineldeen_radam, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    asr_model = prepare_asr_model(
        training_name, train_job, train_args_amp_zeineldeen_radam, with_prior=False, datasets=train_data_bpe5000,
        get_specific_checkpoint=250
    )
    greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config, seed=3)  # first run wrong, re-run with new seed


    network_module = "aed.conformer_zoneout_prototype_0624.aed_prototype_v2"
    train_args_amp_zeineldeen_radam_modelv2_ctc03 = copy.deepcopy(train_args_amp_zeineldeen_radam)
    train_args_amp_zeineldeen_radam_modelv2_ctc03["net_args"] = {"model_config_dict": asdict(model_zeineldeen_config_v2)}
    train_args_amp_zeineldeen_radam_modelv2_ctc03["network_module"] = network_module

    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_25eps_amp_zeineldeen_decoder_radamv1_ctc03_continue_from_ctc50eps"
    train_job = training(training_name, train_data_bpe5000, train_args_amp_zeineldeen_radam_modelv2_ctc03, num_epochs=250,
                         **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_amp_zeineldeen_radam_modelv2_ctc03, with_prior=False, datasets=train_data_bpe5000,
        get_specific_checkpoint=250
    )
    greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config, seed=3)  # first run wrong, re-run with new seed


    model_zeineldeen_config_v2_fullspec = copy.deepcopy(model_zeineldeen_config_v2)
    model_zeineldeen_config_v2_fullspec.specaug_config = specaug_config_full
    train_args_amp_zeineldeen_radam_modelv2_ctc03_fullspec = copy.deepcopy(train_args_amp_zeineldeen_radam_modelv2_ctc03)
    train_args_amp_zeineldeen_radam_modelv2_ctc03_fullspec["net_args"] = {"model_config_dict": asdict(model_zeineldeen_config_v2_fullspec)}

    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_25eps_amp_zeineldeen_decoder_radamv1_ctc03_fullspec_continue_from_ctc50eps"
    train_job = training(training_name, train_data_bpe5000, train_args_amp_zeineldeen_radam_modelv2_ctc03_fullspec, num_epochs=250,
                         **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_amp_zeineldeen_radam_modelv2_ctc03_fullspec, with_prior=False, datasets=train_data_bpe5000,
        get_specific_checkpoint=250
    )
    greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)
    
    
    train_args_amp_zeineldeen_radam_modelv2_ctc03_fullspec_const = copy.deepcopy(train_args_amp_zeineldeen_radam_modelv2_ctc03_fullspec)
    train_args_amp_zeineldeen_radam_modelv2_ctc03_fullspec_const["config"]["learning_rates"] = list(np.linspace(2e-4, 5e-4, 10)) + list(
        np.linspace(5e-4, 5e-4, 110)) + list(np.linspace(5e-4, 5e-5, 120)) + list(np.linspace(5e-5, 1e-7, 10))

    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_25eps_amp_zeineldeen_decoder_radamv1_ctc03_fullspec_longconst_continue_from_ctc50eps"
    train_job = training(training_name, train_data_bpe5000, train_args_amp_zeineldeen_radam_modelv2_ctc03_fullspec_const, num_epochs=250,
                         **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_amp_zeineldeen_radam_modelv2_ctc03_fullspec_const, with_prior=False, datasets=train_data_bpe5000,
        get_specific_checkpoint=250
    )
    greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)
    
    train_args_amp_zeineldeen_radam_modelv2_ctc03_fullspec_const_morel2 = copy.deepcopy(train_args_amp_zeineldeen_radam_modelv2_ctc03_fullspec_const)
    train_args_amp_zeineldeen_radam_modelv2_ctc03_fullspec_const_morel2["config"]["optimizer"]["weight_decay"] = 1e-2

    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_25eps_amp_zeineldeen_decoder_radamv1_ctc03_fullspec_longconst_morel2_continue_from_ctc50eps"
    train_job = training(training_name, train_data_bpe5000, train_args_amp_zeineldeen_radam_modelv2_ctc03_fullspec_const_morel2, num_epochs=250,
                         **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_amp_zeineldeen_radam_modelv2_ctc03_fullspec_const_morel2, with_prior=False, datasets=train_data_bpe5000,
        get_specific_checkpoint=250
    )
    greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)


    # new baseline from here
    
    train_config_24gbgpu_amp_v2 = {
        "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
        "learning_rates": list(np.linspace(2e-4, 5e-4, 10)) + list(
        np.linspace(5e-4, 5e-4, 110)) + list(np.linspace(5e-4, 5e-5, 120)) + list(np.linspace(5e-5, 1e-7, 10)),
        #############
        "batch_size": 200 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "gradient_clip_norm": 1.0,
        "torch_amp_options": {"dtype": "bfloat16"},
    }

    network_module = "aed.conformer_zoneout_prototype_0624.aed_prototype_v2"
    train_args_new = {
        "config": train_config_24gbgpu_amp_v2,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_zeineldeen_config_v2_fullspec)},
        "debug": True,
    }
    # Debug warprnnt with CTC init
    train_args_new_pretrained = copy.deepcopy(train_args_new)
    train_args_new_pretrained["config"]["preload_from_files"] = {
        "encoder": {
            "filename": get_ctc_model(
                "ls960_ctc_bpe_5k.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6.512dim_sub6_24gbgpu_50eps_ckpt500"
            ).checkpoint,
            "init_for_train": True,
            "ignore_missing": True,
        }
    }

    # the same as before, just as sanity check
    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_25eps_amp_zeineldeen_decoder_baseline_v2_continue_from_ctc50eps"
    train_job = training(training_name, train_data_bpe5000, train_args_new_pretrained, num_epochs=250,
                         **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_new_pretrained, with_prior=False, datasets=train_data_bpe5000,
        get_specific_checkpoint=250
    )
    greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)
    beam_search_prototype(
        training_name,
        asr_model=asr_model,
        decoder_config=BeamSearchDecoderConfig(
            returnn_vocab=label_datastream_bpe5000.vocab,
            beam_search_opts=BeamSearchOpts(
                beam_size=10,
                length_normalization_exponent=1.0,
                length_reward=0.0,
                bos_label=0,
                eos_label=0,
                num_labels=label_datastream_bpe5000.vocab_size,
            )
        ),
        use_gpu=True,
    )

    # with speed perturbation
    train_args_new_pretrained_speedpert = copy.deepcopy(train_args_new_pretrained)
    train_args_new_pretrained_speedpert["use_speed_perturbation"] = True

    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_25eps_amp_zeineldeen_decoder_baseline_v2_sp_continue_from_ctc50eps"
    train_job = training(training_name, train_data_bpe5000, train_args_new_pretrained_speedpert, num_epochs=250,
                         **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_new_pretrained_speedpert, with_prior=False, datasets=train_data_bpe5000,
        get_specific_checkpoint=250
    )
    greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)

    train_args_new_pretrained_speedpert_50eps = copy.deepcopy(train_args_new_pretrained_speedpert)
    train_args_new_pretrained_speedpert_50eps["config"]["learning_rates"] = list(np.linspace(2e-4, 5e-4, 10)) + list(
        np.linspace(5e-4, 5e-4, 110)) + list(np.linspace(5e-4, 5e-5, 360)) + list(np.linspace(5e-5, 1e-7, 20))

    # 50 epochs training
    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_50eps_amp_zeineldeen_decoder_baseline_v2_sp_continue_from_ctc50eps"
    train_job = training(training_name, train_data_bpe5000, train_args_new_pretrained_speedpert_50eps, num_epochs=500,
                         **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_new_pretrained_speedpert_50eps, with_prior=False, datasets=train_data_bpe5000,
        get_specific_checkpoint=500
    )
    greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)
    from ...pytorch_networks.aed.decoder.beam_search_single_v1_with_lm import DecoderConfig as BeamSearchLmDecoderConfig
    beam_search_prototype(
        training_name,
        asr_model=asr_model,
        decoder_config=BeamSearchDecoderConfig(
            returnn_vocab=label_datastream_bpe5000.vocab,
            beam_search_opts=BeamSearchOpts(
                beam_size=10,
                length_normalization_exponent=1.0,
                length_reward=0.0,
                bos_label=0,
                eos_label=0,
                num_labels=label_datastream_bpe5000.vocab_size,
            ),
        ),
        use_gpu=True,
    )

    neural_lm = get_lm_model("bpe5k_2x2024_kazuki_lstmlm_2ep")
    for beam_size in [10, 24]:
        for lm_scale in [0.0, 0.2, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32, 0.34]:
            beam_search_with_lm_prototype(
                training_name + "/lm_%.2f_bs%i" % (lm_scale, beam_size),
                asr_model=asr_model,
                decoder_config=BeamSearchLmDecoderConfig(
                    returnn_vocab=label_datastream_bpe5000.vocab,
                    beam_search_opts=BeamSearchOpts(
                        beam_size=beam_size,
                        length_normalization_exponent=1.0,
                        length_reward=0.0,
                        bos_label=0,
                        eos_label=0,
                        num_labels=label_datastream_bpe5000.vocab_size,
                    ),
                    lm_module="lm.lstm.kazuki_lstm_zijian_variant_v2",
                    lm_args=neural_lm.net_args,
                    lm_checkpoint=neural_lm.checkpoint,
                    lm_scale=lm_scale,
                ),
                use_gpu=True,
            )
            
    neural_lm = get_lm_model("bpe5k_2x2024_kazuki_lstmlm_3ep")
    for beam_size in [10]:
        for lm_scale in [0.26, 0.28, 0.30, 0.32, 0.34]:
            beam_search_with_lm_prototype(
                training_name + "/lm3ep_%.2f_bs%i" % (lm_scale, beam_size),
                asr_model=asr_model,
                decoder_config=BeamSearchLmDecoderConfig(
                    returnn_vocab=label_datastream_bpe5000.vocab,
                    beam_search_opts=BeamSearchOpts(
                        beam_size=beam_size,
                        length_normalization_exponent=1.0,
                        length_reward=0.0,
                        bos_label=0,
                        eos_label=0,
                        num_labels=label_datastream_bpe5000.vocab_size,
                    ),
                    lm_module="lm.lstm.kazuki_lstm_zijian_variant_v2",
                    lm_args=neural_lm.net_args,
                    lm_checkpoint=neural_lm.checkpoint,
                    lm_scale=lm_scale,
                ),
                use_gpu=True,
            )

    # Length reward not implemented yet
    # for beam_size in [10, 24]:
    #     for lm_scale in [0.24, 0.26, 0.28, 0.30, 0.32, 0.34]:
    #         beam_search_with_lm_prototype(
    #             training_name + "/lm_reward_%.2f_bs%i" % (lm_scale, beam_size),
    #             asr_model=asr_model,
    #             decoder_config=BeamSearchLmDecoderConfig(
    #                 returnn_vocab=label_datastream_bpe5000.vocab,
    #                 beam_search_opts=BeamSearchOpts(
    #                     beam_size=beam_size,
    #                     length_normalization_exponent=0.0,
    #                     length_reward=0.1,
    #                     bos_label=0,
    #                     eos_label=0,
    #                     num_labels=label_datastream_bpe5000.vocab_size,
    #                 ),
    #                 lm_module="lm.lstm.kazuki_lstm_zijian_variant_v2",
    #                 lm_args=neural_lm.net_args,
    #                 lm_checkpoint=neural_lm.checkpoint,
    #                 lm_scale=lm_scale,
    #             ),
    #             use_gpu=True,
    #         )
