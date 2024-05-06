from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel
from ...storage import get_ctc_model


def rnnt_bpe_ls960_1023_base():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_rnnt_bpe_5k"

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

    from ...pytorch_networks.rnnt.decoder.experimental_rnnt_decoder import DecoderConfig

    def evaluate_helper(
        training_name: str,
        asr_model: ASRModel,
        base_decoder_config: DecoderConfig,
        beam_size: int = 1,
        use_gpu=False,
    ):
        """
        Example helper to execute tuning over lm_scales and prior scales.
        With the best values runs test-clean and test-other.

        This is just a reference helper and can (should) be freely changed, copied, modified etc...

        :param training_name: for alias and output names
        :param asr_model: ASR model to use
        :param base_decoder_config: any decoder config dataclass

        """
        decoder_config = copy.deepcopy(base_decoder_config)
        decoder_config.beam_size = beam_size
        search_name = training_name + "/search_bs%i" % beam_size
        search_jobs, wers = search(
            search_name,
            forward_config= {"seed": 2} if use_gpu else {},
            asr_model=asr_model,
            decoder_module="rnnt.decoder.experimental_rnnt_decoder",
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            use_gpu=use_gpu,
            **default_returnn,
        )

    decoder_config_bpe5000 = DecoderConfig(
        beam_size=1,  # greedy as default
        returnn_vocab=label_datastream_bpe5000.vocab
    )

    from ...pytorch_networks.rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
        LogMelFeatureExtractionV1Config,
        PredictorConfig
    )

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
    predictor_config = PredictorConfig(
        symbol_embedding_dim=256,
        emebdding_dropout=0.2,
        num_lstm_layers=1,
        lstm_hidden_dim=512,
        lstm_dropout=0.1,
    )
    model_config_v5_sub6_512lstm = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        predictor_config=predictor_config,
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
        specauc_start_epoch=21,
        joiner_dim=640,
        joiner_activation="relu",
        joiner_dropout=0.1,
        ctc_output_loss=0.0
    )
    model_config_v5_sub6_512lstm_start1 = copy.deepcopy(model_config_v5_sub6_512lstm)
    model_config_v5_sub6_512lstm_start1.specauc_start_epoch = 1

    model_config_v5_sub6_512lstm_start1_full_spec = copy.deepcopy(model_config_v5_sub6_512lstm_start1)
    model_config_v5_sub6_512lstm_start1_full_spec.specaug_config = specaug_config_full


    train_config_24gbgpu_amp = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 240))
        + list(np.linspace(5e-4, 5e-5, 240))
        + list(np.linspace(5e-5, 1e-7, 20)),
        #############
        "batch_size": 100 * 16000, # RNN-T has very high memory consumption
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 3,
        "torch_amp_options": {"dtype": "bfloat16"},
    }

    network_module = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9"
    train_args = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config_v5_sub6_512lstm)},
        "debug": False,
    }

    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_50eps_from_scratch"
    train_job = training(training_name, train_data_bpe5000, train_args, num_epochs=500, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=False, datasets=train_data_bpe5000, get_specific_checkpoint=500
    )
    evaluate_helper(
        training_name,
        asr_model,
        decoder_config_bpe5000,
    )
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=False, datasets=train_data_bpe5000, get_best_averaged_checkpoint=(1, "dev_loss_rnnt")
    )
    evaluate_helper(
        training_name + "/best_1",
        asr_model,
        decoder_config_bpe5000,
    )
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=False, datasets=train_data_bpe5000, get_best_averaged_checkpoint=(4, "dev_loss_rnnt")
    )
    evaluate_helper(
        training_name + "/best_4",
        asr_model,
        decoder_config_bpe5000,
    )

    # Debug warprnnt
    network_module = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_warp"
    train_args_warprnnt = copy.deepcopy(train_args)
    train_args_warprnnt["network_module"] = network_module
    train_args_warprnnt["debug"] = True
    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_50eps_from_scratch"
    train_job = training(training_name, train_data_bpe5000, train_args_warprnnt, num_epochs=500, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=False, datasets=train_data_bpe5000, get_specific_checkpoint=500
    )
    evaluate_helper(
        training_name,
        asr_model,
        decoder_config_bpe5000,
    )

    
    # Debug warprnnt
    network_module = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_warp_gather"
    train_args_warprnnt = copy.deepcopy(train_args)
    train_args_warprnnt["network_module"] = network_module
    train_args_warprnnt["debug"] = True
    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_50eps_from_scratch"
    train_job = training(training_name, train_data_bpe5000, train_args_warprnnt, num_epochs=500, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=False, datasets=train_data_bpe5000, get_specific_checkpoint=500
    )
    evaluate_helper(
        training_name,
        asr_model,
        decoder_config_bpe5000,
    )
    
    # Debug warprnnt + CTC
    model_config_v5_sub6_512lstm_ctc02 = copy.deepcopy(model_config_v5_sub6_512lstm)
    model_config_v5_sub6_512lstm_ctc02.ctc_output_loss = 0.2
    network_module = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_warp_gather"
    train_args_warprnnt = copy.deepcopy(train_args)
    train_args_warprnnt["network_module"] = network_module
    train_args_warprnnt["debug"] = True
    train_args_warprnnt["net_args"] = {"model_config_dict": asdict(model_config_v5_sub6_512lstm_ctc02)}
    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_50eps_from_scratch_ctc0.2"
    train_job = training(training_name, train_data_bpe5000, train_args_warprnnt, num_epochs=500, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=False, datasets=train_data_bpe5000, get_specific_checkpoint=500
    )
    evaluate_helper(
        training_name,
        asr_model,
        decoder_config_bpe5000,
    )

    # Debug warprnnt with CTC init
    network_module = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_warp_gather"
    train_args_warprnnt = copy.deepcopy(train_args)
    train_args_warprnnt["network_module"] = network_module
    train_args_warprnnt["debug"] = True
    train_args_warprnnt["config"]["learning_rates"] = list(np.linspace(5e-5, 5e-4, 120)) + list(
        np.linspace(5e-4, 5e-5, 120)) + list(np.linspace(5e-5, 1e-7, 10))
    train_args_warprnnt["config"]["preload_from_files"] = {
        "encoder": {
            "filename": get_ctc_model(
                "ls960_ctc_bpe_5k.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6.512dim_sub6_24gbgpu_50eps_ckpt500"
            ).checkpoint,
            "init_for_train": True,
            "ignore_missing": True,
        }
    }
    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_25eps_continue_from_ctc50eps"
    train_job = training(training_name, train_data_bpe5000, train_args_warprnnt, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=False, datasets=train_data_bpe5000, get_specific_checkpoint=250
    )
    evaluate_helper(
        training_name,
        asr_model,
        decoder_config_bpe5000,
    )

    train_args_warprnnt_accum2 = copy.deepcopy(train_args_warprnnt)
    train_args_warprnnt_accum2["config"]["batch_size"] = 120 * 16000
    train_args_warprnnt_accum2["config"]["accum_grad_multiple_step"] = 2

    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_25eps_accum2_continue_from_ctc50eps"
    train_job = training(training_name, train_data_bpe5000, train_args_warprnnt_accum2, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=False, datasets=train_data_bpe5000, get_specific_checkpoint=250
    )
    evaluate_helper(
        training_name,
        asr_model,
        decoder_config_bpe5000,
    )


    network_module = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native"
    train_args_warprnnt_accum2_fullspec1 = copy.deepcopy(train_args_warprnnt_accum2)
    train_args_warprnnt_accum2_fullspec1["network_module"] = network_module
    train_args_warprnnt_accum2_fullspec1["net_args"] = {"model_config_dict": asdict(model_config_v5_sub6_512lstm_start1_full_spec)}
    train_args_warprnnt_accum2_fullspec1["include_native_ops"] = True

    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_25eps_accum2_fullspec1_continue_from_ctc50eps"
    train_job = training(training_name, train_data_bpe5000, train_args_warprnnt_accum2_fullspec1, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_warprnnt_accum2_fullspec1, with_prior=False, datasets=train_data_bpe5000, get_specific_checkpoint=250
    )
    evaluate_helper(
        training_name,
        asr_model,
        decoder_config_bpe5000,
    )
    evaluate_helper(
        training_name + "/gpu_decode",
        asr_model,
        decoder_config_bpe5000,
        use_gpu=True,
    )
    evaluate_helper(
        training_name,
        asr_model,
        decoder_config_bpe5000,
        beam_size=10,
    )

    train_args_warprnnt_accum2_fullspec1_LR4 = copy.deepcopy(train_args_warprnnt_accum2_fullspec1)
    train_args_warprnnt_accum2_fullspec1_LR4["config"]["learning_rates"] = list(np.linspace(4e-5, 4e-4, 120)) + list(
        np.linspace(4e-4, 4e-5, 120)) + list(np.linspace(4e-5, 1e-7, 10))
    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_25eps_accum2_fullspec1_LR4_continue_from_ctc50eps"
    train_job = training(training_name, train_data_bpe5000, train_args_warprnnt_accum2_fullspec1_LR4, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_warprnnt_accum2_fullspec1, with_prior=False, datasets=train_data_bpe5000, get_specific_checkpoint=250
    )
    evaluate_helper(
        training_name,
        asr_model,
        decoder_config_bpe5000,
    )
    
    
    # i6 native LONG TRAINING NO REDUCE
    network_module = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native"
    train_args_warprnnt_accum2_fullspec1_long = copy.deepcopy(train_args_warprnnt_accum2)
    train_args_warprnnt_accum2_fullspec1_long["network_module"] = network_module
    train_args_warprnnt_accum2_fullspec1_long["net_args"] = {"model_config_dict": asdict(model_config_v5_sub6_512lstm_start1_full_spec)}
    train_args_warprnnt_accum2_fullspec1_long["config"]["learning_rates"] = list(np.linspace(5e-5, 5e-4, 125)) + list(
        np.linspace(5e-4, 5e-5, 375))
    train_args_warprnnt_accum2_fullspec1_long["include_native_ops"] = True

    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_50eps_accum2_fullspec1_nolrred_continue_from_ctc50eps"
    train_job = training(training_name, train_data_bpe5000, train_args_warprnnt_accum2_fullspec1_long, num_epochs=500, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_warprnnt_accum2_fullspec1_long, with_prior=False, datasets=train_data_bpe5000, get_specific_checkpoint=500
    )
    evaluate_helper(
        training_name,
        asr_model,
        decoder_config_bpe5000,
    )
    evaluate_helper(
        training_name,
        asr_model,
        decoder_config_bpe5000,
        beam_size=10,
    )
    asr_model_best = prepare_asr_model(
        training_name, train_job, train_args_warprnnt_accum2_fullspec1_long, with_prior=False,
        datasets=train_data_bpe5000, get_best_averaged_checkpoint=(4, "dev_loss_rnnt"),
    )
    evaluate_helper(
        training_name + "/best_4",
        asr_model_best,
        decoder_config_bpe5000,
    )
    asr_model_last = prepare_asr_model(
        training_name, train_job, train_args_warprnnt_accum2_fullspec1_long, with_prior=False,
        datasets=train_data_bpe5000, get_last_averaged_checkpoint=4,
    )
    evaluate_helper(
        training_name + "/last_4",
        asr_model_last,
        decoder_config_bpe5000,
    )

    # final LR reduction (broken)
    train_args_warprnnt_accum2_fullspec1_lr_reduce_20 = copy.deepcopy(train_args_warprnnt_accum2_fullspec1_long)
    train_args_warprnnt_accum2_fullspec1_lr_reduce_20["config"]["learning_rates"] = list(np.linspace(5e-4, 1e-7, 20))
    train_args_warprnnt_accum2_fullspec1_lr_reduce_20["config"]["cleanup_old_models"] = None  # no cleanup for testing
    train_args_warprnnt_accum2_fullspec1_lr_reduce_20["config"]["import_model_train_epoch1"] = train_job.out_checkpoints[500]
    train_args_warprnnt_accum2_fullspec1_lr_reduce_20["config"].pop("preload_from_files")
    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_50eps_accum2_fullspec1_nolrred_continue_from_ctc50eps.lr_reduce_20"
    train_job = training(training_name, train_data_bpe5000, train_args_warprnnt_accum2_fullspec1_lr_reduce_20, num_epochs=20,
                         **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_warprnnt_accum2_fullspec1_lr_reduce_20, with_prior=False,
        datasets=train_data_bpe5000, get_specific_checkpoint=20
    )
    evaluate_helper(
        training_name,
        asr_model,
        decoder_config_bpe5000,
    )

    # same with fix
    train_args_warprnnt_accum2_fullspec1_lr_reduce_20_fixed = copy.deepcopy(train_args_warprnnt_accum2_fullspec1_lr_reduce_20)
    train_args_warprnnt_accum2_fullspec1_lr_reduce_20_fixed["config"]["learning_rates"] = list(np.linspace(5e-5, 1e-7, 20))
    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_50eps_accum2_fullspec1_nolrred_continue_from_ctc50eps.lr_reduce_20_fixed"
    train_job = training(training_name, train_data_bpe5000, train_args_warprnnt_accum2_fullspec1_lr_reduce_20_fixed, num_epochs=20,
                         **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_warprnnt_accum2_fullspec1_lr_reduce_20_fixed, with_prior=False,
        datasets=train_data_bpe5000, get_specific_checkpoint=20
    )
    evaluate_helper(
        training_name,
        asr_model,
        decoder_config_bpe5000,
    )
    evaluate_helper(
        training_name,
        asr_model,
        decoder_config_bpe5000,
        beam_size=10,
    )
    asr_model_avg10 = prepare_asr_model(
        training_name, train_job, train_args_warprnnt_accum2_fullspec1_lr_reduce_20_fixed, with_prior=False,
        datasets=train_data_bpe5000, get_last_averaged_checkpoint=10,
    )
    evaluate_helper(
        training_name + "/avg10",
        asr_model_avg10,
        decoder_config_bpe5000,
    )

    # tanh joiner
    train_args_warprnnt_accum2_fullspec1_tanh = copy.deepcopy(train_args_warprnnt_accum2_fullspec1)
    model_config_v5_sub6_512lstm_start1_full_spec_tanh = copy.deepcopy(model_config_v5_sub6_512lstm_start1_full_spec)
    model_config_v5_sub6_512lstm_start1_full_spec_tanh.joiner_activation = "tanh"
    train_args_warprnnt_accum2_fullspec1_tanh["net_args"] = {"model_config_dict": asdict(model_config_v5_sub6_512lstm_start1_full_spec_tanh)}
    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_25eps_accum2_fullspec1_tanh_continue_from_ctc50eps"
    train_job = training(training_name, train_data_bpe5000, train_args_warprnnt_accum2_fullspec1_tanh, num_epochs=250,
                         **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_warprnnt_accum2_fullspec1_tanh, with_prior=False,
        datasets=train_data_bpe5000, get_specific_checkpoint=250,
    )
    evaluate_helper(
        training_name,
        asr_model,
        decoder_config_bpe5000,
    )

    # CTC Extra Loss
    train_args_warprnnt_accum2_fullspec1_ctc03 = copy.deepcopy(train_args_warprnnt_accum2_fullspec1)
    model_config_v5_sub6_512lstm_start1_full_spec_ctc03 = copy.deepcopy(model_config_v5_sub6_512lstm_start1_full_spec)
    model_config_v5_sub6_512lstm_start1_full_spec_ctc03.ctc_output_loss = 0.3
    train_args_warprnnt_accum2_fullspec1_ctc03["net_args"] = {"model_config_dict": asdict(model_config_v5_sub6_512lstm_start1_full_spec_ctc03)}
    training_name = prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_25eps_accum2_fullspec1_ctc03_continue_from_ctc50eps"
    train_job = training(training_name, train_data_bpe5000, train_args_warprnnt_accum2_fullspec1_ctc03, num_epochs=250,
                         **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_warprnnt_accum2_fullspec1_ctc03, with_prior=False,
        datasets=train_data_bpe5000, get_specific_checkpoint=250,
    )
    evaluate_helper(
        training_name,
        asr_model,
        decoder_config_bpe5000,
    )

    # Conv first training
    network_module_conv_first = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native_conv_first"
    train_args_warprnnt_accum2_fullspec1_conv_first = copy.deepcopy(train_args_warprnnt_accum2_fullspec1)
    train_args_warprnnt_accum2_fullspec1_conv_first["network_module"] = network_module_conv_first
    train_args_warprnnt_accum2_fullspec1_conv_first["config"]["preload_from_files"] = {
        "encoder": {
            "filename": get_ctc_model(
                "ls960_ctc_bpe_5k.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_conv_first.512dim_sub6_24gbgpu_50eps_ckpt500"
            ).checkpoint,
            "init_for_train": True,
            "ignore_missing": True,
        }
    }
    training_name = prefix_name + "/" + network_module_conv_first + ".512dim_sub6_24gbgpu_25eps_accum2_fullspec1_continue_from_ctc50eps"
    train_job = training(training_name, train_data_bpe5000, train_args_warprnnt_accum2_fullspec1_conv_first, num_epochs=250,
                         **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name, train_job, train_args_warprnnt_accum2_fullspec1_conv_first, with_prior=False, datasets=train_data_bpe5000,
        get_specific_checkpoint=250
    )
    evaluate_helper(
        training_name,
        asr_model,
        decoder_config_bpe5000,
    )

