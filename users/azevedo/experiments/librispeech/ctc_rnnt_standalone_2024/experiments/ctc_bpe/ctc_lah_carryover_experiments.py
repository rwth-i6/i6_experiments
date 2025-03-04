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
from ...storage import add_ctc_model

from ...pytorch_networks.rnnt.auxil.functional import TrainingStrategy


def product_dict(**kwargs):
    keys = kwargs.keys()

    from itertools import product
    for instance in product(*kwargs.values()):
        yield dict(zip(keys, instance))


def get_train_config(model_config, keep, module, accum_grads=1,  **kwargs):
    num_epochs = kwargs.get("num_epochs")

    epochs_r = num_epochs/1000
    # Default configs for continued training
    train_config_24gbgpu = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
        "learning_rates": list(np.linspace(7e-6, 5e-4, int(480 * epochs_r)))
        + list(np.linspace(5e-4, 5e-5, int(480 * epochs_r)))
        + list(np.linspace(5e-5, 1e-7, int(40 * epochs_r))),
        #############
        "batch_size": 240 * 16000 // accum_grads,  # GPU MEM still very moderate, but larger batch did not help
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": accum_grads,
        "torch_amp_options": {"dtype": "bfloat16"},
        "gradient_clip_norm": 1.0,
        "cleanup_old_models": {
            "keep_last_n": 4,
            "keep_best_n": 4,
            "keep": keep
        }
    }
    # train_config_24gbgpu = {
    #     "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
    #     "learning_rates":list(np.linspace(5e-5, 5e-4, int(240 * epochs_r))) + list(
    #             np.linspace(5e-4, 5e-5, int(720 * epochs_r))) + list(
    #                 np.linspace(5e-5, 1e-7, int(40 * epochs_r))),
    #     #############
    #     "batch_size": 240 * 16000 // accum_grads,  # RNN-T has very high memory consumption
    #     "max_seq_length": {"audio_features": 35 * 16000},
    #     "accum_grad_multiple_step": accum_grads,
    #     "gradient_clip_norm": 1.0,
    #     "torch_amp_options": {"dtype": "bfloat16"},
    #     "cleanup_old_models": {
    #         "keep_last_n": 4,
    #         "keep_best_n": 4,
    #         "keep": keep
    #     }
    # }

    network_module = "ctc.conformer_1124.%s" % module
    train_args_24gb_default = {
        "config": train_config_24gbgpu,
        "network_module": network_module,
        "include_native_ops": True,
        "debug": False,
        "net_args": {"model_config_dict": asdict(model_config)}
    }

    return train_args_24gb_default


def run_experiments(**kwargs):
    prefix_name = "example_setups/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_low_bpe_lah_co"
    bpe_size = kwargs.get("bpe_size", 128)
    experiments_config = kwargs.get("experiments_config")

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data_bpe128 = build_bpe_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        bpe_size=128,
        settings=train_settings,
        use_postfix=False,
    )
    label_datastream_bpe128 = cast(LabelDatastream, train_data_bpe128.datastreams["labels"])
    vocab_size_without_blank = label_datastream_bpe128.vocab_size

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

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }
    
    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig as DecoderConfigOffline
    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v2 import DecoderConfig
    from ...pytorch_networks.ctc.decoder.lah_carryover_decoder import DecoderConfig as DecoderConfigV2
    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_v3 import DecoderConfig as OffGreedyDecoderConfig
    from ...pytorch_networks.ctc.decoder.greedy_lah_carryover_decoder import DecoderConfig as GreedyDecoderConfig

    def tune_and_evaluate_helper(
        training_name: str,
        asr_model: ASRModel,
        base_decoder_config: DecoderConfig,
        lm_scales: List[float],
        prior_scales: List[float],
        decoder_module: str = "ctc.decoder.flashlight_ctc_v1"
    ):
        """
        Example helper to execute tuning over lm_scales and prior scales.
        With the best values runs test-clean and test-other.

        This is just a reference helper and can (should) be freely changed, copied, modified etc...

        :param training_name: for alias and output names
        :param asr_model: ASR model to use
        :param base_decoder_config: any decoder config dataclass
        :param lm_scales: lm scales for tuning
        :param prior_scales: prior scales for tuning, same length as lm scales
        """
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        for lm_weight in lm_scales:
            for prior_scale in prior_scales:
                decoder_config = copy.deepcopy(base_decoder_config)
                decoder_config.lm_weight = lm_weight
                decoder_config.prior_scale = prior_scale
                search_name = training_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
                search_jobs, wers = search(
                    search_name,
                    forward_config={},
                    asr_model=asr_model,
                    decoder_module=decoder_module,
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples=dev_dataset_tuples,
                    **default_returnn,
                )
                tune_parameters.append((lm_weight, prior_scale))
                tune_values_clean.append((wers[search_name + "/dev-clean"]))
                tune_values_other.append((wers[search_name + "/dev-other"]))

        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                parameters=tune_parameters, values=tune_values, mode="minimize"
            )
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers = search(
                training_name,
                forward_config={},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples={key: test_dataset_tuples[key]},
                **default_returnn,
            )

    def greedy_search_helper(
            training_name: str,
            asr_model: ASRModel,
            decoder_config: GreedyDecoderConfig,
            decoder_module: str,
    ):
        # remove prior if exists
        asr_model = copy.deepcopy(asr_model)
        asr_model.prior_file = None

        search_name = training_name + "/search_greedy"
        search_jobs, wers = search(
            search_name,
            forward_config={},
            asr_model=asr_model,
            decoder_module=decoder_module,
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples=dev_dataset_tuples,
            **default_returnn,
        )

    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        LogMelFeatureExtractionV1Config,
    )
    from ...pytorch_networks.ctc.conformer_1124.model_lah_carryover_cfg import ModelConfig

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
        pool1_kernel_size=(2, 1),
        pool1_stride=(3, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=512,
        activation=None,
    )

    train_data_bpe = build_bpe_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        bpe_size=bpe_size,
        settings=train_settings,
        use_postfix=True,  # RNN-T now, use postfix
    )
    label_datastream_bpe = cast(LabelDatastream, train_data_bpe.datastreams["labels"])
    vocab_size_without_blank = label_datastream_bpe.vocab_size

    #
    # different encoder param experiments 
    #
    for experiment in experiments_config:
        exp_config = experiments_config[experiment]
        model_params = exp_config["model_params"]

        param_combinations = product_dict(**model_params)

        for param_combi in param_combinations:
            model_config = ModelConfig(
                feature_extraction_config=fe_config,
                frontend_config=frontend_config,
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
                conv_kernel_size=param_combi["kernel_size"],
                final_dropout=0.1,
                specauc_start_epoch=11,  # BPE does not converge otherwise

                chunk_size=param_combi["chunk_size"] * 16e3,
                lookahead_size=param_combi["lookahead_size"],
                online_model_scale=0.5,
                carry_over_size=param_combi["carry_over_size"],
                training_strategy=param_combi["training_strategy"],
            )

            default_decoder_config_bpe128 = DecoderConfigV2(
                lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=128),
                returnn_vocab=label_datastream_bpe128.vocab,
                beam_size=1024,  # Untuned
                beam_size_token=16,  # makes it much faster (0.3 search RTF -> 0.04 search RTF), but looses 0.1% WER over 128
                arpa_lm=arpa_4gram_lm,
                beam_threshold=14,  # Untuned
                chunk_size=int(model_config.chunk_size),
                lookahead_size=int(model_config.lookahead_size*0.06*16e3),
                carry_over_size=model_config.carry_over_size,
                test_version=0.0,
            )
            
            offline_decoder_config_bpe128 = DecoderConfigOffline(
                lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=128),
                returnn_vocab=label_datastream_bpe128.vocab,
                beam_size=1024,  # Untuned
                beam_size_token=16,  # makes it much faster (0.3 search RTF -> 0.04 search RTF), but looses 0.1% WER over 128
                arpa_lm=arpa_4gram_lm,
                beam_threshold=14,  # Untuned
            )

            num_epochs = exp_config.get("num_epochs")
            KEEP = exp_config.get("keep")
            train_args = get_train_config(model_config, keep=KEEP, 
                                          module=exp_config["network_module"],
                                          accum_grads=exp_config["accum_grads"],
                                          num_epochs=num_epochs)

            gpu_mem = exp_config["gpu_mem"]
            train_strat = model_config.training_strategy.split(".")[-1].lower()

            training_name = (
                prefix_name + "/" + str(bpe_size) + "/" + 
                train_args["network_module"] +
                ".512dim_sub6_%dgbgpu_" % gpu_mem + 
                "%deps_" % (num_epochs//10) +
                "from_scratch_radamv1_%s_lah_co_specaug%d" % (train_strat, model_config.specauc_start_epoch) + "/" +
                str(param_combi["chunk_size"]) + "/" +
                "carry%.1f" % model_config.carry_over_size + "/" + 
                "lah%i" % model_config.lookahead_size
            )
            train_job = training(training_name, train_data_bpe, train_args,
                                 num_epochs=num_epochs, **default_returnn)
            train_job.rqmt["gpu_mem"] = gpu_mem
            train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            #
            # decodings
            #
            asr_model = prepare_asr_model(
                training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe128, get_specific_checkpoint=num_epochs
            )
            tune_and_evaluate_helper(
                training_name + "/online",
                asr_model,
                default_decoder_config_bpe128,
                lm_scales=[1.6, 1.8, 2.0, 2.2, 2.4],
                prior_scales=[0.2, 0.3, 0.4, 0.6, 0.8],
                decoder_module="ctc.decoder.lah_carryover_decoder"
            )

            if experiment == 20:
                offline_decoder_config_bpe128 = DecoderConfigV2(
                    lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=128),
                    returnn_vocab=label_datastream_bpe128.vocab,
                    beam_size=1024,
                    beam_size_token=16,
                    arpa_lm=arpa_4gram_lm,
                    beam_threshold=14,
                )
            tune_and_evaluate_helper(
                training_name + "/offline",
                asr_model,
                offline_decoder_config_bpe128,
                lm_scales=[1.6, 1.8, 2.0, 2.2, 2.4],
                prior_scales=[0, 0.2, 0.3, 0.4, 0.6, 0.8],
                decoder_module="ctc.decoder.lah_carryover_decoder"
            )

            if experiment in [20, 30]: #and (model_config.carry_over_size, model_config.lookahead_size) in [(2, 8)]:
                tune_and_evaluate_helper(
                    training_name + "/nolm" + "/offline",
                    asr_model,
                    offline_decoder_config_bpe128,
                    lm_scales=[0],
                    prior_scales=[0.2, 0.3, 0.4, 0.6, 0.8],
                    decoder_module="ctc.decoder.lah_carryover_decoder"
                )

            if experiment == 30 and model_config.lookahead_size == 8:
                online_decoder_greedy = GreedyDecoderConfig(
                    returnn_vocab=label_datastream_bpe128.vocab,

                    chunk_size=int(model_config.chunk_size),
                    lookahead_size=int(model_config.lookahead_size * 0.06 * 16e3),
                    carry_over_size=model_config.carry_over_size,
                    test_version=0.2,
                )
                greedy_search_helper(
                    training_name + "/online",
                    asr_model,
                    online_decoder_greedy,
                    decoder_module="ctc.decoder.greedy_lah_carryover_decoder"
                )
                offline_decoder_greedy = GreedyDecoderConfig(
                    returnn_vocab=label_datastream_bpe128.vocab,

                    chunk_size=None,
                    lookahead_size=None,
                    carry_over_size=None,
                    test_version=0.0,
                )
                greedy_search_helper(
                    training_name + "/offline",
                    asr_model,
                    offline_decoder_greedy,
                    decoder_module="ctc.decoder.greedy_lah_carryover_decoder"
                )


def ctc_lah_carryover_v2_ls960_1023_low_bpe_from_scratch():
    experiment_configs = {
        10: {
            "model_params": {
                "chunk_size": [2.4],
                "lookahead_size": [8],
                "kernel_size": [31],
                "specauc_start_epoch": [11],
                "carry_over_size": [2],
                "training_strategy": [str(TrainingStrategy.UNIFIED)]
            },

            "network_module": "model_streaming_lah_carryover",
            "accum_grads": 1,
            "gpu_mem": 24,
            "num_epochs": 1000,
            "keep": [300, 400, 500, 600, 700, 800, 900, 950, 980]
        },

        20: {
            "model_params": {
                "chunk_size": [0.6],
                "lookahead_size": [0, 8, 16],
                "kernel_size": [31],
                "specauc_start_epoch": [11],
                "carry_over_size": [0, 2, 4, 8],
                "training_strategy": [str(TrainingStrategy.STREAMING)]
            },

            "network_module": "model_streaming_lah_carryover",
            "accum_grads": 1,
            "gpu_mem": 24,
            "num_epochs": 1000,
            "keep": [300, 500, 800, 950]
        },

        25: {
            "model_params": {
                "chunk_size": [0.6],
                "lookahead_size": [8],
                "kernel_size": [31],
                "specauc_start_epoch": [11],
                "carry_over_size": [4],
                "training_strategy": [str(TrainingStrategy.STREAMING)]
            },

            "network_module": "model_streaming_lah_carryover",
            "accum_grads": 1,
            "gpu_mem": 48,
            "num_epochs": 1500,
            "keep": [300, 800, 1200]
        },

        30: {
            "model_params": {
                "chunk_size": [2.4],
                "lookahead_size": [0, 8, 16],
                "kernel_size": [31],
                "specauc_start_epoch": [11],
                "carry_over_size": [0, 0.5, 1, 2],
                "training_strategy": [str(TrainingStrategy.STREAMING)]
            },

            "network_module": "model_streaming_lah_carryover",
            "accum_grads": 1,
            "gpu_mem": 24,
            "num_epochs": 1000,
            "keep": [300, 500, 800, 950]
        },

        35: {
            "model_params": {
                "chunk_size": [2.4],
                "lookahead_size": [8],
                "kernel_size": [31],
                "specauc_start_epoch": [11],
                "carry_over_size": [2],
                "training_strategy": [str(TrainingStrategy.STREAMING)]
            },

            "network_module": "model_streaming_lah_carryover",
            "accum_grads": 1,
            "gpu_mem": 48,
            "num_epochs": 1500,
            "keep": [300, 800, 1200]
        },

        40: {
            "model_params": {
                "chunk_size": [2.4],
                "lookahead_size": [8],
                "kernel_size": [31],
                "specauc_start_epoch": [11],
                "carry_over_size": [2],
                "training_strategy": [str(TrainingStrategy.SWITCHING)]
            },

            "network_module": "model_streaming_lah_carryover",
            "accum_grads": 1,
            "gpu_mem": 24,
            "num_epochs": 1000,
            "keep": [300, 400, 500, 600, 700, 800, 900, 950, 980]
        },

    }

    run_experiments(experiments_config=experiment_configs, bpe_size=128)
