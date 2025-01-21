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


def unified_lookahead_ls960_1023_low_bpe_from_scratch():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_unified_lookahead_low_bpe_from_scratch"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

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
        decoder_module: str = "rnnt.decoder.experimental_rnnt_decoder_v3"
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
            decoder_module=decoder_module,
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples={**dev_dataset_tuples}, #**test_dataset_tuples},
            use_gpu=use_gpu,
            **default_returnn,
            debug=True  # FIXME
        )

    from ...pytorch_networks.rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        LogMelFeatureExtractionV1Config,
        PredictorConfig
    )
    from ...pytorch_networks.rnnt.streaming_conformer.unified_lookahead_transducer_cfg import ModelConfig

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
    predictor_config = PredictorConfig(
        symbol_embedding_dim=256,
        emebdding_dropout=0.2,
        num_lstm_layers=1,
        lstm_hidden_dim=512,
        lstm_dropout=0.1,
    )
    from itertools import product
    # CHUNK_SIZE = max frames in receptfield of f_t
    for CHUNK_SIZE, BPE_SIZE in product([0.5, 1.0, 2.72], [128, ]):   # [128, 512]
        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        for LOOKAHEAD in set([3, 8, int(CHUNK_SIZE//0.06)]):
            lookahead_size = LOOKAHEAD  # e.g. 3 subsampled frames w/ 1 subsampled frame = 60ms => 0.18s lookahead

            train_data_bpe = build_bpe_training_datasets(
                prefix=prefix_name,
                librispeech_key="train-other-960",
                bpe_size=BPE_SIZE,
                settings=train_settings,
                use_postfix=True,  # RNN-T now, use postfix
            )
            label_datastream_bpe = cast(LabelDatastream, train_data_bpe.datastreams["labels"])
            vocab_size_without_blank = label_datastream_bpe.vocab_size

            model_config_v5_sub4_512lstm = ModelConfig(
                feature_extraction_config=fe_config,
                frontend_config=frontend_config,
                specaug_config=specaug_config_full,
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
                ctc_output_loss=0.3,
                use_vgg=None,
                fastemit_lambda=None,
                chunk_size=CHUNK_SIZE*16e3,
                lookahead_size=lookahead_size,
                online_model_scale=0.5,
            )

            # e.g. CHUNK_SIZE = 3.615s, then EIL = 3.615s * 0.5 ≈ 1808ms (average latency per frame)
            layer_chunk_size = int(CHUNK_SIZE*16e3)
            decoder_config = DecoderConfig(
                beam_size=1,  # greedy as default
                returnn_vocab=label_datastream_bpe.vocab,
                left_size=layer_chunk_size,
                right_size=lookahead_size,
                keep_states=True,
                keep_hyps=True,
                test_version=0.0,
            )
            offline_decoder_config = DecoderConfig(
                beam_size=1,  # greedy as default
                returnn_vocab=label_datastream_bpe.vocab,
            )

            if LOOKAHEAD == 3:
                accum_grads = 2
            elif LOOKAHEAD >= 16:
                accum_grads = 4
            else:
                accum_grads = 3

            # Default configs for continued training
            KEEP = [80, 160, 200, 220, 240]
            train_config_11gbgpu = {
                "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
                "learning_rates": list(np.linspace(1e-4, 5e-4, 120)) + list(
                    np.linspace(5e-4, 5e-5, 120)) + list(np.linspace(5e-5, 1e-7, 10)),
                #############
                "batch_size": 200 * 16000 // accum_grads,  # RNN-T has very high memory consumption
                "max_seq_length": {"audio_features": 35 * 16000},
                "accum_grad_multiple_step": accum_grads,
                "gradient_clip_norm": 1.0,
                "cleanup_old_models": {
                    "keep_last_n": 4,
                    "keep_best_n": 2,
                    "keep": KEEP
                }
            }

            network_module = "rnnt.streaming_conformer.unified_lookahead_transducer"
            train_args_11gb_default = {
                "config": train_config_11gbgpu,
                "network_module": network_module,
                "include_native_ops": True,
                "debug": False,
            }

            from dataclasses import replace
            train_args = copy.deepcopy(train_args_11gb_default)

            for KERNEL_SIZE in [31, 7, 3]:

                model_config = replace(model_config_v5_sub4_512lstm, conv_kernel_size=KERNEL_SIZE)

                #
                # training
                #
                train_args["net_args"] = {"model_config_dict": asdict(model_config)}
                training_name = (
                    prefix_name + "/" + str(BPE_SIZE) +
                    "/" + network_module + ".512dim_sub4_11gbgpu_25eps_from_scratch_radamv1_unified_lookahead" +
                    "/" + str(CHUNK_SIZE) + "/" + "conv%i" % KERNEL_SIZE + "/" + "lah%i" % lookahead_size
                )
                # train_job = training(training_name, train_data_bpe, train_args,
                #                     num_epochs=250, **default_returnn)
                # train_job.rqmt["gpu_mem"] = 24 if LOOKAHEAD == 3 else 11
                # train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

                # #
                # # checkpoint decodings
                # #
                # for keep in KEEP:
                #     asr_model = prepare_asr_model(
                #         training_name, train_job, train_args, with_prior=False,
                #         datasets=train_data_bpe, get_specific_checkpoint=keep
                #     )
                #     evaluate_helper(
                #         training_name + "/keep_%i" % keep,
                #         asr_model,
                #         decoder_config,
                #         use_gpu=True,
                #         beam_size=1
                #     )


                # asr_model_online = prepare_asr_model(
                #     training_name, train_job, train_args, with_prior=False,
                #     datasets=train_data_bpe, get_specific_checkpoint=250
                # )
                # asr_model_offline = prepare_asr_model(
                #     training_name, train_job, train_args, with_prior=False,
                #     datasets=train_data_bpe, get_specific_checkpoint=250
                # )
                # for BEAM_SIZE in [1, 12, 24, 64]:
                #     # online
                #     evaluate_helper(
                #         training_name + "/keep_%i" % 250,
                #         asr_model_online,
                #         decoder_config,
                #         use_gpu=True,
                #         beam_size=BEAM_SIZE
                #     )
                #     # offline
                #     evaluate_helper(
                #         training_name + "/offline" + "/keep_%i" % 250,
                #         asr_model_offline,
                #         offline_decoder_config,
                #         use_gpu=True,
                #         beam_size=BEAM_SIZE,
                #         decoder_module="rnnt.decoder.experimental_rnnt_decoder",
                #     )





    #
    # 1000 sub-epochs
    #
    BPE_SIZE = 128
    CHUNK_SIZE = 2.72
    KERNEL_SIZE = 31
    lookahead_size = 3

    train_data_bpe = build_bpe_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        bpe_size=BPE_SIZE,
        settings=train_settings,
        use_postfix=True,  # RNN-T now, use postfix
    )
    label_datastream_bpe = cast(LabelDatastream, train_data_bpe.datastreams["labels"])
    vocab_size_without_blank = label_datastream_bpe.vocab_size

    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config_full,
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
        ctc_output_loss=0.3,
        use_vgg=None,
        fastemit_lambda=None,
        chunk_size=CHUNK_SIZE*16e3,
        lookahead_size=lookahead_size,
        online_model_scale=0.5,
    )

    # e.g. CHUNK_SIZE = 3.615s, then EIL = 3.615s * 0.5 ≈ 1808ms (average latency per frame)
    layer_chunk_size = int(CHUNK_SIZE*16e3)
    decoder_config = DecoderConfig(
        beam_size=1,  # greedy as default
        returnn_vocab=label_datastream_bpe.vocab,
        left_size=layer_chunk_size,
        right_size=lookahead_size,
        keep_states=True,
        keep_hyps=True,
        test_version=0.1,
    )
    offline_decoder_config = DecoderConfig(
        beam_size=1,  # greedy as default
        returnn_vocab=label_datastream_bpe.vocab,
        test_version=0.1
    )

    accum_grads = 2

    # Default configs for continued training
    KEEP = [100, 250, 500, 750, 900, 950, 1000]
    train_config_11gbgpu = {
        "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
        "learning_rates": list(np.linspace(1e-4, 5e-4, 480)) + list(
            np.linspace(5e-4, 5e-5, 480)) + list(np.linspace(5e-5, 1e-7, 40)),
        #############
        "batch_size": 200 * 16000 // accum_grads,  # RNN-T has very high memory consumption
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": accum_grads,
        "gradient_clip_norm": 1.0,
        "cleanup_old_models": {
            "keep_last_n": 4,
            "keep_best_n": 2,
            "keep": KEEP
        }
    }

    network_module = "rnnt.streaming_conformer.unified_lookahead_transducer"
    train_args_11gb_default = {
        "config": train_config_11gbgpu,
        "network_module": network_module,
        "include_native_ops": True,
        "debug": False,
    }

    from dataclasses import replace
    train_args = copy.deepcopy(train_args_11gb_default)

    #
    # training
    #
    train_args["net_args"] = {"model_config_dict": asdict(model_config)}
    training_name = (
        prefix_name + "/" + str(BPE_SIZE) +
        "/" + network_module + ".512dim_sub4_11gbgpu_25eps_from_scratch_radamv1_unified_lookahead_eps1000" +
        "/" + str(CHUNK_SIZE) + "/" + "conv%i" % KERNEL_SIZE + "/" + "lah%i" % lookahead_size
    )
    train_job = training(training_name, train_data_bpe, train_args,
                         num_epochs=1000, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    #
    # checkpoint decodings
    #
    for keep in KEEP[-1:]:
        # online
        asr_model_online = prepare_asr_model(
            training_name, train_job, train_args, with_prior=False,
            datasets=train_data_bpe, get_specific_checkpoint=keep
        )
        evaluate_helper(
            training_name + "/keep_%i" % keep,
            asr_model_online,
            decoder_config,
            use_gpu=True,
            beam_size=12
        )

        # offline
        asr_model_offline = prepare_asr_model(
            training_name, train_job, train_args, with_prior=False,
            datasets=train_data_bpe, get_specific_checkpoint=keep
        )
        evaluate_helper(
            training_name + "/offline" + "/keep_%i" % keep,
            asr_model_offline,
            offline_decoder_config,
            use_gpu=True,
            beam_size=12,
            decoder_module="rnnt.decoder.experimental_rnnt_decoder",
        )


def product_dict(**kwargs):
    keys = kwargs.keys()

    from itertools import product
    for instance in product(*kwargs.values()):
        yield dict(zip(keys, instance))


def get_train_config(model_config, keep, accum_grads=2,  **kwargs):
    num_epochs = kwargs.get("num_epochs", 250)
    # Default configs for continued training
    train_config_11gbgpu = {
        "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
        "learning_rates": list(np.linspace(1e-4, 5e-4, int(0.48 * num_epochs))) + list(
            np.linspace(5e-4, 5e-5, int(0.48 * num_epochs))) + list(
            np.linspace(5e-5, 1e-7, int(0.04 * num_epochs))),
        #############
        "batch_size": 200 * 16000 // accum_grads,  # RNN-T has very high memory consumption
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": accum_grads,
        "gradient_clip_norm": 1.0,
        "cleanup_old_models": {
            "keep_last_n": 4,
            "keep_best_n": 2,
            "keep": keep
        }
    }

    network_module = "rnnt.streaming_conformer.unified_lookahead_transducer"
    train_args_11gb_default = {
        "config": train_config_11gbgpu,
        "network_module": network_module,
        "include_native_ops": True,
        "debug": False,
        "net_args": {"model_config_dict": asdict(model_config)}
    }

    return train_args_11gb_default


def run_experiments(**kwargs):
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_unified_lookahead_low_bpe_from_scratch"
    bpe_size = kwargs.get("bpe_size", 128)
    experiments_config = kwargs.get("experiments_config")

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

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
        decoder_module: str = "rnnt.decoder.experimental_rnnt_decoder_v3"
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
            forward_config={"seed": 2} if use_gpu else {},
            asr_model=asr_model,
            decoder_module=decoder_module,
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples={**dev_dataset_tuples},  # **test_dataset_tuples},
            use_gpu=use_gpu,
            **default_returnn,
            debug=True  # FIXME
        )

    from ...pytorch_networks.rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        LogMelFeatureExtractionV1Config,
        PredictorConfig
    )
    from ...pytorch_networks.rnnt.streaming_conformer.unified_lookahead_transducer_cfg import ModelConfig

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
    predictor_config = PredictorConfig(
        symbol_embedding_dim=256,
        emebdding_dropout=0.2,
        num_lstm_layers=1,
        lstm_hidden_dim=512,
        lstm_dropout=0.1,
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
                specaug_config=specaug_config_full,
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
                conv_kernel_size=param_combi["kernel_size"],
                final_dropout=0.1,
                specauc_start_epoch=21,
                joiner_dim=640,
                joiner_activation="relu",
                joiner_dropout=0.1,
                ctc_output_loss=0.3,
                use_vgg=None,
                fastemit_lambda=None,
                chunk_size=param_combi["chunk_size"] * 16e3,
                lookahead_size=param_combi["lookahead_size"],
                online_model_scale=0.5,
            )

            decoder_config = DecoderConfig(
                beam_size=1,  # greedy as default
                returnn_vocab=label_datastream_bpe.vocab,
                left_size=int(model_config.chunk_size),
                right_size=model_config.lookahead_size,
                keep_states=True,
                keep_hyps=True,
                test_version=0.0,
            )
            offline_decoder_config = DecoderConfig(
                beam_size=1,  # greedy as default
                returnn_vocab=label_datastream_bpe.vocab,
            )

            KEEP = exp_config.get("keep", [80, 160, 200, 220, 240])
            train_args = get_train_config(model_config, keep=KEEP, accum_grads=exp_config["accum_grads"],
                                          num_epochs=exp_config.get("num_epochs", 250))

            name_modifier = exp_config.get("name_modifier", "")
            training_name = (
                prefix_name + "/" + str(bpe_size) +
                "/" + train_args["network_module"] +
                ".512dim_sub4_11gbgpu_25eps_from_scratch_radamv1_unified_lookahead" + name_modifier +
                "/" + str(param_combi["chunk_size"]) + "/" +
                "conv%i" % model_config.conv_kernel_size + "/" + "lah%i" % model_config.lookahead_size
            )
            train_job = training(training_name, train_data_bpe, train_args,
                                 num_epochs=exp_config.get("num_epochs", 250), **default_returnn)
            train_job.rqmt["gpu_mem"] = exp_config.get("gpu_mem", 11)
            train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            #
            # checkpoint decodings
            #
            for keep in KEEP:
                # online
                asr_model_online = prepare_asr_model(
                    training_name, train_job, train_args, with_prior=False,
                    datasets=train_data_bpe, get_specific_checkpoint=keep
                )
                evaluate_helper(
                    training_name + "/keep_%i" % keep,
                    asr_model_online,
                    decoder_config,
                    use_gpu=True,
                    beam_size=1
                )

                # offline
                asr_model_offline = prepare_asr_model(
                    training_name, train_job, train_args, with_prior=False,
                    datasets=train_data_bpe, get_specific_checkpoint=keep
                )
                evaluate_helper(
                    training_name + "/offline" + "/keep_%i" % keep,
                    asr_model_offline,
                    offline_decoder_config,
                    use_gpu=True,
                    beam_size=1,
                    decoder_module="rnnt.decoder.experimental_rnnt_decoder",
                )


def unified_lookahead_ls960_1023_low_bpe_from_scratch_v2():
    experiment_configs = {
        10: {
            "model_params": {
                "chunk_size": [0.5, 1.0, 2.72],
                "lookahead_size": [3],
                "kernel_size": [3, 7, 31]
            },
            "accum_grads": 2,
            "gpu_mem": 11
        },

        20: {
            "model_params": {
                "chunk_size": [0.5, 1.0, 2.72],
                "lookahead_size": [8],
                "kernel_size": [3, 7, 31]
            },
            "accum_grads": 3,
            "gpu_mem": 11
        },
        21: {
            "model_params": {
                "chunk_size": [1.0, 2.72],
                "lookahead_size": [16],
                "kernel_size": [3, 7, 31]
            },
            "accum_grads": 4,
            "gpu_mem": 11
        },
        22: {
            "model_params": {
                "chunk_size": [2.72],
                "lookahead_size": [45],
                "kernel_size": [3, 7, 31]
            },
            "accum_grads": 4,
            "gpu_mem": 11
        },

        30: {
            "model_params": {
                "chunk_size": [2.72],
                "lookahead_size": [3],
                "kernel_size": [31]
            },
            "accum_grads": 2,
            "gpu_mem": 24,
            "num_epochs": 1000,
            "name_modifier": "_eps1000",
            "keep": [100, 250, 500, 750, 900, 950, 1000]
        },

    }

    run_experiments(experiments_config=experiment_configs, bpe_size=128)
