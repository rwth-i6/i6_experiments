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


def rnnt_bpe_ls100_low_vocab0824():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls100_rnnt_bpe"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=3,
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
        out_features=384,
        activation=None,
    )
    frontend_config_kernel_fix = copy.deepcopy(frontend_config)
    frontend_config_kernel_fix.pool1_kernel_size = (3, 1)

    predictor_config = PredictorConfig(
        symbol_embedding_dim=192,
        emebdding_dropout=0.3,
        num_lstm_layers=1,
        lstm_hidden_dim=384,
        lstm_dropout=0.2,
    )

    for BPE_SIZE in [128, 1024]:
        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data_bpe = build_bpe_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-clean-100",
            bpe_size=BPE_SIZE,
            settings=train_settings,
            use_postfix=True,  # RNN-T now, use postfix
        )
        label_datastream_bpe = cast(LabelDatastream, train_data_bpe.datastreams["labels"])
        vocab_size_without_blank = label_datastream_bpe.vocab_size

        decoder_config = DecoderConfig(
            beam_size=1,  # greedy as default
            returnn_vocab=label_datastream_bpe.vocab
        )

        model_config_v5_sub6_384lstm = ModelConfig(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config,
            specaug_config=specaug_config_full,
            predictor_config=predictor_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=384,
            num_layers=12,
            num_heads=4,
            ff_dim=1536,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
            joiner_dim=480,
            joiner_activation="relu",
            joiner_dropout=0.2,
            ctc_output_loss=0.7
        )
        model_config_v5_sub6_384lstm_frontendfix = copy.deepcopy(model_config_v5_sub6_384lstm)
        model_config_v5_sub6_384lstm_frontendfix.frontend_config = frontend_config_kernel_fix
        
        # Default configs for continued training
        KEEP = [160, 200, 240, 280]
        train_config_11gbgpu = {
            "optimizer": {"class": "radam", "epsilon": 1e-16, "weight_decay": 1e-2, "decoupled_weight_decay": True},
            "learning_rates": list(np.linspace(7e-5, 7e-4, 140)) + list(
                np.linspace(7e-4, 7e-5, 140)) + list(np.linspace(7e-5, 1e-7, 20)),
            #############
            "batch_size": 160 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "gradient_clip_norm": 1.0,
            "cleanup_old_models": {
                "keep_last_n": 4,
                "keep_best_n": 2,
                "keep": KEEP
            }
        }
        network_module = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native_conv_first"
        train_args_11gb = {
            "config": train_config_11gbgpu,
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config_v5_sub6_384lstm)},
            "include_native_ops": True,
            "use_speed_perturbation": True,
            "debug": False,
        }


        training_name = prefix_name + "/" + str(
            BPE_SIZE) + "/" + network_module + ".384dim_sub6_11gbgpu_100eps_radam"
        train_job = training(training_name, train_data_bpe, train_args_11gb,
                             num_epochs=300, **default_returnn)
        train_job.rqmt["gpu_mem"] = 11
        train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        for keep in KEEP:
            asr_model = prepare_asr_model(
                training_name, train_job, train_args_11gb, with_prior=False,
                datasets=train_data_bpe, get_specific_checkpoint=keep
            )
            evaluate_helper(
                training_name + "/keep_%i" % keep,
                asr_model,
                decoder_config,
                use_gpu=True
            )
        asr_model = prepare_asr_model(
            training_name, train_job, train_args_11gb, with_prior=False,
            datasets=train_data_bpe, get_specific_checkpoint=300
        )
        evaluate_helper(
            training_name + "/keep_%i" % 300,
            asr_model,
            decoder_config,
            use_gpu=True,
        )
        
        # With kernel fix
        train_args_11gb_frontendfix = copy.deepcopy(train_args_11gb)
        train_args_11gb_frontendfix["net_args"] = {"model_config_dict": asdict(model_config_v5_sub6_384lstm_frontendfix)}

        training_name = prefix_name + "/" + str(
            BPE_SIZE) + "/" + network_module + ".384dim_sub6_11gbgpu_100eps_radam_frontendfix"
        train_job = training(training_name, train_data_bpe, train_args_11gb_frontendfix,
                             num_epochs=300, **default_returnn)
        train_job.rqmt["gpu_mem"] = 11
        train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        for keep in KEEP:
            asr_model = prepare_asr_model(
                training_name, train_job, train_args_11gb_frontendfix, with_prior=False,
                datasets=train_data_bpe, get_specific_checkpoint=keep
            )
            evaluate_helper(
                training_name + "/keep_%i" % keep,
                asr_model,
                decoder_config,
                use_gpu=True
            )
        asr_model = prepare_asr_model(
            training_name, train_job, train_args_11gb_frontendfix, with_prior=False,
            datasets=train_data_bpe, get_specific_checkpoint=300
        )
        evaluate_helper(
            training_name + "/keep_%i" % 300,
            asr_model,
            decoder_config,
            use_gpu=True,
        )
