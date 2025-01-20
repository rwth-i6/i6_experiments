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


def rnnt_bpe_ls960_1023_base(
        prefix_name: str = "example_setups/librispeech/ctc_rnnt_standalone_2024/ls960_rnnt_bpe_5k"
    ):
    #prefix_name = "example_setups/librispeech/ctc_rnnt_standalone_2024/ls960_rnnt_bpe_5k"

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
    vocab_size_without_blank = label_datastream_bpe5000.vocab_size  # 5048

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
        use_gpu: bool = False,
        decoder_module: str = None
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
            forward_config={},
            asr_model=asr_model,
            decoder_module="rnnt.decoder.experimental_rnnt_decoder" if decoder_module is None else decoder_module,
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples={**dev_dataset_tuples}, #**test_dataset_tuples},
            use_gpu=use_gpu,
            debug=True,
            **default_returnn,
        )

    default_decoder_config_bpe5000 = DecoderConfig(
        beam_size=1, returnn_vocab=label_datastream_bpe5000.vocab  # greedy as default
    )

    default_decoder_config_bpe5000_v2 = DecoderConfig(
        beam_size=1, returnn_vocab=label_datastream_bpe5000.vocab,
        left_size=int(0.2*16e3), keep_states=True, keep_hyps=True, test_version=0.0
    )

    from ...pytorch_networks.rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
        LogMelFeatureExtractionV1Config,
        PredictorConfig,
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
        ctc_output_loss=0.0,
    )
    model_config_v5_sub6_512lstm_start1 = copy.deepcopy(model_config_v5_sub6_512lstm)
    model_config_v5_sub6_512lstm_start1.specauc_start_epoch = 1

    model_config_v5_sub6_512lstm_start1_full_spec = copy.deepcopy(model_config_v5_sub6_512lstm_start1)
    model_config_v5_sub6_512lstm_start1_full_spec.specaug_config = specaug_config_full

    train_config_24gbgpu_amp = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": list(np.linspace(5e-5, 5e-4, 240))
        + list(np.linspace(5e-4, 5e-5, 240))
        + list(np.linspace(5e-5, 1e-7, 20)),
        #############
        "batch_size": 120 * 16000,  # RNN-T has very high memory consumption
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 2,
        "torch_amp_options": {"dtype": "bfloat16"}, # FIXME: use this or not?
    }

    network_module = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9"
    train_args = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config_v5_sub6_512lstm)},
        "debug": False,
    }

    network_module = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native"
    train_args_warprnnt_accum2_fullspec1 = copy.deepcopy(train_args)
    train_args_warprnnt_accum2_fullspec1["network_module"] = network_module
    train_args_warprnnt_accum2_fullspec1["net_args"] = {
        "model_config_dict": asdict(model_config_v5_sub6_512lstm_start1_full_spec)
    }
    train_args_warprnnt_accum2_fullspec1["include_native_ops"] = True
    train_args_warprnnt_accum2_fullspec1["config"]["learning_rates"] = (
        list(np.linspace(5e-5, 5e-4, 120)) + list(np.linspace(5e-4, 5e-5, 120)) + list(np.linspace(5e-5, 1e-7, 10))
    )
    # TODO: How does this work exactly? I'm guessing it somehow uses pretrained CTC Conformer as encoder for above Model
    train_args_warprnnt_accum2_fullspec1["config"]["preload_from_files"] = {
        "encoder": {
            "filename": get_ctc_model(
                "ls960_ctc_bpe_5k.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6.512dim_sub6_24gbgpu_50eps_ckpt500"
            ).checkpoint,
            "init_for_train": True,
            "ignore_missing": True,
        }
    }

    training_name = (
        prefix_name + "/" + network_module + ".512dim_sub6_24gbgpu_25eps_accum2_fullspec1_continue_from_ctc50eps"
    )
    train_job = training(
        training_name, train_data_bpe5000, train_args_warprnnt_accum2_fullspec1, num_epochs=250, **default_returnn
    )
    train_job.rqmt["gpu_mem"] = 24
    train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model = prepare_asr_model(
        training_name,
        train_job,
        train_args_warprnnt_accum2_fullspec1,
        with_prior=False,
        datasets=train_data_bpe5000,
        get_specific_checkpoint=250,
        )

    # ===== Training Naive Streamer Transducer =====
    """network_module_streaming = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_i6_streaming"
    train_args_warprnnt_streaming = copy.deepcopy(train_args_warprnnt_accum2_fullspec1)
    train_args_warprnnt_streaming["network_module"] = network_module_streaming
    train_args_warprnnt_streaming["net_args"]["chunk_size"] = int(1.5 * 16000)  # 1.5s

    training_name_streaming = (
            prefix_name + "/" + network_module_streaming + ".512dim_sub6_24gbgpu_25eps_accum2_fullspec1_streaming_continue_from_ctc50eps"
    )
    train_job_streaming = training(
        training_name_streaming, train_data_bpe5000, train_args_warprnnt_streaming, num_epochs=250, **default_returnn
    )
    train_job_streaming.rqmt["gpu_mem"] = 24
    train_job_streaming.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    asr_model_streaming = prepare_asr_model(
        training_name_streaming,
        train_job_streaming,
        train_args_warprnnt_streaming,
        with_prior=False,
        datasets=train_data_bpe5000,
        get_specific_checkpoint=250,
    )"""

    # ===== Training: Streaming Transducer with chunked attention =====
    network_module_streaming = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_i6_streamingv2"
    train_args_warprnnt_streaming = copy.deepcopy(train_args_warprnnt_accum2_fullspec1)
    train_args_warprnnt_streaming["network_module"] = network_module_streaming
    train_args_warprnnt_streaming["net_args"]["chunk_size"] = int(0.2 * 16e3)
    train_args_warprnnt_streaming["config"]["gradient_clip_norm"] = 1.0

    training_name_streaming = (
            prefix_name + "/" + network_module_streaming + ".512dim_sub6_24gbgpu_25eps_accum2_fullspec1_streaming_continue_from_ctc50eps"
    )
    train_job_streaming = training(
        training_name_streaming, train_data_bpe5000, train_args_warprnnt_streaming, num_epochs=250, **default_returnn
    )
    train_job_streaming.rqmt["gpu_mem"] = 24
    train_job_streaming.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    asr_model_streaming = prepare_asr_model(
        training_name_streaming,
        train_job_streaming,
        train_args_warprnnt_streaming,
        with_prior=False,
        datasets=train_data_bpe5000,
        get_specific_checkpoint=204,
    )
    chunked_future_decoder_config = copy.deepcopy(default_decoder_config_bpe5000_v2)
    chunked_future_decoder_config.left_size = train_args_warprnnt_streaming["net_args"]["chunk_size"]
    evaluate_helper(
        training_name_streaming + "/padded_dep/chunk%.1fs/" % (chunked_future_decoder_config.left_size/16e3),
        asr_model_streaming,
        chunked_future_decoder_config,
        beam_size=1,
        use_gpu=True,
        decoder_module="rnnt.decoder.experimental_rnnt_decoder_v2"
    )

    #
    # experiments
    #
    # define streaming decoder config for all upcoming experiments
    default_streaming_decoder_config: DecoderConfig = copy.deepcopy(default_decoder_config_bpe5000)
    default_streaming_decoder_config.right_size = 0
    default_streaming_decoder_config.keep_states = False
    default_streaming_decoder_config.keep_hyps = False
    default_streaming_decoder_config.pad_value = 0.0

    # **** independent chunks experiment (pad last chunk) ****
    chunked_padded_decoder_config = copy.deepcopy(default_streaming_decoder_config)
    chunked_padded_decoder_config.test_version = 0.6

    left_sizes = np.linspace(1.5, 5, 8)
    for left_size in left_sizes:
        chunked_padded_decoder_config.left_size = int(left_size * 16000)

        evaluate_helper(
            training_name + "/padded_indep/chunk%.1fs" % left_size,
            asr_model,
            chunked_padded_decoder_config,
            beam_size=12,
            use_gpu=True
        )

    # **** chunks with future context and previous hypothesis dependence ****
    chunked_future_decoder_config = copy.deepcopy(default_streaming_decoder_config)
    chunked_future_decoder_config.keep_states = True
    chunked_future_decoder_config.keep_hyps = True
    chunked_future_decoder_config.test_version = 0.7

    left_sizes = np.linspace(1.5, 5.0, 8)
    right_sizes = np.linspace(0.0, 5, 11)
    from itertools import product
    for left_size, right_size in product(left_sizes, right_sizes):
        chunked_future_decoder_config.left_size = int(left_size * 16000)
        chunked_future_decoder_config.right_size = int(right_size * 16000)

        evaluate_helper(
            training_name + "/dep_fut_decs/left%.1fs/right%.1fs" % (left_size, right_size),
            asr_model,
            chunked_future_decoder_config,
            beam_size=5,
            use_gpu=True
        )