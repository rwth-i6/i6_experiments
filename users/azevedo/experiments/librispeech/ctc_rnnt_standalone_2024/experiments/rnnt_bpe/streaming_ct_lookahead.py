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


def streaming_lookahead_ls960_1023_low_bpe_from_scratch():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_streaming_lookahead_low_bpe_from_scratch"

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
            decoder_module="rnnt.decoder.experimental_rnnt_decoder_v3",
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
    from ...pytorch_networks.rnnt.streaming_conformer.streaming_conformer_v2_cfg import ModelConfig

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
        lookahead_size = 3  # 3 subsampled frames w/ 1 subsampled frame = 60ms => 0.18s lookahead

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
            fastemit_lambda=None,
            chunk_size=CHUNK_SIZE*16e3,
            lookahead_size=lookahead_size,
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

        # Default configs for continued training
        KEEP = [80, 160, 200, 220, 240]
        train_config_11gbgpu = {
            "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
            "learning_rates": list(np.linspace(1e-4, 5e-4, 120)) + list(
                np.linspace(5e-4, 5e-5, 120)) + list(np.linspace(5e-5, 1e-7, 10)),
            #############
            "batch_size": 100 * 16000,  # RNN-T has very high memory consumption
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 2,
            "gradient_clip_norm": 1.0,
            "cleanup_old_models": {
                "keep_last_n": 4,
                "keep_best_n": 2,
                "keep": KEEP
            }
        }

        network_module = "rnnt.streaming_conformer.streaming_conformer_v2"
        train_args_11gb_default = {
            "config": train_config_11gbgpu,
            "network_module": network_module,
            "include_native_ops": True,
            "debug": False,
        }

        from dataclasses import replace
        train_args = copy.deepcopy(train_args_11gb_default)

        for KERNEL_SIZE in [31, 3]:

            model_config = replace(model_config_v5_sub4_512lstm, conv_kernel_size=KERNEL_SIZE)

            #
            # training
            #
            train_args["net_args"] = {"model_config_dict": asdict(model_config)}
            training_name = (
                prefix_name + "/" + str(BPE_SIZE) +
                "/" + network_module + ".512dim_sub4_11gbgpu_25eps_from_scratch_radamv1_lookahead" +
                "/" + str(CHUNK_SIZE) + "/" + "conv%i" % KERNEL_SIZE
            )
            train_job = training(training_name, train_data_bpe, train_args,
                                 num_epochs=250, **default_returnn)
            train_job.rqmt["gpu_mem"] = 11
            train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            #
            # checkpoint decodings
            #
            for keep in KEEP + [250]:
                asr_model = prepare_asr_model(
                    training_name, train_job, train_args, with_prior=False,
                    datasets=train_data_bpe, get_specific_checkpoint=keep
                )
                evaluate_helper(
                    training_name + "/keep_%i" % keep,
                    asr_model,
                    decoder_config,
                    use_gpu=True,
                    beam_size=1
                )
