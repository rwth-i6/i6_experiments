import dataclasses

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


def rnnt_bpe_ls960_1023_low_bpe_from_scratch():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_rnnt_bpe_low_bpe_from_scratch"

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
            decoder_module="rnnt.decoder.experimental_rnnt_decoder",
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples={**dev_dataset_tuples}, #**test_dataset_tuples},
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

    for BPE_SIZE in [128, ]:
        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data_bpe = build_bpe_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-other-960",
            bpe_size=BPE_SIZE,
            settings=train_settings,
            use_postfix=True,  # RNN-T now, use postfix
        )
        label_datastream_bpe = cast(LabelDatastream, train_data_bpe.datastreams["labels"])
        vocab_size_without_blank = label_datastream_bpe.vocab_size

        decoder_config = DecoderConfig(
            beam_size=1,  # greedy as default
            returnn_vocab=label_datastream_bpe.vocab,
        )

        decoder_config_streaming = DecoderConfig(
            beam_size=1,
            returnn_vocab=label_datastream_bpe.vocab,
            keep_states=True,
            keep_hyps=True,
            test_version=0.0,
        )

        decoder_config_streaming = DecoderConfig(
            beam_size=1,
            returnn_vocab=label_datastream_bpe.vocab,
            keep_states=True,
            keep_hyps=True,
            test_version=0.0,
        )

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
        )

        # Default configs for continued training
        KEEP = [160, 200, 220, 240]
        train_config_11gbgpu = {
            "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
            "learning_rates": list(np.linspace(1e-4, 5e-4, 120)) + list(
                np.linspace(5e-4, 5e-5, 120)) + list(np.linspace(5e-5, 1e-7, 10)),
            #############
            "batch_size": 100 * 16000,  # RNN-T has very high memory consumption
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 2,
            #"torch_amp_options": {"dtype": "bfloat16"},  # FIXME this was commented out
            "gradient_clip_norm": 1.0,
            "cleanup_old_models": {
                "keep_last_n": 4,
                "keep_best_n": 2,
                "keep": KEEP
            }
        }

        from dataclasses import replace
        model_config_v5_sub4_512lstm_conv3 = replace(model_config_v5_sub4_512lstm, conv_kernel_size=3)

        network_module = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native"
        train_args_11gb = {
            "config": train_config_11gbgpu,
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config_v5_sub4_512lstm)},
            "include_native_ops": True,
            "debug": False,
        }

        train_args_11gb_offline = copy.deepcopy(train_args_11gb)
        for model_config, alias in [(model_config_v5_sub4_512lstm, "conv31"), (model_config_v5_sub4_512lstm_conv3, "conv3")]:
            train_args_11gb_offline["net_args"] = {"model_config_dict": asdict(model_config)}

            training_name = (
                prefix_name + "/" + str(BPE_SIZE) + 
                "/" + network_module + ".512dim_sub4_11gbgpu_25eps_from_scratch_radamv1" +
                "/" + alias
            )

            train_job = training(training_name, train_data_bpe, train_args_11gb_offline,
                                num_epochs=250, **default_returnn)
            train_job.rqmt["gpu_mem"] = 11
            train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            decoder_config_buffered = copy.deepcopy(decoder_config_streaming)

            from itertools import product
            for chunk_size, lookahead in product([0.5, 1.0, 2.72], [0.0, 0.1, 0.2, 0.5, 1.0, 5.0]):
                decoder_config_buffered.left_size = int(chunk_size*16e3)
                decoder_config_buffered.right_size = int(lookahead*16e3)
                # prep asr model for checkpoints in KEEP
                for keep in [250]:  # KEEP + [250]:
                    asr_model = prepare_asr_model(
                        training_name, train_job, train_args_11gb_offline, with_prior=False,
                        datasets=train_data_bpe, get_specific_checkpoint=keep
                    )
                    if lookahead in [0.1, 0.2]:
                        beam_sizes = [1, 12, 24, 64]
                    else:
                        beam_sizes = [1, 12]

                    for beam_size in beam_sizes:
                        # decode in buffering mode (offline model with extra frames at the edge(s) of chunks)
                        evaluate_helper(
                            training_name + "/streaming/cs" + str(chunk_size) + "/la" + str(lookahead) + "/keep_%i" % keep,
                            asr_model,
                            decoder_config_buffered,
                            use_gpu=True,
                            beam_size=beam_size
                        )
                        evaluate_helper(
                            training_name + "/offline/keep_%i" % keep,
                            asr_model,
                            decoder_config,
                            use_gpu=True,
                            beam_size=beam_size
                        )

        #
        # --- Streaming Models: Isolated Chunks ---
        #    
        network_module_streaming = "rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_i6_streaming"
        train_args_warprnnt_streaming = copy.deepcopy(train_args_11gb)
        train_args_warprnnt_streaming["network_module"] = network_module_streaming
        train_args_warprnnt_streaming["config"]["gradient_clip_norm"] = 1.0


        # fixed chunk size
        train_args_warprnnt_fixed_chunk_size = copy.deepcopy(train_args_warprnnt_streaming)    
        train_args_warprnnt_fixed_chunk_size["debug"] = False

        for KERNEL_SIZE in [31, 3]:

            model_config = replace(model_config_v5_sub4_512lstm, conv_kernel_size=KERNEL_SIZE)
            train_args_warprnnt_fixed_chunk_size["net_args"] = {"model_config_dict": asdict(model_config)}

            for chunk_size in [0.5, 1.0, 1.1, 15.0] + [2.72]:
                
                if chunk_size in [10.0, 15.0] and KERNEL_SIZE == 3:
                    continue

                train_args_warprnnt_fixed_chunk_size["net_args"]["chunk_size"] = int(chunk_size * 16000)

                training_name_streaming_fc = (
                        prefix_name + "/" + str(BPE_SIZE) +
                        "/" + network_module_streaming +
                        ".512dim_sub4_11gbgpu_25eps_from_scratch_radamv1_fixed_chunks" + 
                        "/" + str(chunk_size) + "/" + "conv%i" % KERNEL_SIZE
                )
                train_job_streaming_fc = training(
                    training_name_streaming_fc, train_data_bpe, train_args_warprnnt_fixed_chunk_size,
                    num_epochs=250, **default_returnn
                )
                train_job_streaming_fc.rqmt["gpu_mem"] = 24 if chunk_size >= 10 else 11
                train_job_streaming_fc.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

                asr_model_streaming_fc = prepare_asr_model(
                    training_name_streaming_fc, train_job_streaming_fc, train_args_warprnnt_fixed_chunk_size, with_prior=False,
                    datasets=train_data_bpe, get_specific_checkpoint=250
                )

                decoder_config_fixed_chunks = copy.deepcopy(decoder_config_streaming)
                decoder_config_fixed_chunks.left_size = int(chunk_size * 16000)
                evaluate_helper(
                    training_name_streaming_fc,
                    asr_model_streaming_fc,
                    decoder_config_fixed_chunks,
                    beam_size=1,
                    use_gpu=True
                )


        """
        # greatest divisor chunk size
        train_args_warprnnt_div_chunks = copy.deepcopy(train_args_warprnnt_streaming)
        #train_args_warprnnt_div_chunks["net_args"]["div_chunk_size"] = True
        #train_args_warprnnt_div_chunks["net_args"]["test_version"] = 0.1
        train_args_warprnnt_div_chunks["net_args"]["chunk_size"] = 20800    # FIXME: just for testing

        training_name_div_chunks = (
                prefix_name + "/" + str(BPE_SIZE) +
                "/" + network_module_streaming +
                ".512dim_sub4_11gbgpu_25eps_from_scratch_radamv1_div_chunks" 
        )
        train_job_div_chunks = training(
            training_name_div_chunks, train_data_bpe, train_args_warprnnt_div_chunks,
            num_epochs=250, **default_returnn
        )
        train_job_div_chunks.rqmt["gpu_mem"] = 11
        train_job_div_chunks.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        asr_model_div_chunks = prepare_asr_model(
            training_name_div_chunks, train_job_div_chunks, train_args_warprnnt_div_chunks, with_prior=False,
            datasets=train_data_bpe, get_specific_checkpoint=250
        )

        decoder_config_div_chunks = copy.deepcopy(decoder_config_streaming)

        chunk_sizes = list(np.linspace(0.5, 5, 10)) + [2.72]
        for chunk_size in chunk_sizes:
            decoder_config_div_chunks.left_size = int(chunk_size * 16e3)

            evaluate_helper(
                training_name_div_chunks + "/chunk%.1fs" % chunk_size,
                asr_model_div_chunks,
                decoder_config_div_chunks,
                beam_size=1,
                use_gpu=True
            )

        # Training Naive Streamer Transducer with dynamic random chunk size [FAILED] 
        train_args_warprnnt_streaming = copy.deepcopy(train_args_warprnnt_streaming)
        # TODO maybe add binary random var to say whether to use full seq_len or dynamic length (different experiment)
        train_args_warprnnt_streaming["net_args"].update({
            "min_chunk_size": 0.15*16000,
            "chunk_ranges_weights": [5, 2, 1, 0, 1, 0, 0, 0, 0],
            "choose_seq_len_p": 0.2,
        })
        
        training_name_streaming_dc = (
                prefix_name + "/" + str(BPE_SIZE) +
                "/" + network_module_streaming +
                ".512dim_sub4_11gbgpu_25eps_from_scratch_radamv1_streaming" +
                "/" + "dynamic_chunks"
        )
        train_job_streaming = training(
            training_name_streaming_dc, train_data_bpe, train_args_warprnnt_streaming,
            num_epochs=250, **default_returnn
        )
        train_job_streaming.rqmt["gpu_mem"] = 11
        train_job_streaming.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        """