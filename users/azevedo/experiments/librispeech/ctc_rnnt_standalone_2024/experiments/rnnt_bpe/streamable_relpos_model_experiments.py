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
from ...pipeline import training, prepare_asr_model, search, ASRModel, latency
from ...storage import get_ctc_model, get_ctc_forced_alignment
from ...latency import BPEToWordAlignmentsJob


from ...pytorch_networks.rnnt.auxil.functional import TrainingStrategy, Mode



def product_dict(**kwargs):
    keys = kwargs.keys()

    from itertools import product
    for instance in product(*kwargs.values()):
        yield dict(zip(keys, instance))


def get_train_config(model_config, keep, module, accum_grads=1,  **kwargs):
    num_epochs = kwargs.get("num_epochs")

    epochs_r = num_epochs/1000
    # Default configs for continued training
    train_config = {
        "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
        "learning_rates":list(np.linspace(5e-5, 5e-4, int(240 * epochs_r))) + list(
                np.linspace(5e-4, 5e-5, int(720 * epochs_r))) + list(
                    np.linspace(5e-5, 1e-7, int(40 * epochs_r))),
        #############
        "batch_size": 240 * 16000 // accum_grads,  # RNN-T has very high memory consumption
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": accum_grads,
        "gradient_clip_norm": 1.0,
        "torch_amp_options": {"dtype": "bfloat16"},
        "cleanup_old_models": {
            "keep_last_n": 4,
            "keep_best_n": 4,
            "keep": keep
        }
    }

    network_module = "rnnt.conformer_0325.%s" % module
    train_args_default = {
        "config": train_config,
        "network_module": network_module,
        "include_native_ops": True,
        "debug": False,
        "use_speed_perturbation": True,
        "net_args": {"model_config_dict": asdict(model_config)}
    }

    return train_args_default


def run_experiments(**kwargs):
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_relpos_streaming_0325_low_bpe_from_scratch"
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

    from ...pytorch_networks.rnnt.decoder.streaming_decoder_v1 import DecoderConfig

    def evaluate_helper(
        training_name: str,
        asr_model: ASRModel,
        base_decoder_config: DecoderConfig,
        decoder_module: str,
        beam_size: int = 1,
        use_gpu=False,
        with_align=False,
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
            debug=False,
            with_align=with_align
        )

        return search_jobs

    from ...pytorch_networks.rnnt.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        LogMelFeatureExtractionV1Config,
        PredictorConfig,
        ConformerPosEmbConfig
    )
    from ...pytorch_networks.rnnt.conformer_0325.model_dual_0325_v1_cfg import ModelConfig

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
    posemb_config = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
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
                pos_emb_config=posemb_config,
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
                mhsa_with_bias=True,
                conv_kernel_size=31,
                final_dropout=0.1,
                specauc_start_epoch=11,
                joiner_dim=640,
                joiner_activation="relu",
                joiner_dropout=0.1,
                dropout_broadcast_axes=None,  # No dropout broadcast yet to properly compare
                module_list=["ff", "conv", "mhsa", "ff"],
                module_scales=[0.5, 1.0, 1.0, 0.5],
                aux_ctc_loss_layers=[11],
                aux_ctc_loss_scales=[0.3],
                ctc_output_loss=0.3,

                fastemit_lambda=None,
                chunk_size=param_combi["chunk_size"] * 16e3,
                lookahead_size=param_combi["lookahead_size"],
                online_model_scale=0.5,
                carry_over_size=param_combi["carry_over_size"],
                training_strategy=param_combi["training_strategy"],
                dual_mode=param_combi["dual_mode"]
            )

            decoder_config_streaming = DecoderConfig(
                beam_size=12,
                mode=str(Mode.STREAMING),
                returnn_vocab=label_datastream_bpe.vocab,
                chunk_size=int(model_config.chunk_size),
                lookahead_size=int(model_config.lookahead_size * 0.06 * 16e3),
                carry_over_size=model_config.carry_over_size,
                test_version=0.0,
            )
            decoder_config_offline = DecoderConfig(
                beam_size=12,  # greedy as default
                mode=str(Mode.OFFLINE),
                returnn_vocab=label_datastream_bpe.vocab,
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
            # checkpoint decodings
            #
            # for keep in KEEP + [num_epochs]:
            for keep in KEEP + [num_epochs]:
                asr_model = prepare_asr_model(
                    training_name, train_job, train_args, with_prior=False,
                    datasets=train_data_bpe, get_specific_checkpoint=keep
                )
                evaluate_helper(
                    training_name + "/keep_%i" % keep,
                    asr_model,
                    decoder_config_streaming,
                    use_gpu=True,
                    beam_size=12,
                    decoder_module="rnnt.decoder.streaming_decoder_v1"
                )
                evaluate_helper(
                    training_name + "/offline" + "/keep_%i" % keep,
                    asr_model,
                    decoder_config_offline,
                    use_gpu=True,
                    beam_size=12,
                    decoder_module="rnnt.decoder.streaming_decoder_v1",
                )
            
            #
            # search job + alignments
            #
            if model_config.training_strategy == str(TrainingStrategy.UNIFIED):
                decoder_align_config = copy.deepcopy(decoder_config_streaming)
                decoder_align_config.test_version = 0.2
                search_jobs = evaluate_helper(
                    training_name + "/keepv2_%i" % keep,
                    asr_model,
                    decoder_align_config,
                    use_gpu=True,
                    beam_size=12,
                    decoder_module="rnnt.aligner.experimental_rnnt_aligner_v1",
                    with_align=True
                )
                word_aligns_job = BPEToWordAlignmentsJob(
                    alignment_path=search_jobs["dev-other"].out_files["aligns_out.json"],
                    labels_path=label_datastream_bpe.vocab
                )
                word_aligns_job.add_alias(training_name + "/dev-other" + "/word_aligns_job")
                latency(
                    training_name + "/latency",
                    None,
                    ref_paths={"dev-other": get_ctc_forced_alignment("dev-other")},
                    hyp_paths={"dev-other": word_aligns_job.word_alignments},
                )



def relpos_streaming_ls960_0325_low_bpe_from_scratch():
    experiment_configs = {
        10: {
            "model_params": {
                "chunk_size": [2.4],
                "lookahead_size": [8],
                "kernel_size": [31],
                "specauc_start_epoch": [11],
                "carry_over_size": [2],
                "training_strategy": [str(TrainingStrategy.UNIFIED)],
                "dual_mode": [True],
            },

            "network_module": "model_dual_0325_v1",
            "accum_grads": 1,
            "gpu_mem": 48,
            "num_epochs": 1000,
            "keep": [300, 800, 950, 980]
        },

        20: {
            "model_params": {
                "chunk_size": [2.4],
                "lookahead_size": [8],
                "kernel_size": [31],
                "specauc_start_epoch": [11],
                "carry_over_size": [2],
                "training_strategy": [str(TrainingStrategy.STREAMING)],
                "dual_mode": [False],
            },

            "network_module": "model_dual_0325_v1",
            "accum_grads": 1,
            "gpu_mem": 48,
            "num_epochs": 1000,
            "keep": [300, 800, 950, 980]
        },

        30: {
            "model_params": {
                "chunk_size": [0.6],
                "lookahead_size": [8],
                "kernel_size": [31],
                "specauc_start_epoch": [11],
                "carry_over_size": [4],
                "training_strategy": [str(TrainingStrategy.STREAMING)],
                "dual_mode": [False],
            },

            "network_module": "model_dual_0325_v1",
            "accum_grads": 1,
            "gpu_mem": 48,
            "num_epochs": 1000,
            "keep": [300, 800, 950, 980]
        },

    }

    run_experiments(experiments_config=experiment_configs, bpe_size=128)
