from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List, Optional


from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ....data.common import DatasetSettings, build_test_dataset
from ....data.bpe import build_bpe_training_datasets
from ....default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ....lm import get_4gram_binary_lm
from ....pipeline import training, prepare_asr_model, search, ASRModel

from ....pytorch_networks.common import Mode
from ....pytorch_networks.trainers.train_handler import TrainMode

from .... import PACKAGE


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

    train_args_default = {
        "config": train_config,
        "network_module": module,
        "include_native_ops": True,
        "debug": True,
        "use_speed_perturbation": True,
        "net_args": {"model_config_dict": asdict(model_config)}
    }

    return train_args_default


def run_experiments(**kwargs):
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2025/ls960_streamable_rnnt_bpe"
    bpe_size = kwargs["bpe_size"]
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

    from ....pytorch_networks.search.rnnt_streamable_decoder_v1 import DecoderConfig, ExtraConfig

    def evaluate_helper(
        training_name: str,
        asr_model: ASRModel,
        base_decoder_config: DecoderConfig,
        decoder_module: str,
        unhashed_decoder_config: Optional[ExtraConfig] = None,
        beam_size: int = 1,
        use_gpu=False,
        with_align=False,
        out_files=["search_out.py"],
        debug=False
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
            unhashed_decoder_args={"extra_config": asdict(unhashed_decoder_config)} if unhashed_decoder_config else None,
            test_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},  # {**dev_dataset_tuples, **test_dataset_tuples},
            use_gpu=use_gpu,
            **default_returnn,
            debug=debug,
            with_align=with_align,
            out_files=out_files
        )

        return search_jobs

    # rnnt-specific imports
    from ....pytorch_networks.rnnt.base_streamable_rnnt import StreamableRNNTConfig
    from ....pytorch_networks.rnnt.joiners.streamable_joiner_v1 import StreamableJoinerConfig
    from ....pytorch_networks.rnnt.predictors.lstm_predictor_v1 import LSTMPredictorConfig

    # encoder-specific imports
    from ....pytorch_networks.encoders.base_encoder import StreamableEncoderConfig
    from ....pytorch_networks.encoders.components.frontend.streamable_vgg_act import VGG4LayerActFrontendV1Config
    from ....pytorch_networks.encoders.components.feature_extractor.streamable_feature_extractor_v1 import (
        StreamableFeatureExtractorV1Config,
        SpecaugConfig,
        LogMelFeatureExtractionV1Config
    )
    from ....pytorch_networks.encoders.encoder_blocks.v2505.streamable_relpos_conformer_block import StreamableRelPosConformerBlockConfigV1
    from ....pytorch_networks.encoders.components.feedforward.streamable_conformer_feedforward import StreamableConformerPositionwiseFeedForwardConfig
    from ....pytorch_networks.encoders.components.convolution.streamable_conv import StreamableConformerConvolutionV1Config
    from ....pytorch_networks.encoders.components.attention.streamable_mhsa_relpos import ConformerMHSARelPosV1Config
    


    # TODO: move all configs to experiments-loop
    logmel_config = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.025,
        hop_size=0.01,
        f_min=60,
        f_max=7600,
        min_amp=1e-10,
        num_filters=80,
        center=True,
    )
    specaug_config_full = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,  # Old style
        num_repeat_feat=5,
    )
    fe_config = StreamableFeatureExtractorV1Config(
        logmel_cfg=logmel_config,
        specaug_cfg=specaug_config_full,
        specaug_start_epoch=21  # TODO: change according to param_combi
    )
    frontend_config = VGG4LayerActFrontendV1Config(
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

    # datasets w/ labels
    dev_dataset_tuples_withlabels = {}
    for testset in ["dev-clean", "dev-other"]:
        dev_dataset_tuples_withlabels[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
            label_datastream=label_datastream_bpe
        )


    #
    # different encoder param experiments 
    #
    for experiment in experiments_config:
        exp_config = experiments_config[experiment]
        model_params = exp_config["model_params"]
        param_combinations = product_dict(**model_params)

        for param_combi in param_combinations:

            joiner_dim = 640
            conformer_size = 512

            predictor_config = LSTMPredictorConfig(
                symbol_embedding_dim=256,
                emebdding_dropout=0.2,
                num_lstm_layers=1,
                lstm_hidden_dim=512,
                lstm_dropout=0.1,

                label_target_size=vocab_size_without_blank + 1,  # FIXME: why +1?
                output_dim=joiner_dim,
                dropout_broadcast_axes=None,
            )
            # TODO: define config with repetitive params and automatically build like in old Model class 
            encoder_config = StreamableEncoderConfig(
                feature_extractor=fe_config,
                frontend=frontend_config,
                encoder_blocks=StreamableRelPosConformerBlockConfigV1(
                    ff_cfg=StreamableConformerPositionwiseFeedForwardConfig(
                        input_dim=conformer_size,
                        hidden_dim=2048,
                        dropout=0.1,
                        activation="silu",
                        dropout_broadcast_axes=None,
                    ),
                    mhsa_cfg=ConformerMHSARelPosV1Config(
                        input_dim=conformer_size,
                        num_att_heads=8,
                        with_bias=True,
                        att_weights_dropout=0.1,
                        learnable_pos_emb=False,
                        rel_pos_clip=16,
                        with_linear_pos=True,
                        with_pos_bias=True,
                        separate_pos_emb_per_head=True,
                        pos_emb_dropout=0.0,
                        dropout=0.1,
                        dropout_broadcast_axes=None,
                    ),
                    conv_cfg=StreamableConformerConvolutionV1Config(
                        channels=conformer_size,
                        kernel_size=31,
                        dropout=0.1,
                        activation="silu"
                    ),
                    dual_mode=param_combi["dual_mode"],
                ),

                num_layers=12,
                encoder_size=conformer_size,
                out_dim=joiner_dim,
            )
            joiner_config = StreamableJoinerConfig(
                input_dim=joiner_dim,
                output_dim=vocab_size_without_blank + 1,
                activation="relu",
                dropout=0.1,
                dropout_broadcast_axes=None,
                dual_mode=param_combi["dual_mode"],
            )
            model_config = StreamableRNNTConfig(
                encoder=encoder_config,
                predictor=predictor_config,
                joiner=joiner_config,
                label_target_size=vocab_size_without_blank,

                chunk_size=param_combi["chunk_size"] * 16e3,
                lookahead_size=param_combi["lookahead_size"],
                carry_over_size=param_combi["carry_over_size"],
                dual_mode=param_combi["dual_mode"],
                streaming_scale=0.5,

                train_mode=str(param_combi["training_strategy"]),
                ctc_output_loss=0.3,  # TODO: change to 0.7 for ffnn predictor
            )

            decoder_config_streaming = DecoderConfig(
                beam_size=12,
                returnn_vocab=label_datastream_bpe.vocab,
                
                lm_model_args=None,
                lm_checkpoint=None,
                lm_module=None,

                mode=str(Mode.STREAMING),
                chunk_size=int(model_config.chunk_size),
                lookahead_size=int(model_config.lookahead_size * 0.06 * 16e3),
                carry_over_size=model_config.carry_over_size,
                test_version=0.0,

            )
            decoder_config_offline = DecoderConfig(
                beam_size=12,
                returnn_vocab=label_datastream_bpe.vocab,

                lm_model_args=None,
                lm_checkpoint=None,
                lm_module=None,

                mode=str(Mode.OFFLINE),
                test_version=0.0,
            )

            num_epochs = exp_config.get("num_epochs")
            KEEP = exp_config.get("keep")
            train_args = get_train_config(
                model_config, keep=KEEP, 
                module=exp_config["network_module"],
                accum_grads=exp_config["accum_grads"],
                num_epochs=num_epochs
            )

            gpu_mem = exp_config["gpu_mem"]
            train_strat = param_combi["training_strategy"].name.lower()
            training_name = (
                prefix_name + "/" + str(bpe_size) + "/" + 
                train_args["network_module"] + ".512dim_sub6_%dgbgpu_" % gpu_mem + 
                "%deps_radamv1_%s_specaug%d" % (num_epochs//10, train_strat, fe_config.specaug_start_epoch)
            )
            if param_combi["training_strategy"] != TrainMode.OFFLINE:
                assert model_config.carry_over_size is not None and model_config.lookahead_size is not None, "Need to define carry and FAC"
                training_name += (
                    "/" + str(param_combi["chunk_size"]) + "/" +
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
            for keep in KEEP + [num_epochs]:
                asr_model = prepare_asr_model(
                    training_name, train_job, train_args, with_prior=False,
                    datasets=train_data_bpe, get_specific_checkpoint=keep
                )
                evaluate_helper(
                    training_name + "/streaming/keep_%i" % keep,
                    asr_model,
                    decoder_config_streaming,
                    use_gpu=True,
                    beam_size=12,
                    decoder_module="search.rnnt_streamable_decoder_v1"  # TODO
                )
                evaluate_helper(
                    training_name + "/offline/keep_%i" % keep,
                    asr_model,
                    decoder_config_offline,
                    use_gpu=True,
                    beam_size=12,
                    decoder_module="search.rnnt_streamable_decoder_v1",  # TODO
                )


            #
            # experiments on each config
            #

            if experiment == "baseline":
                pass

            if experiment == "streaming":
                pass



def ls960_streamable_rnnt():
    experiment_configs = {
        "baseline": {
            "model_params": {
                "chunk_size": [2.39],
                "lookahead_size": [8],
                "carry_over_size": [2],
                "dual_mode": [False],

                "kernel_size": [31],
                "specauc_start_epoch": [11],
                "training_strategy": [TrainMode.OFFLINE],
            },

            "network_module": "rnnt.models.streamable_rnnt_v1",
            "accum_grads": 1,
            "gpu_mem": 48,
            "num_epochs": 1000,
            "keep": [300, 800, 950, 980]
        },

        "streaming": {
            "model_params": {
                "chunk_size": [2.39],
                "lookahead_size": [8],
                "carry_over_size": [2],
                "dual_mode": [False],

                "kernel_size": [31],
                "specauc_start_epoch": [11],
                "training_strategy": [TrainMode.STREAMING],
            },
            "network_module": "rnnt.models.streamable_rnnt_v1",
            "accum_grads": 1,
            "gpu_mem": 48,
            "num_epochs": 1000,
            "keep": [300, 800, 950, 980]
        },
    }

    run_experiments(experiments_config=experiment_configs, bpe_size=128)
    # run_experiments(experiments_config=experiment_configs, bpe_size=512)
