import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ....data.common import DatasetSettings, build_test_dataset
from ....data.bpe import build_bpe_training_datasets, get_text_lexicon
from ....default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ....lm import get_4gram_binary_lm
from ....pipeline import training, prepare_asr_model, search, force_align, latency, ASRModel
from ....storage import add_ctc_model, add_ctc_forced_alignment, get_ctc_forced_alignment

from ....pytorch_networks.common import Mode
from ....pytorch_networks.trainers.train_handler import TrainMode

from ....report import tune_and_evalue_report

from .... import PACKAGE


def product_dict(**kwargs):
    keys = kwargs.keys()

    from itertools import product
    for instance in product(*kwargs.values()):
        yield dict(zip(keys, instance))


def get_train_config(model_config, keep, module, accum_grads=1, **kwargs):
    num_epochs = kwargs.get("num_epochs")

    epochs_r = num_epochs / 1000
    # Default configs for continued training
    train_config = {
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

    train_args_default = {
        "config": train_config,
        "network_module": module,
        "include_native_ops": True,
        "debug": True,
        "net_args": {"model_config_dict": asdict(model_config)}
    }

    return train_args_default


def run_experiments(**kwargs):
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2025/ls960_streamable_ctc_bpe_refactored"
    bpe_size = kwargs["bpe_size"]
    experiments_config = kwargs.get("experiments_config")

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
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

    from ....storage import get_lm_model, NeuralLM
    lstm_2x1024: NeuralLM = get_lm_model("bpe%i_2x2024_kazuki_lstmlm_3ep" % bpe_size)

    from ....pytorch_networks.search.decoder_module import DecoderConfig, ExtraConfig
    # from ....pytorch_networks.ctc.search.beam_search import CTCSearchConfig
    from ....pytorch_networks.ctc.search import CTCBeamSearchConfig as CTCSearchConfig
    from ....pytorch_networks.ctc.search import CTCGreedySearchConfig
    from ....pytorch_networks.ctc.search import CTCBeamSearchLMConfig
    from ....pytorch_networks.ctc.search.beam_search_bpe_ctc_v4 import DecoderConfig as NickDecoderConfig
    from ....pytorch_networks.ctc.search.beam_search_bpe_ctc_v4 import DecoderExtraConfig


    def tune_and_evaluate_helper(
            training_name: str,
            asr_model: ASRModel,
            base_decoder_config: DecoderConfig,
            lm_scales: List[float],
            prior_scales: List[float],
            decoder_module: str = "search.decoder_module",
            ilm_scale: int = 0,
            debug: bool = False,
            use_gpu: bool = False,
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
        asr_model = copy.deepcopy(asr_model)
        for lm_scale in lm_scales:
            for prior_scale in prior_scales:
                decoder_config_lm = copy.deepcopy(base_decoder_config)
                search_config: CTCSearchConfig = decoder_config_lm.search_config
                search_config.lm_scale = lm_scale
                search_config.zero_ilm_scale = ilm_scale
                search_config.prior_scale = prior_scale
                asr_model.prior_file = None
                search_name = training_name + "/lm%.1f_ilm%.2f_prior%.2f" % (lm_scale, ilm_scale, prior_scale)
                search_jobs, wers = search(
                    search_name,
                    debug=debug,
                    forward_config={},
                    asr_model=asr_model,
                    decoder_module=decoder_module,
                    decoder_args={"config": asdict(decoder_config_lm)},
                    test_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},  # dev_dataset_tuples,
                    use_gpu=use_gpu,
                    **default_returnn,
                )
                tune_parameters.append((lm_scale, prior_scale))
                tune_values_other.append((wers[search_name + "/dev-other"]))
                

    def tune_and_evaluate_helper_new(
            training_name, dev_dataset_tuples, test_dataset_tuples, 
            asr_model, base_decoder_config, lm_scales, prior_scales, decoder_module,
            unhashed_decoder_config = None, debug=False, use_gpu=False, extra_forward_config=None
    ):
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        report_values = {}
        for lm_weight in lm_scales:
            for prior_scale in prior_scales:
                decoder_config = copy.deepcopy(base_decoder_config)
                if hasattr(decoder_config, "lm_scale"):
                    decoder_config.lm_scale = lm_weight
                else:
                    decoder_config.lm_weight = lm_weight
                decoder_config.prior_scale = prior_scale
                search_name = training_name + "/search_lm%.2f_prior%.2f" % (lm_weight, prior_scale)
                search_jobs, wers = search(
                    search_name,
                    forward_config=extra_forward_config if extra_forward_config else {},
                    asr_model=asr_model,
                    decoder_module=decoder_module,
                    decoder_args={"config": asdict(decoder_config)},
                    unhashed_decoder_args={"extra_config": asdict(unhashed_decoder_config)} if unhashed_decoder_config else None,
                    test_dataset_tuples=dev_dataset_tuples,
                    debug=debug,
                    use_gpu=use_gpu,
                    **default_returnn
                )
                tune_parameters.append((lm_weight, prior_scale))
                tune_values_clean.append((wers[search_name + "/dev-clean"]))
                tune_values_other.append((wers[search_name + "/dev-other"]))

        # for key, tune_values in [("test-other", tune_values_other)]:
        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(parameters=tune_parameters, values=tune_values, mode="minimize")
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            decoder_config = copy.deepcopy(base_decoder_config)
            if hasattr(decoder_config, "lm_scale"):
                decoder_config.lm_scale = pick_optimal_params_job.out_optimal_parameters[0]
            else:
                decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers = search(
                training_name,
                forward_config=extra_forward_config if extra_forward_config else {},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                unhashed_decoder_args={
                    "extra_config": asdict(unhashed_decoder_config)} if unhashed_decoder_config else None,
                test_dataset_tuples={key: test_dataset_tuples[key]},
                use_gpu=use_gpu,
                **default_returnn
            )
            report_values[key] = wers[training_name + "/" + key]

        tune_and_evalue_report(
            training_name=training_name,
            tune_parameters=tune_parameters,
            tuning_names=["LM", "Prior"],
            tune_values_clean=tune_values_clean,
            tune_values_other=tune_values_other,
            report_values=report_values
        )

    def greedy_search_helper(
        training_name: str,
        asr_model: ASRModel,
        decoder_config: DecoderConfig,
        decoder_module: str = "search.decoder_module",
        debug: bool = False,
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
            test_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},  # dev_dataset_tuples,
            **default_returnn,
            debug=debug,
        )

    # ctc-specific imports
    from ....pytorch_networks.ctc._base_streamable_ctc import StreamableCTCConfig

    # encoder-specific imports
    from ....pytorch_networks.encoders._base_encoder import StreamableEncoderConfig
    from ....pytorch_networks.encoders.components.frontend.streamable_vgg_act import VGG4LayerActFrontendV1Config
    from ....pytorch_networks.encoders.components.feature_extractor.streamable_feature_extractor_v1 import (
        StreamableFeatureExtractorV1Config,
        SpecaugConfig,
        LogMelFeatureExtractionV1Config
    )
    from ....pytorch_networks.encoders.encoder_blocks.v2505.streamable_relpos_conformer_block import \
        StreamableRelPosConformerBlockConfigV1
    from ....pytorch_networks.encoders.components.feedforward.streamable_conformer_feedforward import \
        StreamableConformerPositionwiseFeedForwardConfig
    from ....pytorch_networks.encoders.components.convolution.streamable_conv import \
        StreamableConformerConvolutionV1Config
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
        max_dim_feat=8,  # Jingjing style
        num_repeat_feat=5,
    )
    fe_config = StreamableFeatureExtractorV1Config(
        logmel_cfg=logmel_config,
        specaug_cfg=specaug_config_full,
        specaug_start_epoch=11
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

    conformer_size = 512

    #
    # different encoder param experiments
    #
    for experiment in experiments_config:
        exp_config = experiments_config[experiment]
        model_params = exp_config["model_params"]
        param_combinations = product_dict(**model_params)

        for param_combi in param_combinations:

            network_module = exp_config["network_module"]
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
            )

            model_config = StreamableCTCConfig(
                encoder=encoder_config,
                label_target_size=vocab_size_without_blank,
                final_dropout=0.1,

                # streaming params
                chunk_size=param_combi["chunk_size"] * 16e3,
                lookahead_size=param_combi["lookahead_size"],
                carry_over_size=param_combi["carry_over_size"],
                dual_mode=param_combi["dual_mode"],
                streaming_scale=0.5,

                train_mode=str(param_combi["training_strategy"]),
            )

            num_epochs = exp_config.get("num_epochs")
            KEEP = exp_config.get("keep")
            train_args = get_train_config(
                model_config, keep=KEEP,
                module=network_module,
                accum_grads=exp_config["accum_grads"],
                num_epochs=num_epochs
            )

            gpu_mem = exp_config["gpu_mem"]
            train_strat = param_combi["training_strategy"].name.lower()
            training_name = (
                    prefix_name + "/" + str(bpe_size) + "/" +
                    network_module + ".512dim_sub6_%dgbgpu_" % gpu_mem +
                    "%deps_adamw_%s_specaug%d" % (num_epochs // 10, train_strat, fe_config.specaug_start_epoch)
            )
            if param_combi["training_strategy"] != TrainMode.OFFLINE:
                assert model_config.carry_over_size is not None and model_config.lookahead_size is not None, (
                    "Need to define carry and FAC if not training in offline mode"
                )
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

            # greedy
            greedy_search_config = CTCGreedySearchConfig(
                lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=bpe_size),
            )
            greedy_streaming_config = DecoderConfig(
                returnn_vocab=label_datastream_bpe.vocab,
                beam_size=None,

                search_config=greedy_search_config,

                mode=Mode.STREAMING.name,
                chunk_size=int(model_config.chunk_size),
                lookahead_size=int(model_config.lookahead_size * 0.06 * 16e3),
                carry_over_size=model_config.carry_over_size,
                test_version=0.0,

            )

            greedy_offline_config = DecoderConfig(
                returnn_vocab=label_datastream_bpe.vocab,
                beam_size=None,

                search_config=greedy_search_config,

                mode=Mode.OFFLINE.name,
                test_version=0.0,
            )

            prior_mem_rqmt = 16 if model_config.chunk_size > 0.6 * 16e3 else 24
            asr_model = prepare_asr_model(
                training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe,
                get_specific_checkpoint=num_epochs,
                mem_rqmt=prior_mem_rqmt
            )
            greedy_search_helper(
                training_name + "/streaming/keep_%i" % num_epochs,
                asr_model,
                greedy_streaming_config,
                debug=True,
            )
            greedy_search_helper(
                training_name + "/offline/keep_%i" % num_epochs,
                asr_model,
                greedy_offline_config,
                debug=True,
            )


            # beam-search w/ arpa 4gram lm
            asr_model = prepare_asr_model(
                training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe,
                get_specific_checkpoint=num_epochs,
                mem_rqmt=prior_mem_rqmt
            )

            search_config = CTCSearchConfig(
                lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=bpe_size),
                beam_size_token=16,
                beam_threshold=14,  # Untuned,
                lm_package=arpa_4gram_lm,
                prior_file=asr_model.prior_file,
            )
            decoder_config_streaming = DecoderConfig(
                beam_size=1024,
                returnn_vocab=label_datastream_bpe.vocab,

                search_config=search_config,

                mode=Mode.STREAMING.name,
                chunk_size=int(model_config.chunk_size),
                lookahead_size=int(model_config.lookahead_size * 0.06 * 16e3),
                carry_over_size=model_config.carry_over_size,
                test_version=0.0,
            )
            decoder_config_offline = DecoderConfig(
                beam_size=1024,
                returnn_vocab=label_datastream_bpe.vocab,
                search_config=search_config,

                mode=Mode.OFFLINE.name,
                test_version=0.0,
            )

            tune_and_evaluate_helper(
                training_name + "/streaming/4gram_lm",
                asr_model,
                decoder_config_streaming,
                lm_scales=[0, 1.4, 1.5, 1.6, 2.0],
                prior_scales=[0.2, 0.7, 1.0, 1.5],
                debug=True,
            )
            tune_and_evaluate_helper(
                training_name + "/offline/4gram_lm",
                asr_model,
                decoder_config_offline,
                lm_scales=[0, 1.4, 1.5, 1.6, 2.0, 2.2],
                prior_scales=[0.2, 0.3, 0.7, 1.0],
                debug=True,
            )



            # beam-search w/ LSTM-LM
            asr_model = prepare_asr_model(
                training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe,
                get_specific_checkpoint=num_epochs,
                mem_rqmt=prior_mem_rqmt
            )

            search_config = CTCBeamSearchLMConfig(
                lm_module="pytorch_networks.lm.lstm.kazuki_lstm_zijian_variant_v2.Model",
                lm_model_args=lstm_2x1024.net_args,
                lm_checkpoint=lstm_2x1024.checkpoint,
                lm_package=PACKAGE,
                prior_file=asr_model.prior_file,
            )
            decoder_config_streaming = DecoderConfig(
                beam_size=16,  # 16
                returnn_vocab=label_datastream_bpe.vocab,

                search_config=search_config,

                mode=Mode.STREAMING.name,
                chunk_size=int(model_config.chunk_size),
                lookahead_size=int(model_config.lookahead_size * 0.06 * 16e3),
                carry_over_size=model_config.carry_over_size,
                test_version=0.0,
            )
            decoder_config_offline = DecoderConfig(
                beam_size=16,  # 16
                returnn_vocab=label_datastream_bpe.vocab,
                search_config=search_config,

                mode=Mode.OFFLINE.name,
                test_version=0.0,
            )

            tune_and_evaluate_helper(
                training_name + "/streaming/lstm_lm",
                asr_model,
                decoder_config_streaming,
                lm_scales=[0, 1.4, 1.5, 1.6, 2.0],
                prior_scales=[0.2, 0.7, 1.0, 1.5],
                debug=True,
                use_gpu=True,
            )
            tune_and_evaluate_helper(
                training_name + "/offline/lstm_lm",
                asr_model,
                decoder_config_offline,
                lm_scales=[0, 1.4, 1.5, 1.6, 2.0, 2.2],
                prior_scales=[0.2, 0.3, 0.7, 1.0],
                debug=True,
                use_gpu=True,
            )



            # nicks decoder (testing)
            beam_search_decoder_config_nick_offline = NickDecoderConfig(
                returnn_vocab=label_datastream_bpe.vocab,
                beam_size=10,
                lm_model_args=lstm_2x1024.net_args,
                lm_checkpoint=lstm_2x1024.checkpoint,
                lm_module="pytorch_networks.lm.lstm.kazuki_lstm_zijian_variant_v2.Model",
                lm_states_need_label_axis=False,
                test_version=0.0,
            )
            decoder_unhashed_config = DecoderExtraConfig(
                lm_package=PACKAGE,
            )

            tune_and_evaluate_helper_new(
                training_name + "/offline/lstm_lm_compare_old",
                dev_dataset_tuples=dev_dataset_tuples,
                test_dataset_tuples=test_dataset_tuples,
                asr_model=asr_model,
                base_decoder_config=beam_search_decoder_config_nick_offline,
                unhashed_decoder_config=decoder_unhashed_config,
                lm_scales=[0, 1.4, 1.5, 1.6, 2.0, 2.2],
                prior_scales=[0.2, 0.3, 0.7, 1.0],
                decoder_module="ctc.search.beam_search_bpe_ctc_v4",
                debug=True,
                use_gpu=True,
            )
            tune_and_evaluate_helper_new(
                training_name + "/offline/lstm_lm",
                dev_dataset_tuples=dev_dataset_tuples,
                test_dataset_tuples=test_dataset_tuples,
                asr_model=asr_model,
                base_decoder_config=beam_search_decoder_config_nick_offline,
                unhashed_decoder_config=decoder_unhashed_config,
                lm_scales=[0, 0.65, 0.7, 0.8],
                prior_scales=[0, 0.25, 0.35, 0.4],
                decoder_module="ctc.search.beam_search_bpe_ctc_v4",
                debug=True,
                use_gpu=True,
            )

            #
            # experiments on each config
            #

            if experiment == "ctc.baseline":
                pass
                # tune_and_evaluate_helper(
                #     training_name + "/offline",
                #     asr_model,
                #     decoder_config_offline,
                #     lm_scales=[1.4, 1.5, 1.6],
                #     prior_scales=[0.7, 0.8, 0.95, 1.05],
                #     decoder_module="search.ctc_streamable_decoder_v1"
                # )

            if experiment == "ctc.streaming":
                pass


def ls960_streamable_ctc_refactored():
    experiment_configs = {
        # "ctc.baseline": {
        #     "model_params": {
        #         "chunk_size": [2.39],
        #         "lookahead_size": [8],
        #         "carry_over_size": [2],
        #         "dual_mode": [False],

        #         "kernel_size": [31],
        #         "specauc_start_epoch": [11],
        #         "training_strategy": [TrainMode.OFFLINE],
        #     },

        #     "network_module": "ctc.models.streamable_ctc_v1",
        #     "accum_grads": 1,
        #     "gpu_mem": 48,
        #     "num_epochs": 1000,
        #     "keep": [300]  # early checkpoint to see if model training working
        # },

        "ctc.streaming": {
            "model_params": {
                "chunk_size": [2.39],
                "lookahead_size": [8],
                "carry_over_size": [2],
                "dual_mode": [False],

                "kernel_size": [31],
                "specauc_start_epoch": [11],
                "training_strategy": [TrainMode.STREAMING],
            },
            "network_module": "ctc.models.model",
            "accum_grads": 1,
            "gpu_mem": 48,
            "num_epochs": 1000,
            "keep": [300]  # early checkpoint to see if model training working
        },

        # "ctc.28frames": {
        #     "model_params": {
        #         "chunk_size": [1.67],
        #         "lookahead_size": [8],
        #         "carry_over_size": [1],
        #         "dual_mode": [False],

        #         "kernel_size": [31],
        #         "specauc_start_epoch": [11],
        #         "training_strategy": [TrainMode.OFFLINE, TrainMode.STREAMING, TrainMode.SWITCHING]
        #     },
        #     "network_module": "ctc.models.model",
        #     "accum_grads": 1,
        #     "gpu_mem": 48,
        #     "num_epochs": 1000,
        #     "keep": [300]  # early checkpoint to see if model training working
        # },
    }

    run_experiments(experiments_config=experiment_configs, bpe_size=128)
