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
from ...pipeline import training, prepare_asr_model, search, force_align, ASRModel
from ...storage import add_ctc_model, add_ctc_forced_alignment
from ...latency import BPEToWordAlignmentsJob

from ...pytorch_networks.rnnt.auxil.functional import TrainingStrategy, Mode


def product_dict(**kwargs):
    keys = kwargs.keys()

    from itertools import product
    for instance in product(*kwargs.values()):
        yield dict(zip(keys, instance))


def get_train_config(model_config, keep, module, accum_grads=1, **kwargs):
    num_epochs = kwargs.get("num_epochs")

    epochs_r = num_epochs / 1000
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

    train_args_24gb_default = {
        "config": train_config_24gbgpu,
        "network_module": module,
        "include_native_ops": True,
        "debug": True,
        "net_args": {"model_config_dict": asdict(model_config)}
    }

    return train_args_24gb_default


def run_experiments(**kwargs):
    prefix_name = "example_setups/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_relpos_streamable_0425"
    bpe_size = kwargs.get("bpe_size", 128)
    experiments_config = kwargs.get("experiments_config")

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
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
    from ...pytorch_networks.ctc.decoder.streamable_ctc_decoder_v1 import DecoderConfig

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

    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        LogMelFeatureExtractionV1Config,
    )
    from ...pytorch_networks.rnnt.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1_cfg import \
        ConformerPosEmbConfig
    from ...pytorch_networks.ctc.conformer_0425.model_dual_0425_cfg import ModelConfig

    fe_config = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.025,
        hop_size=0.01,
        f_min=60,
        f_max=7600,
        min_amp=1e-10,
        num_filters=80,
        center=True,        # TODO: changed to True
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
    dev_dataset_tuples_withlabels = {}
    for testset in ["dev-clean", "dev-other"]:
        # add labels to dev for alignment jobs
        dev_dataset_tuples_withlabels[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
            label_datastream=label_datastream_bpe
        )
    for experiment in experiments_config:
        exp_config = experiments_config[experiment]
        param_combinations = product_dict(**exp_config["model_params"])

        for param_combi in param_combinations:
            assert (int(param_combi["chunk_size"]/0.01) + 1) % 6 == 0, "`chunk_size` + 10ms should be divisible by 60ms"
            
            model_config = ModelConfig(
                feature_extraction_config=fe_config,
                frontend_config=frontend_config,
                specaug_config=specaug_config,
                pos_emb_config=posemb_config,
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
                conv_kernel_size=param_combi["kernel_size"],
                final_dropout=0.1,
                specauc_start_epoch=11,
                dropout_broadcast_axes=None,
                module_list=["ff", "conv", "mhsa", "ff"],
                module_scales=[0.5, 1.0, 1.0, 0.5],

                fastemit_lambda=None,
                chunk_size=param_combi["chunk_size"] * 16e3,
                lookahead_size=param_combi["lookahead_size"],
                online_model_scale=0.5,
                carry_over_size=param_combi["carry_over_size"],
                training_strategy=str(param_combi["training_strategy"]),
                dual_mode=param_combi["dual_mode"]
            )

            num_epochs = exp_config.get("num_epochs")
            KEEP = exp_config.get("keep")
            train_args = get_train_config(model_config, keep=KEEP,
                                          module=exp_config["network_module"],
                                          accum_grads=exp_config["accum_grads"],
                                          num_epochs=num_epochs)

            gpu_mem = exp_config["gpu_mem"]
            train_strat = param_combi["training_strategy"].name.lower()
            training_name = (
                    prefix_name + "/" + str(bpe_size) + "/" +
                    train_args["network_module"] +
                    ".512dim_sub6_%dgbgpu_" % gpu_mem +
                    "%deps_" % (num_epochs // 10) +
                    "from_scratch_adamw_%s_specaug%d" % (train_strat, model_config.specauc_start_epoch) + "/" +
                    str(param_combi["chunk_size"]) + "/" +
                    "carry%.1f" % model_config.carry_over_size + "/" +
                    "lah%i" % model_config.lookahead_size
            )
            train_job = training(training_name, train_data_bpe, train_args,
                                 num_epochs=num_epochs, **default_returnn)
            train_job.rqmt["gpu_mem"] = gpu_mem
            train_job.set_env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

            #
            # decoding jobs
            #
            default_decoder_config_bpe128 = DecoderConfig(
                lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=128),
                returnn_vocab=label_datastream_bpe128.vocab,
                beam_size=1024,  # Untuned
                beam_size_token=16,
                arpa_lm=arpa_4gram_lm,
                beam_threshold=14,  # Untuned
                mode=str(Mode.STREAMING),
                chunk_size=int(model_config.chunk_size),
                lookahead_size=int(model_config.lookahead_size * 0.06 * 16e3),
                carry_over_size=model_config.carry_over_size,
                test_version=0.0,
            )
            offline_decoder_config_bpe128 = DecoderConfigOffline(
                lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=128),
                returnn_vocab=label_datastream_bpe128.vocab,
                beam_size=1024,  # Untuned
                beam_size_token=16,
                arpa_lm=arpa_4gram_lm,
                beam_threshold=14,  # Untuned,
            )

            asr_model = prepare_asr_model(
                training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe128,
                get_specific_checkpoint=num_epochs
            )
            tune_and_evaluate_helper(
                training_name + "/online",
                asr_model,
                default_decoder_config_bpe128,
                lm_scales=[1.6, 1.8, 2.0, 2.2, 2.4],
                prior_scales=[0.2, 0.3, 0.4, 0.6, 0.8],
                decoder_module="ctc.decoder.streamable_ctc_decoder_v1"
            )
            tune_and_evaluate_helper(
                training_name + "/offline",
                asr_model,
                offline_decoder_config_bpe128,
                lm_scales=[1.6, 1.8, 2.0, 2.2, 2.4],
                prior_scales=[0, 0.2, 0.3, 0.4, 0.6, 0.8],
                decoder_module="ctc.decoder.streamable_ctc_decoder_v1"
            )

            # #
            # # force alignment jobs
            # #
            # from ...pytorch_networks.ctc.aligner.experimental_ctc_aligner_v1 import AlignerConfig
            # aligner_config = AlignerConfig(
            #     returnn_vocab=label_datastream_bpe128.vocab,
            #     prior_scale=0.2,
            #     prior_file=asr_model.prior_file,
            #     chunk_size=int(model_config.chunk_size),
            #     lookahead_size=int(model_config.lookahead_size * 0.06 * 16e3),
            #     carry_over_size=model_config.carry_over_size,
            #     test_version=0.0,
            # )
            # search_name = training_name + "/falign_prior%.1f" % aligner_config.prior_scale
            # if experiment == 10 and model_config.training_strategy == str(TrainingStrategy.UNIFIED):
            #     add_ctc_model("unified_relpos_large", asr_model)
            #     mode = Mode.OFFLINE
            #     aligner_config.mode = str(mode)

            #     align_jobs = force_align(
            #         search_name + "/%s" % mode.name.lower(),
            #         forward_config={},
            #         asr_model=asr_model,
            #         decoder_module="ctc.aligner.experimental_ctc_aligner_v1",
            #         decoder_args={"config": asdict(aligner_config)},
            #         test_dataset_tuples=dev_dataset_tuples_withlabels,
            #         **default_returnn,
            #     )
            #     word_aligns_job = BPEToWordAlignmentsJob(
            #         alignment_path=align_jobs["dev-other"].out_files["aligns_out.json"],
            #         labels_path=label_datastream_bpe.vocab
            #     )
            #     word_aligns_job.add_alias(training_name + "/dev-other" + "/word_aligns_job")
            #     add_ctc_forced_alignment("dev-other", word_aligns_job.word_alignments)
            # elif experiment == 20 and model_config.carry_over_size == 2:
            #     mode = Mode.STREAMING
            #     aligner_config.mode = str(mode)

            #     # force align ctc on correct labels
            #     align_jobs = force_align(
            #         search_name + "/%s" % mode.name.lower(),
            #         forward_config={},
            #         asr_model=asr_model,
            #         decoder_module="ctc.aligner.experimental_ctc_aligner_v1",
            #         decoder_args={"config": asdict(aligner_config)},
            #         test_dataset_tuples=dev_dataset_tuples_withlabels,
            #         **default_returnn,
            #     )
            #     word_aligns_job = BPEToWordAlignmentsJob(
            #         alignment_path=align_jobs["dev-other"].out_files["aligns_out.json"],
            #         labels_path=label_datastream_bpe.vocab
            #     )
            #     word_aligns_job.add_alias(training_name + "/dev-other" + "/word_aligns_job")
            #     add_ctc_forced_alignment("streaming/dev-other", word_aligns_job.word_alignments)
            # elif experiment == 15 and model_config.training_strategy == str(TrainingStrategy.UNIFIED):
            #     for mode in [Mode.OFFLINE, Mode.STREAMING]:
            #         aligner_config.mode = str(mode)
            #         align_jobs = force_align(
            #             search_name + "/%s" % mode.name.lower(),
            #             forward_config={},
            #             asr_model=asr_model,
            #             decoder_module="ctc.aligner.experimental_ctc_aligner_v1",
            #             decoder_args={"config": asdict(aligner_config)},
            #             test_dataset_tuples=dev_dataset_tuples_withlabels,
            #             **default_returnn,
            #         )
            #         word_aligns_job = BPEToWordAlignmentsJob(
            #             alignment_path=align_jobs["dev-other"].out_files["aligns_out.json"],
            #             labels_path=label_datastream_bpe.vocab
            #         )
            #         word_aligns_job.add_alias(
            #             training_name + "/dev-other/%s" + mode.name.lower() + "/word_aligns_job"
            #         )
            #         add_ctc_forced_alignment(
            #             "2.39/%s/dev-other" % mode.name.lower(), word_aligns_job.word_alignments
            #         )


def ls960_ctc_relpos_streamable_0425_low_bpe_from_scratch():
    experiment_configs = {
        10: {
            "model_params": {
                "chunk_size": [2.39],
                "lookahead_size": [8],
                "kernel_size": [31],
                "specauc_start_epoch": [11],
                "carry_over_size": [2],
                "training_strategy": [TrainingStrategy.UNIFIED, TrainingStrategy.STREAMING],
                "dual_mode": [False],
            },

            "network_module": "ctc.conformer_0425.model_dual_0425_v1",
            "accum_grads": 1,
            "gpu_mem": 48,
            "num_epochs": 1000,
            "keep": [300, 500, 800, 950]
        },
    }

    run_experiments(experiments_config=experiment_configs, bpe_size=128)
