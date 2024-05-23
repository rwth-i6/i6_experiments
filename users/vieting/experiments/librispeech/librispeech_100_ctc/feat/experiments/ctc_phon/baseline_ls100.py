import copy
import os.path
from dataclasses import asdict
import numpy as np
from typing import cast, List, Optional, Dict, Any

from sisyphus import tk
from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.vieting.tools.report import Report

from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel


def eow_phon_ls100_1023_base():
    prefix_name = "experiments/librispeech/librispeech_100_ctc/feat"

    report = Report(
        columns_start=["train_name"],
        columns_end=["lm_scale", "prior_scale", "wer"],
    )

    train_settings = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=3,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_eow_phon_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-clean-100",
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

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

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig

    def tune_and_evaluate_helper(
        training_name: str,
        asr_model: ASRModel,
        base_decoder_config: DecoderConfig,
        lm_scales: List[float],
        prior_scales: List[float],
        forward_config: Optional[Dict[str, Any]] = None,
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
                    forward_config=forward_config or {},
                    asr_model=asr_model,
                    decoder_module="ctc.decoder.flashlight_ctc_v1",
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples=dev_dataset_tuples,
                    **default_returnn,
                )
                tune_parameters.append((lm_weight, prior_scale))
                tune_values_clean.append((wers[search_name + "/dev-clean"]))
                tune_values_other.append((wers[search_name + "/dev-other"]))
                for key, wer in wers.items():
                    report.add(
                        {
                            "train_name": training_name,
                            "prior_scale": prior_scale,
                            "lm_scale": lm_weight,
                            "eval_set": key.split("/")[-1],
                            "wer": wer,
                        }
                    )

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
                forward_config=forward_config or {},
                asr_model=asr_model,
                decoder_module="ctc.decoder.flashlight_ctc_v1",
                decoder_args={"config": asdict(decoder_config)},
                test_dataset_tuples={key: test_dataset_tuples[key]},
                **default_returnn,
            )
            for key, wer in wers.items():
                report.add(
                    {
                        "train_name": training_name,
                        "prior_scale": decoder_config.prior_scale.get(),
                        "lm_scale": decoder_config.lm_weight.get(),
                        "eval_set": key.split("/")[-1],
                        "wer": wer,
                    }
                )

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
        LogMelFeatureExtractionV1Config,
    )

    lgm_config = LogMelFeatureExtractionV1Config(
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
        max_dim_feat=16,
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
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )

    model_config = ModelConfig(
        feature_extraction_config=lgm_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
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
        specauc_start_epoch=1,
    )

    train_config_24gbgpu_amp = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": list(np.linspace(7e-6, 7e-4, 140))
        + list(np.linspace(7e-4, 7e-5, 140))
        + list(np.linspace(7e-5, 1e-8, 30)),
        #############
        "batch_size": 360 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "gradient_clip": 1.0,
        "torch_amp_options": {"dtype": "bfloat16"},
    }

    # Log mel baseline
    network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6"
    train_args = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
    }

    training_name = prefix_name + "/" + network_module + ".lgmV1.384dim_sub4_24gbgpu_100eps"
    train_job = training(training_name, train_data, train_args, num_epochs=300, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=300
    )
    tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[3.5], prior_scales=[0.3, 0.5]
    )

    # Vanilla SCF
    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_ScfV1_v1_cfg import (
        ModelConfig as ScfModelConfig,
        SupervisedConvolutionalFeatureExtractionV1Config,
    )
    scf_config = SupervisedConvolutionalFeatureExtractionV1Config(
        wave_norm=True,
        num_tf=150,
        size_tf=256,
        stride_tf=10,
        num_env=5,
        size_env=40,
        stride_env=16
    )
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=750,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )
    model_config = ScfModelConfig(
        feature_extraction_config=scf_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
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
        specaug_start_epoch=1,
        feature_training_start_epoch=0,
        feature_training_end_epoch=-1,
    )

    # accumulate gradient
    network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_ScfV1_v1"
    train_args = {
        "config": {
            **copy.deepcopy(train_config_24gbgpu_amp),
            "batch_size": 180 * 16000,
            "accum_grad_multiple_step": 2,
        },
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
    }

    training_name = prefix_name + "/" + network_module + ".384dim_sub4_24gbgpu_100eps_bs2x180"
    train_job = training(training_name, train_data, train_args, num_epochs=300, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=300,
        prior_config={"batch_size": 50 * 16000},
    )
    tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[3.5], prior_scales=[0.3, 0.5],
        forward_config={"batch_size": 100 * 16000},
    )

    # use trained features
    from ..convert_scf_checkpoint import get_scf_checkpoint
    network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_ScfV1_v1"
    model_config_v1 = copy.deepcopy(model_config)
    model_config_v1.feature_extraction_config.size_tf = 160
    model_config_v1.feature_training_start_epoch = 1000
    train_args = {
        "config": {
            **copy.deepcopy(train_config_24gbgpu_amp),
            "batch_size": 180 * 16000,
            "accum_grad_multiple_step": 2,
            "preload_from_files": {
                "scf": {
                    "filename": get_scf_checkpoint(),
                    "ignore_missing": True,
                    "checkpoint_key": "model",
                    "prefix": "feature_extraction.",
                    "init_for_train": True,
                },
            },
        },
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config_v1)},
        "debug": False,
    }

    training_name = prefix_name + "/" + network_module + ".384dim_sub4_24gbgpu_100eps_bs2x180_scfinitfix"
    train_job = training(training_name, train_data, train_args, num_epochs=300, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=300,
        prior_config={"batch_size": 50 * 16000},
    )
    tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[3.5], prior_scales=[0.3, 0.5],
        forward_config={"batch_size": 100 * 16000},
    )

    # adjust SpecAugment in frequency dimension
    scf_config = SupervisedConvolutionalFeatureExtractionV1Config(
        wave_norm=True,
        num_tf=150,
        size_tf=256,
        stride_tf=10,
        num_env=5,
        size_env=40,
        stride_env=16
    )
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=int(16 / 80 * 750),
        num_repeat_feat=5,
    )
    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=750,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )
    model_config = ScfModelConfig(
        feature_extraction_config=scf_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
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
        specaug_start_epoch=1,
        feature_training_start_epoch=0,
        feature_training_end_epoch=-1,
    )
    for specaug_max_dim_feat, specaug_start_epoch in [(32, 1), (64, 1), (128, 1), (150, 1), (150, 50)]:
        network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_ScfV1_v1"
        model_config.specaug_config.max_dim_feat = specaug_max_dim_feat
        model_config.specaug_start_epoch = specaug_start_epoch
        train_args = {
            "config": {
                **copy.deepcopy(train_config_24gbgpu_amp),
                "batch_size": 180 * 16000,
                "accum_grad_multiple_step": 2,
            },
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
        }

        training_name = (
            prefix_name + "/" + network_module + f".384dim_sub4_24gbgpu_100eps_bs2x180_sa{specaug_max_dim_feat}" +
            (f"start{specaug_start_epoch}" if specaug_start_epoch > 1 else "")
        )
        train_job = training(training_name, train_data, train_args, num_epochs=300, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        asr_model = prepare_asr_model(
            training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=300,
            prior_config={"batch_size": 50 * 16000},
        )
        tune_and_evaluate_helper(
            training_name, asr_model, default_decoder_config, lm_scales=[3.5], prior_scales=[0.3, 0.5],
            forward_config={"batch_size": 100 * 16000},
        )

    # modified SCF variants
    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_feat_v1_cfg import (
        ModelConfig as CustomFeatureModelConfig,
    )
    from ...pytorch_networks.ctc.conformer_1023.feature_extraction import (
        SupervisedConvolutionalFeatureExtractionV2Config,
    )
    scf_config_v2 = SupervisedConvolutionalFeatureExtractionV2Config(
        module_class="SupervisedConvolutionalFeatureExtractionV2",
        scf_config=scf_config,
        convs=[(10, 320, 1), (10, 80, 1)],
        init_tf="gammatone",
        init_env="hann",
        init_convs="ones",
    )
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
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
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )
    model_config = CustomFeatureModelConfig(
        feature_extraction_config=scf_config_v2,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
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
        specaug_start_epoch=1,
        feature_training_start_epoch=0,
        feature_training_end_epoch=-1,
    )
    for exp_name, convs in [
        ("none_init", []),
        ("v3_init", [(10, 150, 150)]),
        ("v4_init", [(3, 150, 150)]),
        ("v5_init", [(10, 50, 50)]),
        ("v6_init", [(10, 150, 50)]),
        ("v7_init", [(10, 150, 50), (5, 50, 5)]),
        # ("v8_init", [(1, 80, 1)]),
        ("v9_init", [(4, 50, 50)]),
        ("v10_init", [(2, 50, 50)]),
        ("v11_init", [(1, 50, 50)]),
    ]:
        network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_feat_v1"
        model_config.feature_extraction_config.convs = convs
        model_config.frontend_config.in_features = 750 if len(convs) == 0 else convs[-1][1]
        train_args = {
            "config": {
                **copy.deepcopy(train_config_24gbgpu_amp),
                "batch_size": 180 * 16000,
                "accum_grad_multiple_step": 2,
            },
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": True,
        }

        training_name = prefix_name + "/" + network_module + f".384dim_sub4_24gbgpu_100eps_bs2x180.convred{exp_name}"
        train_job = training(training_name, train_data, train_args, num_epochs=300, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        asr_model = prepare_asr_model(
            training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=300,
            prior_config={"batch_size": 50 * 16000},
        )
        tune_and_evaluate_helper(
            training_name, asr_model, default_decoder_config, lm_scales=[3.5], prior_scales=[0.3, 0.5],
            forward_config={"batch_size": 100 * 16000},
        )

    # sorted SpecAugment masks
    from ...pytorch_networks.features.feature_extraction_v3 import SupervisedConvolutionalFeatureExtractionV3Config
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,
        num_repeat_feat=5,
    )
    scf_config_v3 = SupervisedConvolutionalFeatureExtractionV3Config(
        module_class="SupervisedConvolutionalFeatureExtractionV3",
        wave_norm=True,
        num_tf=150,
        size_tf=256,
        stride_tf=10,
        num_env=5,
        size_env=40,
        stride_env=16,
        specaug_config=specaug_config,
        specaug_start_epoch=1,
    )
    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=750,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )
    model_config = CustomFeatureModelConfig(
        feature_extraction_config=scf_config_v3,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
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
        specaug_start_epoch=9999,
        feature_training_start_epoch=0,
        feature_training_end_epoch=-1,
    )
    for exp_name, f_dim in [
        ("v1", 16),
        ("v1", 32),
        ("v1", 64),
        ("v1", 128),
    ]:
        model_config.feature_extraction_config.specaug_config.max_dim_feat = f_dim
        network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_feat_v1"
        train_args = {
            "config": {
                **copy.deepcopy(train_config_24gbgpu_amp),
                "batch_size": 180 * 16000,
                "accum_grad_multiple_step": 2,
            },
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
        }

        training_name = (
            prefix_name + "/" + network_module +
            f".384dim_sub4_24gbgpu_100eps_bs2x180.sasort{exp_name}dim{f_dim}"
        )
        train_job = training(training_name, train_data, train_args, num_epochs=300, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        asr_model = prepare_asr_model(
            training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=300,
            prior_config={"batch_size": 50 * 16000},
        )
        tune_and_evaluate_helper(
            training_name, asr_model, default_decoder_config, lm_scales=[3.5], prior_scales=[0.3, 0.5],
            forward_config={"batch_size": 100 * 16000},
        )

    # finish report
    report.delete_redundant_columns()
    report.delete_redundant_rows()
    report.merge_eval_sets("eval_set")
    tk.register_report(
        os.path.join(prefix_name, "report.csv"),
        values=report.get_values(),
        template=report.get_template(),
    )
