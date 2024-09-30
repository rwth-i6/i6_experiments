import copy
import os.path
from dataclasses import asdict
import numpy as np
from typing import cast, Any, Dict, List, Optional

from sisyphus import tk
from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.vieting.tools.report import Report

from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel


def eow_phon_ls960_1023_base():
    prefix_name = "experiments/librispeech/librispeech_960_ctc/feat"

    report = Report(
        columns_start=["train_name"],
        columns_end=["lm_scale", "prior_scale", "wer"],
    )

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_eow_phon_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
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
        assert training_name.startswith(prefix_name + "/"),  "Assumption for report"
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
                            "train_name": training_name[len(prefix_name + "/"):],
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
                        "train_name": training_name[len(prefix_name + "/"):],
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
        max_dim_feat=8,  # Jingjing style
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
        out_features=512,
        activation=None,
    )

    model_config = ModelConfig(
        feature_extraction_config=lgm_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
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
        specauc_start_epoch=1,
    )

    train_config_24gbgpu_amp = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 240))
        + list(np.linspace(5e-4, 5e-5, 240))
        + list(np.linspace(5e-5, 1e-7, 20)),
        #############
        "batch_size": 360 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
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

    training_name = prefix_name + "/" + network_module + ".lgmV1.512dim_sub4_24gbgpu_50eps"
    train_job = training(training_name, train_data, train_args, num_epochs=500, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=500
    )
    tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4]
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
        out_features=512,
        activation=None,
    )
    model_config = ScfModelConfig(
        feature_extraction_config=scf_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
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
        specaug_start_epoch=1,
        feature_training_start_epoch=0,
        feature_training_end_epoch=-1,
    )
    # accumulate gradient
    network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_ScfV1_v1"
    train_args = {
        "config": {
            **train_config_24gbgpu_amp,
            "batch_size": 180 * 16000,
            "accum_grad_multiple_step": 2,
        },
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
    }

    training_name = prefix_name + "/" + network_module + ".512dim_sub4_24gbgpu_50eps_bs2x180"
    train_job = training(training_name, train_data, train_args, num_epochs=500, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=500,
        prior_config={"batch_size": 50 * 16000},
    )
    tune_and_evaluate_helper(
        training_name, asr_model, default_decoder_config, lm_scales=[2.3, 2.5, 2.7], prior_scales=[0.2, 0.3, 0.4],
        forward_config={"batch_size": 100 * 16000},
    )

    # check updates from LS100
    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_feat_v1_cfg import (
        ModelConfig as CustomFeatureModelConfig,
    )
    from ...pytorch_networks.features.feature_extraction_v4 import (
        SupervisedConvolutionalFeatureExtractionV4Config,
        SpecaugConfig as FeatureSpecaugConfig,
    )
    specaug_config_scf = FeatureSpecaugConfig(
        start_epoch=1,
        time_min_num_masks=2,
        time_max_mask_per_n_frames=25,
        time_mask_max_size=20,
        freq_min_num_masks=2,
        freq_max_num_masks=5,
        freq_mask_max_size=64,
        freq_mask_max_size_delta_per_epoch=(256 - 64) / 300,
    )
    scf_config = SupervisedConvolutionalFeatureExtractionV4Config(
        module_class="SupervisedConvolutionalFeatureExtractionV4",
        wave_norm=True,
        num_tf=150,
        size_tf=256,
        stride_tf=10,
        init_tf="gammatone",
        num_env=5,
        size_env=40,
        stride_env=16,
        init_env="hann",
        interleaved_resolutions=True,
        convs=[(1, 50, 50)],
        init_convs="ones",
        specaug_config=specaug_config_scf,
        specaug_before_conv_red=True,
    )
    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=50,
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
        out_features=512,
        activation=None,
    )
    model_config = CustomFeatureModelConfig(
        feature_extraction_config=scf_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
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
        specaug_start_epoch=9999,
        feature_training_start_epoch=0,
        feature_training_end_epoch=-1,
    )

    for specaug_dim_start, specaug_dim_end in [(32, 64), (64, 128), (64, 256)]:
        network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_feat_v1"
        model_config.feature_extraction_config.specaug_config.freq_mask_max_size = specaug_dim_start
        model_config.feature_extraction_config.specaug_config.freq_mask_max_size_delta_per_epoch = (
            (specaug_dim_end - specaug_dim_start) / 300
        )
        train_args = {
            "config": {
                **copy.deepcopy(train_config_24gbgpu_amp),
                "batch_size": 360 * 16000,
            },
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
        }
        training_name = (
                prefix_name + "/" + network_module +
                f".512dim_sub4_24gbgpu_50eps_bs360.convredv11.init."
                f"sasortv2dim{specaug_dim_start}to{specaug_dim_end}beforeTrue"
        )
        train_job = training(training_name, train_data, train_args, num_epochs=500, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        asr_model = prepare_asr_model(
            training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=500,
            prior_config={"batch_size": 50 * 16000},
        )
        tune_and_evaluate_helper(
            training_name, asr_model, default_decoder_config, lm_scales=[3.5], prior_scales=[0.3, 0.5],
            forward_config={"batch_size": 100 * 16000},
        )

    # repeat with different random seed
    model_config = CustomFeatureModelConfig(
        feature_extraction_config=scf_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
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
    for specaug_dim_start, specaug_dim_end in [(64, 128)]:
        network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_feat_v1"
        model_config.feature_extraction_config.specaug_config.freq_mask_max_size = specaug_dim_start
        model_config.feature_extraction_config.specaug_config.freq_mask_max_size_delta_per_epoch = (
            (specaug_dim_end - specaug_dim_start) / 300
        )
        train_args = {
            "config": {
                **copy.deepcopy(train_config_24gbgpu_amp),
                "batch_size": 360 * 16000,
                "random_seed": 43,
            },
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
        }
        training_name = (
                prefix_name + "/" + network_module +
                f".512dim_sub4_24gbgpu_50eps_bs360.convredv11.init."
                f"sasortv2dim{specaug_dim_start}to{specaug_dim_end}beforeTrue.rndseed43"
        )
        train_job = training(training_name, train_data, train_args, num_epochs=500, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        asr_model = prepare_asr_model(
            training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=500,
            prior_config={"batch_size": 50 * 16000},
        )
        tune_and_evaluate_helper(
            training_name, asr_model, default_decoder_config, lm_scales=[3.5], prior_scales=[0.3, 0.5],
            forward_config={"batch_size": 100 * 16000},
        )

    # test higher dropout
    model_config = CustomFeatureModelConfig(
        feature_extraction_config=scf_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
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
    for specaug_dim_start, specaug_dim_end in [(64, 128)]:
        network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_feat_v1"
        model_config.feature_extraction_config.specaug_config.freq_mask_max_size = specaug_dim_start
        model_config.feature_extraction_config.specaug_config.freq_mask_max_size_delta_per_epoch = (
            (specaug_dim_end - specaug_dim_start) / 300
        )
        train_args = {
            "config": {
                **copy.deepcopy(train_config_24gbgpu_amp),
                "batch_size": 360 * 16000,
            },
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
        }
        training_name = (
                prefix_name + "/" + network_module +
                f".512dim_sub4_24gbgpu_50eps_bs360.convredv11.init."
                f"sasortv2dim{specaug_dim_start}to{specaug_dim_end}beforeTrue.drop0.2"
        )
        train_job = training(training_name, train_data, train_args, num_epochs=500, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        asr_model = prepare_asr_model(
            training_name, train_job, train_args, with_prior=True, datasets=train_data, get_specific_checkpoint=500,
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