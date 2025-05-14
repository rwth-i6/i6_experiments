from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
import os.path
from typing import cast

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.vieting.tools.report import Report

from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search
from ...report import tune_and_evalue_report, DelayedMin
from ...storage import add_ctc_model


def eow_phon_ls960_relposencoder_0924_base():
    prefix_name = "experiments/librispeech/librispeech_960_ctc_eow_phon/feat_torch"

    report = Report(columns_start=["training_name"], columns_end=["dev-clean", "dev-other", "test-clean", "test-other"])

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

    train_settings_nonorm = copy.deepcopy(train_settings)
    train_settings_nonorm.peak_normalization = False
    train_data_nonorm = build_eow_phon_training_datasets(
        prefix=prefix_name,
        librispeech_key="train-other-960",
        settings=train_settings_nonorm,
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

    def tune_and_evaluate_helper(
        training_name, asr_model, base_decoder_config, lm_scales, prior_scales,
        decoder_module="ctc.decoder.flashlight_ctc_v1", forward_config=None,
    ):
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        report_values = {}
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
                    decoder_module=decoder_module,
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples=dev_dataset_tuples,
                    **default_returnn
                )
                tune_parameters.append((lm_weight, prior_scale))
                tune_values_clean.append((wers[search_name + "/dev-clean"]))
                tune_values_other.append((wers[search_name + "/dev-other"]))

        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                parameters=tune_parameters, values=tune_values, mode="minimize",
            )
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            decoder_config = copy.deepcopy(base_decoder_config)
            decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers = search(
                training_name, forward_config=forward_config or {}, asr_model=asr_model, decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)}, test_dataset_tuples={key: test_dataset_tuples[key]},
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
        assert training_name.startswith(prefix_name)
        report.add({
            "training_name": training_name[len(prefix_name):].strip("/"),
            "dev-clean": DelayedMin(tune_values_clean),
            "dev-other": DelayedMin(tune_values_other),
            **report_values
        })

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    from ...pytorch_networks.ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1_cfg import \
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig, LogMelFeatureExtractionV1Config, \
        ConformerPosEmbConfig

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
        max_dim_feat=16,  # classic style
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

    # Try to do like returnn frontend
    posemb_config = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )

    model_base_args = dict(
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
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=11,
        dropout_broadcast_axes=None,  # No dropout broadcast yet to properly compare
        module_list=["ff", "conv", "mhsa", "ff"],
        module_scales=[0.5, 1.0, 1.0, 0.5],
        aux_ctc_loss_layers=None,
        aux_ctc_loss_scales=None,
    )

    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        **model_base_args,
    )

    def run_with_standard_settings(
        network_module, model_cfg, name_ext="", train_data_custom=None, prior_batch_size=None, forward_config=None,
        train_rqmt=None, move_to_hpc=False, debug=False,
    ):
        train_config_24gbgpu_amp = {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
            "learning_rates": list(np.linspace(7e-6, 7e-4, 480)) + list(
                    np.linspace(7e-4, 5e-5, 480)) + list(np.linspace(5e-5, 1e-7, 40)),
            #############
            "batch_size": 360 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
            "torch_amp_options": {"dtype": "bfloat16"},
            "use_speed_perturbation": True,
            "gradient_clip_norm": 1.0,
        }

        train_args = {
            "config": train_config_24gbgpu_amp,
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_cfg)},
            "debug": debug,
            "use_speed_perturbation": True,
            "post_config": {"num_workers_per_gpu": 8},
        }

        name = ".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_lr07_work8" + name_ext
        training_name = prefix_name + "/" + network_module + name
        train_job = training(
            training_name, train_data_custom or train_data, train_args, num_epochs=1000, rqmt=train_rqmt,
            **default_returnn
        )
        train_job.rqmt["gpu_mem"] = 48
        if move_to_hpc and not debug:
            train_job.hold()
            train_job.move_to_hpc = True
        asr_model = prepare_asr_model(
            training_name, train_job, train_args, with_prior=True, datasets=train_data,
            prior_config={"batch_size": prior_batch_size * 16000} if prior_batch_size else None,
            get_specific_checkpoint=1000,
        )
        if prior_batch_size:
            asr_model.prior_file.creator.rqmt["time"] = 4
        tune_and_evaluate_helper(
            training_name, asr_model, default_decoder_config, lm_scales=[1.6, 1.8, 2.0], prior_scales=[0.2, 0.3, 0.4],
            forward_config=forward_config,
        )
        asr_model.returnn_vocab = label_datastream.vocab
        asr_model.settings = train_settings
        asr_model.label_datastream = label_datastream
        add_ctc_model(network_module + ".eow_phon" + name, asr_model)

    # baseline log Mel setup
    run_with_standard_settings(
        network_module="ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1", model_cfg=model_config,
        name_ext=".logmel", move_to_hpc=True,
    )

    # SCF experiments with minimal modifications
    from ...pytorch_networks.ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_feat_v1_cfg import (
        ModelConfig as FeatureModelConfigV1,
    )
    from ...pytorch_networks.ctc.features.scf import (
        SupervisedConvolutionalFeatureExtractionV1Config,
        SupervisedConvolutionalFeatureExtractionV2Config,
    )
    scf_config_base = SupervisedConvolutionalFeatureExtractionV1Config(
        wave_norm=True,
        num_tf=150,
        size_tf=256,
        stride_tf=10,
        num_env=5,
        size_env=40,
        stride_env=16,
    )
    scf_config = SupervisedConvolutionalFeatureExtractionV2Config(
        module_class="SupervisedConvolutionalFeatureExtractionV2",
        scf_config=scf_config_base,
        convs=[],
        init_tf="gammatone",
        init_env="hann",
        init_convs="ones",
    )
    model_base_args_feat = copy.deepcopy(model_base_args)
    model_base_args_feat["specaug_start_epoch"] = model_base_args_feat.pop("specauc_start_epoch")
    model_config = FeatureModelConfigV1(
        feature_extraction_config=scf_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        **model_base_args_feat,
    )

    for exp_name, convs in [
        (".scf", []),
        (".scf_init", []),
        (".scf_init_convred", [(1, 50, 50)]),
    ]:
        model_config_exp = copy.deepcopy(model_config)
        model_config_exp.feature_extraction_config.convs = convs
        model_config_exp.frontend_config.in_features = 750 if len(convs) == 0 else convs[-1][1]
        if "init" not in exp_name:
            model_config_exp.feature_extraction_config.init_tf = None
            model_config_exp.feature_extraction_config.init_env = None
        run_with_standard_settings(
            network_module="ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_feat_v1",
            model_cfg=model_config_exp, name_ext=exp_name, move_to_hpc=True,
            forward_config={"batch_size": 16000 * 250}, prior_batch_size=160,
        )

    # SCF experiments with STFT SpecAugment and configurable VGG front end
    from ...pytorch_networks.ctc.conformer_0924.i6models_relposV1_VGGNLayerActFrontendV1_feat_v2_cfg import (
        ModelConfig as FeatureModelConfigV2,
        SpecaugStftConfig,
        SpecaugStftV2Config,
        SpecaugMultiplierLinearConfig,
        VGGNLayerActFrontendV1Config,
        IdentityConfig,
    )

    frontend_config = VGGNLayerActFrontendV1Config(
        in_features=80,
        convs=[(32, (3, 3), 1), (64, (3, 3), 1), (64, (3, 3), 1), (32, (3, 3), 1)],
        activations=[None, "ReLU", None, "ReLU"],
        poolings=[None, ((2, 1), (2, 1), None), None, ((2, 1), (2, 1), None)],
        out_features=512,
    )
    specaug_configs = {
        "default": specaug_config,
        "default_v11": SpecaugConfig(
            repeat_per_n_frames=25,
            max_dim_time=55,
            max_dim_feat=16,
            num_repeat_feat=5,
        ),
        "default_v12": SpecaugConfig(
            repeat_per_n_frames=25,
            max_dim_time=55,
            max_dim_feat=int(16 / 80 * 201),
            num_repeat_feat=5,
        ),
        "stft_v1": SpecaugStftConfig(
            repeat_per_n_frames=25,
            max_dim_time=20,
            max_dim_feat=16,  # classic style
            num_repeat_feat=5,
            window_size=400,
            window_shift=320,
            fft_size=1023,
        ),
        "stft_v21": SpecaugStftConfig(  # as close to log Mel baseline as possible
            repeat_per_n_frames=25,
            max_dim_time=20,
            max_dim_feat=int(16 / 80 * 201),
            num_repeat_feat=5,
            window_size=400,
            window_shift=160,
            fft_size=400,
        ),
        "stft_v22": SpecaugStftConfig(
            repeat_per_n_frames=25,
            max_dim_time=20,
            max_dim_feat=int(16 / 80 * 201 * 1.3),
            num_repeat_feat=5,
            window_size=400,
            window_shift=160,
            fft_size=400,
        ),
        "stft_v23": SpecaugStftConfig(
            repeat_per_n_frames=25,
            max_dim_time=20,
            max_dim_feat=int(16 / 80 * 201 * 0.7),
            num_repeat_feat=5,
            window_size=400,
            window_shift=160,
            fft_size=400,
        ),
        "stft_v24": SpecaugStftConfig(
            repeat_per_n_frames=25,
            max_dim_time=20,
            max_dim_feat=int(16 / 80 * 201),
            num_repeat_feat=5,
            window_size=400,
            window_shift=160,
            fft_size=512,
        ),
        "stft_v25": SpecaugStftConfig(
            repeat_per_n_frames=25,
            max_dim_time=20,
            max_dim_feat=int(16 / 80 * 201),
            num_repeat_feat=5,
            window_size=400,
            window_shift=320,
            fft_size=400,
        ),
        "stft_v26": SpecaugStftConfig(
            repeat_per_n_frames=25,
            max_dim_time=20,
            max_dim_feat=int(16 / 80 * 201 * 1.3),
            num_repeat_feat=5,
            window_size=400,
            window_shift=320,
            fft_size=512,
        ),
        "stft_v27": SpecaugStftConfig(
            repeat_per_n_frames=25,
            max_dim_time=20,
            max_dim_feat=int(16 / 80 * 201 * 1.6),
            num_repeat_feat=5,
            window_size=400,
            window_shift=320,
            fft_size=512,
        ),
        "stft_v28": SpecaugStftConfig(
            repeat_per_n_frames=25,
            max_dim_time=20,
            max_dim_feat=int(16 / 80 * 201 * 2),
            num_repeat_feat=5,
            window_size=400,
            window_shift=320,
            fft_size=512,
        ),
        "stft_v29": SpecaugStftConfig(
            repeat_per_n_frames=25,
            max_dim_time=20,
            max_dim_feat=int(16 / 80 * 201 * 1.3),
            num_repeat_feat=10,
            window_size=400,
            window_shift=320,
            fft_size=512,
        ),
        "stft_v31": SpecaugStftConfig(  # try to imitate best swb variant
            repeat_per_n_frames=21,
            max_dim_time=15,
            max_dim_feat=16,
            num_repeat_feat=5,
            window_size=800,
            window_shift=320,
            fft_size=1024,
        ),
        "stft_v41": SpecaugStftConfig(
            repeat_per_n_frames=25,
            max_dim_time=40,
            max_dim_feat=int(16 / 80 * 201 * 1.3),
            num_repeat_feat=5,
            window_size=400,
            window_shift=160,
            fft_size=512,
        ),
        "stft_v42": SpecaugStftConfig(
            repeat_per_n_frames=25,
            max_dim_time=30,
            max_dim_feat=int(16 / 80 * 201 * 1.3),
            num_repeat_feat=5,
            window_size=400,
            window_shift=160,
            fft_size=512,
        ),
        "stft_v43": SpecaugStftConfig(
            repeat_per_n_frames=25,
            max_dim_time=50,
            max_dim_feat=int(16 / 80 * 201 * 1.3),
            num_repeat_feat=5,
            window_size=400,
            window_shift=160,
            fft_size=512,
        ),
        "stft_v44": SpecaugStftConfig(
            repeat_per_n_frames=25,
            max_dim_time=40,
            max_dim_feat=int(16 / 80 * 257 * 0.7),
            num_repeat_feat=5,
            window_size=400,
            window_shift=160,
            fft_size=512,
        ),
        "stft_v45": SpecaugStftConfig(
            repeat_per_n_frames=25,
            max_dim_time=40,
            max_dim_feat=int(16 / 80 * 257 * 1.3),
            num_repeat_feat=5,
            window_size=400,
            window_shift=160,
            fft_size=512,
        ),
        "stft_v46": SpecaugStftConfig(
            repeat_per_n_frames=25,
            max_dim_time=45,
            max_dim_feat=int(16 / 80 * 201 * 1.3),
            num_repeat_feat=5,
            window_size=400,
            window_shift=160,
            fft_size=512,
        ),
        "stft_v47": SpecaugStftConfig(
            repeat_per_n_frames=25,
            max_dim_time=55,
            max_dim_feat=int(16 / 80 * 201 * 1.3),
            num_repeat_feat=5,
            window_size=400,
            window_shift=160,
            fft_size=512,
        ),
        "stft_v48": SpecaugStftConfig(
            repeat_per_n_frames=25,
            max_dim_time=60,
            max_dim_feat=int(16 / 80 * 201 * 1.3),
            num_repeat_feat=5,
            window_size=400,
            window_shift=160,
            fft_size=512,
        ),
        "stft_v51": SpecaugStftV2Config(
            repeat_per_n_frames=25,
            max_dim_time=20,
            min_num_time=4,
            max_dim_feat=int(16 / 80 * 201 * 1.3),
            min_num_feat=4,
            num_repeat_feat=5,
            window_size=400,
            window_shift=320,
            fft_size=512,
        ),
        "stft_v52": SpecaugStftV2Config(
            repeat_per_n_frames=25,
            max_dim_time=20,
            min_num_time=2,
            max_dim_feat=int(16 / 80 * 201 * 1.3),
            min_num_feat=2,
            num_repeat_feat=5,
            multiplier=SpecaugMultiplierLinearConfig(
                start_epoch=500,
                end_epoch=1000,
                start_factor=1.,
                end_factor=1.5,
            ),
            window_size=400,
            window_shift=320,
            fft_size=512,
        ),
        "stft_v53": SpecaugStftV2Config(
            repeat_per_n_frames=25,
            max_dim_time=20,
            min_num_time=2,
            max_dim_feat=int(16 / 80 * 201 * 1.3),
            min_num_feat=2,
            num_repeat_feat=5,
            multiplier=SpecaugMultiplierLinearConfig(
                start_epoch=500,
                end_epoch=1000,
                start_factor=1.,
                end_factor=2.0,
            ),
            window_size=400,
            window_shift=320,
            fft_size=512,
        ),
    }
    model_config = FeatureModelConfigV2(
        specaug_config=specaug_configs["stft_v1"],
        feature_extraction_config=scf_config,
        frontend_config=frontend_config,
        frontend_config_class="VGGNLayerActFrontendV1Config",
        **model_base_args_feat,
    )

    for exp_name, convs in [
        (".scf", []),
        (".sa64.scf", []),
        (".sa128.scf", []),
        (".stftsa.scf", []),
        (".stftsav41.scf", []),
        (".stftsav43.scf", []),
        (".defaultsav11.scf", []),
    ]:
        model_config_exp = copy.deepcopy(model_config)
        model_config_exp.feature_extraction_config.convs = convs
        model_config_exp.frontend_config.in_features = 750 if len(convs) == 0 else convs[-1][1]
        if "stftsa" in exp_name or "defaultsa" in exp_name:
            specaug_version = exp_name.split(".")[1].replace("sa", "_")
            if "v" not in specaug_version:
                specaug_version = "stft_v1"
            model_config_exp.specaug_config = copy.deepcopy(specaug_configs[specaug_version])
        else:
            model_config_exp.specaug_config = copy.deepcopy(specaug_config)
            if exp_name.startswith(".sa"):
                model_config_exp.specaug_config.max_dim_feat = int(exp_name[3:].split(".")[0])
        if "init" not in exp_name:
            model_config_exp.feature_extraction_config.init_tf = None
            model_config_exp.feature_extraction_config.init_env = None
        run_with_standard_settings(
            network_module="ctc.conformer_0924.i6models_relposV1_VGGNLayerActFrontendV1_feat_v2",
            model_cfg=model_config_exp, name_ext=exp_name, move_to_hpc=True,
            forward_config={"batch_size": 16000 * 120}, prior_batch_size=120,
        )

    # 2D experiments with STFT SpecAugment: different 2D configurations
    from ...pytorch_networks.ctc.features.stft import (
        StftFeatureExtractionV1Config,
    )
    frontend_configs = {
        "2Dx6v1": VGGNLayerActFrontendV1Config(
            in_features=400 // 2 + 1,
            convs=[(32, (3, 3), (2, 1))] + [(64, (3, 3), (2, 1))] * 4 + [(32, (3, 3), (2, 1))],
            activations=["ReLU"] * 6,
            poolings=[None] * 6,
            out_features=512,
        ),
        "2Dx6v2": VGGNLayerActFrontendV1Config(
            in_features=400 // 2 + 1,
            convs=[(32, (3, 3), 1), (64, (3, 3), (2, 1))] + [(64, (3, 3), 1)] * 9 + [(32, (3, 3), 1)],
            activations=[None, "ReLU"] * 6,
            poolings=[None, None] + [None, ((2, 1), (2, 1), None)] * 5,
            out_features=512,
        ),
        "2Dx6v3": VGGNLayerActFrontendV1Config(
            in_features=400 // 2 + 1,
            convs=[(32, (3, 3), (2, 1))] + [(64, (3, 3), 1), (64, (3, 3), (2, 1))] * 5 + [(32, (3, 3), 1)],
            activations=[None, "ReLU"] * 6,
            poolings=[None] * 12,
            out_features=512,
        ),
        "2Dx6v4": VGGNLayerActFrontendV1Config(
            in_features=400 // 2 + 1,
            convs=[(32, (5, 5), (2, 1))] + [(64, (5, 5), (2, 1))] * 4 + [(32, (5, 5), (2, 1))],
            activations=["ReLU"] * 6,
            poolings=[None] * 6,
            out_features=512,
        ),
        "2Dx6v5": VGGNLayerActFrontendV1Config(
            in_features=400 // 2 + 1,
            convs=[(32, (3, 3), (2, 1))] + [(64, (3, 3), (2, 1))] * 4 + [(32, (3, 3), (2, 1))],
            activations=["ReLU_Dropout0.1"] * 6,
            poolings=[None] * 6,
            out_features=512,
        ),
        "2Dx6v6": VGGNLayerActFrontendV1Config(
            in_features=400 // 2 + 1,
            convs=[(32, (3, 3), (2, 1))] + [(64, (3, 3), (2, 1))] * 4 + [(32, (3, 3), (2, 1))],
            activations=["ReLU_Dropout0.3"] * 6,
            poolings=[None] * 6,
            out_features=512,
        ),
        "2Dx7v1": VGGNLayerActFrontendV1Config(
            in_features=400 // 2 + 1,
            convs=[(32, (3, 3), (2, 1))] + [(64, (3, 3), (2, 1))] * 5 + [(32, (3, 3), (2, 1))],
            activations=["ReLU"] * 7,
            poolings=[None] * 7,
            out_features=512,
        ),
        "2Dx5v1": VGGNLayerActFrontendV1Config(
            in_features=400 // 2 + 1,
            convs=[(32, (3, 3), (2, 1))] + [(64, (3, 3), (2, 1))] * 3 + [(32, (3, 3), (2, 1))],
            activations=["ReLU"] * 5,
            poolings=[None] * 5,
            out_features=512,
        ),
        "2Dx4v1": VGGNLayerActFrontendV1Config(
            in_features=400 // 2 + 1,
            convs=[(32, (3, 3), (2, 1))] + [(64, (3, 3), (2, 1))] * 2 + [(32, (3, 3), (2, 1))],
            activations=["ReLU"] * 4,
            poolings=[None] * 4,
            out_features=512,
        ),
        "2Dx3v1": VGGNLayerActFrontendV1Config(
            in_features=400 // 2 + 1,
            convs=[(32, (3, 3), (2, 1)), (64, (3, 3), (2, 1)), (32, (3, 3), (2, 1))],
            activations=["ReLU"] * 3,
            poolings=[None] * 3,
            out_features=512,
        ),
        "2Dx2v1": VGGNLayerActFrontendV1Config(
            in_features=400 // 2 + 1,
            convs=[(32, (3, 3), (2, 1))] + [(32, (3, 3), (2, 1))],
            activations=["ReLU"] * 2,
            poolings=[None] * 2,
            out_features=512,
        ),
        "2Dx2v2": VGGNLayerActFrontendV1Config(
            in_features=400 // 2 + 1,
            convs=[(32, (3, 3), 1), (64, (3, 3), 1), (64, (3, 3), 1), (32, (3, 3), 1)],
            activations=[None, "ReLU"] * 2,
            poolings=[None, ((2, 1), (2, 1), None)] * 2,
            out_features=512,
        ),
        "2Dx2v3": VGGNLayerActFrontendV1Config(
            in_features=400 // 2 + 1,
            convs=[(32, (3, 3), (2, 1))] + [(32, (3, 3), (2, 1))],
            activations=["ReLU", "ReLU_Log1p"],
            poolings=[None] * 2,
            out_features=512,
        ),
    }

    for exp_name, window_size, window_shift, n_fft in [
        (f".stftsa.2Dx6v1", 400, 10, None),
        (f".stftsa.2Dx6v1", 400, 10, 512),
        (f".stftsa.2Dx6v1", 256, 10, 256),
        # (f".stftsa.2Dx6v2", 400, 10, None),  # very slow (almost factor 3) and worse scores, aborted after 591 epochs
        (f".stftsa.2Dx5v1", 400, 20, None),
        (f".stftsa.2Dx2v1", 400, 160, None),
    ]:
        stft_config = StftFeatureExtractionV1Config(
            window_size=window_size,
            window_shift=window_shift,
            n_fft=n_fft,
            center=False,
            magnitude=True,
            module_class="StftFeatureExtractionV1",
        )
        fe_key = exp_name.split(".")[2]
        assert fe_key.startswith("2D")
        frontend_config = copy.deepcopy(frontend_configs[fe_key])
        frontend_config.in_features = (n_fft or window_size) // 2 + 1
        model_config = FeatureModelConfigV2(
            specaug_config=specaug_configs["stft_v1"],
            feature_extraction_config=stft_config,
            frontend_config=frontend_config,
            frontend_config_class="VGGNLayerActFrontendV1Config",
            **model_base_args_feat,
        )
        name_ext = f"{exp_name}.stft{window_size}x{window_shift}x{n_fft or window_size}"
        run_with_standard_settings(
            network_module="ctc.conformer_0924.i6models_relposV1_VGGNLayerActFrontendV1_feat_v2",
            model_cfg=model_config, name_ext=name_ext, train_rqmt={"mem_rqmt": 64}, move_to_hpc=True,
            forward_config={"batch_size": (16000 * 250 if exp_name == ".stftsa.2Dx2v1" else 16000 * 120)},
            prior_batch_size=140,
        )

    # 2D experiments: Tune STFT SpecAugment settings
    for exp_name, window_size, window_shift, n_fft, specaug_version in [
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v21"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v22"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v23"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v24"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v25"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v26"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v27"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v28"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v29"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v31"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v41"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v42"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v43"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v44"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v45"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v46"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v47"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v48"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v51"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v52"),
        (f".stftsa.2Dx2v1", 400, 160, None, "stft_v53"),
        (f".stftsa.2Dx2v2", 400, 160, None, "stft_v43"),
        (f".stftsa.2Dx6v1", 400, 10, None, "stft_v43"),
        (f".stftsa.2Dx5v1", 400, 20, None, "stft_v43"),
        (f".stftsa.2Dx6v1", 400, 10, None, "stft_v47"),
        (f".stftsa.2Dx5v1", 400, 20, None, "stft_v47"),
        (f".stftsa.2Dx4v1", 400, 40, None, "stft_v47"),
        (f".stftsa.2Dx3v1", 400, 80, None, "stft_v47"),
        (f".defaultsa.2Dx2v1", 400, 160, None, "default_v11"),
        (f".defaultsa.2Dx2v1", 400, 160, None, "default_v12"),
        (f".stftsa.2Dx2v1.nonorm", 400, 160, None, "stft_v47"),
        (f".stftsa.2Dx2v3", 400, 160, None, "stft_v47"),
    ]:
        stft_config = StftFeatureExtractionV1Config(
            window_size=window_size,
            window_shift=window_shift,
            n_fft=n_fft,
            center=False,
            magnitude=True,
            module_class="StftFeatureExtractionV1",
        )
        fe_key = exp_name.split(".")[2]
        assert fe_key.startswith("2D")
        frontend_config = copy.deepcopy(frontend_configs[fe_key])
        frontend_config.in_features = (n_fft or window_size) // 2 + 1
        model_config = FeatureModelConfigV2(
            specaug_config=specaug_configs[specaug_version],
            feature_extraction_config=stft_config,
            frontend_config=frontend_config,
            frontend_config_class="VGGNLayerActFrontendV1Config",
            **model_base_args_feat,
        )
        exp_name = exp_name.replace("stftsa", "stftsa" + specaug_version.split("_")[1])
        exp_name = exp_name.replace("defaultsa", "defaultsa" + specaug_version.split("_")[1])
        name_ext = f"{exp_name}.stft{window_size}x{window_shift}x{n_fft or window_size}"
        if "nonorm" in exp_name:
            run_with_standard_settings(
                network_module="ctc.conformer_0924.i6models_relposV1_VGGNLayerActFrontendV1_feat_v2",
                model_cfg=model_config, name_ext=name_ext, train_rqmt={"mem_rqmt": 64}, move_to_hpc=True,
                forward_config={"batch_size": (16000 * 250 if exp_name == ".stftsa.2Dx2v1" else 16000 * 120)},
                prior_batch_size=140, train_data_custom=train_data_nonorm,
            )
        else:
            run_with_standard_settings(
                network_module="ctc.conformer_0924.i6models_relposV1_VGGNLayerActFrontendV1_feat_v2",
                model_cfg=model_config, name_ext=name_ext, train_rqmt={"mem_rqmt": 64}, move_to_hpc=True,
                forward_config={"batch_size": (16000 * 250 if exp_name == ".stftsa.2Dx2v1" else 16000 * 120)},
                prior_batch_size=140,
            )

    # 2D experiments with STFT SpecAugment: Replace STFT by conv layer
    from ...pytorch_networks.ctc.features.conv import (
        ConvFeatureExtractionV1Config, ConvFeatureExtractionV2Config
    )
    for fe_key, specaug_version, out_channels, kernel_size, stride, freeze, init, activation in [
        ("2Dx6v1", "stft_v22", 80, 256, 10, False, "gammatone", None),
        ("2Dx6v1", "stft_v22", 80, 256, 10, False, None, None),
        ("2Dx6v1", "stft_v47", 80, 256, 10, True, "gammatone", None),
        ("2Dx6v1", "stft_v47", 80, 256, 10, False, "gammatone", None),
        ("2Dx6v1", "stft_v47", 80, 256, 10, False, None, None),
        ("2Dx5v1", "stft_v47", 80, 256, 20, False, None, None),
        ("2Dx4v1", "stft_v47", 80, 256, 40, False, None, None),
        ("2Dx3v1", "stft_v47", 80, 256, 80, False, None, None),
        ("2Dx2v1", "stft_v47", 80, 256, 160, False, None, None),
        ("2Dx7v1", "stft_v47", 80, 256, 5, False, None, None),
        ("2Dx6v1", "stft_v47", 80, 400, 10, False, None, None),
        ("2Dx6v1", "stft_v47", 80, 64, 10, False, None, None),
        ("2Dx6v1", "stft_v47", 80, 16, 10, False, None, None),
        ("2Dx6v3", "stft_v47", 80, 256, 10, False, None, None),
        ("2Dx6v4", "stft_v47", 80, 256, 10, False, None, None),
        ("2Dx6v5", "stft_v47", 80, 256, 10, False, None, None),
        ("2Dx6v6", "stft_v47", 80, 256, 10, False, None, None),
    ]:
        if freeze:
            conv_config = ConvFeatureExtractionV2Config(
                wave_norm=True,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
                freeze=freeze,
                init=init,
                activation=activation,
                module_class="ConvFeatureExtractionV2",
            )
        else:
            conv_config = ConvFeatureExtractionV1Config(
                wave_norm=True,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
                init=init,
                activation=activation,
                module_class="ConvFeatureExtractionV1",
            )
        frontend_config = copy.deepcopy(frontend_configs[fe_key])
        frontend_config.in_features = out_channels
        model_config = FeatureModelConfigV2(
            specaug_config=specaug_configs[specaug_version],
            feature_extraction_config=conv_config,
            frontend_config=frontend_config,
            frontend_config_class="VGGNLayerActFrontendV1Config",
            **model_base_args_feat,
        )
        exp_name = ".stftsa" + specaug_version.split("_")[1] + f".{fe_key}.conv{out_channels}x{kernel_size}x{stride}"
        exp_name = (
            exp_name +
            (f"_{activation}" if activation else "") +
            (f"_{init}" if init else "") +
            ("_freeze" if freeze else "")
        )
        run_with_standard_settings(
            network_module="ctc.conformer_0924.i6models_relposV1_VGGNLayerActFrontendV1_feat_v2",
            model_cfg=model_config, name_ext=exp_name, train_rqmt={"mem_rqmt": 64}, move_to_hpc=True,
            forward_config={"batch_size": 16000 * 120}, prior_batch_size=140,
        )

    tk.register_report(
        os.path.join(prefix_name, "report.csv"),
        values=report.get_values(),
        template=report.get_template(),
    )
