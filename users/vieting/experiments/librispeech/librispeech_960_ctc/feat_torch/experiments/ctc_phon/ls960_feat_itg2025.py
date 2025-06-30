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
    prefix_name = "experiments/librispeech/librispeech_960_ctc_eow_phon/feat_torch/itg_2025"

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

        training_name = prefix_name + "/" + name_ext
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
        add_ctc_model(network_module + ".eow_phon" + name_ext, asr_model)

    # baseline log Mel setup
    logmel_config = LogMelFeatureExtractionV1Config(
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
    frontend_logmel_config = VGG4LayerActFrontendV1Config_mod(
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

    model_logmel_config = ModelConfig(
        feature_extraction_config=logmel_config,
        frontend_config=frontend_logmel_config,
        specaug_config=specaug_config,
        **model_base_args,
    )
    run_with_standard_settings(
        network_module="ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1", model_cfg=model_logmel_config,
        name_ext="logmel", move_to_hpc=True,
    )

    # SCF experiments
    from ...pytorch_networks.ctc.conformer_0924.i6models_relposV1_VGGNLayerActFrontendV1_feat_v2_cfg import (
        ModelConfig as FeatureModelConfigV2,
        SpecaugStftConfig,
        VGGNLayerActFrontendV1Config,
        VGGNLayerActFrontendV2Config,
        LinearConfig,
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
        init_tf=None,
        init_env=None,
        init_convs="ones",
    )
    model_base_args_feat = copy.deepcopy(model_base_args)
    model_base_args_feat["specaug_start_epoch"] = model_base_args_feat.pop("specauc_start_epoch")

    specaug_stft_config = SpecaugStftConfig(
        repeat_per_n_frames=25,
        max_dim_time=55,
        max_dim_feat=int(16 / 80 * 201 * 1.3),
        num_repeat_feat=5,
        window_size=400,
        window_shift=160,
        fft_size=512,
    )
    frontend_scf_config = VGGNLayerActFrontendV1Config(
        in_features=750,
        convs=[(32, (3, 3), 1), (64, (3, 3), 1), (64, (3, 3), 1), (32, (3, 3), 1)],
        activations=[None, "ReLU", None, "ReLU"],
        poolings=[None, ((2, 1), (2, 1), None), None, ((2, 1), (2, 1), None)],
        out_features=512,
    )

    for exp_name, specaug_config_exp in [
        ("scf.vanilla_specaug", specaug_config),
        ("scf", specaug_stft_config),
    ]:
        model_config_exp = FeatureModelConfigV2(
            specaug_config=copy.deepcopy(specaug_config_exp),
            feature_extraction_config=scf_config,
            frontend_config=frontend_scf_config,
            frontend_config_class="VGGNLayerActFrontendV1Config",
            **model_base_args_feat,
        )
        run_with_standard_settings(
            network_module="ctc.conformer_0924.i6models_relposV1_VGGNLayerActFrontendV1_feat_v2",
            model_cfg=model_config_exp, name_ext=exp_name, move_to_hpc=True,
            forward_config={"batch_size": 16000 * 120}, prior_batch_size=120,
        )

    # wav2vec feature extractor
    from ...pytorch_networks.ctc.features.wav2vec import (
        Wav2vecFeatureExtractionV1Config
    )
    w2v_config = Wav2vecFeatureExtractionV1Config(
        conv_layers=[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] * 3,
        module_class="Wav2vecFeatureExtractionV1",
    )
    model_config = FeatureModelConfigV2(
        specaug_config=specaug_stft_config,
        feature_extraction_config=w2v_config,
        frontend_config=LinearConfig(in_features=512, out_features=512),
        frontend_config_class="LinearConfig",
        **model_base_args_feat,
    )
    run_with_standard_settings(
        network_module="ctc.conformer_0924.i6models_relposV1_VGGNLayerActFrontendV1_feat_v2",
        model_cfg=model_config, name_ext="w2v_fe", train_rqmt={"mem_rqmt": 64}, move_to_hpc=True,
        forward_config={"batch_size": 16000 * 120}, prior_batch_size=140,
    )

    # Generic learnable 2D convolutional features
    from ...pytorch_networks.ctc.features.stft import (
        StftFeatureExtractionV1Config, StftFeatureExtractionV2Config,
    )
    from ...pytorch_networks.ctc.features.conv import (
        ConvFeatureExtractionV1Config, ConvFeatureExtractionV2Config
    )

    frontend_2d_configs = {  # keys are (subsampling factor, num layers) w.r.t. this 2d frontend
        (64, 6): VGGNLayerActFrontendV1Config(
            in_features=0,  # needs to be overwritten below
            convs=[
                (32, (3, 3), (2, 1)),
                (64, (3, 3), (2, 1)),
                (64, (3, 3), (2, 1)),
                (64, (3, 3), (2, 1)),
                (64, (3, 3), (2, 1)),
                (32, (3, 3), (2, 1)),
            ],
            activations=["ReLU"] * 6,
            poolings=[None] * 6,
            out_features=512,
        ),
        (64, 8): VGGNLayerActFrontendV1Config(
            in_features=0,  # needs to be overwritten below
            convs=[
                (32, (3, 3), (2, 1)),
                (64, (3, 3), (2, 1)),
                (64, (3, 3), (2, 1)), (64, (3, 3), 1),
                (64, (3, 3), (2, 1)),
                (64, (3, 3), (2, 1)),
                (64, (3, 3), (2, 1)), (32, (3, 3), 1),
            ],
            activations=["ReLU", "ReLU", None, "ReLU"] * 2,
            poolings=[None] * 8,
            out_features=512,
        ),
        (64, 10): VGGNLayerActFrontendV1Config(
            in_features=0,  # needs to be overwritten below
            convs=[
                (32, (3, 3), (2, 1)),
                (64, (3, 3), (2, 1)), (64, (3, 3), 1),
                (64, (3, 3), (2, 1)), (64, (3, 3), 1),
                (64, (3, 3), (2, 1)),
                (64, (3, 3), (2, 1)), (64, (3, 3), 1),
                (64, (3, 3), (2, 1)), (32, (3, 3), 1),
            ],
            activations=["ReLU", None, "ReLU", None, "ReLU"] * 2,
            poolings=[None] * 10,
            out_features=512,
        ),
        (16, 4): VGGNLayerActFrontendV1Config(
            in_features=0,  # needs to be overwritten below
            convs=[
                (32, (3, 3), (2, 1)),
                (64, (3, 3), (2, 1)),
                (64, (3, 3), (2, 1)),
                (32, (3, 3), (2, 1)),
            ],
            activations=["ReLU"] * 4,
            poolings=[None] * 4,
            out_features=512,
        ),
        (16, 6): VGGNLayerActFrontendV1Config(
            in_features=0,  # needs to be overwritten below
            convs=[
                (32, (3, 3), (2, 1)),
                (64, (3, 3), (2, 1)), (64, (3, 3), 1),
                (64, (3, 3), (2, 1)),
                (64, (3, 3), (2, 1)), (32, (3, 3), 1),
            ],
            activations=["ReLU", None, "ReLU"] * 2,
            poolings=[None] * 6,
            out_features=512,
        ),
        (16, 8): VGGNLayerActFrontendV1Config(
            in_features=0,  # needs to be overwritten below
            convs=[
                (32, (3, 3), (2, 1)), (64, (3, 3), 1),
                (64, (3, 3), (2, 1)), (64, (3, 3), 1),
                (64, (3, 3), (2, 1)), (64, (3, 3), 1),
                (64, (3, 3), (2, 1)), (32, (3, 3), 1),
            ],
            activations=[None, "ReLU"] * 4,
            poolings=[None] * 8,
            out_features=512,
        ),
        (16, 10): VGGNLayerActFrontendV1Config(
            in_features=0,  # needs to be overwritten below
            convs=[
                (32, (3, 3), (2, 1)), (64, (3, 3), 1),
                (64, (3, 3), (2, 1)), (64, (3, 3), 1), (64, (3, 3), 1),
                (64, (3, 3), (2, 1)), (64, (3, 3), 1),
                (64, (3, 3), (2, 1)), (64, (3, 3), 1), (32, (3, 3), 1),
            ],
            activations=[None, "ReLU", None, None, "ReLU"] * 2,
            poolings=[None] * 10,
            out_features=512,
        ),
        (4, 2): VGGNLayerActFrontendV1Config(
            in_features=0,  # needs to be overwritten below
            convs=[
                (32, (3, 3), (2, 1)),
                (32, (3, 3), (2, 1)),
            ],
            activations=["ReLU"] * 2,
            poolings=[None] * 2,
            out_features=512,
        ),
        (4, 4): VGGNLayerActFrontendV1Config(
            in_features=0,  # needs to be overwritten below
            convs=[
                (32, (3, 3), (2, 1)), (64, (3, 3), 1),
                (64, (3, 3), (2, 1)), (32, (3, 3), 1),
            ],
            activations=[None, "ReLU"] * 2,
            poolings=[None] * 4,
            out_features=512,
        ),
        (4, 6): VGGNLayerActFrontendV1Config(
            in_features=0,  # needs to be overwritten below
            convs=[
                (32, (3, 3), (2, 1)), (64, (3, 3), 1), (64, (3, 3), 1),
                (64, (3, 3), (2, 1)), (64, (3, 3), 1), (32, (3, 3), 1),
            ],
            activations=[None, "ReLU"] * 3,
            poolings=[None] * 6,
            out_features=512,
        ),
        (4, 8): VGGNLayerActFrontendV1Config(
            in_features=0,  # needs to be overwritten below
            convs=[
                (32, (3, 3), (2, 1)), (64, (3, 3), 1), (64, (3, 3), 1), (64, (3, 3), 1),
                (64, (3, 3), (2, 1)), (64, (3, 3), 1), (64, (3, 3), 1), (32, (3, 3), 1),
            ],
            activations=[None, "ReLU"] * 4,
            poolings=[None] * 8,
            out_features=512,
        ),
        (4, 10): VGGNLayerActFrontendV1Config(
            in_features=0,  # needs to be overwritten below
            convs=[
                (32, (3, 3), (2, 1)), (64, (3, 3), 1), (64, (3, 3), 1), (64, (3, 3), 1), (64, (3, 3), 1),
                (64, (3, 3), (2, 1)), (64, (3, 3), 1), (64, (3, 3), 1), (64, (3, 3), 1), (32, (3, 3), 1),
            ],
            activations=[None, "ReLU"] * 5,
            poolings=[None] * 10,
            out_features=512,
        ),
    }

    # 2D experiments: filterbank first layer
    for out_channels, kernel_size, stride, num_2d_layers, freeze, init in [
        # initialization and freezing (Table 2)
        (80, 256, 10, 6, True, "gammatone"),
        (80, 256, 10, 6, False, "gammatone"),
        (80, 256, 10, 6, False, None),
        # subsampling in the first layer and number of 2D layers
        # (80, 256, 10, 6, False, None),  # see above
        (80, 256, 10, 8, False, None),
        (80, 256, 10, 10, False, None),
        (80, 256, 40, 4, False, None),
        (80, 256, 40, 6, False, None),
        (80, 256, 40, 8, False, None),
        (80, 256, 40, 10, False, None),
        (80, 256, 160, 2, False, None),
        (80, 256, 160, 4, False, None),
        (80, 256, 160, 6, False, None),
        (80, 256, 160, 8, False, None),
        (80, 256, 160, 10, False, None),
        # different kernel sizes (Figure 3)
        (80, 400, 10, 6, False, None),
        (80, 128, 10, 6, False, None),
        (80, 64, 10, 6, False, None),
        (80, 32, 10, 6, False, None),
        (80, 28, 10, 6, False, None),
        (80, 24, 10, 6, False, None),
        (80, 20, 10, 6, False, None),
        (80, 16, 10, 6, False, None),
        # different numbers of channels (Figure 3)
        (8, 256, 10, 6, False, None),  # parameter-efficient configuration in Table 1
        (16, 256, 10, 6, False, None),
        (32, 256, 10, 6, False, None),
        (64, 256, 10, 6, False, None),
        # (80, 256, 10, 6, False, None),  # see above
        (128, 256, 10, 6, False, None),  # better-performing configuration in Table 1
        (256, 256, 10, 6, False, None),
    ]:
        if freeze:
            conv_config_exp = ConvFeatureExtractionV2Config(
                wave_norm=True,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
                freeze=freeze,
                init=init,
                activation=None,
                module_class="ConvFeatureExtractionV2",
            )
        else:
            conv_config_exp = ConvFeatureExtractionV1Config(
                wave_norm=True,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=False,
                init=init,
                activation=None,
                module_class="ConvFeatureExtractionV1",
            )
        frontend_config_exp = copy.deepcopy(frontend_2d_configs[(640 / stride, num_2d_layers)])
        frontend_config_exp.in_features = out_channels
        model_config = FeatureModelConfigV2(
            specaug_config=specaug_stft_config,
            feature_extraction_config=conv_config_exp,
            frontend_config=frontend_config_exp,
            frontend_config_class="VGGNLayerActFrontendV1Config",
            **model_base_args_feat,
        )
        exp_name = f"2D.conv{out_channels}x{kernel_size}x{stride}"
        exp_name = (
            exp_name +
            (f"_{init}" if init else "") +
            ("_freeze" if freeze else "")
        )
        exp_name += f".2Dx{num_2d_layers}"
        run_with_standard_settings(
            network_module="ctc.conformer_0924.i6models_relposV1_VGGNLayerActFrontendV1_feat_v2",
            model_cfg=model_config, name_ext=exp_name, train_rqmt={"mem_rqmt": 64}, move_to_hpc=True,
            forward_config={"batch_size": 16000 * 120}, prior_batch_size=140,
        )

    # 2D experiments: STFT magnitude as the first layer
    for window_size, window_shift, n_fft, num_2d_layers in [
        (400, 10, None, 6),
    ]:
        stft_config_exp = StftFeatureExtractionV1Config(
            window_size=window_size,
            window_shift=window_shift,
            n_fft=n_fft,
            center=False,
            magnitude=True,
            module_class="StftFeatureExtractionV1",
        )
        frontend_config_exp = copy.deepcopy(frontend_2d_configs[(640 / window_shift, num_2d_layers)])
        frontend_config_exp.in_features = (n_fft or window_size) // 2 + 1
        model_config = FeatureModelConfigV2(
            specaug_config=specaug_stft_config,
            feature_extraction_config=stft_config_exp,
            frontend_config=frontend_config_exp,
            frontend_config_class="VGGNLayerActFrontendV1Config",
            **model_base_args_feat,
        )
        name_ext = f"2D.stft{window_size}x{window_shift}x{n_fft or window_size}mag.2Dx{num_2d_layers}"
        run_with_standard_settings(
            network_module="ctc.conformer_0924.i6models_relposV1_VGGNLayerActFrontendV1_feat_v2",
            model_cfg=model_config, name_ext=name_ext, train_rqmt={"mem_rqmt": 64}, move_to_hpc=True,
            forward_config={"batch_size": (16000 * 120)}, prior_batch_size=100,
        )

    # 2D experiments: STFT Re + Im as the first layer
    for window_size, window_shift, n_fft in [
        (400, 10, None),
    ]:
        re_im_proc_config = VGGNLayerActFrontendV2Config(
            in_features=(n_fft or window_size) // 2 + 1,
            convs=[(32, (3, 3), (2, 1))],
            activations=["ReLU"],
            poolings=[None],
            out_features=512,
            in_channels=1,
            project_out=False,
        )
        stft_config_exp = StftFeatureExtractionV2Config(
            window_size=window_size,
            window_shift=window_shift,
            n_fft=n_fft,
            center=False,
            module_class="StftFeatureExtractionV2",
            proc_config=re_im_proc_config,
            proc_module="VGGNLayerActFrontendV2",
        )
        n_fe_layers = int(np.log2(640 / window_shift)) - 1
        frontend_config_exp = VGGNLayerActFrontendV2Config(
            in_features=(n_fft or window_size) // 2 + 1,
            convs=[(64, (3, 3), (2, 1))] * (n_fe_layers - 1) + [(32, (3, 3), (2, 1))],
            activations=["ReLU"] * n_fe_layers,
            poolings=[None] * n_fe_layers,
            out_features=512,
            in_channels=32,
            project_out=True,
        )
        model_config_exp = FeatureModelConfigV2(
            specaug_config=specaug_stft_config,
            feature_extraction_config=stft_config_exp,
            frontend_config=frontend_config_exp,
            frontend_config_class="VGGNLayerActFrontendV2Config",
            **model_base_args_feat,
        )
        name_ext = f"2D.stft{window_size}x{window_shift}x{n_fft or window_size}reim.2Dx{n_fe_layers + 1}"
        run_with_standard_settings(
            network_module="ctc.conformer_0924.i6models_relposV1_VGGNLayerActFrontendV1_feat_v2",
            model_cfg=model_config_exp, name_ext=name_ext, train_rqmt={"mem_rqmt": 64}, move_to_hpc=True,
            forward_config={"batch_size": (16000 * 120)}, prior_batch_size=140,
        )

    tk.register_report(
        os.path.join(prefix_name, "report.csv"),
        values=report.get_values(),
        template=report.get_template(),
    )
