from dataclasses import asdict
import numpy as np
from typing import cast
import copy
from sisyphus import tk

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream
from i6_experiments.users.hilmes.experiments.tedlium2.standalone.experiments.ctc_phon.tune_eval import QuantArgs
from i6_experiments.users.hilmes.experiments.tedlium2.standalone.data.common import (
    DatasetSettings,
    build_test_dataset,
    build_st_test_dataset,
)
from i6_experiments.users.hilmes.experiments.tedlium2.standalone.data.phon import (
    build_eow_phon_training_datasets,
    get_text_lexicon,
)
from i6_experiments.users.hilmes.experiments.tedlium2.standalone.default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from i6_experiments.users.hilmes.experiments.tedlium2.standalone.lm import get_4gram_binary_lm
from i6_experiments.users.hilmes.experiments.tedlium2.standalone.pipeline import training, prepare_asr_model
from i6_experiments.users.hilmes.experiments.tedlium2.standalone.report import generate_report
from i6_experiments.users.hilmes.experiments.tedlium2.standalone.experiments.ctc_phon.tune_eval import (
    tune_and_evaluate_helper,
)


def eow_phon_ted_distill_hubert():
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/distill_hubert"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data = build_eow_phon_training_datasets(
        prefix=prefix_name,
        settings=train_settings,
    )
    label_datastream = cast(LabelDatastream, train_data.datastreams["labels"])
    vocab_size_without_blank = label_datastream.vocab_size

    dev_dataset_tuples = {}
    for testset in ["dev"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    test_dataset_tuples = {}
    for testset in ["test"]:
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

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    from ...pytorch_networks.ctc.conformer_distill_1206.conformerv2_distill_v1_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
        LogMelFeatureExtractionV1Config,
        HubertTeacherConfig,
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
    for scale in [0.0, 0.1, 1.0, 10.0, 1000]:
        # 1000: 10.4
        for dim in [384]:
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
                out_features=dim,
                activation=None,
            )

            model_config = ModelConfig(
                feature_extraction_config=fe_config,
                frontend_config=frontend_config,
                specaug_config=specaug_config,
                label_target_size=vocab_size_without_blank,
                conformer_size=dim,
                num_layers=12,
                num_heads=4,
                ff_dim=4 * dim,
                att_weights_dropout=0.2,
                conv_dropout=0.2,
                ff_dropout=0.2,
                mhsa_dropout=0.2,
                conv_kernel_size=31,
                final_dropout=0.2,
                specauc_start_epoch=1,
                module_list=["ff", "mhsa", "conv", "ff"],
                module_scales=[0.5, 1.0, 1.0, 0.5],
                aux_ctc_loss_layers=[3, 7, 11],  # 4, 8, 12 when counting from 1
                aux_ctc_loss_scales=[0.2, 0.3, 1.0],
            )
            teacher_config = HubertTeacherConfig(
                model_name="base-ls960",
                distill_scale=scale,
            )
            model_config_decoding = copy.deepcopy(model_config)
            model_config_decoding.aux_ctc_loss_scales = [0.0, 0.0, 1.0]  # for decoding use result only of last layer
            network_module = "ctc.conformer_distill_1206.conformerv2_distill_v1"

            train_config_24gbgpu = {
                "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
                "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
                + list(np.linspace(5e-4, 5e-5, 110))
                + list(np.linspace(5e-5, 1e-7, 30)),
                #############
                "batch_size": 120 * 16000 if not dim == 768 else 90 * 16000,
                "max_seq_length": {"audio_features": 35 * 16000},
                "accum_grad_multiple_step": 3 if not dim == 768 else 4,
            }
            train_args = {
                "config": train_config_24gbgpu,
                "network_module": network_module,
                "net_args": {"model_config_dict": asdict(model_config), "hubert_config_dict": asdict(teacher_config)},
                "debug": False,
            }
            train_args_decoding = copy.deepcopy(train_args)
            train_args_decoding["net_args"] = {
                "model_config_dict": asdict(model_config_decoding),
                "hubert_config_dict": asdict(teacher_config),
            }
            results = {}
            training_name = prefix_name + "/" + network_module + f"_{dim}_scale_{scale}"
            train_job = training(training_name, train_data, train_args, num_epochs=250, **default_returnn)
            train_job.rqmt["gpu_mem"] = 24
            asr_model = prepare_asr_model(
                training_name,
                train_job,
                train_args_decoding,
                with_prior=True,
                datasets=train_data,
                get_specific_checkpoint=250,
            )
            lm_scales = [2.0, 2.2, 2.4, 2.6, 2.8]
            prior_scales = [0.7, 0.9]
            res, _ = tune_and_evaluate_helper(
                training_name,
                asr_model,
                default_decoder_config,
                lm_scales=lm_scales,
                prior_scales=prior_scales,
                dev_dataset_tuples=dev_dataset_tuples,
                decoder_module="ctc.decoder.flashlight_ctc_distill_v1",
            )
            results.update(res)
            asr_model_best4 = prepare_asr_model(
                training_name + "/best4",
                train_job,
                train_args_decoding,
                with_prior=True,
                datasets=train_data,
                get_best_averaged_checkpoint=(4, "ctc_loss_layer12"),
            )
            res, _ = tune_and_evaluate_helper(
                training_name + "/best4",
                asr_model_best4,
                default_decoder_config,
                lm_scales=lm_scales,
                prior_scales=prior_scales,
                dev_dataset_tuples=dev_dataset_tuples,
                decoder_module="ctc.decoder.flashlight_ctc_distill_v1",
            )
            results.update(res)
            asr_model_best = prepare_asr_model(
                training_name + "/best",
                train_job,
                train_args_decoding,
                with_prior=True,
                datasets=train_data,
                get_best_averaged_checkpoint=(1, "ctc_loss_layer12"),
            )
            res, _ = tune_and_evaluate_helper(
                training_name + "/best",
                asr_model_best,
                default_decoder_config,
                lm_scales=lm_scales,
                prior_scales=prior_scales,
                dev_dataset_tuples=dev_dataset_tuples,
                decoder_module="ctc.decoder.flashlight_ctc_distill_v1",
            )
            results.update(res)
            generate_report(results=results, exp_name=training_name)
            del results
