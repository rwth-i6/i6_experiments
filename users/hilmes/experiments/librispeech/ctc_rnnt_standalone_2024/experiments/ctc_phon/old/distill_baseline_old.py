import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from sisyphus import tk
from functools import partial
from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ....data.common import DatasetSettings, build_test_dataset
from ....data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ....default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ....lm import get_4gram_binary_lm
from ....pipeline import training, prepare_asr_model, calculate_blank_counts
from ..tune_eval import build_base_report, eval_model, build_hubert_distill_report
from ....report import generate_report


def eow_phon_ls960_distill_base_old():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_eow_phon/baselines_old"

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

    from ....pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )
    from ....pytorch_networks.ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1_cfg import (
        LogMelFeatureExtractionV1Config,
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

    from ....pytorch_networks.ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1_cfg import (
        ModelConfig as RelPosModelConfigV1,
        ConformerPosEmbConfig,
        VGG4LayerActFrontendV1Config_mod,
        SpecaugConfig,
    )

    report = {}
    for dim in [384, 512]:
        for spec_start in [1]:
            for epochs in [500, 1000]:
                for spec in [16]:
                    for num_heads in [8]:
                        if dim == 512 and num_heads == 12:
                            continue
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
                        specaug_config_test = SpecaugConfig(
                            repeat_per_n_frames=25,
                            max_dim_time=20,
                            max_dim_feat=spec,
                            num_repeat_feat=5,
                        )
                        pos_emb_cfg = ConformerPosEmbConfig(
                            learnable_pos_emb=False,
                            rel_pos_clip=16,
                            with_linear_pos=True,
                            with_pos_bias=True,
                            separate_pos_emb_per_head=True,
                            pos_emb_dropout=0.0,
                        )
                        model_config_pos_enc = RelPosModelConfigV1(
                            feature_extraction_config=fe_config,
                            frontend_config=frontend_config,
                            specaug_config=specaug_config_test,
                            label_target_size=vocab_size_without_blank,
                            pos_emb_config=pos_emb_cfg,
                            conformer_size=dim,
                            num_layers=12,
                            num_heads=num_heads,
                            ff_dim=4 * dim,
                            att_weights_dropout=0.1,
                            conv_dropout=0.1,
                            ff_dropout=0.1,
                            mhsa_dropout=0.1,
                            mhsa_with_bias=True,
                            conv_kernel_size=31,
                            final_dropout=0.1,
                            dropout_broadcast_axes=None,
                            specauc_start_epoch=spec_start,
                            module_list=["ff", "conv", "mhsa", "ff"],
                            module_scales=[0.5, 1.0, 1.0, 0.5],
                            aux_ctc_loss_layers=None,
                            aux_ctc_loss_scales=None,
                        )
                        network_module_pos_enc = (
                            "ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"
                        )
                        train_config = {
                            "optimizer": {
                                "class": "radam",
                                "epsilon": 1e-16,
                                "weight_decay": 1e-2,
                                "decoupled_weight_decay": True,
                            },
                            "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))  # try higher start
                            + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
                            + list(np.linspace(5e-5, 1e-7, 30)),
                            #############
                            "batch_size": 180 * 16000,
                            "max_seq_length": {"audio_features": 35 * 16000},
                            "accum_grad_multiple_step": 1,
                        }
                        train_args = {
                            "config": train_config,
                            "network_module": network_module_pos_enc,
                            "net_args": {"model_config_dict": asdict(model_config_pos_enc)},
                            "debug": True,
                        }
                        results = {}
                        training_name = (
                            prefix_name
                            + "/"
                            + network_module_pos_enc
                            + f"_{epochs}_{dim}_{num_heads}_{spec}_{spec_start}"
                        )
                        train_job = training(
                            training_name, train_data, train_args, num_epochs=epochs, **default_returnn
                        )

                        results = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args,
                            train_data=train_data,
                            decoder_config=default_decoder_config,
                            dev_dataset_tuples=dev_dataset_tuples,
                            result_dict=results,
                            loss_name=f"ctc_loss_layer12",
                            specific_epoch=epochs,
                            lm_scales=[2.2, 2.4, 2.6, 2.8, 3.0],
                            prior_scales=[0.1, 0.2, 0.3, 0.5],
                            run_test=True,
                            test_dataset_tuples=test_dataset_tuples,
                        )
                        generate_report(results=results, exp_name=training_name)
                        report[training_name] = results
                        del results
    tk.register_report(
        "reports/ls_baselines_report", partial(build_base_report, report), required=report, update_frequency=900
    )

    from ....pytorch_networks.ctc.conformer_distill_1007.i6modelsV2_VGG4LayerActFrontendV1_auxloss_v1_cfg import (
        ModelConfig as ModelConfigAux,
    )

    for dim in [384]:
        for spec_start in [1]:
            for epochs in [500, 1000]:
                for spec in [8]:
                    for num_heads in [4, 8]:
                        for drop in [0.2, 0.1, 0.0]:
                            if dim == 512 and num_heads == 12:
                                continue
                            if num_heads > 4 and not drop == 0.1:
                                continue
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
                            specaug_config_test = SpecaugConfig(
                                repeat_per_n_frames=25,
                                max_dim_time=20,
                                max_dim_feat=spec,
                                num_repeat_feat=5,
                            )
                            model_config_aux = ModelConfigAux(
                                feature_extraction_config=fe_config,
                                frontend_config=frontend_config,
                                specaug_config=specaug_config_test,
                                label_target_size=vocab_size_without_blank,
                                conformer_size=dim,
                                num_layers=12,
                                num_heads=num_heads,
                                ff_dim=4 * dim,
                                att_weights_dropout=drop,
                                conv_dropout=drop,
                                ff_dropout=drop,
                                mhsa_dropout=drop,
                                conv_kernel_size=31,
                                final_dropout=drop,
                                specauc_start_epoch=spec_start,
                                module_list=["ff", "conv", "mhsa", "ff"],
                                module_scales=[0.5, 1.0, 1.0, 0.5],
                                aux_ctc_loss_layers=[3, 7, 11],  # 4, 8, 12 when counting from 1
                                aux_ctc_loss_scales=[0.3, 0.3, 1.0],
                            )
                            model_config_decoding = copy.deepcopy(model_config_aux)
                            model_config_decoding.aux_ctc_loss_scales = [
                                0.0,
                                0.0,
                                1.0,
                            ]  # for decoding use result only of last layer
                            network_module_aux = (
                                "ctc.conformer_distill_1007.i6modelsV2_VGG4LayerActFrontendV1_auxloss_v1"
                            )
                            train_config = {
                                "optimizer": {
                                    "class": "radam",
                                    "epsilon": 1e-16,
                                    "weight_decay": 1e-2,
                                    "decoupled_weight_decay": True,
                                },
                                "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))  # try higher start
                                + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
                                + list(np.linspace(5e-5, 1e-7, 30)),
                                #############
                                "batch_size": 300 * 16000,
                                "max_seq_length": {"audio_features": 35 * 16000},
                                "accum_grad_multiple_step": 1,
                                "torch_amp_options": {"dtype": "bfloat16"},
                                "gradient_clip_norm": 1.0,
                            }
                            # batch size, adamw, speed pert, gradient clip,
                            train_args = {
                                "config": train_config,
                                "network_module": network_module_aux,
                                "net_args": {"model_config_dict": asdict(model_config_aux)},
                                "debug": False,
                                "use_speed_perturbation": True,
                            }
                            train_args_decoding = copy.deepcopy(train_args)
                            train_args_decoding["net_args"] = {"model_config_dict": asdict(model_config_decoding)}
                            results = {}
                            training_name = (
                                prefix_name
                                + "/"
                                + network_module_aux
                                + f"_baseline_{epochs}_{dim}_{num_heads}_{spec}_{spec_start}_drop{drop}"
                            )
                            train_job = training(
                                training_name, train_data, train_args, num_epochs=epochs, **default_returnn
                            )
                            train_job.rqmt["gpu_mem"] = 48
                            results = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args_decoding,
                                train_data=train_data,
                                decoder_config=default_decoder_config,
                                dev_dataset_tuples=dev_dataset_tuples,
                                result_dict=results,
                                loss_name=f"ctc_loss_layer12",
                                specific_epoch=epochs,
                                lm_scales=[2.0, 2.2, 2.4, 2.6],
                                prior_scales=[0.1, 0.2, 0.3],
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                            )
                            generate_report(results=results, exp_name=training_name)
                            report[training_name] = results
                            del results
