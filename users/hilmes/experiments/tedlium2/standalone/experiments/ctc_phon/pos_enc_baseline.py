import os.path
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.phon import build_eow_phon_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, calculate_blank_counts
from ...report import generate_report
from functools import partial
from sisyphus import tk
from .tune_eval import eval_model, build_base_report, RTFArgs


def eow_phon_ted_pos_enc_baseline(get_report=False):
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/ctc_eow_phon/pos_enc_baseline"

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
    train_dataset_tuples = {}
    for testset in ["train"]:
        train_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    test_dataset_tuples = {}
    for testset in ["test"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings,
        )

    train_settings_4000 = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.4000",
    )

    train_data_4000 = build_eow_phon_training_datasets(
        prefix=prefix_name,
        settings=train_settings_4000,
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
    from ...pytorch_networks.ctc.conformer_0106.i6modelsV2_VGG4LayerActFrontendV1_auxloss_v1_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
        ModelConfig,
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
    from ...pytorch_networks.ctc.conformer_0106.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1_cfg import (
        ModelConfig as RelPosModelConfig,
        ConformerPosEmbConfig,
    )

    network_module_pos_enc_v1 = "ctc.conformer_0106.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"
    RTFS = [
        # prefix_name + "/" + network_module_pos_enc_v1 + f"_500_384_4_8_1",
        # prefix_name + "/" + network_module_pos_enc_v1 + f"_500_384_4_16_1",
        # prefix_name + "/" + network_module_pos_enc_v1 + f"_500_384_8_8_1",
        prefix_name
        + "/"
        + network_module_pos_enc_v1
        + f"_500_384_8_16_1",
    ]
    prior_scales = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    lm_scales = [1.2, 1.3, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
    chkpts = {}
    new_rep = {}
    # Best: 384, 1, 500, 16, 8
    for dim in [384]:
        for spec_start in [1]:
            for epochs in [500, 1000]:
                for spec in [16]:
                    for num_heads in [8]:
                        for drop in [0.2]:
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
                            model_config_pos_enc = RelPosModelConfig(
                                feature_extraction_config=fe_config,
                                frontend_config=frontend_config,
                                specaug_config=specaug_config_test,
                                label_target_size=vocab_size_without_blank,
                                pos_emb_config=pos_emb_cfg,
                                conformer_size=dim,
                                num_layers=12,
                                num_heads=num_heads,
                                ff_dim=4 * dim,
                                att_weights_dropout=drop,
                                conv_dropout=drop,
                                ff_dropout=drop,
                                mhsa_dropout=drop,
                                mhsa_with_bias=True,
                                conv_kernel_size=31,
                                final_dropout=drop,
                                dropout_broadcast_axes=None,
                                specauc_start_epoch=spec_start,
                                module_list=["ff", "conv", "mhsa", "ff"],
                                module_scales=[0.5, 1.0, 1.0, 0.5],
                                aux_ctc_loss_layers=None,
                                aux_ctc_loss_scales=None,
                            )

                            train_config = {
                                "optimizer": {
                                    "class": "radam",
                                    "epsilon": 1e-16,
                                    "weight_decay": 1e-2,
                                    "decoupled_weight_decay": True,
                                },
                                "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))
                                + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
                                + list(np.linspace(5e-5, 1e-7, 30)),
                                #############
                                "batch_size": 180 * 16000,
                                "max_seq_length": {"audio_features": 35 * 16000},
                                "accum_grad_multiple_step": 1,
                            }
                            if epochs == 1000:
                                train_config["torch_amp_options"] = {"dtype": "bfloat16"}
                            train_args = {
                                "config": train_config,
                                "network_module": network_module_pos_enc_v1,
                                "net_args": {"model_config_dict": asdict(model_config_pos_enc)},
                                "debug": True,
                            }
                            results = {}
                            if drop == 0.2:
                                training_name = (
                                    prefix_name
                                    + "/"
                                    + network_module_pos_enc_v1
                                    + f"_{epochs}_{dim}_{num_heads}_{spec}_{spec_start}"
                                )
                            else:
                                training_name = (
                                    prefix_name
                                    + "/"
                                    + network_module_pos_enc_v1
                                    + f"_{epochs}_{dim}_{num_heads}_{spec}_{spec_start}_drop{drop}"
                                )
                            train_job = training(
                                training_name, train_data, train_args, num_epochs=epochs, **default_returnn
                            )
                            if epochs == 1000:
                                train_job.rqmt["gpu_mem"] = 48

                            if training_name in RTFS:
                                rtf_args = RTFArgs(
                                    #beam_sizes=[256, 512, 1024, 4096],
                                    beam_sizes=[64, 128, 256, 512, 1024, 4096],
                                    #beam_size_tokens=[4, 6, 8, 10, 12, 20, 30],  # makes it much faster
                                    beam_size_tokens=[4, 8, 20, 30],
                                    #beam_thresholds=[8, 10, 12, 14, 20, 30],
                                    beam_thresholds=[8, 20, 30],
                                    decoder_module="ctc.decoder.flashlight_ctc_v4_rescale_measure",
                                )
                            else:
                                rtf_args = None
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
                                prior_scales=prior_scales,
                                lm_scales=lm_scales,
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                                run_rtf=training_name in RTFS,
                                rtf_args=rtf_args,
                            )

                            generate_report(results=results, exp_name=training_name)
                            blank_counts = calculate_blank_counts(
                                prefix_name=training_name,
                                train_job=train_job,
                                train_args=train_args,
                                train_data=train_data,
                                checkpoint=epochs,
                                debug=True,
                            )
                            tk.register_output(training_name + "/" + "blank_counts", blank_counts)
                            new_rep[training_name] = results
                            chkpts[training_name] = train_job.out_checkpoints[250]

                            del results

                            train_config = {
                                "optimizer": {"class": "radam", "epsilon": 1e-16, "weight_decay": 1e-2,
                                    "decoupled_weight_decay": True},
                                "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))
                                                  + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
                                                  + list(np.linspace(5e-5, 1e-7, 30)),
                                #############
                                "batch_size": 180 * 16000,
                                "max_seq_length": {"audio_features": 35 * 16000},
                                "accum_grad_multiple_step": 1,
                                "gradient_clip_norm": 1.0,
                            }
                            train_args = {
                                "config": train_config,
                                "network_module": network_module_pos_enc_v1,
                                "net_args": {"model_config_dict": asdict(model_config_pos_enc)},
                                "debug": False,
                                "use_speed_perturbation": True,
                            }
                            results = {}
                            training_name = (
                                prefix_name
                                + "/"
                                + network_module_pos_enc_v1
                                + f"_{epochs}_{dim}_{num_heads}_{spec}_{spec_start}_{drop}_radam_180bs"
                            )
                            train_job = training(
                                training_name, train_data, train_args, num_epochs=epochs, **default_returnn
                            )
                            if not os.path.exists(
                                f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                                train_job.hold()
                                train_job.move_to_hpc = True

                            train_job.rqmt["gpu_mem"] = 48

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
                                prior_scales=prior_scales,
                                lm_scales=lm_scales,
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                            )
                            generate_report(results=results, exp_name=training_name)
                            new_rep[training_name] = results
                            del results

                            train_config = {
                                "optimizer": {"class": "radam", "epsilon": 1e-16, "weight_decay": 1e-2,
                                    "decoupled_weight_decay": True},
                                "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))
                                                  + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
                                                  + list(np.linspace(5e-5, 1e-7, 30)),
                                #############
                                "batch_size": 180 * 16000,
                                "max_seq_length": {"audio_features": 35 * 16000},
                                "accum_grad_multiple_step": 1,
                                "gradient_clip_norm": 1.0,
                                "torch_amp_options": {"dtype": "bfloat16"},
                            }
                            train_args = {
                                "config": train_config,
                                "network_module": network_module_pos_enc_v1,
                                "net_args": {"model_config_dict": asdict(model_config_pos_enc)},
                                "debug": False,
                                "use_speed_perturbation": True,
                            }
                            results = {}
                            training_name = (
                                prefix_name
                                + "/"
                                + network_module_pos_enc_v1
                                + f"_{epochs}_{dim}_{num_heads}_{spec}_{spec_start}_{drop}_radam_180bs_amp"
                            )
                            train_job = training(
                                training_name, train_data, train_args, num_epochs=epochs, **default_returnn
                            )
                            if not os.path.exists(
                                f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                                train_job.hold()
                                train_job.move_to_hpc = True

                            train_job.rqmt["gpu_mem"] = 48

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
                                prior_scales=prior_scales,
                                lm_scales=lm_scales,
                                run_test=True,
                                test_dataset_tuples=test_dataset_tuples,
                            )
                            generate_report(results=results, exp_name=training_name)
                            new_rep[training_name] = results
                            del results

    for dim in [256, 384, 512]:
        for ff_factor in [2, 4]:
            for layer_count in [8, 10, 12]:
                for conv_size in [15, 31]:
                    for spec_start in [1]:
                        for epochs in [500]:
                            for spec in [16]:
                                for num_heads in [8]:
                                    for drop in [0.2]:
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
                                        model_config_pos_enc = RelPosModelConfig(
                                            feature_extraction_config=fe_config,
                                            frontend_config=frontend_config,
                                            specaug_config=specaug_config_test,
                                            label_target_size=vocab_size_without_blank,
                                            pos_emb_config=pos_emb_cfg,
                                            conformer_size=dim,
                                            num_layers=layer_count,
                                            num_heads=num_heads,
                                            ff_dim=ff_factor * dim,
                                            att_weights_dropout=drop,
                                            conv_dropout=drop,
                                            ff_dropout=drop,
                                            mhsa_dropout=drop,
                                            mhsa_with_bias=True,
                                            conv_kernel_size=conv_size,
                                            final_dropout=drop,
                                            dropout_broadcast_axes=None,
                                            specauc_start_epoch=spec_start,
                                            module_list=["ff", "conv", "mhsa", "ff"],
                                            module_scales=[0.5, 1.0, 1.0, 0.5],
                                            aux_ctc_loss_layers=None,
                                            aux_ctc_loss_scales=None,
                                        )

                                        train_config = {
                                            "optimizer": {
                                                "class": "radam",
                                                "epsilon": 1e-16,
                                                "weight_decay": 1e-2,
                                                "decoupled_weight_decay": True,
                                            },
                                            "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))
                                            + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
                                            + list(np.linspace(5e-5, 1e-7, 30)),
                                            #############
                                            "batch_size": 180 * 16000,
                                            "max_seq_length": {"audio_features": 35 * 16000},
                                            "accum_grad_multiple_step": 1,
                                        }
                                        train_config["torch_amp_options"] = {"dtype": "bfloat16"}
                                        train_args = {
                                            "config": train_config,
                                            "network_module": network_module_pos_enc_v1,
                                            "net_args": {"model_config_dict": asdict(model_config_pos_enc)},
                                            "debug": True,
                                        }
                                        results = {}
                                        training_name = (
                                            prefix_name
                                            + "/"
                                            + network_module_pos_enc_v1
                                            + f"_{epochs}_{layer_count}_{dim}_{ff_factor}_{conv_size}_{num_heads}_{spec}_{spec_start}_{drop}"
                                        )
                                        train_job = training(
                                            training_name, train_data, train_args, num_epochs=epochs, **default_returnn
                                        )
                                        train_job.rqmt["gpu_mem"] = 48

                                        rtf_args = RTFArgs(
                                            beam_sizes=[512, 1024],
                                            beam_size_tokens=[10, 12, 20],  # makes it much faster
                                            beam_thresholds=[12, 14, 20, 30],
                                        )
                                        results = eval_model(
                                            training_name=training_name,
                                            train_job=train_job,
                                            train_args=train_args,
                                            train_data=train_data,
                                            decoder_config=default_decoder_config,
                                            dev_dataset_tuples=dev_dataset_tuples,
                                            result_dict=results,
                                            loss_name=f"ctc_loss_layer{layer_count}",
                                            specific_epoch=epochs,
                                            prior_scales=prior_scales,
                                            lm_scales=lm_scales,
                                            run_test=True,
                                            test_dataset_tuples=test_dataset_tuples,
                                            #run_rtf=True,
                                            #rtf_args=rtf_args,
                                            run_best_4=False,
                                        )

                                        generate_report(results=results, exp_name=training_name)
                                        # blank_counts = calculate_blank_counts(
                                        #     prefix_name=training_name,
                                        #     train_job=train_job,
                                        #     train_args=train_args,
                                        #     train_data=train_data,
                                        #     checkpoint=epochs,
                                        #     debug=True,
                                        # )
                                        # tk.register_output(training_name + "/" + "blank_counts", blank_counts)
                                        new_rep[training_name] = results
                                        # chkpts[training_name] = train_job.out_checkpoints[250]

                                        del results

    tk.register_report("reports/pos_enc_report", partial(build_base_report, new_rep), required=new_rep)
    if get_report is True:
        rep = {}
        rep.update(new_rep)
        return rep, chkpts
