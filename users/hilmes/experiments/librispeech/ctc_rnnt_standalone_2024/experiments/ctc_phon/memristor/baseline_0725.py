from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast
import os
from functools import partial

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ....data.common import DatasetSettings, build_test_dataset
from ....data.phon import build_eow_phon_training_datasets, get_text_lexicon, get_bliss_phoneme_lexicon
from ....default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ....lm import get_4gram_binary_lm, get_arpa_lm_config
from ....pipeline import training, compute_statistics, prepare_memristor
from ....config import get_stats_config, get_mem_init_config
from ....report import generate_report, build_qat_report, build_qat_report_v2

from ..tune_eval import tune_and_evaluate_helper, eval_model


def eow_phon_ls960_0725_memristor():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_eow_phon_memristor"

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
    from ....pytorch_networks.ctc.decoder.flashlight_qat_phoneme_ctc import DecoderConfig as DecoderConfigMemristor

    default_decoder_config = DecoderConfig(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    decoder_config_memristor = DecoderConfigMemristor(
        lexicon=get_text_lexicon(),
        returnn_vocab=label_datastream.vocab,
        beam_size=1024,
        beam_size_token=12,  # makes it much faster
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,
    )

    decoder_config_no_memristor = copy.deepcopy(decoder_config_memristor)
    decoder_config_no_memristor.turn_off_quant = "leave_as_is"

    from ....pytorch_networks.ctc.decoder.rasr_ctc_v1 import DecoderConfig as RasrDecoderConfig
    from ....rasr_recog_config import get_tree_timesync_recog_config, get_no_op_label_scorer_config

    recog_rasr_config, recog_rasr_post_config = get_tree_timesync_recog_config(
        lexicon_file=get_bliss_phoneme_lexicon(),
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        blank_index=vocab_size_without_blank,
        max_beam_size=2048,
        score_threshold=18.0,
        logfile_suffix="recog",
        lm_config=get_arpa_lm_config("4gram", lexicon_file=get_bliss_phoneme_lexicon(), scale=0.0),
    )

    as_training_rasr_config = RasrDecoderConfig(
        rasr_config_file=recog_rasr_config,
        rasr_post_config=recog_rasr_post_config,
        blank_log_penalty=None,
        prior_scale=0.0,  # this will be overwritten internally
        prior_file=None,
        turn_off_quant="leave_as_is",  # this does not have memristor
    )
    rasr_config_memristor = copy.deepcopy(as_training_rasr_config)
    rasr_config_memristor.turn_off_quant = False

    rasr_prior_scales = [0.2, 0.3, 0.4, 0.5]
    rasr_lm_scales = [0.9, 1.0, 1.1, 1.2]

    from ....pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import \
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig, LogMelFeatureExtractionV1Config

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

    specaug_config_full = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=16,  # Normal Style
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
    from ....pytorch_networks.ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1_cfg import (
        ModelConfig as RelPosModelConfigV1,
        ConformerPosEmbConfig,
    )
    pos_emb_cfg = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )

    memristor_report = {}
    train_config_24gbgpu = {
        "optimizer": {
            "class": "radam",
            "epsilon": 1e-12,
            "weight_decay": 1e-2,
            "decoupled_weight_decay": True,
        },
        "learning_rates": list(np.linspace(7e-6, 5e-4, 480))
                          + list(np.linspace(5e-4, 5e-5, 480))
                          + list(np.linspace(5e-5, 1e-7, 40)),
        #############
        "batch_size": 360 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
    }
    baseline_network_module = "ctc.conformer_distill_1007.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"


    model_config = RelPosModelConfigV1(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config_full,
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
        pos_emb_config=pos_emb_cfg,
        module_list=["ff", "conv", "mhsa", "ff"],
        module_scales=[0.5, 1.0, 1.0, 0.5],
        aux_ctc_loss_layers=None,
        aux_ctc_loss_scales=None,
        dropout_broadcast_axes=None,
        mhsa_with_bias=True,
    )
    train_args_base = {
        "config": train_config_24gbgpu,
        "network_module": baseline_network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "use_speed_perturbation": True,
        "post_config": {"num_workers_per_gpu": 8}
    }
    name = ".baseline_512dim_sub4_48gbgpu_100eps_radam_bs360_sp"
    training_name = prefix_name + "/" + baseline_network_module + name
    train_job = training(training_name, train_data, train_args_base, num_epochs=1000, **default_returnn)

    if not os.path.exists(
        f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
        train_job.hold()
        train_job.move_to_hpc = True

    results = {}
    results, best_params_job = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args_base,
        train_data=train_data,
        decoder_config=as_training_rasr_config,
        dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
        result_dict=results,
        decoder_module="ctc.decoder.rasr_ctc_v1",
        prior_scales=rasr_prior_scales,
        lm_scales=rasr_lm_scales,
        import_memristor=True,
        get_best_params=True,
        run_rasr=True,
        run_best_4=False,
        run_best=False,
    )
    generate_report(results=results, exp_name=training_name)
    memristor_report[training_name] = results

    network_module_mem_v9 = "ctc.qat_0711.memristor_v9"
    network_module_mem_v10 = "ctc.qat_0711.memristor_v10"
    network_module_mem_v10_weight_decay = "ctc.qat_0711.memristor_v10_weight_decay"
    network_module_mem_v11 = "ctc.qat_0711.memristor_v11"
    network_module_mem_v10_keep_encs = "ctc.qat_0711.memristor_v10_keep_encs"

    from ....pytorch_networks.ctc.qat_0711.memristor_v7_cfg import QuantModelTrainConfigV7 as MemristorModelTrainConfigV7
    from ....pytorch_networks.ctc.qat_0711.memristor_v8_cfg import QuantModelTrainConfigV8 as MemristorModelTrainConfigV8
    from ....pytorch_networks.ctc.qat_0711.memristor_v10_weight_decay_cfg import QuantModelTrainConfigV8 as MemristorModelTrainConfigV10
    from ....pytorch_networks.ctc.qat_0711.memristor_v11_cfg import QuantModelTrainConfigV11 as MemristorModelTrainConfigV11

    memristor_prior = 0.5
    memristor_lm = 1.0

    from torch_memristor.memristor_modules import DacAdcHardwareSettings, CycleCorrectionSettings

    for activation_bit in [8]:
        for epochs in [1000]:
            for dim in [512]:
                for weight_bit in [3, 4, 5, 6, 7, 8]:
                    frontend_config_dim = VGG4LayerActFrontendV1Config_mod(
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
                    prior_train_dac_settings = DacAdcHardwareSettings(
                        input_bits=0,
                        output_precision_bits=0,
                        output_range_bits=0,
                        hardware_input_vmax=0.6,
                        hardware_output_current_scaling=8020.0,
                    )
                    model_config = MemristorModelTrainConfigV8(
                        feature_extraction_config=fe_config,
                        frontend_config=frontend_config_dim,
                        specaug_config=specaug_config_full,
                        label_target_size=vocab_size_without_blank,
                        conformer_size=dim,
                        num_layers=12,
                        num_heads=8,
                        ff_dim=dim * 4,
                        att_weights_dropout=0.1,
                        conv_dropout=0.1,
                        ff_dropout=0.1,
                        mhsa_dropout=0.1,
                        conv_kernel_size=31,
                        final_dropout=0.1,
                        specauc_start_epoch=1,
                        weight_quant_dtype="qint8",
                        weight_quant_method="per_tensor_symmetric",
                        activation_quant_dtype="qint8",
                        activation_quant_method="per_tensor_symmetric",
                        dot_quant_dtype="qint8",
                        dot_quant_method="per_tensor_symmetric",
                        Av_quant_dtype="qint8",
                        Av_quant_method="per_tensor_symmetric",
                        moving_average=None,
                        weight_bit_prec=weight_bit,
                        activation_bit_prec=activation_bit,
                        quantize_output=False,
                        converter_hardware_settings=prior_train_dac_settings,
                        quant_in_linear=True,
                        num_cycles=0,
                        correction_settings=None,
                        weight_noise_func=None,
                        weight_noise_values=None,
                        weight_noise_start_epoch=None,
                        pos_emb_config=pos_emb_cfg,
                        module_list=["ff", "conv", "mhsa", "ff"],
                        module_scales=[0.5, 1.0, 1.0, 0.5],
                        aux_ctc_loss_layers=None,
                        aux_ctc_loss_scales=None,
                        dropout_broadcast_axes=None,
                    )
                    res_seeds_total = {}
                    for seed in range(3):
                        train_config_24gbgpu = {
                            "optimizer": {
                                "class": "radam",
                                "epsilon": 1e-12,
                                "weight_decay": 1e-2,
                                "decoupled_weight_decay": True,
                            },
                            "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs // 2 - 20)))
                                              + list(np.linspace(5e-4, 5e-5, (epochs // 2 - 20)))
                                              + list(np.linspace(5e-5, 1e-7, 40)),
                            #############
                            "batch_size": 360 * 16000,
                            "max_seq_length": {"audio_features": 35 * 16000},
                            "accum_grad_multiple_step": 1,
                            "gradient_clip_norm": 1.0,
                            "seed": seed,
                            "torch_amp_options": {"dtype": "bfloat16"},
                        }
                        train_args = {
                            "config": train_config_24gbgpu,
                            "network_module": network_module_mem_v9,
                            "net_args": {"model_config_dict": asdict(model_config)},
                            "debug": False,
                            "post_config": {"num_workers_per_gpu": 8},
                            "use_speed_perturbation": True,
                        }
                        training_name = prefix_name + "/" + network_module_mem_v9 + f"_{epochs//10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}"
                        train_job = training(training_name, train_data, train_args, num_epochs=epochs, **default_returnn)
                        if not os.path.exists(
                            f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                            train_job.rqmt['cpu'] = 8
                            train_job.hold()
                            train_job.move_to_hpc = True

                        results = {}
                        results, best_params_job = eval_model(
                            training_name=training_name,
                            train_job=train_job,
                            train_args=train_args,
                            train_data=train_data,
                            decoder_config=as_training_rasr_config,
                            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                            result_dict=results,
                            decoder_module="ctc.decoder.rasr_ctc_v1",
                            prior_scales=rasr_prior_scales,
                            lm_scales=rasr_lm_scales,
                            import_memristor=True,
                            get_best_params=True,
                            run_rasr=True,
                            run_best_4=False,
                            run_best=False,
                        )
                        generate_report(results=results, exp_name=training_name + "/non_memristor")
                        memristor_report[training_name] = results

                        results = {}
                        if seed == 0 and weight_bit == 4:
                            training_name = prefix_name + "/" + network_module_mem_v9 + f"_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}_gpu"
                            results, best_params_job = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args,
                                train_data=train_data,
                                decoder_config=as_training_rasr_config,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                result_dict=results,
                                decoder_module="ctc.decoder.rasr_ctc_v1",
                                prior_scales=[0.5],
                                lm_scales=[1.0],
                                import_memristor=True,
                                get_best_params=True,
                                run_rasr=True,
                                run_best_4=False,
                                run_best=False,
                                use_gpu=True,
                                extra_forward_config = {
                                    "max_seqs": 2000000
                                },
                                search_gpu=48,
                            )

                        # compute statistics
                        if epochs == 1000:
                            cfg = get_stats_config(
                                dataset=train_data.cv,
                                network_module=network_module_mem_v10,
                                config={},
                                net_args=train_args['net_args'],
                                unhashed_net_args=train_args.get("unhashed_net_args", None),
                                debug=True,
                                import_memristor=True,
                            )
                            stats = compute_statistics(
                                prefix_name = training_name,
                                returnn_config=cfg,
                                checkpoint=train_job.out_checkpoints[epochs],
                                returnn_exe=RETURNN_EXE,
                                returnn_root=MINI_RETURNN_ROOT,
                            )
                            tk.register_output(training_name + "/mhsa_statistics", stats)
                            tk.register_output("papers/2026_pos_enc/" + network_module_mem_v9 + f"_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}", stats)

                        # res_conv = {}
                        # res_split = {}
                        res_batched = {}
                        max_cycles = 5
                        prior_args = {
                            "config": train_config_24gbgpu,
                            "network_module": network_module_mem_v9,
                            "net_args": {"model_config_dict": asdict(model_config)},
                            "debug": False,
                            "post_config": {"num_workers_per_gpu": 8},
                            "use_speed_perturbation": True,
                        }
                        for num_cycles in range(1, max_cycles+1):
                            recog_dac_settings = DacAdcHardwareSettings(
                                input_bits=8,
                                output_precision_bits=4,
                                output_range_bits=4,
                                hardware_input_vmax=0.6,
                                hardware_output_current_scaling=8020.0,
                            )
                            model_config_recog = copy.deepcopy(model_config)
                            model_config_recog.converter_hardware_settings = recog_dac_settings
                            model_config_recog.num_cycles = num_cycles

                            train_args_recog = {
                                "config": train_config_24gbgpu,
                                "network_module": network_module_mem_v10,
                                "net_args": {"model_config_dict": asdict(model_config_recog)},
                                "debug": False,
                                "post_config": {"num_workers_per_gpu": 8},
                                "use_speed_perturbation": True,
                            }
                            recog_name = prefix_name + "/" + network_module_mem_v10 + f"_batched_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}/cycle_{num_cycles // 11}"
                            res_batched = eval_model(
                                training_name=recog_name + f"_{num_cycles}",
                                train_job=train_job,
                                train_args=train_args_recog,
                                train_data=train_data,
                                decoder_config=rasr_config_memristor,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                result_dict=res_batched,
                                decoder_module="ctc.decoder.rasr_ctc_v1_batched",
                                prior_scales=[memristor_prior],
                                lm_scales=[memristor_lm],
                                use_gpu=True,
                                import_memristor=True,
                                extra_forward_config={
                                    "batch_size": 3500000 if not weight_bit in [8] else 2500000,
                                },
                                run_best_4=False,
                                run_best=False,
                                prior_args=prior_args,
                                run_search_on_hpc=False,
                                run_rasr=True,
                                split_mem_init=True,
                            )
                            res_seeds_total.update(res_batched)
                            if num_cycles == max_cycles:
                                generate_report(results=res_batched, exp_name=recog_name)
                                memristor_report[recog_name] = copy.deepcopy(res_batched)
                            if num_cycles == 2 and weight_bit in [4, 8] and seed == 0:
                                mem_init_config = get_mem_init_config(
                                    training_datasets=train_data,
                                    network_module=network_module_mem_v10,
                                    config={},
                                    net_args=train_args_recog["net_args"],
                                    unhashed_net_args=train_args_recog.get("unhashed_net_args", None),
                                    debug=True,
                                    import_memristor=True,
                                )
                                checkpoint = prepare_memristor(
                                    recog_name,
                                    mem_init_config,
                                    checkpoint=train_job.out_checkpoints[epochs],
                                    returnn_exe=RETURNN_EXE,
                                    returnn_root=MINI_RETURNN_ROOT,
                                )

                                cfg = get_stats_config(
                                    dataset=train_data.cv,
                                    network_module=network_module_mem_v10 + "_mem_inited",
                                    config={},
                                    net_args=train_args_recog['net_args'],
                                    unhashed_net_args=train_args_recog.get("unhashed_net_args", None),
                                    debug=True,
                                    import_memristor=True,
                                )
                                stats = compute_statistics(
                                    prefix_name=recog_name,
                                    returnn_config=cfg,
                                    checkpoint=checkpoint,
                                    returnn_exe=RETURNN_EXE,
                                    returnn_root=MINI_RETURNN_ROOT,
                                    device="gpu",
                                )
                                tk.register_output(recog_name + "/mhsa_statistics", stats)
                                tk.register_output(
                                    "papers/2026_pos_enc/" + network_module_mem_v10 + f"_batched_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}", stats)

                                if weight_bit == 8 and seed == 0:

                                    cfg = get_stats_config(
                                        dataset=train_data.cv,
                                        network_module=network_module_mem_v10 + "_mem_inited_calc",
                                        config={"tmp": None},
                                        net_args=train_args_recog['net_args'],
                                        unhashed_net_args=train_args_recog.get("unhashed_net_args", None),
                                        debug=True,
                                        import_memristor=True,
                                    )
                                    stats = compute_statistics(
                                        prefix_name=recog_name,
                                        returnn_config=cfg,
                                        checkpoint=checkpoint,
                                        returnn_exe=RETURNN_EXE,
                                        returnn_root=MINI_RETURNN_ROOT,
                                        device="gpu",
                                        output_file="num_clipped.pkl"
                                    )
                                    tk.register_output(recog_name + "/mhsa_statistics_num_clipped", stats)


                        res_keep_enc = {}
                        if weight_bit in [4, 8]:
                            for num_cycles in range(1, max_cycles + 1):
                                model_config_recog = copy.deepcopy(model_config)
                                model_config_recog.converter_hardware_settings = recog_dac_settings
                                model_config_recog.num_cycles = num_cycles
                                train_args_recog = {
                                    "config": train_config_24gbgpu,
                                    "network_module": network_module_mem_v10_keep_encs,
                                    "net_args": {"model_config_dict": asdict(model_config_recog)},
                                    "debug": False,
                                    "post_config": {"num_workers_per_gpu": 8},
                                    "use_speed_perturbation": True,
                                }
                                recog_name = prefix_name + "/" + network_module_mem_v10_keep_encs + f"_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}/cycle_{num_cycles // 11}"
                                res_keep_enc = eval_model(
                                    training_name=recog_name + f"_{num_cycles}",
                                    train_job=train_job,
                                    train_args=train_args_recog,
                                    train_data=train_data,
                                    decoder_config=rasr_config_memristor,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    result_dict=res_keep_enc,
                                    decoder_module="ctc.decoder.rasr_ctc_v1_batched",
                                    prior_scales=[memristor_prior], # prior_scales=[best_params_job.out_optimal_parameters[1]],
                                    lm_scales=[memristor_lm], # lm_scales=[(best_params_job.out_optimal_parameters[0], "best")],
                                    use_gpu=True,
                                    import_memristor=True,
                                    extra_forward_config={
                                        "batch_size": 3500000 if not weight_bit in [8] else 2500000,
                                    },
                                    run_best_4=False,
                                    run_best=False,
                                    prior_args=prior_args,
                                    run_search_on_hpc=False,
                                    run_rasr=True,
                                    split_mem_init=True,
                                )
                                if num_cycles == max_cycles:
                                    generate_report(results=res_keep_enc, exp_name=recog_name)
                                    memristor_report[recog_name] = copy.deepcopy(res_keep_enc)

                        if epochs == 1000 and weight_bit in [4, 8]:
                            # [(1, 7), (2, 6),(7, 1), (1, 12),
                            for prec, ran in [(2, 6), (3, 5), (4, 8), (4, 12), (12, 12), (8, 8), (6, 6), (8, 4)]:
                                if (prec, ran) in [(8, 4)] and weight_bit not in [4, 8]:
                                    continue
                                res_balance = {}
                                res_fixed = {}
                                for num_cycles in range(1, max_cycles + 1):

                                    model_config_recog = copy.deepcopy(model_config)
                                    model_config_recog.converter_hardware_settings = recog_dac_settings
                                    model_config_recog.num_cycles = num_cycles
                                    model_config_balanced_adc = copy.deepcopy(model_config_recog)
                                    recog_dac_settings_larger = DacAdcHardwareSettings(
                                        input_bits=8,
                                        output_precision_bits=prec,
                                        output_range_bits=ran,
                                        hardware_input_vmax=0.6,
                                        hardware_output_current_scaling=8020.0,
                                    )
                                    model_config_balanced_adc.converter_hardware_settings = recog_dac_settings_larger

                                    train_args_recog = {
                                        "config": train_config_24gbgpu,
                                        "network_module": network_module_mem_v10,
                                        "net_args": {"model_config_dict": asdict(model_config_balanced_adc)},
                                        "debug": False,
                                        "post_config": {"num_workers_per_gpu": 8},
                                        "use_speed_perturbation": True,
                                    }
                                    recog_name = prefix_name + "/" + network_module_mem_v10 + f"_fixed_bal_{ran}_{prec}_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}/cycle_{num_cycles // 11}"
                                    res_fixed = eval_model(
                                        training_name=recog_name + f"_{num_cycles}",
                                        train_job=train_job,
                                        train_args=train_args_recog,
                                        train_data=train_data,
                                        decoder_config=rasr_config_memristor,
                                        dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                        result_dict=res_fixed,
                                        decoder_module="ctc.decoder.rasr_ctc_v1_batched",
                                        prior_scales=[memristor_prior],
                                        lm_scales=[memristor_lm],
                                        use_gpu=True,
                                        import_memristor=True,
                                        extra_forward_config={
                                            "batch_size": 3500000 if not weight_bit in [8] else 2500000,
                                        },
                                        run_best_4=False,
                                        run_best=False,
                                        prior_args=prior_args,
                                        run_search_on_hpc=False,
                                        run_rasr=True,
                                        split_mem_init=True,
                                    )
                                    if num_cycles == max_cycles:
                                        generate_report(results=res_fixed, exp_name=recog_name)
                                        memristor_report[recog_name] = copy.deepcopy(res_fixed)

                                    if num_cycles == 2 and weight_bit in [4, 8] and (prec, ran) in [(4, 8), (8, 8)] and seed == 0:
                                        mem_init_config = get_mem_init_config(
                                            training_datasets=train_data,
                                            network_module=network_module_mem_v10,
                                            config={},
                                            net_args=train_args_recog["net_args"],
                                            unhashed_net_args=train_args_recog.get("unhashed_net_args", None),
                                            debug=True,
                                            import_memristor=True,
                                        )
                                        checkpoint = prepare_memristor(
                                            recog_name,
                                            mem_init_config,
                                            checkpoint=train_job.out_checkpoints[epochs],
                                            returnn_exe=RETURNN_EXE,
                                            returnn_root=MINI_RETURNN_ROOT,
                                        )

                                        cfg = get_stats_config(
                                            dataset=train_data.cv,
                                            network_module=network_module_mem_v10 + "_mem_inited",
                                            config={},
                                            net_args=train_args_recog['net_args'],
                                            unhashed_net_args=train_args_recog.get("unhashed_net_args", None),
                                            debug=True,
                                            import_memristor=True,
                                        )
                                        stats = compute_statistics(
                                            prefix_name=recog_name,
                                            returnn_config=cfg,
                                            checkpoint=checkpoint,
                                            returnn_exe=RETURNN_EXE,
                                            returnn_root=MINI_RETURNN_ROOT,
                                            device="gpu",
                                        )
                                        tk.register_output(recog_name + "/mhsa_statistics", stats)
                                        tk.register_output("papers/2026_pos_enc/" + network_module_mem_v10 + f"_fixed_bal_{ran}_{prec}_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}", stats)

                                    if weight_bit == 4 and seed == 0 and (prec, ran) in [(4, 4), (4, 8), (8, 8)]:
                                        model_config_recog = copy.deepcopy(model_config)
                                        model_config_recog.converter_hardware_settings = recog_dac_settings
                                        model_config_recog.num_cycles = num_cycles
                                        model_config_balanced_adc = copy.deepcopy(model_config_recog)
                                        recog_dac_settings_larger = DacAdcHardwareSettings(
                                            input_bits=8,
                                            output_precision_bits=prec,
                                            output_range_bits=ran,
                                            hardware_input_vmax=0.6,
                                            hardware_output_current_scaling=8020.0,
                                        )
                                        model_config_balanced_adc.converter_hardware_settings = recog_dac_settings_larger

                                        train_args_recog = {
                                            "config": train_config_24gbgpu,
                                            "network_module": network_module_mem_v10,
                                            "net_args": {"model_config_dict": asdict(model_config_balanced_adc)},
                                            "debug": False,
                                            "post_config": {"num_workers_per_gpu": 8},
                                            "use_speed_perturbation": True,
                                        }

                                        model_config_4_4 = copy.deepcopy(model_config_recog)
                                        recog_dac_4_4 = DacAdcHardwareSettings(
                                            input_bits=8,
                                            output_precision_bits=4,
                                            output_range_bits=4,
                                            hardware_input_vmax=0.6,
                                            hardware_output_current_scaling=8020.0,
                                        )
                                        model_config_4_4.converter_hardware_settings = recog_dac_4_4

                                        train_args_4_4 = {
                                            "config": train_config_24gbgpu,
                                            "network_module": network_module_mem_v10,
                                            "net_args": {"model_config_dict": asdict(model_config_4_4)},
                                            "debug": False,
                                            "post_config": {"num_workers_per_gpu": 8},
                                            "use_speed_perturbation": True,
                                        }
                                        res_same = {}
                                        recog_name = prefix_name + "/" + network_module_mem_v10 + f"_fixed_samemem_{ran}_{prec}_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}/cycle_{num_cycles // 11}"
                                        res_same = eval_model(
                                            training_name=recog_name + f"_{num_cycles}",
                                            train_job=train_job,
                                            train_args=train_args_recog,
                                            train_data=train_data,
                                            decoder_config=rasr_config_memristor,
                                            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                            result_dict=res_same,
                                            decoder_module="ctc.decoder.rasr_ctc_v1_batched",
                                            prior_scales=[memristor_prior],
                                            lm_scales=[memristor_lm],
                                            use_gpu=True,
                                            import_memristor=True,
                                            extra_forward_config={
                                                "batch_size": 3500000 if not weight_bit in [8] else 2500000,
                                            },
                                            run_best_4=False,
                                            run_best=False,
                                            prior_args=prior_args,
                                            run_search_on_hpc=False,
                                            run_rasr=True,
                                            split_mem_init=True,
                                            split_args=train_args_4_4,
                                        )
                                        if num_cycles == max_cycles:
                                            generate_report(results=res_same, exp_name=recog_name)
                                            memristor_report[recog_name] = copy.deepcopy(res_same)

                            for prec, ran in [(0, 8), (1, 7), (2, 6), (3, 5), (4, 8), (4, 12), (12, 12), (8, 8)]:
                                if not seed == 0 and not weight_bit in [4, 8]:
                                    continue
                                if (prec, ran) in [(0, 8), (1, 7), (2, 6), (3, 5)] and weight_bit not in [4, 8]:
                                    continue
                                res_balance = {}
                                res_fixed = {}
                                for num_cycles in range(1, max_cycles + 1):
                                    # model_config_recog = copy.deepcopy(model_config)
                                    # model_config_recog.converter_hardware_settings = recog_dac_settings
                                    # model_config_recog.num_cycles = num_cycles
                                    model_config_balanced_adc = MemristorModelTrainConfigV11(
                                        **model_config.__dict__,
                                        pos_enc_converter_hardware_settings=None,
                                    )
                                    model_config_balanced_adc.converter_hardware_settings = recog_dac_settings
                                    model_config_balanced_adc.num_cycles = num_cycles
                                    recog_dac_settings_larger = DacAdcHardwareSettings(
                                        input_bits=8,
                                        output_precision_bits=prec,
                                        output_range_bits=ran,
                                        hardware_input_vmax=0.6,
                                        hardware_output_current_scaling=8020.0,
                                    )
                                    model_config_balanced_adc.pos_enc_converter_hardware_settings = recog_dac_settings_larger

                                    train_args_recog = {
                                        "config": train_config_24gbgpu,
                                        "network_module": network_module_mem_v11,
                                        "net_args": {"model_config_dict": asdict(model_config_balanced_adc)},
                                        "debug": False,
                                        "post_config": {"num_workers_per_gpu": 8},
                                        "use_speed_perturbation": True,
                                    }
                                    recog_name = prefix_name + "/" + network_module_mem_v11 + f"_fixed_posadc_{ran}_{prec}_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}/cycle_{num_cycles // 11}"
                                    res_fixed = eval_model(
                                        training_name=recog_name + f"_{num_cycles}",
                                        train_job=train_job,
                                        train_args=train_args_recog,
                                        train_data=train_data,
                                        decoder_config=rasr_config_memristor,
                                        dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                        result_dict=res_fixed,
                                        decoder_module="ctc.decoder.rasr_ctc_v1_batched",
                                        prior_scales=[memristor_prior],
                                        lm_scales=[memristor_lm],
                                        use_gpu=True,
                                        import_memristor=True,
                                        extra_forward_config={
                                            "batch_size": 3500000 if not weight_bit in [8] else 2500000,
                                        },
                                        run_best_4=False,
                                        run_best=False,
                                        prior_args=prior_args,
                                        run_search_on_hpc=False,
                                        run_rasr=True,
                                        split_mem_init=True,
                                    )
                                    if num_cycles == max_cycles:
                                        generate_report(results=res_fixed, exp_name=recog_name)
                                        memristor_report[recog_name] = copy.deepcopy(res_fixed)

                                    if num_cycles == 2 and weight_bit in [4, 8] and (prec, ran) in [(4, 8), (8, 8), (1, 7), (0, 8)] and seed == 0:
                                        mem_init_config = get_mem_init_config(
                                            training_datasets=train_data,
                                            network_module=network_module_mem_v11,
                                            config={},
                                            net_args=train_args_recog["net_args"],
                                            unhashed_net_args=train_args_recog.get("unhashed_net_args", None),
                                            debug=True,
                                            import_memristor=True,
                                        )
                                        checkpoint = prepare_memristor(
                                            recog_name,
                                            mem_init_config,
                                            checkpoint=train_job.out_checkpoints[epochs],
                                            returnn_exe=RETURNN_EXE,
                                            returnn_root=MINI_RETURNN_ROOT,
                                        )

                                        cfg = get_stats_config(
                                            dataset=train_data.cv,
                                            network_module=network_module_mem_v11 + "_mem_inited",
                                            config={},
                                            net_args=train_args_recog['net_args'],
                                            unhashed_net_args=train_args_recog.get("unhashed_net_args", None),
                                            debug=True,
                                            import_memristor=True,
                                        )
                                        stats = compute_statistics(
                                            prefix_name=recog_name,
                                            returnn_config=cfg,
                                            checkpoint=checkpoint,
                                            returnn_exe=RETURNN_EXE,
                                            returnn_root=MINI_RETURNN_ROOT,
                                            device="gpu",
                                        )
                                        tk.register_output(recog_name + "/mhsa_statistics", stats)
                                        tk.register_output("papers/2026_pos_enc/" + network_module_mem_v11 + f"_fixed_posadc_{ran}_{prec}_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}", stats)

                        if weight_bit in [4, 5, 6, 7, 8] and seed == 0:
                            train_config_24gbgpu = {
                                "optimizer": {
                                    "class": "radam",
                                    "epsilon": 1e-12,
                                    "weight_decay": 1e-2,
                                    "decoupled_weight_decay": True,
                                },
                                "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs // 2 - 20)))
                                                  + list(np.linspace(5e-4, 5e-5, (epochs // 2 - 20)))
                                                  + list(np.linspace(5e-5, 1e-7, 40)),
                                #############
                                "batch_size": 360 * 16000,
                                "max_seq_length": {"audio_features": 35 * 16000},
                                "accum_grad_multiple_step": 1,
                                "gradient_clip_norm": 1.0,
                                "seed": seed,
                                "torch_amp_options": {"dtype": "bfloat16"},
                            }
                            model_config_output = copy.deepcopy(model_config)
                            model_config_output.quantize_output = True
                            train_args = {
                                "config": train_config_24gbgpu,
                                "network_module": network_module_mem_v9,
                                "net_args": {"model_config_dict": asdict(model_config_output)},
                                "debug": False,
                                "post_config": {"num_workers_per_gpu": 8},
                                "use_speed_perturbation": True,
                            }
                            training_name = prefix_name + "/" + network_module_mem_v9 + f"_quantout_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}"
                            train_job = training(training_name, train_data, train_args, num_epochs=epochs,
                                **default_returnn)
                            if not os.path.exists(
                                f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                                train_job.hold()
                                train_job.rqmt['cpu'] = 8
                                train_job.move_to_hpc = True

                            results = {}
                            results, best_params_job = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args,
                                train_data=train_data,
                                decoder_config=as_training_rasr_config,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                result_dict=results,
                                decoder_module="ctc.decoder.rasr_ctc_v1",
                                prior_scales=rasr_prior_scales,
                                lm_scales=rasr_lm_scales,
                                import_memristor=True,
                                get_best_params=True,
                                run_rasr=True,
                                run_best_4=False,
                                run_best=False,
                            )
                            generate_report(results=results, exp_name=training_name + "/non_memristor")
                            memristor_report[training_name] = results

                        if weight_bit in [4, 8]:
                            train_config_24gbgpu = {
                                "optimizer": {
                                    "class": "radam",
                                    "epsilon": 1e-12,
                                    "weight_decay": 1e-2,
                                    "decoupled_weight_decay": True,
                                },
                                "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs // 2 - 20)))
                                                  + list(np.linspace(5e-4, 5e-5, (epochs // 2 - 20)))
                                                  + list(np.linspace(5e-5, 1e-7, 40)),
                                #############
                                "batch_size": 360 * 16000,
                                "max_seq_length": {"audio_features": 35 * 16000},
                                "accum_grad_multiple_step": 1,
                                "gradient_clip_norm": 1.0,
                                "seed": seed,
                                "torch_amp_options": {"dtype": "bfloat16"},
                            }
                            no_lin_pos_emb_cfg = ConformerPosEmbConfig(
                                learnable_pos_emb=False,
                                rel_pos_clip=16,
                                with_linear_pos=False,
                                with_pos_bias=True,
                                separate_pos_emb_per_head=True,
                                pos_emb_dropout=0.0,
                            )
                            model_config_learn_pos = copy.deepcopy(model_config)
                            model_config_learn_pos.pos_emb_config = no_lin_pos_emb_cfg
                            train_args = {
                                "config": train_config_24gbgpu,
                                "network_module": network_module_mem_v9,
                                "net_args": {"model_config_dict": asdict(model_config_learn_pos)},
                                "debug": False,
                                "post_config": {"num_workers_per_gpu": 8},
                                "use_speed_perturbation": True,
                            }
                            training_name = prefix_name + "/" + network_module_mem_v9 + f"_nolinpos_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}"

                            train_job = training(training_name, train_data, train_args, num_epochs=epochs,
                                **default_returnn)
                            if not os.path.exists(
                                f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                                train_job.rqmt['cpu'] = 8
                                train_job.hold()
                                train_job.move_to_hpc = True
                            # train_job.rqmt["gpu_mem"] = 48
                            results = {}
                            results, best_params_job = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args,
                                train_data=train_data,
                                decoder_config=as_training_rasr_config,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                result_dict=results,
                                decoder_module="ctc.decoder.rasr_ctc_v1",
                                prior_scales=rasr_prior_scales,
                                lm_scales=rasr_lm_scales,
                                import_memristor=True,
                                get_best_params=True,
                                run_rasr=True,
                                run_best_4=False,
                                run_best=False,
                            )
                            generate_report(results=results, exp_name=training_name + "/non_memristor")
                            memristor_report[training_name] = results
                            res_conv = {}
                            for num_cycles in range(1, max_cycles+1):
                                recog_dac_settings = DacAdcHardwareSettings(
                                    input_bits=8,
                                    output_precision_bits=4,
                                    output_range_bits=4,
                                    hardware_input_vmax=0.6,
                                    hardware_output_current_scaling=8020.0,
                                )
                                model_config_recog = copy.deepcopy(model_config_learn_pos)
                                model_config_recog.converter_hardware_settings = recog_dac_settings
                                model_config_recog.num_cycles = num_cycles

                                prior_args = {
                                    "config": train_config_24gbgpu,
                                    "network_module": network_module_mem_v9,
                                    "net_args": {"model_config_dict": asdict(model_config_learn_pos)},
                                    "debug": False,
                                    "post_config": {"num_workers_per_gpu": 8},
                                    "use_speed_perturbation": True,
                                }

                                train_args_recog = {
                                    "config": train_config_24gbgpu,
                                    "network_module": network_module_mem_v10,
                                    "net_args": {"model_config_dict": asdict(model_config_recog)},
                                    "debug": False,
                                    "post_config": {"num_workers_per_gpu": 8},
                                    "use_speed_perturbation": True,
                                }
                                recog_name = prefix_name + "/" + network_module_mem_v10 + f"_nolinpos_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}/cycle_{num_cycles // 11}"
                                res_conv = eval_model(
                                    training_name=recog_name + f"_{num_cycles}",
                                    train_job=train_job,
                                    train_args=train_args_recog,
                                    train_data=train_data,
                                    decoder_config=rasr_config_memristor,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    result_dict=res_conv,
                                    decoder_module="ctc.decoder.rasr_ctc_v1_batched",
                                    prior_scales=[memristor_prior],
                                    # prior_scales=[best_params_job.out_optimal_parameters[1]],
                                    lm_scales=[memristor_lm],
                                    # lm_scales=[(best_params_job.out_optimal_parameters[0], "best")],
                                    use_gpu=True,
                                    import_memristor=True,
                                    extra_forward_config={
                                        "batch_size": 3500000 if not weight_bit in [8] else 2500000,
                                    },
                                    run_best_4=False,
                                    run_best=False,
                                    prior_args=prior_args,
                                    run_search_on_hpc=False,
                                    run_rasr=True,
                                    split_mem_init=True,
                                )
                                if num_cycles == max_cycles:
                                    generate_report(results=res_conv, exp_name=recog_name)
                                    memristor_report[recog_name] = copy.deepcopy(res_conv)

                            learn_pos_emb_cfg = ConformerPosEmbConfig(
                                learnable_pos_emb=True,
                                rel_pos_clip=16,
                                with_linear_pos=True,
                                with_pos_bias=True,
                                separate_pos_emb_per_head=True,
                                pos_emb_dropout=0.0,
                            )
                            model_config_learn_pos = copy.deepcopy(model_config)
                            model_config_learn_pos.pos_emb_config = learn_pos_emb_cfg
                            train_args = {
                                "config": train_config_24gbgpu,
                                "network_module": network_module_mem_v9,
                                "net_args": {"model_config_dict": asdict(model_config_learn_pos)},
                                "debug": False,
                                "post_config": {"num_workers_per_gpu": 8},
                                "use_speed_perturbation": True,
                            }
                            training_name = prefix_name + "/" + network_module_mem_v9 + f"_learnpos_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}"
                            train_job = training(training_name, train_data, train_args, num_epochs=epochs,
                                **default_returnn)
                            if not os.path.exists(
                                f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                                train_job.rqmt['cpu'] = 8
                                train_job.hold()
                                train_job.move_to_hpc = True
                            # train_job.rqmt["gpu_mem"] = 48
                            results = {}
                            results, best_params_job = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args,
                                train_data=train_data,
                                decoder_config=as_training_rasr_config,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                result_dict=results,
                                decoder_module="ctc.decoder.rasr_ctc_v1",
                                prior_scales=rasr_prior_scales,
                                lm_scales=rasr_lm_scales,
                                import_memristor=True,
                                get_best_params=True,
                                run_rasr=True,
                                run_best_4=False,
                                run_best=False,
                                # split_mem_init=True,
                            )
                            generate_report(results=results, exp_name=training_name + "/non_memristor")
                            memristor_report[training_name] = results

                            res_conv = {}
                            for num_cycles in range(1, max_cycles + 1):
                                recog_dac_settings = DacAdcHardwareSettings(
                                    input_bits=8,
                                    output_precision_bits=4,
                                    output_range_bits=4,
                                    hardware_input_vmax=0.6,
                                    hardware_output_current_scaling=8020.0,
                                )
                                model_config_recog = copy.deepcopy(model_config_learn_pos)
                                model_config_recog.converter_hardware_settings = recog_dac_settings
                                model_config_recog.num_cycles = num_cycles

                                prior_args = {
                                    "config": train_config_24gbgpu,
                                    "network_module": network_module_mem_v9,
                                    "net_args": {"model_config_dict": asdict(model_config_learn_pos)},
                                    "debug": False,
                                    "post_config": {"num_workers_per_gpu": 8},
                                    "use_speed_perturbation": True,
                                }

                                train_args_recog = {
                                    "config": train_config_24gbgpu,
                                    "network_module": network_module_mem_v10,
                                    "net_args": {"model_config_dict": asdict(model_config_recog)},
                                    "debug": False,
                                    "post_config": {"num_workers_per_gpu": 8},
                                    "use_speed_perturbation": True,
                                }
                                recog_name = prefix_name + "/" + network_module_mem_v10 + f"_learnpos_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}/cycle_{num_cycles // 11}"
                                res_conv = eval_model(
                                    training_name=recog_name + f"_{num_cycles}",
                                    train_job=train_job,
                                    train_args=train_args_recog,
                                    train_data=train_data,
                                    decoder_config=rasr_config_memristor,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    result_dict=res_conv,
                                    decoder_module="ctc.decoder.rasr_ctc_v1_batched",
                                    prior_scales=[memristor_prior],
                                    lm_scales=[memristor_lm],
                                    use_gpu=True,
                                    import_memristor=True,
                                    extra_forward_config={
                                        "batch_size": 3500000 if not weight_bit in [8] else 2500000,
                                    },
                                    run_best_4=False,
                                    run_best=False,
                                    prior_args=prior_args,
                                    run_search_on_hpc=False,
                                    run_rasr=True,
                                    split_mem_init=True,
                                    search_gpu=48,
                                )
                                if num_cycles == max_cycles:
                                    generate_report(results=res_conv, exp_name=recog_name)
                                    memristor_report[recog_name] = copy.deepcopy(res_conv)


                            if weight_bit in [4, 8]:
                                learn_pos_emb_cfg = ConformerPosEmbConfig(
                                    learnable_pos_emb=True,
                                    rel_pos_clip=16,
                                    with_linear_pos=False,
                                    with_pos_bias=True,
                                    separate_pos_emb_per_head=True,
                                    pos_emb_dropout=0.0,
                                )
                                model_config_learn_pos = copy.deepcopy(model_config)
                                model_config_learn_pos.pos_emb_config = learn_pos_emb_cfg
                                train_args = {
                                    "config": train_config_24gbgpu,
                                    "network_module": network_module_mem_v9,
                                    "net_args": {"model_config_dict": asdict(model_config_learn_pos)},
                                    "debug": False,
                                    "post_config": {"num_workers_per_gpu": 8},
                                    "use_speed_perturbation": True,
                                }
                                training_name = prefix_name + "/" + network_module_mem_v9 + f"_learnpos_nolin{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}"
                                train_job = training(training_name, train_data, train_args, num_epochs=epochs,
                                                     **default_returnn)
                                train_job.rqmt["gpu_mem"] = 48
                                train_job.rqmt["mem"] = 36
                                # if not os.path.exists(
                                #     f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                                #     train_job.rqmt['cpu'] = 8
                                #     train_job.hold()
                                #     train_job.move_to_hpc = True
                                results = {}
                                results, best_params_job = eval_model(
                                    training_name=training_name,
                                    train_job=train_job,
                                    train_args=train_args,
                                    train_data=train_data,
                                    decoder_config=as_training_rasr_config,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    result_dict=results,
                                    decoder_module="ctc.decoder.rasr_ctc_v1",
                                    prior_scales=rasr_prior_scales,
                                    lm_scales=rasr_lm_scales,
                                    import_memristor=True,
                                    get_best_params=True,
                                    run_rasr=True,
                                    run_best_4=False,
                                    run_best=False,
                                    # split_mem_init=True,
                                )
                                generate_report(results=results, exp_name=training_name + "/non_memristor")
                                memristor_report[training_name] = results

                                res_conv = {}
                                for num_cycles in range(1, max_cycles + 1):
                                    recog_dac_settings = DacAdcHardwareSettings(
                                        input_bits=8,
                                        output_precision_bits=4,
                                        output_range_bits=4,
                                        hardware_input_vmax=0.6,
                                        hardware_output_current_scaling=8020.0,
                                    )
                                    model_config_recog = copy.deepcopy(model_config_learn_pos)
                                    model_config_recog.converter_hardware_settings = recog_dac_settings
                                    model_config_recog.num_cycles = num_cycles

                                    prior_args = {
                                        "config": train_config_24gbgpu,
                                        "network_module": network_module_mem_v9,
                                        "net_args": {"model_config_dict": asdict(model_config_learn_pos)},
                                        "debug": False,
                                        "post_config": {"num_workers_per_gpu": 8},
                                        "use_speed_perturbation": True,
                                    }

                                    train_args_recog = {
                                        "config": train_config_24gbgpu,
                                        "network_module": network_module_mem_v10,
                                        "net_args": {"model_config_dict": asdict(model_config_recog)},
                                        "debug": False,
                                        "post_config": {"num_workers_per_gpu": 8},
                                        "use_speed_perturbation": True,
                                    }
                                    recog_name = prefix_name + "/" + network_module_mem_v10 + f"_learnpos_nolin{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}/cycle_{num_cycles // 11}"
                                    res_conv = eval_model(
                                        training_name=recog_name + f"_{num_cycles}",
                                        train_job=train_job,
                                        train_args=train_args_recog,
                                        train_data=train_data,
                                        decoder_config=rasr_config_memristor,
                                        dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                        result_dict=res_conv,
                                        decoder_module="ctc.decoder.rasr_ctc_v1_batched",
                                        prior_scales=[memristor_prior],
                                        lm_scales=[memristor_lm],
                                        use_gpu=True,
                                        import_memristor=True,
                                        extra_forward_config={
                                            "batch_size": 3500000 if not weight_bit in [8] else 2500000,
                                        },
                                        run_best_4=False,
                                        run_best=False,
                                        prior_args=prior_args,
                                        run_search_on_hpc=False,
                                        run_rasr=True,
                                        split_mem_init=True,
                                    )
                                    if num_cycles == max_cycles:
                                        generate_report(results=res_conv, exp_name=recog_name)
                                        memristor_report[recog_name] = copy.deepcopy(res_conv)

                            train_config_24gbgpu = {
                                "optimizer": {
                                    "class": "radam",
                                    "epsilon": 1e-12,
                                    "weight_decay": 1e-2,
                                    "decoupled_weight_decay": True,
                                },
                                "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs // 2 - 20)))
                                                  + list(np.linspace(5e-4, 5e-5, (epochs // 2 - 20)))
                                                  + list(np.linspace(5e-5, 1e-7, 40)),
                                #############
                                "batch_size": 360 * 16000,
                                "max_seq_length": {"audio_features": 35 * 16000},
                                "accum_grad_multiple_step": 1,
                                "gradient_clip_norm": 1.0,
                                "seed": seed,
                                "torch_amp_options": {"dtype": "bfloat16"},
                            }
                            dropout_pos_emb_cfg = ConformerPosEmbConfig(
                                learnable_pos_emb=False,
                                rel_pos_clip=16,
                                with_linear_pos=True,
                                with_pos_bias=True,
                                separate_pos_emb_per_head=True,
                                pos_emb_dropout=0.1,
                            )
                            model_config_dropout = copy.deepcopy(model_config)
                            model_config_dropout.pos_emb_config = dropout_pos_emb_cfg
                            train_args = {
                                "config": train_config_24gbgpu,
                                "network_module": network_module_mem_v9,
                                "net_args": {"model_config_dict": asdict(model_config_dropout)},
                                "debug": False,
                                "post_config": {"num_workers_per_gpu": 8},
                                "use_speed_perturbation": True,
                            }
                            training_name = prefix_name + "/" + network_module_mem_v9 + f"_drop0.1_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}"
                            train_job = training(training_name, train_data, train_args, num_epochs=epochs,
                                **default_returnn)
                            if not os.path.exists(
                                f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                                train_job.rqmt['cpu'] = 8
                                train_job.hold()
                                train_job.move_to_hpc = True
                            results = {}
                            results, best_params_job = eval_model(
                                training_name=training_name,
                                train_job=train_job,
                                train_args=train_args,
                                train_data=train_data,
                                decoder_config=as_training_rasr_config,
                                dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                result_dict=results,
                                decoder_module="ctc.decoder.rasr_ctc_v1",
                                prior_scales=rasr_prior_scales,
                                lm_scales=rasr_lm_scales,
                                import_memristor=True,
                                get_best_params=True,
                                run_rasr=True,
                                run_best_4=False,
                                run_best=False,
                            )
                            generate_report(results=results, exp_name=training_name + "/non_memristor")
                            memristor_report[training_name] = results
                            for prec, ran in [(4, 4), (4, 8), (8, 8)]:
                                if seed > 0:
                                    continue
                                res_conv = {}
                                recog_dac_settings = DacAdcHardwareSettings(
                                    input_bits=8,
                                    output_precision_bits=prec,
                                    output_range_bits=ran,
                                    hardware_input_vmax=0.6,
                                    hardware_output_current_scaling=8020.0,
                                )
                                #for num_cycles in range(1, max_cycles + 1):
                                for num_cycles in range(1, max_cycles+1):
                                    model_config_recog = copy.deepcopy(model_config_dropout)
                                    model_config_recog.converter_hardware_settings = recog_dac_settings
                                    model_config_recog.num_cycles = num_cycles

                                    prior_args = {
                                        "config": train_config_24gbgpu,
                                        "network_module": network_module_mem_v9,
                                        "net_args": {"model_config_dict": asdict(model_config_dropout)},
                                        "debug": False,
                                        "post_config": {"num_workers_per_gpu": 8},
                                        "use_speed_perturbation": True,
                                    }

                                    train_args_recog = {
                                        "config": train_config_24gbgpu,
                                        "network_module": network_module_mem_v10,
                                        "net_args": {"model_config_dict": asdict(model_config_recog)},
                                        "debug": False,
                                        "post_config": {"num_workers_per_gpu": 8},
                                        "use_speed_perturbation": True,
                                    }
                                    recog_name = prefix_name + "/" + network_module_mem_v10 + f"_drop0.1_{ran}_{prec}_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}/cycle_{num_cycles // 11}"
                                    res_conv = eval_model(
                                        training_name=recog_name + f"_{num_cycles}",
                                        train_job=train_job,
                                        train_args=train_args_recog,
                                        train_data=train_data,
                                        decoder_config=rasr_config_memristor,
                                        dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                        result_dict=res_conv,
                                        decoder_module="ctc.decoder.rasr_ctc_v1_batched",
                                        prior_scales=[memristor_prior],
                                        # prior_scales=[best_params_job.out_optimal_parameters[1]],
                                        lm_scales=[memristor_lm],
                                        # lm_scales=[(best_params_job.out_optimal_parameters[0], "best")],
                                        use_gpu=True,
                                        import_memristor=True,
                                        extra_forward_config={
                                            "batch_size": 3500000 if not weight_bit in [8] else 2500000,
                                        },
                                        run_best_4=False,
                                        run_best=False,
                                        prior_args=prior_args,
                                        run_search_on_hpc=False,
                                        run_rasr=True,
                                        split_mem_init=True,
                                    )
                                    if num_cycles == max_cycles:
                                        generate_report(results=res_conv, exp_name=recog_name)
                                        memristor_report[recog_name] = copy.deepcopy(res_conv)

                            if weight_bit not in [4, 8] or seed > 0:
                                continue
                            for norm in [0.05, 0.01, 0.001]:
                                model_config_weight_decay = MemristorModelTrainConfigV10(
                                    feature_extraction_config=fe_config,
                                    frontend_config=frontend_config_dim,
                                    specaug_config=specaug_config_full,
                                    label_target_size=vocab_size_without_blank,
                                    conformer_size=dim,
                                    num_layers=12,
                                    num_heads=8,
                                    ff_dim=dim * 4,
                                    att_weights_dropout=0.1,
                                    conv_dropout=0.1,
                                    ff_dropout=0.1,
                                    mhsa_dropout=0.1,
                                    conv_kernel_size=31,
                                    final_dropout=0.1,
                                    specauc_start_epoch=1,
                                    weight_quant_dtype="qint8",
                                    weight_quant_method="per_tensor_symmetric",
                                    activation_quant_dtype="qint8",
                                    activation_quant_method="per_tensor_symmetric",
                                    dot_quant_dtype="qint8",
                                    dot_quant_method="per_tensor_symmetric",
                                    Av_quant_dtype="qint8",
                                    Av_quant_method="per_tensor_symmetric",
                                    moving_average=None,
                                    weight_bit_prec=weight_bit,
                                    activation_bit_prec=activation_bit,
                                    quantize_output=False,
                                    converter_hardware_settings=prior_train_dac_settings,
                                    quant_in_linear=True,
                                    num_cycles=0,
                                    correction_settings=None,
                                    weight_noise_func=None,
                                    weight_noise_values=None,
                                    weight_noise_start_epoch=None,
                                    pos_emb_config=pos_emb_cfg,
                                    module_list=["ff", "conv", "mhsa", "ff"],
                                    module_scales=[0.5, 1.0, 1.0, 0.5],
                                    aux_ctc_loss_layers=None,
                                    aux_ctc_loss_scales=None,
                                    dropout_broadcast_axes=None,
                                    l2_norm=norm,
                                )
                                train_args = {
                                    "config": train_config_24gbgpu,
                                    "network_module": network_module_mem_v10_weight_decay,
                                    "net_args": {"model_config_dict": asdict(model_config_weight_decay)},
                                    "debug": False,
                                    "post_config": {"num_workers_per_gpu": 8},
                                    "use_speed_perturbation": True,
                                }
                                training_name = prefix_name + "/" + network_module_mem_v10_weight_decay + f"_l2{norm}_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}"
                                train_job = training(training_name, train_data, train_args, num_epochs=epochs,
                                    **default_returnn)
                                if not os.path.exists(
                                    f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                                    train_job.rqmt['cpu'] = 8
                                    train_job.hold()
                                    train_job.move_to_hpc = True
                                # train_job.rqmt['gpu_mem'] = 24
                                # train_job.rqmt['mem'] = 36
                                results = {}
                                results, best_params_job = eval_model(
                                    training_name=training_name,
                                    train_job=train_job,
                                    train_args=train_args,
                                    train_data=train_data,
                                    decoder_config=as_training_rasr_config,
                                    dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                    result_dict=results,
                                    decoder_module="ctc.decoder.rasr_ctc_v1",
                                    prior_scales=rasr_prior_scales,
                                    lm_scales=rasr_lm_scales,
                                    import_memristor=True,
                                    get_best_params=True,
                                    run_rasr=True,
                                    run_best_4=False,
                                    run_best=False,
                                )
                                generate_report(results=results, exp_name=training_name + "/non_memristor")
                                memristor_report[training_name] = results
                                for prec, ran in [(4, 4), (4, 8)]:
                                    if seed > 0:
                                        continue
                                    res_conv = {}
                                    recog_dac_settings = DacAdcHardwareSettings(
                                        input_bits=8,
                                        output_precision_bits=prec,
                                        output_range_bits=ran,
                                        hardware_input_vmax=0.6,
                                        hardware_output_current_scaling=8020.0,
                                    )
                                    # for num_cycles in range(1, max_cycles + 1):
                                    for num_cycles in range(1, 3):
                                        model_config_weight_decay = copy.deepcopy(model_config)
                                        model_config_weight_decay.converter_hardware_settings = recog_dac_settings
                                        model_config_weight_decay.num_cycles = num_cycles

                                        prior_args = {
                                            "config": train_config_24gbgpu,
                                            "network_module": network_module_mem_v9,
                                            "net_args": {"model_config_dict": asdict(model_config_weight_decay)},
                                            "debug": False,
                                            "post_config": {"num_workers_per_gpu": 8},
                                            "use_speed_perturbation": True,
                                        }

                                        train_args_recog = {
                                            "config": train_config_24gbgpu,
                                            "network_module": network_module_mem_v10,
                                            "net_args": {"model_config_dict": asdict(model_config_weight_decay)},
                                            "debug": False,
                                            "post_config": {"num_workers_per_gpu": 8},
                                            "use_speed_perturbation": True,
                                        }
                                        recog_name = prefix_name + "/" + network_module_mem_v10 + f"_l2{norm}_{ran}_{prec}_{epochs // 10}eps_{dim}dim_{weight_bit}_{activation_bit}_seed_{seed}/cycle_{num_cycles // 11}"
                                        res_conv = eval_model(
                                            training_name=recog_name + f"_{num_cycles}",
                                            train_job=train_job,
                                            train_args=train_args_recog,
                                            train_data=train_data,
                                            decoder_config=rasr_config_memristor,
                                            dev_dataset_tuples={"dev-other": dev_dataset_tuples["dev-other"]},
                                            result_dict=res_conv,
                                            decoder_module="ctc.decoder.rasr_ctc_v1_batched",
                                            prior_scales=[memristor_prior],
                                            # prior_scales=[best_params_job.out_optimal_parameters[1]],
                                            lm_scales=[memristor_lm],
                                            # lm_scales=[(best_params_job.out_optimal_parameters[0], "best")],
                                            use_gpu=True,
                                            import_memristor=True,
                                            extra_forward_config={
                                                "batch_size": 3500000 if not weight_bit in [8] else 2500000,
                                            },
                                            run_best_4=False,
                                            run_best=False,
                                            prior_args=prior_args,
                                            run_search_on_hpc=False,
                                            run_rasr=True,
                                            split_mem_init=True,
                                        )
                                        if num_cycles == 2:
                                            generate_report(results=res_conv, exp_name=recog_name)
                                            memristor_report[recog_name] = copy.deepcopy(res_conv)

                    if epochs == 1000:
                        training_name = (
                            prefix_name
                            + "/"
                            + network_module_mem_v9
                            + f"_{epochs//10}eps_{dim}dim_{weight_bit}_{activation_bit}_seeds_combined_cycle"
                        )
                        if len(res_seeds_total) > 0:
                            generate_report(results=res_seeds_total, exp_name=training_name)
                            memristor_report[training_name] = copy.deepcopy(res_seeds_total)

    tk.register_report("reports/lbs/memristor_report_phon", partial(build_qat_report, memristor_report), required=memristor_report, update_frequency=400)
    tk.register_report("reports/lbs/v2/memristor_phon", partial(build_qat_report_v2, memristor_report),
                       required=memristor_report, update_frequency=400)