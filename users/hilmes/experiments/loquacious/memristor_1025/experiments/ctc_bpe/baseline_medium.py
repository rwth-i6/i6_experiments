from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast
from functools import partial

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset, build_short_dev_dataset
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon, get_bpe_bliss_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm, get_arpa_lm_config
from ...pipeline import training
from ...report import generate_report
from ...rasr_recog_config import get_tree_timesync_recog_config, get_no_op_label_scorer_config
import os

from ...tune_eval import eval_model, build_base_report

def bpe_loq_medium_1025():
    BPE_SIZE = 128
    loquacious_key = "train.medium"
    PARTITION_EPOCH = 25

    prefix_name = f"experiments/loquacious/medium/memristor_1025/bpe_ctc_bpe/{BPE_SIZE}"
    train_settings_4k = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        # training
        train_partition_epoch=PARTITION_EPOCH,
        train_seq_ordering="laplace:.4000",
    )

    short_dev_dataset_tuples = {
            "dev": build_short_dev_dataset(train_settings_4k)
    }

    dev_dataset_tuples = {}
    for testset in ["dev.commonvoice", "dev.librispeech", "dev.voxpopuli", "dev.yodas"]:
        dev_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings_4k,
        )

    test_dataset_tuples = {}
    for testset in ["test.commonvoice", "test.librispeech", "test.voxpopuli", "test.yodas"]:
        test_dataset_tuples[testset] = build_test_dataset(
            dataset_key=testset,
            settings=train_settings_4k,
        )

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.rasr_ctc_v1 import DecoderConfig as RasrDecoderConfig

    from ...pytorch_networks.ctc.memristor_1025.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1_cfg import SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig, LogMelFeatureExtractionV1Config, ConformerPosEmbConfig

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
        max_dim_feat=16,
        num_repeat_feat=5,
    )

    frontend_config_sub4 = VGG4LayerActFrontendV1Config_mod(
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

    pos_emb_cfg = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )

    network_module_pos_enc_v1 = "ctc.memristor_1025.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"

    global_train_args = {
        "debug": False,
        "use_speed_perturbation": True,
        "post_config": {"num_workers_per_gpu": 4},
    }

    train_data_bpe = build_bpe_training_datasets(
        prefix=prefix_name,
        bpe_size=BPE_SIZE,
        settings=train_settings_4k,
        use_postfix=False,
        loquacious_key=loquacious_key,
    )

    label_datastream_bpe = cast(LabelDatastream, train_data_bpe.datastreams["labels"])
    vocab_size_without_blank = label_datastream_bpe.vocab_size

    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub4,
        specaug_config=specaug_config,
        pos_emb_config=pos_emb_cfg,
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
        dropout_broadcast_axes=None,
        module_list=["ff", "conv", "mhsa", "ff"],
        module_scales=[0.5, 1.0, 1.0, 0.5],
        aux_ctc_loss_layers=None,
        aux_ctc_loss_scales=None,
    )

    recog_rasr_config, recog_rasr_post_config = get_tree_timesync_recog_config(
        lexicon_file=get_bpe_bliss_lexicon(bpe_size=BPE_SIZE, add_blank=True, loquacious_key=loquacious_key),
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        blank_index=vocab_size_without_blank,
        max_beam_size=4096,
        score_threshold=20.0,
        logfile_suffix="recog",
        lm_config=get_arpa_lm_config("default",
                                     get_bpe_bliss_lexicon(bpe_size=BPE_SIZE, add_blank=True, loquacious_key=loquacious_key), scale=0.0),
    )

    as_training_rasr_config = RasrDecoderConfig(
        rasr_config_file=recog_rasr_config,
        rasr_post_config=recog_rasr_post_config,
        blank_log_penalty=None,
        prior_scale=0.0,  # this will be overwritten internally
        prior_file=None,
        turn_off_quant="leave_as_is",
    )
    rasr_config_memristor = copy.deepcopy(as_training_rasr_config)
    rasr_config_memristor.turn_off_quant = False

    rasr_prior_scales = [0.2, 0.3, 0.4, 0.5]
    rasr_lm_scales = [0.4, 0.5, 0.6, 0.7, 0.8]

    full_results = {}
    for epochs in [1000]:
        train_config_24gbgpu_amp = {
            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
            "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 20) // 2))
                              + list(np.linspace(5e-4, 5e-5, (epochs - 20) // 2))
                              + list(np.linspace(5e-5, 1e-7, 20)),
            #############
            "batch_size": 240 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
            "torch_amp_options": {"dtype": "bfloat16"},
            "gradient_clip_norm": 1.0,
        }
        train_args = copy.deepcopy(global_train_args)
        train_args["network_module"] = network_module_pos_enc_v1
        train_args["net_args"] = {"model_config_dict": asdict(model_config)}
        train_args["config"] = train_config_24gbgpu_amp

        training_name = prefix_name + "/" + network_module_pos_enc_v1 + f".512dim_sub{4}_48gbgpu_{epochs//PARTITION_EPOCH}eps_sp_lp_fullspec_gradnorm_smallbatch"
        train_job = training(training_name, train_data_bpe, train_args, num_epochs=epochs, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
            train_job.rqmt["cpu"] = 12
            train_job.hold()
            train_job.move_to_hpc = True

        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe,
            decoder_config=as_training_rasr_config,
            dev_dataset_tuples=short_dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.rasr_ctc_v1",
            prior_scales=rasr_prior_scales,
            lm_scales=rasr_lm_scales,
            run_rasr=True,
            run_best_4=False,
            run_best=False,
            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            run_test=True,
            loss_name="dev_loss_ctc_loss_layer12"
        )
        full_results[training_name + "_full_dev"] = {"dev_all": results.pop(training_name + f"/{epochs}" + "_dev_all", None)}
        full_results[training_name + "_full_test"] = {"test_all": results.pop(training_name + f"/{epochs}" + "_test_all", None)}
        generate_report(results=results, exp_name=training_name)
        full_results[training_name] = copy.deepcopy(results)

    network_module_mem_v9 = "ctc.memristor_1025.memristor_v9"
    network_module_mem_v10 = "ctc.memristor_1025.memristor_v10"

    from torch_memristor.memristor_modules import DacAdcHardwareSettings, CycleCorrectionSettings
    train_dac_settings = DacAdcHardwareSettings(
        input_bits=0,
        output_precision_bits=0,
        output_range_bits=0,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    from ...pytorch_networks.ctc.memristor_1025.memristor_v8_cfg import QuantModelTrainConfigV8 as MemristorModelTrainConfigV8

    global_model_config = MemristorModelTrainConfigV8(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub4,
        specaug_config=specaug_config,
        pos_emb_config=pos_emb_cfg,
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
        specauc_start_epoch=11,
        weight_quant_dtype="qint8",
        weight_quant_method="per_tensor_symmetric",
        activation_quant_dtype="qint8",
        activation_quant_method="per_tensor_symmetric",
        dot_quant_dtype="qint8",
        dot_quant_method="per_tensor_symmetric",
        Av_quant_dtype="qint8",
        Av_quant_method="per_tensor_symmetric",
        moving_average=None,
        weight_bit_prec=0, # will be filled out in loop
        activation_bit_prec=0,  # will be filled out in loop
        quantize_output=False,
        converter_hardware_settings=train_dac_settings,
        quant_in_linear=True,
        num_cycles=0,
        correction_settings=None,
        weight_noise_func=None,
        weight_noise_values=None,
        weight_noise_start_epoch=None,
        module_list=["ff", "conv", "mhsa", "ff"],
        module_scales=[0.5, 1.0, 1.0, 0.5],
        aux_ctc_loss_layers=None,
        aux_ctc_loss_scales=None,
        dropout_broadcast_axes=None,
    )
    for epochs in [1000]:
        for activation_bit in [8]:
            # for weight_bit in [3, 4, 5, 6, 7, 8]:
            for weight_bit in [3, 4, 5, 8]:
                res_seeds_total = {}
                for seed in range(3):
                    train_config_24gbgpu_amp = {
                        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
                        "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 20) // 2))
                                          + list(np.linspace(5e-4, 5e-5, (epochs - 20) // 2))
                                          + list(np.linspace(5e-5, 1e-7, 20)),
                        #############
                        "batch_size": 240 * 16000,
                        "max_seq_length": {"audio_features": 35 * 16000},
                        "accum_grad_multiple_step": 1,
                        "torch_amp_options": {"dtype": "bfloat16"},
                        "gradient_clip_norm": 1.0,
                        "seed": seed,
                    }
                    model_config = copy.deepcopy(global_model_config)
                    model_config.weight_bit_prec = weight_bit
                    model_config.activation_bit_prec = activation_bit
                    train_args = copy.deepcopy(global_train_args)
                    train_args["net_args"] = {"model_config_dict": asdict(model_config)}
                    train_args["config"] = train_config_24gbgpu_amp
                    train_args["network_module"] = network_module_mem_v9

                    training_name = prefix_name + "/" + network_module_mem_v9 + f"_{epochs//PARTITION_EPOCH}eps_{weight_bit}_{activation_bit}_seed_{seed}"

                    train_job = training(training_name, train_data_bpe, train_args, num_epochs=epochs,
                                         **default_returnn)
                    if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                        train_job.rqmt["cpu"] = 12
                        train_job.hold()
                        train_job.move_to_hpc = True

                    results = {}
                    results, best_params_job = eval_model(
                        training_name=training_name,
                        train_job=train_job,
                        train_args=train_args,
                        train_data=train_data_bpe,
                        decoder_config=as_training_rasr_config,
                        dev_dataset_tuples=short_dev_dataset_tuples,
                        result_dict=results,
                        decoder_module="ctc.decoder.rasr_ctc_v1",
                        prior_scales=rasr_prior_scales,
                        lm_scales=rasr_lm_scales,
                        import_memristor=True,
                        get_best_params=True,
                        run_rasr=True,
                        run_best_4=False,
                        run_best=False,
                        test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
                        run_test=True,
                    )
                    full_results[training_name + "_full_dev"] = {"dev_all": results.pop(training_name + f"/{epochs}" + "_dev_all", None)}
                    full_results[training_name + "_full_test"] = {"test_all": results.pop(training_name + f"/{epochs}" + "_test_all", None)}
                    generate_report(results=results, exp_name=training_name + "/non_memristor")
                    full_results[training_name] = results

                    res_conv = {}
                    recog_dac_settings = DacAdcHardwareSettings(
                        input_bits=8,
                        output_precision_bits=4,
                        output_range_bits=4,
                        hardware_input_vmax=0.6,
                        hardware_output_current_scaling=8020.0,
                    )
                    for num_cycles in range(1, 3):
                        model_config_recog = copy.deepcopy(model_config)
                        model_config_recog.converter_hardware_settings = recog_dac_settings
                        model_config_recog.num_cycles = num_cycles

                        prior_args = copy.deepcopy(train_args)
                        train_args_recog = copy.deepcopy(train_args)
                        train_args_recog["net_args"] = {"model_config_dict": asdict(model_config_recog)}
                        train_args_recog["network_module"] = network_module_mem_v10

                        recog_name = prefix_name + "/" + network_module_mem_v10 + f"_{weight_bit}_{activation_bit}_seed_{seed}/cycle_{num_cycles // 11}"
                        res_conv = eval_model(
                            training_name=recog_name + f"_{num_cycles}",
                            train_job=train_job,
                            train_args=train_args_recog,
                            train_data=train_data_bpe,
                            decoder_config=rasr_config_memristor,
                            dev_dataset_tuples=short_dev_dataset_tuples,
                            result_dict=res_conv,
                            decoder_module="ctc.decoder.rasr_ctc_v1",
                            prior_scales=[best_params_job.out_optimal_parameters[1]],
                            lm_scales=[(best_params_job.out_optimal_parameters[0], "best")],
                            use_gpu=True,
                            import_memristor=True,
                            extra_forward_config={
                                "batch_size": 7000000 * 2,
                            },
                            run_best_4=False,
                            run_best=False,
                            prior_args=prior_args,
                            run_search_on_hpc=False,
                            run_rasr=True,
                            split_mem_init=True,
                        )
                    res_seeds_total.update(res_conv)
                    recog_name = prefix_name + "/" + network_module_mem_v9 + f"_{weight_bit}_{activation_bit}_seed_{seed}_cycle"
                    generate_report(results=res_conv, exp_name=recog_name)
                    full_results[recog_name] = copy.deepcopy(res_conv)

                continue  # TODO: remove once pipeline is working
                recog_name = prefix_name + "/" + network_module_mem_v9 + f"_{weight_bit}_{activation_bit}_seeds_combined_cycle"
                generate_report(results=res_seeds_total, exp_name=recog_name)
                full_results[recog_name] = copy.deepcopy(res_seeds_total)


    tk.register_report("reports/loquacious/medium_baseline_report", partial(build_base_report, full_results, False), required=full_results, update_frequency=600)

