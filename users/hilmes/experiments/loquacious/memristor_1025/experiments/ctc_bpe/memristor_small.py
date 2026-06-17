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

from ...tune_eval import eval_model, build_qat_report, build_qat_report_v2, build_qat_report_v2


def collect_checkpoint_results(results, full_results, training_name, epochs, datasets, with_best=False):
    checkpoints = [("", f"/{epochs}")]
    if with_best:
        checkpoints.append(("_best", "/best"))
    for suffix, ckpt in checkpoints:
        full_results[training_name + "_full_dev" + suffix] = {
            "dev_all": results.pop(training_name + ckpt + "_dev_all", None)
        }
        full_results[training_name + "_full_test" + suffix] = {
            "test_all": results.pop(training_name + ckpt + "_test_all", None)
        }
        for set_name in datasets:
            full_results[training_name + f"_dev_{set_name}" + suffix] = {
                set_name: results.pop(training_name + ckpt + f"/dev.{set_name}", None)
            }
            full_results[training_name + f"_test_{set_name}" + suffix] = {
                set_name: results.pop(training_name + ckpt + f"/test.{set_name}", None)
            }


def bpe_loq_small_memristor_width_1125():

    prefix_name = "experiments/loquacious/small/memristor_1025/bpe_ctc_bpe/128/width"

    loquacious_key = "train.small"

    train_settings_4k = DatasetSettings(
        preemphasis=0.97,
        peak_normalization=True,
        # training
        train_partition_epoch=5,
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
    pos_emb_cfg = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )

    global_train_args = {
        "debug": False,
        "use_speed_perturbation": True,
        "post_config": {"num_workers_per_gpu": 4},
    }
    network_module_mem_v9 = "ctc.memristor_1025.memristor_v9"
    network_module_mem_v11 = "ctc.memristor_1025.memristor_v11"

    train_data_bpe = build_bpe_training_datasets(
        prefix=prefix_name,
        bpe_size=128,
        settings=train_settings_4k,
        use_postfix=False,
        loquacious_key=loquacious_key,
    )

    label_datastream_bpe = cast(LabelDatastream, train_data_bpe.datastreams["labels"])
    vocab_size_without_blank = label_datastream_bpe.vocab_size

    recog_rasr_config, recog_rasr_post_config = get_tree_timesync_recog_config(
        lexicon_file=get_bpe_bliss_lexicon(bpe_size=128, add_blank=True, loquacious_key=loquacious_key),
        collapse_repeated_labels=True,
        label_scorer_config=get_no_op_label_scorer_config(),
        blank_index=vocab_size_without_blank,
        max_beam_size=4096,
        score_threshold=20.0,
        logfile_suffix="recog",
        lm_config=get_arpa_lm_config("default",
                                     get_bpe_bliss_lexicon(bpe_size=128, add_blank=True, loquacious_key=loquacious_key), scale=0.0),
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

    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_quant_v1 import DecoderConfig as GreedyDecoderConfig
    as_training_greedy_decoder_config = GreedyDecoderConfig(
        returnn_vocab=label_datastream_bpe.vocab,
        turn_off_quant="leave_as_is",
    )
    greedy_decoder_memristor = copy.deepcopy(as_training_greedy_decoder_config)
    greedy_decoder_memristor.turn_off_quant = False

    rasr_prior_scales = [0.3, 0.4, 0.5]
    rasr_lm_scales = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]

    full_results = {}
    from ...pytorch_networks.ctc.memristor_1025.memristor_v8_cfg import QuantModelTrainConfigV8 as MemristorModelTrainConfigV8
    from ...pytorch_networks.ctc.memristor_1025.memristor_v11_cfg import QuantModelTrainConfigV11 as MemristorModelTrainConfigV11
    from torch_memristor.memristor_modules import DacAdcHardwareSettings, CycleCorrectionSettings
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
    train_dac_settings = DacAdcHardwareSettings(
        input_bits=0,
        output_precision_bits=0,
        output_range_bits=0,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    recog_dac_settings = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=4,
        output_range_bits=4,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    posenc_dac_settings_posenc = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=1,
        output_range_bits=7,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )
    max_runs = 3

    def _make_frontend_config(out_features):
        cfg = copy.deepcopy(frontend_config_sub4)
        cfg.out_features = out_features
        return cfg

    def _make_model_config_kwargs(dim, quant_method="per_tensor_symmetric"):
        return dict(
            feature_extraction_config=fe_config,
            frontend_config=_make_frontend_config(dim),
            specaug_config=specaug_config,
            pos_emb_config=pos_emb_cfg,
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
            specauc_start_epoch=11,
            weight_quant_dtype="qint8",
            weight_quant_method=quant_method,
            activation_quant_dtype="qint8",
            activation_quant_method=quant_method,
            dot_quant_dtype="qint8",
            dot_quant_method=quant_method,
            Av_quant_dtype="qint8",
            Av_quant_method=quant_method,
            moving_average=None,
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

    def _run_cycle_loop(train_job, train_args, model_config, name_suffix, final_name_suffix,
                        prior_scales, lm_scales, batch_size, recog_network_module,
                        recog_model_config_class=None, extra_eval_kwargs=None,
                        greedy_config=None):
        res, res_greedy = {}, {}
        for num_cycles in range(1, max_runs + 1):
            if recog_model_config_class is not None:
                model_config_recog = recog_model_config_class(
                    **model_config.__dict__,
                    pos_enc_converter_hardware_settings=None,
                )
            else:
                model_config_recog = copy.deepcopy(model_config)
            model_config_recog.converter_hardware_settings = recog_dac_settings
            model_config_recog.num_cycles = num_cycles
            model_config_recog.pos_enc_converter_hardware_settings = posenc_dac_settings_posenc

            prior_args = copy.deepcopy(train_args)
            train_args_recog = copy.deepcopy(train_args)
            train_args_recog["net_args"] = {"model_config_dict": asdict(model_config_recog)}
            train_args_recog["network_module"] = recog_network_module

            recog_name = prefix_name + "/" + recog_network_module + name_suffix + f"/cycle_{num_cycles // 11}"
            kwargs = dict(
                training_name=recog_name + f"_{num_cycles}",
                train_job=train_job,
                train_args=train_args_recog,
                train_data=train_data_bpe,
                decoder_config=rasr_config_memristor,
                dev_dataset_tuples=short_dev_dataset_tuples,
                result_dict=res,
                decoder_module="ctc.decoder.rasr_ctc_v1_batched",
                prior_scales=prior_scales,
                lm_scales=lm_scales,
                use_gpu=True,
                import_memristor=True,
                extra_forward_config={"batch_size": batch_size},
                run_best_4=False,
                run_best=False,
                prior_args=prior_args,
                run_search_on_hpc=False,
                run_rasr=True,
                split_mem_init=True,
            )
            if extra_eval_kwargs:
                kwargs.update(extra_eval_kwargs)
            res = eval_model(**kwargs)

            if greedy_config is not None:
                res_greedy = eval_model(
                    training_name=recog_name + f"_{num_cycles}",
                    train_job=train_job,
                    train_args=train_args_recog,
                    train_data=train_data_bpe,
                    decoder_config=greedy_config,
                    dev_dataset_tuples=short_dev_dataset_tuples,
                    result_dict=res_greedy,
                    decoder_module="ctc.decoder.greedy_bpe_ctc_quant_v1",
                    prior_scales=[0.0],
                    lm_scales=[0.0],
                    use_gpu=True,
                    import_memristor=True,
                    extra_forward_config={"batch_size": batch_size},
                    run_best_4=False,
                    run_best=False,
                    prior_args=None,
                    with_prior=False,
                    run_search_on_hpc=False,
                    run_rasr=False,
                    split_mem_init=True,
                    **(extra_eval_kwargs or {}),
                )

            if num_cycles == max_runs:
                final_name = prefix_name + "/" + recog_network_module + final_name_suffix + "_cycle"
                generate_report(results=res, exp_name=final_name)
                full_results[final_name] = copy.deepcopy(res)
                if greedy_config is not None:
                    generate_report(results=res_greedy, exp_name=final_name + "_greedy")
                    full_results[final_name + "_greedy"] = copy.deepcopy(res_greedy)
        return res

    for epochs in [500, 1000, 1500]:
        for activation_bit in [8]:
            for weight_bit in [4]:
                for dim in [384, 512, 768, 1024, 1536, 2048]:
                    res_seeds_total = {}
                    for seed in range(3):
                        if seed > 0 and not epochs == 500:
                            continue
                        if seed > 0 and dim > 1024:
                            continue
                        train_config_24gbgpu_amp = {
                            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
                            "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 20) // 2))
                                              + list(np.linspace(5e-4, 5e-5, (epochs - 20) // 2))
                                              + list(np.linspace(5e-5, 1e-7, 20)),
                            #############
                            "batch_size": 240 * 16000 if not (epochs == 1500 and dim == 2048) else 120 * 16000,
                            "max_seq_length": {"audio_features": 35 * 16000},
                            "accum_grad_multiple_step": 1 if not (epochs == 1500 and dim == 2048) else 2,
                            "torch_amp_options": {"dtype": "bfloat16"},
                            "gradient_clip_norm": 1.0,
                            "seed": seed,
                        }
                        model_config = MemristorModelTrainConfigV8(
                            **_make_model_config_kwargs(dim),
                            weight_bit_prec=weight_bit,
                            activation_bit_prec=activation_bit,
                        )
                        train_args = copy.deepcopy(global_train_args)
                        train_args["net_args"] = {"model_config_dict": asdict(model_config)}
                        train_args["config"] = train_config_24gbgpu_amp
                        train_args["network_module"] = network_module_mem_v9

                        training_name = prefix_name + "/" + network_module_mem_v9 + f"_{epochs//5}eps_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}"
                        train_job = training(training_name, train_data_bpe, train_args, num_epochs=epochs,
                                             **default_returnn)

                        if epochs == 1500:
                            train_job.rqmt['cpu'] = 8
                            train_job.rqmt['gpu_mem'] = 48
                        else:
                            if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                                train_job.rqmt['cpu'] = 8
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
                            run_best=True,
                            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
                            run_test=True,
                            loss_name="ctc_loss_layer12"
                        )
                        collect_checkpoint_results(results, full_results, training_name, epochs,
                                                   ['yodas', 'librispeech', 'voxpopuli', 'commonvoice'],
                                                   with_best=True)
                        generate_report(results=results, exp_name=training_name + "/non_memristor")
                        full_results[training_name] = results

                        greedy_results = eval_model(
                            training_name=training_name + "/greedy",
                            train_job=train_job,
                            train_args=train_args,
                            train_data=train_data_bpe,
                            decoder_config=as_training_greedy_decoder_config,
                            dev_dataset_tuples=short_dev_dataset_tuples,
                            result_dict={},
                            decoder_module="ctc.decoder.greedy_bpe_ctc_quant_v1",
                            prior_scales=[0.0],
                            lm_scales=[0.0],
                            import_memristor=True,
                            run_rasr=False,
                            run_best_4=False,
                            run_best=False,
                            with_prior=False,
                        )
                        generate_report(results=greedy_results, exp_name=training_name + "/greedy/non_memristor")
                        full_results[training_name + "_greedy"] = greedy_results

                        name_suffix = f"_{epochs//5}eps_{dim}dim_w{weight_bit}_a{activation_bit}_seed_{seed}"
                        _run_cycle_loop(
                            train_job=train_job,
                            train_args=train_args,
                            model_config=model_config,
                            name_suffix=name_suffix,
                            final_name_suffix=name_suffix,
                            prior_scales=[0.5],
                            lm_scales=[1.0],
                            batch_size=2500000,
                            recog_network_module=network_module_mem_v11,
                            recog_model_config_class=MemristorModelTrainConfigV11,
                            extra_eval_kwargs={"search_gpu": 24},
                            greedy_config=greedy_decoder_memristor,
                        )

                        if seed == 0 and epochs == 500 and dim in [384, 512, 1024]:
                            # try better QAT settings
                            # should be worse for memristor
                            # TODO: try per channel quant
                            model_config_per = MemristorModelTrainConfigV8(
                                **_make_model_config_kwargs(dim, "per_tensor"),
                                weight_bit_prec=weight_bit,
                                activation_bit_prec=activation_bit,
                            )

                            train_args = copy.deepcopy(global_train_args)
                            train_args["net_args"] = {"model_config_dict": asdict(model_config_per)}
                            train_args["config"] = train_config_24gbgpu_amp
                            train_args["network_module"] = network_module_mem_v9

                            training_name = prefix_name + "/" + network_module_mem_v9 + f"_{epochs // 5}eps_{dim}dim_w{weight_bit}_a{activation_bit}_pertensor_seed_{seed}"
                            train_job = training(training_name, train_data_bpe, train_args, num_epochs=epochs,
                                **default_returnn)

                            train_job.rqmt['cpu'] = 8
                            train_job.rqmt['gpu_mem'] = 24

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
                                run_test=False,
                            )
                            collect_checkpoint_results(results, full_results, training_name, epochs,
                                                       ['yodas', 'librispeech', 'voxpopuli', 'commonvoice'])
                            generate_report(results=results, exp_name=training_name + "/non_memristor")
                            full_results[training_name] = results

                            greedy_results = eval_model(
                                training_name=training_name + "/greedy",
                                train_job=train_job,
                                train_args=train_args,
                                train_data=train_data_bpe,
                                decoder_config=as_training_greedy_decoder_config,
                                dev_dataset_tuples=short_dev_dataset_tuples,
                                result_dict={},
                                decoder_module="ctc.decoder.greedy_bpe_ctc_quant_v1",
                                prior_scales=[0.0],
                                lm_scales=[0.0],
                                import_memristor=True,
                                run_rasr=False,
                                run_best_4=False,
                                run_best=False,
                                with_prior=False,
                            )
                            generate_report(results=greedy_results, exp_name=training_name + "/greedy/non_memristor")
                            full_results[training_name + "_greedy"] = greedy_results

                            per_name_suffix = f"_{epochs // 5}eps_{dim}dim_w{weight_bit}_a{activation_bit}_pertensor_seed_{seed}"
                            # TODO: update to fixed prior and lm scale
                            _run_cycle_loop(
                                train_job=train_job,
                                train_args=train_args,
                                model_config=model_config_per,
                                name_suffix=per_name_suffix,
                                final_name_suffix=per_name_suffix,
                                prior_scales=[best_params_job.out_optimal_parameters[1]],
                                lm_scales=[(best_params_job.out_optimal_parameters[0], "best")],
                                batch_size=2500000 if dim <= 512 else 1000000,
                                recog_network_module=network_module_mem_v11,
                                recog_model_config_class=MemristorModelTrainConfigV11,
                                greedy_config=greedy_decoder_memristor,
                            )

    # tk.register_report("reports/loquacious/small_width", partial(build_qat_report, full_results, False), required=full_results, update_frequency=600)
    tk.register_report("reports/loquacious/v2/small_width", partial(build_qat_report_v2, full_results),
                       required=full_results, update_frequency=600)