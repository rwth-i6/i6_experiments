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
from ...report import generate_report, multi_scale_cycle_report_format
from ...rasr_recog_config import get_tree_timesync_recog_config, get_no_op_label_scorer_config
import os

from ...tune_eval import eval_model, build_qat_report, build_qat_report_v2


def bpe_loq_small_noise_0826():
    _bpe_loq_noise("small")


def bpe_loq_medium_noise_0826():
    _bpe_loq_noise("medium")


def bpe_loq_small_noise_from100eps_0826():
    """Parallel small-corpus graph: same noise finetunes, but from the 500-subep (100 full
    epoch) QAT baselines. Old baseline HW numbers come from the posadc_7_1 cycle evals."""
    baseline_prefix = "experiments/loquacious/small/memristor_1025/bpe_ctc_bpe/128"
    noise_prefix = baseline_prefix + "/noise"
    loquacious_key = "train.small"
    PARTITION_EPOCH = 5
    base_epochs = 500

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

    global_train_args = {
        "debug": False,
        "use_speed_perturbation": True,
        "post_config": {"num_workers_per_gpu": 4},
    }

    network_module_mem_v9 = "ctc.memristor_1025.memristor_v9"
    network_module_mem_v11 = "ctc.memristor_1025.memristor_v11"
    network_module_mem_v15 = "ctc.memristor_1025.memristor_v15"

    train_data_bpe = build_bpe_training_datasets(
        prefix=baseline_prefix,
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
    greedy_decoder_memristor = GreedyDecoderConfig(
        returnn_vocab=label_datastream_bpe.vocab,
        turn_off_quant=False,
    )

    from ...pytorch_networks.ctc.memristor_1025.memristor_v8_cfg import QuantModelTrainConfigV8 as MemristorModelTrainConfigV8
    from ...pytorch_networks.ctc.memristor_1025.memristor_v11_cfg import QuantModelTrainConfigV11 as MemristorModelTrainConfigV11
    from ...pytorch_networks.ctc.memristor_1025.memristor_v15_cfg import (
        QuantModelTrainConfigV15 as MemristorModelTrainConfigV15,
        GaussianWeightNoiseConfig,
        BitFlipSTEWeightNoiseConfig,
        SynaptogenPoolNoiseConfig,
        AsymmetricBitNoiseConfig,
        UniformBitNoiseConfig,
        StudentTWeightNoiseConfig,
        RelativeGaussianWeightNoiseConfig,
    )
    from torch_memristor.memristor_modules import DacAdcHardwareSettings

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
    posenc_dac_settings = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=1,
        output_range_bits=7,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )

    rasr_prior_scales = [0.3, 0.4, 0.5]
    rasr_lm_scales = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
    rasr_noise_prior_scales = [0.3, 0.4, 0.5, 0.6, 0.7]
    rasr_noise_lm_scales = [0.9, 1.0, 1.1, 1.2, 1.3]
    cycle_lm_scale = 1.0

    full_results = {}

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
        weight_bit_prec=0,  # will be filled out in loop
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

    def _make_v15_model_config(dim, weight_noise, weight_bit, activation_bit, weight_dropout=0.0):
        model_config = MemristorModelTrainConfigV15(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub4,
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
            weight_quant_method="per_tensor_symmetric",
            activation_quant_dtype="qint8",
            activation_quant_method="per_tensor_symmetric",
            dot_quant_dtype="qint8",
            dot_quant_method="per_tensor_symmetric",
            Av_quant_dtype="qint8",
            Av_quant_method="per_tensor_symmetric",
            moving_average=None,
            quantize_output=False,
            converter_hardware_settings=train_dac_settings,
            quant_in_linear=True,
            num_cycles=0,
            correction_settings=None,
            weight_noise=weight_noise,
            module_list=["ff", "conv", "mhsa", "ff"],
            module_scales=[0.5, 1.0, 1.0, 0.5],
            aux_ctc_loss_layers=None,
            aux_ctc_loss_scales=None,
            dropout_broadcast_axes=None,
            weight_bit_prec=weight_bit,
            activation_bit_prec=activation_bit,
            weight_dropout=weight_dropout,
            weight_pruning=None,
            pos_enc_converter_hardware_settings=train_dac_settings,
        )
        # v9 ignored module_list (v8-cfg default order was used); v15 honors it, so pin the
        # effective checkpoint order here or preload crashes on mismatched keys
        model_config.module_list = ["ff", "mhsa", "conv", "ff"]
        return model_config

    def _run_cycle_loop(train_job, prior_train_args, prior_model_config, name_suffix, final_name_suffix,
                        prior_scales, lm_scales, batch_size, max_runs, greedy=False):
        # prior_model_config must already have weight_noise=None; recognition adds HW settings per cycle
        res, res_greedy = {}, {}
        for num_cycles in range(1, max_runs + 1):
            model_config_recog = copy.deepcopy(prior_model_config)
            model_config_recog.converter_hardware_settings = recog_dac_settings
            model_config_recog.num_cycles = num_cycles
            model_config_recog.pos_enc_converter_hardware_settings = posenc_dac_settings

            prior_args = copy.deepcopy(prior_train_args)
            train_args_recog = copy.deepcopy(prior_train_args)
            train_args_recog["net_args"] = {"model_config_dict": asdict(model_config_recog)}
            train_args_recog["network_module"] = network_module_mem_v15

            recog_name = noise_prefix + "/" + network_module_mem_v15 + name_suffix + f"/cycle_{num_cycles // 11}"
            res = eval_model(
                training_name=recog_name + f"_{num_cycles}",
                train_job=train_job,
                train_args=train_args_recog,
                train_data=train_data_bpe,
                decoder_config=rasr_config_memristor,
                dev_dataset_tuples=short_dev_dataset_tuples,
                result_dict=res,
                decoder_module="ctc.decoder.rasr_ctc_v1_batched_fast",
                prior_scales=prior_scales,
                lm_scales=lm_scales,
                use_gpu=True,
                import_memristor="new_v3",
                extra_forward_config={"batch_size": batch_size},
                run_best_4=False,
                run_best=False,
                prior_args=prior_args,
                run_search_on_hpc=False,
                run_rasr=True,
                split_mem_init=True,
                search_gpu=24,
            )

            if greedy:
                res_greedy = eval_model(
                    training_name=recog_name + f"_{num_cycles}",
                    train_job=train_job,
                    train_args=train_args_recog,
                    train_data=train_data_bpe,
                    decoder_config=greedy_decoder_memristor,
                    dev_dataset_tuples=short_dev_dataset_tuples,
                    result_dict=res_greedy,
                    decoder_module="ctc.decoder.greedy_bpe_ctc_quant_v1_fast",
                    prior_scales=[0.0],
                    lm_scales=[0.0],
                    use_gpu=True,
                    import_memristor="new_v3",
                    extra_forward_config={"batch_size": batch_size},
                    run_best_4=False,
                    run_best=False,
                    prior_args=None,
                    with_prior=False,
                    run_search_on_hpc=False,
                    run_rasr=False,
                    split_mem_init=True,
                    search_gpu=24,
                )

            if num_cycles == max_runs:
                final_name = noise_prefix + "/" + network_module_mem_v15 + final_name_suffix + "_cycle"
                generate_report(results=res, exp_name=final_name)
                full_results[final_name] = copy.deepcopy(res)
                if greedy:
                    generate_report(results=res_greedy, exp_name=final_name + "_greedy")
                    full_results[final_name + "_greedy"] = copy.deepcopy(res_greedy)
        return res

    # --- Base QAT trainings (re-declaration of the 500-subep baseline jobs) ---
    FINETUNE_MODELS = {}
    for epochs in [base_epochs]:
        for activation_bit in [8]:
            for weight_bit in [8, 4]:
                for seed in [0]:
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

                    training_name = baseline_prefix + "/" + network_module_mem_v9 + f"_{epochs//PARTITION_EPOCH}eps_{weight_bit}_{activation_bit}_seed_{seed}"

                    train_job = training(training_name, train_data_bpe, train_args, num_epochs=epochs,
                                         **default_returnn)
                    if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                        train_job.rqmt['cpu'] = 8
                        train_job.hold()
                        train_job.move_to_hpc = True
                    FINETUNE_MODELS[training_name] = train_job.out_checkpoints[epochs]

                    # baseline tuning numbers, replicated 1:1 with baseline.py -> dedup to finished jobs
                    results, best_params_job = eval_model(
                        training_name=training_name,
                        train_job=train_job,
                        train_args=train_args,
                        train_data=train_data_bpe,
                        decoder_config=as_training_rasr_config,
                        dev_dataset_tuples=short_dev_dataset_tuples,
                        result_dict={},
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
                    # keep only the short-dev tuning numbers in the noise report
                    results.pop(training_name + f"/{epochs}" + "_dev_all", None)
                    results.pop(training_name + f"/{epochs}" + "_test_all", None)
                    for set_name in ['yodas', 'librispeech', 'voxpopuli', 'commonvoice']:
                        results.pop(training_name + f"/{epochs}" + f"/dev.{set_name}", None)
                        results.pop(training_name + f"/{epochs}" + f"/test.{set_name}", None)
                    full_results[training_name] = results

                    # old (non-fast, v11) baseline cycle numbers: the posadc_7_1 runs match the
                    # standard config (converter 8/4/4, posenc 8/1/7); replicated 1:1 -> finished jobs
                    res_conv = {}
                    for num_cycles in range(1, 6):
                        model_config_recog = MemristorModelTrainConfigV11(
                            **model_config.__dict__,
                            pos_enc_converter_hardware_settings=None,
                        )
                        model_config_recog.converter_hardware_settings = recog_dac_settings
                        model_config_recog.pos_enc_converter_hardware_settings = posenc_dac_settings
                        model_config_recog.num_cycles = num_cycles

                        prior_args_old = copy.deepcopy(train_args)
                        train_args_recog = copy.deepcopy(train_args)
                        train_args_recog["net_args"] = {"model_config_dict": asdict(model_config_recog)}
                        train_args_recog["network_module"] = network_module_mem_v11
                        train_args_recog["debug"] = False

                        recog_name = baseline_prefix + "/" + network_module_mem_v11 + f"_posadc_7_1_{epochs // PARTITION_EPOCH}eps_{weight_bit}_{activation_bit}_seed_{seed}/cycle_{num_cycles // 11}"
                        res_conv = eval_model(
                            training_name=recog_name + f"_{num_cycles}",
                            train_job=train_job,
                            train_args=train_args_recog,
                            train_data=train_data_bpe,
                            decoder_config=rasr_config_memristor,
                            dev_dataset_tuples=short_dev_dataset_tuples,
                            result_dict=res_conv,
                            decoder_module="ctc.decoder.rasr_ctc_v1_batched",
                            prior_scales=[0.5],
                            lm_scales=[1.0],
                            use_gpu=True,
                            import_memristor=True,
                            extra_forward_config={
                                "batch_size": 3500000 if not weight_bit in [8] else 2500000,
                            },
                            run_best_4=False,
                            run_best=False,
                            prior_args=prior_args_old,
                            run_search_on_hpc=False,
                            run_rasr=True,
                            test_dataset_tuples={**dev_dataset_tuples},
                            run_test=True,
                            split_mem_init=True,
                        )
                    # keep only the short-dev cycle numbers
                    res_conv = {
                        k: v for k, v in res_conv.items()
                        if not any(s in k for s in ("yodas", "commonvoice", "librispeech", "voxpopuli", "dev_all", "test_all"))
                    }
                    full_results[baseline_prefix + "/" + network_module_mem_v11 + f"_posadc_7_1_{epochs // PARTITION_EPOCH}eps_{weight_bit}_{activation_bit}_seed_{seed}_cycle"] = res_conv

                    # baseline memristor numbers on the same v15 fast path as the finetunes
                    base_prior_config = _make_v15_model_config(
                        dim=512, weight_noise=None, weight_bit=weight_bit, activation_bit=activation_bit,
                    )
                    base_prior_args = copy.deepcopy(train_args)
                    base_prior_args["network_module"] = network_module_mem_v15
                    base_prior_args["net_args"] = {"model_config_dict": asdict(base_prior_config)}
                    _run_cycle_loop(
                        train_job=train_job,
                        prior_train_args=base_prior_args,
                        prior_model_config=base_prior_config,
                        name_suffix=f"_base{epochs // PARTITION_EPOCH}eps_512dim_w{weight_bit}_a{activation_bit}_seed_{seed}",
                        final_name_suffix=f"_base{epochs // PARTITION_EPOCH}eps_512dim_w{weight_bit}_a{activation_bit}_seed_{seed}",
                        prior_scales=[0.5],
                        lm_scales=[cycle_lm_scale],
                        batch_size=2500000 if weight_bit == 8 else 3500000,
                        max_runs=5,
                        greedy=True,
                    )

    # --- Noise finetunes from the 100eps checkpoint ---
    noise_configs = [
        (GaussianWeightNoiseConfig(dev=0.01, start_epoch=1), "gauss0.01_ep1"),
        (GaussianWeightNoiseConfig(dev=0.035, start_epoch=1), "gauss0.035_ep1"),
        (GaussianWeightNoiseConfig(dev=0.05, start_epoch=1), "gauss0.05_ep1"),
        (GaussianWeightNoiseConfig(dev=0.06, start_epoch=1), "gauss0.06_ep1"),
        (GaussianWeightNoiseConfig(dev=0.1, start_epoch=1), "gauss0.1_ep1"),
        (BitFlipSTEWeightNoiseConfig(flip_p=0.001, start_epoch=1), "bitflipste0.001_ep1"),
        (
            SynaptogenPoolNoiseConfig(
                pool_size=1000000,
                num_cycles=1,
                read_noise="none",
                strength=1.0,
                parametric=False,
                pool_seed=0,
                refresh_per_epoch=False,
                start_epoch=1,
            ),
            "synpool1.0_ep1",
        ),
        (
            AsymmetricBitNoiseConfig(dev=0.05, on_scale=1.34, off_scale=1.0, normalize=True, start_epoch=1),
            "asymbit0.05_r1.34_ep1",
        ),
        # The three models from the final table that were only ever run on LibriSpeech.
        # Settings are the LibriSpeech optima: a=0.1 (not 0.05, which is under-amplitude),
        # Student-t at the same sigma as gauss0.035, dropout at p=0.1.
        (UniformBitNoiseConfig(bit_amplitude=0.1, start_epoch=1), "unifbit0.1_ep1"),
        (StudentTWeightNoiseConfig(dev=0.035, nu=5.9, start_epoch=1), "studentt0.035_nu5.9_ep1"),
        (None, "wdrop0.1"),
        # Full non-combined sweep of the LibriSpeech tuning table, so the collapse can be
        # checked per corpus. Combined settings (dropout x gauss, act-dropout off) stay out.
        (GaussianWeightNoiseConfig(dev=0.02, start_epoch=1), "gauss0.02_ep1"),
        (GaussianWeightNoiseConfig(dev=0.03, start_epoch=1), "gauss0.03_ep1"),
        (GaussianWeightNoiseConfig(dev=0.04, start_epoch=1), "gauss0.04_ep1"),
        (GaussianWeightNoiseConfig(dev=0.07, start_epoch=1), "gauss0.07_ep1"),
        (
            AsymmetricBitNoiseConfig(dev=0.05, on_scale=1.0, off_scale=1.0, normalize=True, start_epoch=1),
            "asymbit0.05_r1.0_ep1",
        ),
        (
            AsymmetricBitNoiseConfig(dev=0.05, on_scale=1.0, off_scale=0.0, normalize=True, start_epoch=1),
            "asymbit0.05_onlyon_ep1",
        ),
        (
            AsymmetricBitNoiseConfig(dev=0.05, on_scale=0.0, off_scale=1.0, normalize=True, start_epoch=1),
            "asymbit0.05_onlyoff_ep1",
        ),
        (RelativeGaussianWeightNoiseConfig(rel_dev=0.05, start_epoch=1), "relgauss0.05_ep1"),
        (RelativeGaussianWeightNoiseConfig(rel_dev=0.3, start_epoch=1), "relgauss0.3_ep1"),
        (UniformBitNoiseConfig(bit_amplitude=0.05, start_epoch=1), "unifbit0.05_ep1"),
        (UniformBitNoiseConfig(bit_amplitude=0.15, start_epoch=1), "unifbit0.15_ep1"),
        (StudentTWeightNoiseConfig(dev=0.02, nu=5.9, start_epoch=1), "studentt0.02_nu5.9_ep1"),
        (StudentTWeightNoiseConfig(dev=0.05, nu=5.9, start_epoch=1), "studentt0.05_nu5.9_ep1"),
        (BitFlipSTEWeightNoiseConfig(flip_p=0.0003, start_epoch=1), "bitflipste0.0003_ep1"),
        (BitFlipSTEWeightNoiseConfig(flip_p=0.003, start_epoch=1), "bitflipste0.003_ep1"),
        (BitFlipSTEWeightNoiseConfig(flip_p=0.01, start_epoch=1), "bitflipste0.01_ep1"),
        (
            SynaptogenPoolNoiseConfig(
                pool_size=1000000,
                num_cycles=1,
                read_noise="none",
                strength=0.7,
                parametric=False,
                pool_seed=0,
                refresh_per_epoch=False,
                start_epoch=1,
            ),
            "synpool0.7_ep1",
        ),
        (
            SynaptogenPoolNoiseConfig(
                pool_size=1000000,
                num_cycles=1,
                read_noise="none",
                strength=1.4,
                parametric=False,
                pool_seed=0,
                refresh_per_epoch=False,
                start_epoch=1,
            ),
            "synpool1.4_ep1",
        ),
        (None, "wdrop0.05"),
        (None, "wdrop0.2"),
        (None, "nonoise"),
    ]

    # Weight dropout per variant; unlisted stays 0.0, so existing hashes are untouched.
    FINETUNE_WEIGHT_DROPOUT = {"wdrop0.05": 0.05, "wdrop0.1": 0.1, "wdrop0.2": 0.2}
    # Format: finetune_epochs, dim, weight_bit, dropout, seed, noise_name
    diverged_list = [
    ]
    for finetune_epochs in [250]:
        for activation_bit in [8]:
            for dim in [512]:
                for weight_bit in [8, 4]:
                    for dropout in [0.1]:
                        for seed in [0]:
                            for noise_cfg, noise_name in noise_configs:
                                training_name = noise_prefix + "/" + network_module_mem_v15 + f"_{finetune_epochs // PARTITION_EPOCH}eps_from{base_epochs // PARTITION_EPOCH}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}"
                                if (finetune_epochs, dim, weight_bit, dropout, seed, noise_name) in diverged_list:
                                    full_results[training_name] = "Diverged"
                                    continue
                                base_checkpoint_name = baseline_prefix + "/" + network_module_mem_v9 + f"_{base_epochs // PARTITION_EPOCH}eps_{weight_bit}_{activation_bit}_seed_{seed}"
                                model_config = _make_v15_model_config(
                                    dim=dim, weight_noise=noise_cfg, weight_bit=weight_bit, activation_bit=activation_bit,
                                    weight_dropout=FINETUNE_WEIGHT_DROPOUT.get(noise_name, 0.0),
                                )
                                train_config_24gbgpu = {
                                    "optimizer": {
                                        "class": "radam",
                                        "epsilon": 1e-12,
                                        "weight_decay": 1e-2,
                                        "decoupled_weight_decay": True,
                                    },
                                    "learning_rates": list(np.linspace(7e-6, 1e-4, finetune_epochs // 2))
                                                      + list(np.linspace(1e-4, 1e-7, finetune_epochs // 2)),
                                    "batch_size": 240 * 16000,
                                    "max_seq_length": {"audio_features": 35 * 16000},
                                    "accum_grad_multiple_step": 1,
                                    "gradient_clip_norm": 1.0,
                                    "seed": seed,
                                    "torch_amp_options": {"dtype": "bfloat16"},
                                    "preload_from_files": {
                                        "model": {
                                            "filename": FINETUNE_MODELS[base_checkpoint_name],
                                            "init_for_train": True,
                                            # always False: key mismatches must crash loudly and be
                                            # fixed in the config, never papered over
                                            "ignore_missing": False,
                                        }
                                    },
                                }
                                train_args = {
                                    "config": train_config_24gbgpu,
                                    "network_module": network_module_mem_v15,
                                    "net_args": {"model_config_dict": asdict(model_config)},
                                    "debug": False,
                                    "post_config": {"num_workers_per_gpu": 4},
                                    "use_speed_perturbation": True,
                                }
                                train_args_training = train_args
                                if isinstance(noise_cfg, SynaptogenPoolNoiseConfig):
                                    # the lazy pool build needs synaptogen_ml importable inside the
                                    # training job; same pin as the LBS synpool runs
                                    train_args_training = {**train_args, "import_memristor": "new_v3"}
                                train_job = training(training_name, train_data_bpe, train_args_training,
                                                     num_epochs=finetune_epochs, **default_returnn)
                                if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                                    train_job.rqmt['cpu'] = 8
                                    train_job.rqmt['time'] = 36
                                    train_job.hold()
                                    train_job.move_to_hpc = True
                                    # submit as 12h c25g dependency-chain segments
                                    train_job.use_new_partition = True

                                prior_config = copy.deepcopy(model_config)
                                prior_config.weight_noise = None
                                prior_args = copy.deepcopy(train_args)
                                prior_args["net_args"] = {"model_config_dict": asdict(prior_config)}

                                if noise_cfg is not None:
                                    # with_noise eval is meaningless for the no-noise control
                                    results = {}
                                    results, best_params_job_noise = eval_model(
                                        training_name=training_name + "_with_noise",
                                        train_job=train_job,
                                        train_args=train_args,
                                        train_data=train_data_bpe,
                                        decoder_config=as_training_rasr_config,
                                        dev_dataset_tuples=short_dev_dataset_tuples,
                                        result_dict=results,
                                        decoder_module="ctc.decoder.rasr_ctc_v1",
                                        prior_scales=rasr_noise_prior_scales,
                                        lm_scales=rasr_noise_lm_scales,
                                        prior_args=prior_args,
                                        import_memristor=True,
                                        get_best_params=True,
                                        run_rasr=True,
                                        run_best_4=False,
                                        run_best=False,
                                    )
                                    generate_report(results=results, exp_name=training_name + "/with_noise")
                                    full_results[training_name + "/with_noise"] = results

                                results = {}
                                results, best_params_job = eval_model(
                                    training_name=training_name + "_without_noise",
                                    train_job=train_job,
                                    train_args=prior_args,
                                    train_data=train_data_bpe,
                                    decoder_config=as_training_rasr_config,
                                    dev_dataset_tuples=short_dev_dataset_tuples,
                                    result_dict=results,
                                    decoder_module="ctc.decoder.rasr_ctc_v1",
                                    prior_scales=rasr_prior_scales,
                                    lm_scales=rasr_lm_scales,
                                    prior_args=prior_args,
                                    import_memristor=True,
                                    get_best_params=True,
                                    run_rasr=True,
                                    run_best_4=False,
                                    run_best=False,
                                )
                                generate_report(results=results, exp_name=training_name + "/without_noise")
                                full_results[training_name + "/without_noise"] = results

                                name_suffix = f"_{finetune_epochs // PARTITION_EPOCH}eps_from{base_epochs // PARTITION_EPOCH}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}"
                                _run_cycle_loop(
                                    train_job=train_job,
                                    prior_train_args=prior_args,
                                    prior_model_config=prior_config,
                                    name_suffix=name_suffix,
                                    final_name_suffix=name_suffix,
                                    prior_scales=[0.5],
                                    lm_scales=[cycle_lm_scale],
                                    batch_size=2500000 if weight_bit == 8 else 3500000,
                                    max_runs=5,
                                    greedy=True,
                                )

    tk.register_report(
        "reports/loquacious/v2/noise_small_from100eps",
        partial(build_qat_report_v2, full_results, max_seeds=1),
        required=full_results,
        update_frequency=600,
    )


def _bpe_loq_noise(corpus: str):
    assert corpus in ["small", "medium"]
    # dataset build keeps the baseline prefix so all data jobs and the base QAT trainings dedup
    baseline_prefix = f"experiments/loquacious/{corpus}/memristor_1025/bpe_ctc_bpe/128"
    noise_prefix = baseline_prefix + "/noise"
    loquacious_key = f"train.{corpus}"
    PARTITION_EPOCH = 5 if corpus == "small" else 25
    train_cpu_rqmt = 8 if corpus == "small" else 12

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

    global_train_args = {
        "debug": False,
        "use_speed_perturbation": True,
        "post_config": {"num_workers_per_gpu": 4},
    }

    network_module_mem_v9 = "ctc.memristor_1025.memristor_v9"
    network_module_mem_v11 = "ctc.memristor_1025.memristor_v11"
    network_module_mem_v15 = "ctc.memristor_1025.memristor_v15"

    train_data_bpe = build_bpe_training_datasets(
        prefix=baseline_prefix,
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
    greedy_decoder_memristor = GreedyDecoderConfig(
        returnn_vocab=label_datastream_bpe.vocab,
        turn_off_quant=False,
    )

    from ...pytorch_networks.ctc.memristor_1025.memristor_v8_cfg import QuantModelTrainConfigV8 as MemristorModelTrainConfigV8
    from ...pytorch_networks.ctc.memristor_1025.memristor_v11_cfg import QuantModelTrainConfigV11 as MemristorModelTrainConfigV11
    from ...pytorch_networks.ctc.memristor_1025.memristor_v15_cfg import (
        QuantModelTrainConfigV15 as MemristorModelTrainConfigV15,
        GaussianWeightNoiseConfig,
        BitFlipSTEWeightNoiseConfig,
        SynaptogenPoolNoiseConfig,
        AsymmetricBitNoiseConfig,
        UniformBitNoiseConfig,
        StudentTWeightNoiseConfig,
        RelativeGaussianWeightNoiseConfig,
    )
    from torch_memristor.memristor_modules import DacAdcHardwareSettings

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
    posenc_dac_settings = DacAdcHardwareSettings(
        input_bits=8,
        output_precision_bits=1,
        output_range_bits=7,
        hardware_input_vmax=0.6,
        hardware_output_current_scaling=8020.0,
    )

    if corpus == "small":
        # noise pushes the prior optimum up (LBS observation), hence the wider noisy grid
        rasr_prior_scales = [0.3, 0.4, 0.5]
        rasr_lm_scales = [0.9, 1.0, 1.1, 1.2, 1.3, 1.4]
        rasr_noise_prior_scales = [0.3, 0.4, 0.5, 0.6, 0.7]
        rasr_noise_lm_scales = [0.9, 1.0, 1.1, 1.2, 1.3]
        cycle_lm_scale = 1.0
        multi_prior_scales = [0.3, 0.4, 0.5, 0.6]
        multi_lm_scales = [0.9, 1.1, 1.3]
    else:
        rasr_prior_scales = [0.2, 0.3, 0.4, 0.5]
        rasr_lm_scales = [0.4, 0.5, 0.6, 0.7, 0.8]
        rasr_noise_prior_scales = [0.2, 0.3, 0.4, 0.5, 0.6]
        rasr_noise_lm_scales = [0.4, 0.5, 0.6, 0.7, 0.8]
        cycle_lm_scale = 0.6
        multi_prior_scales = None
        multi_lm_scales = None

    full_results = {}

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
        weight_bit_prec=0,  # will be filled out in loop
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

    def _make_v15_model_config_kwargs(dim, weight_noise):
        return dict(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub4,
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
            weight_quant_method="per_tensor_symmetric",
            activation_quant_dtype="qint8",
            activation_quant_method="per_tensor_symmetric",
            dot_quant_dtype="qint8",
            dot_quant_method="per_tensor_symmetric",
            Av_quant_dtype="qint8",
            Av_quant_method="per_tensor_symmetric",
            moving_average=None,
            quantize_output=False,
            converter_hardware_settings=train_dac_settings,
            quant_in_linear=True,
            num_cycles=0,
            correction_settings=None,
            weight_noise=weight_noise,
            module_list=["ff", "conv", "mhsa", "ff"],
            module_scales=[0.5, 1.0, 1.0, 0.5],
            aux_ctc_loss_layers=None,
            aux_ctc_loss_scales=None,
            dropout_broadcast_axes=None,
        )

    def _make_v15_model_config(dim, weight_noise, weight_bit, activation_bit, weight_dropout=0.0):
        model_config = MemristorModelTrainConfigV15(
            **_make_v15_model_config_kwargs(dim, weight_noise),
            weight_bit_prec=weight_bit,
            activation_bit_prec=activation_bit,
            weight_dropout=weight_dropout,
            weight_pruning=None,
            pos_enc_converter_hardware_settings=train_dac_settings,
        )
        # v9 ignored module_list (v8-cfg default order was used); v15 honors it, so pin the
        # effective checkpoint order here or preload crashes on mismatched keys
        model_config.module_list = ["ff", "mhsa", "conv", "ff"]
        return model_config

    def _run_cycle_loop(train_job, prior_train_args, prior_model_config, name_suffix, final_name_suffix,
                        prior_scales, lm_scales, batch_size, max_runs,
                        run_rasr_multi=False, search_gpu=24, greedy=False):
        # prior_model_config must already have weight_noise=None; recognition adds HW settings per cycle
        res, res_greedy = {}, {}
        for num_cycles in range(1, max_runs + 1):
            model_config_recog = copy.deepcopy(prior_model_config)
            model_config_recog.converter_hardware_settings = recog_dac_settings
            model_config_recog.num_cycles = num_cycles
            model_config_recog.pos_enc_converter_hardware_settings = posenc_dac_settings

            prior_args = copy.deepcopy(prior_train_args)
            train_args_recog = copy.deepcopy(prior_train_args)
            train_args_recog["net_args"] = {"model_config_dict": asdict(model_config_recog)}
            train_args_recog["network_module"] = network_module_mem_v15

            recog_name = noise_prefix + "/" + network_module_mem_v15 + name_suffix + f"/cycle_{num_cycles // 11}"
            res = eval_model(
                training_name=recog_name + f"_{num_cycles}",
                train_job=train_job,
                train_args=train_args_recog,
                train_data=train_data_bpe,
                decoder_config=rasr_config_memristor,
                dev_dataset_tuples=short_dev_dataset_tuples,
                result_dict=res,
                decoder_module="ctc.decoder.rasr_ctc_v1_batched_multi_fast" if run_rasr_multi
                else "ctc.decoder.rasr_ctc_v1_batched_fast",
                prior_scales=prior_scales,
                lm_scales=lm_scales,
                use_gpu=True,
                import_memristor="new_v3",
                extra_forward_config={"batch_size": batch_size},
                run_best_4=False,
                run_best=False,
                prior_args=prior_args,
                run_search_on_hpc=False,
                run_rasr=True,
                split_mem_init=True,
                search_gpu=search_gpu,
                run_rasr_multi=run_rasr_multi,
                num_search_workers=8,
            )

            if greedy:
                res_greedy = eval_model(
                    training_name=recog_name + f"_{num_cycles}",
                    train_job=train_job,
                    train_args=train_args_recog,
                    train_data=train_data_bpe,
                    decoder_config=greedy_decoder_memristor,
                    dev_dataset_tuples=short_dev_dataset_tuples,
                    result_dict=res_greedy,
                    decoder_module="ctc.decoder.greedy_bpe_ctc_quant_v1_fast",
                    prior_scales=[0.0],
                    lm_scales=[0.0],
                    use_gpu=True,
                    import_memristor="new_v3",
                    extra_forward_config={"batch_size": batch_size},
                    run_best_4=False,
                    run_best=False,
                    prior_args=None,
                    with_prior=False,
                    run_search_on_hpc=False,
                    run_rasr=False,
                    split_mem_init=True,
                    search_gpu=search_gpu,
                )

            if num_cycles == max_runs:
                final_name = noise_prefix + "/" + network_module_mem_v15 + final_name_suffix + "_cycle"
                if run_rasr_multi:
                    generate_report(results=res, exp_name=final_name, report_template=multi_scale_cycle_report_format)
                else:
                    generate_report(results=res, exp_name=final_name)
                full_results[final_name] = copy.deepcopy(res)
                if greedy:
                    generate_report(results=res_greedy, exp_name=final_name + "_greedy")
                    full_results[final_name + "_greedy"] = copy.deepcopy(res_greedy)
        return res

    # --- Base QAT trainings (re-declaration of the baseline jobs, same hashes -> no new work) ---
    FINETUNE_MODELS = {}
    for epochs in [1000]:
        for activation_bit in [8]:
            for weight_bit in [8, 4]:
                for seed in [0]:
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

                    training_name = baseline_prefix + "/" + network_module_mem_v9 + f"_{epochs//PARTITION_EPOCH}eps_{weight_bit}_{activation_bit}_seed_{seed}"

                    train_job = training(training_name, train_data_bpe, train_args, num_epochs=epochs,
                                         **default_returnn)
                    if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                        train_job.rqmt['cpu'] = train_cpu_rqmt
                        train_job.hold()
                        train_job.move_to_hpc = True
                    FINETUNE_MODELS[training_name] = train_job.out_checkpoints[epochs]

                    # baseline tuning numbers for the noise report; the call replicates the
                    # baseline files 1:1 so all jobs dedup to the finished ones
                    results, best_params_job = eval_model(
                        training_name=training_name,
                        train_job=train_job,
                        train_args=train_args,
                        train_data=train_data_bpe,
                        decoder_config=as_training_rasr_config,
                        dev_dataset_tuples=short_dev_dataset_tuples,
                        result_dict={},
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
                    # keep only the short-dev tuning numbers in the noise report
                    results.pop(training_name + f"/{epochs}" + "_dev_all", None)
                    results.pop(training_name + f"/{epochs}" + "_test_all", None)
                    for set_name in ['yodas', 'librispeech', 'voxpopuli', 'commonvoice']:
                        results.pop(training_name + f"/{epochs}" + f"/dev.{set_name}", None)
                        results.pop(training_name + f"/{epochs}" + f"/test.{set_name}", None)
                    full_results[training_name] = results

                    # old (non-fast, v11) baseline cycle numbers, re-declared 1:1 with the
                    # baseline files so they dedup to the finished jobs. Small w8 has none:
                    # baseline.py cycles only the 500-subep models, width only ran w4.
                    if corpus == "medium":
                        # from baseline_medium.py: cycles 1-3 at prior 0.5 / tuned-best lm
                        res_conv = {}
                        for num_cycles in range(1, 4):
                            model_config_recog = MemristorModelTrainConfigV11(
                                **model_config.__dict__,
                                pos_enc_converter_hardware_settings=None,
                            )
                            model_config_recog.converter_hardware_settings = recog_dac_settings
                            model_config_recog.num_cycles = num_cycles
                            model_config_recog.pos_enc_converter_hardware_settings = posenc_dac_settings

                            prior_args_old = copy.deepcopy(train_args)
                            train_args_recog = copy.deepcopy(train_args)
                            train_args_recog["net_args"] = {"model_config_dict": asdict(model_config_recog)}
                            train_args_recog["network_module"] = network_module_mem_v11

                            recog_name = baseline_prefix + "/" + network_module_mem_v11 + f"_{weight_bit}_{activation_bit}_seed_{seed}/cycle_{num_cycles // 11}"
                            res_conv = eval_model(
                                training_name=recog_name + f"_{num_cycles}",
                                train_job=train_job,
                                train_args=train_args_recog,
                                train_data=train_data_bpe,
                                decoder_config=rasr_config_memristor,
                                dev_dataset_tuples=short_dev_dataset_tuples,
                                result_dict=res_conv,
                                decoder_module="ctc.decoder.rasr_ctc_v1_batched",
                                prior_scales=[0.5],
                                lm_scales=[(best_params_job.out_optimal_parameters[0], "best")],
                                use_gpu=True,
                                import_memristor=True,
                                extra_forward_config={
                                    "batch_size": 3500000 if not weight_bit in [8] else 2500000,
                                },
                                run_best_4=False,
                                run_best=False,
                                prior_args=prior_args_old,
                                run_search_on_hpc=False,
                                run_rasr=True,
                                split_mem_init=True,
                            )
                        full_results[baseline_prefix + "/" + network_module_mem_v11 + f"_{weight_bit}_{activation_bit}_seed_{seed}_cycle"] = copy.deepcopy(res_conv)

                    if corpus == "small" and weight_bit == 4:
                        # from memristor_small.py (width): cycles 1-3 at prior 0.5 / lm 1.0 + greedy
                        width_prefix = baseline_prefix + "/width"
                        width_suffix = f"_{epochs // PARTITION_EPOCH}eps_512dim_w{weight_bit}_a{activation_bit}_seed_{seed}"
                        res_conv, res_greedy_old = {}, {}
                        for num_cycles in range(1, 4):
                            model_config_recog = MemristorModelTrainConfigV11(
                                **model_config.__dict__,
                                pos_enc_converter_hardware_settings=None,
                            )
                            model_config_recog.converter_hardware_settings = recog_dac_settings
                            model_config_recog.num_cycles = num_cycles
                            model_config_recog.pos_enc_converter_hardware_settings = posenc_dac_settings

                            prior_args_old = copy.deepcopy(train_args)
                            train_args_recog = copy.deepcopy(train_args)
                            train_args_recog["net_args"] = {"model_config_dict": asdict(model_config_recog)}
                            train_args_recog["network_module"] = network_module_mem_v11

                            recog_name = width_prefix + "/" + network_module_mem_v11 + width_suffix + f"/cycle_{num_cycles // 11}"
                            res_conv = eval_model(
                                training_name=recog_name + f"_{num_cycles}",
                                train_job=train_job,
                                train_args=train_args_recog,
                                train_data=train_data_bpe,
                                decoder_config=rasr_config_memristor,
                                dev_dataset_tuples=short_dev_dataset_tuples,
                                result_dict=res_conv,
                                decoder_module="ctc.decoder.rasr_ctc_v1_batched",
                                prior_scales=[0.5],
                                lm_scales=[1.0],
                                use_gpu=True,
                                import_memristor=True,
                                extra_forward_config={"batch_size": 2500000},
                                run_best_4=False,
                                run_best=False,
                                prior_args=prior_args_old,
                                run_search_on_hpc=False,
                                run_rasr=True,
                                split_mem_init=True,
                                search_gpu=24,
                            )
                            res_greedy_old = eval_model(
                                training_name=recog_name + f"_{num_cycles}",
                                train_job=train_job,
                                train_args=train_args_recog,
                                train_data=train_data_bpe,
                                decoder_config=greedy_decoder_memristor,
                                dev_dataset_tuples=short_dev_dataset_tuples,
                                result_dict=res_greedy_old,
                                decoder_module="ctc.decoder.greedy_bpe_ctc_quant_v1",
                                prior_scales=[0.0],
                                lm_scales=[0.0],
                                use_gpu=True,
                                import_memristor=True,
                                extra_forward_config={"batch_size": 2500000},
                                run_best_4=False,
                                run_best=False,
                                prior_args=None,
                                with_prior=False,
                                run_search_on_hpc=False,
                                run_rasr=False,
                                split_mem_init=True,
                                search_gpu=24,
                            )
                        full_results[width_prefix + "/" + network_module_mem_v11 + width_suffix + "_cycle"] = copy.deepcopy(res_conv)
                        full_results[width_prefix + "/" + network_module_mem_v11 + width_suffix + "_cycle_greedy"] = copy.deepcopy(res_greedy_old)

                    base_prior_config = _make_v15_model_config(
                        dim=512, weight_noise=None, weight_bit=weight_bit, activation_bit=activation_bit,
                    )
                    base_prior_args = copy.deepcopy(train_args)
                    base_prior_args["network_module"] = network_module_mem_v15
                    base_prior_args["net_args"] = {"model_config_dict": asdict(base_prior_config)}

                    # baseline memristor numbers at the fixed point, same v15 fast path as the
                    # finetunes (the 1000-subep baselines have no cycle evals in baseline.py)
                    _run_cycle_loop(
                        train_job=train_job,
                        prior_train_args=base_prior_args,
                        prior_model_config=base_prior_config,
                        name_suffix=f"_base{epochs // PARTITION_EPOCH}eps_512dim_w{weight_bit}_a{activation_bit}_seed_{seed}",
                        final_name_suffix=f"_base{epochs // PARTITION_EPOCH}eps_512dim_w{weight_bit}_a{activation_bit}_seed_{seed}",
                        prior_scales=[0.5],
                        lm_scales=[cycle_lm_scale],
                        batch_size=2500000 if weight_bit == 8 else 3500000,
                        max_runs=5,
                        greedy=True,
                    )

                    if corpus == "small":
                        # scale-sweep calibration of the fixed cycle-eval point on the un-finetuned
                        # QAT baseline, recognized through the v15 path (weight_noise=None)
                        _run_cycle_loop(
                            train_job=train_job,
                            prior_train_args=base_prior_args,
                            prior_model_config=base_prior_config,
                            name_suffix=f"_base{epochs // PARTITION_EPOCH}eps_512dim_w{weight_bit}_a{activation_bit}_multi_sweep_seed_{seed}",
                            final_name_suffix=f"_base{epochs // PARTITION_EPOCH}eps_512dim_w{weight_bit}_a{activation_bit}_multi_sweep_seed_{seed}",
                            prior_scales=multi_prior_scales,
                            lm_scales=multi_lm_scales,
                            batch_size=2500000 if weight_bit == 8 else 3500000,
                            max_runs=3,
                            run_rasr_multi=True,
                            search_gpu=48,
                        )

    # --- Noise finetunes ---
    if corpus == "small":
        noise_configs = [
            (GaussianWeightNoiseConfig(dev=0.01, start_epoch=1), "gauss0.01_ep1"),
            (GaussianWeightNoiseConfig(dev=0.035, start_epoch=1), "gauss0.035_ep1"),
            (GaussianWeightNoiseConfig(dev=0.05, start_epoch=1), "gauss0.05_ep1"),
            (GaussianWeightNoiseConfig(dev=0.06, start_epoch=1), "gauss0.06_ep1"),
            (GaussianWeightNoiseConfig(dev=0.1, start_epoch=1), "gauss0.1_ep1"),
            (BitFlipSTEWeightNoiseConfig(flip_p=0.001, start_epoch=1), "bitflipste0.001_ep1"),
            (
                SynaptogenPoolNoiseConfig(
                    pool_size=1000000,
                    num_cycles=1,
                    read_noise="none",
                    strength=1.0,
                    parametric=False,
                    pool_seed=0,
                    refresh_per_epoch=False,
                    start_epoch=1,
                ),
                "synpool1.0_ep1",
            ),
            (
                AsymmetricBitNoiseConfig(dev=0.05, on_scale=1.34, off_scale=1.0, normalize=True, start_epoch=1),
                "asymbit0.05_r1.34_ep1",
            ),
            # Same three additions as the 100eps graph, at the LibriSpeech optima.
            (UniformBitNoiseConfig(bit_amplitude=0.1, start_epoch=1), "unifbit0.1_ep1"),
            (StudentTWeightNoiseConfig(dev=0.035, nu=5.9, start_epoch=1), "studentt0.035_nu5.9_ep1"),
            (None, "wdrop0.1"),
            # Full non-combined sweep of the LibriSpeech tuning table, so the collapse can be
            # checked per corpus. Combined settings (dropout x gauss, act-dropout off) stay out.
            (GaussianWeightNoiseConfig(dev=0.02, start_epoch=1), "gauss0.02_ep1"),
            (GaussianWeightNoiseConfig(dev=0.03, start_epoch=1), "gauss0.03_ep1"),
            (GaussianWeightNoiseConfig(dev=0.04, start_epoch=1), "gauss0.04_ep1"),
            (GaussianWeightNoiseConfig(dev=0.07, start_epoch=1), "gauss0.07_ep1"),
            (
                AsymmetricBitNoiseConfig(dev=0.05, on_scale=1.0, off_scale=1.0, normalize=True, start_epoch=1),
                "asymbit0.05_r1.0_ep1",
            ),
            (
                AsymmetricBitNoiseConfig(dev=0.05, on_scale=1.0, off_scale=0.0, normalize=True, start_epoch=1),
                "asymbit0.05_onlyon_ep1",
            ),
            (
                AsymmetricBitNoiseConfig(dev=0.05, on_scale=0.0, off_scale=1.0, normalize=True, start_epoch=1),
                "asymbit0.05_onlyoff_ep1",
            ),
            (RelativeGaussianWeightNoiseConfig(rel_dev=0.05, start_epoch=1), "relgauss0.05_ep1"),
            (RelativeGaussianWeightNoiseConfig(rel_dev=0.3, start_epoch=1), "relgauss0.3_ep1"),
            (UniformBitNoiseConfig(bit_amplitude=0.05, start_epoch=1), "unifbit0.05_ep1"),
            (UniformBitNoiseConfig(bit_amplitude=0.15, start_epoch=1), "unifbit0.15_ep1"),
            (StudentTWeightNoiseConfig(dev=0.02, nu=5.9, start_epoch=1), "studentt0.02_nu5.9_ep1"),
            (StudentTWeightNoiseConfig(dev=0.05, nu=5.9, start_epoch=1), "studentt0.05_nu5.9_ep1"),
            (BitFlipSTEWeightNoiseConfig(flip_p=0.0003, start_epoch=1), "bitflipste0.0003_ep1"),
            (BitFlipSTEWeightNoiseConfig(flip_p=0.003, start_epoch=1), "bitflipste0.003_ep1"),
            (BitFlipSTEWeightNoiseConfig(flip_p=0.01, start_epoch=1), "bitflipste0.01_ep1"),
            (
                SynaptogenPoolNoiseConfig(
                    pool_size=1000000,
                    num_cycles=1,
                    read_noise="none",
                    strength=0.7,
                    parametric=False,
                    pool_seed=0,
                    refresh_per_epoch=False,
                    start_epoch=1,
                ),
                "synpool0.7_ep1",
            ),
            (
                SynaptogenPoolNoiseConfig(
                    pool_size=1000000,
                    num_cycles=1,
                    read_noise="none",
                    strength=1.4,
                    parametric=False,
                    pool_seed=0,
                    refresh_per_epoch=False,
                    start_epoch=1,
                ),
                "synpool1.4_ep1",
            ),
            (None, "wdrop0.05"),
            (None, "wdrop0.2"),
            (None, "nonoise"),
        ]
    else:
        # 2500h ran only the two Gaussians. Add the rest of the final-table models so the
        # corpus-size comparison is like for like.
        noise_configs = [
            (GaussianWeightNoiseConfig(dev=0.035, start_epoch=1), "gauss0.035_ep1"),
            (GaussianWeightNoiseConfig(dev=0.05, start_epoch=1), "gauss0.05_ep1"),
            (GaussianWeightNoiseConfig(dev=0.06, start_epoch=1), "gauss0.06_ep1"),
            (BitFlipSTEWeightNoiseConfig(flip_p=0.001, start_epoch=1), "bitflipste0.001_ep1"),
            (
                SynaptogenPoolNoiseConfig(
                    pool_size=1000000,
                    num_cycles=1,
                    read_noise="none",
                    strength=1.0,
                    parametric=False,
                    pool_seed=0,
                    refresh_per_epoch=False,
                    start_epoch=1,
                ),
                "synpool1.0_ep1",
            ),
            (UniformBitNoiseConfig(bit_amplitude=0.1, start_epoch=1), "unifbit0.1_ep1"),
            (StudentTWeightNoiseConfig(dev=0.035, nu=5.9, start_epoch=1), "studentt0.035_nu5.9_ep1"),
            (None, "wdrop0.1"),
            # Full non-combined sweep of the LibriSpeech tuning table, so the collapse can be
            # checked per corpus. Combined settings (dropout x gauss, act-dropout off) stay out.
            (GaussianWeightNoiseConfig(dev=0.01, start_epoch=1), "gauss0.01_ep1"),
            (GaussianWeightNoiseConfig(dev=0.02, start_epoch=1), "gauss0.02_ep1"),
            (GaussianWeightNoiseConfig(dev=0.03, start_epoch=1), "gauss0.03_ep1"),
            (GaussianWeightNoiseConfig(dev=0.04, start_epoch=1), "gauss0.04_ep1"),
            (GaussianWeightNoiseConfig(dev=0.07, start_epoch=1), "gauss0.07_ep1"),
            (GaussianWeightNoiseConfig(dev=0.1, start_epoch=1), "gauss0.1_ep1"),
            (
                AsymmetricBitNoiseConfig(dev=0.05, on_scale=1.0, off_scale=1.0, normalize=True, start_epoch=1),
                "asymbit0.05_r1.0_ep1",
            ),
            (
                AsymmetricBitNoiseConfig(dev=0.05, on_scale=1.34, off_scale=1.0, normalize=True, start_epoch=1),
                "asymbit0.05_r1.34_ep1",
            ),
            (
                AsymmetricBitNoiseConfig(dev=0.05, on_scale=1.0, off_scale=0.0, normalize=True, start_epoch=1),
                "asymbit0.05_onlyon_ep1",
            ),
            (
                AsymmetricBitNoiseConfig(dev=0.05, on_scale=0.0, off_scale=1.0, normalize=True, start_epoch=1),
                "asymbit0.05_onlyoff_ep1",
            ),
            (RelativeGaussianWeightNoiseConfig(rel_dev=0.05, start_epoch=1), "relgauss0.05_ep1"),
            (RelativeGaussianWeightNoiseConfig(rel_dev=0.3, start_epoch=1), "relgauss0.3_ep1"),
            (UniformBitNoiseConfig(bit_amplitude=0.05, start_epoch=1), "unifbit0.05_ep1"),
            (UniformBitNoiseConfig(bit_amplitude=0.15, start_epoch=1), "unifbit0.15_ep1"),
            (StudentTWeightNoiseConfig(dev=0.02, nu=5.9, start_epoch=1), "studentt0.02_nu5.9_ep1"),
            (StudentTWeightNoiseConfig(dev=0.05, nu=5.9, start_epoch=1), "studentt0.05_nu5.9_ep1"),
            (BitFlipSTEWeightNoiseConfig(flip_p=0.0003, start_epoch=1), "bitflipste0.0003_ep1"),
            (BitFlipSTEWeightNoiseConfig(flip_p=0.003, start_epoch=1), "bitflipste0.003_ep1"),
            (BitFlipSTEWeightNoiseConfig(flip_p=0.01, start_epoch=1), "bitflipste0.01_ep1"),
            (
                SynaptogenPoolNoiseConfig(
                    pool_size=1000000,
                    num_cycles=1,
                    read_noise="none",
                    strength=0.7,
                    parametric=False,
                    pool_seed=0,
                    refresh_per_epoch=False,
                    start_epoch=1,
                ),
                "synpool0.7_ep1",
            ),
            (
                SynaptogenPoolNoiseConfig(
                    pool_size=1000000,
                    num_cycles=1,
                    read_noise="none",
                    strength=1.4,
                    parametric=False,
                    pool_seed=0,
                    refresh_per_epoch=False,
                    start_epoch=1,
                ),
                "synpool1.4_ep1",
            ),
            (None, "wdrop0.05"),
            (None, "wdrop0.2"),
            (None, "nonoise"),
        ]

    # Weight dropout per variant; unlisted stays 0.0, so existing hashes are untouched.
    FINETUNE_WEIGHT_DROPOUT = {"wdrop0.05": 0.05, "wdrop0.1": 0.1, "wdrop0.2": 0.2}
    # Format: finetune_epochs, dim, weight_bit, dropout, seed, noise_name
    diverged_list = [
    ]
    for finetune_epochs in [250]:
        for activation_bit in [8]:
            for dim in [512]:
                for weight_bit in [8, 4]:
                    for dropout in [0.1]:
                        for seed in [0]:
                            for noise_cfg, noise_name in noise_configs:
                                training_name = noise_prefix + "/" + network_module_mem_v15 + f"_{finetune_epochs // PARTITION_EPOCH}eps_from{1000 // PARTITION_EPOCH}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}"
                                if (finetune_epochs, dim, weight_bit, dropout, seed, noise_name) in diverged_list:
                                    full_results[training_name] = "Diverged"
                                    continue
                                base_checkpoint_name = baseline_prefix + "/" + network_module_mem_v9 + f"_{1000 // PARTITION_EPOCH}eps_{weight_bit}_{activation_bit}_seed_{seed}"
                                model_config = _make_v15_model_config(
                                    dim=dim, weight_noise=noise_cfg, weight_bit=weight_bit, activation_bit=activation_bit,
                                    weight_dropout=FINETUNE_WEIGHT_DROPOUT.get(noise_name, 0.0),
                                )
                                train_config_24gbgpu = {
                                    "optimizer": {
                                        "class": "radam",
                                        "epsilon": 1e-12,
                                        "weight_decay": 1e-2,
                                        "decoupled_weight_decay": True,
                                    },
                                    "learning_rates": list(np.linspace(7e-6, 1e-4, finetune_epochs // 2))
                                                      + list(np.linspace(1e-4, 1e-7, finetune_epochs // 2)),
                                    "batch_size": 240 * 16000,
                                    "max_seq_length": {"audio_features": 35 * 16000},
                                    "accum_grad_multiple_step": 1,
                                    "gradient_clip_norm": 1.0,
                                    "seed": seed,
                                    "torch_amp_options": {"dtype": "bfloat16"},
                                    "preload_from_files": {
                                        "model": {
                                            "filename": FINETUNE_MODELS[base_checkpoint_name],
                                            "init_for_train": True,
                                            # always False: key mismatches must crash loudly and be
                                            # fixed in the config, never papered over
                                            "ignore_missing": False,
                                        }
                                    },
                                }
                                train_args = {
                                    "config": train_config_24gbgpu,
                                    "network_module": network_module_mem_v15,
                                    "net_args": {"model_config_dict": asdict(model_config)},
                                    "debug": False,
                                    "post_config": {"num_workers_per_gpu": 4},
                                    "use_speed_perturbation": True,
                                }
                                train_args_training = train_args
                                if isinstance(noise_cfg, SynaptogenPoolNoiseConfig):
                                    # the lazy pool build needs synaptogen_ml importable inside the
                                    # training job; same pin as the LBS synpool runs
                                    train_args_training = {**train_args, "import_memristor": "new_v3"}
                                train_job = training(training_name, train_data_bpe, train_args_training,
                                                     num_epochs=finetune_epochs, **default_returnn)
                                if not os.path.exists(f"{train_job._sis_path()}/finished.run.1"):  # sync back was successful
                                    train_job.rqmt['cpu'] = train_cpu_rqmt
                                    train_job.rqmt['time'] = 36
                                    train_job.hold()
                                    train_job.move_to_hpc = True
                                    # submit as 12h c25g dependency-chain segments
                                    train_job.use_new_partition = True

                                prior_config = copy.deepcopy(model_config)
                                prior_config.weight_noise = None
                                prior_args = copy.deepcopy(train_args)
                                prior_args["net_args"] = {"model_config_dict": asdict(prior_config)}

                                if noise_cfg is not None:
                                    # with_noise eval is meaningless for the no-noise control
                                    results = {}
                                    results, best_params_job_noise = eval_model(
                                        training_name=training_name + "_with_noise",
                                        train_job=train_job,
                                        train_args=train_args,
                                        train_data=train_data_bpe,
                                        decoder_config=as_training_rasr_config,
                                        dev_dataset_tuples=short_dev_dataset_tuples,
                                        result_dict=results,
                                        decoder_module="ctc.decoder.rasr_ctc_v1",
                                        prior_scales=rasr_noise_prior_scales,
                                        lm_scales=rasr_noise_lm_scales,
                                        prior_args=prior_args,
                                        import_memristor=True,
                                        get_best_params=True,
                                        run_rasr=True,
                                        run_best_4=False,
                                        run_best=False,
                                    )
                                    generate_report(results=results, exp_name=training_name + "/with_noise")
                                    full_results[training_name + "/with_noise"] = results

                                results = {}
                                results, best_params_job = eval_model(
                                    training_name=training_name + "_without_noise",
                                    train_job=train_job,
                                    train_args=prior_args,
                                    train_data=train_data_bpe,
                                    decoder_config=as_training_rasr_config,
                                    dev_dataset_tuples=short_dev_dataset_tuples,
                                    result_dict=results,
                                    decoder_module="ctc.decoder.rasr_ctc_v1",
                                    prior_scales=rasr_prior_scales,
                                    lm_scales=rasr_lm_scales,
                                    prior_args=prior_args,
                                    import_memristor=True,
                                    get_best_params=True,
                                    run_rasr=True,
                                    run_best_4=False,
                                    run_best=False,
                                )
                                generate_report(results=results, exp_name=training_name + "/without_noise")
                                full_results[training_name + "/without_noise"] = results

                                name_suffix = f"_{finetune_epochs // PARTITION_EPOCH}eps_from{1000 // PARTITION_EPOCH}eps_{dim}dim_w{weight_bit}_a{activation_bit}_noise_{noise_name}_drop{dropout}_seed_{seed}"
                                _run_cycle_loop(
                                    train_job=train_job,
                                    prior_train_args=prior_args,
                                    prior_model_config=prior_config,
                                    name_suffix=name_suffix,
                                    final_name_suffix=name_suffix,
                                    prior_scales=[0.5],
                                    lm_scales=[cycle_lm_scale],
                                    batch_size=2500000 if weight_bit == 8 else 3500000,
                                    max_runs=5,
                                    greedy=True,
                                )

                                if corpus == "small" and noise_name == "gauss0.05_ep1":
                                    # scale-sweep calibration for the best noise finetune
                                    _run_cycle_loop(
                                        train_job=train_job,
                                        prior_train_args=prior_args,
                                        prior_model_config=prior_config,
                                        name_suffix=name_suffix + "_multi_sweep",
                                        final_name_suffix=name_suffix + "_multi_sweep",
                                        prior_scales=multi_prior_scales,
                                        lm_scales=multi_lm_scales,
                                        batch_size=2500000 if weight_bit == 8 else 3500000,
                                        max_runs=3,
                                        run_rasr_multi=True,
                                        search_gpu=48,
                                    )

    tk.register_report(
        f"reports/loquacious/v2/noise_{corpus}",
        partial(build_qat_report_v2, full_results, max_seeds=1),
        required=full_results,
        update_frequency=600,
    )
