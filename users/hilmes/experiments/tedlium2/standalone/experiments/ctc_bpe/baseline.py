from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List, Optional, Union

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon, get_bpe_bliss_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm, get_arpa_lm_config
from ...pipeline import training, prepare_asr_model, search, ASRModel
from ...report import generate_report
from ...pytorch_networks.ctc.decoder.rasr_ctc_v1 import DecoderConfig as RasrDecoderConfig
from i6_core.rasr.config import WriteRasrConfigJob
from .report import build_base_report
from functools import partial

def bpe_ted_1023_base():
    prefix_name = "experiments/tedlium2/ctc_rnnt_standalone_2024/bpe_ctc_bpe/1024"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data_bpe1024 = build_bpe_training_datasets(
        prefix=prefix_name,
        bpe_size=1024,  # TODO tune
        settings=train_settings,
        use_postfix=False,
    )
    label_datastream_bpe1024 = cast(LabelDatastream, train_data_bpe1024.datastreams["labels"])
    vocab_size_without_blank = label_datastream_bpe1024.vocab_size

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

    def tune_and_evaluate_helper(
        training_name: str,
        asr_model: ASRModel,
        base_decoder_config: Union[DecoderConfig, RasrDecoderConfig],
        lm_scales: List[float],
        prior_scales: List[float],
        eval_test: bool = False,
        extra_forward_config: Optional[dict] = None,
        import_memristor: bool = False,
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
        tune_values = []
        results = {}
        for lm_weight in lm_scales:
            for prior_scale in prior_scales:
                decoder_config = copy.deepcopy(base_decoder_config)
                if "rasr" in training_name:
                    decoder_config.rasr_config_file.lib_rasr.lm.scale = lm_weight
                    recog_rasr_config_path = WriteRasrConfigJob(
                        decoder_config.rasr_config_file, decoder_config.rasr_post_config
                    ).out_config
                    decoder_config.rasr_config_file = recog_rasr_config_path
                    decoder_config.rasr_post_config = None
                else:
                    decoder_config.lm_weight = lm_weight
                decoder_config.prior_scale = prior_scale
                search_name = training_name + "/search_lm%.1f_prior%.1f" % (lm_weight, prior_scale)
                search_jobs, wers = search(
                    search_name,
                    forward_config=extra_forward_config or {},
                    asr_model=asr_model,
                    decoder_module="ctc.decoder.flashlight_ctc_v1" if not "rasr" in training_name else "ctc.decoder.rasr_ctc_v1",
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples=dev_dataset_tuples,
                    run_rasr="rasr" in training_name,
                    import_memristor=import_memristor,
                    **default_returnn,
                )
                tune_parameters.append((lm_weight, prior_scale))
                tune_values.append((wers[search_name + "/dev"]))
                results.update(wers)
        if eval_test:
            assert not "rasr" in training_name
            for key, tune_values in [("test", tune_values)]:
                pick_optimal_params_job = GetOptimalParametersAsVariableJob(
                    parameters=tune_parameters, values=tune_values, mode="minimize"
                )
                pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
                decoder_config = copy.deepcopy(base_decoder_config)
                decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
                decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
                search_jobs, wers = search(
                    training_name,
                    forward_config=extra_forward_config or {},
                    asr_model=asr_model,
                    decoder_module="ctc.decoder.flashlight_ctc_v1",
                    decoder_args={"config": asdict(decoder_config)},
                    test_dataset_tuples={key: test_dataset_tuples[key]},
                    import_memristor=import_memristor,
                    **default_returnn,
                )
                results.update(wers)
        return results

    default_decoder_config_bpe1024 = DecoderConfig(
        lexicon=get_text_lexicon(prefix=prefix_name, bpe_size=1024),
        returnn_vocab=label_datastream_bpe1024.vocab,
        beam_size=1024,  # Untuned
        beam_size_token=16,  # makes it much faster (0.3 search RTF -> 0.04 search RTF), but looses 0.1% WER over 128
        arpa_lm=arpa_4gram_lm,
        beam_threshold=14,  # Untuned
    )

    from ...rasr_recog_config import get_tree_timesync_recog_config, get_no_op_label_scorer_config

    rasr_prior_scales = [0.5, 0.7, 0.9, 1.0, 1.1]
    rasr_lm_scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]


    from ...pytorch_networks.ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
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
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=8,  # Jingjing style
        num_repeat_feat=5,
    )
    frontend_config_sub6 = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(3, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=384,
        activation=None,
    )

    model_config = ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub6,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
        specauc_start_epoch=11,  # BPE does not converge otherwise
    )

    train_config_24gbgpu_amp = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
        + list(np.linspace(5e-4, 5e-5, 110))
        + list(np.linspace(5e-5, 1e-7, 30)),
        #############
        "batch_size": 360 * 16000,  # GPU MEM still very moderate, but larger batch did not help
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "torch_amp_options": {"dtype": "bfloat16"},
    }

    network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6"
    train_args = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
    }
    results = {}
    training_name = prefix_name + "/" + network_module + ".384dim_sub6_24gbgpu_50eps_amp"
    train_job = training(training_name, train_data_bpe1024, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe1024, get_specific_checkpoint=250
    )
    res = tune_and_evaluate_helper(
        training_name,
        asr_model,
        default_decoder_config_bpe1024,
        lm_scales=[1.6, 1.8, 2.0],
        prior_scales=[0.2, 0.3, 0.4],
    )
    results.update(res)
    generate_report(results=results, exp_name=training_name)
    del results
    train_config_24gbgpu = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
        + list(np.linspace(5e-4, 5e-5, 110))
        + list(np.linspace(5e-5, 1e-7, 30)),
        #############
        "batch_size": 360 * 16000,  # GPU MEM still very moderate, but larger batch did not help
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
    }

    network_module = "ctc.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6"
    train_args = {
        "config": train_config_24gbgpu,
        "network_module": network_module,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
    }
    results = {}
    training_name = prefix_name + "/" + network_module + ".384dim_sub6_24gbgpu_50eps"
    train_job = training(training_name, train_data_bpe1024, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 24
    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe1024, get_specific_checkpoint=250
    )
    res = tune_and_evaluate_helper(
        training_name,
        asr_model,
        default_decoder_config_bpe1024,
        lm_scales=[1.6, 1.8, 2.0],
        prior_scales=[0.2, 0.3, 0.4],
    )
    results.update(res)
    generate_report(results=results, exp_name=training_name)

    full_res = {}
    lm_scales = [1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    priors = [0.3, 0.5, 0.7, 0.9, 1.0]

    from ...pytorch_networks.ctc.conformer_0106.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1_cfg import (
        ModelConfig as RelPosModelConfig,
        ConformerPosEmbConfig,
    )
    network_module_mem_v9 = "ctc.qat_0711.memristor_v9"
    from ...pytorch_networks.ctc.qat_0711.memristor_v8_cfg import QuantModelTrainConfigV8 as MemristorModelTrainConfigV8

    for epochs in [1000]: # [500, 1500]
        for batch_size in [360]: # [180]
            for dim in [384, 512]: # [768]
                for bpe in [128, 256]: # [64, 256, 1024, 10000]
                    for speed_perturbation in [True]: # [False]
                        prefix_name_bpe = (
                            f"experiments/tedlium2/ctc_rnnt_standalone_2024/bpe_ctc_bpe/pos_enc_{dim}/{bpe}"
                        )
                        train_data_bpe = build_bpe_training_datasets(
                            prefix=prefix_name_bpe,
                            bpe_size=bpe,
                            settings=train_settings,
                            use_postfix=False,
                        )
                        label_datastream_bpe = cast(LabelDatastream, train_data_bpe.datastreams["labels"])
                        vocab_size_without_blank = label_datastream_bpe.vocab_size
                        pos_emb_cfg = ConformerPosEmbConfig(
                            learnable_pos_emb=False,
                            rel_pos_clip=16,
                            with_linear_pos=True,
                            with_pos_bias=True,
                            separate_pos_emb_per_head=True,
                            pos_emb_dropout=0.0,
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
                            out_features=dim,
                            activation=None,
                        )
                        model_config_sub4 = RelPosModelConfig(
                            feature_extraction_config=fe_config,
                            frontend_config=frontend_config_sub4,
                            specaug_config=specaug_config,
                            label_target_size=vocab_size_without_blank,
                            conformer_size=dim,
                            num_layers=12,
                            num_heads=8,
                            ff_dim=2048,
                            att_weights_dropout=0.2,
                            conv_dropout=0.2,
                            ff_dropout=0.2,
                            mhsa_dropout=0.2,
                            conv_kernel_size=31,
                            final_dropout=0.2,
                            specauc_start_epoch=11,  # BPE does not converge otherwise
                            dropout_broadcast_axes=None,
                            module_list=["ff", "conv", "mhsa", "ff"],
                            module_scales=[0.5, 1.0, 1.0, 0.5],
                            aux_ctc_loss_layers=None,
                            aux_ctc_loss_scales=None,
                            pos_emb_config=pos_emb_cfg,
                            mhsa_with_bias=True,
                        )
                        train_config_24gbgpu_amp = {
                            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
                            "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))
                            + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
                            + list(np.linspace(5e-5, 1e-7, 30)),
                            #############
                            "batch_size": batch_size
                            * 16000,  # GPU MEM still very moderate, but larger batch did not help
                            "max_seq_length": {"audio_features": 35 * 16000},
                            "accum_grad_multiple_step": 1,
                            "torch_amp_options": {"dtype": "bfloat16"},
                        }

                        network_module_pos_enc_v1 = "ctc.conformer_0106.i6modelsRelPosEncV1_VGG4LayerActFrontendV1_v1"
                        train_args = {
                            "config": train_config_24gbgpu_amp,
                            "network_module": network_module_pos_enc_v1,
                            "net_args": {"model_config_dict": asdict(model_config_sub4)},
                            "debug": False,
                            "use_speed_perturbation": speed_perturbation,
                        }
                        if speed_perturbation:
                            training_name = (
                                prefix_name_bpe
                                + "/"
                                + network_module_pos_enc_v1
                                + f".{dim}dim_sub4_{batch_size}bs_speedpert_24gbgpu_{epochs}_amp"
                            )
                        else:
                            training_name = (
                                prefix_name_bpe
                                + "/"
                                + network_module_pos_enc_v1
                                + f".{dim}dim_sub4_{batch_size}bs_24gbgpu_{epochs}_amp"
                            )
                        train_job = training(
                            training_name, train_data_bpe, train_args, num_epochs=epochs, **default_returnn
                        )
                        train_job.rqmt["gpu_mem"] = 24
                        asr_model = prepare_asr_model(
                            training_name,
                            train_job,
                            train_args,
                            with_prior=True,
                            datasets=train_data_bpe,
                            get_specific_checkpoint=epochs,
                        )
                        recog_rasr_config, recog_rasr_post_config = get_tree_timesync_recog_config(
                            lexicon_file=get_bpe_bliss_lexicon(bpe_size=bpe, add_blank=True),
                            collapse_repeated_labels=True,
                            label_scorer_config=get_no_op_label_scorer_config(),
                            blank_index=vocab_size_without_blank,
                            max_beam_size=4096,
                            score_threshold=20.0, # this large search is too much when having higher bpe sizes, for small this is fine
                            logfile_suffix="recog",
                            lm_config=get_arpa_lm_config("4gram",
                                get_bpe_bliss_lexicon(bpe_size=bpe, add_blank=True), scale=0.0),
                        )
                        as_training_rasr_config = RasrDecoderConfig(
                            rasr_config_file=recog_rasr_config,
                            rasr_post_config=recog_rasr_post_config,
                            blank_log_penalty=None,
                            prior_scale=0.0,  # this will be overwritten internally
                            prior_file=None,
                            turn_off_quant="leave_as_is",
                        )

                        results = {}
                        res = tune_and_evaluate_helper(
                            training_name + "_rasr_larger",
                            asr_model,
                            as_training_rasr_config,
                            lm_scales=rasr_lm_scales,
                            prior_scales=rasr_prior_scales,
                        )
                        results.update(res)
                        generate_report(results=results, exp_name=training_name + "_rasr_larger")
                        full_res[training_name + f"_{bpe}" + "_rasr"] = copy.deepcopy(results)
                        del results

                        model_config_sub4 = MemristorModelTrainConfigV8(
                            feature_extraction_config=fe_config,
                            frontend_config=frontend_config_sub4,
                            specaug_config=specaug_config,
                            label_target_size=vocab_size_without_blank,
                            conformer_size=dim,
                            num_layers=12,
                            num_heads=8,
                            ff_dim=2048,
                            att_weights_dropout=0.2,
                            conv_dropout=0.2,
                            ff_dropout=0.2,
                            mhsa_dropout=0.2,
                            conv_kernel_size=31,
                            final_dropout=0.2,
                            specauc_start_epoch=11,  # BPE does not converge otherwise
                            dropout_broadcast_axes=None,
                            module_list=["ff", "conv", "mhsa", "ff"],
                            module_scales=[0.5, 1.0, 1.0, 0.5],
                            aux_ctc_loss_layers=None,
                            aux_ctc_loss_scales=None,
                            pos_emb_config=pos_emb_cfg,
                            weight_quant_dtype="qint8",
                            weight_quant_method="per_tensor",
                            activation_quant_dtype="qint8",
                            activation_quant_method="per_tensor",
                            dot_quant_dtype="qint8",
                            dot_quant_method="per_tensor",
                            Av_quant_dtype="qint8",
                            Av_quant_method="per_tensor",
                            moving_average=None,
                            weight_bit_prec=8,
                            activation_bit_prec=8,
                            quantize_output=True,
                            converter_hardware_settings=None,
                            quant_in_linear=True,
                            num_cycles=0,
                            correction_settings=None,
                            weight_noise_start_epoch=None,
                            weight_noise_func=None,
                            weight_noise_values=None,
                        )
                        train_config_24gbgpu_amp = {
                            "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-3},
                            "learning_rates": list(np.linspace(7e-6, 5e-4, (epochs - 30) // 2))
                                              + list(np.linspace(5e-4, 5e-5, (epochs - 30) // 2))
                                              + list(np.linspace(5e-5, 1e-7, 30)),
                            #############
                            "batch_size": batch_size
                                          * 16000,  # GPU MEM still very moderate, but larger batch did not help
                            "max_seq_length": {"audio_features": 35 * 16000},
                            "accum_grad_multiple_step": 1,
                            "torch_amp_options": {"dtype": "bfloat16"},
                        }

                        train_args = {
                            "config": train_config_24gbgpu_amp,
                            "network_module": network_module_mem_v9,
                            "net_args": {"model_config_dict": asdict(model_config_sub4)},
                            "debug": False,
                            "use_speed_perturbation": True,
                        }
                        training_name = (
                            prefix_name_bpe
                            + "/"
                            + network_module_mem_v9
                            + f".{dim}dim_sub4_{batch_size}bs_speedpert_24gbgpu_{epochs}_amp"
                        )
                        train_job = training(
                            training_name, train_data_bpe, train_args, num_epochs=epochs, **default_returnn
                        )
                        train_job.rqmt["gpu_mem"] = 24
                        asr_model = prepare_asr_model(
                            training_name,
                            train_job,
                            train_args,
                            with_prior=True,
                            datasets=train_data_bpe,
                            get_specific_checkpoint=epochs,
                            prior_config={"import_memristor": True}
                        )
                        recog_rasr_config, recog_rasr_post_config = get_tree_timesync_recog_config(
                            lexicon_file=get_bpe_bliss_lexicon(bpe_size=bpe, add_blank=True),
                            collapse_repeated_labels=True,
                            label_scorer_config=get_no_op_label_scorer_config(),
                            blank_index=vocab_size_without_blank,
                            max_beam_size=4096,
                            score_threshold=20.0,
                            # this large search is too much when having higher bpe sizes, for small this is fine
                            logfile_suffix="recog",
                            lm_config=get_arpa_lm_config("4gram",
                                get_bpe_bliss_lexicon(bpe_size=bpe, add_blank=True), scale=0.0),
                        )
                        as_training_rasr_config = RasrDecoderConfig(
                            rasr_config_file=recog_rasr_config,
                            rasr_post_config=recog_rasr_post_config,
                            blank_log_penalty=None,
                            prior_scale=0.0,  # this will be overwritten internally
                            prior_file=None,
                            turn_off_quant="leave_as_is",
                        )

                        results = {}
                        res = tune_and_evaluate_helper(
                            training_name + "_rasr_larger",
                            asr_model,
                            as_training_rasr_config,
                            lm_scales=rasr_lm_scales,
                            prior_scales=rasr_prior_scales,
                            import_memristor=True,
                        )
                        results.update(res)
                        generate_report(results=results, exp_name=training_name + "_rasr_larger")
                        full_res[training_name + f"_{bpe}" + "_rasr"] = copy.deepcopy(results)
                        del results

    tk.register_report("reports/ted/bpe_tune_report", partial(build_base_report, full_res, False), required=full_res, update_frequency=1200)
