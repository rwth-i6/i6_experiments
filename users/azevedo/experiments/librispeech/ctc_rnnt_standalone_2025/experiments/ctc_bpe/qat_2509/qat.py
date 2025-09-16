from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast, List
from functools import partial

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ....data.common import DatasetSettings, build_test_dataset
from ....data.bpe import build_bpe_training_datasets, get_text_lexicon
from ....default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ....lm import get_4gram_binary_lm
from ....pipeline import training, prepare_asr_model, search, ASRModel
# from ....report import generate_report
from ....report import tune_and_evalue_report
# from ...experiments.ctc_phon.tune_eval import build_qat_report

# from ..ctc_phon.tune_eval import eval_model

from ....pytorch_networks.common import Mode
from ....pytorch_networks.trainers.train_handler import TrainMode


def bpe_lib_qat_comparisons():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2025/ctc_bpe/256/qat_comparison"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=5,
        train_seq_ordering="laplace:.1000",
    )

    # build the training datasets object containing train, cv, dev-train and the extern_data dict
    train_data_bpe256 = build_bpe_training_datasets(
        prefix=prefix_name,
        bpe_size=256,  # TODO tune
        settings=train_settings,
        use_postfix=False,
    )
    label_datastream_bpe256 = cast(LabelDatastream, train_data_bpe256.datastreams["labels"])
    vocab_size_without_blank = label_datastream_bpe256.vocab_size

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

    def tune_and_evaluate_helper(
            training_name, dev_dataset_tuples, test_dataset_tuples, 
            asr_model, base_decoder_config, lm_scales, prior_scales, decoder_module,
            unhashed_decoder_config = None, debug=False, use_gpu=False, extra_forward_config=None
    ):
        tune_parameters = []
        tune_values_clean = []
        tune_values_other = []
        report_values = {}
        for lm_weight in lm_scales:
            for prior_scale in prior_scales:
                decoder_config = copy.deepcopy(base_decoder_config)
                if hasattr(decoder_config, "lm_scale"):
                    decoder_config.lm_scale = lm_weight
                else:
                    decoder_config.lm_weight = lm_weight
                decoder_config.prior_scale = prior_scale
                search_name = training_name + "/search_lm%.2f_prior%.2f" % (lm_weight, prior_scale)
                search_jobs, wers = search(
                    search_name,
                    forward_config=extra_forward_config if extra_forward_config else {},
                    asr_model=asr_model,
                    decoder_module=decoder_module,
                    decoder_args={"config": asdict(decoder_config)},
                    unhashed_decoder_args={"extra_config": asdict(unhashed_decoder_config)} if unhashed_decoder_config else None,
                    test_dataset_tuples=dev_dataset_tuples,
                    debug=debug,
                    use_gpu=use_gpu,
                    **default_returnn
                )
                tune_parameters.append((lm_weight, prior_scale))
                tune_values_clean.append((wers[search_name + "/dev-clean"]))
                tune_values_other.append((wers[search_name + "/dev-other"]))

        # for key, tune_values in [("test-other", tune_values_other)]:
        for key, tune_values in [("test-clean", tune_values_clean), ("test-other", tune_values_other)]:
            pick_optimal_params_job = GetOptimalParametersAsVariableJob(parameters=tune_parameters, values=tune_values, mode="minimize")
            pick_optimal_params_job.add_alias(training_name + f"/pick_best_{key}")
            decoder_config = copy.deepcopy(base_decoder_config)
            if hasattr(decoder_config, "lm_scale"):
                decoder_config.lm_scale = pick_optimal_params_job.out_optimal_parameters[0]
            else:
                decoder_config.lm_weight = pick_optimal_params_job.out_optimal_parameters[0]
            decoder_config.prior_scale = pick_optimal_params_job.out_optimal_parameters[1]
            search_jobs, wers = search(
                training_name,
                forward_config=extra_forward_config if extra_forward_config else {},
                asr_model=asr_model,
                decoder_module=decoder_module,
                decoder_args={"config": asdict(decoder_config)},
                unhashed_decoder_args={
                    "extra_config": asdict(unhashed_decoder_config)} if unhashed_decoder_config else None,
                test_dataset_tuples={key: test_dataset_tuples[key]},
                use_gpu=use_gpu,
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
        
        
    # from ....pytorch_networks.ctc.decoder.flashlight_qat_phoneme_ctc import DecoderConfig

    # default_decoder_config_bpe256 = DecoderConfig(
    #     lexicon=get_text_lexicon(prefix=prefix_name, bpe_size=256),
    #     returnn_vocab=label_datastream_bpe256.vocab,
    #     beam_size=1024,  # Untuned
    #     beam_size_token=16,  # makes it much faster (0.3 search RTF -> 0.04 search RTF), but looses 0.1% WER over 128
    #     arpa_lm=arpa_4gram_lm,
    #     beam_threshold=14,  # Untuned
    # )
    # as_training_decoder_config = DecoderConfig(
    #     lexicon=get_text_lexicon(prefix=prefix_name, bpe_size=256),
    #     returnn_vocab=label_datastream_bpe256.vocab,
    #     beam_size=1024,
    #     beam_size_token=12,  # makes it much faster
    #     arpa_lm=arpa_4gram_lm,
    #     beam_threshold=14,
    #     turn_off_quant="leave_as_is",
    # )

    # from ....pytorch_networks.ctc.decoder.greedy_bpe_ctc_v3 import DecoderConfig as GreedyCTCDecoderConfig

    # as_training_greedy_decoder_config = GreedyCTCDecoderConfig(
    #     returnn_vocab=label_datastream_bpe256.vocab,
    #     turn_off_quant="leave_as_is",
    # )

    from ....pytorch_networks.ctc.qat_2509.full_qat_v1_cfg import (
        SpecaugConfig,
        VGG4LayerActFrontendV1Config_mod,
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
        center=True,
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
    qat_report = {}

    train_config = {
        "optimizer": {
            "class": "radam",
            "epsilon": 1e-16,
            "weight_decay": 1e-2,
            "decoupled_weight_decay": True,
        },
        "learning_rates": list(np.linspace(7e-6, 5e-4, 110))
        + list(np.linspace(5e-4, 5e-5, 110))
        + list(np.linspace(5e-5, 1e-7, 30)),
        #############
        "batch_size": 180 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
    }
    # from ..ctc_phon.tune_eval import RTFArgs

    # rtf_args = RTFArgs(
    #     beam_sizes=[256, 512, 1024, 4096],
    #     beam_size_tokens=[4, 8, 12, 20, 30],
    #     beam_thresholds=[4, 8, 20, 30],
    #     decoder_module="ctc.decoder.flashlight_ctc_v5_rescale_measure",
    #     include_gpu=True,
    # )

    ####################################################################################################
    # QAT Baseline
    # network_module_v4 = "ctc.qat_2509.baseline_qat_v4"
    # from ....pytorch_networks.ctc.qat_2509.baseline_qat_v4_cfg import QuantModelTrainConfigV4
    
    network_module_v4_streamable = "ctc.qat_25-0.baseline_qat_v4_streamable"
    from ....pytorch_networks.ctc.qat_2509.baseline_qat_v4_streamable_cfg import QuantModelTrainConfigV4

    model_config = QuantModelTrainConfigV4(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub6,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=4,
        ff_dim=1536,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
        specauc_start_epoch=11,
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
        quantize_output=False,
        extra_act_quant=False,
        quantize_bias=None,
        observer_only_in_train=False,

        # streaming params
        chunk_size=0.27 * 16000,  # samples corresponding to 28 frames
        lookahead_size=8,
        carry_over_size=1,
        dual_mode=False,
        streaming_scale=1,
        train_mode=str(TrainMode.STREAMING),

    )

    train_args = {
        "config": train_config,
        "network_module": network_module_v4,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "post_config": {"num_workers_per_gpu": 8},
        "use_speed_perturbation": True,
    }

    training_name = prefix_name + "/" + network_module_v4 + f"_8_8_bpe"
    train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    results = {}
    # results = eval_model(
    #     training_name=training_name,
    #     train_job=train_job,
    #     train_args=train_args,
    #     train_data=train_data_bpe256,
    #     decoder_config=as_training_decoder_config,
    #     dev_dataset_tuples=dev_dataset_tuples,
    #     result_dict=results,
    #     decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
    #     prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
    #     lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
    #     run_rtf=True,
    #     rtf_args=rtf_args,
    # )

    # generate_report(results=results, exp_name=training_name)
    # qat_report[training_name] = results

    # # `Succesful test for fixing the AM RTF issue
    # rtf_args_big = RTFArgs(
    #     beam_sizes=[4096],
    #     beam_size_tokens=[30],
    #     beam_thresholds=[30],
    #     decoder_module="ctc.decoder.flashlight_ctc_v6_rescale_measure",
    # )
    # results = eval_model(
    #     training_name=training_name + "_big",
    #     train_job=train_job,
    #     train_args=train_args,
    #     train_data=train_data_bpe256,
    #     decoder_config=as_training_decoder_config,
    #     dev_dataset_tuples=dev_dataset_tuples,
    #     result_dict=results,
    #     decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
    #     prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
    #     lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
    #     run_rtf=True,
    #     rtf_args=rtf_args_big,
    # )

    # rtf_args_greedy = RTFArgs(
    #     beam_sizes=None,
    #     beam_size_tokens=None,
    #     beam_thresholds=None,
    #     decoder_module="ctc.decoder.greedy_bpe_ctc_rescale_measure_v1",
    #     type="greedy",
    #     include_gpu=True,
    # )
    # results = {}
    # results = eval_model(
    #     training_name=training_name + "/greedy",
    #     train_job=train_job,
    #     train_args=train_args,
    #     train_data=train_data_bpe256,
    #     decoder_config=as_training_greedy_decoder_config,
    #     dev_dataset_tuples=dev_dataset_tuples,
    #     result_dict=results,
    #     decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
    #     prior_scales=[0.0],
    #     lm_scales=[0.0],
    #     with_prior=False,
    #     run_rtf=True,
    #     rtf_args=rtf_args_greedy,
    # )
    # generate_report(results=results, exp_name=training_name + "_greedy")
    # qat_report[training_name + "_greedy"] = results

    # # Neural LM
    # from ...pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v3 import DecoderConfig as BeamSearchDecoderConfig
    # from ...pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v4 import DecoderConfig as BeamSearchDecoderConfigv4
    # from ...pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v3 import DecoderExtraConfig
    # from ... import PACKAGE

    # from ...pytorch_networks.lm.kazuki_lstm_zijian_variant_v1_cfg import ModelConfig

    # default_init_args = {
    #     "init_args_w": {"func": "normal", "arg": {"mean": 0.0, "std": 0.1}},
    #     "init_args_b": {"func": "normal", "arg": {"mean": 0.0, "std": 0.1}},
    # }

    # lm_config = ModelConfig(
    #     vocab_dim=vocab_size_without_blank,
    #     embed_dim=512,
    #     hidden_dim=2048,
    #     n_lstm_layers=2,
    #     use_bottle_neck=False,
    #     dropout=0.2,
    #     init_args=default_init_args,
    # )
    # lm = "/work/asr3/zyang/share/zhan/torch_setup/work/i6_core/returnn/training/ReturnnTrainingJob.qIIaRxZQmaBL/output/models/epoch.200.pt"
    # lm_net_args = asdict(lm_config)

    # rtf_lm = RTFArgs(
    #     beam_sizes=None,
    #     beam_size_tokens=None,
    #     beam_thresholds=None,
    #     include_gpu=True,
    #     type="nn_lm",
    #     decoder_module="ctc.decoder.beam_search_bpe_ctc_v4_rescale_measure_v3",
    # )

    from ....pytorch_networks.search.decoder_module import DecoderConfig, ExtraConfig
    from ....pytorch_networks.ctc.search import CTCBeamSearchConfig

    asr_model = prepare_asr_model(
        training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe256,
        get_specific_checkpoint=250,
    )
    search_config = CTCBeamSearchConfig(
        lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=256),
        beam_size_token=16,
        beam_threshold=14,  # Untuned,
        lm_package=arpa_4gram_lm,
        prior_file=asr_model.prior_file,
    )

    for beam in [10, 30, 128, 200, 256, 300, 315]:
        decoder_config_streaming = DecoderConfig(
            beam_size=beam,
            returnn_vocab=label_datastream_bpe256.vocab,

            search_config=search_config,

            mode=Mode.STREAMING.name,
            chunk_size=int(model_config.chunk_size),
            lookahead_size=int(model_config.lookahead_size * 0.06 * 16e3),
            carry_over_size=model_config.carry_over_size,
            test_version=0.0,
        )
        decoder_config_offline = DecoderConfig(
            beam_size=beam,
            returnn_vocab=label_datastream_bpe256.vocab,
            search_config=search_config,

            mode=Mode.OFFLINE.name,
            test_version=0.0,
        )

        tune_and_evaluate_helper(
            training_name + "/offline/4gram_lm",
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=test_dataset_tuples,
            asr_model=asr_model,
            base_decoder_config=decoder_config_offline,
            lm_scales=[0, 0.5, 0.7, 0.9, 1.4, 1.5, 1.6, 2.0],
            prior_scales=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            decoder_module="search.decoder_module",
            debug=True,
            use_gpu=False,
        )
        tune_and_evaluate_helper(
            training_name + "/streaming/4gram_lm",
            dev_dataset_tuples=dev_dataset_tuples,
            test_dataset_tuples=test_dataset_tuples,
            asr_model=asr_model,
            base_decoder_config=decoder_config_streaming,
            lm_scales=[0, 0.5, 0.7, 0.9, 1.4, 1.5, 1.6, 2.0],
            prior_scales=[0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
            decoder_module="search.decoder_module",
            debug=True,
            use_gpu=False,
        )


        # beam_search_decoder_config_v4_lstmlm = BeamSearchDecoderConfigv4(
        #     returnn_vocab=label_datastream_bpe256.vocab,
        #     beam_size=beam,
        #     lm_model_args=lm_net_args,
        #     lm_checkpoint=lm,
        #     lm_module="pytorch_networks.lm.kazuki_lstm_zijian_variant_v1.Model",
        #     lm_states_need_label_axis=False,
        # )
        # decoder_unhashed_config_v3 = DecoderExtraConfig(
        #     lm_package=PACKAGE,
        # )
        # train_args["debug"] = True
        # results = {}
        # eval_model(
        #     training_name=training_name + f"/lstm_lm_{beam}",
        #     train_job=train_job,
        #     train_args=train_args,
        #     train_data=train_data_bpe256,
        #     decoder_config=beam_search_decoder_config_v4_lstmlm,
        #     unhashed_decoder_args=decoder_unhashed_config_v3,
        #     dev_dataset_tuples=dev_dataset_tuples,
        #     result_dict=results,
        #     decoder_module="ctc.decoder.beam_search_bpe_ctc_v4",
        #     prior_scales=[0.3, 0.5, 0.7, 0.9, 1.0],
        #     lm_scales=[0.5, 0.7, 0.9, 1.0, 1.1, 1.2, 1.4],
        #     # prior_scales=[0.5],
        #     # lm_scales=[2.0],
        #     with_prior=True,
        #     run_rtf=True,
        #     use_gpu=True,
        #     run_best=False,
        #     run_best_4=False,
        #     extra_forward_config={"batch_size": 200 * 16000},
        #     rtf_args=rtf_lm,
        # )
        # generate_report(results=results, exp_name=training_name + f"_lstm_lm_{beam}")
        # qat_report[training_name + f"_lstm_lm_{beam}"] = results

    # from ...pytorch_networks.lm.kazuki_trafo_zijian_variant_v1_cfg import (
    #     TransformerLMConfig,
    #     TransformerMHSAConfig,
    #     TransformerLinearConfig,
    #     TransformerBlockConfig,
    # )

    # hidden_dim = 768
    # ff_dim = 4096
    # input_dim = hidden_dim
    # output_dim = hidden_dim
    # linear_config = TransformerLinearConfig(
    #     input_dim=input_dim, ff_dim=ff_dim, output_dim=output_dim, dropout=0.0, batch_first=True
    # )
    # mhsa_config = TransformerMHSAConfig(input_dim=input_dim, num_heads=4, dropout=0.0, batch_first=True)
    # block_config = TransformerBlockConfig(linear_config=linear_config, mhsa_config=mhsa_config)
    # trafo_base_config = TransformerLMConfig(
    #     embed_dim=128,
    #     hidden_dim=hidden_dim,
    #     vocab_dim=vocab_size_without_blank,
    #     num_layers=12,
    #     block_config=block_config,
    #     batch_first=True,  # very important, state management in decoder does not work otherwise
    #     dropout=0.0,
    # )
    # lm = "/work/asr3/zyang/share/zhan/torch_setup/work/i6_core/returnn/training/ReturnnTrainingJob.JiXjvdOlsRMv/output/models/epoch.200.pt"
    # lm_net_args = asdict(trafo_base_config)
    # for beam in [32, 64, 128, 200, 256]:
    #     beam_search_decoder_config_v4_trafo = BeamSearchDecoderConfigv4(
    #         returnn_vocab=label_datastream_bpe256.vocab,
    #         beam_size=beam,
    #         lm_model_args=lm_net_args,
    #         lm_checkpoint=lm,
    #         lm_module="pytorch_networks.lm.kazuki_trafo_zijian_variant_v1_decoding.Model",
    #         lm_states_need_label_axis=True,
    #     )
    #     decoder_unhashed_config_v3 = DecoderExtraConfig(
    #         lm_package=PACKAGE,
    #     )
    #     train_args["debug"] = True
    #     results = {}
    #     eval_model(
    #         training_name=training_name + f"/trafo_lm_{beam}",
    #         train_job=train_job,
    #         train_args=train_args,
    #         train_data=train_data_bpe256,
    #         decoder_config=beam_search_decoder_config_v4_trafo,
    #         unhashed_decoder_args=decoder_unhashed_config_v3,
    #         dev_dataset_tuples=dev_dataset_tuples,
    #         result_dict=results,
    #         decoder_module="ctc.decoder.beam_search_bpe_ctc_v4",
    #         prior_scales=[0.3, 0.5, 0.7, 0.9],
    #         lm_scales=[0.5, 0.7, 0.9, 1.0, 1.2],
    #         with_prior=True,
    #         run_rtf=True,
    #         rtf_args=rtf_lm,
    #         use_gpu=True,
    #         run_best=False,
    #         run_best_4=False,
    #         extra_forward_config={"batch_size": 150 * 16000} if beam <= 128 else {"max_seqs": 1},
    #     )
    #     generate_report(results=results, exp_name=training_name + f"_trafo_lm_{beam}")
    #     qat_report[training_name + f"_trafo_lm_{beam}"] = results

    # from ...pytorch_networks.lm.kazuki_trafo_zijian_variant_v1_cfg import (
    #     TransformerLMConfig,
    #     TransformerMHSAConfig,
    #     TransformerLinearConfig,
    #     TransformerBlockConfig,
    # )

    # hidden_dim = 768
    # ff_dim = 4096
    # input_dim = hidden_dim
    # output_dim = hidden_dim
    # linear_config = TransformerLinearConfig(
    #     input_dim=input_dim, ff_dim=ff_dim, output_dim=output_dim, dropout=0.0, batch_first=True
    # )
    # mhsa_config = TransformerMHSAConfig(input_dim=input_dim, num_heads=4, dropout=0.0, batch_first=True)
    # block_config = TransformerBlockConfig(linear_config=linear_config, mhsa_config=mhsa_config)
    # trafo_base_config = TransformerLMConfig(
    #     embed_dim=128,
    #     hidden_dim=hidden_dim,
    #     vocab_dim=vocab_size_without_blank,
    #     num_layers=24,
    #     block_config=block_config,
    #     batch_first=True,  # very important, state management in decoder does not work otherwise
    #     dropout=0.0,
    # )
    # lm = "/work/asr3/zyang/share/zhan/torch_setup/work/i6_core/returnn/training/ReturnnTrainingJob.L1cJPJEMZufI/output/models/epoch.200.pt"
    # lm_net_args = asdict(trafo_base_config)
    # for beam in [32, 64, 100, 128, 150]:
    #     beam_search_decoder_config_v4_trafo = BeamSearchDecoderConfigv4(
    #         returnn_vocab=label_datastream_bpe256.vocab,
    #         beam_size=beam,
    #         lm_model_args=lm_net_args,
    #         lm_checkpoint=lm,
    #         lm_module="pytorch_networks.lm.kazuki_trafo_zijian_variant_v1_decoding.Model",
    #         lm_states_need_label_axis=True,
    #     )
    #     decoder_unhashed_config_v3 = DecoderExtraConfig(
    #         lm_package=PACKAGE,
    #     )
    #     train_args["debug"] = True
    #     results = {}
    #     eval_model(
    #         training_name=training_name + f"/trafo_24l_lm_{beam}",
    #         train_job=train_job,
    #         train_args=train_args,
    #         train_data=train_data_bpe256,
    #         decoder_config=beam_search_decoder_config_v4_trafo,
    #         unhashed_decoder_args=decoder_unhashed_config_v3,
    #         dev_dataset_tuples=dev_dataset_tuples,
    #         result_dict=results,
    #         decoder_module="ctc.decoder.beam_search_bpe_ctc_v4",
    #         prior_scales=[0.3, 0.5, 0.7, 0.9],
    #         lm_scales=[0.5, 0.7, 0.9, 1.0, 1.2],
    #         with_prior=True,
    #         run_rtf=beam < 128,
    #         rtf_args=rtf_lm if beam < 128 else None,
    #         use_gpu=True,
    #         run_best=False,
    #         run_best_4=False,
    #         extra_forward_config={"batch_size": 150 * 16000} if beam < 128 else {"max_seqs": 1},
    #     )
    #     generate_report(results=results, exp_name=training_name + f"_trafo_24l_lm_{beam}")
    #     qat_report[training_name + f"_trafo_24l_lm_{beam}"] = results


    # TODO RASR Search


    #########################################################################################
    """
    # Full Quant Baseline
    network_module_v1 = "ctc.qat_0711.full_qat_v1"
    from ...pytorch_networks.ctc.qat_0711.full_qat_v1_cfg import QuantModelTrainConfigV4

    model_config = QuantModelTrainConfigV4(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config_sub6,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=384,
        num_layers=12,
        num_heads=4,
        ff_dim=1536,
        att_weights_dropout=0.2,
        conv_dropout=0.2,
        ff_dropout=0.2,
        mhsa_dropout=0.2,
        conv_kernel_size=31,
        final_dropout=0.2,
        specauc_start_epoch=11,
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
        quantize_output=False,
        extra_act_quant=False,
        quantize_bias=None,
        observer_only_in_train=False,
    )
    train_args = {
        "config": train_config,
        "network_module": network_module_v1,
        "net_args": {"model_config_dict": asdict(model_config)},
        "debug": False,
        "post_config": {"num_workers_per_gpu": 8},
        "use_speed_perturbation": True,
    }

    training_name = prefix_name + "/" + network_module_v1 + f"_{8}_{8}"
    train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
    train_job.rqmt["gpu_mem"] = 48
    results = {}
    results = eval_model(
        training_name=training_name,
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
        prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
        lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
        run_rtf=False,
        rtf_args=None,
    )
    generate_report(results=results, exp_name=training_name)
    qat_report[training_name] = results

    results = {}
    results = eval_model(
        training_name=training_name + "/greedy",
        train_job=train_job,
        train_args=train_args,
        train_data=train_data_bpe256,
        decoder_config=as_training_greedy_decoder_config,
        dev_dataset_tuples=dev_dataset_tuples,
        result_dict=results,
        decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
        prior_scales=[0.0],
        lm_scales=[0.0],
        with_prior=False,
        run_rtf=False,
    )
    generate_report(results=results, exp_name=training_name + "_greedy")
    qat_report[training_name + "_greedy"] = results

    for ff_dim in [512, 1024]:
        # ########################################################################
        # # FF 512 and 1024
        # frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
        #     in_features=80,
        #     conv1_channels=32,
        #     conv2_channels=64,
        #     conv3_channels=64,
        #     conv4_channels=32,
        #     conv_kernel_size=(3, 3),
        #     conv_padding=None,
        #     pool1_kernel_size=(3, 1),
        #     pool1_stride=(2, 1),
        #     pool1_padding=None,
        #     pool2_kernel_size=(2, 1),
        #     pool2_stride=(2, 1),
        #     pool2_padding=None,
        #     activation_str="ReLU",
        #     out_features=512,
        #     activation=None,
        # )
        #
        # model_config = QuantModelTrainConfigV4(
        #     feature_extraction_config=fe_config,
        #     frontend_config=frontend_config_sub6_512,
        #     specaug_config=specaug_config,
        #     label_target_size=vocab_size_without_blank,
        #     conformer_size=512,
        #     num_layers=12,
        #     num_heads=4,
        #     ff_dim=1024,
        #     att_weights_dropout=0.2,
        #     conv_dropout=0.2,
        #     ff_dropout=0.2,
        #     mhsa_dropout=0.2,
        #     conv_kernel_size=31,
        #     final_dropout=0.2,
        #     specauc_start_epoch=11,
        #     weight_quant_dtype="qint8",
        #     weight_quant_method="per_tensor",
        #     activation_quant_dtype="qint8",
        #     activation_quant_method="per_tensor",
        #     dot_quant_dtype="qint8",
        #     dot_quant_method="per_tensor",
        #     Av_quant_dtype="qint8",
        #     Av_quant_method="per_tensor",
        #     moving_average=None,
        #     weight_bit_prec=8,
        #     activation_bit_prec=8,
        #     quantize_output=False,
        #     extra_act_quant=False,
        #     quantize_bias=None,
        #     observer_only_in_train=False,
        # )
        # train_args = {
        #     "config": train_config,
        #     "network_module": network_module_v1,
        #     "net_args": {"model_config_dict": asdict(model_config)},
        #     "debug": False,
        #     "post_config": {"num_workers_per_gpu": 8},
        #     "use_speed_perturbation": True,
        # }
        #
        # training_name = prefix_name + "/" + network_module_v1 + f"_8_8_512_1024"
        # train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        # train_job.rqmt["gpu_mem"] = 48
        # results = {}
        # results = eval_model(
        #     training_name=training_name,
        #     train_job=train_job,
        #     train_args=train_args,
        #     train_data=train_data_bpe256,
        #     decoder_config=as_training_decoder_config,
        #     dev_dataset_tuples=dev_dataset_tuples,
        #     result_dict=results,
        #     decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
        #     prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
        #     lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
        # )
        # generate_report(results=results, exp_name=training_name)
        # qat_report[training_name] = results
        #
        # results = {}
        # results = eval_model(
        #     training_name=training_name + "/greedy",
        #     train_job=train_job,
        #     train_args=train_args,
        #     train_data=train_data_bpe256,
        #     decoder_config=as_training_greedy_decoder_config,
        #     dev_dataset_tuples=dev_dataset_tuples,
        #     result_dict=results,
        #     decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
        #     prior_scales=[0.0],
        #     lm_scales=[0.0],
        #     with_prior=False,
        # )
        # generate_report(results=results, exp_name=training_name + "_greedy")
        # qat_report[training_name + "_greedy"] = results

        ########################################################################
        # FF 512 and 512
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
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
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
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
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )
        train_args = {
            "config": train_config,
            "network_module": network_module_v1,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1 + f"_8_8_512_{ff_dim}"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results
        ########################################################################
        # FF 512 and 512 with mean abs
        network_module_v1_mean = "ctc.qat_0711.full_qat_v1_mean_abs_norm"
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
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
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
            specauc_start_epoch=11,
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
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )
        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean + f"_8_8_512_{ff_dim}"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_best_4=False,
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
            run_best_4=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

        ########################################################################
        # FF 512 and 512 with sym and means abs
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
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
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
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
            weight_bit_prec=8,
            activation_bit_prec=8,
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean + f"_8_8_512_{ff_dim}_sym"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

        ########################################################################
        # FF 512 and 512 with sym and means abs and ReLu
        network_module_v1_mean_relu = "ctc.qat_0711.full_qat_v1_relu_mean_abs"
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
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
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
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
            weight_bit_prec=8,
            activation_bit_prec=8,
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu + f"_8_8_512_{ff_dim}_sym"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

        ########################################################################
        # sym and means abs and ReLu and shared observers (faulty old)
        network_module_v1_mean_relu_shared_obs = "ctc.qat_0711.full_qat_v1_relu_mean_abs_shared_obs"
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
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
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
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
            weight_bit_prec=8,
            activation_bit_prec=8,
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu_shared_obs,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu_shared_obs + f"_8_8_512_{ff_dim}_sym"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

        ########################################################################
        # sym and means abs and ReLu and shared observers
        network_module_v1_mean_relu_shared_obs_v2 = "ctc.qat_0711.full_qat_v1_relu_mean_abs_shared_obs_v2"
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
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
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=12,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
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
            weight_bit_prec=8,
            activation_bit_prec=8,
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu_shared_obs_v2,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu_shared_obs_v2 + f"_8_8_512_{ff_dim}_sym"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

        ########################################################################
        # FF 512 and 512 with sym and means abs and ReLu 16 L
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
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
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=16,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
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
            weight_bit_prec=8,
            activation_bit_prec=8,
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu + f"_8_8_512_{ff_dim}_sym_16l"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_best_4=False,
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
            run_best_4=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

        ########################################################################
        # FF 512 and 512 with sym and means abs and ReLu 20 L
        frontend_config_sub6_512 = VGG4LayerActFrontendV1Config_mod(
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
            out_features=512,
            activation=None,
        )

        model_config = QuantModelTrainConfigV4(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config_sub6_512,
            specaug_config=specaug_config,
            label_target_size=vocab_size_without_blank,
            conformer_size=512,
            num_layers=20,
            num_heads=4,
            ff_dim=ff_dim,
            att_weights_dropout=0.2,
            conv_dropout=0.2,
            ff_dropout=0.2,
            mhsa_dropout=0.2,
            conv_kernel_size=31,
            final_dropout=0.2,
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
            weight_bit_prec=8,
            activation_bit_prec=8,
            quantize_output=False,
            extra_act_quant=False,
            quantize_bias=None,
            observer_only_in_train=False,
        )

        train_args = {
            "config": train_config,
            "network_module": network_module_v1_mean_relu,
            "net_args": {"model_config_dict": asdict(model_config)},
            "debug": False,
            "post_config": {"num_workers_per_gpu": 8},
            "use_speed_perturbation": True,
        }

        training_name = prefix_name + "/" + network_module_v1_mean_relu + f"_8_8_512_{ff_dim}_sym_20l"
        train_job = training(training_name, train_data_bpe256, train_args, num_epochs=250, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        results = {}
        results = eval_model(
            training_name=training_name,
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.flashlight_qat_phoneme_ctc",
            prior_scales=[0.1, 0.3, 0.5, 0.7, 0.9],
            lm_scales=[1.8, 2.0, 2.2, 2.4, 2.6, 2.8],
            run_best_4=False,
            run_rtf=False,
            rtf_args=None,
        )
        generate_report(results=results, exp_name=training_name)
        qat_report[training_name] = results

        results = {}
        results = eval_model(
            training_name=training_name + "/greedy",
            train_job=train_job,
            train_args=train_args,
            train_data=train_data_bpe256,
            decoder_config=as_training_greedy_decoder_config,
            dev_dataset_tuples=dev_dataset_tuples,
            result_dict=results,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            prior_scales=[0.0],
            lm_scales=[0.0],
            with_prior=False,
            run_best_4=False,
        )
        generate_report(results=results, exp_name=training_name + "_greedy")
        qat_report[training_name + "_greedy"] = results

    tk.register_report("reports/qat_report_bpe_comparison", partial(build_qat_report, qat_report), required=qat_report)
"""