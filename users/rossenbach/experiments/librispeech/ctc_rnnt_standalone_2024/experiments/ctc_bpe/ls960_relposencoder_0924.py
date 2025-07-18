from sisyphus import tk

import copy
from dataclasses import asdict
import numpy as np
from typing import cast

from i6_core.tools.parameter_tuning import GetOptimalParametersAsVariableJob

from i6_experiments.common.setups.returnn.datastreams.vocabulary import LabelDatastream

from ...data.common import DatasetSettings, build_test_dataset
from ...data.bpe import build_bpe_training_datasets, get_text_lexicon
from ...default_tools import RETURNN_EXE, MINI_RETURNN_ROOT
from ...lm import get_4gram_binary_lm
from ...pipeline import training, prepare_asr_model, search, ASRModel
from ...storage import add_ctc_model, get_lm_model, NeuralLM
from ...report import tune_and_evalue_report
from ... import PACKAGE


def bpe_ls960_0924_relposencoder():
    prefix_name = "experiments/librispeech/ctc_rnnt_standalone_2024/ls960_ctc_bpe_relposencoder_0924"

    train_settings = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.1000",
    )
    
    train_settings_laplace4 = DatasetSettings(
        preemphasis=0.97,  # TODO: Check if this is really useful
        peak_normalization=True,  # TODO: Also check if really useful, older Attention setups did not have that
        # training
        train_partition_epoch=10,
        train_seq_ordering="laplace:.4000",
    )

    arpa_4gram_lm = get_4gram_binary_lm(prefix_name=prefix_name)

    default_returnn = {
        "returnn_exe": RETURNN_EXE,
        "returnn_root": MINI_RETURNN_ROOT,
    }

    from ...pytorch_networks.ctc.decoder.flashlight_ctc_v1 import DecoderConfig
    from ...pytorch_networks.ctc.decoder.greedy_bpe_ctc_v3 import DecoderConfig as GreedyDecoderConfig



    from ...pytorch_networks.ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1_cfg import \
        SpecaugConfig, VGG4LayerActFrontendV1Config_mod, ModelConfig, LogMelFeatureExtractionV1Config, ConformerPosEmbConfig

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
        max_dim_feat=16,  # classic style
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

    # Try to do like returnn frontend
    posemb_config = ConformerPosEmbConfig(
        learnable_pos_emb=False,
        rel_pos_clip=16,
        with_linear_pos=True,
        with_pos_bias=True,
        separate_pos_emb_per_head=True,
        pos_emb_dropout=0.0,
    )



    train_config_24gbgpu_amp = {
        "optimizer": {"class": "adamw", "epsilon": 1e-16, "weight_decay": 1e-2},
        "learning_rates": list(np.linspace(7e-6, 5e-4, 480)) + list(
                np.linspace(5e-4, 5e-5, 480)) + list(np.linspace(5e-5, 1e-7, 40)),
        #############
        "batch_size": 240 * 16000,
        "max_seq_length": {"audio_features": 35 * 16000},
        "accum_grad_multiple_step": 1,
        "gradient_clip_norm": 1.0,
        "torch_amp_options": {"dtype": "bfloat16"},
    }

    network_module = "ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1"
    global_train_args = {
        "config": train_config_24gbgpu_amp,
        "network_module": network_module,
        "use_speed_perturbation": True,
        "debug": True,
    }

    def tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, base_decoder_config, lm_scales, prior_scales, unhashed_decoder_config = None, decoder_module="ctc.decoder.flashlight_ctc_v1", debug=False, use_gpu=False, extra_forward_config=None):
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

    def greedy_search_helper(
            training_name: str,
            asr_model: ASRModel,
            decoder_config: GreedyDecoderConfig
        ):
        # remove prior if exists
        asr_model = copy.deepcopy(asr_model)
        asr_model.prior_file = None

        search_name = training_name + "/search_greedy"
        search_jobs, wers = search(
            search_name,
            forward_config={},
            asr_model=asr_model,
            decoder_module="ctc.decoder.greedy_bpe_ctc_v3",
            decoder_args={"config": asdict(decoder_config)},
            test_dataset_tuples={**dev_dataset_tuples, **test_dataset_tuples},
            **default_returnn,
        )

    # for BPE_SIZE in [0, 128, 512]:
    for BPE_SIZE in [128]:

        # build the training datasets object containing train, cv, dev-train and the extern_data dict
        train_data_bpe = build_bpe_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-other-960",
            bpe_size=BPE_SIZE,
            settings=train_settings,
            use_postfix=False,
        )
        train_data_bpe_laplace4 = build_bpe_training_datasets(
            prefix=prefix_name,
            librispeech_key="train-other-960",
            bpe_size=BPE_SIZE,
            settings=train_settings_laplace4,
            use_postfix=False,
        )
        label_datastream_bpe = cast(LabelDatastream, train_data_bpe.datastreams["labels"])
        vocab_size_without_blank = label_datastream_bpe.vocab_size

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

        model_config = ModelConfig(
            feature_extraction_config=fe_config,
            frontend_config=frontend_config,
            pos_emb_config=posemb_config,
            specaug_config=specaug_config,
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
            dropout_broadcast_axes=None, # No dropout broadcast yet to properly compare
            module_list=["ff", "conv", "mhsa", "ff"],
            module_scales=[0.5, 1.0, 1.0, 0.5],
            aux_ctc_loss_layers=None,
            aux_ctc_loss_scales=None,
        )

        model_config_dropbt = copy.deepcopy(model_config)
        model_config_dropbt.dropout_broadcast_axes = "BT"

        #model_config_decoding = copy.deepcopy(model_config)
        #model_config_decoding.aux_ctc_loss_scales = [0.0, 0.0, 1.0]  # for decoding use result only of last layer

        default_decoder_config_bpe = DecoderConfig(
            lexicon=get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=BPE_SIZE),
            returnn_vocab=label_datastream_bpe.vocab,
            beam_size=1024,
            beam_size_token=16,  # makes it much faster
            arpa_lm=arpa_4gram_lm,
            beam_threshold=14,
        )

        greedy_decoder_config = GreedyDecoderConfig(
            returnn_vocab=label_datastream_bpe.vocab,
        )
        
        train_args = copy.deepcopy(global_train_args)
        train_args["net_args"] = {"model_config_dict": asdict(model_config)}

        training_name = prefix_name + "/" + str(BPE_SIZE) + "/" + network_module + ".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_smallbatch"
        train_job = training(training_name, train_data_bpe, train_args, num_epochs=1000, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        asr_model = prepare_asr_model(
            training_name, train_job, train_args, with_prior=True, datasets=train_data_bpe, get_specific_checkpoint=1000
        )
        tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, default_decoder_config_bpe, lm_scales=[1.6, 1.8, 2.0], prior_scales=[0.2, 0.3, 0.4])
        greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)


        # LM STUFF HERE
        from ...pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v3 import DecoderConfig as BeamSearchDecoderConfig
        from ...pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v4 import DecoderConfig as BeamSearchDecoderConfigv4
        from ...pytorch_networks.ctc.decoder.beam_search_bpe_ctc_v3 import DecoderExtraConfig
        trafo_12x768 : NeuralLM = get_lm_model("bpe%i_trafo12x768_2ep" % BPE_SIZE)
        trafo_32x768 : NeuralLM = get_lm_model("bpe%i_trafo32x768_5ep" % BPE_SIZE)
        lstm_2x2048 : NeuralLM = get_lm_model("bpe%i_2x2024_kazuki_lstmlm_3ep" % BPE_SIZE)
        beam_search_decoder_config = BeamSearchDecoderConfig(
            returnn_vocab=label_datastream_bpe.vocab,
            beam_size=10,
            lm_model_args = trafo_12x768.net_args,
            lm_checkpoint = trafo_12x768.checkpoint,
            lm_module = "pytorch_networks.lm.trafo.kazuki_trafo_zijian_variant_v2.Model",
        )
        beam_search_decoder_config_v4 = BeamSearchDecoderConfigv4(
            returnn_vocab=label_datastream_bpe.vocab,
            beam_size=10,
            lm_model_args = trafo_12x768.net_args,
            lm_checkpoint = trafo_12x768.checkpoint,
            lm_module = "pytorch_networks.lm.trafo.kazuki_trafo_zijian_variant_v2.Model",
            lm_states_need_label_axis=True,
        )
        beam_search_decoder_config_v4_32lm = BeamSearchDecoderConfigv4(
            returnn_vocab=label_datastream_bpe.vocab,
            beam_size=10,
            lm_model_args = trafo_32x768.net_args,
            lm_checkpoint = trafo_32x768.checkpoint,
            lm_module = "pytorch_networks.lm.trafo.kazuki_trafo_zijian_variant_v2.Model",
            lm_states_need_label_axis=True,
        )
        beam_search_decoder_config_v4_lstmlm = BeamSearchDecoderConfigv4(
            returnn_vocab=label_datastream_bpe.vocab,
            beam_size=10,
            lm_model_args = lstm_2x2048.net_args,
            lm_checkpoint = lstm_2x2048.checkpoint,
            lm_module = "pytorch_networks.lm.lstm.kazuki_lstm_zijian_variant_v2.Model",
            lm_states_need_label_axis=False,
        )
        decoder_unhashed_config_v3 = DecoderExtraConfig(
            lm_package=PACKAGE,
        )

        tune_and_evaluate_helper(
            training_name + "/trafolm_test", dev_dataset_tuples, test_dataset_tuples, asr_model, beam_search_decoder_config,
            unhashed_decoder_config=decoder_unhashed_config_v3, extra_forward_config={"batch_size": 200 * 16000},
            lm_scales=[0.4, 0.5, 0.6, 0.7, 0.8], prior_scales=[0.2, 0.3, 0.4, 0.5], decoder_module="ctc.decoder.beam_search_bpe_ctc_v3", debug=True, use_gpu=True
        )
        tune_and_evaluate_helper(
            training_name + "/trafolm_test_v4", dev_dataset_tuples, test_dataset_tuples, asr_model, beam_search_decoder_config_v4,
            unhashed_decoder_config=decoder_unhashed_config_v3, extra_forward_config={"batch_size": 200 * 16000},
            lm_scales=[0.8], prior_scales=[0.3], decoder_module="ctc.decoder.beam_search_bpe_ctc_v4", debug=True, use_gpu=True
        )
        tune_and_evaluate_helper(
            training_name + "/no_lm_test", dev_dataset_tuples, test_dataset_tuples, asr_model, beam_search_decoder_config,
            unhashed_decoder_config=decoder_unhashed_config_v3, extra_forward_config={"batch_size": 200 * 16000},
            lm_scales=[0.0], prior_scales=[0.0], decoder_module="ctc.decoder.beam_search_bpe_ctc_v3", debug=True, use_gpu=True
        )
        tune_and_evaluate_helper(
            training_name + "/trafolm_32x768", dev_dataset_tuples, test_dataset_tuples, asr_model, beam_search_decoder_config_v4_32lm,
            unhashed_decoder_config=decoder_unhashed_config_v3, extra_forward_config={"batch_size": 200 * 16000},
            lm_scales=[0.7, 0.75, 0.8, 0.85, 0.9], prior_scales=[0.3, 0.35, 0.4], decoder_module="ctc.decoder.beam_search_bpe_ctc_v4", debug=True, use_gpu=True
        )
        tune_and_evaluate_helper(
            training_name + "/trafolm_32x768_test_against_example", dev_dataset_tuples, test_dataset_tuples, asr_model, beam_search_decoder_config_v4_32lm,
            unhashed_decoder_config=decoder_unhashed_config_v3, extra_forward_config={"batch_size": 200 * 16000},
            lm_scales=[0.91], prior_scales=[.35], decoder_module="ctc.decoder.beam_search_bpe_ctc_v4", debug=False, use_gpu=True
        )
        tune_and_evaluate_helper(
            training_name + "/lstmlm_2x2048", dev_dataset_tuples, test_dataset_tuples, asr_model, beam_search_decoder_config_v4_lstmlm,
            unhashed_decoder_config=decoder_unhashed_config_v3, extra_forward_config={"batch_size": 200 * 16000},
            lm_scales=[0.65, 0.7, 0.75, 0.8, 0.85], prior_scales=[0.25, 0.3, 0.35, 0.4], decoder_module="ctc.decoder.beam_search_bpe_ctc_v4", debug=True, use_gpu=True
        )
        tune_and_evaluate_helper(
            training_name + "/lstmlm_2x2048_v5", dev_dataset_tuples, test_dataset_tuples, asr_model, beam_search_decoder_config_v4_lstmlm,
            unhashed_decoder_config=decoder_unhashed_config_v3, extra_forward_config={"batch_size": 200 * 16000},
            lm_scales=[0.65, 0.7, 0.75, 0.8, 0.85], prior_scales=[0.25, 0.3, 0.35, 0.4], decoder_module="ctc.decoder.beam_search_bpe_ctc_v5", debug=True, use_gpu=True
        )

        beam_search_decoder_config = BeamSearchDecoderConfig(
            returnn_vocab=label_datastream_bpe.vocab,
            beam_size=1,
            lm_model_args = trafo_12x768.net_args,
            lm_checkpoint = trafo_12x768.checkpoint,
            lm_module = "pytorch_networks.lm.trafo.kazuki_trafo_zijian_variant_v2.Model",
        )
        tune_and_evaluate_helper(
            training_name + "/trafolm_test_beam1", dev_dataset_tuples, test_dataset_tuples, asr_model, beam_search_decoder_config,
            unhashed_decoder_config=decoder_unhashed_config_v3, extra_forward_config={"batch_size": 200 * 16000},
            lm_scales=[0.3, 0.5, 0.7, 1.0], prior_scales=[0.0], decoder_module="ctc.decoder.beam_search_bpe_ctc_v3", debug=True, use_gpu=True
        )


        # test with bs360
        train_args_bs360 = copy.deepcopy(train_args)
        train_args_bs360["config"]["batch_size"] = 360 * 16000
        training_name = prefix_name + "/" + str(BPE_SIZE) + "/" + network_module + ".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm"
        train_job = training(training_name, train_data_bpe, train_args_bs360, num_epochs=1000, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        asr_model = prepare_asr_model(
            training_name, train_job, train_args_bs360, with_prior=True, datasets=train_data_bpe, get_specific_checkpoint=1000
        )
        tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, default_decoder_config_bpe, lm_scales=[1.6, 1.8, 2.0], prior_scales=[0.2, 0.3, 0.4])
        greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)

        # test dropout broadcasting
        train_args_dropbt = copy.deepcopy(global_train_args)
        train_args_dropbt["net_args"] = {"model_config_dict": asdict(model_config_dropbt)}

        training_name = prefix_name + "/" + str(BPE_SIZE) + "/" + network_module + ".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_smallbatch_dropbt"
        train_job = training(training_name, train_data_bpe, train_args_dropbt, num_epochs=1000, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        asr_model = prepare_asr_model(
            training_name, train_job, train_args_dropbt, with_prior=True, datasets=train_data_bpe, get_specific_checkpoint=1000
        )
        tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, default_decoder_config_bpe, lm_scales=[1.6, 1.8, 2.0], prior_scales=[0.2, 0.3, 0.4])
        greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)

        # test with bs 360
        train_args_dropbt_bs360 = copy.deepcopy(train_args_dropbt)
        train_args_dropbt_bs360["config"]["batch_size"] = 360 * 16000
        training_name = prefix_name + "/" + str(BPE_SIZE) + "/" + network_module + ".512dim_sub4_48gbgpu_100eps_sp_lp_fullspec_gradnorm_dropbt"
        train_job = training(training_name, train_data_bpe, train_args_dropbt_bs360, num_epochs=1000, **default_returnn)
        train_job.rqmt["gpu_mem"] = 48
        asr_model = prepare_asr_model(
            training_name, train_job, train_args_dropbt_bs360, with_prior=True, datasets=train_data_bpe, get_specific_checkpoint=1000
        )
        tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, default_decoder_config_bpe, lm_scales=[1.6, 1.8, 2.0], prior_scales=[0.2, 0.3, 0.4])
        greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)

        # higher lr
        train_args_lr07 = copy.deepcopy(train_args)
        train_args_lr07["config"]["learning_rates"] = list(np.linspace(7e-6, 7e-4, 480)) + list(
                np.linspace(7e-4, 5e-5, 480)) + list(np.linspace(5e-5, 1e-7, 40))

        training_name = prefix_name + "/" + str(BPE_SIZE) + "/" + network_module + ".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_smallbatch_lr07"
        train_job = training(training_name, train_data_bpe, train_args_lr07, num_epochs=1000, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        train_job.hold()
        asr_model = prepare_asr_model(
            training_name, train_job, train_args_lr07, with_prior=True, datasets=train_data_bpe, get_specific_checkpoint=1000
        )
        tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, default_decoder_config_bpe, lm_scales=[1.6, 1.8, 2.0], prior_scales=[0.2, 0.3, 0.4])
        greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)
        
        # higher lr bs360
        train_args_lr07 = copy.deepcopy(train_args_bs360)
        train_args_lr07["config"]["learning_rates"] = list(np.linspace(7e-6, 7e-4, 480)) + list(
                np.linspace(7e-4, 5e-5, 480)) + list(np.linspace(5e-5, 1e-7, 40))

        training_name = prefix_name + "/" + str(BPE_SIZE) + "/" + network_module + ".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_lr07"
        train_job = training(training_name, train_data_bpe, train_args_lr07, num_epochs=1000, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        train_job.hold()
        asr_model = prepare_asr_model(
            training_name, train_job, train_args_lr07, with_prior=True, datasets=train_data_bpe, get_specific_checkpoint=1000
        )
        tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, default_decoder_config_bpe, lm_scales=[1.6, 1.8, 2.0], prior_scales=[0.2, 0.3, 0.4])
        greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)
        
        # higher lr bs360, laplace4
        train_args_lr07 = copy.deepcopy(train_args_bs360)
        train_args_lr07["config"]["learning_rates"] = list(np.linspace(7e-6, 7e-4, 480)) + list(
                np.linspace(7e-4, 5e-5, 480)) + list(np.linspace(5e-5, 1e-7, 40))
        train_args_lr07["post_config"] = {"num_workers_per_gpu": 8}

        training_name = prefix_name + "/" + str(BPE_SIZE) + "/" + network_module + ".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_lr07_work8"
        train_job = training(training_name, train_data_bpe_laplace4, train_args_lr07, num_epochs=1000, **default_returnn)
        train_job.rqmt["gpu_mem"] = 24
        train_job.hold()
        asr_model = prepare_asr_model(
            training_name, train_job, train_args_lr07, with_prior=True, datasets=train_data_bpe, get_specific_checkpoint=1000
        )
        tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model, default_decoder_config_bpe, lm_scales=[1.6, 1.8, 2.0], prior_scales=[0.2, 0.3, 0.4])
        greedy_search_helper(training_name, asr_model=asr_model, decoder_config=greedy_decoder_config)

        asr_model.lexicon = get_text_lexicon(prefix=prefix_name, librispeech_key="train-other-960", bpe_size=BPE_SIZE)
        asr_model.returnn_vocab = label_datastream_bpe.vocab
        asr_model.settings = train_settings
        asr_model.label_datastream = label_datastream_bpe
        add_ctc_model(network_module + ".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_lr07_work8", asr_model)


        # New test more closer to other setup
        train_config_24gbgpu_amp_radam = {
            "optimizer": {"class": "radam", "epsilon": 1e-12, "weight_decay": 1e-2, "decoupled_weight_decay": True},
            "learning_rates":list(np.linspace(5e-5, 5e-4, 480)) + list(
            np.linspace(5e-4, 5e-5, 480)) + list(np.linspace(5e-5, 1e-7, 40)),
            #############
            "batch_size": 240 * 16000,
            "max_seq_length": {"audio_features": 35 * 16000},
            "accum_grad_multiple_step": 1,
            "gradient_clip_norm": 1.0,
            "torch_amp_options": {"dtype": "bfloat16"},
        }

        network_module = "ctc.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1"
        train_args_radam = {
            "config": train_config_24gbgpu_amp_radam,
            "network_module": network_module,
            "net_args": {"model_config_dict": asdict(model_config)},
            "use_speed_perturbation": True,
            "debug": False,
        }

        training_name = prefix_name + "/" + str(
            BPE_SIZE) + "/" + network_module + ".512dim_sub4_24gbgpu_100eps_sp_lp_fullspec_gradnorm_radam_lr5e-4"
        train_job = training(training_name, train_data_bpe_laplace4, train_args_radam, num_epochs=1000,
                             **default_returnn)
        train_job.rqmt["gpu_mem"] = 24

        asr_model = prepare_asr_model(
            training_name, train_job, train_args_radam, with_prior=True, datasets=train_data_bpe,
            get_specific_checkpoint=1000
        )
        tune_and_evaluate_helper(training_name, dev_dataset_tuples, test_dataset_tuples, asr_model,
                                 default_decoder_config_bpe, lm_scales=[1.6, 1.8, 2.0], prior_scales=[0.2, 0.3, 0.4])
